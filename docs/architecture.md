# 設計・MVPの境界

## 1. アーキテクチャ

```text
React / Vite source (frontend/)
  ├─ dev: 127.0.0.1:5173 ──proxy /api,/files──> Flask
  └─ build ──> src/manga_scan/static/ (generated, gitignored)
                                      │
CLI / Flask loopback Web UI (127.0.0.1:8765)
  └─ ingest → video → motion → candidate sampling → hand + score
       → candidate selection → dedupe → configured rotation
       → page contour + per-page perspective OR spread perspective + split
       → finger repair → dewarp → illumination → white normalization / enhancement
       → review state → export
       └─ project manifest + per-stage inspectable artifacts
```

- `ingest.py`: ローカル動画確認、ffprobeメタデータ、表紙/見開き基準フレーム選択、プロジェクト作成。
- `video.py`: FFmpeg境界。回転メタデータを適用した画像、表示時刻でのseek、縮小パイプ。
- `motion.py`: ROI差分、stable/turning状態機械、時間分散した候補抽出。
- `hand.py`: MediaPipe IMAGEモード、最大4手、landmark凸包を膨張したマスクとROIの交差。
- `finger_alignment.py`: donorのglobal ECC、component単位local translation、clean-context検証、bounded photometric alignment。
- `finger_repair.py`: alignment済みdonorの実画素合成、境界feather、未補修fallback、repair metadata。
- `score.py`: 品質指標、合成スコア、suspect判定。
- `selection.py`: 候補見開きを左右に分けたページ単位スコアと、左右別候補IDの選択。
- `page_detect.py`: ユーザー指定の見開きROIを外側へ広げない保守的な外周微調整。
- `spread_boundary.py`: GrabCutで見開きと机の境界を推定し、ROIを内外両方向へ補正。表紙らしい平行な帯を検出しても、ページ内の印刷帯と単一フレームでは区別できない。初期ROIが内側の境界を支持する場合のみそれを採用し、それ以外は外側を維持してレビューに曖昧さを示す。表紙の色は固定せず、変化量・元ROI外周の前景率で危険な変更を棄却する。
- `page_contour.py`: 見開き内の左右ページ外周を各候補フレームで個別検出し、外れ値を除いたconfidence加重consensus quadを返す。片側が指や影で欠けたフレームも別候補で補完し、低confidence時は既存ROI分割へfallback。
- `page_warp.py`: 左右ページquadを独立した `warpPerspective` で長方形化する。
- `perspective.py`: ROI検証、見開き射影変換、90°単位のROI回転。
- `split.py`: 設定回転、背の推定、左右分割、自動湾曲推定、左右別の保守的remap、白背景正規化、グレースケール、コントラスト。
- `illumination.py`: ページ輝度の低周波マップ推定と、Lab輝度/グレースケールへの保守的な照明補正。
- `dedupe.py`: dHash / SSIM と ORB 特徴点の幾何照合。RANSAC 後の一致点を左右ページ別に数え、指で隠れた見開きも比較する。
- `export.py`: 画像PDF、分割コンタクトシート。
- `pipeline_render_helpers.py`: page geometry、mask変換、page-level overrideなどrenderingの共通helper。
- `spread_render.py`: 見開き1枚出力のrendering実装。
- `pipeline.py`: scan/render/export/editのorchestrationと互換facade。画素アルゴリズムをUIから分離。
- `storage.py`: atomic JSON保存、画像保存、プロジェクト排他ロック。
- `frontend/`: React + ViteのUIソース。開発時は5173番で起動し、`/api` と `/files` をFlaskへproxyする。
- `src/manga_scan/static/`: `npm --prefix frontend run build` の生成物。Git管理せず、Flaskが通常起動時に配信する。
- `ui.py`: loopback限定のFlask APIと静的配信。処理は背景スレッド、状態はディスクに保存。

Homebrew / pip / npmの依存導入とMediaPipeモデル取得にはネット接続が必要になり得る。
**スキャン実行時**にはモデル自動取得、クラウドAPI、OCR、画像生成、テレメトリ、外部CDN通信を行わない。
Node.jsはフロントのinstall/build/devに必要だが、build済み静的ファイルをFlaskから使う通常実行では不要。

## 2. OSS

[調査とライセンス一覧](../THIRD_PARTY.md)参照。参照実装のコードを移植せず独立実装する。

## 3. MVP範囲

動画入力、メタデータ、任意の表紙1ページ、見開き基準フレーム/ROI、めくりと静止検出、ベストフレーム、手の重なり評価、
重複除外、射影変換、左右分割、画像/PDF保存、ログ、レビューまで。
除外はmanifestのフラグで行い、重複候補も画像を残す。
初回解析で自動PDF生成。編集後は `pdf_stale=true` とし再出力を明示する。

改善余地: より高度な2D/3D曲面推定、長時間4Kでの性能最適化、
実写条件ごとの輪郭/追跡confidence校正。

## 4. アルゴリズム

### 時間・動き

FFmpegの `setpts=PTS-STARTPTS,fps=<sample_fps>,scale=...` でプレゼンテーション時刻を基準に
サンプルする。通常は `video_sample_fps=10` が上限で、入力fpsがそれ未満なら入力側に合わせる。UIで見開き基準フレームを確定した場合は、その時刻からFFmpeg入力を開始し、
それ以前の表紙区間は見開き解析へ入れない。サンプル時刻は
`analysis_start + index / sample_fps` で、元動画のフレーム番号ではない。
VFRでの候補seekはその時刻に対応する元フレームをFFmpegで取得するため、解析フレームと
最大1元フレーム程度ずれる場合がある。候補取得後に画質・手・幾何を再評価する。
iPhoneスローモーションの実撮影fpsを推測して時間を圧縮しない。

ROI射影画像のmotion判定はv2で、grayscale化後に縮小・低域化し、10/50/90 percentileから
中程度のglobal gain/bias（AE変化）を補正する。さらにgradient structureのphase correlationが
高confidenceかつ数px以内のときだけtranslationを打ち消し、軽いAF変化・微振動による差分を抑える。
低contrast画像、大きなshift、低confidence alignmentは補正せず、最後に平均絶対差 / 255をmotionとする。
既存の `motion_threshold` / `turn_threshold` のスケールは維持する。
連続 `stable_frames` 回の低motionでstable。高閾値以上で区間を閉じturningへ。
2つの閾値の間は候補に含めず、安定状態の区間は維持する。
冒頭は前フレームがないため候補から外し、末尾の確定済み静止区間は必ずflush。
短すぎる区間は採用しない。ページを長く静止させても一つの区間として扱う。

区間を最大 `candidates_per_spread` 個（デフォルト7）の時間ビンへ分割し、各ビンで
`log1p(sharpness) - 30*motion` 最大を選ぶ。
最初の鋭い候補だけに偏らず、後半に手が引かれたフレームを調べる。
候補時刻を元動画から縮小再取得しMediaPipeを実行する。同じ静止区間内で鮮鋭度・motion・
hand overlap・反射・合成スコアを相対評価する。最高点の候補より指の検出面積が1ポイント以上小さく、
ピント・動き・反射・ページ形状が許容範囲にある候補を優先する。指だけ少なくてもブレや欠けがあれば採用しない。
`candidate_selection_mode="spread"` は見開き全体で選ぶ。`"per_page"` では各候補のROI射影画像を左右に分割し、
左/右それぞれについて鮮鋭度・hand overlap・clipping・exposureを再計算する。motionとROI幾何ペナルティは
同じ候補時刻/ROI由来の値を共有し、左右ごとに候補IDを独立して選ぶ。

元解像度の再取得は実際に採用された候補IDだけに限定し、左右が同じ候補なら1回、異なる候補なら最大2回。
manifestには後方互換用の `selected` に加えて `selected_pages.left/right` を保存し、各pageにも
`candidate_id` / `candidate_time` を記録する。レビューUIでは左右片側だけ候補を差し替えられる。
`perspective_mode="per_page"` も同時に有効な場合は、同じ見開きの候補フレーム群を採用候補へoptical-flow homographyで位置合わせしてからページ輪郭のconsensusを作る。候補間で本やカメラが数十pxずれても、移動そのものを輪郭outlierとして捨てず、同じ座標系で左右ページ境界を比較できる。左右が別候補なら左右それぞれの採用候補をanchorとして独立にconsensusし、選択側だけ結果を反映する。輪郭・alignment・fallback状態もmanifestへ保持する。

レビューUIの動画タイムラインは各見開きの区間と採用候補時刻を動画全体へ配置する。
欠落候補v2は高motionのページめくりイベントを時間方向にまとめ、隣接する2イベントの間に
accepted stable intervalが存在しない場合だけ候補化する。候補時刻はその窓内の最小motion sample。
新規解析ではこのpage-turn判定を使い、v2 metadataの無い旧projectだけ開始時刻gap heuristicへfallbackする。

`auto_high_fps_fallback=true` では欠落候補の窓だけ `auto_high_fps_fallback_fps` で再サンプルする。
`auto_high_fps_min_stable_seconds` 以上、連続して `motion_threshold` 以下となるrunが確認できた場合だけ
そのrunをstable intervalへ追加し、page-turn解析を再計算する。確認できない候補は自動追加せずReviewへ残す。
また通常の見開きでも、全候補（per-page選択では片側の全候補）が
low sharpness / hand overlap / glare overlap / high motion のいずれかで要確認の場合だけ、
そのstable interval内を高fps再探索する。追加候補は既存のcandidate scoringへ合流し、
通常の自動選択と同じ規則で再選択する。source fps以下では追加サンプリングしない。

### 合成スコア

```text
 + sharpness_weight * log1p(Laplacian variance) / 8
 - motion_weight * min(1, motion / turn_threshold)
 - hand_overlap_weight * union(hand_masks ∩ ROI) / area(ROI)
 - glare_overlap_weight * likely_specular_glare / area(ROI)
 - distortion_weight * mean(abs(cos(adjacent edges)))
 - flatness_weight * opposing-edge-length-imbalance
 - clipping_weight * fraction(gray <= 2 or gray >= 253)
 - exposure_weight * max(0, 55 - mean(gray)) / 55
```

評価は同じ縮小幅で行う。`flatness_proxy` は四辺形の幾何的な代理指標で、3Dの湾曲を
測定していない。clippingは漫画の黒ベタ・白紙でも増えるため重みを小さくする。
手のマスクはsegmentationではない。指の間の空白も含む保守的な凸包。
手検出無効は `null` として保存し、検出して重なりゼロだった結果と区別する。

### 指・反射の遮蔽補修

候補フレームではMediaPipe手マスクに加え、極端な高輝度・低彩度・局所輝度差を満たす領域から
保守的な `glare_mask` を作る。候補scoreには `glare_overlap_weight` で反射重なりを減点し、
白紙そのものを反射と誤認しにくいよう、局所コントラストをseedにして連結成分単位で判定する。

補修時は既存のfinger repair engineを `occlusion repair` として一般化し、
`target_mask = finger_mask OR glare_mask` を入力する。donor側も同様に
`finger_mask OR glare_mask` をcleanではない領域として扱うため、別フレームでも指や反射に
隠れている画素はコピーしない。finger repairは `finger_repair=true`、反射補修は
`glare_repair=true` で独立に有効化できる。

採用ページに遮蔽がある場合は、同じstable interval内の別候補をhand overlap・glare overlap・
page scoreで順位付けし、最大5件までdonor候補として使う。各donorも同じ回転・ページ幾何へ射影した後、
OpenCV ECCのEUCLIDEAN alignment（小さな回転＋平行移動のみ）で採用ページへglobal alignmentし、
遮蔽connected componentごとにbounded translationのlocal alignmentを追加する。
採用側・donor側それぞれの遮蔽maskはECCの評価対象から除外し、
score 0.72未満・5度超の回転・ページ寸法の8%超のtranslationは拒否する。scale/shearは許可しない。

置換対象は採用ページの遮蔽mask内だけで、donor側でもcleanな実画素だけを利用する。
複数donorを順番に使い、1枚で埋まらない領域を補う。境界はdistance transformに基づくfeatherで
ページ外へ変更を広げない。生成AI、inpaintingモデル、OCRによる補完は行わず、全候補で隠れている画素は
元画像をそのまま残す。復元率が `finger_repair_min_coverage`（既定0.9）未満なら
遮蔽補修を要確認として残す。coverage / donor ID / alignment score / target mask /
glare mask / unresolved maskはmanifestと `debug/finger_repair/` に残す。

### 幾何

正規化ROIは表示方向のTL/TR/BR/BL順。凸性・面積・範囲・順序を検証。
表紙は独立した時刻・ROIで1ページとして切り出し、見開きROIとは共有しない。
基準見開きは、回転済みの基準フレーム全体からまずlandscapeな外周候補を探索し、その範囲内で左右ページquadを検証する。
両ページがconfidence閾値を満たし、外側4点から作る見開きquadも有効な場合だけ自動ROIとして採用する。
検出結果はセットアップ画面へ4点プレビューし、ユーザーはそのまま確定できる。どの段階でも不確実ならROIを保存せず、
従来の4点手動指定へfallbackする。解析本体はこの確定済み基準ROIを使って机を外す。`rotation` が指定されている場合は、左右を決める前に見開き全体・ROI・手マスクを同じ向きへ回転する。これにより90°/270°の横向き撮影でも上/下ではなく見た目上の左/右ページを得る。auto splitは回転後画像の中央±4%の暗い縦谷を検索する実験機能。
stable intervalごとの候補生成前に、直前の採用見開きと現在区間の先頭候補を比較し、
前回ROI内部の特徴点だけをLucas-Kanade optical flowで追跡する。forward/backward一致、
RANSAC homographyのinlier率・再投影誤差・ROI内特徴点coverageを満たす場合だけ前回ROIを現在座標へ写し、
その見開きのtracking base ROIとして使う。1見開きのcorner移動は `roi_tracking_max_step`（既定0.08）、
確定した基準ROIからの累積移動は `roi_tracking_max_total`（既定0.16）で制限する。
追跡失敗や上限超過では直前のtrusted ROIを維持し、机の静止特徴を本の移動と誤認しないよう
特徴点探索を本ROI内部へ限定する。そのtracking base ROIに対して輪郭を各点最大2.5%まで微調整し、GrabCutの前景境界が十分明確なら机を除く内向き補正または見切れを戻す外向き補正を行う。補正結果と採否理由は候補に保存する。
`output_layout="spread"` では、採用候補の回転後元フレームから左右ページのquadを検出する。両方が
`page_contour_min_confidence` を満たした場合は、左quadの左上・左下と右quadの右上・右下を
見開き外周として1回だけ射影変換する。内側4点は使わないため、中央の綴じ目を分割・再結合しない。
片側でもconfidence不足なら見開き境界の補正結果へfallbackし、`page_contour_low_confidence` を残す。旧プロジェクトの候補に補正情報がない場合は再出力時に境界補正を試す。手動ROI overrideは変更しない。
重なった紙の内側を選んだとき、左右ページ輪郭の外側4点がさらに外側の表紙を指す場合は、その輪郭で内側の補正を上書きしない。
`page_background_fill` はこの `auto_pages` が成功した見開き出力だけを対象にする。左右ページquadと内側エッジ間のノドをunionした保護maskを同じ射影変換で出力座標へ写し、mask外だけを `paper / white` で埋める。ページ境界には小さな保護marginと外向きfeatherを設け、ページ画素や中央の綴じ目を変更しない。`paper` はページ内縁の明るい低彩度画素から紙色を推定し、十分な候補がなければ `white_target` を使う。輪郭fallback・手動crop・`preserve` では背景を変更しない。
`perspective_mode="per_page"` では、回転後の元フレーム上で左右ページの外周を別々に検出し、
両方が `page_contour_min_confidence` を満たした場合だけ各ページを独立して射影変換する。
片側でもconfidence不足なら、その見開きは従来の「見開き全体を射影変換 → 左右分割」へfallbackし、
`page_contour_low_confidence` を要確認理由として残す。検出quadとdebug overlayはmanifest / `debug/page_contours/` に保存する。
新規設定の既定は `refine_quad=true` + `perspective_mode="per_page"`。見開きROIの境界補正と左右ページ別の輪郭検出を試し、検出に十分なconfidenceがない場合は補正済みまたは元のROIへfallbackする。手で隠された外周や背景と紙面が似たケースは安全のため補正を見送る場合がある。

湾曲補正は `off / manual / auto` を選べ、新規設定の既定は `auto`。manualは従来の対称cylindrical remapを維持する。
autoは左右ページを分割した後、ページ高の9地点を中心にした複数scanline帯でSobel-x由来の
縦エッジピークを測る。各高さで背表紙側のエッジ間隔とページ中央側の間隔を比較し、
「背側へ近づくほど横方向に圧縮されている量」を独立に推定する。

各高さの推定値は欠損を補間して平滑化し、ページ上端〜下端の `strength_profile` にする。
auto補正では1つの固定gammaを全行へ使わず、行ごとに異なるgammaを持つ2D x-remapを生成する。
そのため、上側だけ強く反る、中央が最も持ち上がる、といった実物の本に近い非一様な湾曲にも
追従できる。各行のx写像は単調かつ画像内に制限し、出力幅・高さは変えない。

右ページは左端、左ページは右端を背側として補正し、各行の補正量は
`dewarp_max_strength` で上限を設ける。測定できる帯が少ない、またはノイズが大きい場合は
confidenceを下げ、`dewarp_min_confidence` 未満なら画素を一切変更せずfallbackする。
完全な3D形状復元ではなく、背側の横方向圧縮と高さ方向の変動を軽減する保守的2D dewarpである。

auto時は補正前画像と、実際の高さ別remapを描いたグリッドを `debug/dewarp/` に残す。
manifestの各pageには peak strength / mean strength / profile variation / strength_profile /
confidence / statusを保存する。レビューUIでは最大補正量と高さ方向差を表示し、左右ページ単位で
autoを無効化できる。低confidence時は `dewarp_low_confidence` を要確認理由へ追加する。
表紙は背側を決められないためauto対象外で、manualのみ適用可能。文字認識・生成AI・描き足しは行わない。

指補修は回転済み見開きの左右ページ射影/分割後、湾曲・照明・白背景などの画質補正より前に行う。

照明ムラ補正は新規設定で既定有効。回転済み見開きの左右分割/湾曲補正後、白背景正規化・grayscale・contrast前に
ページ単位で補正する。カラー画像はLabのL成分だけ、グレースケール画像はその輝度を直接扱う。
照明マップの推定だけを最大512pxへ縮小し、大きめのmorphological closingで線画・網点などの
暗い高周波成分を抑えた後、Gaussian blurで低周波成分へ限定する。元画像を二値化せず、
global thresholdや背景画素の分類は行わない。

照明マップの90 percentileを基準に乗算ゲインを求め、0.9〜1.4倍へ制限した上で
`illumination_strength` (0..1) で1倍との間を補間する。黒に近い画素は乗算なので大きく持ち上がりにくく、
大きな黒ベタが照明マップを誤らせた場合もゲイン上限で影響を抑える。補正前の見開きは従来どおり
`selected/` に残る。

白背景正規化は新規設定で既定有効。照明ムラ補正の後に、ページの明るい低彩度画素からrobustなwhite levelを推定し、
white targetへ緩やかに寄せる。補正重みはwhite levelの約45階調下からsmoothstepで立ち上げるため、
黒ベタや中間調は原則そのまま残す。カラー画像ではLab色空間を使い、低彩度の明部だけ色かぶりを
弱める。十分な明るい紙面候補がない暗いページでは補正せずfallbackする。

### 重複

直近3つの採用見開きとdHash + SSIM + 左右それぞれのSSIMを比較。画素比較で足りない場合は縮小画像のORB特徴点をRANSACで位置合わせする。OCRは行わない。per-page選択で左右の候補時刻が異なる場合は、
採用した左ページと右ページの低解像度previewを合成した見開きを重複判定に使う。
指が大きく重なって全体の画素相関が低い場合も、位置合わせ後の特徴点が左右両ページに十分残り、広い範囲で一致すれば重複とする。片側のページだけ一致する場合は自動除外せず要確認にとどめる。
閾値全てを満たすと重複候補をいったん除外する。ただし後から見つかった重複画像の方が
指の面積や補修結果で優れ、ピントが許容範囲にあり、新たな外周・最終品質の警告を増やさない場合は採用ページを自動で切り替える。
手動で有効・無効や候補を変更した重複グループは、後続の自動切替から保護する。
コントラストが低い画像（白紙等）は自動除外しない。
近似候補は `duplicate_suspected` として残す。既存の読書順と絵の反復を尊重し、
全巻全ページを総当たりで消さない。左右別の白紙や繰り返し絵の重複は意図的に除外しない。
候補切替後に重複除外を勝手に再適用しない。元の警告と除外状態を維持し、UIで復元できる。

## 5. ディレクトリ

README参照。`manifest.json` の `pages` 配列がページ順の唯一の根拠。
画像ファイル名のsortはPDFの順序として使用しない。

## 6. 実装・検証ステップ

1. 設定、ffprobe/ffmpeg、ROI座標系。
2. 動き状態機械、候補時間分散、手・品質スコア。
3. 重複、補正、分割、可逆画像PDF。
4. CLI、四隅UI、レビュー操作。
5. 独立ユニットテスト、合成動画結合テスト、PDF内画像の画素・JPEGバイト検査。
6. Mac arm64で実モデルのロード・推論確認、公開用ドキュメント。

## 7. 技術的リスク

- デフォルト10fpsではサンプル間に完了するページは見えない。デフォルトの静止5フレームは約0.5秒必要。
  `video_sample_fps` / `stable_frames` は設定可能だが、上げるほど解析量や誤検出とのトレードオフがある。
- 手で全候補にわたって常に隠れる箇所は指補修でも復元不能。生成せず元画素を残し、要確認にする。
- donor候補の位置合わせが誤ると別の線・文字を貼る危険があるため、ECC scoreを高めに取り、
  residual transformを小さな回転＋平行移動だけに限定する。条件を外れたdonorは補修せずfallbackする。
- MediaPipeは一部だけ見える指、漫画に描かれた手で誤判定し得る。
- 白飛び、黒つぶれ、照明反射、コマ内黒ベタを完全には区別できない。反射maskは保守的にし、全候補で同じ領域がmaskされる場合は実画素置換せず要確認として残す。
- 射影変換は平面を仮定。背の湾曲・厚み・机から浮く紙は完全には直らない。
- HDR→8bitは校正済みトーンマップではない。SDR推奨、HDR入力は警告。
- タイムラインの欠落候補・時間間隔警告はページ欠落の証明にならない。OCRなしで実ページ番号は分からない。
- FFmpegは圧縮動画の中間フレームを内部でdecodeする。4K全フレームをPythonで解析する
  ことは避けるが、decode自体をサンプル数まで削減できるわけではない。
- 長いGOPを何度もseekすると候補抽出が遅い。今後は候補を時間順にまとめてdecode可能。
- Pythonに保持する高解像度画像は一見開きずつ。PDF生成はReportLabが圧縮ページを
  内部保持するため、長大な本ではPDFのサイズに応じてメモリを使う。
- 背景スレッドはブラウザを閉じても続くが、サーバー終了・Macスリープ中は完了しない。
  異常終了後は同一プロジェクトで最初から再解析。処理途中からの自動再開は未実装。
