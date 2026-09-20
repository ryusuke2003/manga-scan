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
- `finger_repair.py`: 別候補ページの保守的位置合わせ、手マスクで保護した実画素置換、境界feather。
- `score.py`: 品質指標、合成スコア、suspect判定。
- `selection.py`: 候補見開きを左右に分けたページ単位スコアと、左右別候補IDの選択。
- `page_detect.py`: ユーザー指定の見開きROIを外側へ広げない保守的な外周微調整。
- `page_contour.py`: 見開き内の左右ページ外周を個別検出し、confidence付きquadを返す。低confidence時は既存ROI分割へfallback。
- `page_warp.py`: 左右ページquadを独立した `warpPerspective` で長方形化する。
- `perspective.py`: ROI検証、見開き射影変換、90°単位のROI回転。
- `split.py`: 設定回転、背の推定、左右分割、自動湾曲推定、左右別の保守的remap、白背景正規化、グレースケール、コントラスト。
- `illumination.py`: ページ輝度の低周波マップ推定と、Lab輝度/グレースケールへの保守的な照明補正。
- `dedupe.py`: dHashと局所SSIM。左右半分も比較。
- `export.py`: 画像PDF、分割コンタクトシート。
- `pipeline.py`: 処理の接続とレビュー操作。画素アルゴリズムをUIから分離。
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

改善余地: optical flow併用、より高度な2D/3D曲面推定、
追跡によるROI移動、複数動画の統合、画像追加、ドラッグ並べ替え、ジョブ再開。

## 4. アルゴリズム

### 時間・動き

FFmpegの `setpts=PTS-STARTPTS,fps=<sample_fps>,scale=...` でプレゼンテーション時刻を基準に
サンプルする。通常は `video_sample_fps=10` が上限で、入力fpsがそれ未満なら入力側に合わせる。UIで見開き基準フレームを確定した場合は、その時刻からFFmpeg入力を開始し、
それ以前の表紙区間は見開き解析へ入れない。サンプル時刻は
`analysis_start + index / sample_fps` で、元動画のフレーム番号ではない。
VFRでの候補seekはその時刻に対応する元フレームをFFmpegで取得するため、解析フレームと
最大1元フレーム程度ずれる場合がある。候補取得後に画質・手・幾何を再評価する。
iPhoneスローモーションの実撮影fpsを推測して時間を圧縮しない。

ROI射影画像をgrayscale → Gaussian blur → 平均絶対差 / 255。
連続 `stable_frames` 回の低motionでstable。高閾値以上で区間を閉じturningへ。
2つの閾値の間は候補に含めず、安定状態の区間は維持する。
冒頭は前フレームがないため候補から外し、末尾の確定済み静止区間は必ずflush。
短すぎる区間は採用しない。ページを長く静止させても一つの区間として扱う。

区間を最大 `candidates_per_spread` 個（デフォルト7）の時間ビンへ分割し、各ビンで
`log1p(sharpness) - 30*motion` 最大を選ぶ。
最初の鋭い候補だけに偏らず、後半に手が引かれたフレームを調べる。
候補時刻を元動画から縮小再取得しMediaPipeを実行する。`candidate_selection_mode="spread"` は従来どおり
見開き全体の合成スコア最大を採用する。`"per_page"` では各候補のROI射影画像を左右に分割し、
左/右それぞれについて鮮鋭度・hand overlap・clipping・exposureを再計算する。motionとROI幾何ペナルティは
同じ候補時刻/ROI由来の値を共有し、左右ごとに合成スコア最大の候補IDを独立して選ぶ。

元解像度の再取得は実際に採用された候補IDだけに限定し、左右が同じ候補なら1回、異なる候補なら最大2回。
manifestには後方互換用の `selected` に加えて `selected_pages.left/right` を保存し、各pageにも
`candidate_id` / `candidate_time` を記録する。レビューUIでは左右片側だけ候補を差し替えられる。
`perspective_mode="per_page"` も同時に有効な場合は、採用された各候補フレームごとにページ輪郭検出と
左右別射影変換を行う。左右が別候補なら輪郭・fallback状態も `*_by_side` としてmanifestへ保持する。

レビューUIの動画タイムラインは各見開きの区間と採用候補時刻を動画全体へ配置する。
欠落候補の判定には候補選択位置の揺れを使わず、重複除外済み見開きの安定区間開始時刻 `spread.start` を使う。
隣接開始時刻差の中央値を通常のページ送り間隔とみなし、フロント側の既定ではその約1.8倍以上の
空白を「欠落ページ候補」として表示する。これはUIの候補表示用閾値であり、pipeline側で
`interval_gap` 警告を付ける `interval_gap_factor`（デフォルト3.0）とは別のヒューリスティックである。
空白の長さから最大4件まで候補時刻を等間隔に推定する。
これはOCRや実ページ番号による欠落判定ではなく時間間隔のヒューリスティックなので、自動追加はせず、
ユーザーが元動画を確認した上で既存の手動追加へ渡す。手動追加後はspread時刻列へ入るため候補を再計算する。

### 合成スコア

```text
 + sharpness_weight * log1p(Laplacian variance) / 8
 - motion_weight * min(1, motion / turn_threshold)
 - hand_overlap_weight * union(hand_masks ∩ ROI) / area(ROI)
 - distortion_weight * mean(abs(cos(adjacent edges)))
 - flatness_weight * opposing-edge-length-imbalance
 - clipping_weight * fraction(gray <= 2 or gray >= 253)
 - exposure_weight * max(0, 55 - mean(gray)) / 55
```

評価は同じ縮小幅で行う。`flatness_proxy` は四辺形の幾何的な代理指標で、3Dの湾曲を
測定していない。clippingは漫画の黒ベタ・白紙でも増えるため重みを小さくする。
手のマスクはsegmentationではない。指の間の空白も含む保守的な凸包。
手検出無効は `null` として保存し、検出して重なりゼロだった結果と区別する。

### 指の写り込み補修

`finger_repair=true` のときだけ実行する。候補評価時に保存したMediaPipe手マスクを元フレーム解像度へ
nearest-neighborで戻し、設定回転を同じように適用してから、採用ページと同じROI / page contour / split座標へ射影する。
採用ページで手と判定された領域がなければ追加decodeは行わない。

手領域がある場合は、同じstable interval内の別候補をhand overlapの少ない順・page scoreの高い順で
最大5件までdonor候補として元解像度再取得する。各donorも同じ回転・ページ幾何へ射影した後、
OpenCV ECCのEUCLIDEAN alignment（小さな回転＋平行移動のみ）で採用ページへ追加位置合わせする。
採用側・donor側それぞれの手マスク領域は各画像の自座標で中立値へ置換してECCから実質除外し、
score 0.72未満・5度超の回転・ページ寸法の8%超のtranslationは拒否する。scale/shearは許可しない。

置換対象は採用ページの手マスク内だけで、donor側でも手に隠れていない画素だけを利用する。
複数donorを順番に使い、1枚で埋まらない領域を補う。境界はdistance transformに基づくfeatherで
ページ外へ変更を広げない。生成AI、inpaintingモデル、OCRによる補完は行わず、全候補で隠れている画素は
元画像をそのまま残す。復元率が `finger_repair_min_coverage`（既定0.9）未満なら
`finger_repair_incomplete` を要確認理由に追加する。coverage / donor ID / alignment score /
target mask / unresolved maskはmanifestと `debug/finger_repair/` に残す。

### 幾何

正規化ROIは表示方向のTL/TR/BR/BL順。凸性・面積・範囲・順序を検証。
表紙は独立した時刻・ROIで1ページとして切り出し、見開きROIとは共有しない。
見開きはユーザーが選んだ基準フレームのROIで透視補正して机を外す。`rotation` が指定されている場合は、左右を決める前に見開き全体・ROI・手マスクを同じ向きへ回転し、その表示向きで中央から左右分割する。これにより90°/270°の横向き撮影でも上/下ではなく見た目上の左/右ページを得る。auto splitは回転後画像の中央±4%の暗い縦谷を検索する実験機能。
自動外周補正は元ROIより外へ広げず、各点最大2.5%の移動まで。検出失敗はROI fallbackと警告。
`perspective_mode="per_page"` では、回転後の元フレーム上で左右ページの外周を別々に検出し、
両方が `page_contour_min_confidence` を満たした場合だけ各ページを独立して射影変換する。
片側でもconfidence不足なら、その見開きは従来の「見開き全体を射影変換 → 左右分割」へfallbackし、
`page_contour_low_confidence` を要確認理由として残す。検出quadとdebug overlayはmanifest / `debug/page_contours/` に保存する。
デフォルトは手動ROI固定 + spread方式なので、漫画が動いた場合の背景混入を自動保証できない。

湾曲補正は `off / manual / auto` を選べる。manualは従来の対称cylindrical remapを維持する。
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

照明ムラ補正は既定無効。ON時は回転済み見開きの左右分割/湾曲補正後、白背景正規化・grayscale・contrast前に
ページ単位で補正する。カラー画像はLabのL成分だけ、グレースケール画像はその輝度を直接扱う。
照明マップの推定だけを最大512pxへ縮小し、大きめのmorphological closingで線画・網点などの
暗い高周波成分を抑えた後、Gaussian blurで低周波成分へ限定する。元画像を二値化せず、
global thresholdや背景画素の分類は行わない。

照明マップの90 percentileを基準に乗算ゲインを求め、0.9〜1.4倍へ制限した上で
`illumination_strength` (0..1) で1倍との間を補間する。黒に近い画素は乗算なので大きく持ち上がりにくく、
大きな黒ベタが照明マップを誤らせた場合もゲイン上限で影響を抑える。補正前の見開きは従来どおり
`selected/` に残る。

白背景正規化は既定無効。ON時は照明ムラ補正の後に、ページの明るい低彩度画素からrobustなwhite levelを推定し、
white targetへ緩やかに寄せる。補正重みはwhite levelの約45階調下からsmoothstepで立ち上げるため、
黒ベタや中間調は原則そのまま残す。カラー画像ではLab色空間を使い、低彩度の明部だけ色かぶりを
弱める。十分な明るい紙面候補がない暗いページでは補正せずfallbackする。

### 重複

直近3つの採用見開きとdHash + SSIM + 左右それぞれのSSIMを比較。per-page選択で左右の候補時刻が異なる場合は、
採用した左ページと右ページの低解像度previewを合成した見開きを重複判定に使う。
閾値全てを満たすと両ページを除外。コントラストが低い画像（白紙等）は自動除外しない。
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
- 白飛び、黒つぶれ、照明反射、コマ内黒ベタを正確に区別できない。
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

