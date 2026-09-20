# 設計・MVPの境界

## 1. アーキテクチャ

```text
React / Vite source (frontend/)
  ├─ dev: 127.0.0.1:5173 ──proxy /api,/files──> Flask
  └─ build ──> src/manga_scan/static/ (generated, gitignored)
                                      │
CLI / Flask loopback Web UI (127.0.0.1:8765)
  └─ ingest → video → motion → candidate sampling → hand + score
       → dedupe → perspective → split → page enhancement → export
       └─ project manifest + per-stage inspectable artifacts
```

- `ingest.py`: ローカル動画確認、ffprobeメタデータ、表紙/見開き基準フレーム選択、プロジェクト作成。
- `video.py`: FFmpeg境界。回転メタデータを適用した画像、表示時刻でのseek、縮小パイプ。
- `motion.py`: ROI差分、stable/turning状態機械、時間分散した候補抽出。
- `hand.py`: MediaPipe IMAGEモード、最大4手、landmark凸包を膨張したマスクとROIの交差。
- `score.py`: 品質指標、合成スコア、suspect判定。
- `page_detect.py` / `perspective.py`: 保守的な外周微調整、ROI検証、射影変換。
- `split.py`: 背の推定、左右分割、自動湾曲推定、左右別の保守的remap、白背景正規化、グレースケール、コントラスト、回転。
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

改善余地: optical flow併用、ページ単位で別々のベストフレーム選択、より高度な2D/3D曲面推定、
追跡によるROI移動、複数動画の統合、画像追加、ドラッグ並べ替え、ジョブ再開。

## 4. アルゴリズム

### 時間・動き

FFmpegの `setpts=PTS-STARTPTS,fps=10,scale=...` でプレゼンテーション時刻を基準に
サンプルする。UIで見開き基準フレームを確定した場合は、その時刻からFFmpeg入力を開始し、
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

区間を最大7つの時間ビンへ分割し、各ビンで `log1p(sharpness) - 30*motion` 最大を選ぶ。
最初の鋭い候補だけに偏らず、後半に手が引かれたフレームを調べる。
候補時刻を元動画から縮小再取得しMediaPipeを実行、最高スコアの1枚だけ元解像度で取得。

レビューUIの動画タイムラインは、各見開きの採用候補時刻（無ければ区間中央）を動画全体へ配置する。
重複除外済み見開きを除いた隣接時刻差の中央値を通常のページ送り間隔とみなし、その約1.8倍以上の
空白を「欠落ページ候補」として表示する。空白の長さから最大4件まで候補時刻を等間隔に推定する。
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

### 幾何

正規化ROIは表示方向のTL/TR/BR/BL順。凸性・面積・範囲・順序を検証。
表紙は独立した時刻・ROIで1ページとして切り出し、見開きROIとは共有しない。
見開きはユーザーが選んだ基準フレームのROIで透視補正して机を外し、中央で左右分割。auto splitは中央±4%の暗い縦谷を検索する実験機能。
自動外周補正は元ROIより外へ広げず、各点最大2.5%の移動まで。検出失敗はROI fallbackと警告。
デフォルトは手動ROI固定なので、漫画が動いた場合の背景混入を自動保証できない。

湾曲補正は `off / manual / auto` を選べる。manualは従来の対称cylindrical remapを維持する。
autoは左右ページを分割した後、それぞれ5つの高さ帯でSobel-x由来の縦エッジピークを取り、
背表紙側のエッジ間隔中央値とページ中央側の中央値を比較する。複数帯で圧縮比が一貫している
場合だけconfidenceを上げ、`dewarp_min_confidence` 未満なら画素を変更せずfallbackする。

補正自体はページ外へ画素を生成せず、出力幅を維持した1次元の単調なx remap。
右ページは左端、左ページは右端を背側として、その側へ近づくほど元画像の狭い範囲を
多くの出力画素へ割り当てる。強度は `dewarp_max_strength` で上限を設ける。
これは完全な3D復元ではなく、背側の横方向圧縮を軽減する保守的MVPである。

auto時は補正前画像とremapグリッドを `debug/dewarp/` に残し、manifestの各pageへ
strength / confidence / statusを保存する。レビューUIから左右ページ単位でautoを無効化できる。
低confidence時は `dewarp_low_confidence` を要確認理由へ追加する。表紙は背側を決められないため
auto対象外で、manualのみ適用可能。文字認識・生成AI・描き足しは行わない。

照明ムラ補正は既定無効。ON時は左右分割/湾曲補正後、白背景正規化・grayscale・contrast・rotation前に
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

直近3つの採用見開きとdHash + SSIM + 左右それぞれのSSIMを比較。
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

- 10fpsでサンプル間に完了するページは見えない。静止5フレームは約0.5秒必要。
- 手で常に隠れる箇所・ブレしかないページは復元不能。疑わしい結果をレビューする。
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

