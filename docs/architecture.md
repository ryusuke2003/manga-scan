# 設計・MVPの境界

## 1. アーキテクチャ

```text
CLI / loopback Web UI
  └─ ingest → video → motion → candidate sampling → hand + score
       → dedupe → perspective → split → page enhancement → export
       └─ project manifest + per-stage inspectable artifacts
```

- `ingest.py`: ローカル動画確認、ffprobeメタデータ、最初のフレーム、プロジェクト作成。
- `video.py`: FFmpeg境界。回転メタデータを適用した画像、表示時刻でのseek、縮小パイプ。
- `motion.py`: ROI差分、stable/turning状態機械、時間分散した候補抽出。
- `hand.py`: MediaPipe IMAGEモード、最大4手、landmark凸包を膨張したマスクとROIの交差。
- `score.py`: 品質指標、合成スコア、suspect判定。
- `page_detect.py` / `perspective.py`: 保守的な外周微調整、ROI検証、射影変換。
- `split.py`: 背の推定、左右分割、グレースケール、コントラスト、回転、任意の円筒リマップ。
- `dedupe.py`: dHashと局所SSIM。左右半分も比較。
- `export.py`: 画像PDF、分割コンタクトシート。
- `pipeline.py`: 処理の接続とレビュー操作。画素アルゴリズムをUIから分離。
- `storage.py`: atomic JSON保存、画像保存、プロジェクト排他ロック。
- `ui.py` / `static/`: ローカルUI。処理は背景スレッド、状態はディスクに保存。

ネットワークを使うのはユーザーが明示実行する初期セットアップのモデル取得スクリプトだけ。
ランタイムにモデル自動取得、クラウドAPI、OCR、画像生成、テレメトリはない。

## 2. OSS

[調査とライセンス一覧](../THIRD_PARTY.md)参照。参照実装のコードを移植せず独立実装する。

## 3. MVP範囲

動画入力、メタデータ、ROI、めくりと静止検出、ベストフレーム、手の重なり評価、
重複除外、射影変換、左右分割、画像/PDF保存、ログ、レビューまで。
除外はmanifestのフラグで行い、重複候補も画像を残す。
初回解析で自動PDF生成。編集後は `pdf_stale=true` とし再出力を明示する。

改善余地: optical flow併用、ページ単位で別々のベストフレーム選択、曲面推定、
追跡によるROI移動、複数動画の統合、画像追加、ドラッグ並べ替え、ジョブ再開。

## 4. アルゴリズム

### 時間・動き

FFmpegの `setpts=PTS-STARTPTS,fps=10,scale=...` でプレゼンテーション時刻を基準に
サンプルする。`index / sample_fps` はサンプルの時刻であって元動画のフレーム番号ではない。
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
透視補正で机を外し、中央で左右分割。auto splitは中央±4%の暗い縦谷を検索する実験機能。
自動外周補正は元ROIより外へ広げず、各点最大2.5%の移動まで。検出失敗はROI fallbackと警告。
デフォルトは手動ROI固定なので、漫画が動いた場合の背景混入を自動保証できない。

円筒dewarpは明示設定時だけ水平方向に既存画素をリサンプルする。文字行も生成AIも使わない。
この単純モデルは実際の本の曲面を推定しない。既定無効。

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
- 時間間隔警告はページ欠落の証明にならない。OCRなしで実ページ番号は分からない。
- FFmpegは圧縮動画の中間フレームを内部でdecodeする。4K全フレームをPythonで解析する
  ことは避けるが、decode自体をサンプル数まで削減できるわけではない。
- 長いGOPを何度もseekすると候補抽出が遅い。今後は候補を時間順にまとめてdecode可能。
- Pythonに保持する高解像度画像は一見開きずつ。PDF生成はReportLabが圧縮ページを
  内部保持するため、長大な本ではPDFのサイズに応じてメモリを使う。
- 背景スレッドはブラウザを閉じても続くが、サーバー終了・Macスリープ中は完了しない。
  異常終了後は同一プロジェクトで最初から再解析。処理途中からの自動再開は未実装。

