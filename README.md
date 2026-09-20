# Manga Scan Local

Macで撮影した漫画の動画から、**机などの背景を除いた左右ページ画像とPDF**を作るローカルアプリです。

- Python / OpenCV / FFmpeg / MediaPipeで画像処理
- React + ViteのローカルWeb UI
- ページめくり中を避け、静止した候補からベストフレームを選択
- 任意で左右ページの外周を自動検出し、各ページを別々に台形補正
- 任意で低周波の照明ムラ・緩い影をページ単位に補正
- 手の重なり、ブレ、重複候補などを「要確認」として表示
- 候補切替、除外/復元、ページ順、左右交換、分割位置を後から修正可能
- OCR、クラウドAPI、有料API、生成AIによる画像補完なし
- セットアップ後のスキャン処理はオフラインで実行可能

> [!IMPORTANT]
> このMVPは、各見開きを **0.5〜1秒程度静止して撮影する**使い方を想定しています。
> 高速なパラパラめくり、隠れた絵、強く湾曲したページの完全復元はできません。

## 最短セットアップ

主な対象環境は **Mac Apple Silicon** です。

| 必要なもの | 目安 | 用途 |
|---|---|---|
| macOS arm64 | Apple Silicon | 主な検証環境 |
| Python | 3.11以上。推奨3.12〜3.14 | 画像処理・ローカルAPI |
| FFmpeg / ffprobe | Homebrew版で可 | 動画解析 |
| Node.js | 22.12以上 | React/Viteの初回ビルド |
| Homebrew | 任意だが推奨 | Python / FFmpeg / Nodeの導入 |

Homebrewがない場合は [brew.sh](https://brew.sh/) から導入してください。

### 1. リポジトリを取得

```bash
git clone https://github.com/ryusuke2003/manga-scan.git
cd manga-scan
```

### 2. 必要なツールを入れる

```bash
brew install python@3.14 ffmpeg node
```

確認:

```bash
python3.14 --version
ffmpeg -version
ffprobe -version
node --version
```

Node.jsは **22.12以上** が必要です。

### 3. Python環境を作る

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hands]'
```

通常利用では `dev` 依存は不要です。テストや開発をする場合だけ後述の開発手順を使ってください。

### 4. React/Viteフロントをビルド

```bash
npm --prefix frontend install --no-audit --no-fund --no-package-lock
npm --prefix frontend run build
```

ビルド結果は `src/manga_scan/static/` に生成されます。このディレクトリは生成物なのでGit管理しません。

### 5. 手検出モデルと設定ファイルを用意

```bash
python scripts/download_hand_model.py
cp config.example.toml config.toml
```

`download_hand_model.py` はMediaPipe公式モデルを取得し、スクリプト内に固定したSHA-256と照合します。

### 6. 起動

```bash
manga-scan ui --config config.toml --projects projects
```

ブラウザで **http://127.0.0.1:8765** を開けば準備完了です。

サーバーは `127.0.0.1` のみで待ち受けます。外部CDN、解析タグ、クラウド通信はありません。

### 2回目以降

フロントを変更していなければ、通常はこれだけです。

```bash
cd manga-scan
source .venv/bin/activate
manga-scan ui --config config.toml --projects projects
```

## Web UIの使い方

1. **動画を選ぶ**
   - `.mov` / `.mp4` のローカルファイルを選択します。
   - ブラウザへ巨大動画をアップロードするのではなく、ローカルパスだけを渡します。
2. **読み方と出力を選ぶ**
   - 日本漫画なら通常は「右 → 左」。
   - PNGは画質優先、JPEGは容量優先です。
3. **表紙を追加する（任意）**
   - 表紙が映っている時刻を `±0.1秒 / ±1秒` で選びます。
   - 表紙の外周だけを4点指定し、1ページとして保存します。
   - 表紙が不要ならスキップできます。
4. **見開きの基準フレームを選ぶ**
   - 最初に本を開いて左右2ページが見えている時刻を選びます。
   - この時刻より前は、自動の見開き検出から除外されます。
5. **見開きの外周を4点指定**
   - `左上 → 右上 → 右下 → 左下` の順にクリックします。
   - 表紙用ROIとは別で、実際の見開きサイズに合わせます。
6. **抽出を開始**
   - 基準時刻以降から動きが小さい区間を探し、候補フレームを評価します。
7. **要確認ページと動画タイムラインをチェック**
   - 手の重なり、ブレ、重複候補、時間間隔などを確認します。
   - 検出済み見開きを動画全体のタイムラインで確認できます。
   - 普段より長い見開き間隔は「欠落候補」として表示され、候補時刻を手動追加欄へ入れられます。
8. **必要なら修正**
   - 候補フレーム切替
   - ページ除外 / 復元
   - 前後移動
   - 左右交換
   - 分割位置修正
9. **見落とした見開きを追加**
   - 動画の秒数を指定して追加できます。
   - タイムラインの欠落候補は推定なので、元動画を確認してから追加してください。
10. **PDFを出力**
   - 編集後は「PDFを出力」で変更を反映します。

除外は非破壊です。自動で重複除外されたページも「除外ページも表示」から復元できます。

左のプロジェクト一覧から不要なプロジェクトを削除できます。削除すると、そのプロジェクト内の生成ページ・候補画像・PDFも消え、元に戻せません。プロジェクト外の元動画ファイルは削除しませんが、`--copy-source` でプロジェクト内に作った動画コピーは一緒に削除されます。

ブラウザを閉じてもサーバープロセスが動いている間は処理を続けます。ターミナルを終了すると停止します。
Macのスリープを避ける場合は次のように起動できます。

```bash
caffeinate -i manga-scan ui --config config.toml --projects projects
```

## 撮影のコツ

- スマホを真上の固定スタンドに置く
- 表紙を撮ったあと、本を開いた最初の見開きでも少し静止する（ここを基準フレームにする）
- 見開きを画面いっぱいに入れる
- 本とスマホを途中で動かさない
- 反射や強い影を避け、可能ならAF/AEを固定する
- まずは **4K・30/60fps・SDR** を推奨
- 1枚めくるごとに **0.5〜1秒程度静止**する
- 静止中は指をページから離す
- 最後の見開きも少し静止してから録画を止める

120/240fps入力も扱えますが、解析フレーム数は設定したsample fpsまで落とします。iPhoneスローモーションの実撮影fpsを推測して時間を変換することはしません。

## よくあるエラー

### `manga-scan: command not found`

仮想環境を有効にしてください。

```bash
source .venv/bin/activate
```

### `Frontend build missing`

React/Viteの生成物がありません。

```bash
npm --prefix frontend install --no-audit --no-fund --no-package-lock
npm --prefix frontend run build
```

### `Hand model missing`

手検出モデルを取得してください。

```bash
python scripts/download_hand_model.py
```

すでに `models/hand_landmarker.task` が存在する場合、ダウンロードスクリプトは上書きしません。

手検出なしで動作確認だけする場合は `config.toml` の次の値を変更できます。

```toml
hand_backend = "none"
```

この場合、手の重なりを判定できないため全ページが要確認扱いになります。

### `ffmpeg` / `ffprobe` が見つからない

```bash
brew install ffmpeg
```

### ポート8765が使用中

別ポートで起動できます。

```bash
manga-scan ui --config config.toml --projects projects --port 8766
```

### `No stable intervals found`

各見開きで静止する時間を長くしてください。それでも検出できない場合は `config.toml` の `stable_frames` や `motion_threshold` を調整します。

### MediaPipeがmacOSで異常終了する

- Rosettaとarm64 Pythonを混在させない
- 通常のMacターミナル/GUIセッションから実行する
- `mediapipe==0.10.35` の固定をむやみに外さない

検証環境では新しいMediaPipe版でCPU指定時にもMetal helper内部の異常終了を確認したため、現在は0.10.35に固定しています。

## CLIで使う

Web UIを使わず処理することもできます。

```bash
# 動画情報だけ確認
manga-scan probe '/path/to/video.mov'

# プロジェクト作成 + 解析
manga-scan scan '/path/to/video.mov' projects/book01 \
  --config config.toml \
  --roi '[[0.10,0.10],[0.90,0.10],[0.90,0.90],[0.10,0.90]]'

# 初期化だけして、あとでUIから四隅を指定
manga-scan init '/path/to/video.mov' projects/book02 --config config.toml

# 失敗・中断したプロジェクトを最初から再解析
manga-scan run projects/book02

# 12.4秒の見開きを追加
manga-scan add projects/book01 12.4

# 現在のページ順でPDFを再出力
manga-scan export projects/book01
```

`--copy-source` を付けない限り、元動画はプロジェクトへコピーせず参照だけを保存します。候補の再取得に必要なので、レビュー中は元動画を移動・削除しないでください。

## 出力

主な出力は次の通りです。

```text
projects/book01/
├── manifest.json              # ページ順、除外状態、候補、警告、進捗
├── config.resolved.json       # 作成時の設定スナップショット
├── source/
│   ├── video.json
│   ├── first_frame.png
│   ├── cover_frame.png        # 表紙を使う場合
│   └── reference_frame.png    # 見開き基準フレーム
├── candidates/                # 候補フレーム・手マスク・評価値
├── selected/                  # 採用した見開き
├── pages/                     # 左右に分割したページ画像
├── debug/                     # motion / score / ログ / contact sheet / dewarp比較
└── output/
    └── manga.pdf
```

PDFのページ順は `manifest.json` の `pages` 配列で管理します。画像ファイル名の並び順には依存しません。

- PNGページ: PDF内でも画素を維持
- JPEGページ: 保存済みJPEGをPDFへ再圧縮せず埋め込み
- `pdf_dpi`: 物理サイズを決める値で、画像を縮小しません

## 自動湾曲補正

Issue #10 Phase 3として、見開きを左右に分割した後の各ページについて、背表紙側の横方向圧縮を画像内容から推定する自動dewarpを利用できます。

```toml
dewarp_mode = "auto"
dewarp_max_strength = 0.25
dewarp_min_confidence = 0.6
```

自動推定は、ページ内を複数の高さ帯に分けて縦エッジ間隔を測り、背側だけが一貫して詰まっている場合に限って補正します。推定のばらつきが大きい場合は補正せず、ページを「要確認」にします。完全な3D復元や文字認識は行いません。

レビュー画面では、ページごとに推定強度・信頼度を確認できます。自動補正を適用したくないページは「自動補正OFF」で片側だけ無効化できます。自動モードでは `debug/dewarp/` に補正前PNGと、推定した横方向remapを確認するグリッド画像を保存します。

従来の固定補正も残しています。

```toml
dewarp_mode = "manual"
dewarp_strength = 0.15
```

`dewarp_mode = "off"` なら湾曲補正を完全に無効化します。表紙は背表紙側を一意に決められないため、自動モードの対象外です。表紙に固定補正をかけたい場合はmanualモードを使用してください。

補正順は **湾曲補正 → 照明ムラ補正 → 白背景正規化 → grayscale / contrast / rotation** です。

## チューニング

変更後の自動解析は新しいプロジェクトで試すのがおすすめです。プロジェクト作成時の設定はスナップショット保存されます。

| 症状 | 主な調整 |
|---|---|
| 候補がない / 少ない | 静止時間を伸ばす。`stable_frames` を5→3、sample fpsを10→15 |
| 微振動で静止判定されない | `motion_threshold` を少し上げる |
| 別ページが1区間になる | `turn_threshold` を下げる |
| 同じページが繰り返される | `turn_threshold` を上げる / 重複SSIMを少し下げる |
| 指の少ない候補を拾わない | `candidates_per_spread`、`hand_overlap_weight` を上げる |
| 左右ページで台形の向きが違う | `perspective_mode="per_page"`。輪郭検出に自信がない見開きは自動で従来方式へfallback |
| 自動ページ輪郭が不安定 | `page_contour_min_confidence` を上げるとfallbackしやすくなる。従来方式へ固定するなら `perspective_mode="spread"` |
| 背の位置がずれる | UIで分割位置を修正。必要なら `split_mode="auto"` |
| ページの端/中央が緩く暗い | `illumination_correction=true`。強すぎる場合は `illumination_strength` を0.4〜0.7へ下げる |
| 紙が黄ばみ/グレーに見える | `white_normalization=true`。まず `white_strength=0.6`, `white_target=245` から |
| 黒ベタや網点が変わる | `illumination_correction=false`, `white_normalization=false`, `contrast=1.0`, `dewarp_mode="off"`, PNG |
| PDFが大きい | `image_format="jpeg"`, `jpeg_quality=90` 前後 |
| decodeが遅い | Macでは `hwaccel="videotoolbox"` を試す |

ページ別台形補正はデフォルトでは従来互換の `perspective_mode="spread"` です。
`"per_page"` にすると元フレーム上で左右ページの外周を自動検出し、左右を別々の射影変換で補正します。
両ページの輪郭confidenceが閾値に届かない見開きは `page_contour_low_confidence` として要確認にし、従来の「見開き全体を補正 → 左右分割」へ自動fallbackします。
検出quadとconfidenceはmanifestに残り、`debug/page_contours/` に確認画像を保存します。

照明ムラ補正はデフォルトOFFです。ON時はページ単位の低周波な明るさだけを均し、二値化や背景除去は行いません。
黒ベタや網点への影響が気になる場合は `illumination_strength` を下げるかOFFにしてください。

白背景正規化はデフォルトOFFです。ONでも暗部はほぼ触らず、明るい紙面候補だけを白へ寄せます。
薄いトーンを残したい場合は `white_strength` を下げてください。

詳しい判定式やアルゴリズムは [docs/architecture.md](docs/architecture.md) を参照してください。

## 開発する場合

### 開発依存を入れる

```bash
source .venv/bin/activate
python -m pip install -e '.[hands,dev]'
npm --prefix frontend install --no-audit --no-fund --no-package-lock
```

### テスト

```bash
npm --prefix frontend test
npm --prefix frontend run build
python -m pytest -q
ruff check src tests scripts
```

PythonのUI統合テストは、Viteで生成された `src/manga_scan/static/` をFlaskから実際に配信できることも確認します。そのため **Pythonテストの前にフロントをbuild** してください。

GitHub Actionsでは、Ubuntu/Python 3.14でNode 22のフロントテスト・ビルドと軽量Pythonテストを先に実行します。Ubuntu/Python 3.12ではUI以外のPythonテストとソース互換性を確認し、macOS/Python 3.14ではFFmpegを使うパイプライン統合テストを実行します。PRブランチではpushとpull_requestの二重実行を避け、同じPRへの古い実行は新しいpush時に自動キャンセルします。

### React/Viteを開発モードで動かす

ターミナル1:

```bash
source .venv/bin/activate
manga-scan ui --config config.toml --projects projects
```

ターミナル2:

```bash
npm --prefix frontend run dev
```

開発中は **http://127.0.0.1:5173** を開きます。Viteが `/api` と `/files` を `127.0.0.1:8765` のFlaskへproxyします。

フロント変更を通常の `8765` 側へ反映したいときは、再度 `npm --prefix frontend run build` を実行してください。

## 制約

- 10fps解析では、約0.5秒未満しか安定して見えないページを取りこぼす可能性があります
- 手で常に隠れている領域は復元できません
- MediaPipeは指先だけの手や漫画に描かれた手を誤判定する可能性があります
- 射影変換は平面を仮定するため、背の強い湾曲は完全には補正できません
- HDR / Dolby Visionの色忠実度は未検証で、MVPは8bit出力です
- 見開き基準フレーム以降は固定ROIなので、本自体が大きく移動すると背景が混ざる可能性があります
- OCRを使わないため、実際のページ番号から欠落を自動推定しません
- 処理途中からの自動再開、複数動画統合、外部画像追加は未実装です

実写漫画での精度や長時間4K動画の性能はまだ十分に評価できていません。最初は数ページの短い動画で撮影条件を調整してください。

## ドキュメント

- [設計・アルゴリズム・MVPの境界](docs/architecture.md)
- [検証記録](docs/validation.md)
- [OSS・モデル・ライセンス調査](THIRD_PARTY.md)
- [設定例](config.example.toml)

## ライセンス

このリポジトリの独自コードはMIT Licenseです。

依存ライブラリ、MediaPipeモデル、ユーザーが別途インストールするFFmpegにはそれぞれのライセンスが適用されます。詳細は [THIRD_PARTY.md](THIRD_PARTY.md) を参照してください。

動画、モデル、プロジェクト出力、仮想環境、Viteのbuild生成物はGit管理対象外です。公開前に私有動画や漫画ページが含まれていないことを確認してください。
