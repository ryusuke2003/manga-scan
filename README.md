# Manga Scan Local

Macで撮影済みの漫画動画から、机を除いた左右ページ画像と画像PDFを作るローカルMVPです。
Python / OpenCV / FFmpeg / MediaPipe。OCR、クラウドAPI、有料API、生成AIによる指消しはありません。
自動抽出後に候補の切替・除外・復元・ページ順・分割位置を確認できます。

**このMVPは、各見開きが約0.5秒以上安定して映る撮影を対象にします。**
一瞬だけ見える高速なパラパラめくり、隠れた絵、強く湾曲したページの完全復元はできません。
実写漫画での精度とMacBook Air M5での処理時間は未評価です。まず数ページで撮影条件を調整してください。

## セットアップ（Mac Apple Silicon）

macOSのarm64 Python 3.11以降。推奨はPython 3.12〜3.14です。Rosetta環境を混ぜないでください。

```bash
brew install python@3.14 ffmpeg node
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hands,dev]'
npm --prefix frontend install
npm --prefix frontend run build
python scripts/download_hand_model.py \
  --sha256 fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
cp config.example.toml config.toml
```

Homebrewが未導入の場合は[公式手順](https://brew.sh/)で導入してください。
`ffmpeg -version` と `ffprobe -version`、`node --version`（22.12以上）で確認できます。

依存とモデルの**インストール時だけ**ネット接続が必要です。その後はネットを切って処理できます。
別のオンラインPCで必要なwheelとモデルを用意して持ち込むこともできます。
検証環境の固定依存は `requirements-macos-arm64.lock.txt`。他のOS/Pythonでは
`pyproject.toml` から解決してください。FFmpegは別インストールです。

### MediaPipe依存

Hand Landmarker **Tasks API**を使用し、CPUで実行します。従来の `mp.solutions.hands` は使いません。
MediaPipeは **0.10.35** に固定しています。検証したMac arm64環境では1.0.1がCPU指定でも
Metal helper内部でネイティブ異常終了したため、動作確認なしでバージョンを上げないでください。
macOS版はCPU推論指定でも内部でgraphics contextを初期化します。強いサンドボックスや
GUIセッションのない環境では異常終了する場合があるため、通常のMacターミナルから起動してください。
`hand_model` はローカルの `.task` ファイルを指します。設定ファイルに記述した相対パスは
その設定ファイルのあるディレクトリから解決します。処理中の自動ダウンロードはありません。
`opencv-python` と `opencv-contrib-python` を同時に入れないでください。このプロジェクトは
MediaPipeと共通の `opencv-contrib-python` に統一しています。

モデルや依存がなければ明確なエラーで停止します。手検出なしで試すには設定の
`hand_backend = "none"` を明示してください。その場合は全ページが要確認になります。
ライセンスとモデルの出所は [THIRD_PARTY.md](THIRD_PARTY.md) を参照してください。

## Web UIで実行

Web UIはReact + Viteです。初回セットアップ後やフロントエンド変更後は
`npm --prefix frontend run build` で `src/manga_scan/static/` を生成します。

```bash
source .venv/bin/activate
manga-scan ui --config config.toml --projects projects
```

[http://127.0.0.1:8765](http://127.0.0.1:8765) をブラウザで開きます。
サーバーは127.0.0.1だけで待ち受けます。外部CDN、解析タグ、遠隔通信はありません。

1. 「ファイルを選択」でMacの動画を選択（または絶対パスを入力）。巨大動画をブラウザへアップロードしません。
2. 右→左 / 左→右、PNG / JPEG、手検出等を選び「動画を読み込む」。
3. 最初のフレームで **左上 → 右上 → 右下 → 左下** の順に紙の外周をクリック。
4. 「抽出を開始」。背景スレッドで処理し、進捗と中間データを保存します。
5. ページ一覧の要確認表示をチェック。画像クリックで元サイズ、候補一覧で手のマスクも確認できます。
6. 候補フレーム切替、ページ除外/復元、前後移動、左右交換、分割位置修正を行います。
7. 見落とした見開きは秒数を指定して追加できます。左右2ページが時系列位置に挿入されます。
8. 「PDFを出力」で編集を反映。「PDFを開く」で確認。

除外は非破壊です。「除外ページも表示」で自動重複除外も復元できます。
時刻指定追加では動きを測定しないため、必ず要確認扱いです。
候補切替はその見開き両ページを再生成しますが、除外状態と現在のページ順は保持します。
原動画は候補再取得に必要です。移動・削除しないでください。

ブラウザを閉じても処理は続きます。ターミナルを終了すると処理も止まります。
スリープを避けたい場合は `caffeinate -i manga-scan ui --config config.toml` を利用できます。

## CLIで実行

```bash
manga-scan probe '/path/to/video.mov'

manga-scan scan '/path/to/video.mov' projects/book01 \
  --config config.toml \
  --roi '[[0.10,0.10],[0.90,0.10],[0.90,0.90],[0.10,0.90]]'

# 初期化だけ行い、UIで四隅指定する場合
manga-scan init '/path/to/video.mov' projects/book02 --config config.toml
manga-scan ui --projects projects --config config.toml

# 中断・失敗したプロジェクトを最初から再解析
manga-scan run projects/book02

# 元動画の表示時刻12.4秒から見開きを追加 / PDF再出力
manga-scan add projects/book01 12.4
manga-scan export projects/book01
```

ROIは自動回転適用後の画像に対する0〜1の座標。`init --copy-source` / `scan --copy-source`
で動画を `source/video.mov` 等へコピーできます。既定は元ファイルへの参照のみです。
新規作成には空のディレクトリが必要です。完了済みプロジェクトを再解析で上書きしません。

## 出力

```text
projects/book01/
├── manifest.json                 # 順番、選択候補、除外、警告、処理状態
├── config.resolved.json          # 初期設定スナップショット
├── source/
│   ├── video.json                # 元ファイルとffprobe情報
│   ├── first_frame.png           # ROI指定用、表示方向適用済み
│   └── video.mov                 # --copy-source時だけ
├── frames_lowres/                # save_lowres=true時のみ画像を書き出す
├── candidates/spread_0001/
│   ├── candidate_00.png          # 縮小候補。フル画像は必要時に元動画から取得
│   ├── candidate_00_spread.png   # ROI補正プレビュー
│   ├── candidate_00_hand_mask.png
│   └── candidate_00.json        # 時刻、ROI、スコア内訳
├── selected/spread_0001.png      # 元解像度から補正した見開き
├── pages/
│   ├── spread_0001_right.png    # または.jpg
│   ├── spread_0001_left.png
│   └── ..._thumb.jpg            # UI用のみ。PDFには使わない
├── debug/
│   ├── process.log
│   ├── motion.csv
│   ├── intervals.json
│   ├── scores.csv
│   └── contact_sheet.jpg        # 80ページごとに分割
└── output/manga.pdf
```

PDF順は `manifest.json` の `pages` 配列で管理し、ファイル名順とは独立しています。
`selected/` は必ずPNGで保持し、`pages/` を設定形式で保存します。
PNGページはPDF内でも画素を維持（Flate圧縮。PNGファイルのバイト列そのものではありません）。
JPEGページは指定品質で一度だけ保存し、PDFには再圧縮せず埋め込みます。
`pdf_dpi` はPDFの物理寸法を決めるだけで、画像を縮小しません。ページごとに画像比率を保ちます。

## 推奨撮影

- スマホを真上の固定スタンドへ置き、見開きを画面いっぱいに。最初から本を開いた状態で撮影開始。
- 拡散照明を左右から当て、反射と影を避ける。AF/AEを固定できる場合は固定する。
- まず **4K・30/60fps・SDR** を推奨。120/240fpsは明るさと撮影端末の解像度制限に注意。
- 普通に一枚ずつめくった後、**0.5〜1秒程度静止**させ、指を紙から離す。
- 本・スマホを動かさず、最初のROI内で撮影。背をできるだけ平らにする。
- ページを飛ばさず一定方向に進む。日本漫画は右→左。表紙等は必要箇所を別途レビュー。
- 最後の見開きも静止してから録画終了。

30/60/120/240fpsをフレーム番号に依存せず扱います。iPhoneスローモーションの
編集済み動画は再生時間に従います。実際の撮影時刻/fpsを自動復元する機能はありません。

## チューニング

変更後の自動解析は新規プロジェクトで試してください。プロジェクトは初期設定を保存します。

| 症状 | 調整 |
|---|---|
| 候補がない / 少ない | `debug/motion.csv` を確認。`stable_frames` を5→3、sample fpsを10→15。撮影時の静止も延ばす |
| 微振動で静止にならない | `motion_threshold` を少し上げる。ただしブレた候補が増える |
| 別ページが一つの区間になる | `turn_threshold` を下げる。必ずmotion_threshold以上にする |
| 同じページが何度も候補になる | `turn_threshold` を上げる / 重複SSIMを少し下げる。ただし似たコマの誤除外に注意 |
| 指の少ない候補を拾わない | `candidates_per_spread` を7→12、`hand_overlap_weight` を上げる。候補とhand maskを確認 |
| 背の位置がずれる | UIの分割位置を修正。`split_mode="auto"` は黒いコマで誤るため任意 |
| 本の輪郭が少し変わる | `refine_quad=true`。確実に検出できない候補は初期ROIへ戻り警告 |
| 黒ベタや網点が変わる | `contrast=1.0`, `dewarp_strength=0.0`, PNGを維持 |
| PDFが大きい | `image_format="jpeg"`, `jpeg_quality=90` 程度で新規処理。PNGは無劣化だが大きい |
| decodeが遅い | `hwaccel="videotoolbox"` を試す。非対応時CPUへfallback。長GOPの候補seekが律速の場合もある |

`motion_threshold` は0〜1の平均画素差、`duplicate_hash_distance` は0〜64bitの距離です。
suspect: 低鮮鋭度、手の重なり、重複疑い、外周不確か、時間間隔異常、高motion等。
flatnessは3D湾曲ではなく四辺形の辺比を使う代理指標です。
詳細な式・状態機械・限界は [docs/architecture.md](docs/architecture.md) を参照してください。

## 制約・性能

- 縮小して5〜15fps程度を解析します。FFmpeg内部ではコーデック上必要な中間フレームもdecodeします。
- 高解像度は選択フレームのみ。候補を元動画へseekするため、長GOP動画では時間がかかります。
- 一見開きずつ処理し、フル動画をメモリ保持しません。PDF作成時は圧縮画像量に応じたメモリが必要です。
- 逐次パイプラインがMVPの既定です。M5専用最適化・multiprocessingは未導入。
- 数分以内という速度目標は動画長・codec・候補数に左右され、M5実写ベンチマークは未実施です。
- 本が動けば固定ROIの外にはみ出せます。追跡は未実装。台形補正だけでは湾曲や綴じ部の欠落は直りません。
- 手が検出されない場合、重なり0でも指がない保証にはなりません。特に指先だけ・手袋・漫画中の手に注意。
- 重複は直近の見開き単位。白紙や似たページを誤って消さないよう保守的です。
- HDR/Dolby Visionの色忠実度、実写での欠落率、端末別性能は未検証です。MVPでは8bit出力です。
- ROIの途中変更、自動的な欠落ページ番号推定、外部画像の追加、複数動画統合、自動再開は未実装です。

## テスト・サンプル

```bash
npm --prefix frontend test
npm --prefix frontend run build
python -m pytest -q
ruff check src tests scripts
python scripts/make_demo.py projects/demo-input.mp4 --fps 30
manga-scan scan projects/demo-input.mp4 projects/demo --config config.toml \
  --roi '[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]'
```

デモはオリジナルの図形で作った6秒動画。4静止区間（1つ重複）から6ページを期待します。
動き検出・重複・スコア・ROI・分割・手マスク交差・PDF画素保持・レビュー操作をテストします。
通常テストは手モデルを要求しません。実モデルの空画像推論は別途手動で検証します。
合成テストの成功は、実際の漫画の撮影精度を保証しません。
実行した項目・環境・測定値は [docs/validation.md](docs/validation.md) に記録しています。

## 公開・開発

MITライセンス。設計は [docs/architecture.md](docs/architecture.md)、OSS調査は
[THIRD_PARTY.md](THIRD_PARTY.md)。GitHub Actionsでテストを実行します。
動画・モデル・プロジェクト出力・仮想環境はGit管理対象外です。
公開前に私有動画や漫画ページが含まれていないことを確認してください。
