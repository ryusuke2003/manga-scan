# セットアップガイド

Manga Scan LocalをMacで初めて起動するまでの手順です。

## 対象環境

主な対象環境は **Mac Apple Silicon / arm64** です。

| 必要なもの | 目安 | 用途 |
|---|---|---|
| macOS arm64 | Apple Silicon | 主な検証環境 |
| Python | 3.11以上。推奨3.12〜3.14 | 画像処理・ローカルAPI |
| FFmpeg / ffprobe | Homebrew版で可 | 動画解析 |
| Node.js | 22.12以上 | React/Viteの初回ビルド |
| Homebrew | 任意だが推奨 | Python / FFmpeg / Nodeの導入 |

Homebrewがない場合は [brew.sh](https://brew.sh/) から導入してください。

## 1. リポジトリを取得

```bash
git clone https://github.com/ryusuke2003/manga-scan.git
cd manga-scan
```

## 2. 必要なツールを入れる

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

フロントエンド依存は `frontend/package-lock.json` で固定し、
セットアップ/CIとも `npm ci` で同一バージョンを再現します。

## 3. Python環境を作る

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hands]'
```

通常利用では `dev` 依存は不要です。
開発やテストを行う場合は [開発・テストガイド](development.md) を参照してください。

## 4. React/Viteフロントをビルド

```bash
npm --prefix frontend ci --no-audit --no-fund
npm --prefix frontend run build
```

ビルド結果は `src/manga_scan/static/` に生成されます。
このディレクトリは生成物なのでGit管理しません。

## 5. 手検出モデルと設定ファイルを用意

```bash
python scripts/download_hand_model.py
cp config.example.toml config.toml
```

`download_hand_model.py` はMediaPipe公式モデルを取得し、
スクリプト内に固定したSHA-256と照合します。

`config.toml` は端末ごとのローカル設定としてGit管理しません。
共有するデフォルト値を変更する場合は `config.example.toml` を更新してください。

## 6. 起動

```bash
manga-scan ui --config config.toml --projects projects
```

ブラウザで **http://127.0.0.1:8765** を開けば準備完了です。

サーバーは `127.0.0.1` のみで待ち受けます。
外部CDN、解析タグ、クラウド通信はありません。

## 入力サイズの安全上限

入力によるメモリ/CPU枯渇を避けるため、デフォルトでは次の上限があります。

- 静止画: 60,000,000 pixels
- 静止画の最大辺: 12,000px
- 静止画フォルダ: 最大5,000枚
- 動画: 40,000,000 pixels/frame
- 動画の最大辺: 8,192px
- 動画の合計長: 最大4時間

必要なら `config.toml` の `max_image_*` / `max_video_*` で上限をさらに引き下げられます。
安全上限を超える値は設定できません。

## 2回目以降

### フロントに変更がない場合

通常はこれだけです。

```bash
cd manga-scan
source .venv/bin/activate
manga-scan ui --config config.toml --projects projects
```

### フロントを変更した場合

`frontend/` 以下を変更した場合や、`git pull` でフロントの変更を取り込んだ場合は、
起動前にReact/Viteをbuildし直します。

```bash
cd manga-scan
source .venv/bin/activate
npm --prefix frontend run build
manga-scan ui --config config.toml --projects projects
```

`frontend/package.json` が変更されている場合や `node_modules` がない場合は、
build前に次も実行してください。

```bash
npm --prefix frontend ci --no-audit --no-fund
```

フロントを継続的に変更する場合は [開発・テストガイド](development.md) の
Vite開発サーバー手順が便利です。

## 次に読む

- 実際の操作: [Web UI・撮影ガイド](usage.md)
- 補正や自動判定の調整: [設定・チューニングガイド](configuration.md)
- 起動できない場合: [トラブルシューティング](troubleshooting.md)
