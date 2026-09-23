# 開発・テスト

このページは、Manga Scanを変更する開発者向けの手順です。
通常の利用だけなら、[README](../README.md) のセットアップと起動手順だけで十分です。

## 開発用の依存関係を入れる

フロント依存は `frontend/package-lock.json` を正とし、`npm install` ではなく `npm ci` を使います。依存を更新するときだけ `npm install` で lockfile も更新し、その差分をレビューしてください。

初回、または開発用依存をまだ入れていない場合に実行します。

```bash
source .venv/bin/activate
python -m pip install -e '.[hands,dev]'
npm --prefix frontend ci --no-audit --no-fund
```

## フロントエンドを変更したとき

### 通常の `http://127.0.0.1:8765` で確認する

React/Viteの変更は、そのままではFlask側に反映されません。
先にフロントをbuildしてから起動します。

```bash
source .venv/bin/activate
npm --prefix frontend run build
manga-scan ui --config config.toml --projects projects
```

build結果は `src/manga_scan/static/` に出力されます。

`frontend/package.json` の依存関係も変更した場合や、`node_modules` がない場合は、build前に次も実行してください。

```bash
npm --prefix frontend ci --no-audit --no-fund
```

### Viteの開発サーバーで確認する

フロントを何度も変更する場合は、毎回buildするよりViteの開発サーバーを使う方が便利です。

ターミナル1:

```bash
source .venv/bin/activate
manga-scan ui --config config.toml --projects projects
```

ターミナル2:

```bash
npm --prefix frontend run dev
```

ブラウザでは **http://127.0.0.1:5173** を開きます。

Viteが `/api` と `/files` を `127.0.0.1:8765` のFlaskへproxyします。
通常の `8765` 側で最終確認するときは、最後にもう一度 `npm --prefix frontend run build` を実行してください。

## テスト

一通り確認する場合は、次の順で実行します。

```bash
source .venv/bin/activate

npm --prefix frontend test
npm --prefix frontend run test:layout
npm --prefix frontend run build

ruff check src tests scripts
python -m pytest -q
```

PythonのUI統合テストは、Viteで生成された `src/manga_scan/static/` をFlaskから実際に配信できることも確認します。
そのため、**Pythonテストより先にフロントをbuild**してください。
`test:layout` はローカルのViteサーバーとChromeのヘッドレスモードを使い、複数の画面幅でページカードのはみ出しを確認します。

### フロントだけ確認する

```bash
npm --prefix frontend test
npm --prefix frontend run test:layout
npm --prefix frontend run build
```

### Pythonだけ確認する

フロントのbuildが済んでいる状態で実行します。

```bash
ruff check src tests scripts
python -m pytest -q
```

## GitHub Actions

PRではGitHub Actionsが以下を確認します。

- Apple Silicon macOS / Python 3.14: フロントテスト、Chromeでのページカード幅確認、Vite build、Ruff、軽量Pythonテスト
- Apple Silicon macOS / Python 3.11: UI以外のPythonテストと最低対応Pythonのソース互換性
- Apple Silicon macOS / Python 3.14: FFmpegを使うパイプライン統合テスト

同じPRへ新しいcommitをpushした場合、古い実行は自動キャンセルされます。
