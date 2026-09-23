# CLI・出力ガイド

Web UIを使わずに処理する場合のCLIと、
生成されるプロジェクトファイルの概要です。

## CLI

### 動画情報を確認

```bash
manga-scan probe '/path/to/video.mov'
```

### プロジェクトを作成して解析

```bash
manga-scan scan '/path/to/video.mov' projects/book01 --config config.toml --roi '[[0.10,0.10],[0.90,0.10],[0.90,0.90],[0.10,0.90]]'
```

### 初期化だけ行う

あとでUIから四隅を指定する場合:

```bash
manga-scan init '/path/to/video.mov' projects/book02 --config config.toml
```

### 失敗・中断したプロジェクトを最初から再解析

```bash
manga-scan run projects/book02
```

### 指定時刻の見開きを追加

```bash
manga-scan add projects/book01 12.4
```

### 現在のページ順でPDF / CBZを再出力

```bash
manga-scan export projects/book01
```

`--copy-source` を付けない限り、
元動画はプロジェクトへコピーせず参照だけを保存します。

候補の再取得に必要なので、レビュー中は元動画を移動・削除しないでください。

## プロジェクト構成

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
    ├── manga.pdf
    └── manga.cbz
```

PDF / CBZのページ順は `manifest.json` の `pages` 配列で管理します。
画像ファイル名の並び順には依存しません。

## PDF / CBZの保存仕様

- PNGページ: PDF内でも画素を維持
- JPEGページ: 保存済みJPEGをPDFへ再圧縮せず埋め込み
- CBZ: PNG/JPEGの保存済みバイト列をそのまま格納
- CBZのZIP圧縮: `ZIP_STORED`（再圧縮なし）
- `pdf_dpi`: PDFの物理サイズを決める値で、画像を縮小しない

CBZ内では有効ページを
`001.png`, `002.png` … の連番へ並べます。
JPEG設定なら拡張子は `.jpg` です。

## 見開きと左右分割

出力レイアウトの考え方や補正設定は
[設定・チューニングガイド](configuration.md) を参照してください。

## 関連ドキュメント

- Web UIでの操作: [Web UI・撮影ガイド](usage.md)
- 設定値: [config.example.toml](../config.example.toml)
