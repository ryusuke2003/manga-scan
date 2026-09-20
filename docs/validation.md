# 検証記録

2026-09-20、macOS arm64 / Apple M5、Python 3.14.6、FFmpeg 9.0.1、
OpenCV 4.14.0、MediaPipe 0.10.35。依存の全バージョンは
`requirements-macos-arm64.lock.txt` に記録。

## 自動テスト

2026-09-20のMacローカル検証では `python -m pytest -q`: **36 passed**。現在のリポジトリではReact/Vite側のsmoke testとbuild検証も追加している。

- ROI内motionの同一/変化、連続静止、末尾flush、短い静止の棄却。
- 候補が時間区間の前半/後半へ分散すること。
- dHash/SSIM、白紙を自動除外しないこと、片側だけ変わった見開きの保持。
- 品質スコア各項目の減点、無効化した手検出の明示。
- 手maskのROI交差、複数maskの重複を二重計上しないこと。
- ROIの透視補正、机の除外、自己交差/範囲外/NaN/誤順序の拒否。
- 奇数幅の左右分割で画素を欠落させないこと、auto spine、円筒リマップ。
- PNGからPDFへ画素一致、JPEGの圧縮データ一致、PDFページ比率、OCRテキストなし。
- 失敗したPDF出力が以前のPDFを壊さないこと。
- 合成動画の4静止区間 → 1重複除外 → 6ページPDF。
- 候補切替で除外状態を維持、左右交換、手動追加、PDF再出力。
- 60/120/240fps入力でも解析を10fpsに制限。
- 回転メタデータ、可変fps、4K候補の元解像度取得。
- localhost UIのHost制限、Origin検査、変更操作のtoken、Vite生成アセットの静的配信。

フロントエンドには、ローカルファイルURLのエンコードとROI座標正規化のsmoke testがある。
通常の検証順は次の通り。

```bash
npm --prefix frontend test
npm --prefix frontend run build
python -m pytest -q
ruff check src tests scripts
```

PythonのUI統合テストは、Viteの`index.html`が参照するハッシュ付き`/static/assets/...`を実際にFlaskから取得できることを確認する。そのためpytest前にfrontend buildが必要。

GitHub Actionsはpush / pull requestで、Ubuntu・macOS × Python 3.12・3.14を実行する。各jobでNode 22をセットアップし、フロントのtest/build後にPythonのlint/testを行う。個々のCI実行結果はこの文書へ固定せず、GitHub Actions側を参照する。

## M5上のローカル実行

`scripts/make_demo.py` で生成した480×320 / 30fps / 6秒の動画を使い、
**MediaPipe有効・候補7枚/見開き**でCLIとブラウザの両方から処理。

- 4見開き、8保存ページ、重複2ページを除いた6ページのPDF。
- `projects/demo/output/manga.pdf` と `debug/contact_sheet.jpg` を生成・確認。
- CLIの処理本体: **1.77秒**（manifestのelapsed_seconds）。
- これはモデル初期化・初回フォントキャッシュ作成・プロジェクト初期化を含まない。
- 小さな合成入力の結果なので、実写4Kの所要時間へ外挿しない。

MediaPipe公式Pythonチュートリアルのサンプル画像を一時領域に取得して陽性推論を検証。
960×640画像、画像全体のROIに対してhand overlap **0.07396484375**、mask **45,444画素**。
これは手検出APIの実動作確認であり、本の端に指だけが映る場合の精度評価ではない。
その外部サンプル画像はリポジトリへ同梱していない。

## ブラウザ

- 新規プロジェクト、ローカル動画パス入力、メタデータ表示。
- CanvasでTL/TR/BR/BLの四隅をクリックし4点の確定を確認。
- 抽出開始、処理完了、6ページ/重複見開きの表示、PDFリンクを確認。
- 除外/復元、候補切替、左右交換、再出力のデータ更新はPython結合テストで検証。

## 未検証

実写漫画の手検出率、ページ欠落率、4K長時間動画の処理時間・最大メモリ、
HDR/Dolby Visionの色再現、端末ごとのcodec差、強い湾曲、macOS以外の実機UI、
VideoToolboxの機種ごとの性能、パッケージ再配布物の完全なライセンス監査。

