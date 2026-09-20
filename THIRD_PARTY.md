# OSS・参照実装・ライセンス調査

調査日: 2026-09-20。README、公式ドキュメント、公式LICENSEを参照。
FlipScanを含む参照アプリのソースコードをコピー・翻訳・移植していません。
本プロジェクトの独自コードはMIT。依存・モデル・外部バイナリの条件は個別に適用されます。

| 対象 | ライセンス・一次情報 | 流用できるもの / 今回の判断 | 用途との適合 |
|---|---|---|---|
| FlipScan | [PolyForm Noncommercial 1.0.0](https://raw.githubusercontent.com/JamesDavid/FlipScan/main/LICENSE.md) | [README](https://github.com/JamesDavid/FlipScan)にある候補評価・suspect表示・段階構成を設計参考にする。コード、依存、設定実装は取り込まない。商用利用制限がありOSIの意味でのOSSではない | OCR/LLM部分は今回不要。動画から候補を選ぶUXのみ有用 |
| Scan Studio | [Apache-2.0](https://raw.githubusercontent.com/bwerick/scanstudio/main/LICENSE) | 条件を満たせばコード再利用可能だが今回不使用。[README](https://github.com/bwerick/scanstudio)の段階処理・確認工程を参考にする | 本の動画→ページ画像という入力形態に合う。漫画の網点は強い二値化を避ける |
| UVDoc | [MIT](https://github.com/tanguymagne/UVDoc/blob/main/LICENSE) | コードは表示義務等を守って再利用可能。MVPは未採用。重み・データ・推移依存の条件は導入時に別確認 | 文字ベースラインに依存しないgrid unwarpingが将来候補。ただし漫画への変形品質・性能は実測が必要 |
| ScanTailor Advanced | [GPL-3.0](https://github.com/4lex4/scantailor-advanced/blob/master/LICENSE) | 商用利用自体は可能だが配布時にGPL義務がある。MVPへのリンク・移植なし | crop、deskew、内容領域確認の考え方は有用。動画と手の選択は別問題 |
| MediaPipe | [Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) | 依存として使用。Hand Landmarker Tasks APIを呼ぶ | 完全ローカルで手のlandmarkを取得。ただしページ上の指の厳密なmaskではない |
| OpenCV 4.5+ | [Apache-2.0](https://opencv.org/license/) | `opencv-contrib-python` wheel一種類に統一 | 差分、Laplacian、輪郭、homography、画像処理に適する |
| FFmpeg / ffprobe | [LGPL-2.1-or-laterを基本とし、ビルド構成でGPL等](https://ffmpeg.org/legal.html) | ユーザーがインストールした外部コマンドとして利用。バイナリ同梱なし。Homebrew版はGPLオプションを含み得る | MOV/MP4、HEVC、回転、seek、VideoToolboxに適する |

## アプリの直接依存

| パッケージ | ライセンス | 用途 |
|---|---|---|
| NumPy | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | 数値配列 |
| opencv-contrib-python | [MIT（パッケージ） / OpenCV Apache-2.0](https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt) | 画像処理。wheel内の第三者条件も確認対象 |
| Pillow | [HPND / MIT-CMU（PIL由来）](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | サムネイル、JPEG設定 |
| Flask | [BSD-3-Clause](https://github.com/pallets/flask/blob/main/LICENSE.txt) | loopback UI |
| ReportLab | [BSD](https://www.reportlab.com/opensource/) / インストール配布物のLICENSE.txt | JPEGパススルー、PNG画素保持PDF |
| MediaPipe（hands extra） | Apache-2.0 | 手検出 |
| React / React DOM | [MIT](https://github.com/facebook/react/blob/main/LICENSE) | Web UI |
| Vite | [MIT](https://github.com/vitejs/vite/blob/main/packages/vite/LICENSE.md) | フロントの開発サーバー・build。公開パッケージにはMIT以外の許容ライセンスのbundled dependencyも含む |
| @vitejs/plugin-react | [MIT](https://github.com/vitejs/vite-plugin-react/blob/main/LICENSE) | React/Vite build（開発時） |
| Vitest / React Testing Library / jsdom（開発のみ） | [MIT](https://github.com/vitest-dev/vitest/blob/main/LICENSE) / [MIT](https://github.com/testing-library/react-testing-library/blob/main/LICENSE) / [MIT](https://github.com/jsdom/jsdom/blob/main/LICENSE.txt) | フロントエンドテスト |
| pytest / Ruff / pypdf（開発のみ） | MIT / MIT / BSD-3-Clause | Pythonテスト、静的検査、PDFの検査 |

MediaPipeの推移依存にはNumPy、OpenCV contrib、Matplotlib（PSF系）、absl-py（Apache-2.0）、
flatbuffers（Apache-2.0）、sounddevice（MIT）、CFFI（MIT）、certifi（MPL-2.0）等があります。
音声機能・カメラ・マイクは本アプリでは呼び出しません。
FlaskのWerkzeug/Jinja2/MarkupSafe等、PillowやOpenCV wheel内のネイティブライブラリにも
それぞれ条件があります。フロント側もVite/Rollup等の推移依存を持ちます。
`requirements-macos-arm64.lock.txt` はPython側の検証時解決結果を保存するもので、npm依存のlockfileではありません。
全推移依存がMIT/Apache/BSDだけという主張はしません。再配布用アプリを作る場合は、
実際に含めるwheel・FFmpegビルド・モデルに対応した通知とライセンス文書を同梱してください。

## Hand Landmarkerモデル

- [公式モデルガイド](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
  がリンクする [公式モデルカード](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf)
  の2ページ目にApache License, Version 2.0の記載があります。
- 取得対象は `hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`。`latest`を使いません。
- 検証時SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`。
- モデルはGitリポジトリに同梱せず、明示的なセットアップスクリプトで取得します。
  URLと実ファイルのSHAを隣接JSONへ保存し、使用したSHAをプロジェクトmanifestにも記録します。
- モデルカードは隠れた手・物を持つ手等の制約も挙げています。ページの端だけに映った指を
  完全検出できるとは扱わず、候補画像と推定マスクを確認できるようにしています。

