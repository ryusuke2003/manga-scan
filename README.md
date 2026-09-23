# Manga Scan Local

Macで撮影した漫画の動画やページ画像から、**見開き画像・PDF・CBZ**を作るローカルアプリです。
標準では見開きのまま保存し、必要に応じて左右ページへの分割も選べます。

## 主な機能

- ページめくり中を避け、静止した候補からベストフレームを自動選択
- 左右ページを別々に採点し、別時刻の候補を採用可能
- MediaPipeの手マスクと別候補の実画素を使った指領域の補修
- ページ外周検出、台形補正、湾曲補正、照明ムラ補正、白背景正規化
- 手の重なり、ブレ、重複候補、欠落候補をレビュー画面で確認
- 候補切替、除外/復元、並べ替え、左右交換、分割位置修正、Undo/Redo
- 動画だけでなくPNG / JPEG / WebPの静止画フォルダ入力にも対応
- OCR、クラウドAPI、有料API、生成AIによる画像補完なし
- セットアップ後のスキャン処理はオフラインで実行可能

> [!IMPORTANT]
> 各見開きを **0.5〜1秒程度静止して撮影する**使い方を想定しています。
> 高速なパラパラめくり、常に隠れている絵、強く湾曲したページの完全復元はできません。

## クイックスタート

対応環境は **Apple Silicon搭載Mac（macOS）のみ** です。Linux / Windows / Intel Macはサポートしません。

| 必要なもの | 目安 |
|---|---|
| macOS | Apple Silicon / arm64（対応環境） |
| Python | 3.11以上、推奨3.12〜3.14 |
| FFmpeg / ffprobe | Homebrew版で可 |
| Node.js | 22.12以上 |

```bash
git clone https://github.com/ryusuke2003/manga-scan.git
cd manga-scan

brew install python@3.14 ffmpeg node

python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hands]'

npm --prefix frontend ci --no-audit --no-fund
npm --prefix frontend run build

python scripts/download_hand_model.py
cp config.example.toml config.toml

manga-scan ui --config config.toml --projects projects
```

ブラウザで **http://127.0.0.1:8765** を開けば利用できます。

インストール理由、2回目以降の起動、入力サイズ上限などは
[セットアップガイド](docs/getting-started.md) を参照してください。

## 基本的な流れ

1. スマホを固定し、各見開きで0.5〜1秒ほど静止して撮影する
2. Web UIで動画、または静止画フォルダを選ぶ
3. 出力形式・補正設定・基準フレーム・見開き範囲を確認して抽出する
4. 要確認ページをレビューし、必要なら候補や外周、順番などを修正する
5. PDF / CBZを出力する

詳しい画面操作は [Web UI・撮影ガイド](docs/usage.md)、
補正や自動判定の調整は [設定・チューニングガイド](docs/configuration.md) を参照してください。

## ドキュメント

用途ごとの入口は [docs/README.md](docs/README.md) にまとめています。

| 内容 | ドキュメント |
|---|---|
| インストール・起動 | [セットアップガイド](docs/getting-started.md) |
| Web UI・撮影・レビュー | [Web UI・撮影ガイド](docs/usage.md) |
| 補正・候補選択・チューニング | [設定・チューニングガイド](docs/configuration.md) |
| CLI・生成ファイル・PDF / CBZ | [CLI・出力ガイド](docs/cli.md) |
| エラー対応 | [トラブルシューティング](docs/troubleshooting.md) |
| 既知の制約 | [制約・既知の限界](docs/limitations.md) |
| 内部設計・アルゴリズム | [設計・アルゴリズム](docs/architecture.md) |
| 開発環境・テスト | [開発・テストガイド](docs/development.md) |
| 検証状況 | [検証記録](docs/validation.md) |
| OSS・モデルのライセンス | [THIRD_PARTY.md](THIRD_PARTY.md) |
| 設定値の一覧 | [config.example.toml](config.example.toml) |

## ライセンス

このリポジトリの独自コードはMIT Licenseです。

依存ライブラリ、MediaPipeモデル、ユーザーが別途インストールするFFmpegにはそれぞれのライセンスが適用されます。
詳細は [THIRD_PARTY.md](THIRD_PARTY.md) を参照してください。

動画、モデル、プロジェクト出力、仮想環境、Viteのbuild生成物はGit管理対象外です。
公開前に私有動画や漫画ページが含まれていないことを確認してください。
