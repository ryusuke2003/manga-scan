# ドキュメント

Manga Scan Localの詳細資料を目的別に分けています。
最初に使う場合は、READMEのクイックスタートから始めて、必要な項目だけ参照してください。

## 利用者向け

- [セットアップガイド](getting-started.md)
  - 必要環境、初回インストール、起動、2回目以降の手順
- [Web UI・撮影ガイド](usage.md)
  - 撮影方法、画面操作、レビュー、静止画フォルダ入力、中断・再開
- [設定・チューニングガイド](configuration.md)
  - 出力形式、補正プリセット、指補修、湾曲補正、候補選択、性能調整
- [CLI・出力ガイド](cli.md)
  - CLIコマンド、プロジェクト構成、PDF / CBZの生成仕様
- [トラブルシューティング](troubleshooting.md)
  - 起動、FFmpeg、MediaPipe、静止区間検出などの代表的なエラー
- [制約・既知の限界](limitations.md)
  - 自動補正や検出で保証できない範囲

## 開発・内部仕様

- [設計・アルゴリズム](architecture.md)
  - motion判定、候補選択、手検出、補修、ページ検出などの内部設計
- [開発・テストガイド](development.md)
  - 開発依存、Vite、pytest、Vitest、Ruff、GitHub Actions
- [検証記録](validation.md)
  - 自動テスト、実機検証、adversarial validation

## リポジトリ直下の関連資料

- [設定例](../config.example.toml)
- [OSS・モデル・ライセンス調査](../THIRD_PARTY.md)
- [README](../README.md)
