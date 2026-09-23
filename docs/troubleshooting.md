# トラブルシューティング

代表的なエラーと確認方法をまとめています。

## `manga-scan: command not found`

仮想環境を有効にしてください。

```bash
source .venv/bin/activate
```

## `Frontend build missing`

React/Viteの生成物がありません。

```bash
npm --prefix frontend ci --no-audit --no-fund
npm --prefix frontend run build
```

## `Hand model missing`

手検出モデルを取得してください。

```bash
python scripts/download_hand_model.py
```

すでに `models/hand_landmarker.task` が存在する場合、
ダウンロードスクリプトは上書きしません。

手検出なしで動作確認だけする場合は `config.toml` の次の2値を変更します。

```toml
hand_backend = "none"
finger_repair = false
```

この場合、手の重なりを判定できないため全ページが要確認扱いになります。

## `ffmpeg` / `ffprobe` が見つからない

```bash
brew install ffmpeg
```

インストール後に次を確認してください。

```bash
ffmpeg -version
ffprobe -version
```

## ポート8765が使用中

別ポートで起動できます。

```bash
manga-scan ui --config config.toml --projects projects --port 8766
```

## `No stable intervals found`

見開きとして採用できる「連続した低motion区間」が1件も見つからなかった状態です。

デフォルトは `video_sample_fps=10` / `stable_frames=5` なので、
目安として約0.5秒以上、`motion_threshold=0.012` 以下の状態が続く必要があります。

motion判定v2では、中程度のAE変化、軽いAFの揺れ、
数px程度の微振動を正規化してから差分を測ります。

まずは各見開きで手を引いて静止する時間を長くしてください。

それでも検出できない場合は
`projects/<project-id>/debug/motion.csv` の `motion` 列を確認し、
例えば次のように少し緩めます。

```toml
stable_frames = 3
motion_threshold = 0.018
turn_threshold = 0.025
```

`turn_threshold >= motion_threshold` は維持してください。

設定はプロジェクト作成時に `config.resolved.json` へ保存されるため、
`config.toml` を変更した後は新しいプロジェクトを作って再解析するのが確実です。

## MediaPipeがmacOSで異常終了する

- Rosettaとarm64 Pythonを混在させない
- 通常のMacターミナル/GUIセッションから実行する
- `mediapipe==0.10.35` の固定をむやみに外さない

検証環境では新しいMediaPipe版でCPU指定時にもMetal helper内部の異常終了を確認したため、
現在は0.10.35に固定しています。

## それでも解決しない場合

- セットアップを確認: [セットアップガイド](getting-started.md)
- 設定値を確認: [設定・チューニングガイド](configuration.md)
- 検証済み範囲を確認: [検証記録](validation.md)
