# 検証記録

2026-09-20、macOS arm64 / Apple M5、Python 3.14.6、FFmpeg 9.0.1、
OpenCV 4.14.0、MediaPipe 0.10.35。依存の全バージョンは
`requirements-macos-arm64.lock.txt` に記録。

## 自動テスト

2026-09-20の初期Macローカル検証では `python -m pytest -q` が **36 passed** だった。
その後、ページ輪郭・左右別射影変換・補正プリセット・タイムライン・左右別候補選択・横向き回転などの
テストを追加しているため、36件は**当時の基準値**であり現在のテスト総数ではない。
最新の成否はGitHub Actionsと手元の `pytest` / Vitest結果を基準にし、この文書では変動する件数を固定しない。

- ROI内motionの同一/変化、連続静止、末尾flush、短い静止の棄却。
- motion v2が中程度のAE変化、軽いAF blur、数pxのmicro-jitterを低motionとして維持しつつ、実際のページ内容変更はturn thresholdを超えること。
- 候補が時間区間の前半/後半へ分散すること。
- dHash/SSIM、白紙を自動除外しないこと、片側だけ変わった見開きの保持。
- 品質スコア各項目の減点、無効化した手検出の明示。
- 手maskのROI交差、複数maskの重複を二重計上しないこと。
- 同一見開きの候補3枚以上を時間方向に位置合わせし、temporal medianとの差分からページ端の一時物体をMediaPipe maskへ補助的にORすること。安定した漫画本文や、既存hand maskへ接続しないページ中央の一時差分は採用しないこと。
- 指補修で別候補の実画素だけを使うこと、複数donorの合成、未復元領域の保持、危険な位置合わせの拒否。
- 指・反射の遮蔽補修donorとのAE/照明差はclean contextからbounded gain/biasを推定し、context residualが改善するときだけ補修画素へ適用すること。gainは0.9〜1.1、biasは-12〜+12に制限し、幾何alignmentの採否は補正前residualで判定すること。
- ROIの透視補正、机の除外、自己交差/範囲外/NaN/誤順序の拒否。
- 基準フレーム全体からの見開き外周自動検出、左右ページ検証、confidence不足時の4点手動fallback。
- 左右ページ輪郭の自動検出、confidence不足時のfallback、左右別 `warpPerspective`。
- 見開き出力で左右ページ+ノドを保護し、信頼できるページ輪郭の外側だけを紙色/白で隠すこと。confidence不足時は背景を変更しないこと。
- 横向き90°/270°撮影で、回転後の見た目上の左右を分割し、ROI・手mask・左右別採点も同じ向きになること。
- 奇数幅の左右分割で画素を欠落させないこと、auto spine、固定/自動湾曲remap。
- 合成した背側圧縮を左右ページ別・高さ別に検出し、profiled 2D dewarp後のエッジ間隔が1.0へ近づくこと。
- 高さ方向の孤立した湾曲推定値を抑え、低情報ページでは自動湾曲補正を安全にfallbackし、一定輝度画像で黒い穴を作らないこと。
- PNGからPDFへ画素一致、JPEGの圧縮データ一致、PDFページ比率、OCRテキストなし。
- CBZは有効ページをmanifest順の連番へ並べ、PNG/JPEGの元バイト列をZIP_STOREDで再圧縮せず格納すること。
- 失敗したPDF出力が以前のPDFを壊さないこと。
- 合成動画の4静止区間 → 1重複除外 → 6ページPDF。
- 見開き単位/左右ページ別の候補選択、片側だけの候補差し替え、除外状態維持、左右交換、手動追加、PDF再出力。
- 60/120/240fps入力でも通常解析を10fpsに制限。
- 自動high-fps fallbackは、全通常候補がrecoverableな要確認理由を持つ見開きだけ追加候補を探索し、1候補だけ悪い場合やpage quadだけが不確かな場合は発動しないこと。
- ページ抜けv2の窓では、指定秒数以上の連続低motion runを高fpsで確認できた場合だけstable intervalとして復旧し、確認できなければ欠落候補を残すこと。
- 回転メタデータ、可変fps、4K候補の元解像度取得。
- localhost UIのHost制限、Origin検査、変更操作のtoken、Vite生成アセットの静的配信。

フロントエンドには、ローカルファイルURL、ROI座標正規化、補正プリセット、候補選択、
指補修設定、タイムライン欠落候補、バックグラウンドjob状態などの回帰テストがある。
通常の検証順は次の通り。

```bash
npm --prefix frontend test
npm --prefix frontend run build
python -m pytest -q
ruff check src tests scripts
```

PythonのUI統合テストは、Viteの`index.html`が参照するハッシュ付き`/static/assets/...`を実際にFlaskから取得できることを確認する。そのためpytest前にfrontend buildが必要。

GitHub Actionsは、Ubuntu/Python 3.14のfast jobでNode 22のfrontend test/build、Ruff、軽量pytestを実行する。Ubuntu/Python 3.11のcompat-py311 jobではsource compileと、frontend buildが必要なUI統合テストおよびFFmpeg統合テストを除くPythonテストを実行し、`requires-python = ">=3.11"` の最低対応バージョンを継続確認する。pipeline jobではmacOS/Python 3.14でFFmpeg統合テストを実行する。個々のCI実行結果はこの文書へ固定せず、GitHub Actions側を参照する。

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

初期の手動ブラウザ確認では次を確認した。

- 新規プロジェクト、ローカル動画パス入力、メタデータ表示。
- CanvasでTL/TR/BR/BLの四隅をクリックし4点の確定。
- 抽出開始、処理完了、6ページ/重複見開きの表示、PDFリンク。
- 除外/復元、候補切替、左右交換、再出力。

その後追加された表紙/見開き基準フレーム、補正プリセット、動画タイムライン、
左右別候補選択などはReact/Pythonの自動テストで回帰確認している。
実写動画を使った一連のUI操作は「未検証」に含め、合成入力の自動テストと区別する。

## Issue #53: 指補修のadversarial validation

複数donorによる指補修では、**coverageが高いことだけを成功条件にしない**。
復元率100%でも文字、コマ線、網点、絵の位置関係を壊していれば不合格とする。
自動テストは「誤補修をしないための下限の安全制約」、実写確認は「漫画として自然か」の最終判定として扱う。

### 自動テストで固定する安全制約

`tests/test_finger_repair_adversarial.py` では、A/Bのlocal alignment実装に依存しない安全制約を先に固定する。

1. **コマ線が数pxずれるdonor**: target hand mask外の画素がbit-identicalであること。
2. **縦書き文字/細線が数pxずれるdonor**: target hand mask外へdonor画素が漏れないこと。
3. **donorが別の指で隠れている**: donor hand mask内の画素を採用せず、埋められない領域をunresolvedとして残すこと。
4. **ページ湾曲が少し違う**: Aのlocal alignment統合後、下記実写/合成ケースで二重線を目視・自動検証する。C単独PRではproduction algorithm未導入のため成功率テストを先行させない。
5. **全く違うページ**: alignmentを拒否すること。
6. **大きなlocal/global shiftが必要**: 上限を超える位置合わせを拒否すること。
7. **clean context不足**: alignmentを拒否し、無理にcoverageを上げないこと。
8. **指領域が2か所以上**: 複数donorを使ってもtarget mask外を変更しないこと。
9. **target mask外の不変性**: 複数component/donorでもbit-identicalを維持すること。
10. **unresolved**: Review UIの「要確認」件数・フィルタ対象に含め、未補修領域へのリンクを残すこと。

local alignmentのproduction APIが入った後は、上記に加えて
「global alignmentだけでは数px残るがlocal alignmentなら復元できる」
「湾曲差があるdonorでも許容範囲だけ局所補正できる」
をcore/pipeline側で追加検証する。

### Review UIのoptional metadata契約

A/Bが `finger_repair.local_alignment` を出力した場合だけ、既存の指補修欄へコンパクトに表示する。
metadataが存在しない既存manifestでは何も追加表示しない。

想定する最小形は次の通り。

```json
{
  "finger_repair": {
    "status": "complete",
    "coverage": 1.0,
    "donors": [4, 2],
    "local_alignment": {
      "component_count": 2,
      "max_shift_px": 6.0,
      "components": [
        {
          "donor_candidate_id": 4,
          "score": 0.95,
          "coverage": 1.0,
          "dx": 3,
          "dy": 4
        }
      ]
    }
  }
}
```

Reviewカードでは大量のcomponent詳細は展開せず、例えば
`局所補正 2領域 · 最大ずれ 6px`
だけを表示する。`max_shift_px` がない場合はcomponentの `dx/dy`
（または `shift.dx/shift.dy`）から最大移動量を算出する。
donor IDs、coverage、target mask、unresolved maskは既存表示を維持する。

### 実写で必ず確認する項目

最低2〜3種類の実写見開きで、target / 採用donor / 補修後画像 / target mask / unresolved maskを並べて確認する。
100%表示に加えて、細線や網点は必要に応じて200%以上でも確認する。

- [ ] **文字が二重になっていない**。特に縦書き本文、ルビ、細い吹き出し内文字。
- [ ] **コマ線が二重になっていない**。数pxの平行な線や折れが出ていない。
- [ ] **網点が不自然になっていない**。位相ずれ、モアレ、局所的な濃淡の継ぎ目がない。
- [ ] **指の輪郭の一部が残っていない**。肌色の縁、影、爪、半透明な境界が残っていない。
- [ ] **targetとdonorの露出差による継ぎ目が目立たない**。AE/照明差がある候補でも補修領域だけ不自然に明るい・暗い状態になっていない。
- [ ] **donor由来の別位置の絵が混ざっていない**。目、髪、背景線、効果線などが別位置から誤貼付されていない。
- [ ] **target mask外が変わっていない**。補修前後を差分比較し、mask外はbit-identicalである。
- [ ] donor側で別の指に隠れた領域を使っていない。
- [ ] ページ湾曲や押さえ方が違うdonorで、局所補正が大きくなりすぎていない。
- [ ] clean contextが乏しい領域は無理に埋めず、unresolvedとして残る。
- [ ] 2か所以上の指を別donorから補修した場合も、各領域の境界に継ぎ目がない。
- [ ] unresolvedが残るページはReviewで「要確認」として数えられ、要確認フィルタでも消えない。

**不合格例**: coverage 100%でも、文字/コマ線の二重化、網点の破綻、donorの別位置の絵の混入、
target mask外の変更が1つでもあれば不合格とする。coverageを下げてunresolvedを残す方を優先する。

## 未検証

実写漫画の手検出率、ページ欠落率、4K長時間動画の処理時間・最大メモリ、
HDR/Dolby Visionの色再現、端末ごとのcodec差、強い湾曲、macOS以外の実機UI、
VideoToolboxの機種ごとの性能、パッケージ再配布物の完全なライセンス監査。

