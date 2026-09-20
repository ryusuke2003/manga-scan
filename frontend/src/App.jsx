import { fileUrl } from './api.js';
import useScanner from './useScanner.js';
import Setup from './components/Setup.jsx';
import FrameSelector from './components/FrameSelector.jsx';
import RoiSelector from './components/RoiSelector.jsx';
import Review from './components/Review.jsx';
import { rotateNormalizedRoi } from './rotation.js';

export default function App() {
  const scanner = useScanner();
  const { manifest, project, server, busy, error, revision } = scanner;
  const file = path => fileUrl(project, path, revision);

  const cover = manifest?.cover;
  const reference = manifest?.reference;
  const coverStatus = cover?.status ?? 'skipped';
  const coverHadManualCrop = cover?.detection?.source === 'manual';
  const editingCoverCrop = coverStatus === 'frame_selected' && Boolean(cover?.roi);
  // Version 1 projects did not have setup stages; keep their original first-frame flow.
  const referenceConfirmed = reference?.confirmed ?? true;
  const referenceDetected = Boolean(reference?.detection?.detected && manifest?.roi);
  const canConfigure = manifest && manifest.status !== 'complete'
    && !(manifest.status === 'processing' && busy);

  let setupStage = null;
  if (canConfigure && coverStatus === 'pending') {
    setupStage = <FrameSelector
      step="02 / 表紙フレーム（任意）"
      title="表紙にするフレームを選ぶ"
      description="録画冒頭の表紙を1ページとして残す場合は、そのフレームを選びます。不要ならスキップできます。"
      imageUrl={fileUrl(project, cover.preview || cover.frame || 'source/first_frame_preview.png', revision)}
      time={cover.time ?? 0}
      duration={manifest.metadata.duration}
      busy={busy}
      confirmLabel="このフレームを表紙にする →"
      onPreview={time => scanner.coverFrame(time)}
      onConfirm={time => scanner.coverFrame(time, true)}
      onSkip={scanner.skipCover}
      rotation={manifest.config.rotation}
      rotationDetection={manifest.rotation_detection}
      onRotation={scanner.rotation}
    />;
  } else if (canConfigure && coverStatus === 'frame_selected') {
    setupStage = <RoiSelector
      key={`${project}-cover`}
      imageUrl={fileUrl(project, cover.preview || cover.frame, revision)}
      initialPoints={rotateNormalizedRoi(cover.roi, manifest.config.rotation)}
      metadata={manifest.metadata}
      busy={busy}
      onStart={scanner.coverRoi}
      step="03 / 表紙を囲む"
      title={editingCoverCrop ? '表紙の外周を確認・修正' : '表紙の外周を4点で指定'}
      description={editingCoverCrop
        ? '自動検出した外周です。ずれている場合だけ「やり直す」から4点を指定し直してください。'
        : '外周を自動検出できなかったため、表紙だけの大きさに合わせて4点を指定してください。'}
      actionLabel="表紙を追加して次へ →"
    />;
  } else if (canConfigure && !referenceConfirmed) {
    setupStage = <FrameSelector
      step={`${coverHadManualCrop ? '04' : '03'} / 見開き基準フレーム`}
      title="最初に本を開いた見開きを選ぶ"
      description="左右2ページがしっかり見えている場面を選んでください。この時刻より前は自動見開き解析から除外します。"
      imageUrl={fileUrl(project, reference?.preview || reference?.frame || 'source/first_frame_preview.png', revision)}
      time={reference?.time ?? 0}
      duration={manifest.metadata.duration}
      busy={busy}
      confirmLabel="このフレームを基準にする →"
      onPreview={time => scanner.referenceFrame(time)}
      onConfirm={time => scanner.referenceFrame(time, true)}
      rotation={manifest.config.rotation}
      rotationDetection={manifest.rotation_detection}
      onRotation={scanner.rotation}
      candidates={(reference?.candidates ?? []).map(candidate => ({
        ...candidate,
        imageUrl: fileUrl(project, candidate.preview, revision),
      }))}
      notice={coverStatus === 'ready' && <div className="auto-detection-notice">
        <span>{coverHadManualCrop ? '表紙の外周を設定済み' : `表紙の外周を自動検出済み · 信頼度 ${Math.round((cover.detection?.confidence ?? 0) * 100)}%`}</span>
        <button type="button" disabled={busy} onClick={scanner.editCoverRoi}>外周を修正</button>
      </div>}
    />;
  } else if (canConfigure) {
    setupStage = <RoiSelector
      key={`${project}-spread`}
      imageUrl={fileUrl(project, reference?.preview || reference?.frame || 'source/first_frame_preview.png', revision)}
      initialPoints={rotateNormalizedRoi(manifest.roi, manifest.config.rotation)}
      metadata={manifest.metadata}
      busy={busy}
      onStart={scanner.start}
      step={`${coverHadManualCrop ? '05' : '04'} / 見開き外周`}
      title={referenceDetected ? '自動検出した見開き外周を確認' : '見開きの外周を4点で指定'}
      description={referenceDetected
        ? `左右ページから外周を自動検出しました · 信頼度 ${Math.round((reference.detection?.confidence ?? 0) * 100)}%。合っていればそのまま抽出を開始してください。ずれている場合だけ「やり直す」から4点を指定し直せます。`
        : '外周を自動検出できませんでした。左上 → 右上 → 右下 → 左下 の順に4点を指定してください。'}
      actionLabel={referenceDetected ? 'この範囲で抽出開始 →' : '抽出を開始 →'}
    />;
  }
  return <>
    <aside>
      <a className="brand" href="/">Manga<span>Scan</span><small>LOCAL EDITION / 0.1</small></a>
      <p className="privacy"><span className="dot" /> このMacだけで処理</p>
      <button className="primary" disabled={busy} onClick={() => scanner.selectProject(null)}>＋ 新しいスキャン</button>
      <h2>プロジェクト</h2><div id="projects">{server.projects.map(item => <div className="project-item" key={item.id}>
        <button className={`project-select ${item.id === project ? 'active' : ''}`} title={item.id} onClick={() => scanner.selectProject(item.id)}>{item.source_name}</button>
        <button className="project-delete" disabled={busy} aria-label={`${item.source_name}を削除`} title="プロジェクトを削除" onClick={() => {
          if (window.confirm(`「${item.source_name}」を削除しますか？\nプロジェクトID: ${item.id}\n\n生成したページ画像やPDFも削除されます。この操作は元に戻せません。`)) scanner.deleteProject(item.id);
        }}>削除</button>
      </div>)}</div>
      <p className="aside-note">動画から、読むための一冊へ。<br />OCRなし・画像生成なし。</p>
    </aside>
    <main>
      <header><div><p className="eyebrow">VIDEO → PAGES → PDF</p><h1>{project ? (manifest?.source.split('/').pop() || '読み込み中…') : '漫画を、ページに。'}</h1></div><span className="badge">OFFLINE</span></header>
      {error && <div id="error" role="alert">{error}</div>}
      {!project && <Setup busy={busy} defaults={server.defaults} onChoose={scanner.choose} onCreate={scanner.create} />}
      {manifest && <>
        {setupStage}
        <section className="panel" aria-live="polite"><div className="row"><strong id="progress-text">{busy && manifest.status !== 'processing' ? '処理中…' : manifest.message}</strong><span>{Math.round(manifest.progress * 100)}%</span></div><progress max="1" value={manifest.progress} /><p className="muted">{manifest.warnings.join(' / ')}</p></section>
        {(manifest.pages.length > 0 || manifest.roi) && <Review key={project} manifest={manifest} file={file} busy={busy} onEdit={scanner.edit} />}
      </>}
      <footer>完全ローカル · 元動画を変更しません · 手や絵の描き足しは行いません</footer>
    </main>
  </>;
}
