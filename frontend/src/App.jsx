import { fileUrl } from './api.js';
import useScanner from './useScanner.js';
import Setup from './components/Setup.jsx';
import RoiSelector from './components/RoiSelector.jsx';
import Review from './components/Review.jsx';

export default function App() {
  const scanner = useScanner();
  const { manifest, project, server, busy, error, revision } = scanner;
  const file = path => fileUrl(project, path, revision);
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
      {!project && <Setup busy={busy} onChoose={scanner.choose} onCreate={scanner.create} />}
      {manifest && <>
        {manifest.status !== 'complete' && !(manifest.status === 'processing' && busy) && <RoiSelector key={project} imageUrl={fileUrl(project, 'source/first_frame.png')} initialPoints={manifest.roi} metadata={manifest.metadata} busy={busy} onStart={scanner.start} />}
        <section className="panel" aria-live="polite"><div className="row"><strong id="progress-text">{busy && manifest.status !== 'processing' ? '処理中…' : manifest.message}</strong><span>{Math.round(manifest.progress * 100)}%</span></div><progress max="1" value={manifest.progress} /><p className="muted">{manifest.warnings.join(' / ')}</p></section>
        {(manifest.pages.length > 0 || manifest.roi) && <Review key={project} manifest={manifest} file={file} busy={busy} onEdit={scanner.edit} />}
      </>}
      <footer>完全ローカル · 元動画を変更しません · 手や絵の描き足しは行いません</footer>
    </main>
  </>;
}
