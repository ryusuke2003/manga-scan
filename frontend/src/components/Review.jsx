import { useEffect, useState } from 'react';

import VideoTimeline from './VideoTimeline.jsx';

const labels = { low_sharpness: '鮮鋭度が低い', hand_detection_disabled: '手の検出が無効', hand_overlap: '手の重なり', high_motion: '動きが大きい', page_quad_uncertain: '外周を確認', underexposed: '暗い', interval_gap: '時間間隔が長い', duplicate_suspected: '重複候補', manual_frame: '手動追加', manual_frame_motion_unmeasured: '動き未評価', dewarp_low_confidence: '湾曲補正の信頼度が低い' };
const reasons = items => (items || []).map(item => labels[item] || item).join(' / ');
const pageSideLabel = side => side === 'cover' ? '表紙' : (side === 'right' ? '右ページ' : '左ページ');

function ImageLink({ path, preview, file }) {
  return <a href={file(path)} target="_blank" rel="noopener"><img src={file(preview || path)} alt="抽出ページ" loading="lazy" /></a>;
}

function Spread({ spread, config, file, busy, onEdit }) {
  const [ratio, setRatio] = useState(spread.spine_ratio ?? config.spine_ratio);
  useEffect(() => setRatio(spread.spine_ratio ?? config.spine_ratio), [spread.spine_ratio, config.spine_ratio]);
  const selectionMode = spread.candidate_selection_mode ?? config.candidate_selection_mode ?? 'spread';
  const selectedPages = {
    left: spread.selected_pages?.left ?? spread.selected,
    right: spread.selected_pages?.right ?? spread.selected,
  };
  return <details className="spread">
    <summary>{spread.id} · {spread.start.toFixed(1)}–{spread.end.toFixed(1)}s{spread.duplicate_of ? ' · 重複候補' : ''}</summary>
    <p className="muted">{reasons(spread.suspect)}</p>
    {selectionMode === 'per_page' && <p className="muted">左右ページを別々に採点・選択中 · 左 #{selectedPages.left} / 右 #{selectedPages.right}</p>}
    <div className="row">
      <button disabled={busy} onClick={() => onEdit('swap', { spread_id: spread.id })}>左右の順番を入れ替え</button>
      <label htmlFor={`spine-${spread.id}`}>分割位置</label>
      <input id={`spine-${spread.id}`} type="number" min="0.25" max="0.75" step="0.005" value={ratio} onChange={event => setRatio(event.target.value)} />
      <button disabled={busy || ratio === '' || Number(ratio) < .25 || Number(ratio) > .75} onClick={() => onEdit('spine', { spread_id: spread.id, ratio: Number(ratio) })}>反映</button>
    </div>
    <div className="candidates">{spread.candidates.map(candidate => {
      const leftMetrics = candidate.page_metrics?.left ?? candidate.metrics;
      const rightMetrics = candidate.page_metrics?.right ?? candidate.metrics;
      const leftSelected = candidate.id === selectedPages.left;
      const rightSelected = candidate.id === selectedPages.right;
      const selected = selectionMode === 'per_page' ? leftSelected || rightSelected : candidate.id === spread.selected;
      return <div key={candidate.id} className={`candidate ${selected ? 'selected' : ''}`}>
        <ImageLink file={file} path={candidate.path} preview={candidate.preview} />
        <p>{candidate.time.toFixed(2)}s · 全体 score {candidate.metrics.score.toFixed(3)}<br />
          鮮鋭度 {candidate.metrics.sharpness.toFixed(0)} / 手 {candidate.metrics.hand_overlap === null ? '未評価' : `${(candidate.metrics.hand_overlap * 100).toFixed(1)}%`}</p>
        {selectionMode === 'per_page' && <p>
          左 score {leftMetrics.score.toFixed(3)} / 鮮鋭度 {leftMetrics.sharpness.toFixed(0)} / 手 {leftMetrics.hand_overlap === null ? '未評価' : `${(leftMetrics.hand_overlap * 100).toFixed(1)}%`}<br />
          右 score {rightMetrics.score.toFixed(3)} / 鮮鋭度 {rightMetrics.sharpness.toFixed(0)} / 手 {rightMetrics.hand_overlap === null ? '未評価' : `${(rightMetrics.hand_overlap * 100).toFixed(1)}%`}
        </p>}
        <a href={file(candidate.hand_mask)} target="_blank" rel="noopener">手のマスク ↗</a>
        {selectionMode === 'per_page'
          ? <div className="row">
            <button disabled={busy || leftSelected} onClick={() => onEdit('select_candidate', { spread_id: spread.id, candidate_id: candidate.id, side: 'left' })}>{leftSelected ? '左に採用中' : '左に採用'}</button>
            <button disabled={busy || rightSelected} onClick={() => onEdit('select_candidate', { spread_id: spread.id, candidate_id: candidate.id, side: 'right' })}>{rightSelected ? '右に採用中' : '右に採用'}</button>
          </div>
          : <button disabled={busy || candidate.id === spread.selected} onClick={() => onEdit('select_candidate', { spread_id: spread.id, candidate_id: candidate.id })}>{candidate.id === spread.selected ? '採用中' : 'この候補を採用'}</button>}
      </div>;
    })}</div>
  </details>;
}

export default function Review({ manifest, file, busy, onEdit }) {
  const [suspectsOnly, setSuspectsOnly] = useState(false);
  const [showExcluded, setShowExcluded] = useState(false);
  const [timestamp, setTimestamp] = useState('');
  const enabled = manifest.pages.filter(page => page.enabled);
  let number = 0;
  const numbered = manifest.pages.map(page => ({ ...page, number: page.enabled ? ++number : null }));
  return <section>
    <div className="review-head"><div><p className="step">03 / 確認して仕上げる</p><h2>{enabled.length} ページ / 要確認 {enabled.filter(page => page.suspect.length).length}</h2></div>
      <div className="row"><button className="primary" disabled={busy || !enabled.length} onClick={() => onEdit('export')}>PDFを出力</button>
        {manifest.pdf && !manifest.pdf_stale && <a className="button" href={file(manifest.pdf)} target="_blank" rel="noopener">PDFを開く ↗</a>}</div></div>
    <p className="muted">{manifest.pdf_stale ? '編集後のPDFは未出力です。「PDFを出力」で反映してください。' : '現在のページ順・画質でPDFを出力済みです。'}</p>
    <VideoTimeline
      manifest={manifest}
      busy={busy}
      onSelectTime={time => setTimestamp(time.toFixed(2))}
    />
    <div className="toolbar panel">
      <label className="checkbox"><input type="checkbox" checked={suspectsOnly} onChange={event => setSuspectsOnly(event.target.checked)} /> 要確認だけ表示</label>
      <label className="checkbox"><input type="checkbox" checked={showExcluded} onChange={event => setShowExcluded(event.target.checked)} /> 除外ページも表示</label>
      <form className="row" onSubmit={event => { event.preventDefault(); onEdit('add_frame', { time: Number(timestamp) }); }}>
        <input aria-label="追加する動画の秒数" type="number" min="0" max={manifest.metadata.duration - .001} step="any" placeholder="動画の秒数" required value={timestamp} onChange={event => setTimestamp(event.target.value)} />
        <button disabled={busy || timestamp === ''}>この時刻から追加</button></form>
    </div>
    <div className="page-grid">{numbered.filter(page => (page.enabled || showExcluded) && (!suspectsOnly || page.suspect.length)).map(page => <article key={page.id} className={`page-card ${page.suspect.length ? 'suspect' : ''} ${page.enabled ? '' : 'excluded'}`}>
      <ImageLink file={file} path={page.path} preview={page.preview} />
      <h3>{page.number ? String(page.number).padStart(3, '0') : '除外'} · {pageSideLabel(page.side)}</h3>
      <p>{reasons(page.suspect)}</p>
      {page.candidate_time !== undefined && <p className="muted">候補 #{page.candidate_id} · {page.candidate_time.toFixed(2)}s</p>}
      {page.dewarp?.mode === 'auto' && <div className="dewarp-meta">
        <span>湾曲補正: {page.dewarp.status === 'applied' ? `適用 最大 ${(page.dewarp.strength * 100).toFixed(1)}%${page.dewarp.profile_variation ? ` · 上下差 ${(page.dewarp.profile_variation * 100).toFixed(1)}%` : ''}` : page.dewarp.status === 'disabled' ? 'ページ単位でOFF' : page.dewarp.status === 'not_needed' ? '補正不要' : '見送り'}{page.dewarp.confidence !== undefined ? ` · 信頼度 ${Math.round(page.dewarp.confidence * 100)}%` : ''}</span>
        <div className="row">{page.dewarp.before && <a href={file(page.dewarp.before)} target="_blank" rel="noopener">補正前 ↗</a>}
          {page.dewarp.debug_grid && <a href={file(page.dewarp.debug_grid)} target="_blank" rel="noopener">remap ↗</a>}
          <button disabled={busy} onClick={() => onEdit('toggle_dewarp', { spread_id: page.spread_id, side: page.side })}>{page.dewarp.status === 'disabled' ? '自動補正ON' : '自動補正OFF'}</button></div>
      </div>}
      <div className="row"><button disabled={busy} onClick={() => onEdit('toggle_page', { page_id: page.id })}>{page.enabled ? '除外' : '復元'}</button>
        <button disabled={busy} aria-label={`${page.id}を前へ`} onClick={() => onEdit('move_page', { page_id: page.id, delta: -1 })}>←</button>
        <button disabled={busy} aria-label={`${page.id}を後ろへ`} onClick={() => onEdit('move_page', { page_id: page.id, delta: 1 })}>→</button></div>
    </article>)}</div>
    <h2 className="spreads-heading">見開き・候補フレーム</h2><p className="muted">候補を選ぶと元解像度で再抽出します。左右別モードでは各ページのスコアを個別に確認・差し替えできます。</p>
    {manifest.spreads.map(spread => <Spread key={spread.id} spread={spread} config={manifest.config} file={file} busy={busy} onEdit={onEdit} />)}
  </section>;
}
