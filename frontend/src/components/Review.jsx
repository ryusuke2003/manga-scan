import { useEffect, useState } from 'react';

import { rotateNormalizedRoi } from '../rotation.js';
import RoiSelector from './RoiSelector.jsx';
import VideoTimeline from './VideoTimeline.jsx';

const labels = { low_sharpness: '鮮鋭度が低い', hand_detection_disabled: '手の検出が無効', hand_overlap: '手の重なり', high_motion: '動きが大きい', page_quad_uncertain: '外周を確認', underexposed: '暗い', interval_gap: '時間間隔が長い', duplicate_suspected: '重複候補', manual_frame: '手動追加', manual_frame_motion_unmeasured: '動き未評価', dewarp_low_confidence: '湾曲補正の信頼度が低い', finger_repair_incomplete: '指の補修が不完全', source_frame_clipped: '元動画の画面端に接触・見切れを確認', page_contour_low_confidence: 'ページ外周の検出が不確か' };
const reasons = items => (items || []).map(item => labels[item] || item).join(' / ');
const pageSideLabel = side => ({ cover: '表紙', spread: '見開き', right: '右ページ', left: '左ページ' }[side] || side);
const cropStatusLabel = crop => ({
  auto_pages: `左右ページから外周を自動検出${crop.confidence !== undefined ? ` · 信頼度 ${Math.round(crop.confidence * 100)}%` : ''}`,
  fallback: '外周を自動検出できず、基準範囲を使用',
  reference: '外周の自動検出はOFF',
  manual: '外周を手動指定',
}[crop?.status] || '');
const fingerFallbackLabel = repair => {
  const fallback = repair?.fallback;
  if (!fallback?.applied) return '';
  if (fallback.mode === 'white') return ' · 未補修部を白塗り';
  if (fallback.mode === 'paper') return ` · 紙面補完 ${Math.round((fallback.filled_fraction ?? 0) * 100)}%`;
  return '';
};

function ImageLink({ path, preview, file }) {
  return <a href={file(path)} target="_blank" rel="noopener"><img src={file(preview || path)} alt="抽出ページ" loading="lazy" /></a>;
}

function Spread({ spread, config, file, busy, onEdit }) {
  const [ratio, setRatio] = useState(spread.spine_ratio ?? config.spine_ratio);
  const [cropCandidate, setCropCandidate] = useState(null);
  const layout = spread.output_layout ?? config.output_layout ?? 'split';
  useEffect(() => setRatio(spread.spine_ratio ?? config.spine_ratio), [spread.spine_ratio, config.spine_ratio]);
  const selectionMode = layout === 'spread' ? 'spread' : (spread.candidate_selection_mode ?? config.candidate_selection_mode ?? 'spread');
  const selectedPages = {
    left: spread.selected_pages?.left ?? spread.selected,
    right: spread.selected_pages?.right ?? spread.selected,
  };
  return <details className="spread">
    <summary>{spread.id} · {spread.start.toFixed(1)}–{spread.end.toFixed(1)}s{spread.duplicate_of ? ' · 重複候補' : ''}</summary>
    <p className="muted">{reasons(spread.suspect)}</p>
    <label>この見開きの出力形式<select disabled={busy} value={layout} onChange={event => onEdit('output_layout', { spread_id: spread.id, layout: event.target.value })}><option value="spread">見開きのまま</option><option value="split">左右のページに分割</option></select></label>
    {layout === 'spread' && spread.whole_spread_crop && <p className="muted">{cropStatusLabel(spread.whole_spread_crop)}{spread.page_contour_debug && <> · <a href={file(spread.page_contour_debug)} target="_blank" rel="noopener">検出結果 ↗</a></>}</p>}
    {selectionMode === 'per_page' && <p className="muted">左右ページを別々に採点・選択中 · 左 #{selectedPages.left} / 右 #{selectedPages.right}</p>}
    {layout === 'split' && <div className="row">
      <button disabled={busy} onClick={() => onEdit('swap', { spread_id: spread.id })}>左右の順番を入れ替え</button>
      <label htmlFor={`spine-${spread.id}`}>分割位置</label>
      <input id={`spine-${spread.id}`} type="number" min="0.25" max="0.75" step="0.005" value={ratio} onChange={event => setRatio(event.target.value)} />
      <button disabled={busy || ratio === '' || Number(ratio) < .25 || Number(ratio) > .75} onClick={() => onEdit('spine', { spread_id: spread.id, ratio: Number(ratio) })}>反映</button>
    </div>}
    {cropCandidate !== null && (() => {
      const candidate = spread.candidates.find(item => item.id === cropCandidate);
      const renderedCrop = layout === 'spread' && spread.whole_spread_crop?.candidate_id === candidate.id
        ? spread.whole_spread_crop : null;
      const sourceRoi = spread.roi_overrides?.[String(candidate.id)] ?? candidate.roi;
      const initialPoints = renderedCrop?.roi ?? rotateNormalizedRoi(sourceRoi, config.rotation || 0);
      return <div>
        <RoiSelector key={`${spread.id}-${candidate.id}`} imageUrl={file(candidate.path)} initialPoints={initialPoints} rotation={config.rotation || 0} busy={busy}
          step="外周を確認・調整" title={`候補 #${candidate.id} の外周`}
          description="自動検出した外周です。ずれている場合は「やり直す」を押し、左上 → 右上 → 右下 → 左下の順で指定してください。この候補だけに適用します。指で隠れた絵や画面外の絵は復元できません。"
          actionLabel="この範囲で再出力" onStart={points => {
            onEdit('crop', { spread_id: spread.id, candidate_id: candidate.id, roi: points });
            setCropCandidate(null);
          }} />
        <button disabled={busy} onClick={() => setCropCandidate(null)}>閉じる</button>
      </div>;
    })()}
    <div className="candidates">{spread.candidates.map(candidate => {
      const leftMetrics = candidate.page_metrics?.left ?? candidate.metrics;
      const rightMetrics = candidate.page_metrics?.right ?? candidate.metrics;
      const leftSelected = candidate.id === selectedPages.left;
      const rightSelected = candidate.id === selectedPages.right;
      const selected = selectionMode === 'per_page' ? leftSelected || rightSelected : candidate.id === spread.selected;
      const manuallyCropped = Boolean(spread.roi_overrides?.[String(candidate.id)]);
      return <div key={candidate.id} className={`candidate ${selected ? 'selected' : ''}`}>
        <ImageLink file={file} path={candidate.path} preview={candidate.review_preview || candidate.preview} />
        <p>{candidate.time.toFixed(2)}s · 全体 score {candidate.metrics.score.toFixed(3)}<br />
          鮮鋭度 {candidate.metrics.sharpness.toFixed(0)} / 手 {candidate.metrics.hand_overlap === null ? '未評価' : `${(candidate.metrics.hand_overlap * 100).toFixed(1)}%`}</p>
        {selectionMode === 'per_page' && <p>
          左 score {leftMetrics.score.toFixed(3)} / 鮮鋭度 {leftMetrics.sharpness.toFixed(0)} / 手 {leftMetrics.hand_overlap === null ? '未評価' : `${(leftMetrics.hand_overlap * 100).toFixed(1)}%`}<br />
          右 score {rightMetrics.score.toFixed(3)} / 鮮鋭度 {rightMetrics.sharpness.toFixed(0)} / 手 {rightMetrics.hand_overlap === null ? '未評価' : `${(rightMetrics.hand_overlap * 100).toFixed(1)}%`}
        </p>}
        <a href={file(candidate.hand_mask)} target="_blank" rel="noopener">手のマスク ↗</a>
        {selected && <button disabled={busy} onClick={() => setCropCandidate(candidate.id)}>外周を確認・調整</button>}
        {selected && (manuallyCropped || (layout === 'spread' && config.refine_quad !== false)) && <button disabled={busy} onClick={() => onEdit('reset_crop', { spread_id: spread.id, candidate_id: candidate.id })}>{manuallyCropped ? '自動検出に戻す' : '外周を自動検出し直す'}</button>}
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
    <div className={`page-grid ${manifest.pages.some(page => page.side === 'spread') ? 'with-spreads' : ''}`}>{numbered.filter(page => (page.enabled || showExcluded) && (!suspectsOnly || page.suspect.length)).map(page => <article key={page.id} className={`page-card ${page.suspect.length ? 'suspect' : ''} ${page.enabled ? '' : 'excluded'}`}>
      <ImageLink file={file} path={page.path} preview={page.preview} />
      <h3>{page.number ? String(page.number).padStart(3, '0') : '除外'} · {pageSideLabel(page.side)}</h3>
      <p>{reasons(page.suspect)}</p>
      {page.candidate_time !== undefined && <p className="muted">候補 #{page.candidate_id} · {page.candidate_time.toFixed(2)}s</p>}
      {page.finger_repair && page.finger_repair.status !== 'disabled' && <div className="dewarp-meta">
        <span>指補修: {page.finger_repair.status === 'complete' ? '完了' : page.finger_repair.status === 'clean' ? '指を未検出' : page.finger_repair.status === 'unavailable' ? 'マスクなし' : '一部のみ'} · 復元率 {Math.round((page.finger_repair.coverage ?? 0) * 100)}%{page.finger_repair.donors?.length ? ` · donor #${page.finger_repair.donors.join(', #')}` : ''}{fingerFallbackLabel(page.finger_repair)}</span>
        {page.finger_repair.status === 'incomplete' && <p className="muted">隠れた部分を別候補から十分に補修できず、指が残っています。別の候補も確認してください。</p>}
        <div className="row">{page.finger_repair.target_mask && <a href={file(page.finger_repair.target_mask)} target="_blank" rel="noopener">指マスク ↗</a>}
          {page.finger_repair.unresolved_mask && <a href={file(page.finger_repair.unresolved_mask)} target="_blank" rel="noopener">未補修領域 ↗</a>}</div>
      </div>}
      {page.dewarp?.mode === 'auto' && <div className="dewarp-meta">
        <span>湾曲補正: {page.dewarp.status === 'applied' ? `適用 最大 ${(page.dewarp.strength * 100).toFixed(1)}%${page.dewarp.profile_variation ? ` · 高さ方向差 ${(page.dewarp.profile_variation * 100).toFixed(1)}%` : ''}` : page.dewarp.status === 'disabled' ? 'ページ単位でOFF' : page.dewarp.status === 'not_needed' ? '補正不要' : '見送り'}{page.dewarp.confidence !== undefined ? ` · 信頼度 ${Math.round(page.dewarp.confidence * 100)}%` : ''}</span>
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
