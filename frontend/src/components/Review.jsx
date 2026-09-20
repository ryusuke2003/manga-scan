import { useEffect, useState } from 'react';

import { rotateNormalizedRoi } from '../rotation.js';
import RoiSelector from './RoiSelector.jsx';
import VideoTimeline from './VideoTimeline.jsx';

const labels = {
  low_sharpness: '鮮鋭度が低い',
  hand_detection_disabled: '手の検出が無効',
  hand_overlap: '手の重なり',
  glare_overlap: '反射・白飛び',
  high_motion: '動きが大きい',
  page_quad_uncertain: '外周を確認',
  underexposed: '暗い',
  interval_gap: '時間間隔が長い',
  duplicate_suspected: '重複候補',
  manual_frame: '手動追加',
  manual_frame_motion_unmeasured: '動き未評価',
  dewarp_low_confidence: '湾曲補正の信頼度が低い',
  finger_repair_incomplete: '指の補修が不完全',
  occlusion_repair_incomplete: '遮蔽補修が不完全',
  source_frame_clipped: '元動画の画面端に接触・見切れを確認',
  page_contour_low_confidence: 'ページ外周の検出が不確か',
  final_edge_crop_suspected: '完成画像: ページ端を確認',
  final_dewarp_line_regression: '完成画像: 湾曲補正後の直線を確認',
  final_unresolved_finger: '完成画像: 指の未補修',
  final_finger_repair_residual: '完成画像: 指補修境界を確認',
  final_background_fill_large: '完成画像: 白背景補正範囲が大きい',
  final_glare_residual: '完成画像: 反射が残っている',
  final_duplicate_suspected: '完成画像: 前ページとほぼ同一',
  final_near_blank_white: '完成画像: ほぼ真っ白',
  final_near_blank_black: '完成画像: ほぼ真っ黒',
};
const reasons = items => (items || []).map(item => labels[item] || item).join(' / ');
const pageSideLabel = side => ({ cover: '表紙', spread: '見開き', right: '右ページ', left: '左ページ', external: '外部画像' }[side] || side);
const repairTitle = repair => repair?.occlusion_kinds?.includes('glare') ? '遮蔽補修' : '指補修';
const repairCleanLabel = repair => repair?.occlusion_kinds?.includes('glare') ? '遮蔽なし' : '指を未検出';
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

const repairPercentage = value => {
  if (value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.round(numeric * 100) : null;
};

export function fingerRepairCoverageSummary(repair) {
  if (!repair || repair.status === 'clean') return '';

  const donorCoverage = repairPercentage(repair.donor_coverage);
  const coverage = repairPercentage(repair.coverage);
  const parts = [];

  if (donorCoverage !== null) {
    parts.push(`実画素復元率 ${donorCoverage}%`);
  }
  if (coverage !== null && (donorCoverage === null || coverage !== donorCoverage)) {
    parts.push(`処理済み率 ${coverage}%`);
  }
  return parts.join(' · ');
}

function localShift(component) {
  const shift = component?.shift ?? component?.local_shift ?? {};
  const dx = Number(component?.dx ?? component?.shift_dx ?? shift.dx);
  const dy = Number(component?.dy ?? component?.shift_dy ?? shift.dy);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) return null;
  return Math.hypot(dx, dy);
}

export function localAlignmentSummary(repair) {
  const local = repair?.local_alignment;
  if (!local) return '';

  const components = Array.isArray(local.components) ? local.components : [];
  const componentCount = Number.isFinite(Number(local.component_count))
    ? Number(local.component_count)
    : components.length;
  if (componentCount <= 0) return '';

  const explicitMax = Number(local.max_shift_px);
  const shifts = components
    .filter(component => component?.applied !== false && component?.accepted !== false)
    .map(localShift)
    .filter(value => value !== null);
  const maxShift = Number.isFinite(explicitMax)
    ? explicitMax
    : (shifts.length ? Math.max(...shifts) : null);

  const roundedShift = maxShift === null
    ? ''
    : (Math.abs(maxShift - Math.round(maxShift)) < 0.05
      ? String(Math.round(maxShift))
      : maxShift.toFixed(1));
  return `局所補正 ${componentCount}領域${roundedShift ? ` · 最大ずれ ${roundedShift}px` : ''}`;
}

const needsReview = page => Boolean(
  page?.suspect?.length
  || page?.final_quality?.reasons?.length
  || page?.finger_repair?.unresolved_mask
);

const reviewCategories = [
  {
    label: '指補修',
    reasons: ['finger_repair_incomplete', 'final_unresolved_finger', 'final_finger_repair_residual'],
  },
  {
    label: 'ページ輪郭',
    reasons: ['page_quad_uncertain', 'page_contour_low_confidence', 'source_frame_clipped', 'final_edge_crop_suspected'],
  },
  {
    label: '反射',
    reasons: ['glare_overlap', 'final_glare_residual'],
  },
  {
    label: 'dewarp異常',
    reasons: ['dewarp_low_confidence', 'final_dewarp_line_regression'],
  },
  {
    label: '重複疑い',
    reasons: ['duplicate_suspected', 'final_duplicate_suspected'],
  },
  {
    label: '仕上げ異常',
    reasons: ['final_background_fill_large', 'final_near_blank_white', 'final_near_blank_black'],
  },
  {
    label: '入力品質',
    reasons: ['low_sharpness', 'hand_detection_disabled', 'hand_overlap', 'high_motion', 'underexposed', 'interval_gap', 'manual_frame', 'manual_frame_motion_unmeasured'],
  },
];

const pageReasons = page => Array.from(new Set([
  ...(page?.suspect || []),
  ...(page?.final_quality?.reasons || []),
]));

const categoryMatchesPage = (category, page) => {
  const current = pageReasons(page);
  if (category.reasons.some(reason => current.includes(reason))) return true;
  if (!current.includes('occlusion_repair_incomplete')) return false;
  const kinds = page?.finger_repair?.occlusion_kinds || [];
  if (category.label === '指補修') return kinds.includes('finger');
  if (category.label === '反射') return kinds.includes('glare');
  return false;
};

export function qualityReviewSummary(pages = []) {
  const enabled = pages.filter(page => page.enabled);
  return {
    total: enabled.filter(needsReview).length,
    categories: reviewCategories.map(category => ({
      label: category.label,
      count: enabled.filter(page => categoryMatchesPage(category, page)).length,
    })).filter(category => category.count > 0),
  };
}

const safeFixLabels = {
  glare: '反射が少ない',
  hand: '手・指の重なりが少ない',
  sharpness: 'より鮮明',
  motion: 'ブレが少ない',
  risk_count: '既知の警告が少ない',
  selection_score: '総合スコアが高い',
};

export function safeFixSummary(suggestion) {
  const items = (suggestion?.improvements || [])
    .map(item => safeFixLabels[item] || item);
  const gain = Number(suggestion?.score_gain);
  if (Number.isFinite(gain) && gain >= 0.01) {
    items.push(`候補スコア +${Math.round(gain * 100)}pt`);
  }
  return Array.from(new Set(items)).join(' · ');
}

export function pageCountSummary(check) {
  if (!check) return 'ページ数を計算中';
  const output = check.output_items !== check.actual
    ? ` · 出力画像 ${check.output_items}枚`
    : '';
  if (check.status === 'unset') {
    return `現在 ${check.actual}ページ${output} · 期待ページ数は未設定`;
  }
  if (check.status === 'pending') {
    return `現在 ${check.actual}ページ${output} · 完了後に期待 ${check.expected}ページと照合`;
  }
  if (check.status === 'match') {
    return `一致: ${check.actual} / ${check.expected}ページ${output}`;
  }
  if (check.status === 'short') {
    return `不足: ${check.actual} / ${check.expected}ページ · ${Math.abs(check.difference)}ページ不足${output}`;
  }
  return `超過: ${check.actual} / ${check.expected}ページ · ${check.difference}ページ多い${output}`;
}

const backgroundFillLabel = fill => {
  if (!fill || fill.mode === 'preserve') return '';
  if (fill.status === 'unavailable') return 'ページ外背景: 輪郭不確かのため変更なし';
  if (!fill.applied) return 'ページ外背景: 補正不要';
  const mode = fill.mode === 'white' ? '白' : '紙色';
  return `ページ外背景: ${mode}で隠す · ${Math.round((fill.filled_fraction ?? 0) * 100)}%`;
};

function ImageLink({ path, preview, file }) {
  return <a href={file(path)} target="_blank" rel="noopener"><img src={file(preview || path)} alt="抽出ページ" loading="lazy" /></a>;
}

export function reorderPageIds(pageIds, sourceId, targetId) {
  if (sourceId === targetId) return pageIds;
  const source = pageIds.indexOf(sourceId);
  const target = pageIds.indexOf(targetId);
  if (source < 0 || target < 0) return pageIds;

  const reordered = [...pageIds];
  const [moved] = reordered.splice(source, 1);
  reordered.splice(target, 0, moved);
  return reordered;
}

function PageReviewControls({ page, manifest, file, busy, onEdit }) {
  const [editingContour, setEditingContour] = useState(false);
  const spread = manifest.spreads.find(item => item.id === page.spread_id);
  const candidate = spread?.candidates?.find(item => item.id === page.candidate_id);
  const settings = page.render_settings ?? {};
  const splitPage = page.side === 'left' || page.side === 'right';
  const dewarpEnabled = splitPage
    ? (settings.dewarp ?? (page.dewarp?.status !== 'disabled'))
    : false;
  const illuminationEnabled = settings.illumination_correction
    ?? manifest.config.illumination_correction
    ?? false;
  const whiteEnabled = settings.white_normalization
    ?? manifest.config.white_normalization
    ?? false;
  const contourMode = page.page_contour?.mode ?? settings.page_quad_mode ?? 'auto';

  const update = next => onEdit('page_settings', {
    page_id: page.id,
    settings: next,
  });

  return <div className="page-review-controls">
    <div className="page-compare">
      <a href={file(page.source || page.path)} target="_blank" rel="noopener">
        <span>元画像</span>
        <img src={file(page.source || page.path)} alt={page.id + ' 元画像'} loading="lazy" />
      </a>
      <a href={file(page.path)} target="_blank" rel="noopener">
        <span>補正後</span>
        <img src={file(page.preview || page.path)} alt={page.id + ' 補正後'} loading="lazy" />
      </a>
    </div>
    <div className="page-setting-grid">
      {splitPage && <div className="page-setting-row">
        <span>湾曲補正</span>
        <button
          className={dewarpEnabled ? 'active' : ''}
          aria-label={page.id + ' 湾曲補正 ON'}
          disabled={busy}
          onClick={() => update({ dewarp: true })}
        >ON</button>
        <button
          className={!dewarpEnabled ? 'active' : ''}
          aria-label={page.id + ' 湾曲補正 OFF'}
          disabled={busy}
          onClick={() => update({ dewarp: false })}
        >OFF</button>
      </div>}
      <div className="page-setting-row">
        <span>照明補正</span>
        <button
          className={illuminationEnabled ? 'active' : ''}
          aria-label={page.id + ' 照明補正 ON'}
          disabled={busy}
          onClick={() => update({ illumination_correction: true })}
        >ON</button>
        <button
          className={!illuminationEnabled ? 'active' : ''}
          aria-label={page.id + ' 照明補正 OFF'}
          disabled={busy}
          onClick={() => update({ illumination_correction: false })}
        >OFF</button>
      </div>
      <div className="page-setting-row">
        <span>白背景補正</span>
        <button
          className={whiteEnabled ? 'active' : ''}
          aria-label={page.id + ' 白背景補正 ON'}
          disabled={busy}
          onClick={() => update({ white_normalization: true })}
        >ON</button>
        <button
          className={!whiteEnabled ? 'active' : ''}
          aria-label={page.id + ' 白背景補正 OFF'}
          disabled={busy}
          onClick={() => update({ white_normalization: false })}
        >OFF</button>
      </div>
      {splitPage && <div className="page-setting-row">
        <span>ページ輪郭</span>
        <button
          className={contourMode === 'auto' ? 'active' : ''}
          aria-label={page.id + ' ページ輪郭 auto'}
          disabled={busy}
          onClick={() => {
            if (contourMode !== 'auto') update({ page_quad_mode: 'auto' });
          }}
        >auto</button>
        <button
          className={contourMode === 'manual' ? 'active' : ''}
          aria-label={page.id + ' ページ輪郭 manual'}
          disabled={busy || !candidate}
          onClick={() => setEditingContour(true)}
        >manual</button>
      </div>}
    </div>
    <p className="page-rerender-note">変更はこのページを含む見開きだけ再レンダリングします。</p>
    {editingContour && candidate && <div className="page-contour-editor">
      <RoiSelector
        key={page.id + '-' + contourMode}
        imageUrl={file(candidate.path)}
        initialPoints={page.page_contour?.quad || []}
        rotation={manifest.config.rotation || 0}
        busy={busy}
        step="ページ単位の外周補正"
        title={pageSideLabel(page.side) + 'の外周を手動指定'}
        description="左上 → 右上 → 右下 → 左下の順で4点を指定します。このページだけmanual輪郭として保存します。"
        actionLabel="この外周で再レンダリング"
        onStart={points => {
          update({ page_quad_mode: 'manual', manual_quad: points });
          setEditingContour(false);
        }}
      />
      <button disabled={busy} onClick={() => setEditingContour(false)}>閉じる</button>
    </div>}
  </div>;
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
          鮮鋭度 {candidate.metrics.sharpness.toFixed(0)} / 手 {candidate.metrics.hand_overlap === null ? '未評価' : `${(candidate.metrics.hand_overlap * 100).toFixed(1)}%`} / 反射 {((candidate.metrics.glare_overlap ?? 0) * 100).toFixed(1)}%</p>
        {selectionMode === 'per_page' && <p>
          左 score {leftMetrics.score.toFixed(3)} / 鮮鋭度 {leftMetrics.sharpness.toFixed(0)} / 手 {leftMetrics.hand_overlap === null ? '未評価' : `${(leftMetrics.hand_overlap * 100).toFixed(1)}%`} / 反射 {((leftMetrics.glare_overlap ?? 0) * 100).toFixed(1)}%<br />
          右 score {rightMetrics.score.toFixed(3)} / 鮮鋭度 {rightMetrics.sharpness.toFixed(0)} / 手 {rightMetrics.hand_overlap === null ? '未評価' : `${(rightMetrics.hand_overlap * 100).toFixed(1)}%`} / 反射 {((rightMetrics.glare_overlap ?? 0) * 100).toFixed(1)}%
        </p>}
        <a href={file(candidate.hand_mask)} target="_blank" rel="noopener">手のマスク ↗</a>
        {candidate.glare_mask && <> · <a href={file(candidate.glare_mask)} target="_blank" rel="noopener">反射マスク ↗</a></>}
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

const emptyBookMetadata = {
  title: '',
  author: '',
  series: '',
  volume: '',
  publisher: '',
  language: '',
};

function PageCountCheck({ manifest, busy, onSave }) {
  const [draft, setDraft] = useState(
    manifest.expected_page_count === null || manifest.expected_page_count === undefined
      ? ''
      : String(manifest.expected_page_count),
  );
  useEffect(() => {
    setDraft(
      manifest.expected_page_count === null || manifest.expected_page_count === undefined
        ? ''
        : String(manifest.expected_page_count),
    );
  }, [manifest.expected_page_count]);

  const normalized = draft === '' ? null : Number(draft);
  const changed = normalized !== (manifest.expected_page_count ?? null);
  const enabled = manifest.pages.filter(page => page.enabled);
  const actual = enabled.reduce(
    (total, page) => total + (page.side === 'spread' ? 2 : 1),
    0,
  );
  const fallbackDifference = manifest.expected_page_count == null
    ? null
    : actual - manifest.expected_page_count;
  const check = manifest.page_count_check ?? {
    status: manifest.expected_page_count == null
      ? 'unset'
      : manifest.status === 'complete'
        ? (fallbackDifference === 0 ? 'match' : fallbackDifference < 0 ? 'short' : 'over')
        : 'pending',
    expected: manifest.expected_page_count ?? null,
    actual,
    output_items: enabled.length,
    difference: fallbackDifference,
  };
  const status = check.status;

  return <div className={`panel page-count-check ${status}`} aria-label="期待ページ数チェック">
    <div>
      <strong>期待ページ数チェック</strong>
      <p className="muted">{pageCountSummary(check)}</p>
    </div>
    <div className="row">
      <label>期待ページ数
        <input
          aria-label="レビュー期待ページ数"
          type="number"
          min="1"
          max="10000"
          step="1"
          placeholder="未設定"
          value={draft}
          onChange={event => setDraft(event.target.value)}
        />
      </label>
      <button
        disabled={busy || !changed || (draft !== '' && (!Number.isInteger(normalized) || normalized < 1))}
        onClick={() => onSave(normalized)}
      >ページ数を保存</button>
      {manifest.expected_page_count != null && <button
        disabled={busy}
        onClick={() => {
          setDraft('');
          onSave(null);
        }}
      >解除</button>}
    </div>
    <p className="muted">見開きは2ページ、表紙・左右分割・外部画像は1ページとして数えます。</p>
  </div>;
}

function BookMetadataEditor({ metadata = emptyBookMetadata, busy, onSave }) {
  const [draft, setDraft] = useState({ ...emptyBookMetadata, ...metadata });
  useEffect(() => {
    setDraft({ ...emptyBookMetadata, ...metadata });
  }, [metadata]);

  const setField = (field, value) => setDraft(current => ({ ...current, [field]: value }));
  const normalized = Object.fromEntries(
    Object.entries(draft).filter(([, value]) => value.trim()),
  );
  const current = Object.fromEntries(
    Object.entries({ ...emptyBookMetadata, ...metadata }).filter(([, value]) => String(value).trim()),
  );
  const changed = JSON.stringify(normalized) !== JSON.stringify(current);

  return <details className="book-metadata" open>
    <summary>書籍メタデータ</summary>
    <p className="muted">タイトル等をプロジェクトに保存し、PDF内部メタデータとCBZのComicInfo.xmlへ反映します。タイトルは出力ファイル名にも使われます。</p>
    <div className="book-metadata-grid">
      <label>タイトル<input aria-label="書籍タイトル" maxLength="200" value={draft.title} onChange={event => setField('title', event.target.value)} placeholder="例: ONE PIECE 1" /></label>
      <label>著者<input aria-label="著者" maxLength="200" value={draft.author} onChange={event => setField('author', event.target.value)} placeholder="例: 尾田栄一郎" /></label>
      <label>シリーズ<input aria-label="シリーズ" maxLength="200" value={draft.series} onChange={event => setField('series', event.target.value)} /></label>
      <label>巻数<input aria-label="巻数" maxLength="64" value={draft.volume} onChange={event => setField('volume', event.target.value)} placeholder="例: 1" /></label>
      <label>出版社<input aria-label="出版社" maxLength="200" value={draft.publisher} onChange={event => setField('publisher', event.target.value)} /></label>
      <label>言語<input aria-label="言語" maxLength="32" value={draft.language} onChange={event => setField('language', event.target.value)} placeholder="ja" /></label>
    </div>
    <div className="row">
      <button disabled={busy || !changed} onClick={() => onSave(normalized)}>メタデータを保存</button>
      {metadata.title && <span className="muted">タイトルをPDF / CBZのファイル名に使用します。</span>}
    </div>
  </details>;
}

export default function Review({ manifest, file, busy, exporting = false, onEdit, onImportExternal }) {
  const [suspectsOnly, setSuspectsOnly] = useState(false);
  const [showExcluded, setShowExcluded] = useState(false);
  const [timestamp, setTimestamp] = useState('');
  const [draggedPageId, setDraggedPageId] = useState(null);
  const [dragOverPageId, setDragOverPageId] = useState(null);
  const enabled = manifest.pages.filter(page => page.enabled);
  let number = 0;
  const numbered = manifest.pages.map(page => ({ ...page, number: page.enabled ? ++number : null }));
  const pdfReady = Boolean(manifest.pdf && !manifest.pdf_stale);
  const cbzReady = Boolean(manifest.cbz && !manifest.pdf_stale);
  const exportsReady = pdfReady && cbzReady;
  const exportLabel = exporting
    ? 'PDF / CBZ生成中…'
    : (exportsReady ? 'PDF / CBZを再出力' : 'PDF / CBZを出力');
  const exportStatus = exporting
    ? 'PDF / CBZを生成中です。完了すると最新版を確認・保存できます。'
    : manifest.pdf_stale
      ? '編集後のPDF / CBZは未出力です。「PDF / CBZを出力」で反映してください。'
      : exportsReady
        ? '現在のページ順・画質でPDF / CBZを出力済みです。'
        : 'PDFは出力済みです。CBZも作成するには「PDF / CBZを出力」を実行してください。';
  const qualitySummary = qualityReviewSummary(manifest.pages);
  const pageHistory = manifest.page_history ?? {};
  const undoEntries = Array.isArray(pageHistory.undo) ? pageHistory.undo : [];
  const redoEntries = Array.isArray(pageHistory.redo) ? pageHistory.redo : [];
  const undoLabel = undoEntries.at(-1)?.label;
  const redoLabel = redoEntries.at(-1)?.label;

  const dropPage = targetId => {
    const sourceId = draggedPageId;
    setDraggedPageId(null);
    setDragOverPageId(null);
    if (!sourceId || sourceId === targetId || busy) return;
    const pageIds = reorderPageIds(
      manifest.pages.map(page => page.id),
      sourceId,
      targetId,
    );
    if (pageIds.every((pageId, index) => pageId === manifest.pages[index].id)) return;
    onEdit('reorder_pages', { page_ids: pageIds });
  };

  return <section>
    <div className="review-head"><div><p className="step">03 / 確認して仕上げる</p><h2>{enabled.length} ページ / 要確認 {enabled.filter(needsReview).length}</h2></div>
      <div className="row"><button className="primary" aria-busy={exporting ? 'true' : undefined} disabled={busy || !enabled.length} onClick={() => onEdit('export')}>{exportLabel}</button>
        {pdfReady && <a className="button" href={file(manifest.pdf)} target="_blank" rel="noopener">PDFを開く ↗</a>}
        {cbzReady && <a className="button" href={file(manifest.cbz)} download>CBZを保存 ↓</a>}</div></div>
    <p className="muted">{exportStatus}</p>
    <BookMetadataEditor metadata={manifest.book_metadata} busy={busy} onSave={metadata => onEdit('book_metadata', { metadata })} />
    <PageCountCheck
      manifest={manifest}
      busy={busy}
      onSave={expectedPageCount => onEdit('expected_page_count', { expected_page_count: expectedPageCount })}
    />
    {qualitySummary.total > 0 && <div className="panel final-quality-summary" aria-label="最終品質チェック">
      <strong>要確認 {qualitySummary.total}件</strong>
      <div className="final-quality-counts">
        {qualitySummary.categories.map(category => <span key={category.label}>{category.label} {category.count}</span>)}
      </div>
    </div>}
    {manifest.source_type !== 'image_folder' && <VideoTimeline
      manifest={manifest}
      busy={busy}
      onSelectTime={time => setTimestamp(time.toFixed(2))}
    />}
    <div className="toolbar panel">
      <div className="history-controls row" aria-label="ページ編集履歴">
        <button
          disabled={busy || !undoEntries.length}
          title={undoLabel ? `元に戻す: ${undoLabel}` : '元に戻せるページ編集はありません'}
          onClick={() => onEdit('undo_page_edit')}
        >↶ 元に戻す</button>
        <button
          disabled={busy || !redoEntries.length}
          title={redoLabel ? `やり直す: ${redoLabel}` : 'やり直せるページ編集はありません'}
          onClick={() => onEdit('redo_page_edit')}
        >↷ やり直す</button>
        {(undoLabel || redoLabel) && <span className="history-status">
          {undoLabel ? `直前: ${undoLabel}` : `やり直し可: ${redoLabel}`}
        </span>}
      </div>
      <label className="checkbox"><input type="checkbox" checked={suspectsOnly} onChange={event => setSuspectsOnly(event.target.checked)} /> 要確認だけ表示</label>
      <label className="checkbox"><input type="checkbox" checked={showExcluded} onChange={event => setShowExcluded(event.target.checked)} /> 除外ページも表示</label>
      {manifest.source_type !== 'image_folder' && <form className="row" onSubmit={event => { event.preventDefault(); onEdit('add_frame', { time: Number(timestamp) }); }}>
        <input aria-label="追加する動画の秒数" type="number" min="0" max={manifest.metadata.duration - .001} step="any" placeholder="動画の秒数" required value={timestamp} onChange={event => setTimestamp(event.target.value)} />
        <button disabled={busy || timestamp === ''}>この時刻から追加</button></form>}
      <label className="external-page-import">外部画像をページ追加
        <input
          aria-label="外部画像をページ追加"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          disabled={busy}
          onChange={event => {
            const image = event.target.files?.[0];
            if (image) onImportExternal?.(image);
            event.target.value = '';
          }}
        />
      </label>
    </div>
    <p className="muted reorder-help">ページは「⠿ ドラッグ」で並べ替えできます。← / →もそのまま使えます。</p>
    <div className={`page-grid ${manifest.pages.some(page => page.side === 'spread') ? 'with-spreads' : ''}`}>{numbered.filter(page => (page.enabled || showExcluded) && (!suspectsOnly || needsReview(page))).map(page => <article
      key={page.id}
      className={`page-card ${needsReview(page) ? 'suspect' : ''} ${page.enabled ? '' : 'excluded'} ${draggedPageId === page.id ? 'dragging' : ''} ${dragOverPageId === page.id ? 'drag-over' : ''}`}
      onDragOver={event => {
        if (busy || !draggedPageId) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
        setDragOverPageId(page.id);
      }}
      onDragLeave={event => {
        const nextTarget = event.relatedTarget;
        if (!nextTarget || !event.currentTarget.contains(nextTarget)) {
          setDragOverPageId(null);
        }
      }}
      onDrop={event => {
        event.preventDefault();
        dropPage(page.id);
      }}
    >
      <div
        className="page-drag-handle"
        draggable={!busy}
        role="button"
        tabIndex={0}
        aria-label={`${page.id}をドラッグして並べ替え`}
        title="ドラッグしてページを並べ替え"
        onDragStart={event => {
          if (busy) {
            event.preventDefault();
            return;
          }
          setDraggedPageId(page.id);
          setDragOverPageId(page.id);
          event.dataTransfer.effectAllowed = 'move';
          event.dataTransfer.setData('text/plain', page.id);
        }}
        onDragEnd={() => {
          setDraggedPageId(null);
          setDragOverPageId(null);
        }}
      >⠿ ドラッグ</div>
      <ImageLink file={file} path={page.path} preview={page.preview} />
      <h3>{page.number ? String(page.number).padStart(3, '0') : '除外'} · {pageSideLabel(page.side)}</h3>
      <p>{reasons(page.suspect)}</p>
      {page.candidate_time !== undefined && <p className="muted">候補 #{page.candidate_id} · {page.candidate_time.toFixed(2)}s</p>}
      {!['cover', 'external'].includes(page.side) && <PageReviewControls page={page} manifest={manifest} file={file} busy={busy} onEdit={onEdit} />}
      {page.source === 'external_image' && <p className="muted">外部画像: {page.external_name || '読み込み画像'}</p>}
      {page.source_kind === 'image_folder' && <p className="muted">フォルダ画像: {page.external_name || '読み込み画像'}</p>}
      {page.safe_fix_suggestions?.length > 0 && <div className="dewarp-meta safe-fix-suggestions">
        <strong>安全な改善候補</strong>
        <p className="muted">既存の撮影候補だけを比較し、新しい既知リスクを増やさない候補のみ表示しています。</p>
        {page.safe_fix_suggestions.map(suggestion => <div className="row" key={suggestion.candidate_id}>
          <span>候補 #{suggestion.candidate_id} · {safeFixSummary(suggestion)}</span>
          <button disabled={busy} onClick={() => onEdit('select_candidate', {
            spread_id: page.spread_id,
            candidate_id: suggestion.candidate_id,
            ...(suggestion.side ? { side: suggestion.side } : {}),
          })}>この改善候補を採用</button>
        </div>)}
      </div>}
      {page.final_quality?.reasons?.length > 0 && <div className="dewarp-meta">
        <span>完成画像QA: {reasons(page.final_quality.reasons)}</span>
        {page.final_quality.adjacent_duplicate && <p className="muted">前ページ {page.final_quality.adjacent_duplicate.other_page_id} と類似 · SSIM {(page.final_quality.adjacent_duplicate.ssim * 100).toFixed(1)}%</p>}
      </div>}
      {page.finger_repair && page.finger_repair.status !== 'disabled' && <div className="dewarp-meta">
        <span>{repairTitle(page.finger_repair)}: {page.finger_repair.status === 'complete' ? '完了' : page.finger_repair.status === 'clean' ? repairCleanLabel(page.finger_repair) : page.finger_repair.status === 'unavailable' ? 'マスクなし' : '一部のみ'}{fingerRepairCoverageSummary(page.finger_repair) ? ` · ${fingerRepairCoverageSummary(page.finger_repair)}` : ''}{page.finger_repair.donors?.length ? ` · donor #${page.finger_repair.donors.join(', #')}` : ''}{fingerFallbackLabel(page.finger_repair)}{localAlignmentSummary(page.finger_repair) ? ` · ${localAlignmentSummary(page.finger_repair)}` : ''}</span>
        {page.finger_repair.unresolved_mask && <p className="muted">要確認: 未補修領域が残っています。文字・コマ線・網点・指の輪郭に不自然さがないか確認してください。</p>}
        {!page.finger_repair.unresolved_mask && page.finger_repair.status === 'incomplete' && <p className="muted">隠れた部分を別候補から十分に補修できず、指が残っています。別の候補も確認してください。</p>}
        <div className="row">{page.finger_repair.target_mask && <a href={file(page.finger_repair.target_mask)} target="_blank" rel="noopener">遮蔽マスク ↗</a>}
          {page.finger_repair.glare_mask && <a href={file(page.finger_repair.glare_mask)} target="_blank" rel="noopener">反射マスク ↗</a>}
          {page.finger_repair.unresolved_mask && <a href={file(page.finger_repair.unresolved_mask)} target="_blank" rel="noopener">未補修領域 ↗</a>}</div>
      </div>}
      {backgroundFillLabel(page.background_fill) && <div className="dewarp-meta">
        <span>{backgroundFillLabel(page.background_fill)}</span>
        {page.background_fill?.mask && <div className="row"><a href={file(page.background_fill.mask)} target="_blank" rel="noopener">ページmask ↗</a></div>}
      </div>}
      {page.dewarp?.mode === 'auto' && <div className="dewarp-meta">
        <span>湾曲補正: {page.dewarp.status === 'applied' ? `適用 最大 ${(page.dewarp.strength * 100).toFixed(1)}%${page.dewarp.profile_variation ? ` · 高さ方向差 ${(page.dewarp.profile_variation * 100).toFixed(1)}%` : ''}` : page.dewarp.status === 'disabled' ? 'ページ単位でOFF' : page.dewarp.status === 'not_needed' ? '補正不要' : '見送り'}{page.dewarp.confidence !== undefined ? ` · 信頼度 ${Math.round(page.dewarp.confidence * 100)}%` : ''}</span>
        <div className="row">{page.dewarp.before && <a href={file(page.dewarp.before)} target="_blank" rel="noopener">補正前 ↗</a>}
          {page.dewarp.debug_grid && <a href={file(page.dewarp.debug_grid)} target="_blank" rel="noopener">remap ↗</a>}</div>
      </div>}
      <div className="row"><button disabled={busy} onClick={() => onEdit('toggle_page', { page_id: page.id })}>{page.enabled ? '除外' : '復元'}</button>
        <button disabled={busy} aria-label={`${page.id}を前へ`} onClick={() => onEdit('move_page', { page_id: page.id, delta: -1 })}>←</button>
        <button disabled={busy} aria-label={`${page.id}を後ろへ`} onClick={() => onEdit('move_page', { page_id: page.id, delta: 1 })}>→</button>
        <label className="external-page-replace">画像で差し替え
          <input
            aria-label={`${page.id}を外部画像で差し替え`}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            disabled={busy}
            onChange={event => {
              const image = event.target.files?.[0];
              if (image) onImportExternal?.(image, page.id);
              event.target.value = '';
            }}
          />
        </label></div>
    </article>)}</div>
    <h2 className="spreads-heading">見開き・候補フレーム</h2><p className="muted">候補を選ぶと元解像度で再抽出します。左右別モードでは各ページのスコアを個別に確認・差し替えできます。</p>
    {manifest.spreads.map(spread => <Spread key={spread.id} spread={spread} config={manifest.config} file={file} busy={busy} onEdit={onEdit} />)}
  </section>;
}
