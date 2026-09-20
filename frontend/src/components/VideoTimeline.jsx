import {
  detectMissingPageCandidates,
  formatTimelineTime,
  spreadTime,
  timelinePercent,
} from '../timeline.js';

export default function VideoTimeline({ manifest, busy, onSelectTime }) {
  const duration = Number(manifest.metadata?.duration) || 0;
  const spreads = [...(manifest.spreads || [])]
    .filter(spread => Number.isFinite(spread.start))
    .sort((a, b) => a.start - b.start);
  const missing = detectMissingPageCandidates(spreads);

  return <section className="panel timeline-panel">
    <div className="timeline-head">
      <div>
        <p className="step">動画タイムライン</p>
        <h2>検出した見開きと欠落候補</h2>
      </div>
      <span className={`timeline-status ${missing.length ? 'warn' : ''}`}>
        欠落候補 {missing.length}
      </span>
    </div>
    <p className="muted">
      見開きの検出間隔が普段より長い箇所を候補として表示します。ページ番号を認識しているわけではないため、追加前に元動画を確認してください。
    </p>

    <div className="video-timeline" aria-label="動画タイムライン">
      <div className="timeline-track">
        {spreads.map((spread, index) => {
          const start = timelinePercent(spread.start, duration);
          const end = timelinePercent(spread.end, duration);
          const width = Math.max(.45, end - start);
          return <span
            key={spread.id}
            className={`timeline-spread ${spread.duplicate_of ? 'duplicate' : ''}`}
            style={{ left: `${start}%`, width: `${width}%` }}
            title={`${spread.id}: ${spread.start.toFixed(2)}–${spread.end.toFixed(2)}s / 採用 ${spreadTime(spread).toFixed(2)}s`}
          >
            <span>{index + 1}</span>
          </span>;
        })}
        {missing.map(candidate => <button
          key={candidate.id}
          type="button"
          className="timeline-missing"
          style={{ left: `${timelinePercent(candidate.time, duration)}%` }}
          title={`欠落候補 ${candidate.time.toFixed(2)}s`}
          aria-label={`欠落ページ候補 ${candidate.time.toFixed(2)}秒を追加欄へ入力`}
          disabled={busy}
          onClick={() => onSelectTime(candidate.time)}
        />)}
      </div>
      <div className="timeline-scale">
        <span>0:00</span>
        <span>{formatTimelineTime(duration / 2)}</span>
        <span>{formatTimelineTime(duration)}</span>
      </div>
    </div>

    {missing.length ? <div className="missing-candidates">
      {missing.map(candidate => <div className="missing-candidate" key={candidate.id}>
        <div>
          <strong>{candidate.time.toFixed(2)}s</strong>
          <span>
            {candidate.before} → {candidate.after} の間が通常の約 {candidate.gapRatio.toFixed(1)} 倍
          </span>
        </div>
        <button
          type="button"
          disabled={busy}
          onClick={() => onSelectTime(candidate.time)}
        >
          この時刻を追加欄へ
        </button>
      </div>)}
    </div> : <p className="timeline-empty">大きな時間間隔の欠落候補は見つかりませんでした。</p>}
  </section>;
}
