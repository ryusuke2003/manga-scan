function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function spreadTime(spread) {
  const selected = spread.candidates?.find(candidate => candidate.id === spread.selected);
  if (Number.isFinite(selected?.time)) return selected.time;
  if (Number.isFinite(spread.start) && Number.isFinite(spread.end)) {
    return (spread.start + spread.end) / 2;
  }
  return Number(spread.start);
}

export function timelinePercent(time, duration) {
  if (!Number.isFinite(time) || !Number.isFinite(duration) || duration <= 0) return 0;
  return Math.max(0, Math.min(100, time / duration * 100));
}

export function detectMissingPageCandidates(spreads, gapFactor = 1.8, maxPerGap = 4) {
  const points = (spreads || [])
    .filter(spread => !spread.duplicate_of)
    .map(spread => ({ spread, time: spreadTime(spread) }))
    .filter(point => Number.isFinite(point.time))
    .sort((a, b) => a.time - b.time);

  if (points.length < 3) return [];

  const gaps = points.slice(1)
    .map((point, index) => point.time - points[index].time)
    .filter(gap => gap > 0);
  if (gaps.length < 2) return [];

  const typicalGap = median(gaps);
  if (!Number.isFinite(typicalGap) || typicalGap <= 0) return [];

  const candidates = [];
  for (let index = 1; index < points.length; index += 1) {
    const before = points[index - 1];
    const after = points[index];
    const gap = after.time - before.time;
    const ratio = gap / typicalGap;
    if (ratio < gapFactor) continue;

    const missingCount = Math.min(
      maxPerGap,
      Math.max(1, Math.round(ratio) - 1),
    );
    for (let offset = 1; offset <= missingCount; offset += 1) {
      const time = before.time + gap * offset / (missingCount + 1);
      candidates.push({
        id: `${before.spread.id}-${after.spread.id}-${offset}`,
        time,
        gap,
        gapRatio: ratio,
        typicalGap,
        before: before.spread.id,
        after: after.spread.id,
        position: offset,
        count: missingCount,
      });
    }
  }
  return candidates;
}

export function formatTimelineTime(seconds) {
  if (!Number.isFinite(seconds)) return '0:00';
  const whole = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, '0')}`;
}
