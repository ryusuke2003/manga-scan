import { describe, expect, it } from 'vitest';

import { fileUrl } from './api.js';
import { clampTime } from './components/FrameSelector.jsx';
import { normalizedPoint } from './components/RoiSelector.jsx';
import { detectMissingPageCandidates, timelinePercent } from './timeline.js';
import { didJobFinish, shouldReportPollError } from './useScanner.js';

describe('frontend helpers', () => {
  it('builds encoded local file URLs', () => {
    expect(fileUrl('scan 01', 'pages/right page.png', 3))
      .toBe('/files/scan%2001/pages/right%20page.png?v=3');
  });

  it('clamps setup timestamps to the video duration', () => {
    expect(clampTime(-2, 10)).toBe(0);
    expect(clampTime(4.25, 10)).toBe(4.25);
    expect(clampTime(12, 10)).toBeCloseTo(9.999);
  });

  it('normalizes and clamps ROI pointer coordinates', () => {
    const bounds = { left: 100, top: 50, width: 400, height: 200 };
    expect(normalizedPoint(300, 150, bounds)).toEqual([0.5, 0.5]);
    expect(normalizedPoint(0, 500, bounds)).toEqual([0, 1]);
  });

  it('detects long timeline gaps as missing-page candidates', () => {
    const spreads = [2, 4, 6, 10, 12].map((time, index) => ({
      id: `spread_${index + 1}`,
      start: time - 0.3,
      end: time + 0.3,
      selected: 0,
      candidates: [{ id: 0, time }],
    }));
    const missing = detectMissingPageCandidates(spreads);
    expect(missing).toHaveLength(1);
    expect(missing[0].time).toBeCloseTo(8);
    expect(missing[0].gapRatio).toBeCloseTo(2);

    spreads.splice(3, 0, {
      id: 'manual',
      start: 7.9,
      end: 8.1,
      selected: 0,
      candidates: [{ id: 0, time: 8 }],
      extra_suspect: ['manual_frame'],
    });
    expect(detectMissingPageCandidates(spreads)).toEqual([]);
  });

  it('uses stable interval starts instead of selected-frame timing for gap detection', () => {
    const regular = [
      { id: 'a', start: 2, end: 3.9, selected: 0, candidates: [{ id: 0, time: 3.8 }] },
      { id: 'b', start: 4, end: 5, selected: 0, candidates: [{ id: 0, time: 4.1 }] },
      { id: 'c', start: 6, end: 7.9, selected: 0, candidates: [{ id: 0, time: 7.8 }] },
      { id: 'd', start: 8, end: 9, selected: 0, candidates: [{ id: 0, time: 8.1 }] },
    ];
    expect(detectMissingPageCandidates(regular)).toEqual([]);

    const withGap = regular.map(spread => ({ ...spread }));
    withGap[2] = { ...withGap[2], start: 8, end: 9.9, candidates: [{ id: 0, time: 9.8 }] };
    withGap[3] = { ...withGap[3], start: 10, end: 11, candidates: [{ id: 0, time: 10.1 }] };
    const missing = detectMissingPageCandidates(withGap);
    expect(missing).toHaveLength(1);
    expect(missing[0].time).toBeCloseTo(6);
  });

  it('clamps timeline positions to the video bounds', () => {
    expect(timelinePercent(-1, 20)).toBe(0);
    expect(timelinePercent(5, 20)).toBe(25);
    expect(timelinePercent(30, 20)).toBe(100);
  });

  it('refreshes generated assets only when a background job finishes', () => {
    expect(didJobFinish(false, { busy: false })).toBe(false);
    expect(didJobFinish(false, { busy: true })).toBe(false);
    expect(didJobFinish(true, { busy: true })).toBe(false);
    expect(didJobFinish(true, { busy: false })).toBe(true);
  });

  it('ignores stale polling errors while a mutation is changing project state', () => {
    const error = new Error('404');
    expect(shouldReportPollError(error, false, false)).toBe(true);
    expect(shouldReportPollError(error, false, true)).toBe(false);
    expect(shouldReportPollError(error, true, false)).toBe(false);
    expect(shouldReportPollError(Object.assign(new Error('aborted'), { name: 'AbortError' }), false, false)).toBe(false);
  });
});
