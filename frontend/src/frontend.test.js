import { describe, expect, it } from 'vitest';

import { fileUrl } from './api.js';
import { clampTime } from './components/FrameSelector.jsx';
import { normalizedPoint } from './components/RoiSelector.jsx';
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
