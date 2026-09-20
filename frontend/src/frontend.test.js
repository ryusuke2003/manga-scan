import { describe, expect, it } from 'vitest';

import { fileUrl } from './api.js';
import { normalizedPoint } from './components/RoiSelector.jsx';
import { didJobFinish } from './useScanner.js';

describe('frontend helpers', () => {
  it('builds encoded local file URLs', () => {
    expect(fileUrl('scan 01', 'pages/right page.png', 3))
      .toBe('/files/scan%2001/pages/right%20page.png?v=3');
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
});
