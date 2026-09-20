import { describe, expect, it } from 'vitest';
import { normalizedPoint } from './RoiSelector.jsx';

describe('normalizedPoint', () => {
  const bounds = { left: 100, top: 50, width: 400, height: 200 };

  it('normalizes coordinates inside the canvas', () => {
    expect(normalizedPoint(300, 150, bounds)).toEqual([0.5, 0.5]);
  });

  it('clamps coordinates to the canvas bounds', () => {
    expect(normalizedPoint(0, 500, bounds)).toEqual([0, 1]);
  });
});
