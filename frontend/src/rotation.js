export function rotateNormalizedRoi(points, rotation = 0) {
  if (!points?.length || rotation === 0) return points || [];
  if (![90, 180, 270].includes(rotation)) throw new Error('rotation must be 0, 90, 180, or 270');

  const rotated = points.map(([x, y]) => {
    if (rotation === 90) return [1 - y, x];
    if (rotation === 180) return [1 - x, 1 - y];
    return [y, 1 - x];
  });
  const shift = rotation / 90;
  return rotated.slice(shift).concat(rotated.slice(0, shift));
}
