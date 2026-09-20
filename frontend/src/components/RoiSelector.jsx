import { useEffect, useRef, useState } from 'react';

export function normalizedPoint(clientX, clientY, bounds) {
  const clamp = value => Math.min(1, Math.max(0, value));
  return [clamp((clientX - bounds.left) / bounds.width), clamp((clientY - bounds.top) / bounds.height)];
}

export default function RoiSelector({
  imageUrl,
  initialPoints,
  metadata,
  busy,
  onStart,
  step = '02 / 見開きを囲む',
  title = '漫画の外周を4点で指定',
  description = '左上 → 右上 → 右下 → 左下 の順にクリック。机が入らないよう、紙の外周に合わせてください。',
  actionLabel = '抽出を開始 →',
  rotation = 0,
}) {
  const [points, setPoints] = useState(() => initialPoints || []);
  const [image, setImage] = useState(null);
  const [imageError, setImageError] = useState(false);
  const canvas = useRef(null);
  useEffect(() => {
    let cancelled = false;
    const next = new Image();
    setImageError(false);
    next.onload = () => { if (!cancelled) setImage(next); };
    next.onerror = () => { if (!cancelled) setImageError(true); };
    next.src = imageUrl;
    return () => { cancelled = true; };
  }, [imageUrl]);

  useEffect(() => {
    if (!image || !canvas.current) return;
    const element = canvas.current;
    const sideways = rotation === 90 || rotation === 270;
    const width = sideways ? image.naturalHeight : image.naturalWidth;
    const height = sideways ? image.naturalWidth : image.naturalHeight;
    const scale = Math.min(1, 1100 / width);
    element.width = Math.round(width * scale);
    element.height = Math.round(height * scale);
    const ctx = element.getContext('2d');
    ctx.save();
    ctx.translate(element.width / 2, element.height / 2);
    ctx.rotate(rotation * Math.PI / 180);
    ctx.drawImage(image, -image.naturalWidth * scale / 2, -image.naturalHeight * scale / 2, image.naturalWidth * scale, image.naturalHeight * scale);
    ctx.restore();
    if (!points.length) return;
    ctx.beginPath();
    points.forEach(([x, y], index) => index ? ctx.lineTo(x * element.width, y * element.height) : ctx.moveTo(x * element.width, y * element.height));
    if (points.length === 4) { ctx.closePath(); ctx.fillStyle = '#407c5228'; ctx.fill(); }
    ctx.strokeStyle = '#9dffab'; ctx.lineWidth = 3; ctx.stroke();
    points.forEach(([x, y], index) => {
      ctx.fillStyle = '#24563d'; ctx.beginPath(); ctx.arc(x * element.width, y * element.height, 13, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(index + 1, x * element.width, y * element.height + 5);
    });
  }, [image, points, rotation]);

  return <section className="panel">
    <p className="step">{step}</p><h2>{title}</h2>
    <p className="muted">{description}</p>
    {imageError && <p role="alert">最初のフレームを読み込めませんでした。プロジェクトを開き直してください。</p>}
    <div className="canvas-wrap"><canvas ref={canvas} aria-label="見開きの四隅を指定" onClick={event => {
      if (busy || !image || points.length >= 4) return;
      const point = normalizedPoint(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());
      setPoints(current => [...current, point]);
    }} /></div>
    <div className="row"><p id="roi-count">{points.length} / 4 点</p>
      <button disabled={busy} onClick={() => setPoints([])}>やり直す</button>
      <button className="primary" disabled={busy || points.length !== 4 || !image || imageError} onClick={() => onStart(points)}>{actionLabel}</button></div>
    {metadata && <p className="muted">{metadata.display_width} × {metadata.display_height} · {metadata.fps.toFixed(2)} fps · {metadata.duration.toFixed(1)} 秒 · {metadata.codec}</p>}
  </section>;
}
