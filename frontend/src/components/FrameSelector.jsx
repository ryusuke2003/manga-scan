import { useEffect, useState } from 'react';

export function clampTime(value, duration) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.min(Math.max(0, numeric), Math.max(0, duration - 0.001));
}

export default function FrameSelector({
  step,
  title,
  description,
  imageUrl,
  time = 0,
  duration,
  busy,
  confirmLabel,
  onPreview,
  onConfirm,
  onSkip,
}) {
  const [value, setValue] = useState(String(time));

  useEffect(() => setValue(String(time)), [time]);

  const preview = next => {
    const selected = clampTime(next, duration);
    setValue(String(Number(selected.toFixed(3))));
    onPreview(selected);
  };

  const hasInput = String(value).trim() !== '' && Number.isFinite(Number(value));
  const selected = hasInput ? clampTime(value, duration) : null;
  const previewed = clampTime(time, duration);
  const previewIsCurrent = selected !== null && Math.abs(selected - previewed) < 0.0005;

  return <section className="panel">
    <p className="step">{step}</p>
    <h2>{title}</h2>
    <p className="muted">{description}</p>
    <div className="frame-wrap">
      <img className="frame-preview" src={imageUrl} alt="選択中の動画フレーム" />
    </div>
    <div className="frame-controls">
      <button type="button" disabled={busy || selected === null} onClick={() => preview(selected - 1)}>−1秒</button>
      <button type="button" disabled={busy || selected === null} onClick={() => preview(selected - 0.1)}>−0.1秒</button>
      <label>動画の秒数
        <input type="number" min="0" max={Math.max(0, duration - 0.001)} step="0.1"
          value={value} onChange={event => setValue(event.target.value)} />
      </label>
      <button type="button" disabled={busy || selected === null} onClick={() => preview(selected)}>プレビュー更新</button>
      <button type="button" disabled={busy || selected === null} onClick={() => preview(selected + 0.1)}>＋0.1秒</button>
      <button type="button" disabled={busy || selected === null} onClick={() => preview(selected + 1)}>＋1秒</button>
    </div>
    {!previewIsCurrent && <p className="muted">時刻を変更した場合は「プレビュー更新」で画像を確認してから確定してください。</p>}
    <div className="row frame-actions">
      {onSkip && <button type="button" disabled={busy} onClick={onSkip}>表紙なしで進む</button>}
      <button type="button" className="primary" disabled={busy || !previewIsCurrent}
        onClick={() => onConfirm(previewed)}>{confirmLabel}</button>
    </div>
  </section>;
}
