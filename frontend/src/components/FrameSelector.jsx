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
  rotation,
  rotationDetection,
  onRotation,
  notice,
}) {
  const [value, setValue] = useState(String(time));
  const [loadedImageUrl, setLoadedImageUrl] = useState(null);

  useEffect(() => setValue(String(time)), [time]);

  const preview = next => {
    const selected = clampTime(next, duration);
    setValue(String(Number(selected.toFixed(3))));
    onPreview(selected);
  };

  const numericValue = Number(value);
  const hasInput = String(value).trim() !== '' && Number.isFinite(numericValue);
  const maxTime = Math.max(0, duration - 0.001);
  const inputInRange = hasInput && numericValue >= 0 && numericValue <= maxTime;
  const selected = hasInput ? clampTime(numericValue, duration) : null;
  const previewed = clampTime(time, duration);
  const previewIsCurrent = inputInRange
    && Math.abs(numericValue - previewed) < 0.0005
    && loadedImageUrl === imageUrl;

  return <section className="panel">
    <p className="step">{step}</p>
    <h2>{title}</h2>
    <p className="muted">{description}</p>
    {notice}
    <div className="frame-wrap">
      <img
        key={imageUrl}
        className="frame-preview"
        src={imageUrl}
        alt="選択中の動画フレーム"
        onLoad={() => setLoadedImageUrl(imageUrl)}
      />
    </div>
    {onRotation && <div className="rotation-confirm">
      <div>
        <strong>画像の向き</strong>
        {rotationDetection?.source === 'page_geometry' && <p className="muted">自動判定: {rotation}° · 信頼度 {Math.round((rotationDetection.confidence ?? 0) * 100)}%。違って見える場合だけ変更してください。</p>}
        {rotationDetection?.source === 'video_metadata' && <p className="muted">動画の回転メタデータをFFmpegが反映済みです。プレビューが正しければそのままでOKです。</p>}
        {rotationDetection?.source === 'manual' && <p className="muted">手動で向きを指定しています。</p>}
      </div>
      <label>プレビューの向き
        <select value={String(rotation ?? 0)} disabled={busy} onChange={event => onRotation(Number(event.target.value))}>
          <option value="0">そのまま</option>
          <option value="90">右へ90°</option>
          <option value="180">180°</option>
          <option value="270">左へ90°</option>
        </select>
      </label>
    </div>}
    <div className="frame-controls">
      <button type="button" disabled={busy || selected === null} onClick={() => preview(selected - 1)}>−1秒</button>
      <button type="button" disabled={busy || selected === null} onClick={() => preview(selected - 0.1)}>−0.1秒</button>
      <label>動画の秒数
        <input type="number" min="0" max={maxTime} step="0.1"
          value={value} disabled={busy} onChange={event => setValue(event.target.value)} />
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
