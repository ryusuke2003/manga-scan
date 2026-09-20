import { useState } from 'react';

export default function Setup({ busy, onChoose, onCreate }) {
  const [video, setVideo] = useState('');
  const [config, setConfig] = useState({ reading_order: 'rtl', image_format: 'png', jpeg_quality: 92, hand_backend: 'mediapipe', candidate_selection_mode: 'spread', grayscale: false });
  const change = (name, value) => setConfig(current => ({ ...current, [name]: value }));
  return <section className="panel">
    <p className="step">01 / 動画を選ぶ</p><h2>机の上で撮った動画を読み込みます</h2>
    <p className="muted">.mov / .mp4 に対応。カメラを固定し、各見開きで手を引いて少し静止すると、きれいに抽出できます。</p>
    <form onSubmit={event => { event.preventDefault(); onCreate(video.trim(), config); }}>
      <label htmlFor="video">動画のローカルパス</label>
      <div className="row"><input id="video" placeholder="/Users/you/Movies/manga.mov" required value={video} onChange={event => setVideo(event.target.value)} />
        <button type="button" disabled={busy} onClick={() => onChoose(setVideo)}>ファイルを選択</button></div>
      <div className="options">
        <label>読む順番<select value={config.reading_order} onChange={event => change('reading_order', event.target.value)}><option value="rtl">右 → 左（日本漫画）</option><option value="ltr">左 → 右</option></select></label>
        <label>ページ画像<select value={config.image_format} onChange={event => change('image_format', event.target.value)}><option value="png">PNG / 可逆圧縮</option><option value="jpeg">JPEG / 小さいサイズ</option></select></label>
        <label>JPEG品質<input type="number" min="1" max="100" required value={config.jpeg_quality} onChange={event => change('jpeg_quality', Number(event.target.value))} /></label>
        <label>手の検出<select value={config.hand_backend} onChange={event => change('hand_backend', event.target.value)}><option value="mediapipe">有効 / MediaPipe</option><option value="none">無効 / 全ページに警告</option></select></label>
        <label>候補フレーム選択<select value={config.candidate_selection_mode} onChange={event => change('candidate_selection_mode', event.target.value)}><option value="spread">見開き単位 / 従来</option><option value="per_page">左右ページ別</option></select></label>
      </div>
      <label className="checkbox"><input type="checkbox" checked={config.grayscale} onChange={event => change('grayscale', event.target.checked)} /> グレースケールで保存</label>
      <button id="create" className="primary" disabled={busy || !video.trim()}>動画を読み込む →</button>
    </form>
  </section>;
}
