import { useEffect, useRef, useState } from 'react';

const FALLBACK_CONFIG = {
  output_layout: 'spread',
  reading_order: 'rtl',
  image_format: 'png',
  jpeg_quality: 92,
  hand_backend: 'mediapipe',
  finger_repair: true,
  finger_repair_fallback: 'paper',
  page_background_fill: 'paper',
  candidate_selection_mode: 'spread',
  grayscale: false,
  rotation: 0,
  auto_rotation: true,
  refine_quad: true,
  perspective_mode: 'spread',
  page_contour_min_confidence: 0.55,
  split_mode: 'auto',
  dewarp_mode: 'auto',
  dewarp_strength: 0,
  dewarp_max_strength: 0.25,
  dewarp_min_confidence: 0.6,
  illumination_correction: true,
  illumination_strength: 0.7,
  white_normalization: true,
  white_target: 245,
  white_strength: 0.6,
};

export const CORRECTION_PRESETS = {
  original: {
    label: '原画優先',
    description: '色やトーンをできるだけ変えず、必要最小限の補正だけにします。',
    settings: {
      refine_quad: false,
      perspective_mode: 'spread',
      split_mode: 'center',
      dewarp_mode: 'off',
      illumination_correction: false,
      white_normalization: false,
      page_background_fill: 'preserve',
    },
  },
  standard: {
    label: '標準補正',
    description: '左右別台形・背位置・湾曲・照明を自動補正。白背景化は行いません。',
    settings: {
      refine_quad: true,
      perspective_mode: 'per_page',
      page_contour_min_confidence: 0.6,
      split_mode: 'auto',
      dewarp_mode: 'auto',
      dewarp_max_strength: 0.2,
      dewarp_min_confidence: 0.65,
      illumination_correction: true,
      illumination_strength: 0.6,
      white_normalization: false,
      page_background_fill: 'preserve',
    },
  },
  scan: {
    label: 'スキャン風',
    description: '標準補正に加えて紙面の白を整え、スキャナに近い見た目を狙います。',
    settings: {
      refine_quad: true,
      perspective_mode: 'spread',
      page_contour_min_confidence: 0.55,
      split_mode: 'auto',
      dewarp_mode: 'auto',
      dewarp_max_strength: 0.25,
      dewarp_min_confidence: 0.6,
      illumination_correction: true,
      illumination_strength: 0.7,
      white_normalization: true,
      white_target: 245,
      white_strength: 0.6,
      page_background_fill: 'paper',
    },
  },
};

export function buildInitialConfig(defaults = {}) {
  return Object.fromEntries(
    Object.entries(FALLBACK_CONFIG).map(([key, fallback]) => [
      key,
      defaults?.[key] ?? fallback,
    ]),
  );
}

function normalizedPresetValue(config, key) {
  if (key === 'dewarp_mode' && config.dewarp_mode === 'manual' && Number(config.dewarp_strength) === 0) {
    return 'off';
  }
  return config[key];
}

export function correctionPresetForConfig(config) {
  for (const [name, preset] of Object.entries(CORRECTION_PRESETS)) {
    const matches = Object.entries(preset.settings)
      .every(([key, value]) => normalizedPresetValue(config, key) === value);
    if (matches) return name;
  }
  return 'custom';
}

export function applyCorrectionPreset(config, name) {
  const preset = CORRECTION_PRESETS[name];
  if (!preset) return config;
  return {
    ...config,
    ...preset.settings,
    dewarp_strength: name === 'original' ? 0 : config.dewarp_strength,
  };
}

export default function Setup({ busy, defaults, onChoose, onCreate }) {
  const [video, setVideo] = useState('');
  const [config, setConfig] = useState(() => buildInitialConfig(defaults));
  const customized = useRef(false);
  const initializedDefaults = useRef(Boolean(defaults));

  useEffect(() => {
    if (defaults && !initializedDefaults.current && !customized.current) {
      setConfig(buildInitialConfig(defaults));
      initializedDefaults.current = true;
    }
  }, [defaults]);

  const change = (name, value) => {
    customized.current = true;
    setConfig(current => ({ ...current, [name]: value }));
  };
  const applyPreset = name => {
    customized.current = true;
    setConfig(current => applyCorrectionPreset(current, name));
  };
  const changeRotation = value => {
    customized.current = true;
    setConfig(current => value === 'auto'
      ? { ...current, auto_rotation: true, rotation: 0 }
      : { ...current, auto_rotation: false, rotation: Number(value) });
  };
  const preset = correctionPresetForConfig(config);

  return <section className="panel">
    <p className="step">01 / 動画を選ぶ</p><h2>机の上で撮った動画を読み込みます</h2>
    <p className="muted">.mov / .mp4 に対応。カメラを固定し、各見開きで手を引いて少し静止すると、きれいに抽出できます。</p>
    <form onSubmit={event => { event.preventDefault(); onCreate(video.trim(), config); }}>
      <label htmlFor="video">動画のローカルパス</label>
      <div className="row"><input id="video" placeholder="/Users/you/Movies/manga.mov" required value={video} onChange={event => setVideo(event.target.value)} />
        <button type="button" disabled={busy} onClick={() => onChoose(setVideo)}>ファイルを選択</button></div>

      <h3 className="form-heading">出力</h3>
      <div className="options">
        <label>出力形式<select value={config.output_layout} onChange={event => change('output_layout', event.target.value)}><option value="spread">見開きのまま / 標準</option><option value="split">左右のページに分割</option></select></label>
        <label>読む順番<select disabled={config.output_layout === 'spread'} value={config.reading_order} onChange={event => change('reading_order', event.target.value)}><option value="rtl">右 → 左（日本漫画）</option><option value="ltr">左 → 右</option></select></label>
        <label>ページ画像<select value={config.image_format} onChange={event => change('image_format', event.target.value)}><option value="png">PNG / 可逆圧縮</option><option value="jpeg">JPEG / 小さいサイズ</option></select></label>
        <label>JPEG品質<input type="number" min="1" max="100" required value={config.jpeg_quality} onChange={event => change('jpeg_quality', Number(event.target.value))} /></label>
        <label>手の検出<select value={config.hand_backend} onChange={event => {
          const backend = event.target.value;
          change('hand_backend', backend);
          if (backend === 'none') change('finger_repair', false);
        }}><option value="mediapipe">有効 / MediaPipe</option><option value="none">無効 / 全ページに警告</option></select></label>
        <label className="setting-check"><input type="checkbox" checked={config.finger_repair} disabled={config.hand_backend === 'none'} onChange={event => change('finger_repair', event.target.checked)} /><span><strong>別フレームから指を補修</strong><small>同じページの別時刻に写っている実画素だけで指領域を置き換えます。</small></span></label>
        <label>補修できない指<select disabled={!config.finger_repair || config.hand_backend === 'none'} value={config.finger_repair_fallback} onChange={event => change('finger_repair_fallback', event.target.value)}><option value="paper">紙面だけ自然に補完 / おすすめ</option><option value="preserve">元の画像を残す</option><option value="white">未補修部分を白塗り</option></select></label>
        <label>候補フレーム選択<select disabled={config.output_layout === 'spread'} value={config.output_layout === 'spread' ? 'spread' : config.candidate_selection_mode} onChange={event => change('candidate_selection_mode', event.target.value)}><option value="spread">見開き単位</option><option value="per_page">左右ページ別</option></select></label>
      </div>
      <p className="muted">見開きは中央で切らず、1見開きをPDFの1ページに保存します。左右別の台形・分割位置・湾曲補正は、分割出力を選んだときに使います。</p>
      <label className="checkbox"><input type="checkbox" checked={config.grayscale} onChange={event => change('grayscale', event.target.checked)} /> グレースケールで保存</label>

      <div className="correction-head">
        <div><h3 className="form-heading">補正</h3><p className="muted">自動補正はconfidence不足なら従来処理へ戻り、原画を描き足しません。</p></div>
        {preset === 'custom' && <span className="settings-badge">カスタム</span>}
      </div>
      <div className="correction-presets">
        {Object.entries(CORRECTION_PRESETS).map(([name, item]) => <button
          type="button"
          key={name}
          className={preset === name ? 'preset active' : 'preset'}
          aria-pressed={preset === name}
          onClick={() => applyPreset(name)}
        ><strong>{item.label}{name === 'standard' ? ' / おすすめ' : ''}</strong><span>{item.description}</span></button>)}
      </div>

      <details className="advanced-settings">
        <summary>補正の詳細設定</summary>
        <div className="advanced-grid">
          <label>画像の向き<select value={config.auto_rotation ? 'auto' : String(config.rotation)} onChange={event => changeRotation(event.target.value)}><option value="auto">自動判定 / プレビューで確認</option><option value="0">そのまま</option><option value="90">右へ90°</option><option value="180">180°</option><option value="270">左へ90°</option></select></label>
          <label className="setting-check"><input type="checkbox" checked={config.refine_quad} onChange={event => change('refine_quad', event.target.checked)} /><span><strong>見開き外周を自動調整</strong><small>採用候補ごとに左右ページの外周を検出します。</small></span></label>
          <label>左右別の台形補正<select disabled={config.output_layout === 'spread'} value={config.perspective_mode} onChange={event => change('perspective_mode', event.target.value)}><option value="spread">従来方式 / 見開き全体</option><option value="per_page">左右ページを別々に補正</option></select></label>
          {config.refine_quad && (config.output_layout === 'spread' || config.perspective_mode === 'per_page') && <label>ページ輪郭の最低信頼度<input type="number" min="0" max="1" step="0.05" value={config.page_contour_min_confidence} onChange={event => change('page_contour_min_confidence', Number(event.target.value))} /></label>}
          <label>見開きの分割位置<select disabled={config.output_layout === 'spread'} value={config.split_mode} onChange={event => change('split_mode', event.target.value)}><option value="center">中央固定</option><option value="auto">背の位置を自動推定</option></select></label>
          <label>湾曲補正<select disabled={config.output_layout === 'spread'} value={config.output_layout === 'spread' ? 'off' : config.dewarp_mode} onChange={event => change('dewarp_mode', event.target.value)}><option value="off">OFF</option><option value="auto">自動</option><option value="manual">固定強度</option></select></label>
          {config.output_layout === 'split' && config.dewarp_mode === 'manual' && <label>湾曲補正の固定強度<input type="number" min="0" max="0.6" step="0.05" value={config.dewarp_strength} onChange={event => change('dewarp_strength', Number(event.target.value))} /></label>}
          {config.output_layout === 'split' && config.dewarp_mode === 'auto' && <>
            <label>自動湾曲の最大強度<input type="number" min="0" max="0.35" step="0.01" value={config.dewarp_max_strength} onChange={event => change('dewarp_max_strength', Number(event.target.value))} /></label>
            <label>自動湾曲の最低信頼度<input type="number" min="0" max="1" step="0.05" value={config.dewarp_min_confidence} onChange={event => change('dewarp_min_confidence', Number(event.target.value))} /></label>
          </>}
          <label className="setting-check"><input type="checkbox" checked={config.illumination_correction} onChange={event => change('illumination_correction', event.target.checked)} /><span><strong>照明ムラ・影を補正</strong><small>低周波の明るさムラだけを均します。</small></span></label>
          {config.illumination_correction && <label>照明補正の強度<input type="number" min="0" max="1" step="0.05" value={config.illumination_strength} onChange={event => change('illumination_strength', Number(event.target.value))} /></label>}
          <label>ページ外の背景<select disabled={config.output_layout !== 'spread'} value={config.page_background_fill} onChange={event => change('page_background_fill', event.target.value)}><option value="preserve">そのまま残す</option><option value="paper">紙色で隠す / おすすめ</option><option value="white">白で隠す</option></select><small>見開き出力で左右ページ輪郭を高confidenceで検出できたときだけ、机などページ外側を隠します。</small></label>
          <label className="setting-check"><input type="checkbox" checked={config.white_normalization} onChange={event => change('white_normalization', event.target.checked)} /><span><strong>白背景を正規化</strong><small>明るい紙面候補だけを白へ寄せます。</small></span></label>
          {config.white_normalization && <>
            <label>白背景の強度<input type="number" min="0" max="1" step="0.05" value={config.white_strength} onChange={event => change('white_strength', Number(event.target.value))} /></label>
            <label>紙面の白ターゲット<input type="number" min="200" max="255" step="1" value={config.white_target} onChange={event => change('white_target', Number(event.target.value))} /></label>
          </>}
        </div>
      </details>

      <button id="create" className="primary" disabled={busy || !video.trim()}>動画を読み込む →</button>
    </form>
  </section>;
}
