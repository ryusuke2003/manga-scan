import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import Review, {
  fingerRepairCoverageSummary,
  localAlignmentSummary,
  pageCountSummary,
  qualityReviewSummary,
  reorderPageIds,
  safeFixSummary,
} from './Review.jsx';
import Setup from './Setup.jsx';
import VideoTimeline from './VideoTimeline.jsx';

const manifest = {
  config: { output_layout: 'spread', candidate_selection_mode: 'per_page', spine_ratio: .5 },
  metadata: { duration: 10 },
  pages: [{ id: 's_whole', spread_id: 's', side: 'spread', enabled: true, suspect: [], path: 'whole.png' }],
  spreads: [{ id: 's', start: 1, end: 2, selected: 0, suspect: [], whole_spread_crop: {
    status: 'auto_pages', candidate_id: 0, confidence: .86, roi: [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
  }, page_contour_debug: 'debug/contour.jpg', candidates: [{ id: 0, time: 1,
    path: 'original.png', preview: 'preview.png', hand_mask: 'mask.png',
    roi: [[0, 0], [1, 0], [1, 1], [0, 1]], metrics: { score: 1, sharpness: 100, hand_overlap: 0 } }] }],
};

it('shows when the whole-spread crop was corrected from the page boundary', () => {
  const corrected = {
    ...manifest,
    spreads: [{ ...manifest.spreads[0], whole_spread_crop: {
      ...manifest.spreads[0].whole_spread_crop,
      status: 'auto_boundary',
    } }],
  };
  render(<Review manifest={corrected} file={path => path} busy={false} onEdit={vi.fn()} />);
  expect(screen.getByText(/紙面と机の境界から外周を自動補正/)).toBeTruthy();
});

it('keeps the long missing-candidate list collapsed by default', () => {
  const timelineManifest = {
    metadata: { duration: 40 },
    spreads: [],
    page_turn_analysis: {
      version: 2,
      missing_candidates: [{
        id: 'turn_0001-turn_0002',
        time: 17.9,
        left_turn: 'turn_0001',
        right_turn: 'turn_0002',
      }],
    },
  };

  render(<VideoTimeline manifest={timelineManifest} busy={false} onSelectTime={vi.fn()} />);

  const summary = screen.getByText('欠落候補の詳細（1件）');
  const details = summary.closest('details');
  expect(details.open).toBe(false);

  fireEvent.click(summary);
  expect(details.open).toBe(true);
});

it('reorders full page IDs deterministically', () => {
  expect(reorderPageIds(['a', 'b', 'c', 'd'], 'a', 'c')).toEqual(['b', 'c', 'a', 'd']);
  expect(reorderPageIds(['a', 'b', 'c', 'd'], 'd', 'b')).toEqual(['a', 'd', 'b', 'c']);
  expect(reorderPageIds(['a', 'b'], 'missing', 'b')).toEqual(['a', 'b']);
});

it('supports drag reorder and page edit undo redo controls', () => {
  const onEdit = vi.fn();
  const pages = ['a', 'b', 'c'].map(id => ({
    id,
    spread_id: id,
    side: 'cover',
    enabled: true,
    suspect: [],
    path: `${id}.png`,
  }));
  const reviewManifest = {
    ...manifest,
    pages,
    spreads: [],
    page_history: {
      undo: [{ label: 'ページ並び替え', state: { order: ['a', 'b', 'c'], disabled: [] } }],
      redo: [{ label: '除外 / 復元', state: { order: ['a', 'c', 'b'], disabled: ['c'] } }],
    },
  };

  render(<Review manifest={reviewManifest} file={path => path} busy={false} onEdit={onEdit} />);

  fireEvent.click(screen.getByRole('button', { name: /元に戻す/ }));
  expect(onEdit).toHaveBeenCalledWith('undo_page_edit');

  fireEvent.click(screen.getByRole('button', { name: /やり直す/ }));
  expect(onEdit).toHaveBeenCalledWith('redo_page_edit');

  const source = screen.getByLabelText('aをドラッグして並べ替え');
  const target = screen.getByLabelText('cをドラッグして並べ替え').closest('article');
  const dataTransfer = {
    effectAllowed: '',
    dropEffect: '',
    setData: vi.fn(),
  };

  fireEvent.dragStart(source, { dataTransfer });
  fireEvent.dragOver(target, { dataTransfer });
  fireEvent.drop(target, { dataTransfer });

  expect(onEdit).toHaveBeenCalledWith('reorder_pages', {
    page_ids: ['b', 'c', 'a'],
  });
});

it('defaults new projects to whole spreads and retains split controls', () => {
  const onCreate = vi.fn();
  render(<Setup busy={false} onChoose={vi.fn()} onCreate={onCreate} />);
  expect(screen.getByLabelText('出力形式').value).toBe('spread');
  expect(screen.getByLabelText('候補フレーム選択').disabled).toBe(true);
  expect(screen.getByLabelText('ページ輪郭の最低信頼度').value).toBe('0.55');
  fireEvent.change(screen.getByLabelText('出力形式'), { target: { value: 'split' } });
  expect(screen.getByLabelText('候補フレーム選択').disabled).toBe(false);
  expect(screen.getByLabelText('左右別の台形補正').disabled).toBe(false);
  fireEvent.change(screen.getByLabelText('動画のローカルパス'), { target: { value: '/tmp/book.mov' } });
  fireEvent.click(screen.getByRole('button', { name: '動画を読み込む →' }));
  expect(onCreate).toHaveBeenCalledWith(
    '/tmp/book.mov',
    expect.objectContaining({ output_layout: 'split' }),
    null,
  );
});

it('labels whole pages and offers layout switching and crop correction', () => {
  const onEdit = vi.fn();
  render(<Review manifest={manifest} file={path => path} busy={false} onEdit={onEdit} />);
  expect(screen.getByText('001 · 見開き')).toBeTruthy();
  expect(screen.queryByText('左右の順番を入れ替え')).toBeNull();
  expect(screen.queryByText('左に採用中')).toBeNull();
  expect(screen.getByText((_text, node) => node.tagName === 'P'
    && node.textContent.includes('左右ページから外周を自動検出 · 信頼度 86%'))).toBeTruthy();
  expect(screen.getByText('外周を確認・調整')).toBeTruthy();
  expect(screen.getByText('外周を自動検出し直す')).toBeTruthy();
  expect(screen.getByText('検出結果 ↗')).toBeTruthy();
  fireEvent.change(screen.getByLabelText('この見開きの出力形式'), { target: { value: 'split' } });
  expect(onEdit).toHaveBeenCalledWith('output_layout', { spread_id: 's', layout: 'split' });
});

it('opens the fullscreen viewer and changes zoom without leaving Review', () => {
  render(<Review manifest={manifest} file={path => path} busy={false} onEdit={vi.fn()} />);

  fireEvent.click(screen.getByRole('button', { name: '全画面で確認・ズーム' }));
  const dialog = screen.getByRole('dialog', { name: 'ページ全画面ビューア' });
  expect(dialog).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'ズーム 200%' }));
  const image = dialog.querySelector('img[alt="s_whole 補正後"]');
  expect(image).toBeTruthy();
  expect(image.style.width).toBe('200%');

  fireEvent.click(screen.getByRole('button', { name: '元画像' }));
  expect(dialog.querySelector('img[alt="s_whole 元画像"]')).toBeTruthy();

  fireEvent.keyDown(window, { key: 'Escape' });
  expect(screen.queryByRole('dialog', { name: 'ページ全画面ビューア' })).toBeNull();
});

it('requests a high-fps rescan only for the current page', () => {
  const onEdit = vi.fn();
  render(<Review manifest={manifest} file={path => path} busy={false} onEdit={onEdit} />);

  fireEvent.change(screen.getByLabelText('s_whole 再探索範囲'), { target: { value: '2' } });
  fireEvent.change(screen.getByLabelText('s_whole 再探索fps'), { target: { value: '30' } });
  fireEvent.click(screen.getByRole('button', { name: '高fpsで再探索' }));

  expect(onEdit).toHaveBeenCalledWith('rescan_candidates', {
    page_id: 's_whole',
    radius: 2,
    fps: 30,
  });
});

it('can discard a manual crop and return to automatic detection', () => {
  const onEdit = vi.fn();
  const manual = {
    ...manifest,
    spreads: [{ ...manifest.spreads[0], roi_overrides: { 0: [[.1, .1], [.9, .1], [.9, .9], [.1, .9]] },
      whole_spread_crop: { ...manifest.spreads[0].whole_spread_crop, status: 'manual' } }],
  };
  render(<Review manifest={manual} file={path => path} busy={false} onEdit={onEdit} />);
  fireEvent.click(screen.getByText('自動検出に戻す'));
  expect(onEdit).toHaveBeenCalledWith('reset_crop', { spread_id: 's', candidate_id: 0 });
});

it('distinguishes donor pixel recovery from fallback-inclusive coverage', () => {
  expect(fingerRepairCoverageSummary({
    status: 'complete',
    donor_coverage: .7,
    coverage: 1,
  })).toBe('実画素復元率 70% · 処理済み率 100%');

  expect(fingerRepairCoverageSummary({
    status: 'complete',
    donor_coverage: 1,
    coverage: 1,
  })).toBe('実画素復元率 100%');

  expect(fingerRepairCoverageSummary({
    status: 'complete',
    coverage: .96,
  })).toBe('処理済み率 96%');

  expect(fingerRepairCoverageSummary({
    status: 'clean',
    donor_coverage: 1,
    coverage: 1,
  })).toBe('');
});

it('renders donor recovery separately from fallback-inclusive coverage', () => {
  const withFallback = {
    ...manifest,
    pages: [{
      ...manifest.pages[0],
      finger_repair: {
        status: 'complete',
        donor_coverage: .7,
        coverage: 1,
        donors: [4],
        fallback: {
          mode: 'paper',
          applied: true,
          filled_fraction: 1,
        },
      },
    }],
  };
  render(<Review manifest={withFallback} file={path => path} busy={false} onEdit={vi.fn()} />);

  expect(screen.getByText((_text, node) => node.tagName === 'SPAN'
    && node.textContent.includes('実画素復元率 70%')
    && node.textContent.includes('処理済み率 100%')
    && node.textContent.includes('紙面補完 100%'))).toBeTruthy();
});

it('summarizes optional local finger alignment metadata compactly', () => {
  expect(localAlignmentSummary({
    local_alignment: {
      component_count: 2,
      components: [
        { donor_candidate_id: 4, shift: { dx: 3, dy: 4 }, score: .95, coverage: 1 },
        { donor_candidate_id: 2, dx: -6, dy: 0, score: .91, coverage: .96 },
      ],
    },
  })).toBe('局所補正 2領域 · 最大ずれ 6px');

  expect(localAlignmentSummary({ coverage: 1 })).toBe('');
});

it('shows local alignment metadata only when present in finger repair results', () => {
  const withLocalAlignment = {
    ...manifest,
    pages: [{
      ...manifest.pages[0],
      finger_repair: {
        status: 'complete',
        coverage: 1,
        donors: [4, 2],
        target_mask: 'debug/target.png',
        local_alignment: {
          component_count: 2,
          max_shift_px: 6,
          components: [
            { donor_candidate_id: 4, dx: 2, dy: 1, score: .96, coverage: 1 },
            { donor_candidate_id: 2, dx: -6, dy: 0, score: .93, coverage: 1 },
          ],
        },
      },
    }],
  };
  render(<Review manifest={withLocalAlignment} file={path => path} busy={false} onEdit={vi.fn()} />);

  expect(screen.getByText((_text, node) => node.tagName === 'SPAN'
    && node.textContent.includes('指補修: 完了')
    && node.textContent.includes('donor #4, #2')
    && node.textContent.includes('局所補正 2領域 · 最大ずれ 6px'))).toBeTruthy();
});

it('treats an unresolved finger mask as review-required even at high coverage', () => {
  const unresolved = {
    ...manifest,
    pages: [{
      ...manifest.pages[0],
      suspect: [],
      finger_repair: {
        status: 'complete',
        coverage: .96,
        donors: [4],
        target_mask: 'debug/target.png',
        unresolved_mask: 'debug/unresolved.png',
      },
    }],
  };
  render(<Review manifest={unresolved} file={path => path} busy={false} onEdit={vi.fn()} />);

  expect(screen.getByText('1 ページ / 要確認 1')).toBeTruthy();
  expect(screen.getByText(/要確認: 未補修領域が残っています/)).toBeTruthy();
  expect(screen.getByText('未補修領域 ↗')).toBeTruthy();

  fireEvent.click(screen.getByLabelText('要確認だけ表示'));
  expect(screen.getByText('001 · 見開き')).toBeTruthy();
});

it('retains split review controls for legacy projects without the new setting', () => {
  const legacy = { ...manifest, config: { spine_ratio: .5, candidate_selection_mode: 'spread' } };
  render(<Review manifest={legacy} file={path => path} busy={false} onEdit={vi.fn()} />);
  expect(screen.getByLabelText('この見開きの出力形式').value).toBe('split');
  expect(screen.getByText('左右の順番を入れ替え')).toBeTruthy();
});


it('edits and saves book metadata from Review', () => {
  const onEdit = vi.fn();
  render(
    <Review
      manifest={{ ...manifest, book_metadata: { title: '既存タイトル', author: '作者' } }}
      file={path => path}
      busy={false}
      onEdit={onEdit}
    />,
  );

  expect(screen.getByDisplayValue('既存タイトル')).toBeTruthy();
  fireEvent.change(screen.getByLabelText('書籍タイトル'), { target: { value: '新しいタイトル' } });
  fireEvent.change(screen.getByLabelText('巻数'), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'メタデータを保存' }));

  expect(onEdit).toHaveBeenCalledWith('book_metadata', {
    metadata: { title: '新しいタイトル', author: '作者', volume: '2' },
  });
});


it('distinguishes current PDF/CBZ, legacy PDF-only, stale exports, and active export', () => {
  const current = {
    ...manifest,
    pdf: 'output/manga.pdf',
    cbz: 'output/manga.cbz',
    pdf_stale: false,
  };
  const { rerender } = render(
    <Review manifest={current} file={path => path} busy={false} onEdit={vi.fn()} />,
  );
  expect(screen.getByRole('button', { name: 'PDF / CBZを再出力' }).disabled).toBe(false);
  expect(screen.getByRole('link', { name: 'PDFを開く ↗' })).toBeTruthy();
  expect(screen.getByRole('link', { name: 'CBZを保存 ↓' })).toBeTruthy();

  rerender(
    <Review manifest={{ ...current, cbz: undefined }} file={path => path} busy={false} onEdit={vi.fn()} />,
  );
  expect(screen.getByRole('button', { name: 'PDF / CBZを出力' }).disabled).toBe(false);
  expect(screen.getByRole('link', { name: 'PDFを開く ↗' })).toBeTruthy();
  expect(screen.queryByRole('link', { name: 'CBZを保存 ↓' })).toBeNull();
  expect(screen.getByText(/PDFは出力済みです/)).toBeTruthy();

  rerender(
    <Review manifest={{ ...current, pdf_stale: true }} file={path => path} busy={false} onEdit={vi.fn()} />,
  );
  expect(screen.getByRole('button', { name: 'PDF / CBZを出力' }).disabled).toBe(false);
  expect(screen.queryByRole('link', { name: 'PDFを開く ↗' })).toBeNull();
  expect(screen.queryByRole('link', { name: 'CBZを保存 ↓' })).toBeNull();

  rerender(
    <Review manifest={current} file={path => path} busy exporting onEdit={vi.fn()} />,
  );
  const exporting = screen.getByRole('button', { name: 'PDF / CBZ生成中…' });
  expect(exporting.disabled).toBe(true);
  expect(exporting.getAttribute('aria-busy')).toBe('true');
  expect(screen.getByText(/PDF \/ CBZを生成中です/)).toBeTruthy();
});


it('shows glare metrics and generalized occlusion repair metadata', () => {
  const withGlare = {
    ...manifest,
    pages: [{
      ...manifest.pages[0],
      finger_repair: {
        status: 'complete',
        donor_coverage: 1,
        coverage: 1,
        donors: [2],
        occlusion_kinds: ['glare'],
        target_mask: 'debug/target.png',
        glare_mask: 'debug/glare.png',
      },
    }],
    spreads: [{
      ...manifest.spreads[0],
      candidates: [{
        ...manifest.spreads[0].candidates[0],
        glare_mask: 'candidate_glare.png',
        metrics: {
          ...manifest.spreads[0].candidates[0].metrics,
          glare_overlap: .034,
        },
      }],
    }],
  };

  render(<Review manifest={withGlare} file={path => path} busy={false} onEdit={vi.fn()} />);

  expect(screen.getByText((_text, node) => node.tagName === 'P'
    && node.textContent.includes('反射 3.4%'))).toBeTruthy();
  expect(screen.getByText((_text, node) => node.tagName === 'SPAN'
    && node.textContent.includes('遮蔽補修: 完了')
    && node.textContent.includes('donor #2'))).toBeTruthy();
  expect(screen.getAllByText('反射マスク ↗').length).toBeGreaterThan(0);
});


it('groups final output QA reasons in the review summary', () => {
  const pages = [
    {
      id: 'a',
      enabled: true,
      suspect: ['final_unresolved_finger', 'final_edge_crop_suspected'],
      final_quality: { reasons: ['final_unresolved_finger', 'final_edge_crop_suspected'] },
      finger_repair: { occlusion_kinds: ['finger'] },
    },
    {
      id: 'b',
      enabled: true,
      suspect: ['final_glare_residual', 'final_duplicate_suspected'],
      final_quality: { reasons: ['final_glare_residual', 'final_duplicate_suspected'] },
      finger_repair: { occlusion_kinds: ['glare'] },
    },
    {
      id: 'c',
      enabled: false,
      suspect: ['final_near_blank_white'],
      final_quality: { reasons: ['final_near_blank_white'] },
    },
  ];

  expect(qualityReviewSummary(pages)).toEqual({
    total: 2,
    categories: [
      { label: '指補修', count: 1 },
      { label: 'ページ輪郭', count: 1 },
      { label: '反射', count: 1 },
      { label: '重複疑い', count: 1 },
    ],
  });
});


it('classifies generalized occlusion failures by their recorded kind', () => {
  const pages = [
    {
      id: 'finger',
      enabled: true,
      suspect: ['occlusion_repair_incomplete'],
      finger_repair: { occlusion_kinds: ['finger'] },
    },
    {
      id: 'glare',
      enabled: true,
      suspect: ['occlusion_repair_incomplete'],
      finger_repair: { occlusion_kinds: ['glare'] },
    },
  ];

  expect(qualityReviewSummary(pages)).toEqual({
    total: 2,
    categories: [
      { label: '指補修', count: 1 },
      { label: '反射', count: 1 },
    ],
  });
});


it('renders final output QA summary and adjacent duplicate detail', () => {
  const withFinalQa = {
    ...manifest,
    pages: [{
      ...manifest.pages[0],
      suspect: ['final_glare_residual', 'final_duplicate_suspected'],
      final_quality: {
        reasons: ['final_glare_residual', 'final_duplicate_suspected'],
        metrics: { glare_fraction: .004 },
        adjacent_duplicate: {
          other_page_id: 'previous_page',
          ssim: .971,
          hash_distance: 4,
        },
      },
    }],
  };

  render(<Review manifest={withFinalQa} file={path => path} busy={false} onEdit={vi.fn()} />);

  expect(screen.getByLabelText('最終品質チェック')).toBeTruthy();
  expect(screen.getByText('要確認 1件')).toBeTruthy();
  expect(screen.getByText('反射 1')).toBeTruthy();
  expect(screen.getByText('重複疑い 1')).toBeTruthy();
  expect(screen.getByText((_text, node) => node.tagName === 'SPAN'
    && node.textContent.includes('完成画像QA:')
    && node.textContent.includes('反射が残っている')
    && node.textContent.includes('前ページとほぼ同一'))).toBeTruthy();
  expect(screen.getByText(/前ページ previous_page と類似 · SSIM 97.1%/)).toBeTruthy();
});


it('supports per-page correction overrides and before/after comparison', () => {
  const onEdit = vi.fn();
  const split = {
    ...manifest,
    config: {
      ...manifest.config,
      output_layout: 'split',
      candidate_selection_mode: 'spread',
      dewarp_mode: 'auto',
      illumination_correction: true,
      white_normalization: true,
      rotation: 0,
    },
    pages: [{
      id: 's_right',
      spread_id: 's',
      side: 'right',
      enabled: true,
      suspect: [],
      path: 'right.png',
      preview: 'right-thumb.jpg',
      source: 'right-source.png',
      candidate_id: 0,
      candidate_time: 1,
      dewarp: { mode: 'auto', status: 'applied', applied: true },
      render_settings: {
        dewarp: true,
        dewarp_mode: 'auto',
        illumination_correction: true,
        white_normalization: true,
        page_quad_mode: 'auto',
        manual_quad: null,
      },
      page_contour: {
        mode: 'auto',
        quad: [[.52, .1], [.92, .1], [.92, .9], [.52, .9]],
        confidence: .9,
        detected: true,
      },
    }],
    spreads: [{
      ...manifest.spreads[0],
      output_layout: 'split',
      selected_pages: { left: 0, right: 0 },
    }],
  };

  render(<Review manifest={split} file={path => path} busy={false} onEdit={onEdit} />);

  expect(screen.getByText('元画像')).toBeTruthy();
  expect(screen.getByText('補正後')).toBeTruthy();

  fireEvent.click(screen.getByLabelText('s_right 湾曲補正 OFF'));
  expect(onEdit).toHaveBeenCalledWith('page_settings', {
    page_id: 's_right',
    settings: { dewarp: false },
  });

  fireEvent.click(screen.getByLabelText('s_right 照明補正 OFF'));
  expect(onEdit).toHaveBeenCalledWith('page_settings', {
    page_id: 's_right',
    settings: { illumination_correction: false },
  });

  fireEvent.click(screen.getByLabelText('s_right 白背景補正 OFF'));
  expect(onEdit).toHaveBeenCalledWith('page_settings', {
    page_id: 's_right',
    settings: { white_normalization: false },
  });

  fireEvent.click(screen.getByLabelText('s_right ページ輪郭 manual'));
  expect(screen.getByText('右ページの外周を手動指定')).toBeTruthy();
  expect(screen.getByText('この外周で再レンダリング')).toBeTruthy();
});


it('can return a page contour override to auto mode', () => {
  const onEdit = vi.fn();
  const manualPage = {
    ...manifest,
    config: {
      ...manifest.config,
      output_layout: 'split',
      candidate_selection_mode: 'spread',
      rotation: 0,
    },
    pages: [{
      id: 's_left',
      spread_id: 's',
      side: 'left',
      enabled: true,
      suspect: [],
      path: 'left.png',
      source: 'left-source.png',
      candidate_id: 0,
      candidate_time: 1,
      render_settings: {
        dewarp: false,
        illumination_correction: false,
        white_normalization: false,
        page_quad_mode: 'manual',
        manual_quad: [[.08, .1], [.48, .1], [.48, .9], [.08, .9]],
      },
      page_contour: {
        mode: 'manual',
        manual: true,
        quad: [[.08, .1], [.48, .1], [.48, .9], [.08, .9]],
      },
      dewarp: { mode: 'off', status: 'disabled', applied: false },
    }],
    spreads: [{
      ...manifest.spreads[0],
      output_layout: 'split',
      selected_pages: { left: 0, right: 0 },
    }],
  };

  render(<Review manifest={manualPage} file={path => path} busy={false} onEdit={onEdit} />);
  fireEvent.click(screen.getByLabelText('s_left ページ輪郭 auto'));

  expect(onEdit).toHaveBeenCalledWith('page_settings', {
    page_id: 's_left',
    settings: { page_quad_mode: 'auto' },
  });
});


it('creates a project from multiple videos in the entered order', () => {
  const onCreate = vi.fn();
  render(<Setup busy={false} onChoose={vi.fn()} onCreate={onCreate} />);
  fireEvent.change(screen.getByLabelText('動画のローカルパス'), { target: { value: '/tmp/part-1.mov' } });
  fireEvent.click(screen.getByRole('button', { name: '＋ 動画を追加' }));
  fireEvent.change(screen.getByLabelText('動画2のローカルパス'), { target: { value: '/tmp/part-2.mov' } });
  fireEvent.click(screen.getByRole('button', { name: '動画を読み込む →' }));

  expect(onCreate).toHaveBeenCalledWith(
    ['/tmp/part-1.mov', '/tmp/part-2.mov'],
    expect.objectContaining({ output_layout: 'spread' }),
    null,
  );
});

it('uploads an external image as a new page or replacement', () => {
  const onImportExternal = vi.fn();
  render(
    <Review
      manifest={manifest}
      file={path => path}
      busy={false}
      onEdit={vi.fn()}
      onImportExternal={onImportExternal}
    />,
  );
  const image = new File(['image'], 'rescue.jpg', { type: 'image/jpeg' });

  fireEvent.change(screen.getByLabelText('外部画像をページ追加'), {
    target: { files: [image] },
  });
  expect(onImportExternal).toHaveBeenCalledWith(image);

  fireEvent.change(screen.getByLabelText('s_wholeを外部画像で差し替え'), {
    target: { files: [image] },
  });
  expect(onImportExternal).toHaveBeenCalledWith(image, 's_whole');
});

it('renders imported external pages without video-only page controls', () => {
  const external = {
    ...manifest,
    pages: [{
      id: 'external_0001',
      spread_id: null,
      side: 'external',
      enabled: true,
      suspect: [],
      path: 'pages/external_0001.png',
      source: 'external_image',
      external_name: 'phone.jpg',
    }],
    spreads: [],
  };
  render(
    <Review
      manifest={external}
      file={path => path}
      busy={false}
      onEdit={vi.fn()}
      onImportExternal={vi.fn()}
    />,
  );

  expect(screen.getByText('001 · 外部画像')).toBeTruthy();
  expect(screen.getByText('外部画像: phone.jpg')).toBeTruthy();
  expect(screen.queryByText('ページ単位の補正')).toBeNull();
});


it('creates a project from a static image folder with an expected page count', () => {
  const onCreateImages = vi.fn();
  render(
    <Setup
      busy={false}
      onChoose={vi.fn()}
      onChooseFolder={vi.fn()}
      onCreate={vi.fn()}
      onCreateImages={onCreateImages}
    />,
  );

  fireEvent.click(screen.getByRole('button', { name: '静止画フォルダ' }));
  fireEvent.change(screen.getByLabelText('静止画フォルダのローカルパス'), {
    target: { value: '/tmp/book-pages' },
  });
  fireEvent.change(screen.getByLabelText('期待ページ数'), {
    target: { value: '192' },
  });
  fireEvent.click(screen.getByRole('button', { name: '静止画を一括読み込み →' }));

  expect(onCreateImages).toHaveBeenCalledWith(
    '/tmp/book-pages',
    expect.objectContaining({ image_format: 'png' }),
    192,
  );
});

it('formats expected page count safety states', () => {
  expect(pageCountSummary({
    status: 'match',
    actual: 192,
    expected: 192,
    output_items: 97,
    difference: 0,
  })).toContain('一致: 192 / 192ページ');
  expect(pageCountSummary({
    status: 'short',
    actual: 190,
    expected: 192,
    output_items: 96,
    difference: -2,
  })).toContain('2ページ不足');
  expect(pageCountSummary({
    status: 'over',
    actual: 194,
    expected: 192,
    output_items: 98,
    difference: 2,
  })).toContain('2ページ多い');
});

it('saves expected page count from Review', () => {
  const onEdit = vi.fn();
  render(
    <Review
      manifest={{
        ...manifest,
        expected_page_count: 10,
        page_count_check: {
          status: 'short',
          expected: 10,
          actual: 8,
          output_items: 4,
          difference: -2,
        },
      }}
      file={path => path}
      busy={false}
      onEdit={onEdit}
    />,
  );

  expect(screen.getByText(/2ページ不足/)).toBeTruthy();
  fireEvent.change(screen.getByLabelText('レビュー期待ページ数'), {
    target: { value: '12' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'ページ数を保存' }));

  expect(onEdit).toHaveBeenCalledWith('expected_page_count', {
    expected_page_count: 12,
  });
});

it('offers a safe existing candidate fix for a QA-flagged page', () => {
  const onEdit = vi.fn();
  const withSuggestion = {
    ...manifest,
    pages: [{
      ...manifest.pages[0],
      candidate_id: 0,
      safe_fix_suggestions: [{
        candidate_id: 2,
        side: null,
        confidence: 'high',
        improvements: ['glare', 'selection_score'],
        score_gain: .12,
        current_risks: ['glare_overlap'],
        candidate_risks: [],
      }],
    }],
  };

  render(
    <Review
      manifest={withSuggestion}
      file={path => path}
      busy={false}
      onEdit={onEdit}
    />,
  );

  expect(safeFixSummary(withSuggestion.pages[0].safe_fix_suggestions[0]))
    .toContain('反射が少ない');
  fireEvent.click(screen.getByRole('button', { name: 'この改善候補を採用' }));
  expect(onEdit).toHaveBeenCalledWith('select_candidate', {
    spread_id: 's',
    candidate_id: 2,
  });
});

it('renders image-folder projects without video-only controls', () => {
  const imageManifest = {
    ...manifest,
    source_type: 'image_folder',
    source: '/tmp/book-pages',
    metadata: { image_count: 1 },
    pages: [{
      id: 'image_0001',
      spread_id: null,
      side: 'external',
      enabled: true,
      suspect: [],
      path: 'pages/image_0001.png',
      preview: 'pages/image_0001_thumb.jpg',
      source: 'source/images/image_0001.png',
      source_kind: 'image_folder',
      external_name: '001.jpg',
    }],
    spreads: [],
  };

  render(
    <Review
      manifest={imageManifest}
      file={path => path}
      busy={false}
      onEdit={vi.fn()}
      onImportExternal={vi.fn()}
    />,
  );

  expect(screen.getByText('フォルダ画像: 001.jpg')).toBeTruthy();
  expect(screen.queryByLabelText('追加する動画の秒数')).toBeNull();
});
