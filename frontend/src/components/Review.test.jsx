import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import Review, { fingerRepairCoverageSummary, localAlignmentSummary, qualityReviewSummary } from './Review.jsx';
import Setup from './Setup.jsx';

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
  expect(onCreate).toHaveBeenCalledWith('/tmp/book.mov', expect.objectContaining({ output_layout: 'split' }));
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
