import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import Review, { localAlignmentSummary } from './Review.jsx';
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
