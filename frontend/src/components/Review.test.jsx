import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import Review from './Review.jsx';
import Setup from './Setup.jsx';

const manifest = {
  config: { output_layout: 'spread', candidate_selection_mode: 'per_page', spine_ratio: .5 },
  metadata: { duration: 10 },
  pages: [{ id: 's_whole', spread_id: 's', side: 'spread', enabled: true, suspect: [], path: 'whole.png' }],
  spreads: [{ id: 's', start: 1, end: 2, selected: 0, suspect: [], candidates: [{ id: 0, time: 1,
    path: 'original.png', preview: 'preview.png', hand_mask: 'mask.png',
    roi: [[0, 0], [1, 0], [1, 1], [0, 1]], metrics: { score: 1, sharpness: 100, hand_overlap: 0 } }] }],
};

it('defaults new projects to whole spreads and retains split controls', () => {
  const onCreate = vi.fn();
  render(<Setup busy={false} onChoose={vi.fn()} onCreate={onCreate} />);
  expect(screen.getByLabelText('出力形式').value).toBe('spread');
  expect(screen.getByLabelText('候補フレーム選択').disabled).toBe(true);
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
  expect(screen.getByText('切り抜き範囲を調整')).toBeTruthy();
  fireEvent.change(screen.getByLabelText('この見開きの出力形式'), { target: { value: 'split' } });
  expect(onEdit).toHaveBeenCalledWith('output_layout', { spread_id: 's', layout: 'split' });
});

it('retains split review controls for legacy projects without the new setting', () => {
  const legacy = { ...manifest, config: { spine_ratio: .5, candidate_selection_mode: 'spread' } };
  render(<Review manifest={legacy} file={path => path} busy={false} onEdit={vi.fn()} />);
  expect(screen.getByLabelText('この見開きの出力形式').value).toBe('split');
  expect(screen.getByText('左右の順番を入れ替え')).toBeTruthy();
});
