import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

const scanner = vi.hoisted(() => ({
  manifest: null,
  project: null,
  server: {
    projects: [{ id: 'scan-123', source_name: 'book.mov' }],
    job: { busy: true, project: 'scan-123', error: null },
    token: 'token',
  },
  busy: true,
  error: '',
  revision: 0,
  selectProject: () => {},
  choose: () => {},
  create: () => {},
  start: () => {},
  edit: () => {},
  editCoverRoi: vi.fn(),
}));

vi.mock('./useScanner.js', () => ({ default: () => scanner }));

import App from './App.jsx';

it('keeps project navigation available while a background job is running', () => {
  render(<App />);

  expect(screen.getByRole('button', { name: 'book.mov' }).disabled).toBe(false);
  expect(screen.getByRole('button', { name: /新しいスキャン/ }).disabled).toBe(true);
});

it('moves past an automatically detected cover and keeps a crop correction action', () => {
  scanner.project = 'scan-123';
  scanner.busy = false;
  scanner.server.job.busy = false;
  scanner.manifest = {
    source: '/tmp/book.mov',
    status: 'ready',
    progress: 0,
    message: '表紙の外周を自動検出しました',
    warnings: [],
    pages: [],
    spreads: [],
    roi: null,
    metadata: { duration: 10 },
    config: { rotation: 0 },
    rotation_detection: { source: 'manual' },
    cover: {
      status: 'ready',
      roi: [[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]],
      detection: { detected: true, confidence: 0.87 },
    },
    reference: { confirmed: false, time: 1, preview: 'source/reference_preview.png' },
  };
  scanner.referenceFrame = vi.fn();
  scanner.rotation = vi.fn();

  render(<App />);

  expect(screen.getByText('03 / 見開き基準フレーム')).toBeTruthy();
  expect(screen.getByText(/表紙の外周を自動検出済み · 信頼度 87%/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '外周を修正' }));
  expect(scanner.editCoverRoi).toHaveBeenCalledOnce();
});


it('previews an automatically detected reference spread with four ready points', () => {
  scanner.project = 'scan-123';
  scanner.busy = false;
  scanner.server.job.busy = false;
  scanner.start = vi.fn();
  scanner.manifest = {
    source: '/tmp/book.mov',
    status: 'ready',
    progress: 0,
    message: '見開き外周を自動検出しました',
    warnings: [],
    pages: [],
    spreads: [],
    roi: [[0.08, 0.10], [0.92, 0.09], [0.94, 0.91], [0.07, 0.90]],
    metadata: { duration: 10, display_width: 1000, display_height: 600, fps: 30, codec: 'h264' },
    config: { rotation: 0 },
    rotation_detection: { source: 'manual' },
    cover: { status: 'skipped', roi: null },
    reference: {
      confirmed: true,
      time: 1,
      preview: 'source/reference_preview.png',
      detection: { detected: true, confidence: 0.88, source: 'auto_pages' },
    },
  };

  render(<App />);

  expect(screen.getByRole('heading', { name: '見開き外周を自動検出しました' })).toBeTruthy();
  expect(screen.getByText(/前後0.5秒の候補から手・動き・鮮明さも考慮して外周を検出しました · 外周確信度 88%/)).toBeTruthy();
  expect(screen.getByText('4 / 4 点')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'この範囲で抽出開始 →' })).toBeTruthy();
});

it('asks for confirmation when a reference boundary edge is uncertain', () => {
  scanner.project = 'scan-123';
  scanner.busy = false;
  scanner.server.job.busy = false;
  scanner.start = vi.fn();
  scanner.manifest = {
    source: '/tmp/book.mov',
    status: 'ready',
    progress: 0,
    message: '見開き外周を自動検出しました',
    warnings: [],
    pages: [],
    spreads: [],
    roi: [[0.08, 0.10], [0.92, 0.09], [0.94, 0.91], [0.07, 0.90]],
    metadata: { duration: 10, display_width: 1000, display_height: 600, fps: 30, codec: 'h264' },
    config: { rotation: 0 },
    rotation_detection: { source: 'manual' },
    cover: { status: 'skipped', roi: null },
    reference: {
      confirmed: true,
      time: 1,
      preview: 'source/reference_preview.png',
      detection: {
        detected: true,
        confidence: 0.61,
        source: 'auto_pages',
        requires_confirmation: true,
        uncertain_edges: ['right', 'bottom'],
      },
    },
  };

  render(<App />);

  expect(screen.getByRole('heading', {
    name: '外周の一部が不確かです。確認してください',
  })).toBeTruthy();
  expect(screen.getByText(/右辺・下辺の実エッジ証拠が弱いか、手で隠れています/)).toBeTruthy();
  expect(screen.getByText(/外周確信度 61%/)).toBeTruthy();
});


it('falls back to the existing four-point editor when reference detection fails', () => {
  scanner.project = 'scan-123';
  scanner.busy = false;
  scanner.server.job.busy = false;
  scanner.manifest = {
    source: '/tmp/book.mov',
    status: 'ready',
    progress: 0,
    message: '見開き外周を自動検出できませんでした',
    warnings: [],
    pages: [],
    roi: null,
    metadata: { duration: 10, display_width: 1000, display_height: 600, fps: 30, codec: 'h264' },
    config: { rotation: 0 },
    rotation_detection: { source: 'manual' },
    cover: { status: 'skipped', roi: null },
    reference: {
      confirmed: true,
      time: 1,
      preview: 'source/reference_preview.png',
      detection: { detected: false, confidence: 0.22, source: 'auto_pages' },
    },
  };

  render(<App />);

  expect(screen.getByText('見開きの外周を4点で指定')).toBeTruthy();
  expect(screen.getByText('外周を自動検出できませんでした。左上 → 右上 → 右下 → 左下 の順に4点を指定してください。')).toBeTruthy();
  expect(screen.getByText('0 / 4 点')).toBeTruthy();
});


it('shows PDF generation instead of stale 100% scan progress during export', () => {
  scanner.project = 'scan-123';
  scanner.busy = true;
  scanner.server.job = { busy: true, project: 'scan-123', action: 'export', error: null };
  scanner.manifest = {
    source: '/tmp/book.mov',
    status: 'complete',
    progress: 1,
    message: '完了 — 要確認ページを確認してください',
    warnings: [],
    pages: [
      {
        id: 'page-1',
        spread_id: 'spread-1',
        side: 'spread',
        enabled: true,
        suspect: [],
        path: 'pages/page.png',
      },
    ],
    spreads: [],
    metadata: { duration: 10 },
    config: { rotation: 0, output_layout: 'spread', spine_ratio: 0.5 },
    cover: { status: 'skipped' },
    reference: { confirmed: true },
    pdf: 'output/manga.pdf',
    pdf_stale: false,
  };

  render(<App />);

  expect(screen.getByText('PDF / CBZを生成中…')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'PDF / CBZ生成中…' }).disabled).toBe(true);
  expect(screen.queryByText('処理中…')).toBeNull();
});
