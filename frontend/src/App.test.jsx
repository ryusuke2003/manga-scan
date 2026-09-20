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
