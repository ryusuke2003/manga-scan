import { render, screen } from '@testing-library/react';
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
}));

vi.mock('./useScanner.js', () => ({ default: () => scanner }));

import App from './App.jsx';

it('keeps project navigation available while a background job is running', () => {
  render(<App />);

  expect(screen.getByRole('button', { name: 'book.mov' }).disabled).toBe(false);
  expect(screen.getByRole('button', { name: /新しいスキャン/ }).disabled).toBe(true);
});
