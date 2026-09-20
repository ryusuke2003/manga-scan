import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { fileUrl, request } from './api.js';
import FrameSelector, { clampTime } from './components/FrameSelector.jsx';
import { normalizedPoint } from './components/RoiSelector.jsx';
import Setup, {
  applyCorrectionPreset,
  buildInitialConfig,
  correctionPresetForConfig,
} from './components/Setup.jsx';
import { rotateNormalizedRoi } from './rotation.js';
import { detectMissingPageCandidates, timelinePercent } from './timeline.js';
import {
  didJobFinish,
  isStalePoll,
  projectDeleteErrorMessage,
  removeProjectFromServer,
  shouldReportPollError,
} from './useScanner.js';

describe('frontend helpers', () => {
  it('builds encoded local file URLs', () => {
    expect(fileUrl('scan 01', 'pages/right page.png', 3))
      .toBe('/files/scan%2001/pages/right%20page.png?v=3');
  });

  it('clamps setup timestamps to the video duration', () => {
    expect(clampTime(-2, 10)).toBe(0);
    expect(clampTime(4.25, 10)).toBe(4.25);
    expect(clampTime(12, 10)).toBeCloseTo(9.999);
  });

  it('requires the edited frame time to be previewed before confirmation', () => {
    const onPreview = vi.fn();
    const onConfirm = vi.fn();
    const props = {
      step: '04 / 見開き基準フレーム',
      title: '基準フレーム',
      description: 'desc',
      imageUrl: '/frame-3.png',
      time: 3,
      duration: 20,
      busy: false,
      confirmLabel: 'このフレームを基準にする →',
      onPreview,
      onConfirm,
    };
    const { rerender } = render(React.createElement(FrameSelector, props));

    const input = screen.getByLabelText('動画の秒数');
    const confirm = screen.getByRole('button', { name: 'このフレームを基準にする →' });
    expect(confirm.disabled).toBe(true);

    fireEvent.load(screen.getByAltText('選択中の動画フレーム'));
    expect(confirm.disabled).toBe(false);

    fireEvent.change(input, { target: { value: '10' } });
    expect(confirm.disabled).toBe(true);
    expect(screen.getByText(/プレビュー更新.*確認してから確定/).textContent)
      .toContain('確認してから確定');

    fireEvent.click(screen.getByRole('button', { name: 'プレビュー更新' }));
    expect(onPreview).toHaveBeenCalledWith(10);
    expect(confirm.disabled).toBe(true);
    expect(onConfirm).not.toHaveBeenCalled();

    rerender(React.createElement(FrameSelector, {
      ...props,
      imageUrl: '/frame-10.png',
      time: 10,
    }));

    const refreshedConfirm = screen.getByRole('button', { name: 'このフレームを基準にする →' });
    expect(refreshedConfirm.disabled).toBe(true);
    fireEvent.load(screen.getByAltText('選択中の動画フレーム'));
    expect(refreshedConfirm.disabled).toBe(false);
    fireEvent.click(refreshedConfirm);
    expect(onConfirm).toHaveBeenCalledWith(10);
  });

  it('previews a suggested reference frame when its thumbnail is selected', () => {
    const onPreview = vi.fn();
    render(React.createElement(FrameSelector, {
      step: '04 / 見開き基準フレーム',
      title: '基準フレーム',
      description: 'desc',
      imageUrl: '/frame-1.png',
      time: 1,
      duration: 20,
      busy: false,
      confirmLabel: 'このフレームを基準にする →',
      onPreview,
      onConfirm: vi.fn(),
      candidates: [
        {
          time: 4.2,
          confidence: 0.84,
          preview: 'source/reference_candidates/candidate_01.jpg',
          imageUrl: '/candidate-1.jpg',
        },
      ],
    }));

    expect(screen.getByText('候補フレーム')).toBeTruthy();
    expect(screen.getByText(/見開き 84%/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /見開き候補 4.2秒/ }));
    expect(onPreview).toHaveBeenCalledWith(4.2);
  });

  it('does not treat an out-of-range timestamp as the current preview', () => {
    render(React.createElement(FrameSelector, {
      step: '02 / 表紙フレーム（任意）',
      title: '表紙フレーム',
      description: 'desc',
      imageUrl: '/frame-0.png',
      time: 0,
      duration: 20,
      busy: false,
      confirmLabel: 'このフレームを表紙にする →',
      onPreview: vi.fn(),
      onConfirm: vi.fn(),
    }));

    fireEvent.load(screen.getByAltText('選択中の動画フレーム'));
    const input = screen.getByLabelText('動画の秒数');
    const confirm = screen.getByRole('button', { name: 'このフレームを表紙にする →' });
    expect(confirm.disabled).toBe(false);

    fireEvent.change(input, { target: { value: '-1' } });
    expect(confirm.disabled).toBe(true);
  });

  it('normalizes and clamps ROI pointer coordinates', () => {
    const bounds = { left: 100, top: 50, width: 400, height: 200 };
    expect(normalizedPoint(300, 150, bounds)).toEqual([0.5, 0.5]);
    expect(normalizedPoint(0, 500, bounds)).toEqual([0, 1]);
  });

  it('maps saved raw ROIs into rotated preview coordinates', () => {
    const roi = [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]];
    const rotated = rotateNormalizedRoi(roi, 90);
    expect(rotated[0][0]).toBeCloseTo(0.1);
    expect(rotated[0][1]).toBeCloseTo(0.1);
    expect(rotated[2][0]).toBeCloseTo(0.8);
    expect(rotated[2][1]).toBeCloseTo(0.8);
  });

  it('detects long timeline gaps as missing-page candidates', () => {
    const spreads = [2, 4, 6, 10, 12].map((start, index) => ({
      id: `spread_${index + 1}`,
      start,
      end: start + 0.6,
      selected: 0,
      candidates: [{ id: 0, time: start + 0.3 }],
    }));
    const missing = detectMissingPageCandidates(spreads);
    expect(missing).toHaveLength(1);
    expect(missing[0].time).toBeCloseTo(8);
    expect(missing[0].gapRatio).toBeCloseTo(2);

    spreads.splice(3, 0, {
      id: 'manual',
      start: 7.9,
      end: 8.1,
      selected: 0,
      candidates: [{ id: 0, time: 8 }],
      extra_suspect: ['manual_frame'],
    });
    expect(detectMissingPageCandidates(spreads)).toEqual([]);
  });

  it('uses stable interval starts instead of selected-frame timing for gap detection', () => {
    const regular = [
      { id: 'a', start: 2, end: 3.9, selected: 0, candidates: [{ id: 0, time: 3.8 }] },
      { id: 'b', start: 4, end: 5, selected: 0, candidates: [{ id: 0, time: 4.1 }] },
      { id: 'c', start: 6, end: 7.9, selected: 0, candidates: [{ id: 0, time: 7.8 }] },
      { id: 'd', start: 8, end: 9, selected: 0, candidates: [{ id: 0, time: 8.1 }] },
    ];
    expect(detectMissingPageCandidates(regular)).toEqual([]);

    const withGap = regular.map(spread => ({ ...spread }));
    withGap[2] = { ...withGap[2], start: 8, end: 9.9, candidates: [{ id: 0, time: 9.8 }] };
    withGap[3] = { ...withGap[3], start: 10, end: 11, candidates: [{ id: 0, time: 10.1 }] };
    const missing = detectMissingPageCandidates(withGap);
    expect(missing).toHaveLength(1);
    expect(missing[0].time).toBeCloseTo(6);
  });

  it('clamps timeline positions to the video bounds', () => {
    expect(timelinePercent(-1, 20)).toBe(0);
    expect(timelinePercent(5, 20)).toBe(25);
    expect(timelinePercent(30, 20)).toBe(100);
  });

  it('refreshes generated assets only when a background job finishes', () => {
    expect(didJobFinish(false, { busy: false })).toBe(false);
    expect(didJobFinish(false, { busy: true })).toBe(false);
    expect(didJobFinish(true, { busy: true })).toBe(false);
    expect(didJobFinish(true, { busy: false })).toBe(true);
  });

  it('ignores stale polling errors while project selection changes', () => {
    const error = new Error('404');
    expect(shouldReportPollError(error, false, false)).toBe(true);
    expect(shouldReportPollError(error, false, true)).toBe(false);
    expect(shouldReportPollError(error, true, false)).toBe(false);
    expect(shouldReportPollError(error, false, false, true)).toBe(false);
    expect(Object.assign(new Error('aborted'), { name: 'AbortError' })).toMatchObject({ name: 'AbortError' });
    expect(shouldReportPollError(Object.assign(new Error('aborted'), { name: 'AbortError' }), false, false)).toBe(false);
    expect(isStalePoll('scan-old', null, 3, 3)).toBe(true);
    expect(isStalePoll('scan-current', 'scan-current', 3, 4)).toBe(true);
    expect(isStalePoll('scan-current', 'scan-current', 4, 4)).toBe(false);
  });

  it('removes a deleted project from sidebar state immediately', () => {
    const state = {
      projects: [{ id: 'scan-a' }, { id: 'scan-b' }],
      job: { busy: false },
      token: 'token',
    };
    expect(removeProjectFromServer(state, 'scan-a').projects).toEqual([{ id: 'scan-b' }]);
    expect(state.projects).toHaveLength(2);
  });

  it('shows restart guidance when an old backend has no delete route', () => {
    expect(projectDeleteErrorMessage(Object.assign(new Error('NOT FOUND'), { status: 404 })))
      .toContain('再起動');
    expect(projectDeleteErrorMessage(Object.assign(new Error('BUSY'), { status: 409 })))
      .toBe('BUSY');
  });

  it('preserves HTTP status codes on API errors', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'NOT FOUND',
      json: vi.fn().mockRejectedValue(new Error('not json')),
    });
    await expect(request('/missing')).rejects.toMatchObject({
      message: 'NOT FOUND',
      status: 404,
    });
    fetchMock.mockRestore();
  });

  it('initializes correction controls from server defaults', () => {
    const config = buildInitialConfig({ perspective_mode: 'per_page', page_contour_min_confidence: 0.7, illumination_correction: true, illumination_strength: 0.45, rotation: 90 });
    expect(config.perspective_mode).toBe('per_page');
    expect(config.rotation).toBe(90);
    expect(config.page_contour_min_confidence).toBe(0.7);
    expect(config.illumination_strength).toBe(0.45);
  });

  it('applies correction presets and detects custom overrides', () => {
    const base = buildInitialConfig();
    expect(correctionPresetForConfig(base)).toBe('scan');
    const standard = applyCorrectionPreset(base, 'standard');
    expect(standard.perspective_mode).toBe('per_page');
    expect(standard.dewarp_mode).toBe('auto');
    expect(standard.illumination_correction).toBe(true);
    expect(standard.white_normalization).toBe(false);
    expect(standard.page_background_fill).toBe('preserve');
    expect(correctionPresetForConfig(standard)).toBe('standard');
    expect(correctionPresetForConfig({ ...standard, illumination_strength: 0.4 })).toBe('custom');
    const scan = applyCorrectionPreset(standard, 'scan');
    expect(scan.page_background_fill).toBe('paper');
    expect(correctionPresetForConfig(scan)).toBe('scan');
  });

  it('uses the scan-style setup defaults', () => {
    const config = buildInitialConfig();
    expect(config.finger_repair).toBe(true);
    expect(config.finger_repair_fallback).toBe('paper');
    expect(config.page_background_fill).toBe('paper');
    expect(config.grayscale).toBe(false);
    expect(config.candidate_selection_mode).toBe('spread');
    expect(config.perspective_mode).toBe('spread');
    expect(correctionPresetForConfig(config)).toBe('scan');
  });

  it('submits finger repair disabled after toggling the default off', () => {
    const onCreate = vi.fn();
    render(React.createElement(Setup, {
      busy: false,
      defaults: buildInitialConfig(),
      onChoose: vi.fn(),
      onCreate,
    }));
    fireEvent.change(screen.getByLabelText('動画のローカルパス'), {
      target: { value: '/tmp/book.mp4' },
    });
    fireEvent.click(screen.getByLabelText(/別フレームから指を補修/));
    fireEvent.click(screen.getByRole('button', { name: '動画を読み込む →' }));
    expect(onCreate).toHaveBeenCalledWith(
      '/tmp/book.mp4',
      expect.objectContaining({ finger_repair: false }),
    );
  });

  it('submits an explicit white fallback only when selected', () => {
    const onCreate = vi.fn();
    render(React.createElement(Setup, {
      busy: false,
      defaults: buildInitialConfig(),
      onChoose: vi.fn(),
      onCreate,
    }));
    fireEvent.change(screen.getByLabelText('動画のローカルパス'), {
      target: { value: '/tmp/book.mp4' },
    });
    expect(screen.getByLabelText('補修できない指').value).toBe('paper');
    fireEvent.change(screen.getByLabelText('補修できない指'), {
      target: { value: 'white' },
    });
    fireEvent.click(screen.getByRole('button', { name: '動画を読み込む →' }));
    expect(onCreate).toHaveBeenCalledWith(
      '/tmp/book.mp4',
      expect.objectContaining({ finger_repair_fallback: 'white' }),
    );
  });

  it('defaults setup rotation to automatic detection and allows manual override', () => {
    const initial = buildInitialConfig();
    expect(initial.auto_rotation).toBe(true);

    const onCreate = vi.fn();
    render(React.createElement(Setup, { busy: false, defaults: initial, onChoose: vi.fn(), onCreate }));
    fireEvent.change(screen.getByLabelText('動画のローカルパス'), { target: { value: '/tmp/book.mp4' } });
    expect(screen.getByLabelText('画像の向き').value).toBe('auto');
    fireEvent.change(screen.getByLabelText('画像の向き'), { target: { value: '270' } });
    expect(screen.getByLabelText('画像の向き').value).toBe('270');
    fireEvent.change(screen.getByLabelText('画像の向き'), { target: { value: 'auto' } });
    expect(screen.getByLabelText('画像の向き').value).toBe('auto');
    fireEvent.click(screen.getByRole('button', { name: '動画を読み込む →' }));
    expect(onCreate).toHaveBeenCalledWith('/tmp/book.mp4', expect.objectContaining({
      auto_rotation: true,
      rotation: 0,
    }));
  });

  it('shows detected rotation in frame preview and allows manual override', () => {
    const onRotation = vi.fn();
    render(React.createElement(FrameSelector, {
      step: '04 / 見開き基準フレーム',
      title: '基準フレーム',
      description: 'desc',
      imageUrl: '/rotated-preview.png',
      time: 3,
      duration: 20,
      busy: false,
      confirmLabel: 'このフレームを基準にする →',
      onPreview: vi.fn(),
      onConfirm: vi.fn(),
      rotation: 270,
      rotationDetection: { source: 'page_geometry', confidence: 0.82 },
      onRotation,
    }));

    expect(screen.getByText(/自動判定: 270°/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText('プレビューの向き'), { target: { value: '90' } });
    expect(onRotation).toHaveBeenCalledWith(90);
  });

  it('submits the selected correction preset from the setup form', () => {
    const onCreate = vi.fn();
    render(React.createElement(Setup, { busy: false, defaults: buildInitialConfig(), onChoose: vi.fn(), onCreate }));
    fireEvent.change(screen.getByLabelText('動画のローカルパス'), { target: { value: '/tmp/book.mp4' } });
    fireEvent.click(screen.getByRole('button', { name: /^標準補正 \/ おすすめ/ }));
    fireEvent.click(screen.getByRole('button', { name: '動画を読み込む →' }));
    expect(onCreate).toHaveBeenCalledWith('/tmp/book.mp4', expect.objectContaining({
      refine_quad: true,
      perspective_mode: 'per_page',
      split_mode: 'auto',
      dewarp_mode: 'auto',
      illumination_correction: true,
      white_normalization: false,
    }));
  });
});
