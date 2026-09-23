import React from 'react';
import { createRoot } from 'react-dom/client';

import Review from './components/Review.jsx';
import './style.css';

const width = Number(new URLSearchParams(location.search).get('width')) || 1000;
const root = document.getElementById('root');
root.style.width = `${width}px`;
root.style.maxWidth = '100%';

const spreads = Array.from({ length: 4 }, (_, index) => ({
  id: `spread_${index}`,
  start: index * 20,
  end: index * 20 + 12,
  selected: 0,
  candidates: [{
    id: 0,
    time: index * 20 + 1,
    path: 'frame.png',
    roi: [[0, 0], [1, 0], [1, 1], [0, 1]],
    metrics: { score: 1, sharpness: 100, hand_overlap: 0 },
  }],
}));
const pages = spreads.map((spread, index) => ({
  id: `right_${index}`,
  spread_id: spread.id,
  side: 'right',
  enabled: true,
  suspect: ['page_contour_low_confidence'],
  path: 'page.png',
  preview: 'page.png',
  source: 'frame.png',
  candidate_id: 0,
  candidate_time: spread.start + 1,
  dewarp: { status: 'applied', mode: 'auto', strength: 0.1, confidence: 0.8 },
  page_contour: { mode: 'auto' },
}));
const manifest = {
  source_type: 'video',
  config: { output_layout: 'split', rotation: 0, illumination_correction: true },
  metadata: { duration: 100 },
  pages,
  spreads,
  pdf_stale: true,
};

createRoot(root).render(<Review manifest={manifest} file={path => path} busy={false} onEdit={() => {}} />);

setTimeout(() => {
  const cards = [...document.querySelectorAll('.page-card')];
  const failures = cards.flatMap((card, index) => {
    const descendants = [...card.querySelectorAll('button, label, select, input, a, img')];
    const cardBounds = card.getBoundingClientRect();
    const escaping = descendants.filter(element => {
      const bounds = element.getBoundingClientRect();
      return bounds.width > 1 && (bounds.right > cardBounds.right + 1 || bounds.left < cardBounds.left - 1);
    });
    return card.scrollWidth > card.clientWidth + 1 || escaping.length
      ? [{ index, cardWidth: card.clientWidth, scrollWidth: card.scrollWidth,
        escaping: escaping.map(element => element.outerHTML.slice(0, 100)) }]
      : [];
  });
  const report = { width, cardCount: cards.length, failures };
  const output = document.createElement('pre');
  output.id = 'layout-report';
  output.textContent = JSON.stringify(report);
  document.body.append(output);
}, 600);
