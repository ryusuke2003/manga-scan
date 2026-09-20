import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Preserve Host and Origin together so Flask's existing same-origin checks apply
// to POSTs through the development proxy. Never rewrite an untrusted Origin.
const proxy = {
  '/api': { target: 'http://127.0.0.1:8765', changeOrigin: false },
  '/files': { target: 'http://127.0.0.1:8765', changeOrigin: false },
};

export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? '/static/' : '/',
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  build: {
    outDir: fileURLToPath(new URL('../src/manga_scan/static', import.meta.url)),
    emptyOutDir: true,
  },
  test: { environment: 'jsdom', setupFiles: ['./src/test-setup.js'] },
}));
