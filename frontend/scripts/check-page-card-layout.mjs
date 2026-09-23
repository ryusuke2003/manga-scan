import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';

const chrome = [
  process.env.CHROME_BIN,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].find(path => path && existsSync(path));

if (!chrome) throw new Error('Chrome is required for the page-card layout test');

function dumpDom(url, viewport, profile) {
  return new Promise((resolve, reject) => {
    const child = spawn(chrome, [
      '--headless=new',
      '--disable-gpu',
      '--no-first-run',
      '--no-default-browser-check',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--virtual-time-budget=4000',
      `--window-size=${viewport},900`,
      `--user-data-dir=${profile}`,
      '--dump-dom',
      url,
    ]);
    let output = '';
    let errors = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), 20000);
    child.stdout.on('data', data => { output += data; });
    child.stderr.on('data', data => { errors += data; });
    child.on('error', reject);
    child.on('close', code => {
      clearTimeout(timer);
      if (code) reject(new Error(`Chrome exited ${code}: ${errors.slice(-1000)}`));
      else resolve(output);
    });
  });
}

const server = await createServer({ server: { host: '127.0.0.1', port: 0, strictPort: false } });
const profile = mkdtempSync(join(tmpdir(), 'manga-scan-layout-'));
try {
  await server.listen();
  const port = server.httpServer.address().port;
  for (const width of [390, 760, 1000, 1440]) {
    const url = `http://127.0.0.1:${port}/layout-fixture.html?width=${width}`;
    const html = await dumpDom(url, width, profile);
    const match = html.match(/<pre id="layout-report">([^<]+)<\/pre>/);
    if (!match) throw new Error(`No layout report at ${width}px`);
    const report = JSON.parse(match[1].replaceAll('&quot;', '"'));
    if (report.cardCount !== 4 || report.failures.length) {
      throw new Error(`${width}px: ${JSON.stringify(report)}`);
    }
    console.log(`${width}px: ${report.cardCount} page cards fit`);
  }
} finally {
  await server.close();
  rmSync(profile, { recursive: true, force: true });
}
