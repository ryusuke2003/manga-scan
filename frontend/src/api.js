export async function request(path, { body, token, signal } = {}) {
  const response = await fetch(path, {
    signal,
    ...(body === undefined ? {} : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Manga-Token': token },
      body: JSON.stringify(body),
    }),
  });
  if (!response.ok) {
    let message = response.statusText;
    try { message = (await response.json()).error || message; } catch { /* Non-JSON HTTP error. */ }
    const error = new Error(message || '処理に失敗しました');
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export function fileUrl(project, path, revision = 0) {
  return `/files/${encodeURIComponent(project)}/${path.split('/').map(encodeURIComponent).join('/')}?v=${revision}`;
}
