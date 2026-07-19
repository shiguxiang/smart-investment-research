const BASE = '/api/v1';

export async function sendMessage(query, subject, files, sessionId) {
  const form = new FormData();
  form.append('query', query);
  form.append('subject', subject || '');
  form.append('session_id', sessionId || '');

  if (files && files.length > 0) {
    for (const f of files) form.append('files', f);
  }

  const res = await fetch(`${BASE}/chat`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`请求失败: ${res.status}`);
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

export async function uploadFiles(files, subject, chapter) {
  const form = new FormData();
  form.append('subject', subject || '');
  form.append('chapter', chapter || '');
  for (const f of files) form.append('files', f);

  const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`上传失败: ${res.status}`);
  return res.json();
}
