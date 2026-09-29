import { ApiError, MAX_BYTES, SUITE, SUITE_SHA256, UUID, canonical, sha256, summarize, validateReceipt } from './protocol.js';
import { BROWSER_PROTOCOL } from './browser-protocol.js';
import { BOARD_HTML, BOARD_CSS, BOARD_JS } from './board.js';

const API = '/api/benchmarks/v1';
const HEADERS = { 'cache-control': 'no-store', 'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer' };
const json = (value, status = 200, extra = {}) => new Response(JSON.stringify(value), {
  status, headers: { ...HEADERS, 'content-type': 'application/json; charset=utf-8', ...extra },
});
const fail = (status, code, message) => { throw new ApiError(status, code, message); };
function bearer(request) {
  const match = /^Bearer ([A-Za-z0-9_-]{32,256})$/.exec(request.headers.get('authorization') || '');
  if (!match) fail(401, 'authentication_required', 'A contributor bearer token is required.');
  return match[1];
}
async function contributor(request, env) {
  const digest = await sha256(bearer(request));
  const row = await env.BENCHMARKS_DB.prepare('SELECT id, display_name FROM publishers WHERE token_sha256 = ? AND revoked_at IS NULL').bind(digest).first();
  if (!row) fail(401, 'invalid_token', 'The contributor token is invalid or revoked.');
  return row;
}
async function admin(request, env) {
  if (typeof env.ADMIN_TOKEN !== 'string' || env.ADMIN_TOKEN.length < 32) fail(503, 'admin_unconfigured', 'Administration is not configured.');
  const expected = await sha256(env.ADMIN_TOKEN), actual = await sha256(bearer(request));
  let mismatch = 0;
  for (let index = 0; index < expected.length; index++) mismatch |= expected.charCodeAt(index) ^ actual.charCodeAt(index);
  if (mismatch) fail(403, 'admin_required', 'An administrator token is required.');
}
async function readJson(request, max = MAX_BYTES) {
  if (!/^application\/json(?:\s*;|$)/i.test(request.headers.get('content-type') || '')) fail(415, 'content_type', 'Send application/json.');
  const length = request.headers.get('content-length');
  if (length && (!/^\d+$/.test(length) || Number(length) > max)) fail(413, 'receipt_too_large', 'Request exceeds its size limit.');
  if (!request.body) fail(400, 'invalid_json', 'A JSON body is required.');
  const reader = request.body.getReader(), chunks = [];
  let size = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    size += part.value.byteLength;
    if (size > max) { await reader.cancel(); fail(413, 'receipt_too_large', 'Request exceeds its size limit.'); }
    chunks.push(part.value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { fail(400, 'invalid_json', 'Request contains invalid JSON.'); }
}
async function requirePublicCheckpoints(receipt) {
  const repos = [...new Set([receipt.model.repo_id, receipt.model.drafter_repo_id].filter(Boolean))];
  for (const repo of repos) {
    let response;
    try {
      const target = 'https://huggingface.co/api/models/' + repo.split('/').map(encodeURIComponent).join('/');
      response = await fetch(target, { redirect: 'manual', signal: AbortSignal.timeout(8000), headers: { accept: 'application/json' } });
    } catch { fail(503, 'checkpoint_check_unavailable', 'Public checkpoint lookup is temporarily unavailable.'); }
    if (response.status !== 200) {
      if (response.status === 429 || response.status >= 500) fail(503, 'checkpoint_check_unavailable', 'Public checkpoint lookup is temporarily unavailable.');
      fail(400, 'public_checkpoint_required', 'Only publicly accessible Hugging Face checkpoints may be published.');
    }
    let data;
    try { data = await readJson(response, 256 * 1024); }
    catch { fail(503, 'checkpoint_check_unavailable', 'Public checkpoint lookup is temporarily unavailable.'); }
    if (data?.private !== false) fail(400, 'public_checkpoint_required', 'Only publicly accessible Hugging Face checkpoints may be published.');
  }
}
function receiptStatus(row, url) {
  return { id: row.id, status: row.status, content_sha256: row.content_sha256,
    protocol_complete: !!row.protocol_complete, received_at: row.received_at,
    results_url: row.status === 'approved' ? `${url.origin}/benchmarks?result=${row.id}` : null };
}
function publicResult(row) {
  return { id: row.id, display_name: row.display_name, received_at: row.received_at, approved_at: row.approved_at,
    receipt: JSON.parse(row.receipt_json), summary: JSON.parse(row.summary_json),
    protocol_complete: !!row.protocol_complete, self_reported: true };
}
const RESULT_SELECT = `SELECT s.id, s.status, s.received_at, s.approved_at, s.receipt_json, s.summary_json, s.protocol_complete, p.display_name
  FROM submissions s JOIN publishers p ON p.id = s.publisher_id`;
async function submit(request, env, url) {
  const publisher = await contributor(request, env);
  const receipt = validateReceipt(await readJson(request));
  const encoded = canonical(receipt), digest = await sha256(encoded);
  const existing = await env.BENCHMARKS_DB.prepare('SELECT id, status, content_sha256, protocol_complete, received_at FROM submissions WHERE publisher_id = ? AND run_id = ?').bind(publisher.id, receipt.run_id).first();
  if (existing) {
    if (existing.content_sha256 !== digest) fail(409, 'run_conflict', 'This run ID already has different content.');
    return json(receiptStatus(existing, url));
  }
  await requirePublicCheckpoints(receipt);
  const computed = summarize(receipt), id = crypto.randomUUID(), now = new Date().toISOString();
  const label = [...new Set(receipt.hardware.gpus.map(gpu => gpu.name))].join(' + ');
  try {
    await env.BENCHMARKS_DB.prepare(`INSERT INTO submissions
      (id, publisher_id, run_id, content_sha256, received_at, received_day, status, model_repo, backend, gpu_label, protocol_complete, receipt_json, summary_json)
      VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)`).bind(id, publisher.id, receipt.run_id, digest,
      now, now.slice(0, 10), receipt.model.repo_id, receipt.runtime.backend, label, computed.protocol_complete ? 1 : 0,
      encoded, JSON.stringify(computed.summary)).run();
  } catch (error) {
    // The database trigger enforces the quota inside the insert transaction.
    // Concurrent retries may race the first lookup. Resolve them without charging another upload.
    const raced = await env.BENCHMARKS_DB.prepare('SELECT id, status, content_sha256, protocol_complete, received_at FROM submissions WHERE publisher_id = ? AND run_id = ?').bind(publisher.id, receipt.run_id).first();
    if (raced) {
      if (raced.content_sha256 !== digest) fail(409, 'run_conflict', 'This run ID already has different content.');
      return json(receiptStatus(raced, url));
    }
    if (String(error.message).includes('daily_quota_exceeded')) fail(429, 'daily_quota_exceeded', 'The daily upload quota has been reached.');
    if (String(error.message).includes('publisher_revoked')) fail(401, 'invalid_token', 'The contributor token is revoked.');
    throw error;
  }
  return json(receiptStatus({ id, status: 'pending', content_sha256: digest, protocol_complete: computed.protocol_complete, received_at: now }, url), 201);
}
async function listResults(env, url) {
  const allowed = new Set(['model', 'backend', 'gpu', 'limit', 'cursor']);
  for (const key of url.searchParams.keys()) if (!allowed.has(key) || url.searchParams.getAll(key).length !== 1) fail(400, 'invalid_filter', 'Invalid benchmark filter.');
  const where = ["s.status = 'approved'"], values = [];
  for (const [key, column] of [['model', 's.model_repo'], ['backend', 's.backend'], ['gpu', 's.gpu_label']]) {
    if (!url.searchParams.has(key)) continue;
    const value = url.searchParams.get(key);
    if (!value || value.length > 256 || /[\u0000-\u001f]/.test(value)) fail(400, 'invalid_filter', 'Invalid benchmark filter.');
    if (key === 'backend' && !['mlx', 'cuda'].includes(value)) fail(400, 'invalid_filter', 'Unknown backend.');
    where.push(`${column} = ?`); values.push(value);
  }
  const limitText = url.searchParams.get('limit') || '30';
  if (!/^\d+$/.test(limitText) || Number(limitText) < 1 || Number(limitText) > 50) fail(400, 'invalid_filter', 'Page size must be between 1 and 50.');
  const limit = Number(limitText), cursor = url.searchParams.get('cursor');
  if (cursor) {
    const match = /^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z)\|([0-9a-f-]{36})$/.exec(cursor);
    if (!match || !UUID.test(match[2])) fail(400, 'invalid_filter', 'Invalid page cursor.');
    where.push('(s.received_at < ? OR (s.received_at = ? AND s.id < ?))'); values.push(match[1], match[1], match[2]);
  }
  values.push(limit + 1);
  const result = await env.BENCHMARKS_DB.prepare(`${RESULT_SELECT} WHERE ${where.join(' AND ')} ORDER BY s.received_at DESC, s.id DESC LIMIT ?`).bind(...values).all();
  const rows = result.results || [], visible = rows.slice(0, limit), last = visible.at(-1);
  return json({ results: visible.map(publicResult), next_cursor: rows.length > limit ? `${last.received_at}|${last.id}` : null });
}
async function routes(request, env, url) {
  const path = url.pathname.replace(/\/$/, ''), method = request.method;
  if (path === `${API}/suites` && method === 'GET') return json({ suites: [{ ...SUITE, sha256: SUITE_SHA256 }] });
  if (!env.BENCHMARKS_DB) fail(503, 'storage_unconfigured', 'Benchmark storage is not configured.');
  if (path === `${API}/me` && method === 'GET') {
    const publisher = await contributor(request, env);
    return json({ publisher_id: publisher.id, display_name: publisher.display_name });
  }
  if (path === `${API}/submissions` && method === 'POST') return submit(request, env, url);
  let match = new RegExp(`^${API}/submissions/([0-9a-f-]{36})$`).exec(path);
  if (match && UUID.test(match[1]) && ['GET', 'DELETE'].includes(method)) {
    const publisher = await contributor(request, env);
    const row = await env.BENCHMARKS_DB.prepare('SELECT id, status, content_sha256, protocol_complete, received_at FROM submissions WHERE id = ? AND publisher_id = ?').bind(match[1], publisher.id).first();
    if (!row) fail(404, 'not_found', 'Submission not found.');
    if (method === 'GET') return json(receiptStatus(row, url));
    await env.BENCHMARKS_DB.prepare("UPDATE submissions SET status = 'withdrawn', receipt_json = NULL, summary_json = NULL WHERE id = ? AND publisher_id = ?").bind(match[1], publisher.id).run();
    return json({ id: match[1], status: 'withdrawn' });
  }
  if (path === `${API}/results` && method === 'GET') return listResults(env, url);
  match = new RegExp(`^${API}/results/([0-9a-f-]{36})$`).exec(path);
  if (match && UUID.test(match[1]) && method === 'GET') {
    const row = await env.BENCHMARKS_DB.prepare(`${RESULT_SELECT} WHERE s.id = ? AND s.status = 'approved'`).bind(match[1]).first();
    if (!row) fail(404, 'not_found', 'Result not found.');
    return json(publicResult(row));
  }
  if (path === `${API}/admin/publishers` && method === 'POST') {
    await admin(request, env);
    const body = await readJson(request, 1024);
    if (!body || typeof body !== 'object' || Array.isArray(body) ||
      Object.keys(body).some(key => !['display_name', 'daily_quota'].includes(key)) ||
      typeof body.display_name !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_. -]{0,39}$/.test(body.display_name))
      fail(400, 'invalid_publisher', 'Choose a public display name of 1 to 40 letters, numbers, spaces, dots, underscores or hyphens.');
    const quota = body.daily_quota ?? 20;
    if (!Number.isSafeInteger(quota) || quota < 1 || quota > 1000) fail(400, 'invalid_publisher', 'Invalid daily quota.');
    const token = 'tfp_' + Array.from(crypto.getRandomValues(new Uint8Array(32)), byte => byte.toString(16).padStart(2, '0')).join('');
    const id = crypto.randomUUID();
    await env.BENCHMARKS_DB.prepare('INSERT INTO publishers (id, display_name, token_sha256, created_at, daily_quota) VALUES (?, ?, ?, ?, ?)').bind(id, body.display_name, await sha256(token), new Date().toISOString(), quota).run();
    return json({ publisher_id: id, display_name: body.display_name, token, daily_quota: quota }, 201);
  }
  if (path === `${API}/admin/submissions` && method === 'GET') {
    await admin(request, env);
    const result = await env.BENCHMARKS_DB.prepare(`${RESULT_SELECT} WHERE s.status = 'pending' ORDER BY s.received_at ASC LIMIT 50`).all();
    return json({ submissions: (result.results || []).map(publicResult) });
  }
  match = new RegExp(`^${API}/admin/submissions/([0-9a-f-]{36})$`).exec(path);
  if (match && UUID.test(match[1]) && method === 'GET') {
    await admin(request, env);
    const row = await env.BENCHMARKS_DB.prepare(`${RESULT_SELECT} WHERE s.id = ? AND s.receipt_json IS NOT NULL`).bind(match[1]).first();
    if (!row) fail(404, 'not_found', 'Submission not found.');
    return json({ ...publicResult(row), status: row.status });
  }
  match = new RegExp(`^${API}/admin/publishers/([0-9a-f-]{36})$`).exec(path);
  if (match && UUID.test(match[1]) && method === 'DELETE') {
    await admin(request, env);
    const result = await env.BENCHMARKS_DB.prepare('UPDATE publishers SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL').bind(new Date().toISOString(), match[1]).run();
    if (!result.meta?.changes) fail(404, 'not_found', 'Active publisher not found.');
    return json({ publisher_id: match[1], revoked: true });
  }
  match = new RegExp(`^${API}/admin/submissions/([0-9a-f-]{36})/approve$`).exec(path);
  if (match && UUID.test(match[1]) && method === 'POST') {
    await admin(request, env);
    const result = await env.BENCHMARKS_DB.prepare("UPDATE submissions SET status = 'approved', approved_at = ? WHERE id = ? AND status = 'pending' AND receipt_json IS NOT NULL").bind(new Date().toISOString(), match[1]).run();
    if (!result.meta?.changes) fail(404, 'not_found', 'Pending submission not found.');
    return json({ id: match[1], status: 'approved' });
  }
  match = new RegExp(`^${API}/admin/submissions/([0-9a-f-]{36})$`).exec(path);
  if (match && UUID.test(match[1]) && method === 'DELETE') {
    await admin(request, env);
    const result = await env.BENCHMARKS_DB.prepare('DELETE FROM submissions WHERE id = ?').bind(match[1]).run();
    if (!result.meta?.changes) fail(404, 'not_found', 'Submission not found.');
    return json({ id: match[1], deleted: true });
  }
  fail(404, 'not_found', 'Benchmark endpoint not found.');
}

// Returns null for existing site pages. A site's main worker can call this first.
export async function handleRequest(request, env) {
  const url = new URL(request.url), path = url.pathname;
  if (['/benchmarks', '/benchmarks/'].includes(path) && request.method === 'GET') return new Response(BOARD_HTML, { headers: {
    ...HEADERS, 'content-type': 'text/html; charset=utf-8',
    'content-security-policy': "default-src 'none'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
  } });
  if (path === '/benchmarks/app.js' && request.method === 'GET') return new Response(BOARD_JS, { headers: { ...HEADERS, 'content-type': 'text/javascript; charset=utf-8' } });
  if (path === '/benchmarks/protocol.js' && request.method === 'GET') return new Response(BROWSER_PROTOCOL, { headers: { ...HEADERS, 'content-type': 'text/javascript; charset=utf-8' } });
  if (path === '/benchmarks/styles.css' && request.method === 'GET') return new Response(BOARD_CSS, { headers: { ...HEADERS, 'content-type': 'text/css; charset=utf-8' } });
  if (!path.startsWith(API + '/') && path !== API) return null;
  try {
    const origin = request.headers.get('origin');
    if (origin && origin !== url.origin && origin !== env.SITE_ORIGIN) fail(403, 'origin_denied', 'Cross-origin requests are not accepted.');
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: {
      ...HEADERS, 'access-control-allow-origin': origin || url.origin,
      'access-control-allow-methods': 'GET, POST, DELETE, OPTIONS',
      'access-control-allow-headers': 'Authorization, Content-Type', 'access-control-max-age': '600',
    } });
    return await routes(request, env, url);
  } catch (error) {
    if (error instanceof ApiError) return json({ error: { code: error.code, message: error.message } }, error.status,
      error.status === 401 ? { 'www-authenticate': 'Bearer' } : {});
    // Never log bodies, tokens, or database exception strings.
    return json({ error: { code: 'service_error', message: 'The benchmark service is temporarily unavailable.' } }, 503);
  }
}

export default {
  async fetch(request, env) {
    const handled = await handleRequest(request, env);
    if (handled) return handled;
    return env.ASSETS ? env.ASSETS.fetch(request) : new Response('Not found', { status: 404, headers: HEADERS });
  },
};
