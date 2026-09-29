import test from 'node:test';
import assert from 'node:assert/strict';
import { call, environment, fixture, publisher } from './helpers.js';
import { BROWSER_PROTOCOL as VALIDATOR_SOURCE, canonical, sha256, summarize, validateReceipt } from '../src/protocol.js';
import { BROWSER_PROTOCOL } from '../src/browser-protocol.js';
import { BOARD_JS } from '../src/board.js';

const base = '/api/benchmarks/v1';
function publicLookup(t) {
  return t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.match(url, /^https:\/\/huggingface\.co\/api\/models\/example-org\/example-model$/);
    assert.equal(options.redirect, 'manual'); assert.equal(options.headers.authorization, undefined);
    return Response.json({ private: false });
  });
}
test('auth, moderation, ownership, idempotency and withdrawal with a real SQLite schema', async t => {
  publicLookup(t);
  const env = environment(); t.after(() => env.BENCHMARKS_DB.close());
  assert.equal((await call(env, base + '/submissions', { method: 'POST', body: fixture() })).status, 401);
  const one = await publisher(env), two = await publisher(env);
  const me = await call(env, base + '/me', { token: one.token });
  assert.deepEqual(me.body, { publisher_id: one.publisher_id, display_name: 'example-runner' });
  const receipt = fixture(), first = await call(env, base + '/submissions', { method: 'POST', token: one.token, body: receipt });
  assert.equal(first.status, 201); assert.equal(first.body.status, 'pending'); assert.equal(first.body.protocol_complete, true);
  assert.equal(first.body.content_sha256, await sha256(canonical(receipt)));
  const id = first.body.id;
  const retry = await call(env, base + '/submissions', { method: 'POST', token: one.token, body: receipt });
  assert.equal(retry.status, 200); assert.equal(retry.body.id, id);
  const changed = structuredClone(receipt); changed.samples[0].token_sha = 'c'.repeat(12);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: one.token, body: changed })).status, 409);
  assert.deepEqual((await call(env, base + '/results')).body.results, []);
  assert.equal((await call(env, base + '/results/' + id)).status, 404);
  assert.equal((await call(env, base + '/submissions/' + id, { token: two.token })).status, 404);
  assert.equal((await call(env, base + '/submissions/' + id, { token: two.token, method: 'DELETE' })).status, 404);
  const queue = await call(env, base + '/admin/submissions', { token: env.ADMIN_TOKEN });
  assert.equal(queue.body.submissions.length, 1);
  assert.deepEqual(queue.body.submissions[0].receipt, receipt);
  assert.equal((await call(env, base + '/admin/submissions/' + id, { token: env.ADMIN_TOKEN })).body.status, 'pending');
  assert.equal((await call(env, base + '/admin/submissions/' + id + '/approve', { method: 'POST', token: two.token })).status, 403);
  assert.equal((await call(env, base + '/admin/submissions/' + id + '/approve', { method: 'POST', token: env.ADMIN_TOKEN })).status, 200);
  const approved = await call(env, base + '/results/' + id);
  assert.equal(approved.status, 200); assert.equal(approved.body.self_reported, true);
  assert.equal(approved.body.summary[0].delivery_tps.median, 100);
  assert.equal(JSON.stringify(approved.body).includes(one.token), false);
  assert.equal((await call(env, base + '/results?model=other/model')).body.results.length, 0);
  assert.equal((await call(env, base + '/results?gpu=Example%20GPU&backend=mlx')).body.results.length, 1);
  assert.equal((await call(env, base + '/submissions/' + id, { token: one.token, method: 'DELETE' })).status, 200);
  assert.equal((await call(env, base + '/results/' + id)).status, 404);
  const removed = env.BENCHMARKS_DB.sqlite.prepare('SELECT receipt_json, summary_json FROM submissions WHERE id = ?').get(id);
  assert.equal(removed.receipt_json, null); assert.equal(removed.summary_json, null);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: one.token, body: receipt })).body.status, 'withdrawn');
});
test('quotas count accepted uploads, including withdrawn runs, and tokens can be revoked', async t => {
  publicLookup(t); const env = environment(); t.after(() => env.BENCHMARKS_DB.close());
  const owner = await publisher(env, 1), receipt = fixture();
  const first = await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: receipt });
  assert.equal(first.status, 201);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: receipt })).status, 200);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: fixture() })).status, 429);
  await call(env, base + '/submissions/' + first.body.id, { token: owner.token, method: 'DELETE' });
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: fixture() })).status, 429);
  assert.equal((await call(env, base + '/admin/publishers/' + owner.publisher_id, { token: env.ADMIN_TOKEN, method: 'DELETE' })).status, 200);
  assert.equal((await call(env, base + '/me', { token: owner.token })).status, 401);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: fixture() })).status, 401);
});
test('private, redirected, unavailable and oversized checkpoint metadata never get stored', async t => {
  const env = environment(); t.after(() => env.BENCHMARKS_DB.close()); const owner = await publisher(env);
  for (const [response, status, code] of [
    [Response.json({ private: true }), 400, 'public_checkpoint_required'],
    [new Response('', { status: 404 }), 400, 'public_checkpoint_required'],
    [new Response('', { status: 302, headers: { location: 'https://example.invalid/private' } }), 400, 'public_checkpoint_required'],
    [new Response('', { status: 500 }), 503, 'checkpoint_check_unavailable'],
    [Response.json({ private: false, padding: 'x'.repeat(300000) }), 503, 'checkpoint_check_unavailable'],
  ]) {
    const mock = t.mock.method(globalThis, 'fetch', async () => response);
    const result = await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: fixture() });
    assert.equal(result.status, status); assert.equal(result.body.error.code, code); mock.mock.restore();
  }
  assert.equal(env.BENCHMARKS_DB.sqlite.prepare('SELECT count(*) n FROM submissions').get().n, 0);
});
test('strict receipts reject extra data, invalid metrics, paths and nonfinite numbers', () => {
  const mutations = [
    r => r.notes = 'unreviewed content', r => r.hardware.hostname = 'example-host',
    r => r.model.repo_id = '/tmp/example-model', r => r.model.local = true,
    r => r.runtime.dependencies.secret = 'example', r => r.samples[0].generated_text = 'do not upload',
    r => r.samples[0].delivery_tps = 999, r => r.samples[0].completion_tokens = true,
    r => r.samples[0].delivery_seconds = Infinity, r => r.samples[0].cached_tokens = 33,
    r => r.samples.push(structuredClone(r.samples[0])), r => r.samples[0].repeat = -1,
    r => r.samples[0].error_code = 'unbounded_error_text', r => r.suite.sha256 = '0'.repeat(64),
    r => r.settings.rank_count = 3, r => r.model.revision = 'main', r => r.hardware.cpu_model = 'person@example.test',
  ];
  for (const change of mutations) { const receipt = fixture(); change(receipt); assert.throws(() => validateReceipt(receipt)); }
  assert.doesNotThrow(() => validateReceipt(fixture()));
});
test('the shipped browser validator matches the server schema before any upload', async () => {
  assert.equal(BROWSER_PROTOCOL, VALIDATOR_SOURCE);
  const { validateReceipt: browserValidate } = await import('data:text/javascript;base64,' + Buffer.from(BROWSER_PROTOCOL).toString('base64'));
  assert.doesNotThrow(() => browserValidate(fixture()));
  const receipt = fixture(); receipt.samples[0].prompt = 'private user prompt';
  assert.throws(() => browserValidate(receipt));
});
test('server recomputation hides incomplete rates and separates nonstandard and attached runs', () => {
  const standard = fixture(); assert.equal(summarize(standard).protocol_complete, true);
  const failed = fixture(); failed.samples[0].status = 'error'; failed.samples[0].error_code = 'timeout';
  assert.equal(summarize(failed).protocol_complete, false); assert.equal(summarize(failed).summary[0].delivery_tps, null);
  const warmup = fixture(); warmup.samples.push({ ...warmup.samples[0], status: 'error', repeat: -1, error_code: 'timeout' });
  assert.equal(summarize(warmup).summary[0].delivery_tps, null);
  const attached = fixture(); attached.settings.managed = false; attached.settings.rank_count = null;
  assert.equal(summarize(attached).protocol_complete, false);
  const custom = fixture(); custom.settings.repetitions = 2; custom.samples = custom.samples.filter(s => s.repeat < 2);
  assert.equal(summarize(custom).protocol_complete, false); assert.equal(summarize(custom).summary[0].delivery_tps.median, 100);
});
test('bounded JSON, origin protection, pagination, static board and fallback routing', async t => {
  publicLookup(t); const env = environment(); t.after(() => env.BENCHMARKS_DB.close()); const owner = await publisher(env);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: 'x'.repeat(1024 * 1024 + 1) })).status, 413);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: '{}', headers: { 'content-type': 'text/plain' } })).status, 415);
  assert.equal((await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: '{' })).status, 400);
  assert.equal((await call(env, base + '/me', { token: owner.token, headers: { origin: 'https://example.invalid' } })).status, 403);
  assert.equal((await call(env, base + '/results?limit=999')).status, 400);
  assert.equal((await call(env, base + '/results?backend=mlx&backend=cuda')).status, 400);
  for (let i = 0; i < 3; i++) {
    const result = await call(env, base + '/submissions', { method: 'POST', token: owner.token, body: fixture() });
    await call(env, base + '/admin/submissions/' + result.body.id + '/approve', { method: 'POST', token: env.ADMIN_TOKEN });
  }
  const one = await call(env, base + '/results?limit=2'); assert.equal(one.body.results.length, 2); assert.ok(one.body.next_cursor);
  const two = await call(env, base + '/results?limit=2&cursor=' + encodeURIComponent(one.body.next_cursor));
  assert.equal(two.body.results.length, 1); assert.equal(two.body.next_cursor, null);
  assert.equal(new Set([...one.body.results, ...two.body.results].map(r => r.id)).size, 3);
  const board = await call(env, '/benchmarks'); assert.equal(board.status, 200); assert.match(board.body, /<h1>Benchmarks/);
  assert.match(board.headers.get('content-security-policy'), /frame-ancestors 'none'/);
  assert.equal((await call(env, '/')).status, null);
  assert.doesNotThrow(() => new Function(BOARD_JS.replace(/^import[^\n]+\n/, '')));
});
