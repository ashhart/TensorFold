import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { handleRequest } from '../src/index.js';
import { SUITE_SHA256 } from '../src/protocol.js';

export class TestD1 {
  constructor() {
    this.sqlite = new DatabaseSync(':memory:');
    this.sqlite.exec(readFileSync(new URL('../migrations/0001.sql', import.meta.url), 'utf8'));
  }
  prepare(sql) {
    const statement = this.sqlite.prepare(sql);
    const wrap = values => ({
      bind: (...next) => wrap(next),
      first: async () => statement.get(...values) || null,
      all: async () => ({ results: statement.all(...values) }),
      run: async () => ({ meta: { changes: Number(statement.run(...values).changes) } }),
    });
    return wrap([]);
  }
  close() { this.sqlite.close(); }
}
export function environment() {
  return { BENCHMARKS_DB: new TestD1(), ADMIN_TOKEN: crypto.randomUUID() + crypto.randomUUID() };
}
export function fixture() {
  const samples = [];
  for (const fixture_id of ['code', 'chat']) for (const temperature of [1, 0]) for (let repeat = 0; repeat < 5; repeat++) {
    samples.push({ fixture_id, temperature, repeat, seed: 1234 + repeat, status: 'ok', prompt_tokens: 32,
      completion_tokens: 256, cached_tokens: 0, delivery_seconds: 2.55, delivery_tps: 100,
      ttft_seconds: .1, end_to_end_seconds: 2.7, server_decode_tps: null, server_decode_seconds: null,
      prefill_seconds: null, token_sha: 'a'.repeat(12), error_code: null });
  }
  return {
    schema_version: 1, suite: { id: 'community-v1', sha256: SUITE_SHA256 },
    run_id: crypto.randomUUID(), created_at: new Date().toISOString(),
    runtime: { tensorfold_version: 'test-version', backend: 'mlx', python_version: '3.12.0', platform: 'macos', dependencies: { mlx: 'test-version' } },
    model: { repo_id: 'example-org/example-model', revision: 'a'.repeat(40), family: 'example', config_sha256: 'b'.repeat(64),
      tokenizer_sha256: null, quantization: '4bit', drafter_repo_id: null, local: false },
    hardware: { cpu_model: 'Example CPU', system_memory_bytes: 64 * 2 ** 30, memory_type: 'unified',
      gpus: [{ name: 'Example GPU', memory_bytes: 64 * 2 ** 30, cores: 40 }], gpu_count: 1, source: 'detected' },
    settings: { tokens: 256, repetitions: 5, temperatures: [1, 0], serial: false, context_tokens: 4096, managed: true, rank_count: 1 }, samples,
  };
}
export async function call(env, path, { method = 'GET', token, body, headers = {} } = {}) {
  const request = new Request('https://example.test' + path, { method,
    headers: { ...(body !== undefined ? { 'content-type': 'application/json' } : {}), ...(token ? { authorization: 'Bearer ' + token } : {}), ...headers },
    ...(body !== undefined ? { body: typeof body === 'string' ? body : JSON.stringify(body) } : {}),
  });
  const response = await handleRequest(request, env);
  if (!response) return { status: null, body: null };
  const value = response.headers.get('content-type')?.includes('application/json') ? await response.json() : await response.text();
  return { status: response.status, body: value, headers: response.headers };
}
export async function publisher(env, quota = 20) {
  const reply = await call(env, '/api/benchmarks/v1/admin/publishers', { method: 'POST', token: env.ADMIN_TOKEN, body: { display_name: 'example-runner', daily_quota: quota } });
  if (reply.status !== 201) throw new Error('Test publisher creation failed');
  return reply.body;
}
