export const SUITE = Object.freeze({
  id: 'community-v1',
  fixtures: [
    { id: 'code', kind: 'completion', prompt: 'Write a short Python function that computes the Fibonacci sequence and explain it.' },
    { id: 'chat', kind: 'chat', prompt: 'Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.' },
  ],
  max_tokens: 256,
  repetitions: 5,
  warmups_per_cell: 1,
  temperatures: [1, 0],
  seed_base: 1234,
  sampled_top_k: 20,
  sampled_top_p: 0.95,
  thinking: false,
  ignore_eos: false,
  concurrency: 1,
  delivery_rate: 'completion_tokens_minus_one_over_first_to_last_nonempty_sse_delivery',
});
export const SUITE_SHA256 = '9bb717da00666b90b4006174abd9ed46108affc37456b43c64a224a9c938d5be';
export const MAX_BYTES = 1024 * 1024;
const SAMPLE_KEYS = ['fixture_id', 'temperature', 'repeat', 'seed', 'status', 'prompt_tokens', 'completion_tokens',
  'cached_tokens', 'delivery_seconds', 'delivery_tps', 'ttft_seconds', 'end_to_end_seconds', 'server_decode_tps',
  'server_decode_seconds', 'prefill_seconds', 'token_sha', 'error_code'];
const ERROR_CODES = new Set(['http_error', 'connection_error', 'timeout', 'cancelled', 'server_error', 'invalid_stream',
  'truncated_stream', 'missing_usage', 'invalid_usage', 'empty_output', 'unexpected_finish', 'token_count_mismatch', 'unmeasured_delivery']);
const VERSION = /^[A-Za-z0-9][A-Za-z0-9._+ -]{0,79}$/;
const NAME = /^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,119}$/;
const REPO = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,159}$/;
export const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export class ApiError extends Error {
  constructor(status, code, message) { super(message); this.status = status; this.code = code; }
}
const invalid = message => { throw new ApiError(400, 'invalid_receipt', message); };

function object(value, keys, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
    Object.keys(value).length !== keys.length || keys.some(key => !Object.hasOwn(value, key))) invalid(`Invalid ${label} fields.`);
}
function number(value, { nullable = false, integer = false, min = 0, max = 1e15 } = {}) {
  if (nullable && value === null) return;
  if (typeof value !== 'number' || !Number.isFinite(value) || integer && !Number.isSafeInteger(value) || value < min || value > max)
    invalid('A benchmark number is outside its allowed range.');
}
function string(value, pattern, nullable = false) {
  if (nullable && value === null) return;
  if (typeof value !== 'string' || !pattern.test(value)) invalid('Invalid public benchmark identifier.');
}
function hex(value, lengths = [64], nullable = true) {
  if (nullable && value === null) return;
  if (typeof value !== 'string' || !lengths.includes(value.length) || !/^[a-f0-9]+$/.test(value)) invalid('Invalid benchmark digest.');
}

export function validateReceipt(receipt) {
  object(receipt, ['schema_version', 'suite', 'run_id', 'created_at', 'runtime', 'model', 'hardware', 'settings', 'samples'], 'receipt');
  if (receipt.schema_version !== 1) invalid('Unsupported benchmark schema version.');
  object(receipt.suite, ['id', 'sha256'], 'suite');
  if (receipt.suite.id !== SUITE.id || receipt.suite.sha256 !== SUITE_SHA256) invalid('Unsupported benchmark suite or fixture digest.');
  string(receipt.run_id, UUID);
  if (typeof receipt.created_at !== 'string' || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?Z$/.test(receipt.created_at) ||
      !Number.isFinite(Date.parse(receipt.created_at)) ||
      new Date(receipt.created_at).toISOString().slice(0, 19) !== receipt.created_at.slice(0, 19) ||
      Date.parse(receipt.created_at) > Date.now() + 15 * 60 * 1000)
    invalid('Invalid UTC benchmark timestamp.');
  const r = receipt.runtime;
  object(r, ['tensorfold_version', 'backend', 'python_version', 'platform', 'dependencies'], 'runtime');
  if (!['mlx', 'cuda'].includes(r.backend) || !['macos', 'linux'].includes(r.platform) ||
    r.backend === 'mlx' && r.platform !== 'macos' || r.backend === 'cuda' && r.platform !== 'linux') invalid('Unsupported benchmark runtime.');
  string(r.tensorfold_version, VERSION); string(r.python_version, VERSION);
  if (!r.dependencies || typeof r.dependencies !== 'object' || Array.isArray(r.dependencies) ||
    Object.keys(r.dependencies).some(key => !['mlx', 'mlx_lm', 'torch', 'triton', 'cuda', 'driver'].includes(key))) invalid('Invalid runtime dependency fields.');
  for (const value of Object.values(r.dependencies)) string(value, VERSION, true);
  const m = receipt.model;
  object(m, ['repo_id', 'revision', 'family', 'config_sha256', 'tokenizer_sha256', 'quantization', 'drafter_repo_id', 'local'], 'model');
  string(m.repo_id, REPO); string(m.drafter_repo_id, REPO, true);
  if (m.local !== false) invalid('Local model labels cannot be published. Use a public checkpoint ID.');
  hex(m.revision, [40, 64]); hex(m.config_sha256); hex(m.tokenizer_sha256);
  string(m.family, NAME, true); string(m.quantization, NAME, true);
  const h = receipt.hardware;
  object(h, ['cpu_model', 'system_memory_bytes', 'memory_type', 'gpus', 'gpu_count', 'source'], 'hardware');
  string(h.cpu_model, NAME, true); number(h.system_memory_bytes, { nullable: true, integer: true });
  number(h.gpu_count, { nullable: true, integer: true, max: 256 });
  if (!['unified', 'dedicated', 'unknown'].includes(h.memory_type) || !['detected', 'partial', 'unavailable'].includes(h.source)) invalid('Invalid hardware metadata.');
  if (!Array.isArray(h.gpus) || h.gpus.length > 256) invalid('Invalid GPU list.');
  for (const gpu of h.gpus) {
    object(gpu, ['name', 'memory_bytes', 'cores'], 'GPU'); string(gpu.name, NAME);
    number(gpu.memory_bytes, { nullable: true, integer: true }); number(gpu.cores, { nullable: true, integer: true, max: 100000 });
  }
  if (h.gpu_count !== null && h.gpu_count !== h.gpus.length) invalid('GPU count differs from its inventory.');
  const s = receipt.settings;
  object(s, ['tokens', 'repetitions', 'temperatures', 'serial', 'context_tokens', 'managed', 'rank_count'], 'settings');
  number(s.tokens, { integer: true, min: 2, max: 8192 }); number(s.repetitions, { integer: true, min: 1, max: 20 });
  number(s.context_tokens, { nullable: true, integer: true, min: 1, max: 10000000 });
  number(s.rank_count, { nullable: true, integer: true, min: 1, max: 2 });
  if (typeof s.serial !== 'boolean' || typeof s.managed !== 'boolean') invalid('Invalid execution flags.');
  if (!Array.isArray(s.temperatures) || s.temperatures.length < 1 || s.temperatures.length > 4 || new Set(s.temperatures).size !== s.temperatures.length) invalid('Invalid benchmark temperatures.');
  s.temperatures.forEach(value => number(value, { max: 2 }));
  if (!Array.isArray(receipt.samples) || receipt.samples.length < 1 || receipt.samples.length > 200) invalid('Invalid benchmark sample count.');
  const seen = new Set();
  for (const sample of receipt.samples) {
    object(sample, SAMPLE_KEYS, 'sample');
    if (!['code', 'chat'].includes(sample.fixture_id) || !s.temperatures.includes(sample.temperature)) invalid('Sample is outside the requested cells.');
    number(sample.repeat, { integer: true, min: -1, max: s.repetitions - 1 }); number(sample.seed, { integer: true, max: 2 ** 32 - 1 });
    const key = `${sample.fixture_id}:${sample.temperature}:${sample.repeat}`;
    if (seen.has(key)) invalid('Duplicate benchmark repetition.');
    seen.add(key);
    if (!['ok', 'early_eos', 'error', 'unmeasured'].includes(sample.status)) invalid('Invalid sample status.');
    if (sample.repeat === -1 && sample.status !== 'error') invalid('Only a failed warmup may use repetition -1.');
    for (const key of ['prompt_tokens', 'completion_tokens', 'cached_tokens']) number(sample[key], { nullable: true, integer: true, max: 10000000 });
    for (const key of ['delivery_seconds', 'delivery_tps', 'ttft_seconds', 'end_to_end_seconds', 'server_decode_tps', 'server_decode_seconds', 'prefill_seconds'])
      number(sample[key], { nullable: true, max: 1e9 });
    hex(sample.token_sha, [12, 64]);
    if (sample.error_code !== null && !ERROR_CODES.has(sample.error_code)) invalid('Invalid benchmark error code.');
    const count = sample.completion_tokens, seconds = sample.delivery_seconds, rate = sample.delivery_tps;
    if (rate !== null) {
      if (count === null || count < 2 || seconds === null || seconds <= 0) invalid('A delivery rate needs a count and positive duration.');
      const expected = (count - 1) / seconds;
      if (Math.abs(rate - expected) > Math.max(1e-6, 1e-6 * Math.max(rate, expected))) invalid('Delivery rate differs from its count and timing.');
    }
    if (sample.status === 'ok' && (count !== s.tokens || rate === null || sample.error_code !== null)) invalid('A complete sample needs its requested output and timings.');
    if (sample.cached_tokens !== null && sample.prompt_tokens !== null && sample.cached_tokens > sample.prompt_tokens) invalid('Cached count exceeds its prompt.');
    if (sample.completion_tokens !== null && sample.completion_tokens > s.tokens) invalid('Output count exceeds the requested budget.');
  }
  return receipt;
}

export function canonical(value) {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
  return JSON.stringify(value);
}
export async function sha256(value) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}
function spread(values) {
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return { median: values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2,
    min: values[0], max: values.at(-1), spread: values.at(-1) - values[0] };
}
export function summarize(receipt) {
  const groups = new Map();
  for (const row of receipt.samples) {
    const key = `${row.fixture_id}:${row.temperature}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const repetitions = receipt.settings.repetitions;
  const standard = receipt.settings.tokens === SUITE.max_tokens && repetitions === SUITE.repetitions &&
    receipt.settings.managed && receipt.settings.rank_count === 1 && receipt.model.revision !== null &&
    receipt.model.config_sha256 !== null && receipt.hardware.source !== 'unavailable' && receipt.hardware.gpus.length > 0;
  const summary = [];
  for (const rows of groups.values()) {
    const measured = rows.filter(row => row.repeat >= 0);
    const complete = rows.length === repetitions &&
      rows.map(row => row.repeat).sort((a, b) => a - b).every((repeat, index) => repeat === index) &&
      rows.every(row => row.status === 'ok' && row.seed === SUITE.seed_base + row.repeat);
    const cell = { fixture_id: rows[0].fixture_id, temperature: rows[0].temperature, repeats: measured.length,
      status_counts: {}, protocol_complete: complete && repetitions === SUITE.repetitions };
    for (const row of rows) cell.status_counts[row.status] = (cell.status_counts[row.status] || 0) + 1;
    for (const metric of ['delivery_tps', 'ttft_seconds', 'end_to_end_seconds', 'server_decode_tps', 'prefill_seconds']) {
      const values = measured.map(row => row[metric]);
      cell[metric] = complete && values.every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0) ? spread(values) : null;
    }
    summary.push(cell);
  }
  const protocol_complete = standard && receipt.settings.temperatures.length === SUITE.temperatures.length &&
    SUITE.fixtures.every(fixture => SUITE.temperatures.every(temperature =>
      summary.some(cell => cell.fixture_id === fixture.id && cell.temperature === temperature && cell.protocol_complete)));
  return { summary, protocol_complete };
}

// The browser uses the exact upload validator before it sends any receipt.
// This bundle has no server credentials, storage access, or network calls.
export const BROWSER_PROTOCOL = [
  'const SUITE=' + JSON.stringify(SUITE) + ';',
  'const SUITE_SHA256=' + JSON.stringify(SUITE_SHA256) + ';',
  'const SAMPLE_KEYS=' + JSON.stringify(SAMPLE_KEYS) + ';',
  'const ERROR_CODES=new Set(' + JSON.stringify([...ERROR_CODES]) + ');',
  'const VERSION=' + VERSION.toString() + ';', 'const NAME=' + NAME.toString() + ';',
  'const REPO=' + REPO.toString() + ';', 'const UUID=' + UUID.toString() + ';',
  ApiError.toString(), 'const invalid=' + invalid.toString() + ';',
  object.toString(), number.toString(), string.toString(), hex.toString(), validateReceipt.toString(),
  'export {validateReceipt};',
].join('\n');
