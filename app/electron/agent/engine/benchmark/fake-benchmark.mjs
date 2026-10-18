// Synthetic process fixture: never loads a model, downloads or reaches an external API.
import { readFileSync, writeFileSync } from 'node:fs'
import { spawnSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

let args = process.argv.slice(2)
let scenario = 'complete'
if (args[0]?.startsWith('scenario:')) scenario = args.shift().slice(9)
const engineVersion = scenario === 'old-version' ? '1.0.12' : scenario === 'unknown-version' ? '9.9.9' : '1.1.0'
if (args[0] === '--version') { console.log(`tensorfold-native ${engineVersion}`); process.exit(0) }
if (args[0] !== 'benchmark') {
  const fixture = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../test-fixtures/fake-engine.mjs')
  process.exit(spawnSync(process.execPath, [fixture, ...args], { stdio: 'inherit', env: process.env }).status ?? 1)
}
const model = args[1], output = args[args.indexOf('--output') + 1]
const serial = args.includes('--serial')
const root = path.join(process.env.HF_HUB_CACHE ?? '', `models--${model.replace('/', '--')}`)
let revision = 'a'.repeat(40), config = Buffer.from('{}')
try { revision = readFileSync(path.join(root, 'refs', 'main'), 'utf8').trim(); config = readFileSync(path.join(root, 'snapshots', revision, 'config.json')) } catch {}
const receipt = { schema_version: 1, suite: { id: 'community-v1', sha256: '9bb717da00666b90b4006174abd9ed46108affc37456b43c64a224a9c938d5be' }, run_id: '00000000-0000-4000-8000-000000000001', created_at: '2026-01-01T00:00:00Z', runtime: { tensorfold_version: scenario === 'mismatched-receipt' ? '1.0.12' : engineVersion, backend: 'mlx', python_version: 'none', platform: 'macos', dependencies: {} }, model: { repo_id: model, revision, family: 'qwen3_5', config_sha256: createHash('sha256').update(config).digest('hex'), tokenizer_sha256: null, quantization: 'mlx-4bit-g64', drafter_repo_id: null, local: false }, hardware: { cpu_model: 'Synthetic CPU', system_memory_bytes: 16 * 1024 ** 3, memory_type: 'unified', gpus: [{ name: 'Synthetic GPU', memory_bytes: null, cores: 8 }], gpu_count: 1, source: 'detected' }, settings: { tokens: 256, repetitions: 5, temperatures: [1, 0], serial, context_tokens: 4096, managed: true, rank_count: 1 }, samples: [] }
const sample = (fixture_id, temperature, repeat) => ({ fixture_id, temperature, repeat, seed: 1234 + Math.max(0, repeat), status: 'ok', prompt_tokens: 24, completion_tokens: 256, cached_tokens: 0, delivery_seconds: 2.55, delivery_tps: 100, ttft_seconds: 0.1, end_to_end_seconds: 2.8, server_decode_tps: 110, server_decode_seconds: null, prefill_seconds: null, token_sha: 'd'.repeat(12), error_code: null })
const save = () => writeFileSync(output, JSON.stringify(receipt), { mode: 0o600, flag: 'wx' })
const say = text => scenario === 'stderr' ? console.error(text) : console.log(text)
if (scenario === 'missing') process.exit(1)
if (scenario === 'stubborn' || scenario === 'orphan') {
  const child = spawn(process.execPath, ['-e', "process.on('SIGINT',()=>{});setInterval(()=>{},1000)"], { stdio: 'ignore' })
  writeFileSync(output + '.childpid', String(child.pid), { mode: 0o600 })
  process.on('SIGINT', () => { if (scenario === 'orphan') process.exit(130) });
  await new Promise(resolve => setTimeout(resolve, 50))
  console.log('code T=1 warm-up: ok'); setInterval(() => {}, 1000)
} else if (scenario === 'cancel') {
  process.on('SIGINT', () => { const row = sample('code', 1, 0); row.status = 'error'; row.error_code = 'cancelled'; row.completion_tokens = null; row.delivery_seconds = null; row.delivery_tps = null; receipt.samples.push(row); save(); process.exit(130) })
  // Ready must mean the partial-receipt handler is installed before the parent cancels.
  console.log('code T=1 warm-up: ok')
  setInterval(() => {}, 1000)
} else {
  for (const temperature of [1, 0]) for (const fixture_id of ['code', 'chat']) {
    say(`${fixture_id} T=${temperature} warm-up: ok`)
    for (let repeat = 0; repeat < 5; repeat++) { receipt.samples.push(sample(fixture_id, temperature, repeat)); say(`${fixture_id} T=${temperature} repeat ${repeat + 1}/5: ok 100.00 delivery tok/s`) }
  }
  if (scenario === 'early') { Object.assign(receipt.samples[0], { status: 'early_eos', completion_tokens: 12, delivery_seconds: 0.11, delivery_tps: 100 }) }
  if (scenario === 'private') receipt.private_path = '/private/synthetic-operator-path'
  save()
  if (scenario === 'delayed') await new Promise(resolve => setTimeout(resolve, 150))
  process.exit(scenario === 'early' ? 2 : 0)
}
