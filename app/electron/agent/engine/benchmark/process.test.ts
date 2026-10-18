import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, rm, readFile, stat } from 'node:fs/promises'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import path from 'node:path'
import os from 'node:os'
import { benchmarkArguments, benchmarkEnvironment, startBenchmarkProcess } from './process'
import { readBenchmarkReceiptFile } from './storage'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-benchmark-process-'))
const input = { id: 'benchmark_synthetic_process', model: 'example-org/synthetic-model', serial: false }
const fixture = path.join(__dirname, 'fake-benchmark.mjs')
const env = { HOME: home, PATH: process.env.PATH, HF_HUB_CACHE: path.join(home, 'hub'), OPENAI_API_KEY: 'SYNTHETIC_SECRET_NOT_ALLOWED', TENSORFOLD_BENCHMARK_TOKEN: 'SYNTHETIC_UPLOAD_NOT_ALLOWED', NODE_OPTIONS: '--inspect', TENSORFOLD_SEED_SALT: 'SYNTHETIC_NOT_ALLOWED', HTTP_PROXY: 'http://proxy.example.com' }
assert.deepEqual(benchmarkEnvironment(env), { HOME: home, PATH: process.env.PATH, HF_HUB_CACHE: path.join(home, 'hub'), HF_HUB_OFFLINE: '1', TENSORFOLD_NO_LIVE: '1', TENSORFOLD_NO_UPDATE_CHECK: '1' })
assert.ok(!benchmarkArguments(input, '/synthetic/receipt.json').some(value => /publish|login|upload|cortheon|download/.test(value)))
assert.throws(() => benchmarkArguments({ ...input, model: '--publish' }, '/synthetic/receipt.json'), /installed/)
try {
  const progress: unknown[] = []
  let childArgs: string[] = []; let childEnv: NodeJS.ProcessEnv | undefined
  const normal = startBenchmarkProcess({ binary: { command: process.execPath, args: [fixture] }, input, receiptPath: path.join(home, 'receipt.json'), workingDirectory: home, env, progress: value => progress.push(value), spawnImpl: ((command: any, args: any, options: any) => { childArgs = args; childEnv = options.env; return spawn(command, args, options) }) as typeof spawn })
  assert.equal((await normal.done).code, 0)
  assert.equal((progress.at(-1) as any).completedRequests, 24)
  assert.equal(childEnv?.OPENAI_API_KEY, undefined)
  assert.deepEqual(childArgs.slice(1), benchmarkArguments(input, path.join(home, 'receipt.json')))
  const receipt = await readBenchmarkReceiptFile(path.join(home, 'receipt.json'), input)
  assert.equal(receipt?.samples.length, 20)
  assert.equal((await stat(path.join(home, 'receipt.json'))).mode & 0o777, 0o600)
  const beforeAbort = new AbortController(); beforeAbort.abort()
  let abortedSpawns = 0
  const before = startBenchmarkProcess({ binary: { command: process.execPath, args: [fixture] }, input, receiptPath: path.join(home, 'never.json'), workingDirectory: home, env, progress() {}, signal: beforeAbort.signal, spawnImpl: (() => { abortedSpawns++; throw new Error('Must not spawn') }) as typeof spawn })
  assert.equal((await before.done).cancelled, true)
  assert.equal(abortedSpawns, 0)
  const stderrProgress: unknown[] = []
  const stderr = startBenchmarkProcess({ binary: { command: process.execPath, args: [fixture, 'scenario:stderr'] }, input, receiptPath: path.join(home, 'stderr.json'), workingDirectory: home, env, progress: value => stderrProgress.push(value) })
  assert.equal((await stderr.done).code, 0)
  assert.equal((stderrProgress.at(-1) as any).completedRequests, 24, 'only anchored progress is accepted from either stream')
  let ready!: () => void
  const started = new Promise<void>(resolve => { ready = resolve })
  const cancelled = startBenchmarkProcess({ binary: { command: process.execPath, args: [fixture, 'scenario:cancel'] }, input, receiptPath: path.join(home, 'partial.json'), workingDirectory: home, env, progress: () => ready(), cancelGraceMs: 500 })
  await started; cancelled.cancel(); cancelled.cancel()
  const ended = await cancelled.done
  assert.equal(ended.cancelled, true)
  assert.equal(ended.code, 130)
  assert.equal((await readBenchmarkReceiptFile(path.join(home, 'partial.json'), input))?.samples[0].error_code, 'cancelled')
  let stubbornReady!: () => void
  const stubbornStarted = new Promise<void>(resolve => { stubbornReady = resolve })
  const stubborn = startBenchmarkProcess({ binary: { command: process.execPath, args: [fixture, 'scenario:stubborn'] }, input, receiptPath: path.join(home, 'none.json'), workingDirectory: home, env, progress: () => stubbornReady(), cancelGraceMs: 50 })
  await stubbornStarted; stubborn.cancel()
  assert.equal((await stubborn.done).cancelled, true, 'unresponsive owned group is force-stopped after the grace deadline')
  let orphanReady!: () => void
  const orphanStarted = new Promise<void>(resolve => { orphanReady = resolve })
  const orphan = startBenchmarkProcess({ binary: { command: process.execPath, args: [fixture, 'scenario:orphan'] }, input, receiptPath: path.join(home, 'orphan.json'), workingDirectory: home, env, progress: () => orphanReady(), cancelGraceMs: 500 })
  await orphanStarted
  const orphanPid = Number(await readFile(path.join(home, 'orphan.json.childpid'), 'utf8'))
  orphan.cancel()
  const orphanEnd = await orphan.done
  assert.equal(orphanEnd.cleanupFailed, false, 'leader close must verify and stop the surviving owned child before resolving')
  let remaining = ''
  try { remaining = (await promisify(execFile)('/bin/ps', ['-o', 'stat=', '-p', String(orphanPid)])).stdout.trim() } catch {}
  assert.ok(!remaining || remaining.startsWith('Z'), 'no surviving child may keep executing after the leader exits')
  const missing = startBenchmarkProcess({ binary: { command: '/nonexistent/synthetic-binary', args: [] }, input, receiptPath: path.join(home, 'missing.json'), workingDirectory: home, env, progress() {} })
  assert.equal((await missing.done).failed, true)
} finally { await rm(home, { recursive: true, force: true }) }
console.log('Benchmark fixed argv/offline stripped environment, true progress, managed process, partial SIGINT and bounded force cleanup passed')
