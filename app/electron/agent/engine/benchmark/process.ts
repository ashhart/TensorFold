import { spawn, type ChildProcess } from 'node:child_process'
import type { EngineBinary } from '../paths'
import type { ModelBenchmarkInput, ModelBenchmarkProgress } from '../../../../shared/model-benchmark'
import { isRepoId } from '../paths'
import { killGroup } from '../pull'
import { cleanupOwnedBenchmarkGroup } from './owned-group'

export interface BenchmarkProcessResult { code: number | null; cancelled: boolean; overflow: boolean; failed: boolean; cleanupFailed?: boolean }
export interface BenchmarkProcessHandle { done: Promise<BenchmarkProcessResult>; cancel(force?: boolean): void }
export interface BenchmarkProcessOptions {
  binary: EngineBinary
  input: Required<ModelBenchmarkInput>
  receiptPath: string
  workingDirectory: string
  env: NodeJS.ProcessEnv
  progress(value: ModelBenchmarkProgress): void
  spawnImpl?: typeof spawn
  timeoutMs?: number
  cancelGraceMs?: number
  signal?: AbortSignal
}
const ENV_KEYS = ['HOME', 'PATH', 'TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'HF_HOME', 'HF_HUB_CACHE']
export function benchmarkEnvironment(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const key of ENV_KEYS) if (typeof base[key] === 'string') env[key] = base[key]
  env.HF_HUB_OFFLINE = '1'; env.TENSORFOLD_NO_LIVE = '1'; env.TENSORFOLD_NO_UPDATE_CHECK = '1'
  return env
}
export function benchmarkArguments(input: Required<ModelBenchmarkInput>, receiptPath: string): string[] {
  if (!isRepoId(input.model) || !receiptPath || /\0/.test(receiptPath)) throw new Error('Choose an installed Hugging Face checkpoint for the local benchmark.')
  return ['benchmark', input.model, '--context', '4096', '--tokens', '256', '--reps', '5', '--temperatures', '1,0', '--output', receiptPath, ...(input.serial ? ['--serial'] : [])]
}
export function startBenchmarkProcess(options: BenchmarkProcessOptions): BenchmarkProcessHandle {
  if (options.signal?.aborted) return { done: Promise.resolve({ code: null, cancelled: true, overflow: false, failed: false }), cancel() {} }
  const argv = [...options.binary.args, ...benchmarkArguments(options.input, options.receiptPath)]
  let child: ChildProcess | null = null
  let cancelled = false; let overflow = false; let settled = false; let finalizing = false; let bytes = 0
  const partial = { stdout: '', stderr: '' }
  const startedAt = Date.now()
  let forceTimer: ReturnType<typeof setTimeout> | undefined
  let timeout: ReturnType<typeof setTimeout> | undefined
  const attempts = new Set<string>()
  const cancel = (force = false) => {
    if (settled || cancelled && !force) return
    cancelled = true
    killGroup(child, force ? 'SIGKILL' : 'SIGINT')
    if (!force && !forceTimer) forceTimer = setTimeout(() => killGroup(child, 'SIGKILL'), options.cancelGraceMs ?? 15_000)
  }
  const abort = () => cancel()
  options.signal?.addEventListener('abort', abort, { once: true })
  const line = (text: string) => {
    const match = /^(code|chat) T=([01]) (warm-up|repeat ([1-5])\/5): (ok|early_eos|error|unmeasured)(?: \d+(?:\.\d+)? delivery tok\/s)?$/.exec(text)
    if (!match) return
    const repeat = match[3] === 'warm-up' ? -1 : Number(match[4]) - 1
    attempts.add(`${match[1]}:${match[2]}:${repeat}`)
    try { options.progress({ completedRequests: attempts.size, totalRequests: 24, fixture: match[1] as 'code' | 'chat', temperature: Number(match[2]) as 0 | 1, repeat, phase: repeat < 0 ? 'warmup' : 'measured' }) } catch {}
  }
  const consume = (chunk: Buffer | string, stream: 'stdout' | 'stderr') => {
    bytes += Buffer.byteLength(chunk)
    if (bytes > 512 * 1024) { overflow = true; cancel(); return }
    // Only anchored progress reaches the callback; raw diagnostics are always discarded.
    partial[stream] += chunk.toString()
    if (partial[stream].length > 8192 && !partial[stream].includes('\n')) { partial[stream] = ''; return }
    let at: number
    while ((at = partial[stream].indexOf('\n')) >= 0) { const text = partial[stream].slice(0, at).replace(/\r$/, ''); partial[stream] = partial[stream].slice(at + 1); if (text.length <= 4096) line(text) }
  }
  const done = new Promise<BenchmarkProcessResult>((resolve) => {
    const finish = async (code: number | null, failed: boolean) => {
      if (settled || finalizing) return
      finalizing = true
      const cleaned = !child?.pid || await cleanupOwnedBenchmarkGroup({ pid: child.pid, command: options.binary.command, startedAt })
      settled = true; clearTimeout(forceTimer); clearTimeout(timeout); options.signal?.removeEventListener('abort', abort)
      for (const value of Object.values(partial)) if (value.length <= 4096) line(value)
      resolve({ code, cancelled, overflow, failed: failed || !cleaned, cleanupFailed: !cleaned })
    }
    try {
      child = (options.spawnImpl ?? spawn)(options.binary.command, argv, { env: benchmarkEnvironment(options.env), cwd: options.workingDirectory, detached: true, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
      child.stdout?.on('data', chunk => consume(chunk, 'stdout'))
      child.stderr?.on('data', chunk => consume(chunk, 'stderr'))
      child.once('error', () => { void finish(null, true) })
      child.once('close', code => { void finish(code, false) })
      if (cancelled) killGroup(child, 'SIGINT')
      timeout = setTimeout(() => cancel(), options.timeoutMs ?? 45 * 60_000)
    } catch { void finish(null, true) }
  })
  return { done, cancel }
}
