import { spawn, type ChildProcess } from 'node:child_process'
import type { Readable } from 'node:stream'
import type { PullProgress } from '../types'
import { repoCachedBytes, type EngineBinary } from './paths'
import { PullInactivity, type PullTimeout } from './pull-inactivity'
import { pullEnvironment, pullFailure, redactPullDiagnostic } from './pull-network'

// Runs `tensorfold-native pull` for a repo and then any companion (the 27B's drafter), one process
// at a time in its own process group. pull prints a line per finished file but no byte counts, so
// progress is the bytes on disk under each repo's cache folder against the Hub's tree total.
// Cancel kills the group; pulling again resumes from the pieces already on disk.

export interface PullStep {
  repo: string
  revision: string | null
}

export interface PullOptions {
  binary: EngineBinary
  env: NodeJS.ProcessEnv
  hub: string
  steps: PullStep[]
  totalBytes: number | null
  pollMs?: number
  killGraceMs?: number
  inactivityTimer?: PullTimeout
  onUpdate: (progress: PullProgress) => void
}

export interface PullHandle {
  progress(): PullProgress
  cancel(): void
  done: Promise<PullProgress>
}

const KEEP_LINES = 40
const MAX_LINE_CHARS = 64 * 1024

// Cache bytes are measured, not guessed from the advertised model size. A resumed download's
// existing bytes establish the baseline and never count as this session's transfer speed.
export class TransferRateEstimator {
  private samples: Array<{ bytes: number; at: number }> = []
  private lastGrowthAt: number | null = null

  sample(bytes: number, at: number): number | null {
    const previous = this.samples.at(-1)
    if (!Number.isFinite(bytes) || !Number.isFinite(at) || bytes < 0) return null
    if (previous && (at <= previous.at || bytes < previous.bytes)) {
      this.samples = []
      this.lastGrowthAt = null
    }
    if (previous && bytes > previous.bytes) this.lastGrowthAt = at
    this.samples.push({ bytes, at })
    while (this.samples.length > 2 && (this.samples[1].at < at - 10_000 || this.samples.length > 64)) this.samples.shift()
    const first = this.samples[0]
    const elapsed = at - first.at
    if (elapsed < 2000 || this.lastGrowthAt == null || at - this.lastGrowthAt >= 10_000) return null
    const rate = (bytes - first.bytes) / (elapsed / 1000)
    return Number.isFinite(rate) && rate > 0 ? rate : null
  }
}

export function killGroup(child: ChildProcess | null, signal: NodeJS.Signals): void {
  if (!child?.pid || child.exitCode !== null || child.signalCode !== null) return
  try { process.kill(-child.pid, signal) } catch {
    try { child.kill(signal) } catch { /* already gone */ }
  }
}

export function startPull(opts: PullOptions): PullHandle {
  const first = opts.steps[0]
  if (!first) throw new Error('Nothing to pull.')
  const state: PullProgress = {
    repo: first.repo,
    revision: first.revision,
    state: 'running',
    phase: `Downloading ${first.repo}`,
    receivedBytes: 0,
    totalBytes: opts.totalBytes,
    lastLine: null,
    message: null,
    startedAt: Date.now(),
    finishedAt: null,
    bytesPerSecond: null,
    etaSeconds: null,
    updatedAt: Date.now(),
    queuePosition: null,
    companionRepos: opts.steps.slice(1).map((step) => step.repo),
  }
  const lines: string[] = []
  let child: ChildProcess | null = null
  let cancelled = false
  let env = opts.env
  let stalled: string | null = null
  let killTimer: ReturnType<typeof setTimeout> | null = null
  const emit = () => opts.onUpdate({ ...state })
  const terminate = () => {
    killGroup(child, 'SIGTERM')
    if (killTimer) clearTimeout(killTimer)
    killTimer = setTimeout(() => killGroup(child, 'SIGKILL'), opts.killGraceMs ?? 3000)
    killTimer.unref?.()
  }
  const inactivity = new PullInactivity(() => {
    if (!child || cancelled || state.state !== 'running') return
    stalled = pullFailure('', env, true)
    inactivity.stop()
    state.message = stalled
    state.phase = 'Stopping stalled download'
    state.bytesPerSecond = state.etaSeconds = null
    emit()
    // Retry is admitted only after close confirms the cache writer has stopped.
    terminate()
  }, opts.inactivityTimer)

  const rate = new TransferRateEstimator()
  let cacheBytes = 0
  let measuring: Promise<void> | null = null
  const measure = (): Promise<void> => {
    if (measuring) return measuring
    measuring = (async () => {
      let received = 0
      for (const step of opts.steps) received += await repoCachedBytes(opts.hub, step.repo)
      cacheBytes = received
      inactivity.sample(received)
      if (state.totalBytes != null) received = Math.min(received, state.totalBytes)
      if (state.state === 'running') {
        const now = Date.now()
        const measuredRate = stalled ? null : rate.sample(received, now)
        const estimate = measuredRate != null && state.totalBytes != null && state.totalBytes > received ? Math.ceil((state.totalBytes - received) / measuredRate) : null
        state.receivedBytes = received
        state.bytesPerSecond = measuredRate
        state.etaSeconds = estimate
        state.updatedAt = now
        emit()
      }
    })().finally(() => { measuring = null })
    return measuring
  }
  const poll = setInterval(() => { void measure() }, opts.pollMs ?? 1000)
  poll.unref?.()

  const runStep = (step: PullStep, index: number) => new Promise<void>((resolve, reject) => {
    state.phase = index === 0 ? `Downloading ${step.repo}` : `Downloading companion ${step.repo}`
    emit()
    const spec = step.revision ? `${step.repo}@${step.revision}` : step.repo
    const proc = spawn(opts.binary.command, [...opts.binary.args, 'pull', spec], { env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] })
    child = proc
    lines.length = 0
    inactivity.start(cacheBytes)
    const onLines = (records: string[]) => {
      let changed = false
      for (const raw of records) {
        const line = redactPullDiagnostic(raw, env).trim().slice(-MAX_LINE_CHARS)
        if (!line) continue
        lines.push(line)
        if (lines.length > KEEP_LINES) lines.shift()
        state.lastLine = line
        changed = true
      }
      if (changed) emit()
    }
    const capture = (stream: Readable | null) => {
      if (!stream) return
      let pending = ''
      let dropping = false
      stream.setEncoding('utf8')
      stream.on('data', (chunk: string) => {
        const records = (pending + chunk).split('\n')
        pending = records.pop()!
        if (dropping && records.length) { records.shift(); dropping = false }
        onLines(records)
        // Drop an oversized partial line whole, so truncation cannot expose half of a credential.
        if (dropping || pending.length > MAX_LINE_CHARS) { pending = ''; dropping = true }
      })
      stream.once('end', () => { if (!dropping) onLines([pending]) })
    }
    capture(proc.stdout)
    capture(proc.stderr)
    proc.once('error', (error) => reject(new Error(`Unable to start tensorfold-native pull: ${redactPullDiagnostic(error.message, env)}`)))
    proc.once('close', (code, signal) => {
      inactivity.stop()
      if (killTimer) clearTimeout(killTimer)
      child = null
      if (cancelled) reject(new Error('cancelled'))
      else if (stalled) reject(new Error(stalled))
      else if (code === 0) resolve()
      else reject(new Error(pullFailure(lines.at(-1) || `tensorfold-native pull exited with ${signal ?? code}.`, env)))
    })
  })

  const done = (async () => {
    try {
      env = await pullEnvironment(opts.env)
      await measure()
      for (const [index, step] of opts.steps.entries()) {
        if (cancelled) throw new Error('cancelled')
        await runStep(step, index)
      }
      await measure()
      if (cancelled) throw new Error('cancelled')
      state.state = 'done'
      state.phase = 'Downloaded'
      if (state.totalBytes != null) state.receivedBytes = state.totalBytes
    } catch (error) {
      state.state = cancelled ? 'cancelled' : 'failed'
      state.phase = cancelled ? 'Cancelled' : 'Failed'
      state.message = cancelled ? 'Download cancelled. Download again to resume.' : (error as Error).message
    } finally {
      inactivity.stop()
      clearInterval(poll)
      if (killTimer) clearTimeout(killTimer)
      state.finishedAt = Date.now()
      state.updatedAt = state.finishedAt
      state.bytesPerSecond = null
      state.etaSeconds = null
      emit()
    }
    return { ...state }
  })()

  return {
    progress: () => ({ ...state }),
    cancel: () => {
      if (state.state !== 'running' || cancelled) return
      cancelled = true
      inactivity.stop()
      terminate()
    },
    done,
  }
}

interface QueueEntry {
  options: PullOptions
  progress: PullProgress
  native: PullHandle | null
  finalize?: (progress: PullProgress) => Promise<PullProgress>
  resolve: (progress: PullProgress) => void
  done: Promise<PullProgress>
}

// A click on Download explicitly consents to this job running after earlier jobs. Cancelling one
// job never cancels other independently queued jobs; cancelAll closes the queue before teardown.
export class PullQueue {
  private readonly pending: QueueEntry[] = []
  private active: QueueEntry | null = null
  private closed = false

  constructor(private readonly limit = 8) {
    if (!Number.isInteger(limit) || limit < 1) throw new Error('Download queue limit must be a positive integer.')
  }

  get busy(): boolean { return this.active != null || this.pending.length > 0 }

  hasRepo(repo: string): boolean {
    return [this.active, ...this.pending].some((entry) => entry?.options.steps.some((step) => step.repo === repo))
  }

  enqueue(options: PullOptions, finalize?: QueueEntry['finalize']): PullHandle {
    if (this.closed) throw new Error('The download queue is closed.')
    if (!options.steps.length) throw new Error('Nothing to pull.')
    const repos = options.steps.map((step) => step.repo)
    if (new Set(repos).size !== repos.length) throw new Error('The download contains a duplicate model or companion.')
    if (repos.some((repo) => this.hasRepo(repo))) throw new Error('This model or its companion is already in the download queue.')
    if (this.pending.length + (this.active ? 1 : 0) >= this.limit) throw new Error(`The download queue is full. Wait for a download or cancel a queued item, then try again.`)
    let resolve!: QueueEntry['resolve']
    const done = new Promise<PullProgress>((complete) => { resolve = complete })
    const entry: QueueEntry = {
      options, finalize, resolve, done, native: null,
      progress: {
        repo: options.steps[0].repo, revision: options.steps[0].revision,
        state: 'queued', phase: 'Queued', receivedBytes: 0, totalBytes: options.totalBytes,
        lastLine: null, message: null, startedAt: Date.now(), finishedAt: null,
        bytesPerSecond: null, etaSeconds: null, updatedAt: Date.now(), queuePosition: null,
        companionRepos: options.steps.slice(1).map((step) => step.repo),
      },
    }
    this.pending.push(entry)
    this.updatePositions()
    // Defer until the caller has registered its handle, so the first event cannot race the map.
    void Promise.resolve().then(() => this.pump())
    return {
      progress: () => ({ ...entry.progress }), done,
      cancel: () => {
        const index = this.pending.indexOf(entry)
        if (index >= 0) {
          this.pending.splice(index, 1)
          this.finish(entry, { ...entry.progress, state: 'cancelled', phase: 'Cancelled', message: 'Removed from the queue. Download again to resume.', finishedAt: Date.now(), queuePosition: null })
          this.updatePositions()
        } else if (entry === this.active && entry.progress.state === 'running') entry.native?.cancel()
        // Once finalization begins, keep the reservation until the durable write completes.
      },
    }
  }

  cancelAll(): void {
    this.closed = true
    const pending = this.pending.splice(0)
    for (const entry of pending) this.finish(entry, { ...entry.progress, state: 'cancelled', phase: 'Cancelled', message: 'Download stopped when TensorFold closed. Download again to resume.', finishedAt: Date.now(), queuePosition: null })
    if (this.active?.progress.state === 'running') this.active.native?.cancel()
  }

  private updatePositions(): void {
    for (const [index, entry] of this.pending.entries()) {
      entry.progress = { ...entry.progress, queuePosition: index + (this.active ? 1 : 0), updatedAt: Date.now() }
      entry.options.onUpdate({ ...entry.progress })
    }
  }

  private finish(entry: QueueEntry, progress: PullProgress): void {
    entry.progress = { ...progress, queuePosition: null, bytesPerSecond: null, etaSeconds: null, updatedAt: Date.now() }
    entry.options.onUpdate({ ...entry.progress })
    entry.resolve({ ...entry.progress })
  }

  private async pump(): Promise<void> {
    if (this.closed || this.active) return
    const entry = this.pending.shift()
    if (!entry) return
    this.active = entry
    this.updatePositions()
    try {
      entry.native = startPull({ ...entry.options, onUpdate: (progress) => {
        if (progress.state !== 'running') return
        entry.progress = { ...progress }
        entry.options.onUpdate({ ...progress })
      } })
      entry.progress = entry.native.progress()
      entry.options.onUpdate({ ...entry.progress })
      let final = await entry.native.done
      if (final.state === 'done' && entry.finalize) {
        entry.progress = { ...final, state: 'verifying', phase: 'Verifying downloaded checkpoint', finishedAt: null, updatedAt: Date.now() }
        entry.options.onUpdate({ ...entry.progress })
        final = await entry.finalize({ ...final })
      }
      this.finish(entry, { ...final, finishedAt: Date.now() })
    } catch (error) {
      this.finish(entry, { ...entry.progress, state: 'failed', phase: 'Failed', message: error instanceof Error ? error.message : 'Download could not complete.', finishedAt: Date.now() })
    } finally {
      this.active = null
      this.updatePositions()
      void this.pump()
    }
  }
}
