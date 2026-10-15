import { spawn, type ChildProcess } from 'node:child_process'
import type { PullProgress } from '../types'
import { repoCachedBytes, type EngineBinary } from './paths'

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
  onUpdate: (progress: PullProgress) => void
}

export interface PullHandle {
  progress(): PullProgress
  cancel(): void
  done: Promise<PullProgress>
}

const KEEP_LINES = 40

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
  }
  const lines: string[] = []
  let child: ChildProcess | null = null
  let cancelled = false
  let killTimer: ReturnType<typeof setTimeout> | null = null
  const emit = () => opts.onUpdate({ ...state })

  const measure = async () => {
    let received = 0
    for (const step of opts.steps) received += await repoCachedBytes(opts.hub, step.repo)
    if (state.totalBytes != null) received = Math.min(received, state.totalBytes)
    if (received !== state.receivedBytes && state.state === 'running') {
      state.receivedBytes = received
      emit()
    }
  }
  const poll = setInterval(() => { void measure() }, opts.pollMs ?? 1000)
  poll.unref?.()

  const runStep = (step: PullStep, index: number) => new Promise<void>((resolve, reject) => {
    state.phase = index === 0 ? `Downloading ${step.repo}` : `Downloading companion ${step.repo}`
    emit()
    const spec = step.revision ? `${step.repo}@${step.revision}` : step.repo
    const proc = spawn(opts.binary.command, [...opts.binary.args, 'pull', spec], { env: opts.env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] })
    child = proc
    const onText = (chunk: Buffer) => {
      for (const raw of chunk.toString('utf8').split(/\r?\n/)) {
        const line = raw.trimEnd()
        if (!line.trim()) continue
        lines.push(line.trim())
        if (lines.length > KEEP_LINES) lines.shift()
        state.lastLine = line.trim()
      }
      emit()
    }
    proc.stdout?.on('data', onText)
    proc.stderr?.on('data', onText)
    proc.once('error', (error) => reject(new Error(`Could not start tensorfold-native pull: ${error.message}`)))
    proc.once('close', (code, signal) => {
      child = null
      if (cancelled) reject(new Error('cancelled'))
      else if (code === 0) resolve()
      else reject(new Error(lines.at(-1) || `tensorfold-native pull exited with ${signal ?? code}.`))
    })
  })

  const done = (async () => {
    try {
      for (const [index, step] of opts.steps.entries()) {
        if (cancelled) throw new Error('cancelled')
        await runStep(step, index)
      }
      await measure()
      state.state = 'done'
      state.phase = 'Downloaded'
      if (state.totalBytes != null) state.receivedBytes = state.totalBytes
    } catch (error) {
      state.state = cancelled ? 'cancelled' : 'failed'
      state.phase = cancelled ? 'Cancelled' : 'Failed'
      state.message = cancelled ? 'Download cancelled. Download again to resume.' : (error as Error).message
    } finally {
      clearInterval(poll)
      if (killTimer) clearTimeout(killTimer)
      state.finishedAt = Date.now()
      emit()
    }
    return { ...state }
  })()

  return {
    progress: () => ({ ...state }),
    cancel: () => {
      if (state.state !== 'running' || cancelled) return
      cancelled = true
      killGroup(child, 'SIGTERM')
      killTimer = setTimeout(() => killGroup(child, 'SIGKILL'), opts.killGraceMs ?? 3000)
      killTimer.unref?.()
    },
    done,
  }
}
