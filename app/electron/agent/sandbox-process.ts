import { spawn } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import os from 'node:os'
import { StringDecoder } from 'node:string_decoder'
import { commandShell } from './cli-path'
import { terminateChildProcess } from './child-process-lifecycle'
import type { SandboxProfileId, SandboxedCommandResult } from './sandbox-types'

export function runSandboxProcess(argv: string[], opts: { cwd: string; timeoutMs?: number; maxBuffer?: number; sandboxed: boolean; profile: SandboxedCommandResult['profile']; implementation: SandboxedCommandResult['implementation']; signal?: AbortSignal; privateTemp?: string; env?: NodeJS.ProcessEnv; cancelFile?: string }): Promise<SandboxedCommandResult> {
  return new Promise((resolve) => {
    const [cmd, ...args] = argv
    const child = spawn(cmd, args, {
      cwd: opts.cwd,
      env: opts.env ?? { ...sandboxCommandEnv(opts.cwd), ...(opts.privateTemp ? { TMPDIR: opts.privateTemp } : {}) },
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
    })
    const started = Date.now()
    const maxBuffer = opts.maxBuffer ?? 8 * 1024 * 1024
    let stdout = ''
    let stderr = ''
    let killed = false
    let aborted = false
    let overflowed = false
    let stopTimer: ReturnType<typeof setTimeout> | undefined
    const terminate = () => {
      if (opts.cancelFile && !stopTimer) {
        try { writeFileSync(opts.cancelFile, 'cancel', { flag: 'wx' }) } catch {}
        stopTimer = setTimeout(() => terminateChildProcess(child, { processGroup: false }), 5000)
        stopTimer.unref?.()
      } else if (!opts.cancelFile) terminateChildProcess(child, { processGroup: process.platform !== 'win32', graceMs: 1000 })
    }
    const timer = setTimeout(() => {
      killed = true
      terminate()
    }, opts.timeoutMs ?? 120_000)
    const onAbort = () => {
      if (aborted) return
      aborted = true
      clearTimeout(timer)
      terminate()
    }
    opts.signal?.addEventListener('abort', onAbort, { once: true })
    if (opts.signal?.aborted) onAbort()
    // Preserve UTF-8 characters split between stream chunks.
    const outDecoder = new StringDecoder('utf8')
    const errDecoder = new StringDecoder('utf8')
    const append = (kind: 'stdout' | 'stderr', chunk: Buffer) => {
      const text = (kind === 'stdout' ? outDecoder : errDecoder).write(chunk)
      if (kind === 'stdout') stdout = (stdout + text).slice(0, maxBuffer)
      else stderr = (stderr + text).slice(0, maxBuffer)
      if (stdout.length + stderr.length >= maxBuffer && !overflowed) {
        overflowed = true
        // Use the same bounded shutdown path for overflow, cancellation and timeout.
        clearTimeout(timer)
        terminate()
      }
    }
    child.stdout.on('data', (chunk) => append('stdout', chunk))
    child.stderr.on('data', (chunk) => append('stderr', chunk))
    child.on('error', (error) => {
      clearTimeout(timer)
      clearTimeout(stopTimer)
      opts.signal?.removeEventListener('abort', onAbort)
      resolve({ code: 127, stdout, stderr: error.message, sandboxed: opts.sandboxed, profile: opts.profile, implementation: opts.implementation })
    })
    child.on('close', (code, signal) => {
      clearTimeout(timer)
      clearTimeout(stopTimer)
      opts.signal?.removeEventListener('abort', onAbort)
      const timed = killed && !aborted ? `Timed out after ${Math.round((Date.now() - started) / 1000)}s.` : ''
      const cancelled = aborted ? 'Cancelled by the user.' : ''
      const sig = signal ? `Terminated by ${signal}.` : ''
      resolve({
        code: aborted ? 130 : killed ? 124 : overflowed ? 126 : code ?? 1,
        stdout,
        stderr: [stderr.trim(), cancelled, timed, sig].filter(Boolean).join('\n'),
        sandboxed: opts.sandboxed,
        profile: opts.profile,
        implementation: opts.implementation,
      })
    })
  })
}

export function sandboxCommandEnv(cwd: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    PATH: process.env.PATH || '/usr/bin:/bin:/usr/sbin:/sbin',
    SHELL: commandShell(),
    HOME: cwd,
    TMPDIR: process.env.TMPDIR || os.tmpdir(),
    LANG: process.env.LANG || 'C.UTF-8',
    LC_ALL: process.env.LC_ALL || process.env.LANG || 'C.UTF-8',
    GIT_OPTIONAL_LOCKS: '0',
  }
  return env
}
