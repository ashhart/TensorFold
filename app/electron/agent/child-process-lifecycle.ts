import { stopWindowsProcessTree } from './windows-process-tree'
import type { ChildProcess } from 'node:child_process'

export const CHILD_PROCESS_KILL_GRACE_MS = 1_500

const terminating = new WeakSet<object>()

export interface PtyProcess {
  readonly pid: number
  kill(signal?: string): void
}

function signal(proc: Pick<ChildProcess, 'pid' | 'kill'>, value: NodeJS.Signals, processGroup: boolean): void {
  if (processGroup && process.platform !== 'win32' && proc.pid) {
    try {
      process.kill(-proc.pid, value)
      return
    } catch { }
  }
  try { proc.kill(value) } catch { }
}

export function terminateChildProcess(
  proc: Pick<ChildProcess, 'pid' | 'exitCode' | 'kill' | 'once'>,
  options: { processGroup?: boolean; graceMs?: number } = {},
): void {
  if (proc.exitCode != null || terminating.has(proc as object)) return
  terminating.add(proc as object)
  if (process.platform === 'win32' && proc.pid) {
    stopWindowsProcessTree(proc.pid, () => { try { proc.kill('SIGKILL') } catch { } })
    proc.once('close', () => terminating.delete(proc as object))
    return
  }
  const processGroup = options.processGroup === true
  signal(proc, 'SIGTERM', processGroup)
  const timer = setTimeout(() => {

    if (processGroup || proc.exitCode == null) signal(proc, 'SIGKILL', processGroup)
    terminating.delete(proc as object)
  }, Math.max(1, options.graceMs ?? CHILD_PROCESS_KILL_GRACE_MS))
  timer.unref?.()
  proc.once('close', () => {

    if (!processGroup) {
      clearTimeout(timer)
      terminating.delete(proc as object)
    }
  })
}

export function detachedProcessGroup(): boolean {
  return process.platform !== 'win32'
}

function signalPty(pty: PtyProcess, value: NodeJS.Signals): void {
  if (process.platform !== 'win32' && pty.pid > 0) {
    try {
      process.kill(-pty.pid, value)
      return
    } catch { }
  }
  try {
    if (process.platform === 'win32') pty.kill()
    else pty.kill(value)
  } catch { }
}

export function terminatePtyProcess(
  pty: PtyProcess,
  options: { graceMs?: number; force?: boolean } = {},
): void {
  if (terminating.has(pty as object)) return
  terminating.add(pty as object)
  if (process.platform === 'win32') {
    stopWindowsProcessTree(pty.pid, () => { try { pty.kill() } catch { } })
    return
  }

  if (options.force) {
    signalPty(pty, 'SIGKILL')
    terminating.delete(pty as object)
    return
  }

  signalPty(pty, 'SIGTERM')
  const timer = setTimeout(() => {

    signalPty(pty, 'SIGKILL')
    terminating.delete(pty as object)
  }, Math.max(1, options.graceMs ?? CHILD_PROCESS_KILL_GRACE_MS))
  timer.unref?.()
}
