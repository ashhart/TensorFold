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
    } catch { /* fall back to the direct child */ }
  }
  try { proc.kill(value) } catch { /* already gone */ }
}

/**
 * Terminate a CLI plus descendants, escalating after a bounded grace period.
 * Callers using processGroup must spawn the child detached on POSIX.
 */
export function terminateChildProcess(
  proc: Pick<ChildProcess, 'pid' | 'exitCode' | 'kill' | 'once'>,
  options: { processGroup?: boolean; graceMs?: number } = {},
): void {
  if (proc.exitCode != null || terminating.has(proc as object)) return
  terminating.add(proc as object)
  const processGroup = options.processGroup === true
  signal(proc, 'SIGTERM', processGroup)
  const timer = setTimeout(() => {
    // For a process group, always make the best-effort SIGKILL pass. The group
    // leader may have exited on SIGTERM while a descendant ignored it; in that
    // case ChildProcess.exitCode no longer tells us whether the group is gone.
    if (processGroup || proc.exitCode == null) signal(proc, 'SIGKILL', processGroup)
    terminating.delete(proc as object)
  }, Math.max(1, options.graceMs ?? CHILD_PROCESS_KILL_GRACE_MS))
  timer.unref?.()
  proc.once('close', () => {
    // A direct child closing proves there is nothing left to reap. A process
    // group closing does not: descendants can outlive their leader, so retain
    // the bounded escalation timer for the final group-wide SIGKILL.
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
    } catch { /* fall back to the PTY leader */ }
  }
  try {
    if (process.platform === 'win32') pty.kill()
    else pty.kill(value)
  } catch { /* already gone */ }
}

/**
 * Terminate a PTY leader plus its POSIX process group. node-pty's `kill()` only
 * signals the leader PID, so an agent CLI or shell child that ignores the
 * leader's shutdown can otherwise outlive a closed terminal.
 */
export function terminatePtyProcess(
  pty: PtyProcess,
  options: { graceMs?: number; force?: boolean } = {},
): void {
  if (terminating.has(pty as object)) return
  terminating.add(pty as object)

  if (options.force) {
    signalPty(pty, 'SIGKILL')
    terminating.delete(pty as object)
    return
  }

  signalPty(pty, 'SIGTERM')
  const timer = setTimeout(() => {
    // Keep the final group-wide pass even if the PTY leader has already
    // exited: a descendant may have ignored SIGTERM.
    signalPty(pty, 'SIGKILL')
    terminating.delete(pty as object)
  }, Math.max(1, options.graceMs ?? CHILD_PROCESS_KILL_GRACE_MS))
  timer.unref?.()
}
