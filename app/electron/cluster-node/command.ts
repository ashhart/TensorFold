import { spawn, type ChildProcess } from 'node:child_process'

export interface CliOutput { code: number; stdout: string; stderr: string }
export function terminateOwned(child: ChildProcess, signal: NodeJS.Signals): void {
  if (!child.pid || child.exitCode !== null || child.signalCode !== null) return
  try { if (process.platform === 'win32') child.kill(signal); else process.kill(-child.pid, signal) } catch { /* Exit is confirmed separately through close. */ }
}
/** Arguments are passed directly, never through a shell; cancellation waits for physical process close. */
export async function nativeCli(binary: string, args: string[], env: NodeJS.ProcessEnv, signal: AbortSignal, timeoutMs = 30_000): Promise<CliOutput> {
  if (signal.aborted) throw signal.reason ?? new Error('Node command cancelled.')
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, { env, detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'], shell: false })
    let stdout = '', stderr = '', failure: Error | null = null, hard: ReturnType<typeof setTimeout> | null = null
    const cancel = (error: Error) => { failure ??= error; terminateOwned(child, 'SIGTERM'); hard ??= setTimeout(() => terminateOwned(child, 'SIGKILL'), 2_000) }
    const abort = () => cancel(new Error('Node command cancelled before physical completion.'))
    const timer = setTimeout(() => cancel(new Error('The native engine command timed out.')), timeoutMs)
    const append = (part: Buffer, stream: 'stdout' | 'stderr') => {
      if (stream === 'stdout') stdout += part.toString('utf8'); else stderr += part.toString('utf8')
      if (Buffer.byteLength(stdout) > 4 * 1024 * 1024 || Buffer.byteLength(stderr) > 1024 * 1024) cancel(new Error('The native engine exceeded its command output limit.'))
    }
    child.stdout?.on('data', part => append(part, 'stdout')); child.stderr?.on('data', part => append(part, 'stderr'))
    signal.addEventListener('abort', abort, { once: true })
    child.once('error', () => { failure ??= new Error('The configured native engine could not be executed.') })
    child.once('close', code => { clearTimeout(timer); if (hard) clearTimeout(hard); signal.removeEventListener('abort', abort); if (failure) reject(failure); else resolve({ code: code ?? 1, stdout, stderr }) })
    if (signal.aborted) abort()
  })
}
