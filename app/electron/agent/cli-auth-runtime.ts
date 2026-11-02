import type { ChildProcess } from 'node:child_process'
import { spawnCliProcess as spawn } from './cli-process'
import type { CliAuthCheck, CliAuthTransport, CliLoginResult, Provider } from './types'
import { cliAuthCommands, parseAuthStatus } from './cli-auth'
import { resolveCliBinary } from './cli-path'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'

function isCliTransport(t?: string): t is CliAuthTransport {
  return t === 'codex-cli' || t === 'claude-cli'
}

export function cliBinaryNames(transport: CliAuthTransport): string[] {
  return transport === 'codex-cli' ? ['codex'] : ['claude']
}

export function cliAuthBinary(provider: Provider, opts?: { dirs?: string[]; exists?: (p: string) => boolean }): string {
  const names = cliBinaryNames(provider.transport === 'codex-cli' ? 'codex-cli' : 'claude-cli')
  return resolveCliBinary(provider.cliPath, names, opts ?? {})
}

export async function checkCliAuth(provider: Provider): Promise<CliAuthCheck> {
  const transport = provider.transport
  if (!isCliTransport(transport)) return { loggedIn: false, binaryPresent: false, message: 'Not a CLI provider.' }
  const bin = cliAuthBinary(provider)
  const { statusArgs } = cliAuthCommands(transport)
  return await new Promise<CliAuthCheck>((resolve) => {
    let settled = false
    const finish = (r: CliAuthCheck) => { if (!settled) { settled = true; clearTimeout(timer); resolve(r) } }
    const proc = spawn(bin, statusArgs, {
      detached: detachedProcessGroup(),
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, ...(provider.cliEnv ?? {}) },
    })
    const timer = setTimeout(() => {
      terminateChildProcess(proc, { processGroup: true })
      finish({ loggedIn: false, binaryPresent: true, message: 'Auth status check timed out.' })
    }, 15_000)
    let out = ''
    let err = ''
    proc.stdout.on('data', (d) => { out += String(d) })
    proc.stderr.on('data', (d) => { err += String(d) })
    proc.on('error', (e: NodeJS.ErrnoException) => {
      const missing = e.code === 'ENOENT'
      finish({ loggedIn: false, binaryPresent: !missing, message: missing ? `${bin} not found on PATH` : e.message })
    })
    proc.on('close', (code) => {
      finish({ ...parseAuthStatus(transport, out || err, code ?? 0), binaryPresent: true })
    })
  })
}

// In-flight OAuth logins, keyed by provider id, so a Cancel button can kill them.
const activeLogins = new Map<string, ChildProcess>()

export interface StartLoginOptions {
  signal?: AbortSignal
  timeoutMs?: number
}

export async function startCliLogin(provider: Provider, opts: StartLoginOptions = {}): Promise<CliLoginResult> {
  const transport = provider.transport
  if (!isCliTransport(transport)) return { ok: false, loggedIn: false, message: 'Not a CLI provider.' }
  if (opts.signal?.aborted) return { ok: false, loggedIn: false, message: 'Sign-in cancelled.' }
  const bin = cliAuthBinary(provider)
  const { loginArgs } = cliAuthCommands(transport)

  return await new Promise<CliLoginResult>((resolve) => {
    let settled = false
    let endReason: string | undefined
    const proc = spawn(bin, loginArgs, {
      detached: detachedProcessGroup(),
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, ...(provider.cliEnv ?? {}) },
    })
    activeLogins.set(provider.id, proc)

    let out = ''
    const kill = (reason?: string) => {
      if (reason && !endReason) endReason = reason
      terminateChildProcess(proc, { processGroup: true })
    }
    const onAbort = () => kill('Sign-in cancelled.')
    opts.signal?.addEventListener('abort', onAbort, { once: true })
    const timer = setTimeout(() => kill('Sign-in timed out.'), opts.timeoutMs ?? 300_000)

    const finish = async (failMessage?: string) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      activeLogins.delete(provider.id)
      opts.signal?.removeEventListener('abort', onAbort)
      const status = await checkCliAuth(provider).catch(() => null)
      const loggedIn = !!status?.loggedIn
      resolve({
        ok: loggedIn,
        loggedIn,
        detail: status?.detail,
        message: loggedIn ? undefined : (failMessage || out.slice(-400).trim() || 'Sign-in did not complete.'),
      })
    }
    proc.stdout.on('data', (d) => { out = (out + String(d)).slice(-4000) })
    proc.stderr.on('data', (d) => { out = (out + String(d)).slice(-4000) })
    proc.on('error', (e: NodeJS.ErrnoException) => finish(e.code === 'ENOENT' ? `${bin} not found on PATH` : e.message))
    proc.on('close', (code) => finish(endReason ?? (code === 0 ? undefined : `Sign-in exited with code ${code}`)))
  })
}

export function cancelCliLogin(providerId: string): boolean {
  const proc = activeLogins.get(providerId)
  if (!proc) return false
  terminateChildProcess(proc, { processGroup: true })
  activeLogins.delete(providerId)
  return true
}
