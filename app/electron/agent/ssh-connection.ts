// SSH connection management built on `ssh2`. The security-critical decisions
// (host-key trust, private-key encryption detection, fingerprinting) are pure,
// exported functions so they can be unit-tested in isolation. The ssh2 client is
// injected through `SshConnectionDeps.clientFactory`, so the connect/auth/exec
// orchestration is testable with a fake client and no real network.
//
// Secrets (key passphrases, passwords) are obtained through `promptSecret` and
// held only for the lifetime of the connect attempt — never persisted.

import { createHash } from 'node:crypto'
import os from 'node:os'
import { readOpenedTextFile, withFileReadBudget, type WorkspaceReadOptions, type WorkspaceRangeOptions, type WorkspaceTextRange } from './bounded-file-read'
import type { SshHost, SshConnectionState, TrustedSshHostKey } from './types'

export interface SshExecOptions {
  cwd?: string
  env?: Record<string, string>
  pty?: boolean
  /** Cap captured stdout/stderr (bytes). Prevents a huge remote output from OOM-ing the main process. */
  maxBuffer?: number
  /**
   * Kill the remote command and resolve after this many ms if it has not closed. Mirrors the local
   * backend's 120s cap — without it a remote command that never emits 'close' (reads stdin, holds a
   * lock, `tail -f`, `sleep infinity`) would hang the agent turn forever.
   */
  timeoutMs?: number
  /** Abort the remote channel when the owning agent run is cancelled. */
  signal?: AbortSignal
}

/** Default cap for SSH exec output, matching the local backend's cpExec maxBuffer. */
export const SSH_EXEC_MAX_BUFFER = 8_000_000

export interface SshExecResult {
  code: number | null
  signal?: string | null
  stdout: string
  stderr: string
}

// Subset of node's ssh2 SFTPWrapper we depend on, promisified.
export interface SshSftpLike {
  readFile(path: string, options?: WorkspaceReadOptions): Promise<Buffer>
  readFileRange?(path: string, options: WorkspaceRangeOptions): Promise<WorkspaceTextRange>
  writeFile(path: string, data: Buffer | string): Promise<void>
  readdir(path: string): Promise<Array<{ filename: string; attrs: { mode: number; size: number; mtime: number } }>>
  stat(path: string): Promise<{ isDirectory: boolean; isFile: boolean; mode: number; size: number; mtime: number }>
  mkdir(path: string): Promise<void>
  unlink(path: string): Promise<void>
  rename(from: string, to: string): Promise<void>
}

// Subset of the ssh2 Client surface we use. Keeping it minimal lets tests inject
// a fake EventEmitter-shaped client.
export interface SshClientLike {
  on(event: string, cb: (...args: any[]) => void): this
  connect(config: SshClientConnectConfig): void
  exec(command: string, options: Record<string, unknown>, cb: (err: Error | undefined, stream: any) => void): void
  sftp(cb: (err: Error | undefined, sftp: any) => void): void
  end(): void
  /**
   * Open a forwarded TCP channel through this (already-connected) client, originating from
   * srcIP:srcPort and targeting dstIP:dstPort on the remote side. Used to hop through a
   * ProxyJump bastion: the returned channel is passed as the next client's `sock`.
   */
  forwardOut?(srcIP: string, srcPort: number, dstIP: string, dstPort: number, cb: (err: Error | undefined, channel: any) => void): void
}

export interface SshClientConnectConfig {
  host: string
  port: number
  username: string
  privateKey?: string | Buffer
  passphrase?: string
  password?: string
  agent?: string
  tryKeyboard?: boolean
  readyTimeout?: number
  // ssh2 invokes this with the raw server host key so we can pin it.
  hostVerifier?: (key: Buffer, callback: (ok: boolean) => void) => void
  // A duplex stream to use instead of opening a new TCP connection — used for ProxyJump hops
  // (a forwarded channel from an already-connected bastion client).
  sock?: unknown
}

// --- pure, security-critical helpers ---

/** Base64 SHA-256 fingerprint of a server/host public key (OpenSSH style). */
export function fingerprintFromKey(key: Buffer): string {
  return createHash('sha256').update(key).digest('base64').replace(/=+$/, '')
}

export type HostKeyTrust = 'trusted' | 'unknown' | 'changed'

/**
 * Decide whether a presented host key is already trusted, never seen, or
 * conflicts with a pinned key (possible MITM).
 */
export function decideHostKeyTrust(
  trusted: TrustedSshHostKey[],
  host: string,
  port: number,
  fingerprintSha256: string,
): HostKeyTrust {
  const matches = trusted.filter((t) => t.host === host && t.port === port)
  if (!matches.length) return 'unknown'
  if (matches.some((t) => t.fingerprintSha256 === fingerprintSha256)) return 'trusted'
  return 'changed'
}

/**
 * Pin a newly-trusted host key, REPLACING any prior key(s) for the same host:port rather than
 * appending. On key rotation this invalidates the old fingerprint — otherwise a MITM presenting the
 * old (compromised) key would still match `decideHostKeyTrust` and connect silently.
 */
export function upsertTrustedHostKey(keys: TrustedSshHostKey[], key: TrustedSshHostKey): TrustedSshHostKey[] {
  return [...keys.filter((k) => !(k.host === key.host && k.port === key.port)), key]
}

/** Detect whether a private key (PEM, PKCS#8, or OpenSSH format) is passphrase-encrypted. */
export function isEncryptedPrivateKey(text: string): boolean {
  // Classic PEM: "Proc-Type: 4,ENCRYPTED"
  if (/Proc-Type:\s*4,ENCRYPTED/.test(text)) return true
  // PKCS#8 (e.g. `openssl pkcs8 -topk8`) uses a distinct header for encrypted keys; the unencrypted
  // form is "BEGIN PRIVATE KEY" without "ENCRYPTED".
  if (/-----BEGIN ENCRYPTED PRIVATE KEY-----/.test(text)) return true
  const match = text.match(/-----BEGIN OPENSSH PRIVATE KEY-----([\s\S]*?)-----END OPENSSH PRIVATE KEY-----/)
  if (match) {
    try {
      const body = Buffer.from(match[1].replace(/\s+/g, ''), 'base64')
      const magic = 'openssh-key-v1\0'
      if (body.subarray(0, magic.length).toString('binary') === magic) {
        let off = magic.length
        const cipherLen = body.readUInt32BE(off)
        off += 4
        const cipher = body.subarray(off, off + cipherLen).toString('utf8')
        return cipher !== 'none'
      }
    } catch {
      // fall through
    }
  }
  return false
}

/** Parsed `ProxyJump` target: `[user@]host[:port]` (single hop). */
export interface ProxyJumpTarget {
  user?: string
  host: string
  port: number
}

/**
 * Parse a `ProxyJump` directive value into a connectable target. Only the
 * single-hop form is supported — a comma-separated chain (`ProxyJump a,b`) uses
 * just the first hop, which covers the common single-bastion case this fix
 * targets without taking on multi-hop chaining.
 */
export function parseProxyJump(spec: string): ProxyJumpTarget {
  const first = spec.split(',')[0].trim()
  const atIdx = first.lastIndexOf('@')
  const user = atIdx >= 0 ? first.slice(0, atIdx) : undefined
  const hostPort = atIdx >= 0 ? first.slice(atIdx + 1) : first
  const fallback = (host: string): ProxyJumpTarget => ({ user, host, port: 22 })
  const withPort = (host: string, portStr: string): ProxyJumpTarget => {
    const port = Number(portStr)
    return { user, host, port: Number.isInteger(port) && port > 0 ? port : 22 }
  }
  // Bracketed IPv6: `[2001:db8::1]` or `[2001:db8::1]:2222`. Strip the brackets and only treat a
  // trailing `:port` (outside the brackets) as a port.
  if (hostPort.startsWith('[')) {
    const close = hostPort.indexOf(']')
    if (close >= 0) {
      const host = hostPort.slice(1, close)
      const rest = hostPort.slice(close + 1)
      return rest.startsWith(':') ? withPort(host, rest.slice(1)) : fallback(host)
    }
    return fallback(hostPort)
  }
  // A bare IPv6 literal (e.g. `2001:db8::1`) has multiple colons and no port — splitting on the
  // last colon would mangle it. Only treat a single trailing colon as a host:port separator.
  if (hostPort.indexOf(':') !== hostPort.lastIndexOf(':')) {
    return fallback(hostPort)
  }
  const colonIdx = hostPort.lastIndexOf(':')
  if (colonIdx < 0) return fallback(hostPort)
  return withPort(hostPort.slice(0, colonIdx), hostPort.slice(colonIdx + 1))
}

// --- connection manager ---

export interface SshTrustPromptInput {
  hostId: string
  host: string
  port: number
  keyType: string
  fingerprintSha256: string
  changed: boolean
}

export interface SshSecretPromptInput {
  hostId: string
  kind: 'passphrase' | 'password'
  label: string
  retry: boolean
}

export interface SshConnectionDeps {
  clientFactory: () => SshClientLike
  getTrustedKeys: () => TrustedSshHostKey[]
  saveTrustedKey: (key: TrustedSshHostKey) => void | Promise<void>
  /** Ask the user to trust an unknown/changed host key. Resolve true to proceed. */
  promptTrust: (input: SshTrustPromptInput) => Promise<boolean>
  /** Ask the user for a passphrase/password. Resolve null to cancel. */
  promptSecret: (input: SshSecretPromptInput) => Promise<string | null>
  /** Read an identity file's contents (path already tilde-expanded). */
  readIdentityFile: (path: string) => Promise<string | null>
  expandPath: (path: string) => string
  agentSock?: string | null
  emitState: (state: SshConnectionState) => void
  now: () => number
  readyTimeoutMs?: number
  /** False when no UI exists to answer host-key/password prompts (headless server). */
  interactivePrompts?: boolean
}

export class SshConnectionError extends Error {}

export class SshSession {
  constructor(
    readonly host: SshHost,
    private readonly client: SshClientLike,
    private readonly now: () => number,
  ) {}

  exec(command: string, options: SshExecOptions = {}): Promise<SshExecResult> {
    if (options.signal?.aborted) {
      const error = new Error('SSH command was aborted before it started.')
      error.name = 'AbortError'
      return Promise.reject(error)
    }
    const composed = composeRemoteCommand(command, options)
    const maxBuffer = options.maxBuffer ?? SSH_EXEC_MAX_BUFFER
    return new Promise((resolve, reject) => {
      this.client.exec(composed, options.pty ? { pty: true } : {}, (err, stream) => {
        if (err || !stream) return reject(err ?? new SshConnectionError('exec failed'))
        let stdout = ''
        let stderr = ''
        let truncated = false
        let settled = false
        let timer: ReturnType<typeof setTimeout> | undefined
        let onAbort = () => {}
        const finish = (result: SshExecResult) => {
          if (settled) return
          settled = true
          if (timer) clearTimeout(timer)
          options.signal?.removeEventListener('abort', onAbort)
          resolve(result)
        }
        onAbort = () => {
          try { (stream as any).destroy?.() ?? (stream as any).close?.() } catch { /* best-effort */ }
          finish({
            code: null,
            signal: 'SIGINT',
            stdout,
            stderr: stderr ? `${stderr}\n[command cancelled by user]` : '[command cancelled by user]',
          })
        }
        options.signal?.addEventListener('abort', onAbort, { once: true })
        if (options.signal?.aborted) { onAbort(); return }
        // Cap accumulated output so an unbounded remote stream (cat big.log, multi-GB diff) can't
        // grow a JS string until the main process OOMs. Mirrors the local backend's maxBuffer.
        const append = (cur: string, d: Buffer): string => {
          if (truncated || cur.length >= maxBuffer) { truncated = true; return cur }
          const next = cur + d.toString('utf8')
          if (next.length <= maxBuffer) return next
          truncated = true
          try { (stream as any).destroy?.() ?? (stream as any).close?.() } catch { /* best-effort */ }
          return next.slice(0, maxBuffer) + `\n[output truncated at ${maxBuffer} bytes]`
        }
        // Cap wall-clock time so a remote command that never emits 'close' (blocks on stdin/lock,
        // `tail -f`, daemon holding the channel) can't hang the agent turn forever. Destroy the
        // stream and resolve with a timeout result instead of leaving the Promise unsettled.
        if (options.timeoutMs && options.timeoutMs > 0) {
          timer = setTimeout(() => {
            try { (stream as any).destroy?.() ?? (stream as any).close?.() } catch { /* best-effort */ }
            finish({
              code: null,
              signal: 'SIGKILL',
              stdout,
              stderr: stderr ? `${stderr}\n[command timed out after ${options.timeoutMs}ms]` : `[command timed out after ${options.timeoutMs}ms]`,
            })
          }, options.timeoutMs)
        }
        stream.on('data', (d: Buffer) => { stdout = append(stdout, d) })
        stream.stderr?.on('data', (d: Buffer) => { stderr = append(stderr, d) })
        stream.on('close', (code: number | null, signal: string | null) => {
          finish({ code, signal, stdout, stderr })
        })
        stream.on('error', (e: Error) => {
          if (settled) return
          settled = true
          if (timer) clearTimeout(timer)
          options.signal?.removeEventListener('abort', onAbort)
          reject(e)
        })
      })
    })
  }

  sftp(): Promise<SshSftpLike> {
    return new Promise((resolve, reject) => {
      this.client.sftp((err, sftp) => {
        if (err || !sftp) return reject(err ?? new SshConnectionError('sftp unavailable'))
        resolve(wrapSftp(sftp))
      })
    })
  }

  close(): void {
    this.client.end()
  }
}

/** Compose a remote command line with optional cwd + env, safely quoted. */
export function composeRemoteCommand(command: string, options: SshExecOptions): string {
  const parts: string[] = []
  if (options.env) {
    for (const [k, v] of Object.entries(options.env)) {
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(k)) parts.push(`${k}=${shellQuote(v)}`)
    }
  }
  const prefix = parts.length ? `export ${parts.join(' ')}; ` : ''
  const cd = options.cwd ? `cd ${shellQuote(options.cwd)} && ` : ''
  return `${prefix}${cd}${command}`
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

export async function readBoundedSftpFile(sftp: any, filePath: string, options: WorkspaceReadOptions = {}, range?: WorkspaceRangeOptions): Promise<Buffer | WorkspaceTextRange> {
  return withFileReadBudget(options, async (budget) => {
    // SFTP v3 has no per-request cancellation. Tear down its channel on timeout
    // or abort, including a pending open that has not produced a handle yet.
    const abort = () => sftp.destroy?.()
    budget.signal.addEventListener('abort', abort, { once: true })
    try {
      const call = <T>(method: string, ...args: any[]): Promise<T> => new Promise((resolve, reject) => {
        sftp[method](...args, (error: Error | undefined, value: T) => error ? reject(error) : resolve(value))
      })
      const opening = call<Buffer>('open', filePath, 'r')
      opening.then((handle) => { if (budget.signal.aborted) void call('close', handle).catch(() => {}) }, () => {})
      const handle = await budget.wait(opening)
      return await readOpenedTextFile({
        stat: async () => {
          const stat = await call<{ mode: number; size: number }>('fstat', handle)
          return { isFile: (stat.mode & 0o170000) === 0o100000, size: stat.size }
        },
        read: (buffer, length, position) => call<number>('read', handle, buffer, 0, length, position),
        close: () => call<void>('close', handle),
      }, budget, range)
    } finally { budget.signal.removeEventListener('abort', abort) }
  })
}

function wrapSftp(sftp: any): SshSftpLike {
  const call = <T>(fn: string, ...args: any[]): Promise<T> =>
    new Promise((resolve, reject) => {
      sftp[fn](...args, (err: Error | undefined, res: T) => (err ? reject(err) : resolve(res)))
    })
  return {
    readFile: async (p, options) => await readBoundedSftpFile(sftp, p, options) as Buffer,
    readFileRange: async (p, options) => await readBoundedSftpFile(sftp, p, options, options) as WorkspaceTextRange,
    writeFile: (p, data) => call<void>('writeFile', p, data),
    readdir: (p) => call<any[]>('readdir', p).then((list) =>
      list.map((e: any) => ({ filename: e.filename, attrs: { mode: e.attrs.mode, size: e.attrs.size, mtime: e.attrs.mtime } }))),
    stat: (p) => call<any>('stat', p).then((s: any) => ({
      isDirectory: (s.mode & 0o170000) === 0o040000,
      isFile: (s.mode & 0o170000) === 0o100000,
      mode: s.mode, size: s.size, mtime: s.mtime,
    })),
    mkdir: (p) => call<void>('mkdir', p),
    unlink: (p) => call<void>('unlink', p),
    rename: (from, to) => call<void>('rename', from, to),
  }
}

/**
 * Connect to a `ProxyJump` bastion so its transport can carry a forwarded channel to the real
 * target (see `connectSsh`). Reuses the already-resolved auth material (identity/agent) from the
 * target's connect config — the common case is the same identity reaching both hops — and applies
 * the same host-key trust flow as the target, scoped to the bastion's own host:port.
 */
async function connectBastion(
  jump: ProxyJumpTarget,
  targetAuth: SshClientConnectConfig,
  host: SshHost,
  deps: SshConnectionDeps,
): Promise<SshClientLike> {
  const config: SshClientConnectConfig = {
    host: jump.host,
    port: jump.port,
    username: jump.user || targetAuth.username,
    privateKey: targetAuth.privateKey,
    passphrase: targetAuth.passphrase,
    agent: targetAuth.agent,
    tryKeyboard: targetAuth.tryKeyboard,
    readyTimeout: targetAuth.readyTimeout,
  }
  return new Promise<SshClientLike>((resolve, reject) => {
    const client = deps.clientFactory()
    let settled = false
    const fail = (err: Error) => {
      if (settled) return
      settled = true
      reject(err)
    }
    config.hostVerifier = (key, callback) => {
      const fingerprint = fingerprintFromKey(key)
      const trust = decideHostKeyTrust(deps.getTrustedKeys(), jump.host, jump.port, fingerprint)
      if (trust === 'trusted') return callback(true)
      if (deps.interactivePrompts === false) {
        callback(false)
        fail(new SshConnectionError(`SSH host key for ${jump.host}:${jump.port} is not trusted, and interactive prompts are unavailable in headless mode.`))
        return
      }
      deps.promptTrust({
        hostId: host.id,
        host: jump.host,
        port: jump.port,
        keyType: detectKeyType(key),
        fingerprintSha256: fingerprint,
        changed: trust === 'changed',
      }).then(async (ok) => {
        if (!ok) { callback(false); return }
        await deps.saveTrustedKey({
          host: jump.host, port: jump.port, keyType: detectKeyType(key),
          fingerprintSha256: fingerprint, trustedAt: deps.now(),
        })
        callback(true)
      }).catch((e) => { callback(false); fail(e instanceof Error ? e : new SshConnectionError(String(e))) })
    }
    client.on('ready', () => {
      if (settled) return
      settled = true
      resolve(client)
    })
    client.on('error', (err: Error) => fail(err instanceof Error ? err : new SshConnectionError(String(err))))
    try {
      client.connect(config)
    } catch (e) {
      fail(e instanceof Error ? e : new SshConnectionError(String(e)))
    }
  })
}

/**
 * Establish an SSH session: verify the host key (prompting for trust on
 * unknown/changed keys), select an auth method, and resolve a SshSession once
 * the transport is ready.
 */
export async function connectSsh(host: SshHost, deps: SshConnectionDeps): Promise<SshSession> {
  const emit = (phase: SshConnectionState['phase'], extra: Partial<SshConnectionState> = {}) =>
    deps.emitState({ hostId: host.id, phase, ...extra })

  emit('connecting')

  const username = host.user || os.userInfo().username
  const config: SshClientConnectConfig = {
    host: host.hostName,
    port: host.port || 22,
    username,
    readyTimeout: deps.readyTimeoutMs ?? 30_000,
  }

  // Resolve auth material before connecting (ssh2 needs the passphrase up front).
  let usedKey = false
  for (const rawPath of host.identityFiles) {
    const path = deps.expandPath(rawPath)
    const contents = await deps.readIdentityFile(path)
    if (contents == null) continue
    config.privateKey = contents
    if (isEncryptedPrivateKey(contents)) {
      emit('awaiting_password')
      if (deps.interactivePrompts === false) {
        throw new SshConnectionError(`SSH passphrase is required for ${rawPath}, but interactive prompts are unavailable in headless mode.`)
      }
      const passphrase = await deps.promptSecret({
        hostId: host.id, kind: 'passphrase', label: `Passphrase for ${rawPath}`, retry: false,
      })
      if (passphrase == null) throw new SshConnectionError('Passphrase entry cancelled')
      config.passphrase = passphrase
    }
    usedKey = true
    break
  }
  if (!usedKey) {
    // Fall back to ssh-agent, then keyboard-interactive/password.
    if (deps.agentSock) config.agent = deps.agentSock
    config.tryKeyboard = true
  }

  // ProxyJump: dial the bastion first and hand the target client a forwarded channel as its
  // `sock` instead of letting it open a direct TCP connection. Without this, a configured bastion
  // is silently bypassed and TensorFold connects straight to the target.
  let bastionClient: SshClientLike | null = null
  if (host.proxyJump) {
    const jump = parseProxyJump(host.proxyJump)
    bastionClient = await connectBastion(jump, config, host, deps)
    try {
      if (!bastionClient.forwardOut) {
        throw new SshConnectionError('ProxyJump unsupported: client cannot forward channels')
      }
      config.sock = await new Promise<unknown>((resolve, reject) => {
        bastionClient!.forwardOut!('127.0.0.1', 0, host.hostName, config.port, (err, channel) => {
          if (err || !channel) reject(err ?? new SshConnectionError('ProxyJump forwardOut failed'))
          else resolve(channel)
        })
      })
    } catch (e) {
      // The bastion transport only exists to carry the forwarded channel — if forwardOut fails
      // (channel refused, target unreachable from the bastion) or the client can't forward, close
      // it instead of leaking a fully-authenticated SSH connection on every failed/retried connect.
      try { bastionClient.end() } catch { /* best-effort */ }
      throw e instanceof Error ? e : new SshConnectionError(String(e))
    }
  }

  return new Promise<SshSession>((resolve, reject) => {
    const client = deps.clientFactory()
    let settled = false
    const fail = (err: Error) => {
      if (settled) return
      settled = true
      // The bastion leg only exists to carry this target connection — if the target never comes up,
      // don't leak the bastion's transport.
      try { bastionClient?.end() } catch { /* best-effort */ }
      emit('error', { lastError: err.message })
      reject(err)
    }

    config.hostVerifier = (key, callback) => {
      const fingerprint = fingerprintFromKey(key)
      const trust = decideHostKeyTrust(deps.getTrustedKeys(), host.hostName, config.port, fingerprint)
      if (trust === 'trusted') return callback(true)
      if (deps.interactivePrompts === false) {
        callback(false)
        fail(new SshConnectionError(`SSH host key for ${host.hostName}:${config.port} is not trusted, and interactive prompts are unavailable in headless mode.`))
        return
      }
      emit('awaiting_trust')
      deps.promptTrust({
        hostId: host.id,
        host: host.hostName,
        port: config.port,
        keyType: detectKeyType(key),
        fingerprintSha256: fingerprint,
        changed: trust === 'changed',
      }).then(async (ok) => {
        if (!ok) { callback(false); return }
        await deps.saveTrustedKey({
          host: host.hostName, port: config.port, keyType: detectKeyType(key),
          fingerprintSha256: fingerprint, trustedAt: deps.now(),
        })
        callback(true)
      }).catch((e) => { callback(false); fail(e instanceof Error ? e : new SshConnectionError(String(e))) })
    }

    client.on('ready', () => {
      if (settled) return
      settled = true
      emit('connected', { connectedAt: deps.now() })
      resolve(new SshSession(host, client, deps.now))
    })
    client.on('error', (err: Error) => fail(err instanceof Error ? err : new SshConnectionError(String(err))))
    // After the session is live, a transport close/end means the connection dropped — surface it so
    // the manager invalidates the dead session and reconnects on next use (instead of handing out a
    // backend that silently hangs).
    client.on('close', () => {
      if (settled) emit('disconnected', { droppedAt: deps.now() })
      try { bastionClient?.end() } catch { /* best-effort */ }
    })
    client.on('keyboard-interactive', (_name: string, _instr: string, _lang: string, prompts: Array<{ prompt: string; echo: boolean }>, finish: (responses: string[]) => void) => {
      ;(async () => {
        if (deps.interactivePrompts === false) {
          fail(new SshConnectionError('SSH password is required, but interactive prompts are unavailable in headless mode. Configure a usable key or SSH agent.'))
          return
        }
        const responses: string[] = []
        for (const p of prompts) {
          emit('awaiting_password')
          const answer = await deps.promptSecret({ hostId: host.id, kind: 'password', label: p.prompt || 'Password', retry: false })
          if (answer == null) return fail(new SshConnectionError('Password entry cancelled'))
          responses.push(answer)
        }
        finish(responses)
      })().catch((error) => fail(error instanceof Error ? error : new SshConnectionError(String(error))))
    })

    emit('authenticating')
    try {
      client.connect(config)
    } catch (e) {
      fail(e instanceof Error ? e : new SshConnectionError(String(e)))
    }
  })
}

function detectKeyType(key: Buffer): string {
  // OpenSSH keys begin with a length-prefixed type string, e.g. "ssh-ed25519".
  try {
    const len = key.readUInt32BE(0)
    const type = key.subarray(4, 4 + len).toString('utf8')
    if (/^[a-z0-9@.-]+$/.test(type)) return type
  } catch {
    // ignore
  }
  return 'unknown'
}

/** Production ssh2 client factory (lazy-require keeps ssh2 out of pure tests). */
export function createSsh2Client(): SshClientLike {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Client } = require('ssh2')
  return new Client() as SshClientLike
}
