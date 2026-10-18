// Main-process SSH manager. Owns live SSH sessions, bridges ssh2's passphrase /
// password / host-key prompts to the renderer over IPC, persists newly trusted
// host keys, and hands the agent loop a ready WorkspaceBackend for a connected
// host. The pure helpers (config→host, source merge) are exported for tests.

import { promises as fs } from 'node:fs'
import os from 'node:os'
import { randomUUID } from 'node:crypto'
import {
  connectSsh,
  createSsh2Client,
  SshSession,
  type SshClientLike,
  type SshConnectionDeps,
} from '../../agent/ssh-connection'
import { SshBackend, type WorkspaceBackend } from '../../agent/workspace-backend'
import { listSshConfigHosts, resolveSshHostSettings, expandTilde } from '../../agent/ssh-config'
import type {
  SshHost,
  SshConnectionState,
  TrustedSshHostKey,
  WorkspaceLocation,
} from '../../agent/types'

// IPC channels the renderer subscribes to.
export const SSH_STATUS_CHANNEL = 'ssh:status'
export const SSH_SECRET_PROMPT_CHANNEL = 'ssh:secret-prompt'
export const SSH_TRUST_PROMPT_CHANNEL = 'ssh:trust-prompt'
export const SSH_PROMPT_TIMEOUT_MS = 120_000

/** Convert ~/.ssh/config aliases into host-library entries (origin 'config'). */
export function configEntriesToHosts(configText: string): SshHost[] {
  return listSshConfigHosts(configText).map((alias) => {
    const s = resolveSshHostSettings(configText, alias)
    return {
      id: `config:${alias}`,
      label: alias,
      configAlias: alias,
      hostName: s.hostName,
      user: s.user,
      port: s.port,
      identityFiles: s.identityFiles,
      proxyJump: s.proxyJump,
      forwardAgent: s.forwardAgent,
      lastRemoteDir: null,
      platform: null,
      origin: 'config',
      createdAt: 0,
      updatedAt: 0,
    }
  })
}

/** Merge config-derived hosts with the in-app library; library wins per alias. */
export function mergeHostSources(configHosts: SshHost[], libraryHosts: SshHost[]): SshHost[] {
  const aliasKey = (h: SshHost) => h.configAlias || h.label
  const have = new Set(libraryHosts.map(aliasKey))
  return [...libraryHosts, ...configHosts.filter((c) => !have.has(aliasKey(c)))]
}

export interface SshManagerDeps {
  getHosts: () => SshHost[]
  getTrustedKeys: () => TrustedSshHostKey[]
  saveTrustedKey: (key: TrustedSshHostKey) => Promise<void> | void
  sendEvent: (channel: string, payload: unknown) => void
  onStateChange?: (state: SshConnectionState) => void
  now?: () => number
  clientFactory?: () => SshClientLike
  readIdentityFile?: (path: string) => Promise<string | null>
  agentSock?: () => string | null | undefined
  homeDir?: () => string
  sshConfigPath?: () => string
  /** Whether renderer-backed trust/password prompts can currently be answered. */
  interactivePrompts?: () => boolean
  /** Bound renderer-backed prompts so a missing/hidden window cannot park a connection forever. */
  promptTimeoutMs?: number
}

interface ActiveSession {
  session: SshSession
  backend: SshBackend
  state: SshConnectionState
}

interface PendingPrompt<T> {
  resolve(value: T): void
  reject(error: Error): void
  timer: ReturnType<typeof setTimeout>
}

export class SshManager {
  private sessions = new Map<string, ActiveSession>()
  private connecting = new Map<string, Promise<SshBackend>>()
  private pendingSecrets = new Map<string, PendingPrompt<string | null>>()
  private pendingTrust = new Map<string, PendingPrompt<boolean>>()
  private states = new Map<string, SshConnectionState>()

  constructor(private readonly deps: SshManagerDeps) {}

  private now() { return this.deps.now ? this.deps.now() : Date.now() }
  private home() { return this.deps.homeDir ? this.deps.homeDir() : os.homedir() }
  private configPath() { return this.deps.sshConfigPath ? this.deps.sshConfigPath() : `${this.home()}/.ssh/config` }
  private promptTimeoutMs() {
    const configured = this.deps.promptTimeoutMs
    return configured === undefined
      ? SSH_PROMPT_TIMEOUT_MS
      : Math.max(1, Math.min(10 * 60_000, Math.floor(configured)))
  }

  private takePending<T>(pending: Map<string, PendingPrompt<T>>, promptId: string): PendingPrompt<T> | null {
    const entry = pending.get(promptId)
    if (!entry) return null
    pending.delete(promptId)
    clearTimeout(entry.timer)
    return entry
  }

  async listConfigHosts(): Promise<SshHost[]> {
    try {
      const text = await fs.readFile(this.configPath(), 'utf8')
      return configEntriesToHosts(text)
    } catch {
      return []
    }
  }

  async listAllHosts(): Promise<SshHost[]> {
    return mergeHostSources(await this.listConfigHosts(), this.deps.getHosts())
  }

  async resolveHost(hostId: string): Promise<SshHost | null> {
    const fromLibrary = this.deps.getHosts().find((h) => h.id === hostId)
    if (fromLibrary) return fromLibrary
    return (await this.listConfigHosts()).find((h) => h.id === hostId) ?? null
  }

  status(hostId: string): SshConnectionState {
    return this.states.get(hostId) ?? { hostId, phase: 'idle' }
  }

  backendForWorkspace(ws: WorkspaceLocation | null | undefined): WorkspaceBackend | undefined {
    if (!ws || ws.kind !== 'ssh' || !ws.hostId) return undefined
    return this.sessions.get(ws.hostId)?.backend
  }

  /** Resolve a renderer's answer to a passphrase/password prompt. */
  resolveSecret(promptId: string, value: string | null) {
    this.takePending(this.pendingSecrets, promptId)?.resolve(value)
  }

  /** Resolve a renderer's host-key trust decision. */
  resolveTrust(promptId: string, ok: boolean) {
    this.takePending(this.pendingTrust, promptId)?.resolve(ok)
  }

  /** Reject every renderer-backed prompt when the renderer lifecycle ends. */
  cancelPendingPrompts(reason = 'SSH prompt cancelled because the app window closed.') {
    const error = new Error(reason)
    for (const promptId of [...this.pendingTrust.keys()]) this.takePending(this.pendingTrust, promptId)?.reject(error)
    for (const promptId of [...this.pendingSecrets.keys()]) this.takePending(this.pendingSecrets, promptId)?.reject(error)
  }

  disconnect(hostId: string) {
    const active = this.sessions.get(hostId)
    if (active) { try { active.session.close() } catch { /* already closed */ } }
    this.sessions.delete(hostId)
    this.setState({ hostId, phase: 'disconnected' })
  }

  private setState(state: SshConnectionState) {
    this.states.set(state.hostId, state)
    const active = this.sessions.get(state.hostId)
    if (active) active.state = state
    // A dropped/errored transport must invalidate the cached session, or backendForWorkspace keeps
    // handing out a dead backend that hangs. ensureConnected then reconnects on next use.
    if (state.phase === 'disconnected' || state.phase === 'error') this.sessions.delete(state.hostId)
    this.deps.sendEvent(SSH_STATUS_CHANNEL, state)
    try { this.deps.onStateChange?.(state) } catch { /* observers cannot break SSH state */ }
  }

  async ensureConnected(hostId: string): Promise<SshBackend> {
    const existing = this.sessions.get(hostId)
    if (existing) return existing.backend
    const inFlight = this.connecting.get(hostId)
    if (inFlight) return inFlight

    const attempt = this.connect(hostId)
    this.connecting.set(hostId, attempt)
    try {
      return await attempt
    } finally {
      this.connecting.delete(hostId)
    }
  }

  /** Verified session seam for app-owned setup and loopback forwarding, never renderer IPC. */
  async sessionForHost(hostId: string): Promise<SshSession> {
    await this.ensureConnected(hostId)
    const session = this.sessions.get(hostId)?.session
    if (!session) throw new Error('The SSH host disconnected before its verified session was available.')
    return session
  }

  private async connect(hostId: string): Promise<SshBackend> {
    const host = await this.resolveHost(hostId)
    if (!host) throw new Error(`Unknown SSH host: ${hostId}`)

    // Track trust + secret prompts registered during THIS connect attempt so a failure below can
    // clean up exactly these entries — never another in-flight attempt's pending resolver.
    const trustPromptIds = new Set<string>()
    const secretPromptIds = new Set<string>()

    const connDeps: SshConnectionDeps = {
      clientFactory: this.deps.clientFactory ?? createSsh2Client,
      getTrustedKeys: () => this.deps.getTrustedKeys(),
      saveTrustedKey: (k) => this.deps.saveTrustedKey(k),
      promptTrust: (input) => {
        if (this.deps.interactivePrompts && !this.deps.interactivePrompts()) {
          return Promise.reject(new Error(`SSH host key for ${input.host}:${input.port} is not trusted, and interactive prompts are unavailable in headless mode.`))
        }
        return new Promise<boolean>((resolve, reject) => {
          const promptId = randomUUID()
          trustPromptIds.add(promptId)
          const timeoutMs = this.promptTimeoutMs()
          const timer = setTimeout(() => {
            trustPromptIds.delete(promptId)
            this.takePending(this.pendingTrust, promptId)?.reject(new Error(`SSH host-key prompt timed out after ${timeoutMs}ms.`))
          }, timeoutMs)
          timer.unref?.()
          this.pendingTrust.set(promptId, { resolve, reject, timer })
          try {
            this.deps.sendEvent(SSH_TRUST_PROMPT_CHANNEL, { promptId, ...input })
          } catch (error) {
            trustPromptIds.delete(promptId)
            this.takePending(this.pendingTrust, promptId)?.reject(error instanceof Error ? error : new Error(String(error)))
          }
        })
      },
      promptSecret: (input) => {
        if (this.deps.interactivePrompts && !this.deps.interactivePrompts()) {
          return Promise.reject(new Error(`SSH ${input.kind} is required for ${input.label}, but interactive prompts are unavailable in headless mode.`))
        }
        return new Promise<string | null>((resolve, reject) => {
          const promptId = randomUUID()
          secretPromptIds.add(promptId)
          const timeoutMs = this.promptTimeoutMs()
          const timer = setTimeout(() => {
            secretPromptIds.delete(promptId)
            this.takePending(this.pendingSecrets, promptId)?.reject(new Error(`SSH credential prompt timed out after ${timeoutMs}ms.`))
          }, timeoutMs)
          timer.unref?.()
          this.pendingSecrets.set(promptId, { resolve, reject, timer })
          try {
            this.deps.sendEvent(SSH_SECRET_PROMPT_CHANNEL, { promptId, ...input })
          } catch (error) {
            secretPromptIds.delete(promptId)
            this.takePending(this.pendingSecrets, promptId)?.reject(error instanceof Error ? error : new Error(String(error)))
          }
        })
      },
      readIdentityFile: this.deps.readIdentityFile ?? (async (p) => {
        try { return await fs.readFile(p, 'utf8') } catch { return null }
      }),
      expandPath: (p) => expandTilde(p, this.home()),
      agentSock: this.deps.agentSock ? this.deps.agentSock() : process.env.SSH_AUTH_SOCK,
      emitState: (state) => this.setState(state),
      now: () => this.now(),
      interactivePrompts: this.deps.interactivePrompts ? this.deps.interactivePrompts() : true,
    }

    let session: SshSession
    try {
      session = await connectSsh(host, connDeps)
    } catch (err) {
      // The transport can fail (socket drop/timeout/auth error) while a trust prompt OR a
      // passphrase/password prompt registered above is still awaiting the renderer's answer.
      // connectSsh never resolves those prompts itself in this case, so without this cleanup the
      // resolver — and the renderer's dialog — would leak forever. Resolve trust false (deny) and
      // secrets null (cancel), drop them from their Maps, then surface a failure state for this host
      // so the renderer can clear a stale prompt.
      for (const promptId of trustPromptIds) {
        this.takePending(this.pendingTrust, promptId)?.resolve(false)
      }
      for (const promptId of secretPromptIds) {
        this.takePending(this.pendingSecrets, promptId)?.resolve(null)
      }
      this.setState({ hostId, phase: 'error', lastError: err instanceof Error ? err.message : String(err) })
      throw err
    }

    const remoteDir = host.lastRemoteDir || '.'
    const backend = new SshBackend(remoteDir, session)
    const state: SshConnectionState = { hostId, phase: 'connected', connectedAt: this.now() }
    this.sessions.set(hostId, { session, backend, state })
    this.states.set(hostId, state)
    return backend
  }

  /** Connect, run a no-op, and disconnect — used by the "Test connection" button. */
  async test(hostId: string): Promise<{ ok: boolean; message: string }> {
    try {
      const backend = await this.ensureConnected(hostId)
      const res = await backend.exec('uname -a || echo ok')
      return { ok: res.code === 0, message: res.stdout.trim() || 'Connected.' }
    } catch (e) {
      return { ok: false, message: e instanceof Error ? e.message : String(e) }
    }
  }

  /** List directory entries on a connected host (for the remote dir picker). */
  async listRemoteDir(hostId: string, dir: string): Promise<Array<{ name: string; isDirectory: boolean }>> {
    const backend = await this.ensureConnected(hostId)
    const entries = await backend.readdir(dir)
    return entries
      .filter((e) => e.isDirectory)
      .map((e) => ({ name: e.name, isDirectory: true }))
      .sort((a, b) => a.name.localeCompare(b.name))
  }
}
