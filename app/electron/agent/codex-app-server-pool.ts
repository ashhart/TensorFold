import { cliSpawnEnv } from './cli-spawn-env'
import { binary, codexExecApprovalPolicy, codexModelAndEffort, flattenMessages, sandbox } from './codex-cli'
import { planningCommand } from './planning-sandbox'
import { contentText } from './content.ts'
import type { EngineSteeringChannel } from './steering'
import type { ApiMessage, ApprovalMode, Provider } from './types'
import { CodexAppServerSession, CodexThreadConfig, CodexStdioTransport, CodexResumeCursor, createCodexAppServerSession, withProtocolIdleTimeout, CODEX_APP_SERVER_IDLE_TIMEOUT_MS, CodexStreamChunk, CODEX_APP_SERVER_INTERRUPT_GRACE_MS } from './codex-app-server-session'
import { spawnCodexAppServerTransport } from './codex-app-server-transport'

export const POOL_IDLE_MS = 30 * 60_000

export interface PooledSession {
  session: CodexAppServerSession
  idleTimer: ReturnType<typeof setTimeout>

  configKey: string
}

export function threadConfigKey(config: CodexThreadConfig): string {
  return JSON.stringify({
    cwd: config.cwd ?? null,
    approvalPolicy: config.approvalPolicy ?? null,
    sandbox: config.sandbox ?? null,
  })
}

export const pool = new Map<string, PooledSession>()

export function touchIdleTimer(key: string, entry: PooledSession): void {
  clearTimeout(entry.idleTimer)
  entry.idleTimer = setTimeout(() => closeCodexAppServerSession(key), POOL_IDLE_MS)
}

// Evict + close one pooled session (e.g. the thread was deleted, or it idled out).
export function closeCodexAppServerSession(key: string): void {
  const entry = pool.get(key)
  if (!entry) return
  clearTimeout(entry.idleTimer)
  pool.delete(key)
  try { entry.session.close() } catch { }
}

// Close every pooled session — called on app quit, mirroring killAllTerminals().
export function closeAllCodexAppServerSessions(): void {
  for (const key of [...pool.keys()]) closeCodexAppServerSession(key)
}

export function pooledCodexAppServerSessionKeys(): string[] {
  return [...pool.keys()]
}

export async function getOrCreateCodexAppServerSession(
  key: string,
  spawnTransport: () => CodexStdioTransport,
  threadConfig: CodexThreadConfig,
  resumeCursor: CodexResumeCursor | null,
  onError?: (message: string) => void,
  signal?: AbortSignal,
): Promise<{ session: CodexAppServerSession; needsFullHistory: boolean }> {
  const configKey = threadConfigKey(threadConfig)
  const existing = pool.get(key)
  if (existing) {

    if (existing.session.isClosed) {
      closeCodexAppServerSession(key)
    } else if (existing.configKey === configKey) {

      touchIdleTimer(key, existing)
      return { session: existing.session, needsFullHistory: false }
    } else {
      closeCodexAppServerSession(key)
    }
  }
  const session = createCodexAppServerSession({ spawnTransport, onError })

  let needsFullHistory: boolean
  const abortHandshake = () => session.close()
  signal?.addEventListener('abort', abortHandshake, { once: true })
  try {
    if (signal?.aborted) abortHandshake()
    await withProtocolIdleTimeout(session.initialize(), CODEX_APP_SERVER_IDLE_TIMEOUT_MS, 'codex app-server initialize')
    needsFullHistory = !resumeCursor
    if (resumeCursor) {
      await withProtocolIdleTimeout(session.resumeThread(resumeCursor.threadId, threadConfig), CODEX_APP_SERVER_IDLE_TIMEOUT_MS, 'codex app-server thread/resume')
    } else {
      await withProtocolIdleTimeout(session.startThread(threadConfig), CODEX_APP_SERVER_IDLE_TIMEOUT_MS, 'codex app-server thread/start')
    }
  } catch (error) {
    try { session.close() } catch { }
    throw error
  } finally {
    signal?.removeEventListener('abort', abortHandshake)
  }
  const entry: PooledSession = { session, idleTimer: setTimeout(() => { }, 0), configKey }
  pool.set(key, entry)
  touchIdleTimer(key, entry)
  return { session, needsFullHistory }
}

export function latestUserMessageText(messages: ApiMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return contentText(messages[i].content).trim()
  }
  return ''
}

// ---- streaming entry point: resume-aware codex app-server transport --------------------

export interface CodexAppServerStreamParams {
  provider: Provider
  model: string
  messages: ApiMessage[]
  reasoningEffort?: string
  approvalMode?: ApprovalMode
  cwd?: string | null
  // Stable per-conversation key for the session pool — the TensorFold Thread id.
  sessionKey: string
  // The Thread's persisted resume cursor, if any (from a prior app run).
  resumeCursor?: CodexResumeCursor | null
  signal?: AbortSignal
  // Cooperative interrupt grace before the pooled process is forcibly reaped.
  interruptGraceMs?: number
  // Live mid-turn steering channel (VST-033); connected for the duration of the turn.
  steering?: EngineSteeringChannel
  // Injectable for tests; production omits this and a real `codex app-server` process spawns.
  spawnTransport?: () => CodexStdioTransport
  onError?: (message: string) => void
}

export type CodexAppServerChunk = CodexStreamChunk | { type: 'resume_cursor'; cursor: CodexResumeCursor }

export async function* streamCodexAppServer(params: CodexAppServerStreamParams): AsyncGenerator<CodexAppServerChunk> {
  if (params.signal?.aborted) return
  const { model, effort } = codexModelAndEffort(params.model, params.reasoningEffort)
  const threadConfig: CodexThreadConfig = {
    cwd: params.cwd || undefined,
    model,
    approvalPolicy: codexExecApprovalPolicy(params.provider, params.approvalMode),
    sandbox: sandbox(params.provider, params.approvalMode),
  }
  const spawnTransport = params.spawnTransport ?? (() => {
    const run = planningCommand(params.provider, binary(params.provider), ['app-server'], {
      cwd: params.cwd || undefined,
      env: cliSpawnEnv(params.provider.cliEnv),
    })
    return spawnCodexAppServerTransport({ binary: run.command, argv: run.args, cwd: run.cwd, env: run.env })
  })
  let acquired: { session: CodexAppServerSession; needsFullHistory: boolean }
  try {
    acquired = await getOrCreateCodexAppServerSession(
      params.sessionKey,
      spawnTransport,
      threadConfig,
      params.resumeCursor ?? null,
      params.onError,
      params.signal,
    )
  } catch (error) {
    if (params.signal?.aborted) return
    throw error
  }
  const { session, needsFullHistory } = acquired

  if (params.signal?.aborted) return
  const prompt = needsFullHistory ? flattenMessages(params.messages) : latestUserMessageText(params.messages)
  if (!prompt) return

  let interrupted = false
  let forcedCloseTimer: ReturnType<typeof setTimeout> | null = null
  const interrupt = () => {
    if (interrupted) return
    interrupted = true
    void session.interrupt().catch(() => { })

    forcedCloseTimer = setTimeout(() => {
      forcedCloseTimer = null
      if (session.hasActiveTurn) closeCodexAppServerSession(params.sessionKey)
    }, Math.max(1, params.interruptGraceMs ?? CODEX_APP_SERVER_INTERRUPT_GRACE_MS))
    forcedCloseTimer.unref?.()
  }
  const signal = params.signal
  if (signal?.aborted) interrupt()
  const onAbort = () => interrupt()
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    for await (const chunk of session.runTurn(prompt, { model, effort, steering: params.steering })) {
      if (signal?.aborted) {
        interrupt()
        break
      }
      yield chunk
    }
  } catch (error) {

    if (!signal?.aborted) throw error
  } finally {
    signal?.removeEventListener('abort', onAbort)
    if (signal?.aborted) interrupt()
    if (forcedCloseTimer && !session.hasActiveTurn) {
      clearTimeout(forcedCloseTimer)
      forcedCloseTimer = null
    }
  }
  if (signal?.aborted) return
  if (session.resumeCursor) yield { type: 'resume_cursor', cursor: session.resumeCursor }
}
