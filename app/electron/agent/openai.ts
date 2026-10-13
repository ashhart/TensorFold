import { streamCodexCli } from './codex-cli'
import { streamClaudeCli, type ClaudeCliChunk, type ClaudeResumeCursor } from './claude-cli'
import { streamCodexAppServer, latestUserMessageText, type CodexAppServerChunk, type CodexRateLimitSnapshotWire, type CodexResumeCursor } from './codex-app-server'
import { runOpenCodeTurn, createOpenCodeHttpTransport, isOpenCodeProtocol, type OpenCodeStreamChunk, type OpenCodeResumeCursor } from './opencode-engine'
import { streamPiCli, type PiCliChunk, type PiResumeCursor } from './pi-cli'
import { streamCursorAgent, type CursorCliChunk, type CursorResumeCursor } from './cursor-agent-cli'
import { streamGeminiCli } from './gemini-cli'
import { createLocalProviderFetch, createProviderFetch } from './secure-fetch'
import { engineRoute, recordEnginePerformance } from './engine/provider'
import { readBoundedJsonResponse, readBoundedResponseText } from './bounded-response'
import { needsPlanningGuard, planningGuard } from './planning-guard'
import { classifyProvider } from '../../shared/provider-trust'
import { readCortheonReport, readEngineTimings, type CortheonReply, type EngineReplyTimings } from './turn-stats'
import type { EngineSteeringChannel } from './steering'
import type { ApiMessage, ApprovalMode, Provider, ToolCall } from './types'

export interface ChatParams {
  provider?: Provider
  baseUrl?: string
  apiKey?: string
  model: string
  messages: ApiMessage[]
  tools?: unknown[]
  temperature?: number
  reasoningEffort?: string
  serviceTier?: string
  approvalMode?: ApprovalMode
  cwd?: string | null
  signal?: AbortSignal
  // Stable per-conversation key (the Thread id) for the codex-cli transport's resumable
  // app-server session pool. Absent => the one-shot `codex exec` path is used (no pooling).
  threadId?: string
  // The Thread's persisted codex app-server resume cursor, if any (from a prior turn/app run).
  codexResumeCursor?: CodexResumeCursor | null
  // The Thread's persisted Claude Code session cursor, if any — the claude-cli transport
  // resumes that session (`claude -p --resume`) instead of re-flattening the conversation.
  claudeResumeCursor?: ClaudeResumeCursor | null
  // The Thread's persisted OpenCode session cursor, if any — the opencode transport reuses
  // that server-side session instead of creating a new one each turn.
  openCodeResumeCursor?: OpenCodeResumeCursor | null
  // The Thread's persisted Pi session cursor — pi-cli passes it as --session-id
  // (create-or-resume, so a stale id can never fail with "session not found").
  piResumeCursor?: PiResumeCursor | null
  // The Thread's persisted Cursor Agent chat id — cursor-cli resumes it via --resume.
  cursorResumeCursor?: CursorResumeCursor | null
  // Live mid-turn steering channel (VST-033). Only the transports whose protocols support
  // mid-turn input connect to it (codex app-server `turn/steer`, claude stream-json stdin,
  // pi rpc `steer`); see providerSupportsLiveSteering.
  steering?: EngineSteeringChannel
}

export type StreamChunk =
  | { type: 'reasoning'; text: string }
  | { type: 'content'; text: string }
  | { type: 'tool_calls'; toolCalls: ToolCall[] }
  | { type: 'finish'; reason: string | null }
  // Emitted once a codex app-server turn completes, carrying the cursor to persist onto
  // the Thread so the NEXT turn (or a future app run) can resume the same codex thread
  // instead of starting fresh.
  | { type: 'codex_resume_cursor'; threadId: string }
  // Same idea for the claude-cli transport: the Claude Code session id to persist onto the
  // Thread so the next turn resumes the same server-side session.
  | { type: 'claude_resume_cursor'; sessionId: string }
  // Same idea for the opencode transport: the OpenCode session id to persist.
  | { type: 'opencode_resume_cursor'; sessionId: string }
  // Same idea for the pi-cli transport.
  | { type: 'pi_resume_cursor'; sessionId: string }
  // Same idea for the cursor-cli transport (gemini-cli has no documented resume arg yet).
  | { type: 'cursor_resume_cursor'; sessionId: string }
  // Real account rate-limit snapshot from a codex app-server turn (VST-037), for the host
  // to persist a provider-status snapshot feeding the picker + doctor.
  | { type: 'provider_rate_limits'; snapshot: CodexRateLimitSnapshotWire }
  // Timeline note from the planning guard, e.g. what was withheld from an external model.
  | { type: 'notice'; text: string }
  // Last chunk of a chat-completions stream: when tokens arrived (epoch ms), the server's usage and the engine's own timings, if sent.
  | { type: 'usage'; requestAt: number; firstTokenAt: number | null; lastTokenAt: number | null; completionTokens: number | null; promptTokens: number | null; cachedTokens: number | null; engine: EngineReplyTimings | null; cortheon: CortheonReply | null }

const DELEGATED_CLI_HEARTBEAT_MS = 15_000
// After finish_reason, how long a stream that asked for usage may wait for the usage frame.
const USAGE_TAIL_MS = 1_000
export const PROVIDER_SSE_IDLE_TIMEOUT_MS = 60_000
export const PROVIDER_JSON_MAX_BYTES = 4 * 1024 * 1024
export const PROVIDER_ERROR_MAX_BYTES = 64 * 1024
// Above the tool-argument cap: a backend may send one call's whole arguments in a single frame.
export const PROVIDER_SSE_FRAME_MAX_CHARS = 2 * 1024 * 1024
export const PROVIDER_TOOL_ARGUMENTS_MAX_CHARS = 1024 * 1024
const PROVIDER_STREAM_OUTPUT_MAX_CHARS = 8 * 1024 * 1024
const PROVIDER_MAX_TOOL_CALLS = 128

/** A provider's error payload (string or `{ message }`) as one bounded line. */
export function providerErrorText(error: unknown): string {
  const raw = typeof error === 'string' ? error : (error as { message?: unknown } | null)?.message
  const text = typeof raw === 'string' && raw.trim() ? raw : JSON.stringify(error ?? null)
  return text.replace(/\s+/g, ' ').slice(0, 400)
}

function chatCompletionsReasoningEffort(value?: string | null, provider?: Provider | null, model?: string | null): string | null {
  const normalized = (value ?? '').trim().toLowerCase()
  const target = `${provider?.id ?? ''} ${provider?.name ?? ''} ${provider?.source ?? ''} ${provider?.baseUrl ?? ''} ${model ?? ''}`.toLowerCase()
  if (/gemini|google/.test(target)) return null
  if (!normalized || normalized === 'medium' || normalized === 'none' || normalized === 'dynamic' || normalized === '512') return null
  if (normalized === 'xhigh' || normalized === 'max' || normalized === 'ultrathink' || normalized === 'ultracode') return 'high'
  if (normalized === 'minimal' || normalized === 'low' || normalized === 'high') return normalized
  return null
}

function joinUrl(base: string, path: string): string {
  return base.replace(/\/+$/, '') + path
}

function delegatedCliLabel(provider?: Pick<Provider, 'transport' | 'name'>): string | null {
  if (provider?.transport === 'codex-cli') return 'Codex'
  if (provider?.transport === 'claude-cli') return 'Claude'
  if (provider?.transport === 'opencode') return 'OpenCode'
  if (provider?.transport === 'kilo') return 'Kilo'
  if (provider?.transport === 'pi-cli') return 'Pi'
  if (provider?.transport === 'cursor-cli') return 'Cursor'
  if (provider?.transport === 'gemini-cli') return 'Gemini'
  return null
}

function shortDuration(ms: number): string {
  const seconds = Math.max(1, Math.round(ms / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const rest = seconds % 60
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`
}

export function delegatedCliHeartbeatText(label: string, elapsedMs: number, idleMs: number): string {
  return `${label} is still working (${shortDuration(elapsedMs)} elapsed; no live CLI event for ${shortDuration(idleMs)}).\n`
}

type NextOrTimer<T> =
  | { kind: 'next'; next: IteratorResult<T> }
  | { kind: 'timer' }
  | { kind: 'aborted' }
  | { kind: 'error'; error: unknown }

function nextOrTimer<T>(pending: Promise<IteratorResult<T>>, ms: number, signal?: AbortSignal): Promise<NextOrTimer<T>> {
  if (signal?.aborted) return Promise.resolve({ kind: 'aborted' })
  return new Promise((resolve) => {
    let settled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const done = (value: NextOrTimer<T>) => {
      if (settled) return
      settled = true
      if (timer) clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      resolve(value)
    }
    const onAbort = () => done({ kind: 'aborted' })
    timer = setTimeout(() => done({ kind: 'timer' }), Math.max(0, ms))
    signal?.addEventListener('abort', onAbort, { once: true })
    pending.then(
      (next) => done({ kind: 'next', next }),
      (error) => done({ kind: 'error', error }),
    )
  })
}

/** Bound a single provider stream read without blocking cancellation or the event loop. */
export async function readWithIdleTimeout<T>(read: () => Promise<T>, timeoutMs = PROVIDER_SSE_IDLE_TIMEOUT_MS, signal?: AbortSignal): Promise<T> {
  const result = await nextOrTimer(read() as Promise<IteratorResult<unknown>>, Math.max(1, timeoutMs), signal)
  if (result.kind === 'next') return result.next as T
  if (result.kind === 'error') throw result.error
  if (result.kind === 'aborted') {
    const error = new Error('Provider stream was aborted.')
    error.name = 'AbortError'
    throw error
  }
  throw new Error(`Provider event stream was idle for ${Math.max(1, timeoutMs)}ms and was cancelled.`)
}

export async function* withDelegatedCliHeartbeat(
  source: AsyncIterable<StreamChunk>,
  provider: Pick<Provider, 'transport' | 'name'> | undefined,
  opts: { signal?: AbortSignal; heartbeatMs?: number; now?: () => number } = {},
): AsyncGenerator<StreamChunk> {
  const label = delegatedCliLabel(provider)
  if (!label) {
    yield* source
    return
  }

  const heartbeatMs = Math.max(1, opts.heartbeatMs ?? DELEGATED_CLI_HEARTBEAT_MS)
  const now = opts.now ?? Date.now
  const iterator = source[Symbol.asyncIterator]()
  const startedAt = now()
  let lastCliEventAt = startedAt
  let nextHeartbeatAt = startedAt + heartbeatMs
  let closed = false
  let pending = iterator.next()

  try {
    while (true) {
      const waitMs = Math.max(0, nextHeartbeatAt - now())
      const result = await nextOrTimer(pending, waitMs, opts.signal)

      if (result.kind === 'aborted') return
      if (result.kind === 'error') throw result.error

      if (result.kind === 'next') {
        if (result.next.done) {
          closed = true
          return
        }
        lastCliEventAt = now()
        nextHeartbeatAt = lastCliEventAt + heartbeatMs
        yield result.next.value
        pending = iterator.next()
        continue
      }

      const at = now()
      yield { type: 'reasoning', text: delegatedCliHeartbeatText(label, at - startedAt, at - lastCliEventAt) }
      nextHeartbeatAt = at + heartbeatMs
    }
  } finally {
    if (!closed) await iterator.return?.()
  }
}

function contentToText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) {
    return value.map((part) => {
      if (typeof part === 'string') return part
      if (part && typeof part === 'object') {
        const record = part as Record<string, unknown>
        return typeof record.text === 'string' ? record.text : typeof record.content === 'string' ? record.content : ''
      }
      return ''
    }).join('')
  }
  return ''
}

function normalizeToolCalls(value: unknown): ToolCall[] {
  if (!Array.isArray(value)) return []
  return value
    .map((call, index) => {
      if (!call || typeof call !== 'object') return null
      const record = call as Record<string, any>
      const fn = record.function && typeof record.function === 'object' ? record.function : {}
      const name = typeof fn.name === 'string' ? fn.name : typeof record.name === 'string' ? record.name : ''
      const args = typeof fn.arguments === 'string'
        ? fn.arguments
        : typeof record.arguments === 'string'
          ? record.arguments
          : record.arguments ? JSON.stringify(record.arguments) : ''
      if (!name) return null
      return {
        id: typeof record.id === 'string' ? record.id : `call_${index}`,
        type: 'function' as const,
        function: { name, arguments: args },
      }
    })
    .filter((call): call is ToolCall => !!call)
}

async function* streamJsonCompletion(res: Response): AsyncGenerator<StreamChunk> {
  let json: any
  try {
    json = await readBoundedJsonResponse(res, 'Provider completion', PROVIDER_JSON_MAX_BYTES)
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error
    throw new Error('Provider returned a non-streaming response that was not valid bounded JSON.')
  }
  const choice = json?.choices?.[0] ?? {}
  const message = choice.message ?? choice.delta ?? {}
  const reasoning = contentToText(message.reasoning_content ?? message.reasoning ?? choice.reasoning_content ?? choice.reasoning)
  const content = contentToText(message.content ?? choice.text ?? json.output_text)
  const toolCalls = normalizeToolCalls(message.tool_calls ?? choice.tool_calls)
  if (reasoning) yield { type: 'reasoning', text: reasoning }
  if (content) yield { type: 'content', text: content }
  if (toolCalls.length) yield { type: 'tool_calls', toolCalls }
  yield { type: 'finish', reason: choice.finish_reason ?? json.finish_reason ?? 'stop' }
}

// codex-app-server's chunk union carries an extra `resume_cursor` variant not in TensorFold's
// core StreamChunk (so it can't go through withDelegatedCliHeartbeat directly). Splits it
// off via the callback and passes everything else through untouched.
export async function* translateCodexAppServerChunks(
  source: AsyncGenerator<CodexAppServerChunk>,
  onResumeCursor: (threadId: string) => void,
  onRealChunk?: () => void,
): AsyncGenerator<StreamChunk> {
  for await (const chunk of source) {
    if (chunk.type === 'resume_cursor') {
      onResumeCursor(chunk.cursor.threadId)
      continue
    }
    if (chunk.type === 'rate_limits') {
      // Rate-limit updates are metadata, not model output — forward as a typed chunk but
      // do NOT count as a real chunk (they arrive independent of turn success).
      yield { type: 'provider_rate_limits', snapshot: chunk.snapshot }
      continue
    }
    // Counted BEFORE the heartbeat wrapper, so injected keep-alive reasoning chunks don't
    // get mistaken for genuine model output when deciding whether the one-shot fallback is safe.
    onRealChunk?.()
    yield chunk
  }
}

// Sibling of translateCodexAppServerChunks for the claude-cli transport: strip the terminal
// `resume_cursor` chunk into a callback (the caller re-emits it as a typed StreamChunk after
// the heartbeat wrapper) and forward everything else.
export async function* translateClaudeCliChunks(
  source: AsyncGenerator<ClaudeCliChunk>,
  onResumeCursor: (sessionId: string) => void,
): AsyncGenerator<StreamChunk> {
  for await (const chunk of source) {
    if (chunk.type === 'resume_cursor') {
      onResumeCursor(chunk.cursor.sessionId)
      continue
    }
    yield chunk
  }
}

// Sibling of translateClaudeCliChunks for the cursor-cli transport.
export async function* translateCursorCliChunks(
  source: AsyncGenerator<CursorCliChunk>,
  onResumeCursor: (sessionId: string) => void,
): AsyncGenerator<StreamChunk> {
  for await (const chunk of source) {
    if (chunk.type === 'resume_cursor') {
      onResumeCursor(chunk.cursor.sessionId)
      continue
    }
    yield chunk
  }
}

// Sibling of translateClaudeCliChunks for the pi-cli transport.
export async function* translatePiCliChunks(
  source: AsyncGenerator<PiCliChunk>,
  onResumeCursor: (sessionId: string) => void,
): AsyncGenerator<StreamChunk> {
  for await (const chunk of source) {
    if (chunk.type === 'resume_cursor') {
      onResumeCursor(chunk.cursor.sessionId)
      continue
    }
    yield chunk
  }
}

// Sibling of translateClaudeCliChunks for the opencode transport: forward content/reasoning
// chunks; strip the resume_cursor into a callback AND re-yield it as a typed StreamChunk so
// the caller can persist the session id after the heartbeat wrapper.
export async function* translateOpenCodeChunks(
  source: AsyncGenerator<OpenCodeStreamChunk>,
  onResumeCursor: (sessionId: string) => void,
): AsyncGenerator<StreamChunk> {
  for await (const chunk of source) {
    if (chunk.type === 'resume_cursor') {
      onResumeCursor(chunk.cursor.sessionId)
      yield { type: 'opencode_resume_cursor', sessionId: chunk.cursor.sessionId }
      continue
    }
    yield chunk
  }
}

// Streams an OpenAI-compatible /chat/completions response. Works against
// OpenAI, Ollama, LM Studio, vLLM, z.ai GLM, and any compatible endpoint.
export async function* streamChat(p: ChatParams): AsyncGenerator<StreamChunk> {
  // External providers plan only (no tools, redacted, confined); the built-in engine is trusted.
  const planned = planningGuard(p, streamChat)
  if (planned) { yield* planned; return }
  const engine = await engineRoute(p)
  try {
  if (engine) p = engine.params
  if (p.provider?.transport === 'codex-cli') {
    // Prefer the resumable `codex app-server` session over the one-shot `codex exec`: it
    // keeps codex's own thread alive across turns (no re-flattening the whole conversation
    // every time) and persists a resume cursor so a later turn — or a future app run — can
    // reattach to the SAME codex thread. Requires a stable threadId to key the session pool;
    // callers that don't pass one (or a too-old codex without `app-server` support) fall
    // back to the one-shot path below.
    if (p.threadId) {
      let realChunks = 0
      let resumedThreadId: string | null = null
      try {
        const raw = streamCodexAppServer({
          provider: p.provider,
          model: p.model,
          messages: p.messages,
          cwd: p.cwd,
          approvalMode: p.approvalMode,
          reasoningEffort: p.reasoningEffort,
          sessionKey: p.threadId,
          resumeCursor: p.codexResumeCursor ?? null,
          steering: p.steering,
          signal: p.signal,
        })
        const translated = translateCodexAppServerChunks(
          raw,
          (tid) => { resumedThreadId = tid },
          () => { realChunks++ },
        )
        for await (const chunk of withDelegatedCliHeartbeat(translated, p.provider, { signal: p.signal })) {
          yield chunk
        }
        if (resumedThreadId) yield { type: 'codex_resume_cursor', threadId: resumedThreadId }
        yield { type: 'finish', reason: 'stop' }
        return
      } catch (e) {
        // A failure with REAL output already yielded means the turn was genuinely in progress —
        // propagate it rather than silently re-running via a different transport (that could
        // duplicate side effects). A failure before any real output is most likely the
        // app-server session failing to even start (e.g. an older codex without `app-server`
        // support) — safe to fall through to the one-shot path below. We count only genuine
        // model chunks (`realChunks`), NOT the synthetic keep-alive heartbeats injected by
        // withDelegatedCliHeartbeat during a long silent window before any real output.
        if (realChunks > 0) throw e
      }
    }
    const source = streamCodexCli({
      provider: p.provider,
      model: p.model,
      messages: p.messages,
      cwd: p.cwd,
      approvalMode: p.approvalMode,
      reasoningEffort: p.reasoningEffort,
      serviceTier: p.serviceTier,
      signal: p.signal,
    })
    for await (const chunk of withDelegatedCliHeartbeat(source, p.provider, { signal: p.signal })) {
      yield chunk
    }
    yield { type: 'finish', reason: 'stop' }
    return
  }

  if (p.provider?.transport === 'claude-cli') {
    const source = streamClaudeCli({
      provider: p.provider,
      model: p.model,
      messages: p.messages,
      cwd: p.cwd,
      approvalMode: p.approvalMode,
      reasoningEffort: p.reasoningEffort,
      resumeCursor: p.claudeResumeCursor ?? null,
      steering: p.steering,
      signal: p.signal,
    })
    let resumedSessionId: string | null = null
    const translated = translateClaudeCliChunks(source, (sessionId) => { resumedSessionId = sessionId })
    for await (const chunk of withDelegatedCliHeartbeat(translated, p.provider, { signal: p.signal })) {
      yield chunk
    }
    if (resumedSessionId) yield { type: 'claude_resume_cursor', sessionId: resumedSessionId }
    yield { type: 'finish', reason: 'stop' }
    return
  }

  if (p.provider?.transport === 'cursor-cli') {
    const source = streamCursorAgent({
      provider: p.provider,
      model: p.model,
      messages: p.messages,
      cwd: p.cwd,
      approvalMode: p.approvalMode,
      resumeCursor: p.cursorResumeCursor ?? null,
      signal: p.signal,
    })
    let cursorSessionId: string | null = null
    const translated = translateCursorCliChunks(source, (sessionId) => { cursorSessionId = sessionId })
    for await (const chunk of withDelegatedCliHeartbeat(translated, p.provider, { signal: p.signal })) {
      yield chunk
    }
    if (cursorSessionId) yield { type: 'cursor_resume_cursor', sessionId: cursorSessionId }
    yield { type: 'finish', reason: 'stop' }
    return
  }

  if (p.provider?.transport === 'gemini-cli') {
    // One-shot json mode (gemini's stream-json event fields aren't documented with
    // examples); the heartbeat wrapper keeps the UI alive during the silent wait.
    const source = streamGeminiCli({
      provider: p.provider,
      model: p.model,
      messages: p.messages,
      cwd: p.cwd,
      approvalMode: p.approvalMode,
      signal: p.signal,
    })
    for await (const chunk of withDelegatedCliHeartbeat(source, p.provider, { signal: p.signal })) {
      yield chunk
    }
    yield { type: 'finish', reason: 'stop' }
    return
  }

  if (p.provider?.transport === 'pi-cli') {
    const source = streamPiCli({
      provider: p.provider,
      model: p.model,
      messages: p.messages,
      cwd: p.cwd,
      approvalMode: p.approvalMode,
      reasoningEffort: p.reasoningEffort,
      resumeCursor: p.piResumeCursor ?? null,
      steering: p.steering,
      signal: p.signal,
    })
    let piSessionId: string | null = null
    const translated = translatePiCliChunks(source, (sessionId) => { piSessionId = sessionId })
    for await (const chunk of withDelegatedCliHeartbeat(translated, p.provider, { signal: p.signal })) {
      yield chunk
    }
    if (piSessionId) yield { type: 'pi_resume_cursor', sessionId: piSessionId }
    yield { type: 'finish', reason: 'stop' }
    return
  }

  if (p.provider && isOpenCodeProtocol(p.provider.transport)) {
    // The OpenCode/Kilo server protocol currently has no per-turn permission bridge.
    // Sending a prompt in read-only mode would still let the server-side agent edit or
    // run commands, so fail before creating a transport or submitting any prompt.
    if (p.approvalMode === 'readonly') {
      throw new Error(`${p.provider.name || 'OpenCode'} does not support TensorFold Read-only mode yet. Choose a permission-aware provider.`)
    }
    // Kilo is an OpenCode fork on the identical server protocol, so both drive this engine.
    // Only the newest user message goes to the server-side session — opencode keeps the
    // conversation, so re-flattening the whole history would duplicate context (same
    // reasoning as the claude/codex resumable paths).
    const prompt = latestUserMessageText(p.messages)
    if (!prompt) { yield { type: 'finish', reason: 'stop' }; return }
    const transport = createOpenCodeHttpTransport({
      serverUrl: p.provider.serverUrl,
      serverPassword: p.provider.serverPassword,
      cwd: p.cwd,
    })
    const source = translateOpenCodeChunks(
      runOpenCodeTurn({ transport, prompt, model: p.model, resumeCursor: p.openCodeResumeCursor ?? null, signal: p.signal }),
      (sessionId) => { p.openCodeResumeCursor = { sessionId } },
    )
    let resumedSessionId: string | null = null
    for await (const chunk of withDelegatedCliHeartbeat(source, p.provider, { signal: p.signal })) {
      if (chunk.type === 'opencode_resume_cursor') { resumedSessionId = chunk.sessionId; continue }
      yield chunk
    }
    if (resumedSessionId) yield { type: 'opencode_resume_cursor', sessionId: resumedSessionId }
    yield { type: 'finish', reason: 'stop' }
    return
  }

  if (!p.baseUrl) throw new Error('Provider base URL is required.')
  const providerFetch = engine?.fetch ?? (p.provider ? createProviderFetch(p.provider) : createLocalProviderFetch(p.baseUrl))
  const reasoningEffort = chatCompletionsReasoningEffort(p.reasoningEffort, p.provider, p.model)
  // Only trusted local servers (the built-in engine, loopback oMLX/Ollama/LM Studio) are asked for usage.
  const wantUsage = classifyProvider(p.provider, p.baseUrl) === 'trusted-local'
  const requestAt = Date.now()
  const res = await providerFetch(joinUrl(p.baseUrl, '/chat/completions'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(p.apiKey ? { Authorization: `Bearer ${p.apiKey}` } : {}),
    },
    body: JSON.stringify({
      model: p.model,
      messages: p.messages,
      stream: true,
      ...(wantUsage ? { stream_options: { include_usage: true } } : {}),
      ...(p.temperature != null ? { temperature: p.temperature } : {}),
      ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
      ...(p.tools && p.tools.length ? { tools: p.tools, tool_choice: 'auto' } : {}),
    }),
    signal: p.signal,
  })

  if (!res.ok) {
    const detail = await readBoundedResponseText(res, 'Provider error', PROVIDER_ERROR_MAX_BYTES).catch(() => '')
    // Attach status + headers so the model router can honor a real Retry-After / X-RateLimit-Reset
    // window instead of falling back to a fixed cooldown.
    const err = new Error(`HTTP ${res.status} ${res.statusText}${detail ? ` — ${detail.slice(0, 400)}` : ''}`) as Error & { status?: number; headers?: unknown }
    err.status = res.status
    err.headers = res.headers
    throw err
  }
  if (!res.headers.get('content-type')?.toLowerCase().includes('text/event-stream')) {
    yield* streamJsonCompletion(res)
    return
  }
  if (!res.body) throw new Error('Provider response did not include a stream body.')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  // Accumulate tool-call fragments by index across deltas.
  const toolAcc: Record<number, ToolCall> = {}
  let sawToolCalls = false
  let finished = false
  let sawDone = false
  let lastFinishReason: string | null = null

  // Some OpenAI-compatible backends (Ollama, vLLM, LM Studio) omit `index` on every tool_call
  // delta. Collapsing all of those onto index 0 would merge parallel tool calls (e.g. read_file +
  // write_file) into one garbled, unparseable call. When `index` is absent, treat a delta that
  // carries an `id` and/or a non-empty `function.name` as the START of a new call and assign it
  // the next unused auto-index; remember that as the "current" index-less call so that subsequent
  // index-less deltas which only carry `function.arguments` continuations (no id/name) keep
  // accumulating into the SAME call instead of starting another one.
  let nextAutoIndex = 0
  let currentAutoIndex: number | null = null
  const autoPositionIndexes: number[] = []
  const autoIdIndexes = new Map<string, number>()
  let toolCallCount = 0
  let toolArgumentChars = 0
  let streamedOutputChars = 0
  let firstTokenAt: number | null = null
  let lastTokenAt: number | null = null
  let completionTokens: number | null = null
  let promptTokens: number | null = null
  let cachedTokens: number | null = null
  let engineTimings: EngineReplyTimings | null = null
  let cortheonReport: CortheonReply | null = null
  const markToken = () => { const now = Date.now(); firstTokenAt ??= now; lastTokenAt = now }
  const usageCount = (value: unknown): number | null => {
    const n = typeof value === 'number' ? value : Number.NaN
    return Number.isFinite(n) && n >= 0 ? Math.floor(n) : null
  }
  const awaitingUsage = () => wantUsage && completionTokens == null && !sawDone

  const drainToolCalls = function* (): Generator<StreamChunk> {
    if (sawToolCalls && Object.keys(toolAcc).length) {
      const calls = Object.keys(toolAcc).sort((a, b) => +a - +b).map((k) => toolAcc[+k])
      yield { type: 'tool_calls', toolCalls: calls }
    }
  }

  const parseEventData = function* (raw: string): Generator<StreamChunk> {
    const data = raw.trim()
    if (data === '[DONE]') { sawDone = true; return }

    let json: any
    try { json = JSON.parse(data) } catch { return }
    // vLLM-style servers report a mid-stream failure as an error frame with no choices;
    // dropping it would end the turn as a blank "success".
    if (json?.error && !json.choices?.length) throw new Error(`Provider error: ${providerErrorText(json.error)}`)
    if (json?.usage && typeof json.usage === 'object') {
      completionTokens = usageCount(json.usage.completion_tokens) ?? completionTokens
      promptTokens = usageCount(json.usage.prompt_tokens) ?? promptTokens
      cachedTokens = usageCount(json.usage.prompt_tokens_details?.cached_tokens) ?? cachedTokens
    }
    // The built-in engine puts its reply timings on the finish chunk.
    engineTimings = readEngineTimings(json?.tensorfold) ?? engineTimings
    // With --cortheon the engine also reports its research for the reply.
    cortheonReport = readCortheonReport(json?.cortheon) ?? cortheonReport
    // Frames after finish_reason only carry usage.
    if (finished) return
    const choice = json.choices?.[0]
    if (!choice) return
    const delta = choice.delta ?? {}

    // Reasoning content (o-series, GLM, deepseek-style)
    const reasoning = delta.reasoning_content ?? delta.reasoning
    if (typeof reasoning === 'string' && reasoning) {
      markToken()
      streamedOutputChars += reasoning.length
      if (streamedOutputChars > PROVIDER_STREAM_OUTPUT_MAX_CHARS) throw new Error('Provider streamed output exceeded the safety limit.')
      yield { type: 'reasoning', text: reasoning }
    }

    if (typeof delta.content === 'string' && delta.content) {
      markToken()
      streamedOutputChars += delta.content.length
      if (streamedOutputChars > PROVIDER_STREAM_OUTPUT_MAX_CHARS) throw new Error('Provider streamed output exceeded the safety limit.')
      yield { type: 'content', text: delta.content }
    }

    if (Array.isArray(delta.tool_calls)) {
      if (delta.tool_calls.length) markToken()
      sawToolCalls = true
      const positionalIndexlessFrame = delta.tool_calls.length > 1 || autoPositionIndexes.length > 1
      for (let position = 0; position < delta.tool_calls.length; position++) {
        const tc = delta.tool_calls[position]
        let idx: number
        if (typeof tc.index === 'number') {
          // Backend sends real indices — trust them as-is.
          idx = tc.index
          if (!Number.isSafeInteger(idx) || idx < 0 || idx >= PROVIDER_MAX_TOOL_CALLS) {
            throw new Error('Provider emitted an invalid or excessive tool-call index.')
          }
          currentAutoIndex = null
        } else {
          // With multiple index-less calls in one delta, array position is the only stable
          // identity most compatible backends preserve across frames. Map each position once,
          // while retaining the sequential single-call fallback for backends that serialize A
          // completely before starting B.
          const startsNewCall = !!tc.id || !!tc.function?.name
          const knownById = tc.id ? autoIdIndexes.get(tc.id) : undefined
          const knownByPosition = positionalIndexlessFrame ? autoPositionIndexes[position] : undefined
          if (knownById != null) {
            idx = knownById
          } else if (!startsNewCall && knownByPosition != null) {
            idx = knownByPosition
          } else if (!startsNewCall && currentAutoIndex != null && !positionalIndexlessFrame) {
            idx = currentAutoIndex
          } else {
            if (nextAutoIndex >= PROVIDER_MAX_TOOL_CALLS) throw new Error('Provider emitted too many tool calls.')
            idx = nextAutoIndex++
          }
          currentAutoIndex = idx
          autoPositionIndexes[position] = idx
          if (tc.id) autoIdIndexes.set(tc.id, idx)
        }
        if (!toolAcc[idx]) {
          toolCallCount++
          if (toolCallCount > PROVIDER_MAX_TOOL_CALLS) throw new Error('Provider emitted too many tool calls.')
          toolAcc[idx] = { id: tc.id || `call_${idx}`, type: 'function', function: { name: '', arguments: '' } }
        }
        if (tc.id) toolAcc[idx].id = tc.id
        if (tc.function?.name) toolAcc[idx].function.name += tc.function.name
        if (tc.function?.arguments) {
          const fragment = String(tc.function.arguments)
          toolArgumentChars += fragment.length
          if (toolArgumentChars > PROVIDER_TOOL_ARGUMENTS_MAX_CHARS) {
            throw new Error('Provider tool-call arguments exceeded the safety limit.')
          }
          toolAcc[idx].function.arguments += fragment
        }
      }
    }

    if (choice.finish_reason === 'error') throw new Error(`Provider error: ${providerErrorText(json.error ?? choice.error ?? 'the stream ended with finish_reason "error"')}`)
    if (choice.finish_reason) {
      lastFinishReason = choice.finish_reason
      yield* drainToolCalls()
      yield { type: 'finish', reason: choice.finish_reason }
      finished = true
    }
  }

  // SSE data fields belong to one event until its blank line, not to separate JSON frames.
  // Bound all event fields, including ignored metadata, and retain split CRLF/UTF-8 input.
  let eventData: string[] = []
  let eventChars = 0
  let skipNextLF = false
  const flushEvent = function* (): Generator<StreamChunk> {
    const data = eventData.join('\n')
    eventData = []
    eventChars = 0
    if (data) yield* parseEventData(data)
  }
  const consumeLine = function* (line: string): Generator<StreamChunk> {
    if (!line) { yield* flushEvent(); return }
    eventChars += line.length + 1
    if (eventChars > PROVIDER_SSE_FRAME_MAX_CHARS) throw new Error('Provider event stream frame exceeded the safety limit.')
    if (line.startsWith(':')) return
    const colon = line.indexOf(':')
    const field = colon < 0 ? line : line.slice(0, colon)
    if (field !== 'data') return
    const value = colon < 0 ? '' : line.slice(colon + 1)
    eventData.push(value.startsWith(' ') ? value.slice(1) : value)
  }
  const consumeText = function* (text: string): Generator<StreamChunk> {
    let start = 0
    for (let index = 0; index < text.length; index++) {
      const code = text.charCodeAt(index)
      if (skipNextLF) {
        skipNextLF = false
        if (code === 10) { start = index + 1; continue }
      }
      if (code !== 10 && code !== 13) continue
      const line = buffer + text.slice(start, index)
      buffer = ''
      yield* consumeLine(line)
      start = index + 1
      skipNextLF = code === 13
      if (sawDone || (finished && !awaitingUsage())) return
    }
    buffer += text.slice(start)
    if (eventChars + buffer.length > PROVIDER_SSE_FRAME_MAX_CHARS) throw new Error('Provider event stream frame exceeded the safety limit.')
  }

  let completed = false
  let streamed = false
  try {
    read: while (true) {
      if (finished && !awaitingUsage()) break
      const idleMs = finished ? USAGE_TAIL_MS : engine ? (streamed ? engine.idleMs : engine.firstByteMs) : PROVIDER_SSE_IDLE_TIMEOUT_MS
      let next: Awaited<ReturnType<typeof reader.read>>
      try {
        next = await readWithIdleTimeout(() => reader.read(), idleMs, p.signal)
      } catch (error) {
        // A server that keeps the stream open after finish costs only the short usage tail.
        if (finished && !p.signal?.aborted) break
        throw error
      }
      if (next.done) { completed = true; break }
      streamed = true
      yield* consumeText(decoder.decode(next.value, { stream: true }))
      if (sawDone || (finished && !awaitingUsage())) break read
    }

    // Flush the decoder + the final event when the body ends without a trailing newline.
    if (completed) {
      yield* consumeText(decoder.decode())
      if (!sawDone) {
        if (buffer) yield* consumeLine(buffer)
        yield* flushEvent()
      }
    }

    // Some OpenAI-compatible backends end a tool-call stream with `[DONE]` (or a closed body) and no
    // truthy finish_reason. Without this, accumulated tool calls would be silently dropped and the
    // loop would think the turn ended with no action. Emit them + a terminal finish exactly once.
    if (!finished) {
      yield* drainToolCalls()
      yield { type: 'finish', reason: lastFinishReason ?? 'stop' }
    }
    const sample = { requestAt, firstTokenAt, lastTokenAt, completionTokens, promptTokens, cachedTokens, engine: engineTimings, cortheon: cortheonReport }
    if (engine) recordEnginePerformance(p.model, sample)
    yield { type: 'usage', ...sample }
  } finally {
    if (!completed) await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
  } finally { engine?.release?.() }
}

// Quick non-streaming completion used to auto-title threads. External providers take the
// streamChat path so the planning guard redacts the request.
export async function quickComplete(p: Omit<ChatParams, 'tools'>): Promise<string> {
  const engine = await engineRoute(p, { start: false })
  try {
  if (engine) p = engine.params
  if (p.provider?.transport === 'codex-cli' || p.provider?.transport === 'claude-cli' || p.provider?.transport === 'pi-cli' || p.provider?.transport === 'cursor-cli' || p.provider?.transport === 'gemini-cli' || isOpenCodeProtocol(p.provider?.transport) || needsPlanningGuard(p)) {
    let out = ''
    for await (const ch of streamChat(p)) {
      if (ch.type === 'content') out += ch.text
    }
    return out.trim()
  }
  if (!p.baseUrl) throw new Error('Provider base URL is required.')
  const providerFetch = engine?.fetch ?? (p.provider ? createProviderFetch(p.provider) : createLocalProviderFetch(p.baseUrl))
  const res = await providerFetch(joinUrl(p.baseUrl, '/chat/completions'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(p.apiKey ? { Authorization: `Bearer ${p.apiKey}` } : {}) },
    body: JSON.stringify({ model: p.model, messages: p.messages, stream: false, temperature: 0.3, max_tokens: 24 }),
    signal: p.signal,
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const json: any = await readBoundedJsonResponse(res, 'Provider quick completion', PROVIDER_JSON_MAX_BYTES)
  return (json.choices?.[0]?.message?.content ?? '').trim()
  } finally { engine?.release?.() }
}
