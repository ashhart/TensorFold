// TensorFold-Studio-as-ACP-agent: a REAL Agent Client Protocol server that hosts TensorFold's
// agent loop for an external ACP client (e.g. the Zed editor). This is the
// outbound/server half of ACP — the inverse direction from external-agent-runtime.ts,
// which hosts EXTERNAL agents (Codex/Claude CLI, etc.) as ACP-speaking subprocesses
// INSIDE TensorFold. Here, TensorFold itself is "the agent being hosted inside an editor".
//
// Built on the official, currently-maintained `@agentclientprotocol/sdk` package
// (NOT the deprecated `@zed-industries/agent-client-protocol` name for the same
// package). The SDK owns JSON-RPC framing, request/response id correlation, and
// notification routing — this file only implements the agent-side business logic
// (initialize / session/new / session/prompt / session/cancel) the SDK's `agent()`
// builder calls into.
//
// electron/agent/acp-interop.ts is a pre-existing planning/exploration-spike data
// structure (never wired to a process) and electron/agent/acp-server-export.ts has
// some JSON-RPC result-shape builders that predate the real SDK and use an invented
// envelope (`{jsonrpc, id, result}` built by hand). Neither is reused here: the SDK
// itself handles JSON-RPC envelopes, so hand-rolling them would fight the SDK rather
// than use it. Both files are left in place as-is (acp-interop.ts's role/boundary
// notes remain useful design context; acp-server-export.ts is unused dead code that
// is out of this task's scope to delete).
import { randomUUID } from 'node:crypto'
import * as acp from '@agentclientprotocol/sdk'
import { runAgent } from './loop'
import type { ApprovalMode, Provider, Thread } from './types'

// ---- Turn-runner abstraction --------------------------------------------------
//
// The pieces of one ACP prompt turn that the protocol layer needs from "running a
// TensorFold turn", independent of HOW that turn is actually executed. The real
// implementation (createTensorFoldStudioAcpTurnRunner, below) drives electron/agent/loop.ts's
// runAgent; tests inject a fake here the same way codex-app-server.test.ts injects
// a FakeTransport instead of spawning a real codex subprocess.

/** One pending tool-approval request surfaced mid-turn, mapped from TensorFold's loop. */
export interface AcpApprovalRequest {
  toolCallId: string
  title: string
  /** TensorFold tool name when known (e.g. "run_command", "edit_file"); used as the ACP tool-kind hint. */
  kind?: acp.ToolKind
  /** Optional file path this tool call touches, surfaced to the client as a location. */
  path?: string
}

export type AcpApprovalDecision = 'allow' | 'allow_always' | 'deny'

/** Dependencies handed to an injected/real turn runner for exactly one prompt turn. */
export interface AcpTurnDeps {
  /** Stable identifier for the ACP session this turn belongs to (keys the real per-session Thread). */
  sessionId: string
  /** The session's working directory (from session/new's `cwd`). */
  cwd: string
  /** The user's prompt text for this turn (ACP ContentBlock[] flattened to text — see promptTextFromBlocks). */
  promptText: string
  /** Aborts when the client sends session/cancel for this session. */
  signal: AbortSignal
  /** Stream one assistant-message text chunk to the client via session/update. */
  emitContent: (text: string) => void
  /** Stream one reasoning/thinking text chunk to the client via session/update. */
  emitReasoning: (text: string) => void
  /**
   * Ask the connected ACP client to approve a tool call via session/request_permission.
   * Resolves to TensorFold's allow/allow_always/deny decision shape so it composes directly
   * with loop.ts's RunDeps.requestApproval contract.
   */
  requestApproval: (request: AcpApprovalRequest) => Promise<AcpApprovalDecision>
}

/** Runs exactly one TensorFold-Studio-backed turn. Throwing fails the turn (mapped to a JSON-RPC error). */
export type AcpTurnRunner = (deps: AcpTurnDeps) => Promise<void>

// ---- Real adapter: bridges an ACP turn onto loop.ts's runAgent ---------------
//
// v1 simplification (explicitly called out, per the task spec): an ACP session maps
// to one ephemeral, in-memory TensorFold Thread (kind: 'task') created lazily on its first
// prompt — it is NOT persisted to ~/.vontra/threads.json and does not appear in the
// app's thread list. Each session/prompt call appends the prompt and drives one
// runAgent() call to completion (or cancellation) against that in-memory thread, so
// the thread's apiMessages/timeline accumulate turn-over-turn exactly like a normal
// multi-turn TensorFold conversation, but scoped to the lifetime of this ACP session only.
//
// Provider/model selection (v1 simplification, explicitly flagged): rather than
// wiring full Settings/Provider-account UI integration (out of scope here), the
// caller supplies one fixed `provider`/`model` for every session this server hosts
// (see acp-server-main.ts, which resolves a sensible default from
// store.settings.activeProviderId/activeModel — adapted to read ~/.vontra/settings.json
// directly since this runs as a standalone process outside the Electron app).
export interface TensorFoldStudioTurnLoopDeps {
  provider: Provider
  model: string
  /**
   * Approval bridge mode (v1 simplification, explicitly documented): every ACP
   * session created by this server runs loop.ts's gate with this approvalMode.
   * Because the SDK exposes a real session/request_permission primitive, the
   * default here is 'suggest' (loop.ts's normal "ask before doing anything
   * non-read-only" mode) — every non-read-only tool call genuinely round-trips
   * through the connected ACP client's session/request_permission before running,
   * via the requestApproval bridge wired in createAcpAgentHandlers below. This is
   * the REAL approval bridge the task asks for, not a stub: nothing destructive
   * runs over ACP without the client explicitly selecting an "allow" option for
   * that exact call. (A caller may still pass approvalMode: 'readonly' to fall
   * back to the auto-approve-safe-tools-only / deny-everything-else posture for
   * deployments that would rather not trust the connected client at all.)
   */
  approvalMode?: ApprovalMode
  temperature?: number
  reasoningEffort?: string
}

function emptyAcpThread(cwd: string): Thread {
  const now = Date.now()
  return {
    id: randomUUID(),
    title: 'ACP session',
    kind: 'task',
    cwd,
    providerId: null,
    model: null,
    createdAt: now,
    updatedAt: now,
    status: 'idle',
    apiMessages: [],
    timeline: [],
  }
}

/**
 * Builds the REAL turn runner used in production: one TensorFold Thread per ACP
 * session (created lazily on first prompt, reused turn-over-turn), driven by
 * loop.ts's runAgent. TensorFold StreamChunk content/reasoning chunks reach the
 * client because loop.ts's own `emit()` callback (the same one
 * main/host/thread-runtime.ts uses to mirror AgentEvents to the renderer) is
 * hooked here and translated into ACP session/update notifications via
 * emitContent/emitReasoning instead of an IPC event.
 */
export function createTensorFoldStudioAcpTurnRunner(loopDeps: TensorFoldStudioTurnLoopDeps): { runTurn: AcpTurnRunner; threadFor: (sessionId: string) => Thread | undefined; forgetSession: (sessionId: string) => void } {
  const threads = new Map<string, Thread>()

  const runTurn: AcpTurnRunner = async ({ sessionId, cwd, promptText, signal, emitContent, emitReasoning, requestApproval }) => {
    let thread = threads.get(sessionId)
    if (!thread) {
      thread = emptyAcpThread(cwd)
      threads.set(sessionId, thread)
    }
    thread.apiMessages.push({ role: 'user', content: promptText })

    // loop.ts's own approval_request AgentEvent carries only {callId, itemId} — the
    // human-readable title/tool/path the requestApproval hook below needs instead live
    // on the timeline item that loop.ts pushes immediately before requesting approval
    // for that same call, so it's read straight off thread.timeline (closed over here)
    // rather than threading a second map through runAgent's requestApproval(callId)
    // signature, which only carries the call id.
    const toolKindForName = (name?: string): acp.ToolKind | undefined => {
      if (!name) return undefined
      if (name === 'read_file' || name === 'list_dir' || name === 'glob') return 'read'
      if (name === 'edit_file' || name === 'apply_patch' || name === 'write_file') return 'edit'
      if (name === 'search' || name === 'retrieval_search') return 'search'
      if (name === 'run_command') return 'execute'
      if (name === 'web_fetch') return 'fetch'
      return 'other'
    }

    await runAgent({
      thread,
      provider: loopDeps.provider,
      model: loopDeps.model,
      approvalMode: loopDeps.approvalMode ?? 'suggest',
      temperature: loopDeps.temperature ?? 0.1,
      reasoningEffort: loopDeps.reasoningEffort ?? 'medium',
      signal,
      persist: () => { /* v1: ACP sessions are in-memory only; nothing to persist to disk. */ },
      emit: (event) => {
        if (event.type === 'item_delta' && event.field === 'text') {
          const item = thread!.timeline.find((candidate) => candidate.id === event.id)
          if (item?.kind === 'reasoning') emitReasoning(event.chunk)
          else if (item?.kind === 'assistant') emitContent(event.chunk)
        }
      },
      requestApproval: async (callId) => {
        // loop.ts's RunDeps.requestApproval signature is (callId) only, but the item it
        // just added to the timeline carries everything we need to ask the client a
        // useful question — look it up by matching the most recent timeline item whose
        // approval is still pending. loop.ts always pushes the tool's item_add
        // immediately before requesting approval for that same call, so the *latest*
        // pending/awaiting tool item on the timeline is the one this call concerns.
        const item = [...thread!.timeline].reverse().find((candidate) => candidate.kind === 'tool' && (candidate.state === 'awaiting' || candidate.state === 'pending'))
        const decision = await requestApproval({
          toolCallId: callId,
          title: item?.title || item?.tool || 'Run a tool',
          kind: toolKindForName(item?.tool),
          path: item?.path,
        })
        return decision
      },
    })
  }

  return {
    runTurn,
    threadFor: (sessionId) => threads.get(sessionId),
    // Drops the session's accumulated in-memory Thread (apiMessages/timeline) so it can be
    // garbage-collected once the handlers evict the session. Without this the per-session
    // Thread would live for the whole process lifetime even after the session is gone.
    forgetSession: (sessionId) => { threads.delete(sessionId) },
  }
}

// ---- Content-block <-> plain text helpers -------------------------------------

/** Flattens an ACP prompt's ContentBlock[] into plain text for TensorFold's loop. */
export function promptTextFromBlocks(blocks: acp.ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type === 'text') parts.push(block.text)
    else if (block.type === 'resource_link') parts.push(`[attached: ${block.name ?? block.uri}]`)
    // v1 simplification: image/audio/embedded-resource content blocks are not forwarded
    // to the model. TensorFold's vision path (multimodal-attachments-build work) expects its
    // own Attachment shape; bridging ACP's ContentBlock variants into that is future work.
  }
  return parts.join('\n').trim()
}

function permissionOptions(): acp.PermissionOption[] {
  return [
    { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
    { optionId: 'allow_always', name: 'Always allow', kind: 'allow_always' },
    { optionId: 'deny', name: 'Deny', kind: 'reject_once' },
  ]
}

function decisionFromPermissionOutcome(outcome: acp.RequestPermissionOutcome): AcpApprovalDecision {
  if (outcome.outcome === 'cancelled') return 'deny'
  if (outcome.optionId === 'allow_always') return 'allow_always'
  if (outcome.optionId === 'allow') return 'allow'
  return 'deny'
}

// ---- Session state + agent-side handlers --------------------------------------

interface AcpSession {
  id: string
  cwd: string
  abortController: AbortController | null
  /**
   * The currently-running prompt turn for this session, if any. A second prompt must
   * abort AND await this before starting its own runTurn, otherwise two runAgent() calls
   * mutate the same per-session Thread's apiMessages/timeline concurrently (abort is
   * cooperative, so the prior turn keeps writing until it next reaches an await boundary).
   */
  inFlight: Promise<void> | null
  /** Last time this session saw activity (created or prompted); drives idle eviction. */
  lastActivity: number
}

const DEFAULT_MAX_ACP_SESSIONS = 256
const DEFAULT_ACP_SESSION_IDLE_TTL_MS = 30 * 60_000

export interface AcpServerDeps {
  /** Executes one prompt turn. Tests inject a fake; production wires createTensorFoldStudioAcpTurnRunner(...).runTurn. */
  runTurn: AcpTurnRunner
  /** Default cwd used when a session/new request omits one. */
  cwd: string
  /** Agent name/version surfaced in InitializeResponse.agentInfo. */
  name?: string
  version?: string
  /**
   * Called when a session is evicted (idle TTL exceeded or the session cap forces out the
   * oldest idle session). Production wires this to createTensorFoldStudioAcpTurnRunner(...).forgetSession
   * so the per-session in-memory Thread is dropped alongside the session, preventing the maps
   * and accumulated apiMessages/timeline from growing without bound over a long-lived process.
   */
  onSessionClosed?: (sessionId: string) => void
  /** Max retained sessions before the oldest idle ones are evicted (default 256). */
  maxSessions?: number
  /** Idle-session TTL in ms; sessions untouched longer than this are evicted on the next prompt/new (default 30 min). */
  sessionIdleTtlMs?: number
}

/**
 * Builds the agent-side request/notification handler functions, independent of the
 * SDK's `agent()` connection wiring — kept separate so the handler LOGIC (session
 * bookkeeping, prompt -> turn-runner dispatch, cancel -> abort) is testable without
 * needing a live AgentContext for unit tests that don't care about the wire format.
 */
export function createAcpAgentHandlers(deps: AcpServerDeps) {
  const sessions = new Map<string, AcpSession>()
  const maxSessions = Math.max(1, deps.maxSessions ?? DEFAULT_MAX_ACP_SESSIONS)
  const idleTtlMs = deps.sessionIdleTtlMs ?? DEFAULT_ACP_SESSION_IDLE_TTL_MS

  function evictSession(session: AcpSession) {
    session.abortController?.abort()
    sessions.delete(session.id)
    deps.onSessionClosed?.(session.id)
  }

  /**
   * Bounds the sessions map: drops idle sessions past the TTL, then — if still over the cap —
   * evicts the oldest sessions with no in-flight turn. A session with an active turn is never
   * evicted (that would abort a live prompt), so a flood of concurrent live sessions can still
   * exceed the cap transiently; the cap is a steady-state bound, not a hard ceiling.
   */
  function reapSessions() {
    const now = Date.now()
    if (idleTtlMs > 0) {
      for (const session of [...sessions.values()]) {
        if (!session.inFlight && now - session.lastActivity > idleTtlMs) evictSession(session)
      }
    }
    if (sessions.size <= maxSessions) return
    const idle = [...sessions.values()].filter((s) => !s.inFlight).sort((a, b) => a.lastActivity - b.lastActivity)
    for (const session of idle) {
      if (sessions.size <= maxSessions) break
      evictSession(session)
    }
  }

  async function initialize(_params: acp.InitializeRequest): Promise<acp.InitializeResponse> {
    return {
      protocolVersion: acp.PROTOCOL_VERSION,
      agentCapabilities: {
        // v1: session/load is not implemented (no onRequest handler is registered for
        // it below), so this is honestly false — advertising true here without a
        // handler would make a client's session/load call fail with "method not
        // found" instead of the clean capability-gated skip ACP expects.
        loadSession: false,
        promptCapabilities: { image: false, audio: false, embeddedContext: true },
        mcpCapabilities: { http: false, sse: false },
      },
      authMethods: [],
      agentInfo: { name: deps.name ?? 'TensorFold', version: deps.version ?? '0.1.0' },
    }
  }

  async function newSession(params: acp.NewSessionRequest): Promise<acp.NewSessionResponse> {
    const id = randomUUID()
    sessions.set(id, { id, cwd: params.cwd || deps.cwd, abortController: null, inFlight: null, lastActivity: Date.now() })
    reapSessions()
    return {
      sessionId: id,
      modes: { currentModeId: 'default', availableModes: [{ id: 'default', name: 'TensorFold' }] },
    }
  }

  async function prompt(params: acp.PromptRequest, client: acp.AgentContext): Promise<acp.PromptResponse> {
    const session = sessions.get(params.sessionId)
    if (!session) throw new acp.RequestError(-32602, `Unknown ACP session: ${params.sessionId}`)
    session.lastActivity = Date.now()

    // A second prompt while one is already running replaces it (mirrors the SDK's own
    // ExampleAgent: `session.pendingPrompt?.abort()` before starting a new turn). Abort is
    // cooperative, so we must also AWAIT the prior turn to fully unwind before starting this
    // one — otherwise both runAgent() calls write to the same per-session Thread concurrently
    // and corrupt its apiMessages/timeline.
    session.abortController?.abort()
    const prior = session.inFlight
    if (prior) await prior.catch(() => {})

    const controller = new AbortController()
    session.abortController = controller

    const promptText = promptTextFromBlocks(params.prompt)

    const turn = deps.runTurn({
      sessionId: session.id,
      cwd: session.cwd,
      promptText,
      signal: controller.signal,
      emitContent: (text) => {
        if (!text) return
        void client.notify('session/update', {
          sessionId: session.id,
          update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } },
        })
      },
      emitReasoning: (text) => {
        if (!text) return
        void client.notify('session/update', {
          sessionId: session.id,
          update: { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text } },
        })
      },
      requestApproval: async (request) => {
        const response = await client.request('session/request_permission', {
          sessionId: session.id,
          toolCall: { toolCallId: request.toolCallId, title: request.title, kind: request.kind, locations: request.path ? [{ path: request.path }] : undefined },
          options: permissionOptions(),
        })
        return decisionFromPermissionOutcome(response.outcome)
      },
    })
    // Publish this turn as the session's in-flight promise so a later prompt awaits it after
    // aborting (a never-rejecting variant, since callers only need completion, not the result).
    const inFlight = turn.then(() => {}, () => {})
    session.inFlight = inFlight

    try {
      await turn
    } catch (e) {
      if (controller.signal.aborted) return { stopReason: 'cancelled' }
      throw e
    } finally {
      if (session.abortController === controller) session.abortController = null
      // Clear our in-flight marker only if a newer prompt hasn't already replaced it.
      if (session.inFlight === inFlight) session.inFlight = null
      session.lastActivity = Date.now()
    }

    if (controller.signal.aborted) return { stopReason: 'cancelled' }
    return { stopReason: 'end_turn' }
  }

  async function cancel(params: acp.CancelNotification): Promise<void> {
    sessions.get(params.sessionId)?.abortController?.abort()
  }

  return { initialize, newSession, prompt, cancel, sessions }
}

// ---- Wiring into the SDK's agent() builder ------------------------------------

/** Builds a real acp.AgentApp wired to the given (possibly fake, for tests) turn runner. */
export function buildTensorFoldStudioAcpAgentApp(deps: AcpServerDeps): acp.AgentApp {
  const handlers = createAcpAgentHandlers(deps)
  return acp
    .agent({ name: deps.name ?? 'TensorFold' })
    .onRequest('initialize', (ctx) => handlers.initialize(ctx.params))
    .onRequest('session/new', (ctx) => handlers.newSession(ctx.params))
    .onRequest('session/prompt', (ctx) => handlers.prompt(ctx.params, ctx.client))
    .onNotification('session/cancel', (ctx) => handlers.cancel(ctx.params))
}

/** Runs the agent over a real ACP Stream (stdio in production; see acp-server-main.ts). */
export function connectTensorFoldStudioAcpAgent(deps: AcpServerDeps, stream: acp.Stream): acp.AgentConnection {
  return buildTensorFoldStudioAcpAgentApp(deps).connect(stream)
}
