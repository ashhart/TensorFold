import type { ApiMessage, Provider, Seat, SeatHeartbeat } from './types'
import type { RoomRunDeps, RoomSend } from './rooms'
import { streamChat } from './openai'
import { estimateTokens, renderTranscript } from './compaction'
import { ensureTensorFoldStudioStylePrompt } from './prompt-style'
import { isCliSeatProvider, runCliSeatTurn } from './room-cli-seat'
import { ENGINE_FIRST_BYTE_MS, isEngineProvider } from './engine/provider'
import { deliberationProfile } from './room-collaboration'
import { roomOutcomeProfile } from './room-outcomes'

export interface SeatRuntime {
  seat: Seat
  reasoningEffort: string
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[]
  recent: string[]
  heartbeat: SeatHeartbeat
}

const ROOM_KEEP_PREFIX = 2 // system primer + SENSE/task
const ROOM_KEEP_RECENT = 8

async function summarizeRoomTranscript(provider: Provider, model: string, transcript: string, reasoningEffort: string, signal: AbortSignal): Promise<string> {
  const sys = ensureTensorFoldStudioStylePrompt([
    'Compress a multi-agent room seat transcript into a dense private memory.',
    'Preserve the task, decisions, claims, blockers, files/tools used, open tickets, and anything directly addressed to this seat.',
    'Use terse bullets. Output only the summary.',
  ].join('\n'))
  let out = ''
  for await (const ch of streamChat({
    provider,
    baseUrl: provider.baseUrl,
    apiKey: provider.apiKey,
    model,
    messages: [{ role: 'system', content: sys }, { role: 'user', content: transcript }],
    temperature: 0.1,
    reasoningEffort,
    approvalMode: 'readonly',
    cwd: null,
    signal,
  })) {
    if (ch.type === 'content') out += ch.text
  }
  return out.trim()
}

export async function compactSeatRuntime(rt: SeatRuntime, provider: Provider, limit: number, signal: AbortSignal): Promise<{ before: number; after: number; folded: number } | null> {
  if (limit <= 0) return null
  const before = estimateTokens(rt.messages as ApiMessage[])
  if (before <= limit) return null
  const prefix = Math.min(ROOM_KEEP_PREFIX, rt.messages.length)
  const cut = Math.max(prefix, rt.messages.length - ROOM_KEEP_RECENT)
  if (cut - prefix < 2) return null
  const toCompress = rt.messages.slice(prefix, cut)
  const summary = await summarizeRoomTranscript(provider, rt.seat.model, renderTranscript(toCompress as ApiMessage[]), rt.reasoningEffort, signal)
  if (!summary) return null
  rt.messages = [
    ...rt.messages.slice(0, prefix),
    { role: 'user', content: `[Earlier private room context auto-compacted to protect this model's context window. Running memory:]\n\n${summary}` },
    ...rt.messages.slice(cut),
  ]
  return { before, after: estimateTokens(rt.messages as ApiMessage[]), folded: toCompress.length }
}

const DEFAULT_SEAT_TURN_TIMEOUT_MS = 180_000

// First-output window: a cold built-in engine keeps the engine allowance, capped by the room's own deliberation budget.
export function seatFirstOutputWindow(provider: Provider, idleMs: number, roomBudgetMs = 0): number {
  if (!Number.isFinite(idleMs) || idleMs <= 0) return 0
  if (!isEngineProvider(provider)) return idleMs
  const cap = roomBudgetMs > 0 ? Math.min(ENGINE_FIRST_BYTE_MS, roomBudgetMs) : ENGINE_FIRST_BYTE_MS
  return Math.max(idleMs, cap)
}

export function createRoomSeatTurn(deps: RoomRunDeps, send: (event: RoomSend) => void) {
  const { project, signal } = deps
  const chat = deps.chat ?? streamChat
  const cliSeatTurn = deps.cliSeatTurn
    ?? ((params, onDelta, onActivity) => runCliSeatTurn(params, onDelta, undefined, onActivity))
  const streamSeatReply = async (rt: SeatRuntime, provider: Provider, turnSignal: AbortSignal, touch: () => void): Promise<string> => {
    // CLI-backed seats run as a confined one-shot CLI invocation (room-cli-seat.ts); the
    // per-turn idle watchdog/abort wrapper in runSeatTurn applies unchanged either way.
    if (isCliSeatProvider(provider)) {
      return cliSeatTurn(
        { provider, model: rt.seat.model, messages: rt.messages as ApiMessage[], cwd: project.cwd, reasoningEffort: rt.reasoningEffort, signal: turnSignal },
        (text) => { touch(); send({ type: 'delta', seat: rt.seat.name, chunk: text }) },
        touch,
      )
    }
    let reply = ''
    for await (const chunk of chat({
      provider,
      baseUrl: provider.baseUrl, apiKey: provider.apiKey, model: rt.seat.model,
      messages: rt.messages, temperature: deps.temperature, reasoningEffort: rt.reasoningEffort, approvalMode: 'readonly', cwd: project.cwd, signal: turnSignal,
    })) {
      touch()
      if (chunk.type === 'content' && chunk.text) { reply += chunk.text; send({ type: 'delta', seat: rt.seat.name, chunk: chunk.text }) }
    }
    return reply
  }

  // The idle window is a stall watchdog that streamed chunks refresh, not a cap: only a hung seat is skipped.
  return async (rt: SeatRuntime, provider: Provider): Promise<{ reply: string; timedOut: boolean }> => {
    const timeoutMs = deps.turnTimeoutMs ?? DEFAULT_SEAT_TURN_TIMEOUT_MS
    const roomBudgetMs = deliberationProfile(deps.room.deliberationEffort ?? roomOutcomeProfile(deps.room.mode)?.budget).timeBudgetMs
    const ctrl = new AbortController()
    let settled = false
    let timedOut = false
    let firstWindow = true
    let timer: ReturnType<typeof setTimeout> | null = null
    const clear = () => { if (timer) { clearTimeout(timer); timer = null } }
    const onAbort = () => { clear(); ctrl.abort() }
    const onTimer = () => {
      timer = null
      if (settled || signal.aborted || ctrl.signal.aborted) return
      timedOut = true
      ctrl.abort()
    }
    // Arm (or re-arm on each streamed chunk) the stall window; a non-finite or 0 window means no watchdog.
    const arm = () => {
      if (settled || signal.aborted || ctrl.signal.aborted) return
      const windowMs = firstWindow ? seatFirstOutputWindow(provider, timeoutMs, roomBudgetMs) : timeoutMs
      if (!Number.isFinite(windowMs) || windowMs <= 0) return
      firstWindow = false
      clear()
      timer = setTimeout(onTimer, windowMs)
    }
    signal.addEventListener('abort', onAbort, { once: true })
    // A room stopped before the turn: no listener fires, so abort instead of arming a stale watchdog.
    if (signal.aborted) ctrl.abort()
    else arm()
    try {
      const reply = await streamSeatReply(rt, provider, ctrl.signal, arm)
      // CLI transports resolve cleanly on abort, so the watchdog flag (not the resolve) decides the outcome.
      if (timedOut && !signal.aborted) return { reply: '', timedOut: true }
      return { reply, timedOut: false }
    } catch (e) {
      if (timedOut && !signal.aborted) return { reply: '', timedOut: true }
      throw e
    } finally {
      settled = true
      clear()
      signal.removeEventListener('abort', onAbort)
    }
  }
}
