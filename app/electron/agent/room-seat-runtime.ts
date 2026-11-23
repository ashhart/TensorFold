import type { ApiMessage, Provider, Seat, SeatHeartbeat } from './types'
import type { RoomRunDeps, RoomSend } from './rooms'
import { streamChat } from './openai'
import { estimateTokens, renderTranscript } from './compaction'
import { ensureTensorFoldStudioStylePrompt } from './prompt-style'
import { isCliSeatProvider, runCliSeatTurn, type CliSeatTurnParams } from './room-cli-seat'

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

export function createRoomSeatTurn(deps: RoomRunDeps, send: (event: RoomSend) => void) {
  const { project, signal } = deps
  const chat = deps.chat ?? streamChat
  const cliSeatTurn = deps.cliSeatTurn ?? ((params: CliSeatTurnParams, onDelta: (text: string) => void) => runCliSeatTurn(params, onDelta))
  const streamSeatReply = async (rt: SeatRuntime, provider: Provider, turnSignal: AbortSignal): Promise<string> => {
    // CLI-backed seats run as a confined one-shot CLI invocation (room-cli-seat.ts); the
    // per-turn timeout/abort wrapper in runSeatTurn applies unchanged either way.
    if (isCliSeatProvider(provider)) {
      return cliSeatTurn(
        { provider, model: rt.seat.model, messages: rt.messages as ApiMessage[], cwd: project.cwd, reasoningEffort: rt.reasoningEffort, signal: turnSignal },
        (text) => send({ type: 'delta', seat: rt.seat.name, chunk: text }),
      )
    }
    let reply = ''
    for await (const chunk of chat({
      provider,
      baseUrl: provider.baseUrl, apiKey: provider.apiKey, model: rt.seat.model,
      messages: rt.messages, temperature: deps.temperature, reasoningEffort: rt.reasoningEffort, approvalMode: 'readonly', cwd: project.cwd, signal: turnSignal,
    })) {
      if (chunk.type === 'content' && chunk.text) { reply += chunk.text; send({ type: 'delta', seat: rt.seat.name, chunk: chunk.text }) }
    }
    return reply
  }

  // Run one seat turn under a per-turn timeout combined with the room signal. If the seat's model
  // hangs/stalls, the turn is aborted (timedOut=true) and skipped so other models keep going, rather
  // than the whole room freezing behind one slow seat.
  return async (rt: SeatRuntime, provider: Provider): Promise<{ reply: string; timedOut: boolean }> => {
    const timeoutMs = deps.turnTimeoutMs ?? DEFAULT_SEAT_TURN_TIMEOUT_MS
    const ctrl = new AbortController()
    const onAbort = () => ctrl.abort()
    signal.addEventListener('abort', onAbort, { once: true })
    let timedOut = false
    const timer = timeoutMs > 0 ? setTimeout(() => { timedOut = true; ctrl.abort() }, timeoutMs) : null
    try {
      return { reply: await streamSeatReply(rt, provider, ctrl.signal), timedOut: false }
    } catch (e) {
      if (timedOut) return { reply: '', timedOut: true }
      throw e
    } finally {
      if (timer) clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
    }
  }
}
