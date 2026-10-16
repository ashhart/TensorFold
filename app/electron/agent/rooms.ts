import { randomUUID } from 'node:crypto'
import { streamChat } from './openai'
import { isExternalProvider } from '../../shared/provider-trust'
import { PROJECT_CONTEXT_WITHHELD_NOTE } from './planning-guard'
import { estimateTokens, kfmt, renderTranscript } from './compaction'
import { VONTRA_STYLE_PROMPT, ensureTensorFoldStudioStylePrompt } from './prompt-style'
import type { ApiMessage, Project, Provider, Room, RoomDeliberationEffort, RoomEventItem, RoomState, RoomStreamEvent, Seat, SeatHeartbeat } from './types'
import { buildRoomMemoryDiff, buildRoomRunMemoryEntry, deliberationBudgetText, deliberationProfile, formatRoomMemory, inferSeatLane, mergeRoomMemory, normalizeCollaborationMode, normalizeDeliberationEffort, roomShouldCloseOut, seatCollaborationInstructions, updateRoomMemorySummary } from './room-collaboration'
import { validateRoomConfiguration, validateRoomTransition } from './room-invariants'
import { roomModePromptBlock, roomOutcomeGate, roomOutcomeProfile } from './room-outcomes'
import { renderSharedMemory } from './shared-room-memory'
import { isCliSeatProvider, runCliSeatTurn, type CliSeatTurnParams } from './room-cli-seat'
import { recordRoomEvent } from './orchestration-journal'
import { emitRoomLifecycle } from '../main/host/orchestration-events'

// Ported from LLMComposer's room/huddle engine: multiple provider-bound seats
// deliberate on one shared problem as an attributed dialogue, with SENSE
// injection so each seat knows it is one of several real teammates.

const VERBS = new Set(['SPEAK', 'ASK', 'OBJECT', 'ASSIGN', 'CLAIM', 'CHECK', 'VOTE', 'YIELD', 'FINAL', 'TOOL'])

interface SeatRuntime {
  seat: Seat
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[]
  recent: string[]
  heartbeat: SeatHeartbeat
}

// Minimal shape the heartbeat helpers need — lets tests exercise stall detection
// without standing up a full SeatRuntime (messages/recent are irrelevant here).
export interface SeatRuntimeLike {
  seat: Pick<Seat, 'id' | 'name'>
  heartbeat: SeatHeartbeat
}

interface SeatModelRecovery {
  provider: Provider
  providerAccountId: string | null
  model: string
  notice: string
}

// Default grace after nextDueAt before a slow seat is flagged stalled (BUG #53).
// Tolerates transient model latency so a normal slow turn does not false-positive.
const SEAT_STALL_GRACE_SECONDS = 30

// Heartbeat cadence derived from the room deliberation effort. Deeper sessions
// expect longer per-turn thinking, so we widen the interval to avoid false stalls.
export function seatHeartbeatIntervalSeconds(effort?: RoomDeliberationEffort | null): number {
  switch (effort) {
    case 'quick': return 30
    case 'deep': return 90
    case 'max': return 120
    case 'normal':
    default: return 60
  }
}

// Seed a healthy heartbeat for a seat at room start.
export function initSeatHeartbeat(seat: Pick<Seat, 'id' | 'name'>, intervalSeconds: number, now: number): SeatHeartbeat {
  return {
    seatId: seat.id,
    seatName: seat.name,
    intervalSeconds,
    stallGraceSeconds: SEAT_STALL_GRACE_SECONDS,
    required: true,
    count: 0,
    lastBeatAt: now,
    nextDueAt: now + intervalSeconds * 1000,
    missedSince: null,
  }
}

// Record that a seat produced a turn: bump count, advance the clock, clear any stall.
export function recordSeatBeat(hb: SeatHeartbeat, now: number): void {
  hb.count++
  hb.lastBeatAt = now
  hb.nextDueAt = hb.required ? now + hb.intervalSeconds * 1000 : null
  hb.missedSince = null
}

// Names of seats currently flagged stalled (missedSince set), excluding one seat
// by name (the seat trying to FINAL is obviously alive). Used to block a premature
// finalize while a teammate is hung (BUG #53).
export function stalledSeatNames(runtimes: SeatRuntimeLike[], exclude?: string): string[] {
  return runtimes
    .filter((rt) => rt.heartbeat.required && rt.heartbeat.missedSince != null && rt.seat.name !== exclude)
    .map((rt) => rt.seat.name)
}

// Re-arm the stall window for seats that actually took a turn in the previous pass.
// A single pass over several slow models can take longer than interval+grace, so a
// seat that beat last pass and is merely waiting its turn must not be false-flagged;
// pushing its nextDueAt forward to this pass start prevents that. Crucially, seats
// that did NOT run last pass (e.g. one that keeps timing out without ever calling
// recordSeatBeat, or any seat on the very first pass) are left untouched so their
// window keeps counting and evaluateSeatHeartbeats can actually flag a real stall —
// re-arming every non-stalled seat unconditionally is what made stall detection dead
// code (BUG #53 regression).
export function rearmSeatHeartbeats(
  runtimes: SeatRuntimeLike[],
  ranLastPass: Set<string>,
  passStart: number,
): void {
  for (const rt of runtimes) {
    const hb = rt.heartbeat
    if (hb.required && hb.missedSince == null && ranLastPass.has(rt.seat.id)) {
      hb.nextDueAt = passStart + hb.intervalSeconds * 1000
    }
  }
}

// Detect seats whose turn is overdue past the grace window. Mutates each newly
// stalled heartbeat's missedSince so a stall is reported once (not every tick),
// and returns human-readable notices for the moderator to surface.
export function evaluateSeatHeartbeats(runtimes: SeatRuntimeLike[], now: number): string[] {
  const stalled: string[] = []
  for (const rt of runtimes) {
    const hb = rt.heartbeat
    if (!hb.required || hb.lastBeatAt == null || hb.nextDueAt == null) continue
    if (hb.missedSince != null) continue // already flagged; do not re-report
    const stallAt = hb.nextDueAt + hb.stallGraceSeconds * 1000
    if (now >= stallAt) {
      hb.missedSince = hb.nextDueAt
      const ago = Math.round((now - hb.lastBeatAt) / 1000)
      stalled.push(`Seat "${rt.seat.name}": heartbeat overdue (last beat ${ago}s ago)`)
    }
  }
  return stalled
}

const ROOM_KEEP_PREFIX = 2 // system primer + SENSE/task
const ROOM_KEEP_RECENT = 8

function buildPrimer(project: Project, room: Room, seat: Seat, roster: Seat[], provider?: Provider): string {
  const rosterText = roster.map((s) => `- ${s.name}: ${s.role} (model ${s.model})`).join('\n')
  const roomMemory = formatRoomMemory(room)
  const sharedMemory = room.sharedMemory ? renderSharedMemory(room.sharedMemory) : ''
  return [
    `You are ${seat.name} in a live multi-agent working session with the teammates below.`,
    '',
    `ROOM ROLE\nYour visible room role is ${seat.role}. Use it to decide when to volunteer, challenge, defer, or take a ticket, but keep your general reasoning intact.`,
    `\n${seatCollaborationInstructions(room, seat, provider)}`,
    roomOutcomeProfile(room.mode) ? `\nROOM OUTCOME\n${roomModePromptBlock(roomOutcomeProfile(room.mode)!)}` : '',
    project.team ? `\nTEAM CHARTER\n${project.team}` : '',
    // External seats plan only and never see project memory (planning-guard.ts).
    project.memory ? (isExternalProvider(provider) ? `\n${PROJECT_CONTEXT_WITHHELD_NOTE}` : `\nPROJECT MEMORY\n${project.memory}`) : '',
    roomMemory ? `\nROOM MEMORY\n${roomMemory}` : '',
    sharedMemory ? `\n${sharedMemory}` : '',
    '',
    'ROOM CONTRACT',
    'Messages tagged [room] are what teammates actually said; [moderator] hands you the floor.',
    'Speak in first person, briefly and concretely. React to what was just said instead of restating it.',
    "Never write [room] or [moderator] lines yourself, and never invent a teammate's reply.",
    'Never repeat a question or point you already made.',
    'End every reply with exactly one line: ACTION: <VERB> <content>.',
    'Verbs: SPEAK, ASK, OBJECT, ASSIGN, CLAIM, CHECK, VOTE, YIELD, FINAL, TOOL.',
    'ACTION: ASK <teammate> <question> addresses one teammate by name. ACTION: FINAL <answer> ends the room only when the task is genuinely done.',
    'When you answer a specific ASK or ASSIGN, cite it by appending RE: #<id> to your ACTION line so the reply links back to the question.',
    'Only state what you can actually see in the [room] lines above. Do not claim progress that has not happened.',
    '',
    'TEAMWORK CONTRACT',
    'Make handoffs explicit. If a teammate asks or assigns you something, answer that first and say what you are unblocking for them.',
    'Help the room finish the user goal: create or claim concrete tickets, attach evidence, start workers when ready, and yield when waiting.',
    'If you disagree, name the risk and offer the next useful move instead of reopening the whole plan.',
    '',
    VONTRA_STYLE_PROMPT,
    '',
    'SHARED BOARD: coordinate through it so you never duplicate a teammate\'s work:',
    'ACTION: TOOL list_tickets: see every ticket and who is working on what.',
    'ACTION: TOOL claim_ticket {"title":"..."}: claim a ticket BEFORE you start it, so teammates know it is taken.',
    'ACTION: TOOL create_ticket {"title":"...","body":"...","priority":"high","risk":"medium","allowedCapabilities":["read"],"requiredEvidence":["files inspected","verification result"],"acceptanceCriteria":["..."]}: add an execution ticket to the board.',
    'ACTION: TOOL add_ticket_evidence {"title":"...","kind":"file","summary":"What you found","ref":"src/file.ts"}: attach evidence to a claimed ticket.',
    'ACTION: TOOL complete_ticket {"title":"...","evidence":[{"kind":"review","summary":"Reviewed X and found Y"}]}: mark your claimed ticket complete after attaching required evidence.',
    'ACTION: TOOL left_to_work: see the remaining room tickets, active workers, risks, and known file refs.',
    'ACTION: TOOL memory_write {"name":"decisions","content":"...","mode":"append-bullet"}: update the room\'s live shared memory (decisions, plan, blockers). Teammates see updates on their next turn.',
    'ACTION: TOOL start_ticket_work {"title":"...","phase":"build","paths":["src/x.ts"]}: ask the Room Supervisor to start a ticket worker thread in the room worktree. Use phase "review" for reviewer passes. If a file overlap is intentional, include "mergeDecision":"why this overlap is safe and how results will be merged".',
    'ACTION: TOOL delegate {"task":"..."}: hand a focused subtask to a sub-agent and get its result. Add "targetSeat":"SeatName" to run it on that seat\'s provider/model, or "providerId"/"accountId"/"model" for an explicit route; omit them to use your own.',
    'Always claim before working; never touch a ticket already claimed by someone else.',
    'Only start ticket workers for claimed or ready tickets. Declare paths when you know them so the supervisor can prevent overlapping edits.',
    'Do not ACTION: FINAL while room tickets are still open.',
    '',
    'WORKSPACE: inspect the shared project safely:',
    'ACTION: TOOL read_file {"path":"src/x.ts"}: read a file.',
    'ACTION: TOOL search {"query":"..."}: find relevant code.',
    'ACTION: TOOL list_dir {"path":"src"}: inspect folders.',
    'Room agents are read-only by default. Do not attempt edit_file, write_file, or run_command unless the user opens an approved work thread for that ticket.',
    'When you know the change needed, create/claim a ticket and explain the proposed edit for an approved thread to execute.',
    '',
    'TEAM ROSTER',
    rosterText,
  ].filter(Boolean).join('\n')
}

interface Action { verb: string; text: string; raw: string; prose?: string }

// Lenient gate: missing action becomes SPEAK, empty body becomes YIELD.
function gateAction(output: string): Action {
  const raw = (output ?? '').trim()
  if (!raw) return { verb: 'YIELD', text: '', raw }
  const lines = raw.split('\n')
  const actionLine = [...lines].reverse().find((l) => l.trim().toUpperCase().startsWith('ACTION:'))
  const prose = lines.filter((l) => l !== actionLine).join('\n').trim()
  if (!actionLine) return { verb: 'SPEAK', text: prose || raw, raw }
  const body = actionLine.trim().slice(actionLine.trim().indexOf(':') + 1).trim()
  const parts = body.split(/\s+/)
  const verb = (parts[0] || '').toUpperCase()
  const text = parts.slice(1).join(' ').trim()
  if (!VERBS.has(verb)) return { verb: 'SPEAK', text: prose || raw, raw }
  if (!text && verb !== 'YIELD') return { verb: 'YIELD', text: '', raw }
  // Keep the prose so peers receive the model's actual reasoning, not just the one-line ACTION tagline.
  return { verb, text, raw, prose }
}

function renderForRoom(ev: RoomEventItem, compact = false): string {
  const label = ev.kind.toUpperCase()
  // Addressed asks/assigns carry a short ref (#id8) so a teammate can answer with
  // "RE: #<id>"; replies show the ref they answer (re #id8).
  const ref = (ev.kind === 'ask' || ev.kind === 'assign') ? ` #${ev.id.slice(0, 8)}` : ''
  const reply = ev.replyTo ? ` re #${ev.replyTo.slice(0, 8)}` : ''
  if (compact) return `[room] ${ev.seat} (${label}${reply}): ${ev.text}`
  const head = ev.target && ev.target !== 'all' && ev.target !== ev.seat
    ? `[room] ${ev.seat} (${label}, to ${ev.target}${ref}${reply}):`
    : `[room] ${ev.seat} (${label}${ref}${reply}):`
  return `${head}\n${ev.text}`
}

function signature(raw: string): string {
  return raw.split('\n').filter((l) => !l.trim().toUpperCase().startsWith('ACTION:')).join(' ').trim().toLowerCase().slice(0, 400)
}

function blockerSignature(blockers: string[]): string {
  return blockers.map((blocker) => blocker.replace(/\s+/g, ' ').trim().toLowerCase()).filter(Boolean).join('\n')
}

function oneLine(text: string, max = 180): string {
  const compact = text.replace(/\s+/g, ' ').trim()
  return compact.length > max ? `${compact.slice(0, Math.max(0, max - 3))}...` : compact
}

function roomTeamCue(room: Room, seat: Seat): string {
  const recent = room.events.filter((event) => !['task', 'notice', 'user'].includes(event.kind)).slice(-10)
  const directHandoffs = recent.filter((event) => (event.kind === 'ask' || event.kind === 'assign') && event.target === seat.name)
  const latestDirect = directHandoffs[directHandoffs.length - 1]
  const latestOtherHandoff = recent.filter((event) =>
    (event.kind === 'ask' || event.kind === 'assign') &&
    !!event.target &&
    event.target !== 'all' &&
    event.target !== seat.name
  ).slice(-1)[0]
  const latestClaim = recent.filter((event) => event.kind === 'claim').slice(-1)[0]
  const latestUserSteer = room.events
    .filter((event) => event.kind === 'user' && (event.target === 'all' || event.target === seat.name))
    .slice(-1)[0]
  const cue: string[] = []

  if (latestUserSteer) {
    cue.push(`USER STEERING: "${oneLine(latestUserSteer.text, 220)}". Treat this as the user's latest direction and respond to it before lower-priority room chatter.`)
  }

  if (latestDirect) {
    cue.push(`TEAM CUE: ${latestDirect.seat} handed work to you: "${oneLine(latestDirect.text, 220)}". Answer or unblock that first, naming ${latestDirect.seat}.`)
  } else if (latestOtherHandoff) {
    cue.push(`TEAM CUE: ${latestOtherHandoff.seat} handed work to ${latestOtherHandoff.target}. Let that teammate respond unless you can unblock with new evidence.`)
  }

  if (latestClaim) {
    cue.push(`Latest claim: ${latestClaim.seat} claimed "${oneLine(latestClaim.text, 120)}". Build on it and do not duplicate it.`)
  }

  cue.push('Work like a teammate: reference who you are helping, answer direct asks before new topics, create or claim the next useful ticket, start workers when ready, and ACTION: YIELD if you are waiting.')
  return cue.join(' ')
}

function escapeReg(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

const ALWAYS_FULL = new Set(['final', 'task', 'user', 'file_change'])

// `excludeSeatId` is a seat ID (not name): keying on id means two seats that happen to share a name
// don't both get excluded, and neither collapses out of delivery.
function broadcast(runtimes: SeatRuntime[], item: RoomEventItem, excludeSeatId: string) {
  const addressed = new Set<string>()
  if (item.target && item.target !== 'all') addressed.add(item.target)
  for (const rt of runtimes) {
    if (rt.seat.id === excludeSeatId) continue
    const full = ALWAYS_FULL.has(item.kind) || addressed.has(rt.seat.name)
    rt.messages.push({ role: 'user', content: renderForRoom(item, !full) })
  }
}

async function summarizeRoomTranscript(provider: Provider, model: string, transcript: string, signal: AbortSignal): Promise<string> {
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
    approvalMode: 'readonly',
    cwd: null,
    signal,
  })) {
    if (ch.type === 'content') out += ch.text
  }
  return out.trim()
}

async function compactSeatRuntime(rt: SeatRuntime, provider: Provider, limit: number, signal: AbortSignal): Promise<{ before: number; after: number; folded: number } | null> {
  if (limit <= 0) return null
  const before = estimateTokens(rt.messages as ApiMessage[])
  if (before <= limit) return null
  const prefix = Math.min(ROOM_KEEP_PREFIX, rt.messages.length)
  const cut = Math.max(prefix, rt.messages.length - ROOM_KEEP_RECENT)
  if (cut - prefix < 2) return null
  const toCompress = rt.messages.slice(prefix, cut)
  const summary = await summarizeRoomTranscript(provider, rt.seat.model, renderTranscript(toCompress as ApiMessage[]), signal)
  if (!summary) return null
  rt.messages = [
    ...rt.messages.slice(0, prefix),
    { role: 'user', content: `[Earlier private room context auto-compacted to protect this model's context window. Running memory:]\n\n${summary}` },
    ...rt.messages.slice(cut),
  ]
  return { before, after: estimateTokens(rt.messages as ApiMessage[]), folded: toCompress.length }
}

// Distributive Omit so each union member keeps its own payload keys
// (a plain Omit<Union, K> collapses to the members' common keys only).
type RoomSend = RoomStreamEvent extends infer T ? (T extends any ? Omit<T, 'roomId'> : never) : never

export interface RoomRunDeps {
  room: Room
  project: Project
  providers: Provider[]
  problem: string
  temperature: number
  reasoningEffort: string
  emit: (e: RoomStreamEvent) => void
  signal: AbortSignal
  maxTurns?: number
  tools?: Record<string, (args: any, seat: string) => Promise<string>>
  providerForSeat?: (seat: Seat) => Provider | undefined
  recoverSeatModel?: (seat: Seat, error: unknown) => Promise<SeatModelRecovery | null>
  contextLimitForSeat?: (seat: Seat) => number
  finalGate?: () => string[]
  /** Supervisor mode hook: an ASSIGN auto-starts ticket work for the addressed seat.
   *  Returns a short notice for the room, or null when nothing was started. */
  onAssign?: (args: { seat: string; target: string; text: string }) => Promise<string | null>
  /** Injectable chat transport (defaults to streamChat) — lets tests drive distinct per-seat models. */
  chat?: typeof streamChat
  /** Injectable CLI seat turn (defaults to the one-shot dispatch in room-cli-seat.ts) — test seam mirroring `chat`. */
  cliSeatTurn?: (params: CliSeatTurnParams, onDelta: (text: string) => void) => Promise<string>
  /** Per-seat turn timeout (ms). A seat exceeding it is skipped so one hung model can't freeze the room. */
  turnTimeoutMs?: number
}

const DEFAULT_SEAT_TURN_TIMEOUT_MS = 180_000

export async function runRoom(deps: RoomRunDeps): Promise<void> {
  const { room, project, providers, problem, emit, signal } = deps
  const outcome = roomOutcomeProfile(room.mode)
  // Budget (VST-019): an explicit room effort wins; otherwise the mode's default budget.
  const deliberation = deliberationProfile(room.deliberationEffort ?? outcome?.budget)
  const maxTurns = deps.maxTurns ?? deliberation.maxTurns
  const startedAt = Date.now()
  const send = (e: RoomSend) => emit({ ...(e as any), roomId: room.id })
  // Guarded state mutation: an invariant is an enforcement boundary, not telemetry.
  // Illegal transitions are reported but never applied (notably complete -> failed if
  // renderer delivery throws after a FINAL has already committed completion).
  const setRoomState = (next: RoomState): boolean => {
    const error = validateRoomTransition(room.state, next)
    if (error) {
      send({ type: 'error', message: error.message })
      return false
    }
    room.state = next
    return true
  }
  const providerForSeat = (seat: Seat) => deps.providerForSeat?.(seat) ?? providers.find((p) => p.id === seat.providerId)
  const chat = deps.chat ?? streamChat
  const cliSeatTurn = deps.cliSeatTurn ?? ((params: CliSeatTurnParams, onDelta: (text: string) => void) => runCliSeatTurn(params, onDelta))
  const streamSeatReply = async (rt: SeatRuntime, provider: Provider, turnSignal: AbortSignal): Promise<string> => {
    // CLI-backed seats run as a confined one-shot CLI invocation (room-cli-seat.ts); the
    // per-turn timeout/abort wrapper in runSeatTurn applies unchanged either way.
    if (isCliSeatProvider(provider)) {
      return cliSeatTurn(
        { provider, model: rt.seat.model, messages: rt.messages as ApiMessage[], cwd: project.cwd, reasoningEffort: deps.reasoningEffort, signal: turnSignal },
        (text) => send({ type: 'delta', seat: rt.seat.name, chunk: text }),
      )
    }
    let reply = ''
    for await (const chunk of chat({
      provider,
      baseUrl: provider.baseUrl, apiKey: provider.apiKey, model: rt.seat.model,
      messages: rt.messages, temperature: deps.temperature, reasoningEffort: deps.reasoningEffort, approvalMode: 'readonly', cwd: project.cwd, signal: turnSignal,
    })) {
      if (chunk.type === 'content' && chunk.text) { reply += chunk.text; send({ type: 'delta', seat: rt.seat.name, chunk: chunk.text }) }
    }
    return reply
  }

  // Run one seat turn under a per-turn timeout combined with the room signal. If the seat's model
  // hangs/stalls, the turn is aborted (timedOut=true) and skipped so other models keep going, rather
  // than the whole room freezing behind one slow seat.
  const runSeatTurn = async (rt: SeatRuntime, provider: Provider): Promise<{ reply: string; timedOut: boolean }> => {
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

  const heartbeatInterval = seatHeartbeatIntervalSeconds(room.deliberationEffort)
  const heartbeatStart = Date.now()
  const runtimes: SeatRuntime[] = room.seats.map((seat) => ({
    seat,
    messages: [{ role: 'system', content: buildPrimer(project, room, seat, room.seats, providerForSeat(seat)) }],
    recent: [],
    heartbeat: initSeatHeartbeat(seat, heartbeatInterval, heartbeatStart),
  }))
  const rosterNames = room.seats.map((s) => s.name).join(', ')
  // Mirror live seat heartbeats onto the room so the supervisor + UI can read them.
  const publishSeatHeartbeats = () => {
    const map: Record<string, SeatHeartbeat> = {}
    for (const rt of runtimes) map[rt.heartbeat.seatId] = { ...rt.heartbeat }
    room.seatHeartbeats = map
  }
  publishSeatHeartbeats()

  // Live shared memory (decisions/plan/blockers) mutates during the run via the
  // memory_write room tool. Track the revision stamp each seat has seen so a change
  // is injected into that seat's context once, on its next turn.
  const sharedMemoryStamp = (): string =>
    room.sharedMemory ? Object.values(room.sharedMemory.blocks).map((b) => `${b.name}:${b.revision}`).sort().join('|') : ''
  const seenSharedMemoryStamps = new Map(runtimes.map((rt) => [rt.seat.id, sharedMemoryStamp()]))

  // SENSE: composition awareness + the task. This is the "sensory injection".
  for (const rt of runtimes) {
    rt.messages.push({
      role: 'user',
      content: `SENSE you are in a room with: ${rosterNames}. This is a live multi-agent working session. The others are real teammates replying for themselves, not you.\n\nTASK\n${problem}`,
    })
  }

  let tick = 0
  const seenRoomEventIds = new Set(room.events.map((item) => item.id))
  const internalSeats = new Set([...room.seats.map((seat) => seat.name), 'you', 'moderator', 'system'])
  const syncExternalRoomEvents = (): number => {
    const external = room.events.filter((item) =>
      !seenRoomEventIds.has(item.id) &&
      (item.kind === 'user' || item.kind === 'notice') &&
      !internalSeats.has(item.seat)
    )
    for (const item of external) {
      seenRoomEventIds.add(item.id)
      tick = Math.max(tick, item.tick || 0)
      broadcast(runtimes, item, '')
    }
    return external.length
  }
  const commit = (item: RoomEventItem) => {
    seenRoomEventIds.add(item.id)
    tick = Math.max(tick, item.tick || 0)
    room.events.push(item)
    send({ type: 'event', item })
  }
  let hadFatalSeatError = false
  let anySeatContributed = false
  let lastFinalBlockSignature = ''
  const reportedNoProvider = new Set<string>() // emit the "provider not configured" notice once per seat, not every pass
  // Seat ids that actually took a turn (recordSeatBeat) during the previous pass.
  // Only these get their stall window re-armed at the top of the next pass; a seat
  // that never ran (e.g. repeatedly timed out without recordSeatBeat) keeps counting
  // toward stalled so the BUG #53 guard can actually fire instead of being perpetually re-armed.
  let ranThisPass = new Set<string>()
  const rememberRun = () => {
    const previousMemory = room.memory ?? []
    const previousSummary = room.memorySummary ?? ''
    const entry = buildRoomRunMemoryEntry(room, problem)
    room.memory = mergeRoomMemory(room, entry)
    room.memorySummary = updateRoomMemorySummary(room.memorySummary, entry)
    const diff = buildRoomMemoryDiff(previousMemory, room.memory, previousSummary, room.memorySummary ?? '')
    send({ type: 'memory', memory: room.memory ?? [], memorySummary: room.memorySummary ?? '', diff })
  }
  commit({ id: randomUUID(), tick: ++tick, kind: 'task', seat: 'you', target: 'all', text: problem, createdAt: Date.now() })
  commit({
    id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: 'all',
    text: `Room deliberation: ${deliberation.label}. Budget ${deliberationBudgetText(deliberation)}. This is separate from model reasoning effort.`,
    createdAt: Date.now(),
  })

  // Configuration invariants (seat-count vs collaboration mode, deliberation-effort
  // fit). Non-blocking: surface the first violation on the error channel so the user
  // sees a misconfigured room, but still run — this preserves prior behavior where a
  // thin or single-seat room was allowed to proceed.
  const configError = validateRoomConfiguration(
    room.seats.length,
    normalizeCollaborationMode(room.collaborationMode),
    normalizeDeliberationEffort(room.deliberationEffort),
  )
  if (configError) send({ type: 'error', message: configError.message })

  if (!setRoomState('running')) {
    send({ type: 'done' })
    return
  }
  send({ type: 'status', state: 'running' })
  // Durable journal: record the room start alongside the live event stream so the run
  // is reconstructable after a crash. Fire-and-forget — a journal write never blocks or
  // breaks the room.
  void recordRoomEvent(room.id, {
    type: 'started',
    actor: 'room',
    payload: { state: 'running', seats: room.seats.length, effort: room.deliberationEffort, problem: problem.slice(0, 400) },
  })
  // Live push (additive): the room is now running — let the Command Centre reflect it
  // without waiting for the next rooms snapshot.
  emitRoomLifecycle('started', room.id, { state: 'running' })

  let pendingAddressed: string[] = []
  const order = (): SeatRuntime[] => {
    // Key by seat ID, not name: duplicate names must not collapse a distinct-model seat out of the loop.
    const map = new Map(runtimes.map((r) => [r.seat.id, r]))
    const ordered: SeatRuntime[] = []
    for (const id of pendingAddressed) { const r = map.get(id); if (r) { ordered.push(r); map.delete(id) } }
    pendingAddressed = []
    return [...ordered, ...map.values()]
  }

  try {
    for (let t = 0; t < maxTurns; t++) {
      if (signal.aborted) break
      if (deliberation.timeBudgetMs > 0 && Date.now() - startedAt >= deliberation.timeBudgetMs) {
        commit({
          id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: 'all',
          text: `Deliberation budget reached after ${t} room turn${t === 1 ? '' : 's'}. Continuing would require a higher room effort.`,
          createdAt: Date.now(),
        })
        break
      }
      let progressed = syncExternalRoomEvents() > 0
      // Final stretch: push seats to commit (create tickets + FINAL) instead of
      // deliberating until the budget hard-stops and the room ends with nothing done
      // (the classic 2-seat "should I make a ticket? / adjust scope?" stall).
      const closingOut = roomShouldCloseOut(t, maxTurns, Date.now() - startedAt, deliberation.timeBudgetMs)

      // Re-arm the stall window only for seats that actually took a turn last pass
      // (see rearmSeatHeartbeats). A seat that never ran last pass — e.g. one that
      // keeps timing out without ever calling recordSeatBeat — keeps counting toward
      // stalled so the BUG #53 guard below can fire. Re-arming every non-stalled seat
      // unconditionally is what previously made stall detection dead code.
      const passStart = Date.now()
      const ranLastPass = ranThisPass
      ranThisPass = new Set<string>()
      rearmSeatHeartbeats(runtimes, ranLastPass, passStart)

      // BUG #53: surface any seat whose turn has gone overdue past the grace
      // window. Non-blocking — emit a notice + error so the user/supervisor sees
      // the stalled seat instead of the room silently locking.
      const stalledSeats = evaluateSeatHeartbeats(runtimes, Date.now())
      if (stalledSeats.length) {
        for (const msg of stalledSeats) {
          commit({
            id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: 'all',
            text: msg, createdAt: Date.now(),
          })
          send({ type: 'error', message: msg })
          // Pull the seat name out of the "Seat "<name>": …" notice for the journal payload.
          const stalledSeat = msg.match(/^Seat "([^"]+)"/)?.[1]
          void recordRoomEvent(room.id, { type: 'seat_stalled', actor: 'room', payload: { seat: stalledSeat ?? null, note: msg } })
        }
        publishSeatHeartbeats()
      }

      for (const rt of order()) {
        if (signal.aborted) break
        if (syncExternalRoomEvents() > 0) progressed = true
        const seat = rt.seat
        const provider = providerForSeat(seat)
        if (!provider) {
          hadFatalSeatError = true
          // This seat can never take a turn, so it can never beat. Excuse it from
          // heartbeat/stall tracking, otherwise after the grace window it stays
          // flagged forever and blocks every other seat's FINAL via stalledSeatNames.
          if (rt.heartbeat.required) {
            rt.heartbeat.required = false
            rt.heartbeat.nextDueAt = null
            rt.heartbeat.missedSince = null
            publishSeatHeartbeats()
          }
          if (!reportedNoProvider.has(seat.id)) {
            reportedNoProvider.add(seat.id)
            send({ type: 'error', message: `Seat "${seat.name}": provider not configured.` })
          }
          continue
        }

        const contextLimit = deps.contextLimitForSeat?.(seat) ?? 0
        if (contextLimit > 0) {
          try {
            const compacted = await compactSeatRuntime(rt, provider, contextLimit, signal)
            if (compacted) {
              commit({
                id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: seat.name,
                text: `Compacted ${seat.name}'s private room context. Folded ${compacted.folded} earlier messages (~${kfmt(compacted.before)} -> ~${kfmt(compacted.after)} tokens).`,
                createdAt: Date.now(),
              })
            }
          } catch (e: any) {
            if (signal.aborted) break
            send({ type: 'error', message: `Seat "${seat.name}": context compaction skipped (${e?.message ?? e}).` })
          }
        }

        // Inject a fresh shared-memory snapshot when it changed since this seat last saw it.
        const memoryStamp = sharedMemoryStamp()
        if (memoryStamp && seenSharedMemoryStamps.get(seat.id) !== memoryStamp && room.sharedMemory) {
          seenSharedMemoryStamps.set(seat.id, memoryStamp)
          rt.messages.push({ role: 'user', content: `[room] shared memory updated:\n${renderSharedMemory(room.sharedMemory)}` })
        }

        const closingCall = closingOut
          ? ' CLOSING CALL: the room budget is nearly spent. Stop asking each other whether to proceed. If the work is agreed, create the execution tickets now with the ticket tool, then ACTION: FINAL with a one-paragraph summary. If you genuinely cannot proceed, ACTION: FINAL naming exactly what is blocking. Do not end the room with no tickets and no decision.'
          : ''
        const teamCue = roomTeamCue(room, seat)
        rt.messages.push({ role: 'user', content: `[moderator] ${seat.name}, your turn. Answer anything addressed to you, add something new (do not repeat yourself), then end with one ACTION line. Room deliberation is ${deliberation.label}: ${deliberationBudgetText(deliberation)}. Do not FINAL until major objections are resolved or named as dissent. This does not change your model reasoning effort.${closingCall} ${teamCue} Remember the TensorFold style contract: no em dashes and one-line code comments only.` })
        send({ type: 'turn', seat: seat.name })
        void recordRoomEvent(room.id, { type: 'turn', actor: 'seat', payload: { seat: seat.name, tick } })
        // Live push: a seat took the floor — drives the live turn indicator.
        emitRoomLifecycle('turn', room.id, { state: room.state, seat: seat.name })

        const timedOutNotice = () => {
          send({ type: 'error', message: `Seat "${seat.name}": turn timed out; skipped so other models can proceed.` })
          commit({ id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: seat.name, text: `${seat.name} timed out this turn and was skipped.`, createdAt: Date.now() })
        }
        let reply = ''
        try {
          const turn = await runSeatTurn(rt, provider)
          if (turn.timedOut) { timedOutNotice(); continue }
          reply = turn.reply
        } catch (e: any) {
          if (signal.aborted) break
          const recovered = await deps.recoverSeatModel?.(seat, e)
          if (recovered) {
            rt.seat = { ...rt.seat, providerId: recovered.provider.id, providerAccountId: recovered.providerAccountId, model: recovered.model }
            const item: RoomEventItem = {
              id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: seat.name,
              text: recovered.notice, createdAt: Date.now(),
            }
            commit(item)
            broadcast(runtimes, item, 'moderator')
            rt.messages.push({ role: 'user', content: `[moderator] ${recovered.notice} Retry your turn now.` })
            try {
              const retry = await runSeatTurn(rt, recovered.provider)
              if (retry.timedOut) { timedOutNotice(); continue }
              reply = retry.reply
            } catch (retryErr: any) {
              if (signal.aborted) break
              hadFatalSeatError = true
              send({ type: 'error', message: `Seat "${seat.name}": ${retryErr?.message ?? retryErr}` })
              continue
            }
          } else {
            hadFatalSeatError = true
            send({ type: 'error', message: `Seat "${seat.name}": ${e?.message ?? e}` })
            continue
          }
        }

        reply = reply.trim()
        if (!reply) {
          // Empty/reasoning-only output: treat as a silent YIELD. Do NOT push an empty
          // assistant message or commit an empty event; the seat did run, so only the
          // heartbeat is recorded (keeps stall detection honest).
          recordSeatBeat(rt.heartbeat, Date.now())
          ranThisPass.add(rt.seat.id)
          publishSeatHeartbeats()
          continue
        }
        anySeatContributed = true
        rt.messages.push({ role: 'assistant', content: reply })
        // BUG #53: the seat produced a turn — record the heartbeat (clears any stall).
        recordSeatBeat(rt.heartbeat, Date.now())
        // This seat actually ran this pass, so it is safe to re-arm its stall window
        // at the top of the next pass (see the re-arm above).
        ranThisPass.add(rt.seat.id)
        publishSeatHeartbeats()
        const action = gateAction(reply)

        const sig = signature(action.raw)
        if (action.verb !== 'FINAL' && rt.recent.includes(sig)) {
          rt.messages.push({ role: 'user', content: '[moderator] You already said that. Move the work forward or ACTION: YIELD.' })
          continue
        }
        rt.recent.push(sig); if (rt.recent.length > 3) rt.recent.shift()

        if (action.verb === 'TOOL' && deps.tools) {
          const m = action.text.match(/^(\w+)\s*([\s\S]*)$/)
          const name = m?.[1] || ''
          let args: any = {}
          try { if (m?.[2]?.trim()) args = JSON.parse(m[2]) } catch { args = {} }
          let result = `Unknown tool "${name}".`
          try { if (deps.tools[name]) result = await deps.tools[name](args, seat.name) } catch (e: any) { result = `Tool error: ${e?.message ?? e}` }
          const oneLine = result.split('\n')[0].slice(0, 140)
          // Give peers a substantial share of the tool result (not just a 140-char headline) — shared
          // tool/board evidence is how seats coordinate. Exclude the actor by id (not name).
          const peerShare = result.length > 1500 ? `${result.slice(0, 1500)}\n…(truncated)` : result
          rt.messages.push({ role: 'user', content: `[room] tool ${name} returned to you:\n${result}` })
          for (const other of runtimes) if (other.seat.id !== seat.id) other.messages.push({ role: 'user', content: `[room] ${seat.name} used ${name}:\n${peerShare}` })
          commit({
            id: randomUUID(),
            tick: ++tick,
            kind: 'tool',
            seat: seat.name,
            target: 'all',
            tool: name,
            args: args && typeof args === 'object' ? args : undefined,
            subAgentId: typeof args?.id === 'string' ? args.id : undefined,
            parentSeat: typeof args?.parentSeat === 'string' ? args.parentSeat : undefined,
            status: typeof args?.status === 'string' ? args.status : undefined,
            path: typeof args?.path === 'string' ? args.path : undefined,
            text: `${name}: ${oneLine}`,
            createdAt: Date.now(),
          })
          progressed = true
          continue
        }

        if (action.verb === 'FINAL') {
          const blockers = deps.finalGate?.() ?? []
          // Outcome gate (VST-019): a moded room may not FINAL without its expected
          // artifact — the missing pieces are named so the seat can fix the final.
          if (outcome) {
            const gate = roomOutcomeGate(outcome.id, { finalText: action.text })
            for (const item of gate.missing) blockers.push(`${outcome.label} outcome incomplete — the final is missing ${item}.`)
          }
          // BUG #53: do not let the room finalize while a teammate is stalled.
          const stalled = stalledSeatNames(runtimes, seat.name)
          for (const name of stalled) blockers.push(`Seat "${name}" is stalled and has not contributed.`)
          if (blockers.length) {
            const finalBlockSignature = blockerSignature(blockers)
            const text = `Final blocked: ${blockers.join(' ')}`
            rt.messages.push({ role: 'user', content: `[moderator] ${text} Resolve the blockers, then FINAL.` })
            if (finalBlockSignature && finalBlockSignature === lastFinalBlockSignature) {
              rt.messages.push({ role: 'user', content: '[moderator] The same final blocker is already visible in the room transcript. Work a remaining ticket, wait for worker evidence, or ACTION: YIELD.' })
              continue
            }
            lastFinalBlockSignature = finalBlockSignature
            const item: RoomEventItem = { id: randomUUID(), tick: ++tick, kind: 'check', seat: 'moderator', target: seat.name, text, createdAt: Date.now() }
            commit(item)
            broadcast(runtimes, item, 'moderator')
            progressed = true
            continue
          }
        }

        // Deliver the model's reasoning to peers (the prose), not only the ACTION-line summary.
        const roomText = action.prose && action.prose !== action.text
          ? (action.text ? `${action.prose}\n${action.text}` : action.prose)
          : (action.text || action.raw)
        const item: RoomEventItem = { id: randomUUID(), tick: ++tick, kind: action.verb.toLowerCase(), seat: seat.name, target: 'all', text: roomText, createdAt: Date.now() }
        if (action.verb === 'ASK' || action.verb === 'ASSIGN') {
          // Optional reply correlation: a trailing "RE: #<id>" cites the ask/assign
          // item being answered. The # prefix and uppercase RE: are required so prose
          // like "re: testing" does not correlate; only a unique id prefix resolves.
          const replyMatch = action.text.match(/\sRE:\s*#([\w-]{4,})\s*$/)
          if (replyMatch) {
            const token = replyMatch[1]
            const candidates = room.events.filter((e) => e.id === token || e.id.startsWith(token))
            if (candidates.length === 1) item.replyTo = candidates[0].id
          }
          // The word right after the verb is the addressee; only when it is not a seat
          // name do we fall back to scanning the body for seat mentions (so
          // "ASSIGN Alice ... since Bob is busy" targets Alice, not Bob).
          const peers = room.seats.filter((other) => other.name !== seat.name)
          const firstToken = action.text.split(/\s+/)[0] ?? ''
          const direct = peers.find((other) => firstToken.toLowerCase() === other.name.toLowerCase())
          const addressed = direct ? [direct] : peers.filter((other) => new RegExp(`\\b${escapeReg(other.name)}\\b`, 'i').test(action.text))
          for (const other of addressed) {
            item.target = other.name
            if (!pendingAddressed.includes(other.id)) pendingAddressed.push(other.id)
          }
        }
        commit(item)

        if (action.verb === 'FINAL') {
          if (!setRoomState('complete')) {
            send({ type: 'done' })
            return
          }
          room.finalText = action.text
          rememberRun()
          broadcast(runtimes, item, seat.id)
          send({ type: 'final', text: action.text })
          send({ type: 'status', state: 'complete' })
          send({ type: 'done' })
          void recordRoomEvent(room.id, { type: 'finalized', actor: 'seat', payload: { state: 'complete', seat: seat.name } })
          // Live push: a seat reached FINAL — the room is complete.
          emitRoomLifecycle('finalized', room.id, { state: 'complete', seat: seat.name })
          return
        }
        if (action.verb !== 'YIELD') progressed = true
        broadcast(runtimes, item, seat.id)

        // Supervisor mode: an ASSIGN is enforced by the runtime, not by prompt
        // convention — but only the planning-lane (supervisor) seat's ASSIGN auto-starts
        // the addressed seat's ticket worker. ASSIGNs from other seats, and ASSIGNs that
        // resolve to no seat, get a moderator notice instead of a silent no-op.
        if (action.verb === 'ASSIGN' && normalizeCollaborationMode(room.collaborationMode) === 'supervisor') {
          const hasTarget = !!item.target && item.target !== 'all' && item.target !== seat.name
          let notice: string | null = null
          if (inferSeatLane(seat) !== 'planning') {
            notice = `ASSIGN from ${seat.name} noted, but only the supervisor seat's assignments auto-start workers. Create or claim a ticket instead.`
          } else if (!hasTarget) {
            notice = `ASSIGN from ${seat.name} did not name a room seat, so no worker was auto-started.`
          } else if (deps.onAssign) {
            try {
              notice = await deps.onAssign({ seat: seat.name, target: item.target, text: action.text })
            } catch (e: any) {
              send({ type: 'error', message: `ASSIGN auto-start failed: ${e?.message ?? e}` })
            }
          }
          if (notice) {
            const autoItem: RoomEventItem = {
              id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: hasTarget ? item.target : seat.name,
              text: notice, tool: 'auto_assign', createdAt: Date.now(),
            }
            commit(autoItem)
            broadcast(runtimes, autoItem, 'moderator')
          }
        }
      }
      if (!progressed) break
    }

    // Reaching here means no seat emitted a successful FINAL (FINAL returns early).
    // A room that burned its budget deliberating without a decision is NOT "complete" —
    // that green badge with no tickets is misleading (the 2-seat stall). Mark it
    // 'waiting_user' ("Needs you") so the user can raise the effort or give direction.
    // 'failed' only when NO seat ever contributed; 'stopped' when the user aborted.
    if (!signal.aborted && anySeatContributed && !(hadFatalSeatError && !anySeatContributed)) {
      commit({
        id: randomUUID(), tick: ++tick, kind: 'notice', seat: 'moderator', target: 'all',
        text: 'Room ended without a final decision — no FINAL was reached within the budget. Raise the room effort or give the room more direction to continue.',
        createdAt: Date.now(),
      })
    }
    if (!setRoomState(signal.aborted ? 'stopped' : (hadFatalSeatError && !anySeatContributed) ? 'failed' : anySeatContributed ? 'waiting_user' : 'complete')) {
      send({ type: 'done' })
      return
    }
    rememberRun()
    send({ type: 'status', state: room.state })
    send({ type: 'done' })
    void recordRoomEvent(room.id, { type: 'finalized', actor: 'room', payload: { state: room.state, reason: 'no_final', aborted: signal.aborted } })
    // Live push: the room ended without a FINAL (waiting_user / complete / stopped / failed).
    emitRoomLifecycle('finalized', room.id, { state: room.state })
  } catch (e: any) {
    const failedApplied = setRoomState('failed')
    const terminalState = room.state
    send({ type: 'error', message: e?.message ? String(e.message) : String(e), fatal: true })
    send({ type: 'done' })
    void recordRoomEvent(room.id, { type: 'finalized', actor: 'room', payload: { state: terminalState, reason: failedApplied ? 'error' : 'error_after_terminal', error: e?.message ? String(e.message) : String(e) } })
    // Live push the state that was legally retained. A post-FINAL delivery error must not
    // rewrite a completed room to failed in the event stream either.
    emitRoomLifecycle('finalized', room.id, { state: terminalState })
  }
}
