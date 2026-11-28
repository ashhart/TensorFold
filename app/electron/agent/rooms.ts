import { createRoomSeatTurn, compactSeatRuntime, type SeatRuntime } from './room-seat-runtime'
import { seatHeartbeatIntervalSeconds, initSeatHeartbeat, recordSeatBeat, stalledSeatNames, rearmSeatHeartbeats, evaluateSeatHeartbeats } from './room-heartbeats'
export { seatHeartbeatIntervalSeconds, initSeatHeartbeat, recordSeatBeat, stalledSeatNames, rearmSeatHeartbeats, evaluateSeatHeartbeats } from './room-heartbeats'
export type { SeatRuntimeLike } from './room-heartbeats'
import { buildPrimer, gateAction, renderForRoom, signature, blockerSignature, oneLine, roomTeamCue, escapeReg } from './room-protocol'
import { randomUUID } from 'node:crypto'
import { captureAgentReasoningEffort } from '../../shared/reasoning-effort'
import type { streamChat } from './openai'
import { kfmt } from './compaction'
import type { Project, Provider, Room, RoomEventItem, RoomState, RoomStreamEvent, Seat, SeatHeartbeat } from './types'
import { buildRoomMemoryDiff, buildRoomRunMemoryEntry, deliberationBudgetText, deliberationProfile, inferSeatLane, mergeRoomMemory, normalizeCollaborationMode, normalizeDeliberationEffort, roomShouldCloseOut, updateRoomMemorySummary } from './room-collaboration'
import { validateRoomConfiguration, validateRoomTransition } from './room-invariants'
import { roomOutcomeGate, roomOutcomeProfile } from './room-outcomes'
import { renderSharedMemory } from './shared-room-memory'
import type { CliSeatTurnParams } from './room-cli-seat'
import { recordRoomEvent } from './orchestration-journal'
import { emitRoomLifecycle } from '../main/host/orchestration-events'

interface SeatModelRecovery {
  provider: Provider
  providerAccountId: string | null
  model: string
  notice: string
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

// Distributive Omit so each union member keeps its own payload keys
// (a plain Omit<Union, K> collapses to the members' common keys only).
export type RoomSend = RoomStreamEvent extends infer T ? (T extends any ? Omit<T, 'roomId'> : never) : never

export interface RoomRunDeps {
  room: Room
  project: Project
  providers: Provider[]
  problem: string
  temperature: number
  reasoningEffort: string
  reasoningEffortForSeat?: (seat: Seat) => string
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
  cliSeatTurn?: (params: CliSeatTurnParams, onDelta: (text: string) => void, onActivity?: () => void) => Promise<string>
  /** Per-seat idle window (ms), refreshed by streamed chunks; a stalled seat is skipped, not a cap on a turn. */
  turnTimeoutMs?: number
}

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
  const runSeatTurn = createRoomSeatTurn(deps, send)

  const heartbeatInterval = seatHeartbeatIntervalSeconds(room.deliberationEffort)
  const heartbeatStart = Date.now()
  const runtimes: SeatRuntime[] = room.seats.map((seat) => ({
    seat,
    reasoningEffort: deps.reasoningEffortForSeat?.(seat) ?? captureAgentReasoningEffort(seat, deps.reasoningEffort),
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
