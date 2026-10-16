import { randomUUID } from 'node:crypto'
import type { Provider, Room, RoomCollaborationMode, RoomDeliberationEffort, RoomEventItem, RoomMemoryDiff, RoomMemoryEntry, RoomSeatLane, Seat } from './types'

export const DEFAULT_ROOM_COLLABORATION_MODE: RoomCollaborationMode = 'frontier_plans_local_builds'
export const DEFAULT_ROOM_DELIBERATION_EFFORT: RoomDeliberationEffort = 'normal'

export const ROOM_COLLABORATION_MODES: Array<{ id: RoomCollaborationMode; label: string; detail: string }> = [
  {
    id: 'frontier_plans_local_builds',
    label: 'Frontier plans, local builds',
    detail: 'Use frontier/cloud seats for planning and review, and local seats for private implementation/debugging.',
  },
  {
    id: 'balanced',
    label: 'Balanced room',
    detail: 'Let every seat plan, inspect, challenge, and summarize based on its role.',
  },
  {
    id: 'local_private',
    label: 'Local/private room',
    detail: 'Keep sensitive work grounded in local models and avoid pushing private details to cloud seats.',
  },
  {
    id: 'frontier_council',
    label: 'Frontier council',
    detail: 'Use cloud seats as independent critics for architecture, strategy, and final review.',
  },
  {
    id: 'supervisor',
    label: 'Supervisor',
    detail: 'One planning seat decomposes the goal into tickets and assigns them; assigned seats get worker threads started automatically.',
  },
]

export interface RoomDeliberationProfile {
  id: RoomDeliberationEffort
  label: string
  detail: string
  maxTurns: number
  timeBudgetMs: number
  consensusThreshold: number
}

export const ROOM_DELIBERATION_PROFILES: RoomDeliberationProfile[] = [
  {
    id: 'quick',
    label: 'Quick',
    detail: 'A short pass for simple work. Agents should converge fast and avoid broad debate.',
    maxTurns: 4,
    timeBudgetMs: 2 * 60 * 1000,
    consensusThreshold: 0.6,
  },
  {
    id: 'normal',
    label: 'Normal',
    detail: 'The default huddle budget. Agents should challenge assumptions, resolve obvious objections, then land the next step.',
    maxTurns: 8,
    timeBudgetMs: 5 * 60 * 1000,
    consensusThreshold: 0.7,
  },
  {
    id: 'deep',
    label: 'Deep',
    detail: 'A stronger planning and review pass. Agents should explore alternatives and name dissent before finalizing.',
    maxTurns: 16,
    timeBudgetMs: 15 * 60 * 1000,
    consensusThreshold: 0.8,
  },
  {
    id: 'max',
    label: 'Max',
    detail: 'A long council session for hard problems. Still capped for safety, but much less eager to finalize.',
    maxTurns: 48,
    timeBudgetMs: 0,
    consensusThreshold: 0.9,
  },
]

const PLANNING_ROLE = /\b(plan|planner|architect|strategy|strategist|product|lead|manager|pm|orchestrator|director)\b/i
const DEVELOPMENT_ROLE = /\b(build|builder|develop|developer|engineer|implement|coder|code|debug|fix|frontend|backend|fullstack)\b/i
const REVIEW_ROLE = /\b(review|reviewer|qa|test|tester|security|auditor|critic|verify|validation|red team)\b/i
const RESEARCH_ROLE = /\b(research|researcher|analyst|scout|investigate|discovery|data)\b/i
const OPS_ROLE = /\b(ops|operator|assistant|coordinator|dispatch|support|automation|mission)\b/i

export function normalizeCollaborationMode(mode?: RoomCollaborationMode | null): RoomCollaborationMode {
  return mode && ROOM_COLLABORATION_MODES.some((m) => m.id === mode) ? mode : DEFAULT_ROOM_COLLABORATION_MODE
}

export function normalizeDeliberationEffort(effort?: RoomDeliberationEffort | null): RoomDeliberationEffort {
  return effort && ROOM_DELIBERATION_PROFILES.some((profile) => profile.id === effort) ? effort : DEFAULT_ROOM_DELIBERATION_EFFORT
}

export function deliberationProfile(effort?: RoomDeliberationEffort | null): RoomDeliberationProfile {
  return ROOM_DELIBERATION_PROFILES.find((profile) => profile.id === normalizeDeliberationEffort(effort)) ?? ROOM_DELIBERATION_PROFILES[1]
}

// True on a room's final stretch — the last allowed turn, or once at least half the
// time budget is spent. The turn loop uses this to inject a "closing call" so seats
// commit (create tickets + FINAL) instead of politely deliberating until the budget
// hard-stops and the room ends with nothing done.
export function roomShouldCloseOut(turnIndex: number, maxTurns: number, elapsedMs: number, timeBudgetMs: number): boolean {
  if (maxTurns > 0 && turnIndex >= maxTurns - 1) return true
  if (timeBudgetMs > 0 && elapsedMs >= timeBudgetMs * 0.5) return true
  return false
}

export function deliberationBudgetText(profile: RoomDeliberationProfile): string {
  const minutes = profile.timeBudgetMs > 0 ? Math.round(profile.timeBudgetMs / 60000) : 0
  const time = minutes > 0 ? ` or about ${minutes} minute${minutes === 1 ? '' : 's'}` : ''
  return `up to ${profile.maxTurns} room turns${time}, aiming for ${Math.round(profile.consensusThreshold * 100)}% agreement`
}

export function inferSeatLane(seat: Pick<Seat, 'role' | 'lane'>): RoomSeatLane {
  if (seat.lane) return seat.lane
  const role = seat.role || ''
  if (PLANNING_ROLE.test(role)) return 'planning'
  if (REVIEW_ROLE.test(role)) return 'review'
  if (RESEARCH_ROLE.test(role)) return 'research'
  if (OPS_ROLE.test(role)) return 'operations'
  if (DEVELOPMENT_ROLE.test(role)) return 'development'
  return 'development'
}

export function collaborationModeLabel(mode?: RoomCollaborationMode | null): string {
  return ROOM_COLLABORATION_MODES.find((m) => m.id === normalizeCollaborationMode(mode))?.label ?? ROOM_COLLABORATION_MODES[0].label
}

export function collaborationModeDetail(mode?: RoomCollaborationMode | null): string {
  return ROOM_COLLABORATION_MODES.find((m) => m.id === normalizeCollaborationMode(mode))?.detail ?? ROOM_COLLABORATION_MODES[0].detail
}

export function seatCollaborationInstructions(room: Pick<Room, 'collaborationMode' | 'deliberationEffort'>, seat: Seat, provider?: Provider): string {
  const mode = normalizeCollaborationMode(room.collaborationMode)
  const deliberation = deliberationProfile(room.deliberationEffort)
  const lane = inferSeatLane(seat)
  const providerKind = provider?.kind ?? 'cloud'
  const shared = [
    `COLLABORATION MODE: ${collaborationModeLabel(mode)}`,
    collaborationModeDetail(mode),
    `ROOM DELIBERATION: ${deliberation.label}`,
    `${deliberation.detail} This controls group agreement, not your model reasoning effort. Budget: ${deliberationBudgetText(deliberation)}.`,
  ]

  if (mode === 'frontier_plans_local_builds') {
    const laneRule = lane === 'planning'
      ? 'You are the planning lane: decompose the goal, expose assumptions, define acceptance checks, and create focused tickets before implementation.'
      : lane === 'review'
        ? 'You are the review lane: challenge weak claims, demand evidence, and block final answers when tests, checks, or open tickets are missing.'
        : lane === 'research'
          ? 'You are the research lane: gather facts, compare options, and turn uncertainty into narrow tickets or questions.'
          : 'You are the local build lane: ground the plan in repo/local evidence, claim concrete tickets, and report exact files/checks needed for an approved work thread.'
    const cloudRule = providerKind === 'cloud'
      ? 'Because this is a cloud/frontier seat, keep private local details minimal and focus on planning, review, and high-level reasoning.'
      : 'Because this is a local seat, prefer concrete inspection, sensitive context, and implementation detail over broad strategy.'
    return [...shared, laneRule, cloudRule].join('\n')
  }

  if (mode === 'local_private') {
    return [
      ...shared,
      'Treat sensitive repo, connector, and personal context as local-room context. Do not ask cloud seats to process secrets, raw private communications, or credential material.',
      lane === 'review' ? 'Focus on privacy and evidence gates before any final answer.' : 'Prefer local inspection and create tickets when a risky action needs approval.',
    ].join('\n')
  }

  if (mode === 'supervisor') {
    const supervisorRule = lane === 'planning'
      ? 'You are the supervisor: decompose the goal into focused tickets with create_ticket, then hand each ticket to a seat with ACTION: ASSIGN <seat> <ticket title>. Assigned tickets are auto-started as worker threads for that seat, so only assign a ticket once it is ready to build.'
      : 'The supervisor decomposes the goal and assigns tickets. When a ticket is assigned to you, a worker thread is auto-started in your name. Track it with left_to_work, attach evidence when it lands, and name blockers early.'
    return [
      ...shared,
      supervisorRule,
      'The supervisor coordinates; editing happens in the auto-started ticket worker threads, not in room chat.',
    ].join('\n')
  }

  if (mode === 'frontier_council') {
    return [
      ...shared,
      'Act as an independent council member. Give a distinct view, name tradeoffs, and converge only when evidence is strong.',
      lane === 'development' ? 'If implementation is required, translate council decisions into concrete tickets for an approved work thread.' : 'Keep your contribution sharp and non-duplicative.',
    ].join('\n')
  }

  return [
    ...shared,
    'Coordinate naturally with the room: claim work before doing it, avoid repeating other seats, and make decisions durable by naming them explicitly.',
  ].join('\n')
}

export function formatRoomMemory(room: Pick<Room, 'memory' | 'memorySummary'>, maxEntries = 8): string {
  const entries = (room.memory ?? [])
    .filter((entry) => entry.text.trim())
    .slice(0, maxEntries)
    .map((entry) => `- ${entry.kind} (${entry.source}): ${entry.text.trim()}`)
  const parts = [
    room.memorySummary?.trim() ? `Running summary:\n${room.memorySummary.trim()}` : '',
    entries.length ? `Recent durable entries:\n${entries.join('\n')}` : '',
  ].filter(Boolean)
  return parts.join('\n\n').slice(0, 5000)
}

function trimText(text: string, limit: number): string {
  const clean = text.replace(/\s+/g, ' ').trim()
  return clean.length <= limit ? clean : clean.slice(0, limit - 1).trimEnd() + '...'
}

function unique<T>(items: T[]): T[] {
  return [...new Set(items)]
}

function eventSummary(event: RoomEventItem): string {
  const label = event.kind === 'tool' && event.tool ? `${event.tool}` : event.kind
  return `${event.seat} ${label}: ${trimText(event.text, 180)}`
}

export function buildRoomRunMemoryEntry(room: Room, problem: string, now = Date.now()): RoomMemoryEntry | null {
  const events = room.events ?? []
  const final = trimText(room.finalText || [...events].reverse().find((e) => e.kind === 'final')?.text || '', 700)
  const decisions = events
    .filter((e) => ['final', 'check', 'vote', 'object', 'claim'].includes(e.kind))
    .slice(-6)
    .map(eventSummary)
  const files = unique(events.map((e) => e.path).filter((p): p is string => !!p)).slice(0, 8)
  const tools = unique(events.map((e) => e.tool).filter((t): t is string => !!t)).slice(0, 8)
  const asks = events.filter((e) => e.kind === 'ask').slice(-3).map(eventSummary)

  const lines = [
    `Request: ${trimText(problem || room.problem || 'Room run', 500)}`,
    final ? `Outcome: ${final}` : '',
    decisions.length ? `Decisions/findings: ${decisions.join(' | ')}` : '',
    files.length ? `Files touched/inspected: ${files.join(', ')}` : '',
    tools.length ? `Tools used: ${tools.join(', ')}` : '',
    asks.length ? `Open/recent questions: ${asks.join(' | ')}` : '',
  ].filter(Boolean)

  if (!lines.length) return null
  return {
    id: randomUUID(),
    kind: 'summary',
    text: trimText(lines.join('\n'), 1800),
    source: 'room',
    createdAt: now,
    updatedAt: now,
  }
}

export function mergeRoomMemory(room: Room, entry: RoomMemoryEntry | null, maxEntries = 24): RoomMemoryEntry[] {
  if (!entry) return room.memory ?? []
  const existing = room.memory ?? []
  const duplicate = existing.find((item) => item.text === entry.text)
  if (duplicate) return existing
  return [entry, ...existing].slice(0, maxEntries)
}

export function buildRoomMemoryDiff(
  previousMemory: RoomMemoryEntry[] | undefined,
  nextMemory: RoomMemoryEntry[] | undefined,
  previousSummary = '',
  nextSummary = '',
): RoomMemoryDiff {
  const previous = previousMemory ?? []
  const next = nextMemory ?? []
  const previousIds = new Set(previous.map((entry) => entry.id))
  const nextIds = new Set(next.map((entry) => entry.id))
  return {
    added: next.filter((entry) => !previousIds.has(entry.id)),
    removed: previous.filter((entry) => !nextIds.has(entry.id)),
    summaryChanged: previousSummary.trim() !== nextSummary.trim(),
    previousCount: previous.length,
    nextCount: next.length,
  }
}

export function updateRoomMemorySummary(previous: string | undefined, entry: RoomMemoryEntry | null): string {
  if (!entry) return previous ?? ''
  const prior = (previous ?? '').trim()
  const next = `- ${entry.text.trim()}`
  return [next, prior].filter(Boolean).join('\n').slice(0, 5000)
}
