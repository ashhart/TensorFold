import { relativeTime } from '../../lib/format'
import { allowedProviders, projectDefaultModel } from '../../lib/policy'
import { providerAccountChoices } from '../../lib/provider-accounts'
import type {
  AgentRoleProfileId,
  FanOutBudget,
  Job,
  Project,
  Provider,
  Room,
  RoomCollaborationMode, RoomMode,
  RoomDeliberationEffort,
  RoomSeatLane,
  RoomState,
  Seat,
  Ticket,
} from '../../lib/types'

export const FILE_TOOL_VERB: Record<string, string> = { edit_file: 'edited', write_file: 'created', read_file: 'read' }
const TINTS = ['#3a86ff', '#0bb07b', '#e8a33d', '#e5564f', '#8b5cf6', '#d6457f', '#1597a5', '#6366f1']
export const ACTIVE_ROOM_STATES = new Set<RoomState>(['queued', 'running'])
export const ROLE_PROFILES: Array<{ id: AgentRoleProfileId; label: string; role: string; lane: RoomSeatLane; name: string }> = [
  { id: 'planner', label: 'Planner', role: 'Planner', lane: 'planning', name: 'planner' },
  { id: 'builder', label: 'Builder', role: 'Engineer', lane: 'development', name: 'builder' },
  { id: 'reviewer', label: 'Reviewer', role: 'Reviewer', lane: 'review', name: 'reviewer' },
  { id: 'researcher', label: 'Researcher', role: 'Researcher', lane: 'research', name: 'researcher' },
  { id: 'operator', label: 'Operator', role: 'Operator', lane: 'operations', name: 'operator' },
]
export const FAN_OUT_OPTIONS: Array<{ id: FanOutBudget; label: string }> = [
  { id: 'single', label: 'Single' },
  { id: 'paired', label: 'Paired' },
  { id: 'council', label: 'Council' },
  { id: 'swarm', label: 'Swarm' },
]
export const ROOM_STATE_LABEL: Record<RoomState, string> = {
  idle: 'Idle',
  created: 'Created',
  ready: 'Ready',
  degraded: 'Degraded',
  queued: 'Queued',
  running: 'Running',
  waiting_user: 'Needs you',
  waiting_external: 'Waiting',
  complete: 'Complete',
  stopped: 'Stopped',
  failed: 'Failed',
  archived: 'Archived',
  error: 'Failed',
}
const LEFT_TO_WORK_RANK: Record<Ticket['status'], number> = { needs_clarification: 0, review: 1, in_progress: 2, ready: 3, todo: 4, completed: 5 }
const ACTIVE_TICKET_STATUSES = new Set<Ticket['status']>(['in_progress', 'review'])
const ACTIVE_TICKET_RUN_STATES = new Set<Exclude<Ticket['runState'], null>>(['queued', 'running'])
const ROOM_JOB_STATUS_LABEL: Record<Job['status'], string> = {
  queued: 'Queued',
  running: 'Running',
  waiting_approval: 'Waiting approval',
  paused: 'Paused',
  stalled: 'Stalled',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  dead_letter: 'Dead letter',
}
export const ACTIVE_ROOM_JOB_STATES = new Set<Job['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled'])
export const DEFAULT_ROOM_COLLABORATION_MODE: RoomCollaborationMode = 'frontier_plans_local_builds'
export const DEFAULT_ROOM_DELIBERATION_EFFORT: RoomDeliberationEffort = 'normal'
export const DEFAULT_ROOM_WORKER_BUDGET = { maxActive: 2, maxStarts: 4 }
export const ROOM_COLLABORATION_OPTIONS: Array<{ id: RoomCollaborationMode; label: string; detail: string }> = [
  { id: 'frontier_plans_local_builds', label: 'Frontier plans, local builds', detail: 'Cloud/frontier seats plan and review; local seats build with private context.' },
  { id: 'balanced', label: 'Balanced room', detail: 'Every seat plans, inspects, challenges, and summarizes by role.' },
  { id: 'local_private', label: 'Local/private room', detail: 'Keep sensitive work grounded in local models and local evidence.' },
  { id: 'frontier_council', label: 'Frontier council', detail: 'Use cloud seats as independent strategy and review critics.' },
  { id: 'supervisor', label: 'Supervisor', detail: 'A planner seat assigns tickets; workers auto-start for the assigned seat.' },
]
export { ROOM_MODE_OPTIONS, roomModeDefaultEffort } from '../../lib/room-modes'

export const ROOM_DELIBERATION_OPTIONS: Array<{ id: RoomDeliberationEffort; label: string; detail: string; maxTurns: number; timeBudgetMs: number; consensusThreshold: number }> = [
  { id: 'quick', label: 'Quick', detail: 'Short agreement pass for simple work.', maxTurns: 4, timeBudgetMs: 2 * 60 * 1000, consensusThreshold: 0.6 },
  { id: 'normal', label: 'Normal', detail: 'Default room loop: challenge, resolve, then land the next step.', maxTurns: 8, timeBudgetMs: 5 * 60 * 1000, consensusThreshold: 0.7 },
  { id: 'deep', label: 'Deep', detail: 'More rounds for planning, review, and named dissent.', maxTurns: 16, timeBudgetMs: 15 * 60 * 1000, consensusThreshold: 0.8 },
  { id: 'max', label: 'Max', detail: 'Long council mode for hard problems, still capped for safety.', maxTurns: 48, timeBudgetMs: 0, consensusThreshold: 0.9 },
]
// Friendly auto-names so agents address each other by name (e.g. "Aria, what do you think?").
const NAMES = ['Aria', 'Milo', 'Nova', 'Felix', 'Iris', 'Leo', 'Sage', 'June', 'Theo', 'Wren']
export function suggestSeatName(seats: Seat[]): string {
  const used = new Set(seats.map((s) => s.name.toLowerCase()))
  return NAMES.find((n) => !used.has(n.toLowerCase())) || `agent-${seats.length + 1}`
}
export function availableSeatName(preferred: string, seats: Seat[]): string {
  const used = new Set(seats.map((s) => s.name.toLowerCase()))
  if (preferred && !used.has(preferred.toLowerCase())) return preferred
  return suggestSeatName(seats)
}
function seatColor(name: string): string {
  let h = 0
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0
  return TINTS[h % TINTS.length]
}
function initials(name: string): string {
  return name.replace(/[^a-zA-Z0-9]/g, ' ').trim().split(/\s+/).slice(0, 2).map((p) => p[0]?.toUpperCase()).join('') || '?'
}
export function collaborationOption(mode?: RoomCollaborationMode | null) {
  return ROOM_COLLABORATION_OPTIONS.find((o) => o.id === (mode ?? DEFAULT_ROOM_COLLABORATION_MODE)) ?? ROOM_COLLABORATION_OPTIONS[0]
}
export function deliberationOption(effort?: RoomDeliberationEffort | null) {
  return ROOM_DELIBERATION_OPTIONS.find((o) => o.id === (effort ?? DEFAULT_ROOM_DELIBERATION_EFFORT)) ?? ROOM_DELIBERATION_OPTIONS[1]
}
export function deliberationBudgetLabel(effort?: RoomDeliberationEffort | null): string {
  const profile = deliberationOption(effort)
  const minutes = profile.timeBudgetMs > 0 ? Math.round(profile.timeBudgetMs / 60000) : 0
  const time = minutes > 0 ? `${minutes}m` : 'open'
  return `${profile.maxTurns} turns · ${time} · ${Math.round(profile.consensusThreshold * 100)}%`
}
export function roomWorkerBudget(room: Room): NonNullable<Room['workerBudget']> {
  const configured = room.workerBudget ?? DEFAULT_ROOM_WORKER_BUDGET
  const maxActive = Math.max(1, Math.min(8, Math.floor(Number(configured.maxActive) || DEFAULT_ROOM_WORKER_BUDGET.maxActive)))
  const maxStarts = Math.max(maxActive, Math.min(32, Math.floor(Number(configured.maxStarts) || DEFAULT_ROOM_WORKER_BUDGET.maxStarts)))
  return { maxActive, maxStarts }
}
export function roomWorkerBudgetLabel(room: Room, activeWorkers = 0): string {
  const budget = roomWorkerBudget(room)
  return `${activeWorkers}/${budget.maxActive} active · ${budget.maxStarts} starts`
}
export function pickProvider(providers: Provider[], want: 'cloud' | 'local', fallback?: { providerId: string | null; providerAccountId?: string | null; model: string | null }): { provider: Provider | null; providerAccountId: string | null; model: string } {
  const preferred = providers.find((p) => p.kind === want && p.models.length > 0)
  const byDefault = fallback?.providerId ? providers.find((p) => p.id === fallback.providerId && p.models.length > 0) : null
  const provider = preferred ?? byDefault ?? providers.find((p) => p.models.length > 0) ?? providers[0] ?? null
  const accounts = provider ? providerAccountChoices(provider) : []
  const requestedAccountId = provider && fallback?.providerId === provider.id ? fallback.providerAccountId ?? provider.defaultAccountId ?? null : provider?.defaultAccountId ?? null
  const account = accounts.find((choice) => choice.id === requestedAccountId) ?? accounts[0] ?? null
  const models = account?.models?.length ? account.models : (provider?.models ?? [])
  const model = provider && fallback?.providerId === provider.id && fallback.model && models.includes(fallback.model) ? fallback.model : (models[0] ?? '')
  return { provider, providerAccountId: account?.id ?? null, model }
}
export function presetSeats(providers: Provider[], project: Project | null | undefined): Seat[] {
  const allowed = allowedProviders(providers, project)
  const fallback = projectDefaultModel(project, providers)
  const planner = pickProvider(allowed, 'cloud', fallback)
  const builder = pickProvider(allowed, 'local', fallback)
  const reviewer = pickProvider(allowed, 'cloud', fallback)
  const now = crypto.randomUUID()
  const mk = (suffix: string, name: string, role: string, lane: RoomSeatLane, pick: { provider: Provider | null; providerAccountId: string | null; model: string }): Seat | null =>
    pick.provider && pick.model ? { id: `${now}-${suffix}`, name, role, lane, providerId: pick.provider.id, providerAccountId: pick.providerAccountId, model: pick.model } : null
  return [
    mk('planner', 'Planner', 'Planner - decomposes goals, acceptance checks, and handoffs', 'planning', planner),
    mk('builder', 'Builder', 'Builder - grounds work in repo/local context and implementation tickets', 'development', builder),
    mk('reviewer', 'Reviewer', 'Reviewer - checks evidence, risks, tests, and gaps', 'review', reviewer),
  ].filter((seat): seat is Seat => !!seat)
}
// Build a ready-to-add seat for a role profile, with a role-derived name (deduped
// with a numeric suffix) and a model that fits the role — local for builders and
// operators, frontier/cloud for planners, reviewers and researchers (#52/#47).
export function seatForRole(profileId: AgentRoleProfileId, seats: Seat[], providers: Provider[], project: Project | null | undefined): Seat | null {
  const profile = ROLE_PROFILES.find((p) => p.id === profileId) ?? ROLE_PROFILES[1]
  const want: 'cloud' | 'local' = profile.lane === 'development' || profile.lane === 'operations' ? 'local' : 'cloud'
  const pick = pickProvider(allowedProviders(providers, project), want, projectDefaultModel(project, providers))
  if (!pick.provider || !pick.model) return null
  const used = new Set(seats.map((s) => s.name.toLowerCase()))
  let name = profile.name
  if (used.has(name.toLowerCase())) {
    for (let i = 2; i < 99; i++) { if (!used.has(`${profile.name}-${i}`)) { name = `${profile.name}-${i}`; break } }
  }
  return { id: crypto.randomUUID(), name, role: profile.role, lane: profile.lane, roleProfileId: profile.id, fanOutBudget: 'single', providerId: pick.provider.id, providerAccountId: pick.providerAccountId, model: pick.model }
}

export function modelChoiceValue(providerId: string, accountId: string | null | undefined, model: string) {
  return JSON.stringify([providerId, accountId ?? null, model])
}
export function parseModelChoice(value: string): { providerId: string; providerAccountId: string | null; model: string } | null {
  try {
    const [providerId, providerAccountId, model] = JSON.parse(value)
    if (typeof providerId === 'string' && (typeof providerAccountId === 'string' || providerAccountId === null) && typeof model === 'string') {
      return { providerId, providerAccountId, model }
    }
  } catch {
    // Ignore malformed select values.
  }
  return null
}
export function seatModelChoice(seat: Seat, providers: Provider[]) {
  const provider = providers.find((p) => p.id === seat.providerId)
  if (!provider) return ''
  return modelChoiceValue(provider.id, seat.providerAccountId ?? null, seat.model)
}
export function Avatar({ name, size = 26 }: { name: string; size?: number }) {
  const c = seatColor(name)
  return (
    <span className="flex shrink-0 items-center justify-center rounded-full font-semibold text-white" style={{ width: size, height: size, background: c, fontSize: size * 0.4 }}>
      {initials(name)}
    </span>
  )
}

export interface RoomConflictBlocker {
  ticketId: string
  ticketTitle: string
  text: string
}

export function ticketStatusLabel(status: Ticket['status']): string {
  return status.replace(/_/g, ' ')
}

export function isActiveRunState(state: Ticket['runState']): state is Exclude<Ticket['runState'], null> {
  return !!state && ACTIVE_TICKET_RUN_STATES.has(state)
}

export function roomLeftToWork(tickets: Ticket[]): Ticket[] {
  return [...tickets]
    .filter((ticket) => ticket.status !== 'completed')
    .sort((a, b) => (LEFT_TO_WORK_RANK[a.status] ?? 9) - (LEFT_TO_WORK_RANK[b.status] ?? 9) || a.order - b.order || b.updatedAt - a.updatedAt)
}

export function ticketFileRefsForDisplay(ticket: Ticket): string[] {
  const refs: string[] = []
  const push = (value: unknown) => {
    if (typeof value !== 'string') return
    const text = value.trim()
    if (!text) return
    if (looksLikeTicketPath(text)) refs.push(normalizeTicketPathForDisplay(text))
    for (const match of text.matchAll(/\b(?:[\w.-]+\/)+[\w@()[\].-]+\b/g)) refs.push(normalizeTicketPathForDisplay(match[0]))
  }
  push(ticket.title)
  push(ticket.body)
  for (const item of ticket.requiredEvidence ?? []) push(item)
  for (const item of ticket.acceptanceCriteria ?? []) push(item)
  for (const item of ticket.evidence ?? []) {
    push(item.ref)
    push(item.summary)
  }
  return [...new Set(refs.filter(Boolean))]
}

export function ticketConflictBlockers(ticket: Ticket, roomTickets: Ticket[]): string[] {
  const wantedPaths = ticketFileRefsForDisplay(ticket)
  if (!wantedPaths.length) return []
  const active = roomTickets.filter((candidate) => {
    if (candidate.id === ticket.id) return false
    if (candidate.projectId !== ticket.projectId) return false
    if ((candidate.roomId ?? null) !== (ticket.roomId ?? null)) return false
    return ACTIVE_TICKET_STATUSES.has(candidate.status) || isActiveRunState(candidate.runState)
  })
  const blockers: string[] = []
  for (const candidate of active) {
    const activePaths = ticketFileRefsForDisplay(candidate)
    const overlap = wantedPaths.filter((path) => activePaths.includes(path))
    if (overlap.length) blockers.push(`overlaps with "${candidate.title}" on ${overlap.slice(0, 3).join(', ')}`)
  }
  return [...new Set(blockers)]
}

export function roomConflictBlockers(tickets: Ticket[]): RoomConflictBlocker[] {
  return roomLeftToWork(tickets).flatMap((ticket) =>
    ticketConflictBlockers(ticket, tickets).map((text) => ({ ticketId: ticket.id, ticketTitle: ticket.title, text })),
  )
}

export function ticketWorkerLabel(ticket: Ticket): string {
  const owner = ticket.assignee ? ` @${ticket.assignee}` : ''
  if (ticket.runState === 'running') return `Worker running${owner}${ticket.startedAt ? ` · ${relativeTime(ticket.startedAt)}` : ''}`
  if (ticket.runState === 'queued') return `Worker queued${owner}`
  if (ticket.runState === 'done') return `Worker done${owner}`
  if (ticket.status === 'in_progress') return `In progress${owner}`
  if (ticket.status === 'review') return `Review needed${owner}`
  return owner ? `Claimed${owner}` : ''
}

export function ticketJobPhase(job: Job | null | undefined): 'build' | 'review' | null {
  return job?.metadata?.phase === 'review' ? 'review' : job?.metadata?.phase === 'build' ? 'build' : null
}

export function ticketPhaseLabel(ticket: Ticket, job?: Job | null): string {
  const phase = ticketJobPhase(job)
  if (phase === 'review') return 'Work review'
  if (phase === 'build') return 'Build'
  if (ticket.status === 'todo') return 'Ticket draft'
  if (ticket.status === 'ready') return 'Build ready'
  if (ticket.status === 'in_progress') return 'Build'
  if (ticket.status === 'review') return 'Work review'
  if (ticket.status === 'needs_clarification') return 'Clarify'
  if (ticket.status === 'completed') return 'Final gate'
  return ticketStatusLabel(ticket.status)
}

export function roomSupervisorPhase(leftToWork: Ticket[], activeWorkers: Ticket[], ticketJobsByTicket: Map<string, Job>, totalTickets: number): string {
  const active = activeWorkers[0]
  if (active) return ticketPhaseLabel(active, ticketJobsByTicket.get(active.id))
  const next = leftToWork[0]
  if (next) return ticketPhaseLabel(next, ticketJobsByTicket.get(next.id))
  return totalTickets > 0 ? 'Final gate' : 'Planning'
}

export function roomJobTitle(job: Job | null | undefined): string {
  if (!job) return 'No durable room job'
  const title = typeof job.metadata?.name === 'string' ? job.metadata.name : job.kind === 'ticket' ? 'Ticket worker' : 'Room run'
  return `${title} · ${ROOM_JOB_STATUS_LABEL[job.status]}`
}

export function roomJobDetail(job: Job | null | undefined): string | null {
  if (!job) return null
  if (job.nextWakeAt) {
    const wake = job.nextWakeAt <= Date.now()
      ? `wake due ${relativeTime(job.nextWakeAt)}`
      : `wake around ${new Date(job.nextWakeAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
    return job.blockedReason ? `${job.blockedReason} · ${wake}` : wake
  }
  return job.blockedReason || job.resumePlan || null
}

export function latestRoomJob(jobs: Job[]): Job | null {
  const sorted = [...jobs].sort((a, b) => Number(ACTIVE_ROOM_JOB_STATES.has(b.status)) - Number(ACTIVE_ROOM_JOB_STATES.has(a.status)) || b.updatedAt - a.updatedAt)
  return sorted[0] ?? null
}

export function looksLikeTicketPath(value: string): boolean {
  return !/\s/.test(value) && (value.includes('/') || /^[\w.-]+\.(?:[cm]?[jt]sx?|json|md|css|html|ya?ml|toml|rs|go|py|rb|java|kt|swift|sql|sh)$/i.test(value))
}

export function normalizeTicketPathForDisplay(value: string): string {
  return value.trim().replace(/^["'`]+|["'`.,:;]+$/g, '').replace(/\\/g, '/').replace(/^\.\//, '')
}
