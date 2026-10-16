import type { Ticket, TicketCapability, TicketEvidence, TicketExecutionMode, TicketExecutionRecord, TicketRisk, TicketStatus } from './types'

const CAPABILITIES: ReadonlySet<TicketCapability> = new Set(['read', 'write', 'shell', 'network', 'browser', 'connector', 'mcp'])
const RISKS: ReadonlySet<TicketRisk> = new Set(['low', 'medium', 'high'])
const RISKY_CAPABILITIES: ReadonlySet<TicketCapability> = new Set(['write', 'shell', 'network', 'browser', 'connector', 'mcp'])
const EXTERNAL_CAPABILITIES: ReadonlySet<TicketCapability> = new Set(['network', 'browser', 'connector', 'mcp'])
const TICKET_ALWAYS_DISABLED_TOOLS = new Set(['launch_subagent', 'launch_subagents', 'delegate_task'])
const TICKET_WRITE_TOOLS = new Set([
  'write_file', 'edit_file', 'apply_patch',
  'create_project', 'create_room', 'create_kanban_ticket', 'change_setting', 'save_artifact',
  'suggest_skill',
])
const ACTIVE_TICKET_STATUSES: ReadonlySet<Ticket['status']> = new Set(['in_progress', 'review'])
const ACTIVE_RUN_STATES: ReadonlySet<NonNullable<Ticket['runState']>> = new Set(['queued', 'running'])
const REVIEWABLE_TICKET_STATUSES: ReadonlySet<Ticket['status']> = new Set(['review', 'completed'])

export interface TicketExecutionDefaults {
  roomScoped?: boolean
}

export function normalizeTicketCapabilities(input: unknown, fallback: TicketCapability[] = ['read']): TicketCapability[] {
  const raw = Array.isArray(input) ? input : typeof input === 'string' ? input.split(/[,\s]+/) : []
  const cleaned = raw
    .map((item) => String(item).trim().toLowerCase())
    .filter((item): item is TicketCapability => CAPABILITIES.has(item as TicketCapability))
  return [...new Set(cleaned.length ? cleaned : fallback)]
}

/** Capability lane required for a callable tool inside a ticket worker. */
export function ticketToolCapability(toolName: string): TicketCapability | null {
  const name = String(toolName ?? '').trim()
  if (!name || TICKET_ALWAYS_DISABLED_TOOLS.has(name)) return null
  if (name === 'run_command') return 'shell'
  if (name === 'web_search' || name === 'web_fetch') return 'network'
  if (name === 'controller_action' || name.startsWith('browser_')) return 'browser'
  if (/^(email|message|calendar|document|table)_/.test(name)) return 'connector'
  if (name.startsWith('mcp__') || name === 'tool_search' || name === 'tool_describe' || name === 'tool_call') return 'mcp'
  if (TICKET_WRITE_TOOLS.has(name)) return 'write'
  return 'read'
}

export function ticketDisallowedToolNames(toolNames: string[], capabilities: unknown): string[] {
  const allowed = new Set(normalizeTicketCapabilities(capabilities, ['read']))
  return [...new Set(toolNames.filter((name) => {
    const required = ticketToolCapability(name)
    return !required || !allowed.has(required)
  }))]
}

export function normalizeStringList(input: unknown, fallback: string[] = []): string[] {
  const raw = Array.isArray(input) ? input : typeof input === 'string' ? input.split(/\n|;/) : []
  const cleaned = raw.map((item) => String(item).replace(/\s+/g, ' ').trim()).filter(Boolean)
  return [...new Set(cleaned.length ? cleaned : fallback)].slice(0, 12)
}

export function inferTicketRisk(input: Partial<Ticket>): TicketRisk {
  if (RISKS.has(input.risk as TicketRisk)) return input.risk as TicketRisk
  const capabilities = normalizeTicketCapabilities(input.allowedCapabilities, [])
  if (input.priority === 'urgent' || capabilities.includes('shell') || capabilities.includes('write')) return 'high'
  if (input.priority === 'high' || capabilities.some((cap) => cap === 'network' || cap === 'browser' || cap === 'connector' || cap === 'mcp')) return 'medium'
  return 'low'
}

export function defaultRequiredEvidence(input: Partial<Ticket>): string[] {
  if (input.type === 'epic') return ['Child tickets created or linked', 'Completion summary with remaining gaps']
  if (input.type === 'bug') return ['Reproduction or observed failure', 'Fix or mitigation summary', 'Verification result or blocker']
  if (input.status === 'review') return ['Reviewed files or evidence', 'Reviewer verdict', 'Blocking issues or explicit approval']
  return ['Relevant files or sources inspected', 'Implementation or investigation summary', 'Verification result or blocker']
}

export function normalizeTicketEvidence(input: unknown): TicketEvidence[] {
  if (!Array.isArray(input)) return []
  return input
    .filter((item): item is Partial<TicketEvidence> => !!item && typeof item === 'object')
    .filter((item) => typeof item.summary === 'string' && item.summary.trim())
    .map((item, index) => ({
      id: typeof item.id === 'string' && item.id.trim() ? item.id.trim() : `evidence-${index + 1}`,
      kind: isEvidenceKind(item.kind) ? item.kind : 'note',
      summary: item.summary!.replace(/\s+/g, ' ').trim(),
      ref: typeof item.ref === 'string' && item.ref.trim() ? item.ref.trim() : null,
      createdBy: typeof item.createdBy === 'string' && item.createdBy.trim() ? item.createdBy.trim() : null,
      createdAt: typeof item.createdAt === 'number' && Number.isFinite(item.createdAt) ? item.createdAt : Date.now(),
    }))
}

export function normalizeTicketExecution(input: Partial<Ticket>, defaults: TicketExecutionDefaults = {}): Pick<Ticket, 'risk' | 'allowedCapabilities' | 'requiredEvidence' | 'acceptanceCriteria' | 'evidence'> {
  const fallbackCapabilities: TicketCapability[] = defaults.roomScoped ? ['read'] : ['read', 'write', 'shell']
  const allowedCapabilities = normalizeTicketCapabilities(input.allowedCapabilities, fallbackCapabilities)
  const withCapabilities = { ...input, allowedCapabilities }
  return {
    risk: inferTicketRisk(withCapabilities),
    allowedCapabilities,
    requiredEvidence: normalizeStringList(input.requiredEvidence, defaultRequiredEvidence(withCapabilities)),
    acceptanceCriteria: normalizeStringList(input.acceptanceCriteria, []),
    evidence: normalizeTicketEvidence(input.evidence),
  }
}

export function normalizeTickets(input: unknown): Ticket[] {
  if (!Array.isArray(input)) return []
  return input
    .filter((item): item is Ticket => !!item && typeof item === 'object' && typeof (item as Ticket).id === 'string')
    .map((item) => ({ ...item, ...normalizeTicketExecution(item, { roomScoped: !!item.roomId }) }))
}

export function ticketFileRefs(ticket: Partial<Ticket>, extraRefs: unknown = []): string[] {
  const refs: string[] = []
  const push = (value: unknown) => {
    if (typeof value !== 'string') return
    const text = value.trim()
    if (!text) return
    if (looksLikePath(text)) refs.push(normalizeTicketPath(text))
    for (const match of text.matchAll(/\b(?:[\w.-]+\/)+[\w@()[\].-]+\b/g)) refs.push(normalizeTicketPath(match[0]))
  }
  push(ticket.title)
  push(ticket.body)
  for (const item of normalizeStringList(ticket.requiredEvidence, [])) push(item)
  for (const item of normalizeStringList(ticket.acceptanceCriteria, [])) push(item)
  for (const item of normalizeTicketEvidence(ticket.evidence)) {
    push(item.ref)
    push(item.summary)
  }
  const extras = Array.isArray(extraRefs) ? extraRefs : typeof extraRefs === 'string' ? extraRefs.split(/\n|,|;/) : []
  for (const item of extras) push(item)
  return [...new Set(refs.filter(Boolean))]
}

export interface RoomTicketConflictInput {
  ticket: Ticket
  tickets: Ticket[]
  phase?: 'build' | 'review'
  seat?: string
  paths?: string[]
}

export function roomTicketConflictBlockers(input: RoomTicketConflictInput): string[] {
  const blockers: string[] = []
  const ticket = input.ticket
  const phase = input.phase ?? 'build'
  const seat = input.seat?.trim()
  if (ticket.runState && ACTIVE_RUN_STATES.has(ticket.runState)) blockers.push(`"${ticket.title}" already has a ${ticket.runState} worker.`)
  if (phase === 'build' && ticket.assignee && seat && ticket.assignee !== seat) blockers.push(`"${ticket.title}" is claimed by ${ticket.assignee}.`)
  if (phase === 'review') {
    const reviewGate = ticketReviewGateState(ticket)
    if (!REVIEWABLE_TICKET_STATUSES.has(ticket.status)) blockers.push(`"${ticket.title}" is not ready for review; run a build worker first.`)
    if (!reviewGate.hasBuilderEvidence) blockers.push(`"${ticket.title}" has no builder evidence to review.`)
    if (seat && reviewGate.builderEvidenceBy && reviewGate.builderEvidenceBy === seat) blockers.push(`"${ticket.title}" needs a reviewer different from builder ${seat}.`)
  }

  const wantedPaths = ticketFileRefs(ticket, input.paths)
  const active = normalizeTickets(input.tickets).filter((candidate) => {
    if (candidate.id === ticket.id) return false
    if (candidate.projectId !== ticket.projectId) return false
    if ((candidate.roomId ?? null) !== (ticket.roomId ?? null)) return false
    return ACTIVE_TICKET_STATUSES.has(candidate.status) || (candidate.runState != null && ACTIVE_RUN_STATES.has(candidate.runState))
  })

  for (const candidate of active) {
    const activePaths = ticketFileRefs(candidate)
    const overlap = wantedPaths.filter((path) => activePaths.includes(path))
    if (overlap.length) blockers.push(`"${ticket.title}" overlaps with "${candidate.title}" on ${overlap.slice(0, 3).join(', ')}.`)
  }
  return [...new Set(blockers)]
}

export function roomLeftToWork(tickets: Ticket[]): Ticket[] {
  const rank: Record<Ticket['status'], number> = { needs_clarification: 0, review: 1, in_progress: 2, ready: 3, todo: 4, completed: 5 }
  return normalizeTickets(tickets)
    .filter((ticket) => ticket.status !== 'completed')
    .sort((a, b) => (rank[a.status] ?? 9) - (rank[b.status] ?? 9) || a.order - b.order)
}

export function summarizeRoomLeftToWork(tickets: Ticket[]): string {
  const left = roomLeftToWork(tickets)
  if (!left.length) return 'No room tickets left. Final gate can run once any pending worker summaries have landed.'
  return left.map((ticket) => {
    const assignee = ticket.assignee ? ` @${ticket.assignee}` : ''
    const run = ticket.runState ? ` ${ticket.runState}` : ''
    const caps = normalizeTicketCapabilities(ticket.allowedCapabilities, ['read']).join(',')
    const paths = ticketFileRefs(ticket).slice(0, 3)
    return `- [${ticket.status}${run}${assignee} · risk:${ticket.risk ?? 'low'} · caps:${caps}] ${ticket.title}${paths.length ? ` · files:${paths.join(',')}` : ''}`
  }).join('\n')
}

export interface RoomFinalGateInput {
  tickets: Ticket[]
  lastCodeChangeAt?: number | null
  greenEvidenceAt?: number | null
}

export function roomFinalGateBlockers(input: RoomFinalGateInput): string[] {
  const blockers: string[] = []
  const tickets = normalizeTickets(input.tickets)
  const open = tickets.filter((ticket) => ticket.status !== 'completed')
  if (open.length) blockers.push(`room task(s) still open: ${summarizeTicketTitles(open)}.`)

  const completed = tickets.filter((ticket) => ticket.status === 'completed')
  const missingRequired = completed.filter((ticket) => {
    const required = normalizeStringList(ticket.requiredEvidence, [])
    if (!required.length) return false
    return meaningfulTicketEvidence(ticket).length < required.length
  })
  if (missingRequired.length) blockers.push(`required evidence missing on: ${summarizeTicketTitles(missingRequired)}.`)

  const missingReview = completed.filter((ticket) => ticketRequiresReviewGate(ticket) && !ticketReviewGateState(ticket).hasReviewOrSkipAfterBuilder)
  if (missingReview.length) blockers.push(`review evidence after builder or justified skip missing on: ${summarizeTicketTitles(missingReview)}.`)

  const lastCodeChangeAt = Math.max(input.lastCodeChangeAt ?? 0, latestTicketCodeChangeAt(completed) ?? 0)
  const greenEvidenceAt = Math.max(input.greenEvidenceAt ?? 0, latestTicketGreenEvidenceAt(completed) ?? 0)
  if (lastCodeChangeAt && greenEvidenceAt < lastCodeChangeAt) blockers.push('code changed without a later passing command.')

  const unresolvedApproval = completed.filter(hasUnresolvedExternalDestructiveApproval)
  if (unresolvedApproval.length) blockers.push(`external/destructive approval unresolved on: ${summarizeTicketTitles(unresolvedApproval)}.`)
  return blockers
}

function summarizeTicketTitles(tickets: Ticket[]): string {
  const labels = tickets.slice(0, 4).map((ticket) => ticket.title || 'Untitled')
  const extra = tickets.length > labels.length ? `, +${tickets.length - labels.length} more` : ''
  return labels.join(', ') + extra
}

export function ticketRequiresReviewGate(ticket: Ticket): boolean {
  if (ticket.type === 'epic') return false
  if (ticket.risk === 'high') return true
  if (ticket.priority === 'urgent') return true
  const caps = normalizeTicketCapabilities(ticket.allowedCapabilities, ['read'])
  return caps.some((cap) => RISKY_CAPABILITIES.has(cap))
}

export interface TicketReviewGateState {
  hasBuilderEvidence: boolean
  builderEvidenceBy: string | null
  hasReviewOrSkipAfterBuilder: boolean
}

export function ticketReviewGateState(ticket: Ticket): TicketReviewGateState {
  const evidence = meaningfulTicketEvidence(ticket)
  const builder = newestEvidence(evidence.filter(isBuilderEvidence))
  const review = newestReviewPassAfterBuilder(evidence)
  const skip = newestReviewSkipAfterBuilder(evidence, ticket)
  return {
    hasBuilderEvidence: !!builder,
    builderEvidenceBy: builder?.createdBy ?? null,
    hasReviewOrSkipAfterBuilder: !!review || !!skip,
  }
}

export function ticketNeedsReviewAfterBuilder(ticket: Ticket): boolean {
  if (!REVIEWABLE_TICKET_STATUSES.has(ticket.status)) return false
  if (!ticketRequiresReviewGate(ticket)) return false
  const reviewGate = ticketReviewGateState(ticket)
  return reviewGate.hasBuilderEvidence && !reviewGate.hasReviewOrSkipAfterBuilder
}

function meaningfulTicketEvidence(ticket: Partial<Ticket>): TicketEvidence[] {
  return normalizeTicketEvidence(ticket.evidence).filter((item) => !isReservationEvidence(item))
}

function newestEvidence(items: TicketEvidence[]): TicketEvidence | null {
  return items.reduce<TicketEvidence | null>((latest, item) => {
    if (!latest || item.createdAt >= latest.createdAt) return item
    return latest
  }, null)
}

function newestReviewPassAfterBuilder(evidence: TicketEvidence[]): TicketEvidence | null {
  return newestEvidence(evidence.filter((item) => {
    if (!isReviewPassEvidence(item)) return false
    const builder = newestBuilderEvidenceBefore(evidence, item)
    if (!builder) return false
    return !sameEvidenceCreator(builder, item)
  }))
}

function newestReviewSkipAfterBuilder(evidence: TicketEvidence[], ticket: Ticket): TicketEvidence | null {
  if (hasCodeChangeEvidence(ticket)) return null
  return newestEvidence(evidence.filter((item) => {
    if (!isReviewSkipEvidence(item)) return false
    const builder = newestBuilderEvidenceBefore(evidence, item)
    if (!builder) return false
    return !sameEvidenceCreator(builder, item)
  }))
}

function newestBuilderEvidenceBefore(evidence: TicketEvidence[], item: TicketEvidence): TicketEvidence | null {
  return newestEvidence(evidence.filter((candidate) => isBuilderEvidence(candidate) && candidate.createdAt < item.createdAt))
}

function sameEvidenceCreator(a: TicketEvidence, b: TicketEvidence): boolean {
  return !!a.createdBy && !!b.createdBy && a.createdBy === b.createdBy
}

function latestTicketCodeChangeAt(tickets: Ticket[]): number | null {
  return newestEvidence(tickets.flatMap((ticket) => meaningfulTicketEvidence(ticket).filter(isCodeChangeEvidence)))?.createdAt ?? null
}

function latestTicketGreenEvidenceAt(tickets: Ticket[]): number | null {
  return newestEvidence(tickets.flatMap((ticket) => meaningfulTicketEvidence(ticket).filter(isGreenEvidence)))?.createdAt ?? null
}

function hasUnresolvedExternalDestructiveApproval(ticket: Ticket): boolean {
  const evidence = meaningfulTicketEvidence(ticket)
  const actionAt = newestEvidence(evidence.filter((item) => isExternalDestructiveActionEvidence(item, ticket)))?.createdAt ?? null
  if (!actionAt) return false
  const approvalAt = newestEvidence(evidence.filter(isResolvedUserApprovalEvidence))?.createdAt ?? null
  return !approvalAt || approvalAt < actionAt
}

function hasCodeChangeEvidence(ticket: Ticket): boolean {
  return meaningfulTicketEvidence(ticket).some(isCodeChangeEvidence)
}

function isBuilderEvidence(item: TicketEvidence): boolean {
  return !isReviewEvidence(item) && !isReviewSkipEvidence(item)
}

function isReservationEvidence(item: TicketEvidence): boolean {
  return /\breserved for (?:build|review) worker\b/i.test(evidenceText(item))
}

function isReviewEvidence(item: TicketEvidence): boolean {
  if (isReviewSkipEvidence(item)) return false
  const text = evidenceText(item)
  return item.kind === 'review' || /\b(reviewed|reviewer|verdict|approved|approval|blocking issues?|ready for final)\b/i.test(text)
}

function isReviewPassEvidence(item: TicketEvidence): boolean {
  if (isReviewSkipEvidence(item)) return false
  if (isBlockingReviewEvidence(item)) return false
  const text = evidenceText(item)
  const reviewerSignal = item.kind === 'review' || /\b(reviewed|reviewer|review verdict|verdict)\b/i.test(text)
  if (!reviewerSignal) return false
  return /\b(approved|approval|pass(?:ed)?|verified|accepted|ready for final|ready to final|ready|finished|looks good|lgtm|no blockers?|no blocking issues?)\b/i.test(text)
}

function isBlockingReviewEvidence(item: TicketEvidence): boolean {
  const text = evidenceText(item)
  if (/\b(no|zero|without|no remaining|no open)\s+(blockers?|blocking issues?|failures?|errors?|concerns?|issues?)\b/i.test(text)) return false
  if (/\bblockers?:\s*(none|no|zero)\b/i.test(text)) return false
  return /\b(blocked|blockers?|blocking issues?|failed|failure|failing|rejected?|not ready|missing tests?|unsafe|regression|must change|needs changes?|request changes?|changes requested)\b/i.test(text)
}

function isReviewSkipEvidence(item: TicketEvidence): boolean {
  const text = evidenceText(item)
  return /\b(review\s*)?(skip|skipped|waived|waiver)\b/i.test(text)
    && /\b(read-only|no code changes?|no changes?|source lookup|docs? only|investigation only|not required|low[- ]risk)\b/i.test(text)
}

function isCodeChangeEvidence(item: TicketEvidence): boolean {
  const text = evidenceText(item)
  if (/\bno code changes?\b/i.test(text)) return false
  return item.kind === 'diff' || /\b(changed|patched|modified|implemented|fixed)\b/i.test(text)
}

function isGreenEvidence(item: TicketEvidence): boolean {
  const text = evidenceText(item)
  if (/\b(failed|failure|error|blocked|not ok|non[- ]?zero|exit(?:ed)?(?:\s+with)?(?:\s+code)?\s+[1-9])\b/i.test(text)) return false
  if (item.kind === 'test') return true
  if (item.kind !== 'command') return false
  return /\b(test|tsx|vitest|jest|playwright|tsc|lint|typecheck)\b/i.test(text)
    && /\b(pass|passed|green|ok|exit\s+0|succeeded|success|no errors?)\b/i.test(text)
}

function isExternalDestructiveActionEvidence(item: TicketEvidence, ticket: Ticket): boolean {
  const text = evidenceText(item)
  if (isNoSideEffectEvidence(text)) return false
  if (isDestructiveActionEvidence(text)) return true
  const caps = normalizeTicketCapabilities(ticket.allowedCapabilities, ['read'])
  const externalCapable = caps.some((cap) => EXTERNAL_CAPABILITIES.has(cap))
  return isExternalSideEffectEvidence(text, externalCapable)
}

function isNoSideEffectEvidence(text: string): boolean {
  return /\b(read[- ]only|no side effects?|no external action|no destructive action|dry[- ]run|preview only|draft only|not executed|did not execute|not sent|not submitted|not published|not applied|no changes?)\b/i.test(text)
}

function isDestructiveActionEvidence(text: string): boolean {
  return /\b(delete[ds]?|deleted|remove[ds]?|removed|trash(?:ed)?|erase[ds]?|erased|destroy(?:ed)?|wipe[ds]?|wiped|drop(?:ped)?|truncate[ds]?|truncated|reset\s+--hard|rm\s+-[a-z]*r[f]?|git\s+push\s+--force|force[- ]push(?:ed)?|chmod|chown|kill(?:ed)?|shutdown|reboot)\b/i.test(text)
}

function isExternalSideEffectEvidence(text: string, externalCapable: boolean): boolean {
  const externalTarget = /\b(external|connector|mcp|browser|webhook|api|http|https|email|mail|telegram|whatsapp|slack|calendar|github|gitlab|jira|linear|crm|notion|docs?|sheets?|drive|pull request|issue|ticket|customer|recipient|channel)\b/i.test(text)
  const directSideEffect = /\b(send|sent|deliver(?:ed)?|publish(?:ed)?|post(?:ed)?|submit(?:ted)?|upload(?:ed)?|invite[ds]?|invited|schedule[ds]?|scheduled|reschedule[ds]?|rescheduled|reply|replied|respond(?:ed)?|comment(?:ed)?|charge[ds]?|charged|pa(?:id|y)|purchase[ds]?|purchased|order(?:ed)?)\b/i.test(text)
  const externalChange = /\b(create[ds]?|created|update[ds]?|updated)\b/i.test(text) && externalTarget
  return externalChange || (directSideEffect && (externalCapable || externalTarget))
}

function isResolvedUserApprovalEvidence(item: TicketEvidence): boolean {
  const text = evidenceText(item)
  if (/\b(denied|rejected|pending|waiting|requested|required|unresolved|not approved)\b/i.test(text)) return false
  return /\bapproval_(?:approved|resolved)\b/i.test(text)
    || /\b(?:user|human|operator|owner|guide|requester|external|destructive)\s+approval\s+(?:approved|resolved|granted|confirmed)\b/i.test(text)
    || /\b(?:user|human|operator|owner|guide|requester)\b.{0,48}\b(?:approved|confirmed|allowed|granted|resolved)\b/i.test(text)
    || /\b(?:approved|confirmed|allowed|granted|resolved)\b.{0,48}\b(?:by|from)\s+(?:user|human|operator|owner|guide|requester)\b/i.test(text)
}

function evidenceText(item: TicketEvidence): string {
  return `${item.kind} ${item.summary} ${item.ref ?? ''}`
}

function isEvidenceKind(value: unknown): value is TicketEvidence['kind'] {
  return value === 'file' || value === 'diff' || value === 'command' || value === 'test' || value === 'review' || value === 'source' || value === 'note'
}

function looksLikePath(value: string): boolean {
  return !/\s/.test(value) && (value.includes('/') || /^[\w.-]+\.(?:[cm]?[jt]sx?|json|md|css|html|ya?ml|toml|rs|go|py|rb|java|kt|swift|sql|sh)$/i.test(value))
}

function normalizeTicketPath(value: string): string {
  return value.trim().replace(/^["'`]+|["'`.,:;]+$/g, '').replace(/\\/g, '/').replace(/^\.\//, '')
}

// ---- Ticket execution (VST-018): tickets are executable units ------------------------------

export type TicketPhase = 'build' | 'review'

export interface TicketRunOutcome {
  phase: TicketPhase
  failed: boolean
  interrupted: boolean
}

/**
 * The kanban state a ticket lands in when its worker run settles. Extracted from
 * ticket-worker-runtime's settle block so the semantics are testable directly:
 * an aborted/paused run PARKS the ticket (status unchanged, runState cleared for
 * re-dispatch) instead of claiming it reviewed/completed.
 */
export function ticketStateAfterRun(current: Pick<Ticket, 'status'>, outcome: TicketRunOutcome): { runState: Ticket['runState']; status: TicketStatus } {
  const runState: Ticket['runState'] = outcome.interrupted ? null : 'done'
  const status: TicketStatus = outcome.interrupted
    ? current.status
    : outcome.failed
      ? 'needs_clarification'
      : outcome.phase === 'review'
        ? 'completed'
        : 'review'
  return { runState, status }
}

export interface TicketExecutionStart {
  target: TicketExecutionMode
  hostId?: string | null
  providerId?: string | null
  model?: string | null
  branch?: string | null
  threadId?: string | null
  startedAt: number
}

/**
 * A fresh execution record at run START: running, nothing proven. The review verdict always
 * resets — an approval from before this run never covers the new work.
 */
export function startTicketExecution(input: TicketExecutionStart): TicketExecutionRecord {
  return {
    target: input.target,
    hostId: input.hostId ?? null,
    providerId: input.providerId ?? null,
    model: input.model ?? null,
    branch: input.branch ?? null,
    threadId: input.threadId ?? null,
    startedAt: input.startedAt,
    finishedAt: null,
    checks: [],
    blockers: [],
    reviewVerdict: null,
  }
}

export interface TicketExecutionSettle {
  outcome: TicketRunOutcome
  finishedAt: number
  commands?: { command: string; state?: string | null }[]
  blockers?: string[]
}

/**
 * Finalize the execution record when the run settles: checks from the run's commands
 * (passed = clean exit), blockers named, and the review verdict — build success →
 * 'pending' (awaiting review), review pass → 'approved', review fail →
 * 'changes_requested', interrupted → parked with a named blocker.
 */
export function settleTicketExecution(record: TicketExecutionRecord, settle: TicketExecutionSettle): TicketExecutionRecord {
  const { outcome } = settle
  const checks = (settle.commands ?? []).map((item) => ({ command: item.command, passed: (item.state ?? 'ok') === 'ok' }))
  const blockers = [...(settle.blockers ?? [])]
  if (outcome.interrupted) blockers.push('run parked before completion (cancelled or waiting on approval)')
  const reviewVerdict: TicketExecutionRecord['reviewVerdict'] = outcome.interrupted
    ? record.reviewVerdict
    : outcome.phase === 'review'
      ? (outcome.failed ? 'changes_requested' : 'approved')
      : (outcome.failed ? null : 'pending')
  return {
    ...record,
    finishedAt: outcome.interrupted ? null : settle.finishedAt,
    checks,
    blockers,
    reviewVerdict,
  }
}

/**
 * The worker prompt for a ticket run. Shared source of truth for the local worker
 * (ticket-worker-runtime) — the renderer's remote-ticket prompt
 * (src/lib/ticket-remote.ts) mirrors this layout and a parity test pins the two.
 */
export function ticketWorkPrompt(ticket: Ticket, phase: TicketPhase): string {
  const phaseInstruction = phase === 'review'
    ? 'Adversarial review pass. Inspect the ticket and any completed work. Find concrete blockers, missing tests, unsafe assumptions, and either list what must change or say it is ready.'
    : 'Build pass. Implement this ticket in the workspace and keep the result grounded in the project files.'
  const allowedCapabilities = normalizeTicketCapabilities(ticket.allowedCapabilities, ['read'])
  const requiredEvidence = normalizeStringList(ticket.requiredEvidence, [])
  const acceptanceCriteria = normalizeStringList(ticket.acceptanceCriteria, [])
  return [
    phaseInstruction,
    '',
    'Execution policy:',
    `- Risk: ${ticket.risk ?? 'low'}`,
    `- Allowed capabilities: ${allowedCapabilities.join(', ')}`,
    requiredEvidence.length ? `- Required evidence:\n${requiredEvidence.map((item) => `  - ${item}`).join('\n')}` : '- Required evidence: none specified',
    acceptanceCriteria.length ? `- Acceptance criteria:\n${acceptanceCriteria.map((item) => `  - ${item}`).join('\n')}` : '- Acceptance criteria: none specified',
    '',
    'Ticket:',
    ticket.body ? `${ticket.title}\n\n${ticket.body}` : ticket.title,
  ].join('\n')
}
