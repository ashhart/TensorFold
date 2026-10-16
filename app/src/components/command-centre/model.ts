import type { IconName } from '../../lib/icons'
import type { View } from '../../lib/store'
import type {
  AgentRoleProfileId,
  Automation,
  ConnectorId,
  ContextPressureSnapshot,
  FanOutBudget,
  Job,
  ModelRouterSnapshot,
  Mission,
  Provider,
  Room,
  Settings,
  Thread,
  TokenUsageSnapshot,
} from '../../lib/types'

export type Tone = 'critical' | 'high' | 'normal' | 'low' | 'good'
export type CommandCentreSetView = (view: View) => void

export interface AttentionRow {
  id: string
  icon: IconName
  source: string
  title: string
  detail: string
  tone: Tone
  age: string
  action: string
  createdAt: number
  sourceIds?: string[]
  onOpen?: () => void
}

export interface ApprovalRow {
  id: string
  icon: IconName
  sourceGroup: string
  title: string
  detail: string
  meta: string
  tone: Tone
  createdAt: number
  primaryLabel: string
  secondaryLabel?: string
  onPrimary?: () => void
  onSecondary?: () => void
}

export interface ApprovalGroup {
  id: string
  title: string
  detail: string
  tone: Tone
  rows: ApprovalRow[]
}

export interface LiveJob {
  id: string
  icon: IconName
  title: string
  detail: string
  footnote?: string
  status: string
  tone: Tone
  startedAt: number
  remote?: boolean
  remoteLabel?: string
  onOpen?: () => void
  onCancel?: () => void
  onRetry?: () => void
  onResume?: () => void
  onDeadLetter?: () => void
}

export interface RunningWorkSummary {
  total: number
  visibleTotal: number
  statusCounts: Record<Job['status'], number>
  kindCounts: Array<{ kind: Job['kind']; count: number }>
  nextWake: { job: Job; at: number } | null
  leases: { active: number; expired: number; expiringSoon: number; missingRunning: number }
  attempts: { total: number; retried: number; retryable: number; max: number }
  unresolvedApprovals: number
}

export interface WorkLaneSummary {
  id: string
  label: string
  count: number
  tone: Tone
}

export interface ProviderVisibilityRow {
  id: string
  providerName: string
  providerKind: string
  label: string
  model: string
  account: string
  activity: string
  activeCount: number
  health: ProviderSignal
  context: ProviderSignal
  latency: ProviderSignal
  route: ProviderSignal
  spend: ProviderSignal
}

export interface ProviderSignal {
  label: string
  value: string
  detail: string
  tone: Tone
}

export interface ProviderVisibilityInput {
  settings: Settings | null
  threads: Thread[]
  threadStatus: Record<string, string | undefined>
  rooms: Room[]
  roomStatus: Record<string, Room['state'] | undefined>
  automations: Automation[]
  missions: Mission[]
  contextSnapshot: ContextPressureSnapshot | null
  modelRouterSnapshot: ModelRouterSnapshot | null
  usageSnapshot: TokenUsageSnapshot | null
}

export interface ProviderVisibilityDraft {
  provider: Provider
  accountId: string | null
  model: string | null
  labels: Set<string>
  activeCount: number
  latestAt: number
  isDefault: boolean
}

export interface ConnectorHealthSignal {
  label: string
  detail: string
  nextAction: string
  tone: Tone
  action: 'sync-email' | 'open-sources' | 'review-approvals' | 'none'
}

export interface AwayBriefRow {
  id: string
  icon: IconName
  title: string
  detail: string
  meta: string
  tone: Tone
  createdAt: number
  actionLabel?: string
  onOpen?: () => void
}

export interface AwayBriefRowDeps {
  durableJobs: Job[]
  openDurableJob: (job: Job) => void
  openRoom: (id: string) => void
  setView: CommandCentreSetView
}

export interface PowerGuardSignal {
  label: string
  detail: string
  nextAction: string
  tone: Tone
}

export const CONNECTOR_ICONS: Record<ConnectorId, IconName> = {
  email: 'mail',
  slack: 'message',
  discord: 'message',
  mattermost: 'message',
  matrix: 'message',
  googlechat: 'message',
  whatsapp: 'at',
  imessage: 'message',
  docs: 'file',
  browser: 'browser',
  calendar: 'clock',
  telegram: 'send',
  controller: 'monitor',
  webhook: 'plug',
}

export const CONNECTOR_NAMES: Record<ConnectorId, string> = {
  email: 'Email',
  slack: 'Slack',
  discord: 'Discord',
  mattermost: 'Mattermost',
  matrix: 'Matrix',
  googlechat: 'Google Chat',
  whatsapp: 'WhatsApp',
  imessage: 'iMessage',
  docs: 'Documents',
  browser: 'Browser',
  calendar: 'Calendar',
  telegram: 'Telegram',
  controller: 'Controller',
  webhook: 'Webhooks',
}

export const RUNNING_THREAD_STATES = new Set(['thinking', 'streaming', 'tool', 'awaiting', 'running'])
export const RUNNING_ROOM_STATES = new Set(['queued', 'running', 'waiting_user', 'waiting_external'])
export const VISIBLE_JOB_STATUSES = new Set<Job['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled', 'failed', 'dead_letter'])
export const ACTIONABLE_JOB_STATUSES = new Set<Job['status']>(['failed', 'stalled', 'dead_letter'])
export const LEASE_EXPIRING_SOON_MS = 2 * 60_000
export const CONNECTOR_STALE_MS = 24 * 60 * 60_000
export const AWAY_BRIEF_WINDOW_MS = 24 * 60 * 60_000
export const TONE_RANK: Record<Tone, number> = { critical: 0, high: 1, normal: 2, low: 3, good: 4 }
export const MODEL_PRICE_MAP: Array<{
  match: RegExp
  inputPerMillion: number
  outputPerMillion: number
  cacheWritePerMillion?: number
  cacheReadPerMillion?: number
  reasoningPerMillion?: number
}> = [
  { match: /\bclaude.*opus\b/i, inputPerMillion: 15, outputPerMillion: 75, cacheWritePerMillion: 18.75, cacheReadPerMillion: 1.5 },
  { match: /\bclaude.*sonnet\b/i, inputPerMillion: 3, outputPerMillion: 15, cacheWritePerMillion: 3.75, cacheReadPerMillion: 0.3 },
  { match: /\bclaude.*haiku\b/i, inputPerMillion: 0.8, outputPerMillion: 4, cacheWritePerMillion: 1, cacheReadPerMillion: 0.08 },
  { match: /\bgpt-4o-mini\b/i, inputPerMillion: 0.15, outputPerMillion: 0.6 },
  { match: /\bgpt-4o\b/i, inputPerMillion: 2.5, outputPerMillion: 10 },
]
export const cap = (value: string) => value.charAt(0).toUpperCase() + value.slice(1).replace(/_/g, ' ')
export const ROLE_PROFILE_OPTIONS: Array<{ id: AgentRoleProfileId; label: string }> = [
  { id: 'operator', label: 'Operator' },
  { id: 'planner', label: 'Planner' },
  { id: 'builder', label: 'Builder' },
  { id: 'reviewer', label: 'Reviewer' },
  { id: 'researcher', label: 'Researcher' },
]
export const FAN_OUT_OPTIONS: Array<{ id: FanOutBudget; label: string }> = [
  { id: 'single', label: 'Single' },
  { id: 'paired', label: 'Paired' },
  { id: 'council', label: 'Council' },
  { id: 'swarm', label: 'Swarm' },
]

// Threshold above which a collapsible briefing section auto-expands (Task #62).
export const DISCLOSURE_AUTO_EXPAND_THRESHOLD = 3
// How many mission rows the decluttered Mission Control shows before "View all" (Task #62).
export const MISSION_ROW_PREVIEW_LIMIT = 3

export interface CommandStatusSummary {
  maxTone: Tone
  attentionCount: number
  approvalsCount: number
  jobsStatus: { queued: number; stalled: number; failed: number; deadLetter: number }
  missionCount: number
}

export interface CommandStatusSummaryInput {
  attentionCount: number
  attentionMaxTone: Tone
  approvalsCount: number
  liveJobsTone: Tone
  connectorTone: Tone
  readinessTone?: Tone
  jobsStatus: { queued: number; stalled: number; failed: number; deadLetter: number }
  missionCount: number
}

/** Pick the highest-urgency (lowest-ranked) tone from a set of signals. */
export function worstTone(...tones: Tone[]): Tone {
  return tones.reduce<Tone>((worst, tone) => (TONE_RANK[tone] < TONE_RANK[worst] ? tone : worst), 'good')
}

/** Collapse the six command-centre metrics into one glanceable status summary (Task #62). */
export function computeCommandStatusSummary(input: CommandStatusSummaryInput): CommandStatusSummary {
  const approvalsTone: Tone = input.approvalsCount > 0 ? 'high' : 'good'
  return {
    maxTone: worstTone(input.attentionMaxTone, approvalsTone, input.liveJobsTone, input.connectorTone, input.readinessTone ?? 'good'),
    attentionCount: input.attentionCount,
    approvalsCount: input.approvalsCount,
    jobsStatus: input.jobsStatus,
    missionCount: input.missionCount,
  }
}

/** The status dashboard only forces itself open when something needs attention. */
export function summaryDefaultExpanded(maxTone: Tone): boolean {
  return maxTone === 'critical' || maxTone === 'high'
}

/**
 * Default open-state for a collapsible briefing section: open when it has more than
 * `threshold` items, or when it carries a critical signal (Task #62).
 */
export function disclosureDefaultOpen(itemCount: number, threshold: number, maxTone?: Tone): boolean {
  if (maxTone === 'critical' && itemCount > 0) return true
  return itemCount > threshold
}

/** Show the top `limit` mission rows and report overflow for a "View all" affordance (Task #62). */
export function limitMissionRows<T>(rows: T[], limit: number): { visible: T[]; hasMore: boolean; total: number; hidden: number } {
  const total = rows.length
  const visible = rows.slice(0, limit)
  const hidden = Math.max(0, total - visible.length)
  return { visible, hasMore: hidden > 0, total, hidden }
}

/** The Approvals panel only stays pinned while there is something waiting on the user (Task #62). */
export function approvalsShouldStick(pendingCount: number): boolean {
  return pendingCount > 0
}

/**
 * Statuses a LiveJob row can carry that mean the underlying work has stopped making progress
 * (terminal failure or an operator-initiated pause), even though it's still shown in the
 * Live Jobs panel for visibility/follow-up (LOW #30).
 */
export const LIVE_JOB_TERMINAL_STATUSES = new Set(['failed', 'dead letter', 'paused'])

/** Only genuinely in-progress jobs should render with a pulsing "Active" indicator (LOW #30). */
export function liveJobIsActive(status: string): boolean {
  return !LIVE_JOB_TERMINAL_STATUSES.has(status)
}
