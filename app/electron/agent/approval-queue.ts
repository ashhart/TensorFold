// Durable batch-approval queue — a PURE reducer that sits on top of the per-job
// approval substrate (jobs.ts recordJobApproval/resolveJobApproval,
// approval-policy.ts scope keys). Where jobs.ts tracks approvals *inside* one job,
// this queue is the cross-cutting, restart-durable ledger the host persists and
// replays: it enqueues pending approvals under an idempotency key (exactly-once on
// resume), ages them, escalates past an SLA, and resolves them one-at-a-time or in
// a batch by a policy-scope key.
//
// Pure by construction: every function takes the current queue + a `now` clock and
// returns a NEW queue (never mutates the input). No I/O, no globals — the host owns
// persistence and the wall clock. This mirrors the codebase's dependency-injected
// pattern (ssh-connection clientFactory, connector-retry fetchImpl).

export type ApprovalQueueUrgency = 'normal' | 'aging' | 'escalated' | 'expired'
export type ApprovalQueueDecision = 'allow' | 'allow_always' | 'deny'
export type ApprovalQueueState = 'pending' | 'resolved' | 'expired'

export interface ApprovalQueueScope {
  // Mirrors ApprovalPolicyScope (approval-policy.ts) / JobApprovalPolicySnapshot.
  scope: 'tool' | 'connector'
  key: string
  label: string
  explanation?: string
  classification?: string
}

export interface ApprovalQueueEntryInput {
  // The idempotency key: same key => same slot, enqueued exactly once even across
  // restarts. Callers typically derive it from jobId + callId, or from the policy
  // scope key when they want repeat calls of the same signature to coalesce.
  idempotencyKey: string
  jobId: string
  callId: string
  tool: string
  scope: ApprovalQueueScope
  // Optional override of the queue-level SLA for this one entry (ms until escalation).
  slaMs?: number | null
  // Optional wall-clock deadline after which an unresolved entry auto-expires.
  expiresAt?: number | null
  metadata?: Record<string, unknown>
}

export interface ApprovalQueueEntry {
  idempotencyKey: string
  jobId: string
  callId: string
  tool: string
  scope: ApprovalQueueScope
  state: ApprovalQueueState
  enqueuedAt: number
  // Bumped each time the same idempotencyKey is re-enqueued while still pending
  // (a retry of the same call) — surfaces flapping without creating duplicates.
  requeueCount: number
  slaMs: number | null
  expiresAt: number | null
  escalatedAt: number | null
  resolvedAt: number | null
  decision: ApprovalQueueDecision | null
  resolvedBy: string | null
  metadata?: Record<string, unknown>
}

export interface ApprovalQueue {
  entries: ApprovalQueueEntry[]
  // Queue-wide SLA: pending entries older than this (and without their own slaMs)
  // escalate. Aging threshold is derived from it (half the SLA) unless overridden.
  slaMs: number
  agingMs: number
}

export interface CreateApprovalQueueOptions {
  slaMs?: number
  agingMs?: number
  entries?: ApprovalQueueEntry[]
}

export interface EnqueueResult {
  queue: ApprovalQueue
  entry: ApprovalQueueEntry
  // false when the idempotency key already had a live (pending) slot — the call was
  // a duplicate/retry and did NOT create a second approval.
  created: boolean
}

export interface ResolveResult {
  queue: ApprovalQueue
  resolved: ApprovalQueueEntry[]
}

export interface AgeResult {
  queue: ApprovalQueue
  changed: boolean
  // Entries whose urgency rose to 'escalated' on THIS pass (for firing an alert once).
  newlyEscalated: ApprovalQueueEntry[]
  // Entries that hit their hard expiry on THIS pass.
  newlyExpired: ApprovalQueueEntry[]
}

const DEFAULT_SLA_MS = 15 * 60_000

export function createApprovalQueue(options: CreateApprovalQueueOptions = {}, now = Date.now()): ApprovalQueue {
  const slaMs = positive(options.slaMs, DEFAULT_SLA_MS)
  const agingMs = positive(options.agingMs, Math.floor(slaMs / 2))
  return {
    slaMs,
    agingMs,
    entries: normalizeEntries(options.entries ?? [], now),
  }
}

// Rehydrate a persisted queue after a restart — drops malformed rows, clamps fields,
// and preserves resolved/expired state so replay is exactly-once (a resolved entry is
// never resurrected as pending).
export function normalizeApprovalQueue(input: unknown, now = Date.now()): ApprovalQueue {
  const raw = (input && typeof input === 'object' ? input : {}) as Partial<ApprovalQueue>
  return createApprovalQueue(
    {
      slaMs: typeof raw.slaMs === 'number' ? raw.slaMs : undefined,
      agingMs: typeof raw.agingMs === 'number' ? raw.agingMs : undefined,
      entries: Array.isArray(raw.entries) ? (raw.entries as ApprovalQueueEntry[]) : [],
    },
    now,
  )
}

// Exactly-once enqueue. If the idempotency key already maps to a PENDING entry, this
// is a duplicate (a retry, or a replay after restart): we bump requeueCount and return
// created:false without adding a row. If it maps to a RESOLVED/EXPIRED entry, the slot
// is terminal — we do NOT reopen it (that would double-prompt for a decided call);
// the existing terminal entry is returned unchanged with created:false.
export function enqueueApproval(queue: ApprovalQueue, input: ApprovalQueueEntryInput, now = Date.now()): EnqueueResult {
  const key = cleanKey(input.idempotencyKey)
  const existing = queue.entries.find((entry) => entry.idempotencyKey === key)
  if (existing) {
    if (existing.state !== 'pending') {
      return { queue, entry: existing, created: false }
    }
    const bumped: ApprovalQueueEntry = { ...existing, requeueCount: existing.requeueCount + 1 }
    return {
      queue: { ...queue, entries: queue.entries.map((entry) => (entry.idempotencyKey === key ? bumped : entry)) },
      entry: bumped,
      created: false,
    }
  }
  const entry: ApprovalQueueEntry = {
    idempotencyKey: key,
    jobId: String(input.jobId ?? ''),
    callId: String(input.callId ?? ''),
    tool: String(input.tool ?? 'tool'),
    scope: normalizeScope(input.scope, input.tool),
    state: 'pending',
    enqueuedAt: now,
    requeueCount: 0,
    slaMs: input.slaMs == null ? null : positive(input.slaMs, queue.slaMs),
    expiresAt: typeof input.expiresAt === 'number' && Number.isFinite(input.expiresAt) ? input.expiresAt : null,
    escalatedAt: null,
    resolvedAt: null,
    decision: null,
    resolvedBy: null,
    metadata: input.metadata,
  }
  return { queue: { ...queue, entries: [entry, ...queue.entries] }, entry, created: true }
}

// Urgency of a single entry at `now`. 'expired' > 'escalated' > 'aging' > 'normal'.
// Resolved entries report 'normal'. An entry's own slaMs overrides the queue SLA.
export function entryUrgency(queue: ApprovalQueue, entry: ApprovalQueueEntry, now = Date.now()): ApprovalQueueUrgency {
  if (entry.state !== 'pending') return entry.state === 'expired' ? 'expired' : 'normal'
  if (entry.expiresAt != null && now >= entry.expiresAt) return 'expired'
  const age = Math.max(0, now - entry.enqueuedAt)
  const sla = entry.slaMs ?? queue.slaMs
  if (age >= sla) return 'escalated'
  const aging = Math.min(queue.agingMs, sla)
  if (age >= aging) return 'aging'
  return 'normal'
}

// Advance the clock: mark newly-escalated (past SLA) and newly-expired (past hard
// deadline) entries. Idempotent — escalatedAt/state are only stamped once, so calling
// this repeatedly does not re-fire. Returns the sets that CHANGED on this pass.
export function ageApprovalQueue(queue: ApprovalQueue, now = Date.now()): AgeResult {
  const newlyEscalated: ApprovalQueueEntry[] = []
  const newlyExpired: ApprovalQueueEntry[] = []
  let changed = false
  const entries = queue.entries.map((entry) => {
    if (entry.state !== 'pending') return entry
    if (entry.expiresAt != null && now >= entry.expiresAt) {
      changed = true
      const expired: ApprovalQueueEntry = { ...entry, state: 'expired', resolvedAt: now }
      newlyExpired.push(expired)
      return expired
    }
    const sla = entry.slaMs ?? queue.slaMs
    if (entry.escalatedAt == null && now - entry.enqueuedAt >= sla) {
      changed = true
      const escalated: ApprovalQueueEntry = { ...entry, escalatedAt: now }
      newlyEscalated.push(escalated)
      return escalated
    }
    return entry
  })
  return { queue: changed ? { ...queue, entries } : queue, changed, newlyEscalated, newlyExpired }
}

// Resolve exactly one pending entry by idempotency key. No-op (empty resolved[]) if the
// key is unknown or already terminal — safe to replay.
export function resolveApproval(
  queue: ApprovalQueue,
  idempotencyKey: string,
  decision: ApprovalQueueDecision,
  resolvedBy: string | null = null,
  now = Date.now(),
): ResolveResult {
  const key = cleanKey(idempotencyKey)
  return applyResolution(queue, (entry) => entry.idempotencyKey === key, decision, resolvedBy, now)
}

// Batch-resolve every PENDING entry matching a predicate with one decision. The three
// common shapes are provided as helpers below (by scope key, by classification, by job).
export function batchResolveApprovals(
  queue: ApprovalQueue,
  match: (entry: ApprovalQueueEntry) => boolean,
  decision: ApprovalQueueDecision,
  resolvedBy: string | null = null,
  now = Date.now(),
): ResolveResult {
  return applyResolution(queue, match, decision, resolvedBy, now)
}

// Resolve all pending entries sharing a policy scope key — the natural "Approve all
// like this" batch. Scope keys come straight from approval-policy.ts.
export function resolveByScopeKey(queue: ApprovalQueue, scopeKey: string, decision: ApprovalQueueDecision, resolvedBy: string | null = null, now = Date.now()): ResolveResult {
  const key = String(scopeKey ?? '')
  return applyResolution(queue, (entry) => entry.scope.key === key, decision, resolvedBy, now)
}

// Resolve all pending entries for a job (e.g. when the job is cancelled/dead-lettered).
export function resolveByJob(queue: ApprovalQueue, jobId: string, decision: ApprovalQueueDecision, resolvedBy: string | null = null, now = Date.now()): ResolveResult {
  const id = String(jobId ?? '')
  return applyResolution(queue, (entry) => entry.jobId === id, decision, resolvedBy, now)
}

export function pendingApprovals(queue: ApprovalQueue): ApprovalQueueEntry[] {
  return queue.entries.filter((entry) => entry.state === 'pending')
}

// Pending entries ordered most-urgent-first, then oldest-first — the review order a UI
// or notifier should present. Stable for equal urgency (older enqueuedAt wins).
export function orderedPending(queue: ApprovalQueue, now = Date.now()): ApprovalQueueEntry[] {
  const rank: Record<ApprovalQueueUrgency, number> = { expired: 0, escalated: 1, aging: 2, normal: 3 }
  return pendingApprovals(queue).sort((a, b) => {
    const byUrgency = rank[entryUrgency(queue, a, now)] - rank[entryUrgency(queue, b, now)]
    if (byUrgency !== 0) return byUrgency
    return a.enqueuedAt - b.enqueuedAt
  })
}

export interface ApprovalQueueSummary {
  pending: number
  aging: number
  escalated: number
  expired: number
  resolved: number
  oldestPendingAgeMs: number | null
}

export function summarizeApprovalQueue(queue: ApprovalQueue, now = Date.now()): ApprovalQueueSummary {
  let pending = 0
  let aging = 0
  let escalated = 0
  let expired = 0
  let resolved = 0
  let oldest: number | null = null
  for (const entry of queue.entries) {
    if (entry.state === 'resolved') { resolved++; continue }
    if (entry.state === 'expired') { expired++; continue }
    pending++
    const age = now - entry.enqueuedAt
    if (oldest == null || age > oldest) oldest = age
    const urgency = entryUrgency(queue, entry, now)
    if (urgency === 'escalated') escalated++
    else if (urgency === 'expired') { expired++; pending--; }
    else if (urgency === 'aging') aging++
  }
  return { pending, aging, escalated, expired, resolved, oldestPendingAgeMs: oldest }
}

// Drop terminal (resolved/expired) rows older than `olderThanMs` so the persisted queue
// does not grow without bound. Pending rows are always kept.
export function pruneApprovalQueue(queue: ApprovalQueue, olderThanMs: number, now = Date.now()): ApprovalQueue {
  const cutoff = now - Math.max(0, olderThanMs)
  const entries = queue.entries.filter((entry) => {
    if (entry.state === 'pending') return true
    const at = entry.resolvedAt ?? entry.enqueuedAt
    return at > cutoff
  })
  return entries.length === queue.entries.length ? queue : { ...queue, entries }
}

// ---- internals ----

function applyResolution(
  queue: ApprovalQueue,
  match: (entry: ApprovalQueueEntry) => boolean,
  decision: ApprovalQueueDecision,
  resolvedBy: string | null,
  now: number,
): ResolveResult {
  const decisionValue = normalizeDecision(decision)
  const resolved: ApprovalQueueEntry[] = []
  const entries = queue.entries.map((entry) => {
    if (entry.state !== 'pending' || !match(entry)) return entry
    const next: ApprovalQueueEntry = {
      ...entry,
      state: 'resolved',
      decision: decisionValue,
      resolvedAt: now,
      resolvedBy: resolvedBy ?? null,
    }
    resolved.push(next)
    return next
  })
  if (!resolved.length) return { queue, resolved }
  return { queue: { ...queue, entries }, resolved }
}

function normalizeEntries(input: ApprovalQueueEntry[], now: number): ApprovalQueueEntry[] {
  const seen = new Set<string>()
  const out: ApprovalQueueEntry[] = []
  for (const raw of input) {
    if (!raw || typeof raw !== 'object') continue
    const key = cleanKey((raw as ApprovalQueueEntry).idempotencyKey)
    if (!key || seen.has(key)) continue // exactly-once: collapse persisted duplicates
    seen.add(key)
    const state: ApprovalQueueState = raw.state === 'resolved' || raw.state === 'expired' ? raw.state : 'pending'
    out.push({
      idempotencyKey: key,
      jobId: String(raw.jobId ?? ''),
      callId: String(raw.callId ?? ''),
      tool: String(raw.tool ?? 'tool'),
      scope: normalizeScope(raw.scope, raw.tool),
      state,
      enqueuedAt: finiteOr(raw.enqueuedAt, now),
      requeueCount: Math.max(0, Math.floor(finiteOr(raw.requeueCount, 0))),
      slaMs: raw.slaMs == null ? null : positive(raw.slaMs, DEFAULT_SLA_MS),
      expiresAt: typeof raw.expiresAt === 'number' && Number.isFinite(raw.expiresAt) ? raw.expiresAt : null,
      escalatedAt: typeof raw.escalatedAt === 'number' && Number.isFinite(raw.escalatedAt) ? raw.escalatedAt : null,
      resolvedAt: state === 'pending' ? null : finiteOr(raw.resolvedAt, now),
      decision: state === 'resolved' ? normalizeDecision(raw.decision) : null,
      resolvedBy: state === 'resolved' && typeof raw.resolvedBy === 'string' ? raw.resolvedBy : null,
      metadata: raw.metadata && typeof raw.metadata === 'object' ? raw.metadata : undefined,
    })
  }
  return out
}

function normalizeScope(scope: ApprovalQueueScope | undefined, toolFallback: unknown): ApprovalQueueScope {
  const tool = typeof toolFallback === 'string' && toolFallback ? toolFallback : 'tool'
  const s = (scope && typeof scope === 'object' ? scope : {}) as Partial<ApprovalQueueScope>
  return {
    scope: s.scope === 'connector' ? 'connector' : 'tool',
    key: typeof s.key === 'string' && s.key ? s.key : tool,
    label: typeof s.label === 'string' && s.label ? s.label : tool,
    explanation: typeof s.explanation === 'string' ? s.explanation : undefined,
    classification: typeof s.classification === 'string' ? s.classification : undefined,
  }
}

function normalizeDecision(value: unknown): ApprovalQueueDecision {
  return value === 'allow' || value === 'allow_always' || value === 'deny' ? value : 'deny'
}

function cleanKey(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function finiteOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function positive(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback
}
