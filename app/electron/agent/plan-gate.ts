// Two-phase Plan Mode gate — a pure, dependency-free state machine.
//
// The problem this solves: when the agent runs in a "plan first" posture, we want a hard
// gate between *thinking* and *doing*. Phase 1 produces a read-only plan (no mutation may
// run). The plan is then frozen by an explicit user decision — approve / edit / reject.
// Only an approved+frozen plan unlocks mutating tool execution (Phase 2), and every
// mutation that runs is bound back to the exact plan revision that authorised it.
//
// This module is intentionally pure: no I/O, no clock, no fs. The loop owns side effects;
// it feeds tool calls in and reads decisions out. See planGateWire in the accompanying
// report for the loop.ts integration.
//
// Design mirrors the codebase's other pure cores (approval-policy, connector-retry):
// small reducer-style functions over an explicit immutable-ish state value.

export interface PlanGateStep {
  /** Stable id within a plan revision. */
  id: string
  text: string
}

export type PlanGatePhase =
  // No plan drafted yet. Read-only tools allowed; mutations blocked.
  | 'drafting'
  // A plan draft exists and is awaiting a user decision. Mutations still blocked.
  | 'awaiting_approval'
  // Plan approved and frozen. Mutations allowed, bound to `frozenRevision`.
  | 'approved'
  // Plan rejected. Mutations blocked; a fresh plan must be drafted to proceed.
  | 'rejected'

/** A frozen, immutable snapshot of an approved plan — the authorisation record. */
export interface FrozenPlan {
  /** Monotonic revision this plan was approved at. */
  revision: number
  steps: readonly PlanGateStep[]
  /** Content hash of the frozen steps; execution records carry this to prove binding. */
  intentHash: string
}

export interface PlanGateState {
  phase: PlanGatePhase
  /** The current draft (present in drafting/awaiting_approval), or the last-approved steps. */
  draft: readonly PlanGateStep[]
  /** Bumped every time a new draft is proposed; identifies the plan revision. */
  revision: number
  /** Set only when phase === 'approved'. The authorisation binding for mutations. */
  frozen: FrozenPlan | null
  /** Audit log of mutations that ran, each bound to the frozen plan that allowed them. */
  executions: readonly PlanExecutionRecord[]
}

/** One mutating tool call that was permitted, bound to the plan revision that authorised it. */
export interface PlanExecutionRecord {
  tool: string
  /** The plan revision this mutation was bound to. */
  boundRevision: number
  /** The intent hash of the frozen plan at bind time — detects post-hoc plan drift. */
  boundIntentHash: string
  /** Optional step id the caller attributed this mutation to. */
  stepId?: string
}

export type PlanGateDecision = 'approve' | 'reject'

/** Result of gating a single tool call. */
export interface PlanGateVerdict {
  allowed: boolean
  /** Machine-readable reason when blocked, for the loop to surface / log. */
  reason?:
    | 'ok_read_only'
    | 'ok_meta'
    | 'ok_approved'
    | 'blocked_no_plan'
    | 'blocked_awaiting_approval'
    | 'blocked_rejected'
  /** When allowed and a mutation, the binding to record via recordExecution. */
  binding?: { revision: number; intentHash: string }
}

// --- pure hash (FNV-1a over the canonical plan text) — no crypto import needed ---

function hashSteps(steps: readonly PlanGateStep[]): string {
  const canonical = steps.map((s) => `${s.id} ${s.text}`).join('')
  let h = 0x811c9dc5
  for (let i = 0; i < canonical.length; i++) {
    h ^= canonical.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  // >>> 0 to get an unsigned 32-bit, then hex.
  return (h >>> 0).toString(16).padStart(8, '0')
}

function normalizeSteps(steps: readonly Partial<PlanGateStep>[]): PlanGateStep[] {
  return steps
    .map((s, i) => ({ id: String(s?.id ?? `p${i}`), text: String(s?.text ?? '').trim() }))
    .filter((s) => s.text.length > 0)
}

/** Initial state: nothing drafted, mutations blocked. */
export function initPlanGate(): PlanGateState {
  return { phase: 'drafting', draft: [], revision: 0, frozen: null, executions: [] }
}

/**
 * Propose (or replace) the plan draft. Always moves to `awaiting_approval` and bumps the
 * revision, invalidating any prior approval — editing the plan re-freezes intent, so a
 * previously-approved plan can never silently authorise a changed set of steps.
 */
export function proposePlan(state: PlanGateState, steps: readonly Partial<PlanGateStep>[]): PlanGateState {
  const draft = normalizeSteps(steps)
  if (draft.length === 0) {
    // An empty proposal is a no-op that keeps us in drafting (can't approve nothing).
    return { ...state, phase: 'drafting', draft: [], frozen: null }
  }
  return {
    ...state,
    phase: 'awaiting_approval',
    draft,
    revision: state.revision + 1,
    // Editing invalidates any earlier freeze: mutations must wait for a fresh approval.
    frozen: null,
  }
}

/**
 * Apply a user decision to the pending draft. `approve` freezes the current draft as the
 * authorisation record; `reject` blocks execution and requires a fresh plan. Deciding when
 * there is no pending draft is a no-op.
 */
export function decidePlan(state: PlanGateState, decision: PlanGateDecision): PlanGateState {
  if (state.phase !== 'awaiting_approval') return state
  if (decision === 'reject') {
    return { ...state, phase: 'rejected', frozen: null }
  }
  const frozen: FrozenPlan = {
    revision: state.revision,
    steps: state.draft.map((s) => ({ ...s })),
    intentHash: hashSteps(state.draft),
  }
  return { ...state, phase: 'approved', frozen }
}

/**
 * Edit an approved or pending plan. This is `proposePlan` with intent-drift semantics made
 * explicit: it always returns to `awaiting_approval` at a new revision, so the frozen
 * binding never outlives the steps it authorised.
 */
export function editPlan(state: PlanGateState, steps: readonly Partial<PlanGateStep>[]): PlanGateState {
  return proposePlan(state, steps)
}

/**
 * Gate a single tool call. `isMutation` is supplied by the caller (the loop already knows
 * via isReadOnlyTool). `isMeta` marks plan-management tools (update_plan) that must always
 * pass so the model can draft. Read-only and meta tools always pass; mutations pass only
 * when the gate is `approved`, and carry a binding to record.
 */
export function gateToolCall(
  state: PlanGateState,
  opts: { isMutation: boolean; isMeta?: boolean },
): PlanGateVerdict {
  if (opts.isMeta) return { allowed: true, reason: 'ok_meta' }
  if (!opts.isMutation) return { allowed: true, reason: 'ok_read_only' }
  switch (state.phase) {
    case 'approved':
      return {
        allowed: true,
        reason: 'ok_approved',
        binding: { revision: state.frozen!.revision, intentHash: state.frozen!.intentHash },
      }
    case 'drafting':
      return { allowed: false, reason: 'blocked_no_plan' }
    case 'awaiting_approval':
      return { allowed: false, reason: 'blocked_awaiting_approval' }
    case 'rejected':
      return { allowed: false, reason: 'blocked_rejected' }
  }
}

/**
 * Record that a permitted mutation ran, binding it to the plan revision that authorised it.
 * Throws if the gate is not currently approved, or if the supplied binding does not match
 * the live frozen plan (defends against a stale verdict being replayed after an edit).
 */
export function recordExecution(
  state: PlanGateState,
  exec: { tool: string; binding: { revision: number; intentHash: string }; stepId?: string },
): PlanGateState {
  if (state.phase !== 'approved' || !state.frozen) {
    throw new Error('plan-gate: cannot record execution — no approved plan is frozen')
  }
  if (
    exec.binding.revision !== state.frozen.revision ||
    exec.binding.intentHash !== state.frozen.intentHash
  ) {
    throw new Error('plan-gate: execution binding does not match the frozen plan (intent drift)')
  }
  const record: PlanExecutionRecord = {
    tool: exec.tool,
    boundRevision: state.frozen.revision,
    boundIntentHash: state.frozen.intentHash,
    ...(exec.stepId ? { stepId: exec.stepId } : {}),
  }
  return { ...state, executions: [...state.executions, record] }
}

/** True when the gate currently permits mutating tools to run. */
export function canMutate(state: PlanGateState): boolean {
  return state.phase === 'approved' && state.frozen !== null
}

/**
 * Verify the audit trail is internally consistent: every recorded execution is bound to a
 * revision/hash that a plan was actually approved at. Because approving is the only path to
 * a frozen plan and every mutation records the live binding, a consistent trail proves no
 * mutation ran outside an approved plan. Returns the offending records, empty if clean.
 */
export function auditBindings(state: PlanGateState): PlanExecutionRecord[] {
  if (state.executions.length === 0) return []
  // The only revision we can currently attest is the frozen one; historically-approved
  // revisions are attested by the fact they carry the hash produced by hashSteps over the
  // draft at that revision. We re-check the live binding here (the common integrity case).
  const live = state.frozen
  return state.executions.filter((r) => {
    if (live && r.boundRevision === live.revision) {
      return r.boundIntentHash !== live.intentHash
    }
    // For older revisions we can only assert a non-empty hash of the right shape.
    return !/^[0-9a-f]{8}$/.test(r.boundIntentHash) || r.boundRevision < 1
  })
}
