// Pure, dependency-free difficulty / cost-aware model tiering.
//
// The existing model-router.ts does failover + static role/account selection and
// spend controls. What it does NOT do is decide, from the *shape of the task*, how
// much horsepower the job actually needs: an "ok, thanks" ack or a compaction pass
// should route to a cheap model, while a "plan this refactor" or a hard multi-file
// edit should route to a frontier model. This module fills that gap.
//
// It is a pure classifier: (task text + job class + a tier policy) -> a tier + score
// + reasons. It has ZERO side effects and no I/O. The caller feeds the chosen tier
// into model-router's role/account selection (e.g. by picking the tier's model list).

export type ModelTier = 'cheap' | 'standard' | 'frontier'

// Job classes the harness already reasons about (subset of ModelRouterRole plus a
// few internal maintenance jobs). Kept as a string union so callers can map their
// own role names in; unknown classes fall back to `unknown` handling.
export type JobClass =
  | 'ack'
  | 'scoring'
  | 'compaction'
  | 'summarize'
  | 'classify'
  | 'connector'
  | 'chat'
  | 'assistant_reply'
  | 'edit'
  | 'build'
  | 'review'
  | 'plan'
  | 'mission'
  | 'debug'
  | 'unknown'

export interface TierPolicy {
  // Baseline tier each job class starts at before difficulty signals nudge it.
  jobClassBaseline: Partial<Record<JobClass, ModelTier>>
  // Fallback baseline when the job class is not in the map above.
  defaultBaseline: ModelTier
  // If true, difficulty signals can only raise the tier above baseline, never lower it.
  // (Useful when a policy pins e.g. `plan` to frontier regardless of task text.)
  neverDowngradeBaseline?: JobClass[]
  // Hard floor / ceiling applied last (e.g. a budget-constrained policy caps at 'standard').
  maxTier?: ModelTier
  minTier?: ModelTier
  // Number of input tokens (approx) at/above which a job is treated as "large context",
  // which nudges toward a higher tier. Defaults to 24_000.
  largeContextTokens?: number
  // Number of tool calls the job is expected to make; many tool calls => harder.
  // Defaults to 6.
  manyToolsThreshold?: number
}

export interface TaskSignal {
  jobClass: JobClass
  // The user/task text driving the decision (message, instruction, ticket body…).
  text?: string | null
  // Optional structured hints (all optional; classifier degrades gracefully).
  approxInputTokens?: number | null
  expectedToolCalls?: number | null
  filesToEdit?: number | null
  // Caller can force a floor/ceiling for this one task (e.g. user picked "high effort").
  forceMinTier?: ModelTier | null
  forceMaxTier?: ModelTier | null
  // A prior attempt failed / needs a retry at a stronger model.
  retryEscalation?: boolean | null
}

export interface TierReason {
  signal: string
  effect: 'raise' | 'lower' | 'pin' | 'clamp'
  detail: string
}

export interface TierDecision {
  tier: ModelTier
  baseline: ModelTier
  // 0..1 difficulty estimate that produced the tier (before clamps).
  difficulty: number
  jobClass: JobClass
  reasons: TierReason[]
}

const TIER_ORDER: ModelTier[] = ['cheap', 'standard', 'frontier']

export const DEFAULT_TIER_POLICY: TierPolicy = {
  jobClassBaseline: {
    ack: 'cheap',
    scoring: 'cheap',
    compaction: 'cheap',
    summarize: 'cheap',
    classify: 'cheap',
    connector: 'cheap',
    chat: 'standard',
    assistant_reply: 'standard',
    edit: 'standard',
    build: 'standard',
    review: 'frontier',
    plan: 'frontier',
    mission: 'frontier',
    debug: 'frontier',
  },
  defaultBaseline: 'standard',
  neverDowngradeBaseline: ['plan', 'mission', 'review'],
  largeContextTokens: 24_000,
  manyToolsThreshold: 6,
}

function tierIndex(tier: ModelTier): number {
  return TIER_ORDER.indexOf(tier)
}

function tierFromIndex(index: number): ModelTier {
  return TIER_ORDER[Math.max(0, Math.min(TIER_ORDER.length - 1, index))]
}

function raise(tier: ModelTier, steps = 1): ModelTier {
  return tierFromIndex(tierIndex(tier) + steps)
}

function lower(tier: ModelTier, steps = 1): ModelTier {
  return tierFromIndex(tierIndex(tier) - steps)
}

function maxTier(a: ModelTier, b: ModelTier): ModelTier {
  return tierIndex(a) >= tierIndex(b) ? a : b
}

function minTier(a: ModelTier, b: ModelTier): ModelTier {
  return tierIndex(a) <= tierIndex(b) ? a : b
}

// Words that signal a genuinely hard / high-stakes task (raise difficulty).
const HARD_PATTERNS: RegExp[] = [
  /\b(architect|architecture|design|refactor|redesign|migrat(?:e|ion)|rewrite)\b/i,
  /\b(debug|root[\s-]?cause|race condition|deadlock|memory leak|regression)\b/i,
  /\b(plan|strategy|trade[\s-]?off|approach|reason(?:ing)? through|think through)\b/i,
  /\b(security|vulnerabilit|exploit|threat model|cryptograph)/i,
  /\b(multi[\s-]?file|across (?:the|several|multiple) files|whole codebase|end[\s-]?to[\s-]?end)\b/i,
  /\b(prove|proof|invariant|correctness|concurren)/i,
  /\b(complex|tricky|subtle|non[\s-]?trivial|hard\b)/i,
]

// Words that signal a trivial / mechanical task (lower difficulty).
const EASY_PATTERNS: RegExp[] = [
  /\b(ack(?:nowledge)?|thanks|thank you|got it|ok(?:ay)?|sounds good|will do|noted)\b/i,
  /\b(typo|rename|format(?:ting)?|lint|whitespace|comment|bump (?:the )?version)\b/i,
  /\b(yes|no|confirm|approve|reject|👍|👎)\b/i,
  /\b(summar(?:ise|ize|y)|tl;?dr|one[\s-]?liner|short answer)\b/i,
]

function scoreText(text: string): { delta: number; hits: string[]; easyHits: string[] } {
  const hits: string[] = []
  const easyHits: string[] = []
  let delta = 0
  for (const rx of HARD_PATTERNS) {
    const m = text.match(rx)
    if (m) { hits.push(m[0].toLowerCase()); delta += 0.12 }
  }
  for (const rx of EASY_PATTERNS) {
    const m = text.match(rx)
    if (m) { easyHits.push(m[0].toLowerCase()); delta -= 0.12 }
  }
  return { delta: Math.max(-0.5, Math.min(0.5, delta)), hits, easyHits }
}

function baselineTier(policy: TierPolicy, jobClass: JobClass): ModelTier {
  return policy.jobClassBaseline[jobClass] ?? policy.defaultBaseline
}

function baselineDifficulty(tier: ModelTier): number {
  switch (tier) {
    case 'cheap': return 0.2
    case 'standard': return 0.5
    case 'frontier': return 0.8
  }
}

/**
 * Classify a single task into a model tier, purely from its signal + a policy.
 * No I/O, deterministic, side-effect-free.
 */
export function classifyTaskTier(signal: TaskSignal, policy: TierPolicy = DEFAULT_TIER_POLICY): TierDecision {
  const jobClass = signal.jobClass
  const baseline = baselineTier(policy, jobClass)
  const reasons: TierReason[] = []

  let difficulty = baselineDifficulty(baseline)
  reasons.push({ signal: 'job-class', effect: 'pin', detail: `Job class "${jobClass}" baselines at "${baseline}".` })

  // Text difficulty signals.
  const text = (signal.text ?? '').trim()
  if (text) {
    const { delta, hits, easyHits } = scoreText(text)
    difficulty += delta
    if (hits.length) {
      reasons.push({ signal: 'text-hard', effect: 'raise', detail: `Hard-task cues: ${hits.slice(0, 4).join(', ')}.` })
    }
    if (easyHits.length) {
      reasons.push({ signal: 'text-easy', effect: 'lower', detail: `Trivial-task cues: ${easyHits.slice(0, 4).join(', ')}.` })
    }
    // Very short messages are almost always cheap-worthy (unless hard cues present).
    if (text.length <= 24 && !hits.length) {
      difficulty -= 0.15
      reasons.push({ signal: 'text-short', effect: 'lower', detail: `Very short input (${text.length} chars).` })
    }
  }

  // Large context => harder to reason over.
  const largeCtx = policy.largeContextTokens ?? 24_000
  if (typeof signal.approxInputTokens === 'number' && Number.isFinite(signal.approxInputTokens) && signal.approxInputTokens >= largeCtx) {
    difficulty += 0.15
    reasons.push({ signal: 'large-context', effect: 'raise', detail: `~${signal.approxInputTokens} tokens ≥ ${largeCtx}.` })
  }

  // Many tool calls / many files => harder, agentic work.
  const manyTools = policy.manyToolsThreshold ?? 6
  if (typeof signal.expectedToolCalls === 'number' && Number.isFinite(signal.expectedToolCalls) && signal.expectedToolCalls >= manyTools) {
    difficulty += 0.12
    reasons.push({ signal: 'many-tools', effect: 'raise', detail: `Expected ${signal.expectedToolCalls} tool calls ≥ ${manyTools}.` })
  }
  if (typeof signal.filesToEdit === 'number' && Number.isFinite(signal.filesToEdit) && signal.filesToEdit >= 3) {
    difficulty += 0.1
    reasons.push({ signal: 'multi-file', effect: 'raise', detail: `Editing ${signal.filesToEdit} files.` })
  }

  // Retry after a prior failure escalates.
  if (signal.retryEscalation) {
    difficulty += 0.2
    reasons.push({ signal: 'retry', effect: 'raise', detail: 'Escalating after a prior attempt failed.' })
  }

  difficulty = Math.max(0, Math.min(1, difficulty))

  // Map difficulty back onto a tier.
  let tier: ModelTier = difficulty >= 0.66 ? 'frontier' : difficulty >= 0.35 ? 'standard' : 'cheap'

  // Never-downgrade: some job classes are pinned at least to their baseline.
  if (policy.neverDowngradeBaseline?.includes(jobClass) && tierIndex(tier) < tierIndex(baseline)) {
    tier = baseline
    reasons.push({ signal: 'never-downgrade', effect: 'clamp', detail: `Job class "${jobClass}" cannot drop below baseline "${baseline}".` })
  }

  // Per-task caller floor/ceiling.
  if (signal.forceMinTier && tierIndex(tier) < tierIndex(signal.forceMinTier)) {
    tier = signal.forceMinTier
    reasons.push({ signal: 'force-min', effect: 'clamp', detail: `Caller pinned a floor of "${signal.forceMinTier}".` })
  }
  if (signal.forceMaxTier && tierIndex(tier) > tierIndex(signal.forceMaxTier)) {
    tier = signal.forceMaxTier
    reasons.push({ signal: 'force-max', effect: 'clamp', detail: `Caller pinned a ceiling of "${signal.forceMaxTier}".` })
  }

  // Policy floor/ceiling applied last (e.g. budget policy caps at standard).
  if (policy.minTier && tierIndex(tier) < tierIndex(policy.minTier)) {
    tier = maxTier(tier, policy.minTier)
    reasons.push({ signal: 'policy-min', effect: 'clamp', detail: `Policy floor "${policy.minTier}".` })
  }
  if (policy.maxTier && tierIndex(tier) > tierIndex(policy.maxTier)) {
    tier = minTier(tier, policy.maxTier)
    reasons.push({ signal: 'policy-max', effect: 'clamp', detail: `Policy ceiling "${policy.maxTier}".` })
  }

  return { tier, baseline, difficulty: Math.round(difficulty * 1000) / 1000, jobClass, reasons }
}

/**
 * Given a tier decision and a map of tier -> ordered model list, produce the
 * ordered preference of models for that tier, falling back to adjacent tiers so
 * the caller always gets *some* model even if a tier is empty.
 */
export function modelsForTier(
  decision: Pick<TierDecision, 'tier'>,
  tierModels: Partial<Record<ModelTier, string[]>>,
): string[] {
  const order: ModelTier[] = decision.tier === 'frontier'
    ? ['frontier', 'standard', 'cheap']
    : decision.tier === 'standard'
      ? ['standard', 'frontier', 'cheap']
      : ['cheap', 'standard', 'frontier']
  const seen = new Set<string>()
  const out: string[] = []
  for (const tier of order) {
    for (const model of tierModels[tier] ?? []) {
      if (!seen.has(model)) { seen.add(model); out.push(model) }
    }
  }
  return out
}

/**
 * Classify a batch of tasks under a single policy. Convenience for callers that
 * pre-classify a queue (e.g. deciding which of N pending jobs deserve frontier).
 */
export function classifyBatch(signals: TaskSignal[], policy: TierPolicy = DEFAULT_TIER_POLICY): TierDecision[] {
  return signals.map((signal) => classifyTaskTier(signal, policy))
}
