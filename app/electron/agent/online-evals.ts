// Online eval scoring + threshold alerts on live runs.
//
// Pure, dependency-injected scoring pass over a run. Given a stream of run
// samples (assistant turns, each carrying the text produced and optional token
// usage), it applies a set of scorers to a sampled subset of turns, aggregates
// their scores, and evaluates threshold rules that emit typed alerts
// (quality / cost). Every side-effecting dependency — the LLM judge, the RNG
// used for sampling, the clock — is injected, so the whole module is unit
// testable with no real model call, no randomness, and no wall-clock.
//
// This module intentionally does NOT import the shared AgentEvent / token-usage
// types (keeping it standalone + cheap to compile). Callers adapt their run
// into the small RunSample shape below; the parent wires that adapter in.

// ---- Inputs -------------------------------------------------------------

/** One assistant turn from a live run, reduced to what scorers need. */
export interface RunSample {
  /** Monotonic index of this turn within the run (0-based). */
  index: number
  /** The assistant-visible text produced on this turn. */
  text: string
  /** Total tokens billed for this turn, if known. */
  totalTokens?: number
  /** USD cost of this turn, if the caller can compute it. */
  costUsd?: number
  /** Optional freeform tags (e.g. tool names used) a scorer may inspect. */
  tags?: string[]
}

/** A single scorer verdict for one sample: score in [0,1], 1 == best. */
export interface ScoreResult {
  /** Stable id of the scorer that produced this. */
  scorer: string
  /** Normalised quality score in [0,1]. */
  score: number
  /** Optional human-readable justification. */
  rationale?: string
}

/**
 * A scorer. Heuristic scorers are sync; the LLM-judge scorer is async. Both are
 * modelled as possibly-async so the pass can await uniformly. A scorer returns
 * null to abstain (e.g. empty text) — abstentions are excluded from aggregates.
 */
export interface Scorer {
  id: string
  /** 'quality' scorers feed quality alerts; 'cost' scorers feed cost alerts. */
  kind: 'quality' | 'cost'
  score(sample: RunSample): ScoreResult | null | Promise<ScoreResult | null>
}

/**
 * A purely synchronous scorer — the heuristic built-ins. Assignable to `Scorer`
 * but keeps a sync return type so callers (and tests) can use its result
 * without awaiting.
 */
export interface SyncScorer extends Scorer {
  score(sample: RunSample): ScoreResult | null
}

/** The injected LLM judge: returns a score in [0,1] for one sample's text. */
export type JudgeFn = (input: { text: string; index: number }) =>
  Promise<{ score: number; rationale?: string }>

/** A threshold rule over the aggregate of one scorer (or 'cost'/'any'). */
export interface ThresholdRule {
  /** Alert category emitted when this rule trips. */
  kind: 'quality' | 'cost'
  /** Which scorer's mean to test, or 'cost' for total run cost. */
  metric: string
  /** Comparison: quality rules usually 'below'; cost rules 'above'. */
  op: 'below' | 'above'
  /** The threshold value. For quality: mean score in [0,1]. For cost: USD. */
  threshold: number
  severity: 'warn' | 'critical'
  /** Human message template; {value} and {threshold} are substituted. */
  message: string
}

export interface EvalConfig {
  scorers: Scorer[]
  rules: ThresholdRule[]
  /** Sample at most this fraction of turns (0<f<=1). Default 1 (all). */
  sampleRate?: number
  /** Always score turns whose index is a multiple of this (e.g. every 5th). */
  sampleEvery?: number
  /** Injected RNG in [0,1). Default Math.random. */
  rng?: () => number
  /** Injected clock (ms). Default Date.now. */
  now?: () => number
}

// ---- Outputs ------------------------------------------------------------

export interface ScorerAggregate {
  scorer: string
  kind: 'quality' | 'cost'
  /** Mean score across scored (non-abstained) samples, or null if none. */
  mean: number | null
  /** How many samples this scorer actually scored. */
  count: number
}

export interface EvalAlert {
  kind: 'quality' | 'cost'
  metric: string
  severity: 'warn' | 'critical'
  op: 'below' | 'above'
  value: number
  threshold: number
  message: string
  at: number
}

export interface EvalReport {
  /** Indices of turns that were sampled/scored. */
  sampledIndices: number[]
  scores: ScoreResult[]
  aggregates: ScorerAggregate[]
  totalCostUsd: number
  totalTokens: number
  alerts: EvalAlert[]
}

// ---- Built-in heuristic scorers -----------------------------------------

const clamp01 = (n: number): number => (n <= 0 ? 0 : n >= 1 ? 1 : n)

/**
 * Non-emptiness / substance heuristic: penalises trivially short or obviously
 * refusal/error-shaped turns. Cheap first-line quality signal.
 */
export function substanceScorer(): SyncScorer {
  const refusal = /\b(i can'?t|i cannot|i'?m unable|i am unable|as an ai)\b/i
  return {
    id: 'substance',
    kind: 'quality',
    score(sample) {
      const text = (sample.text ?? '').trim()
      if (!text) return null // abstain on empty turns
      const len = text.length
      // Ramp 0..1 over the first ~400 chars, then flat.
      let score = clamp01(len / 400)
      if (refusal.test(text)) score = Math.min(score, 0.2)
      return { scorer: 'substance', score, rationale: `len=${len}` }
    },
  }
}

/**
 * Cost-efficiency scorer: rewards turns that produce output without runaway
 * token spend. Score decays as tokens exceed `budgetTokens`. Abstains when the
 * turn reports no token usage.
 */
export function tokenBudgetScorer(budgetTokens: number): SyncScorer {
  const budget = Math.max(1, budgetTokens)
  return {
    id: 'token_budget',
    kind: 'cost',
    score(sample) {
      const tokens = sample.totalTokens
      if (tokens == null || !Number.isFinite(tokens) || tokens <= 0) return null
      // 1.0 at/under budget, decaying toward 0 as tokens grow past it.
      const score = tokens <= budget ? 1 : clamp01(budget / tokens)
      return { scorer: 'token_budget', score, rationale: `tokens=${tokens}/${budget}` }
    },
  }
}

/**
 * Wraps an injected LLM judge as a quality scorer. The judge fn is the only I/O;
 * pass a fake in tests. Abstains on empty text (no judge call made).
 */
export function judgeScorer(judge: JudgeFn, id = 'llm_judge'): Scorer {
  return {
    id,
    kind: 'quality',
    async score(sample) {
      const text = (sample.text ?? '').trim()
      if (!text) return null
      const out = await judge({ text, index: sample.index })
      const raw = Number(out?.score)
      const score = clamp01(Number.isFinite(raw) ? raw : 0)
      return { scorer: id, score, rationale: out?.rationale }
    },
  }
}

// ---- Sampling -----------------------------------------------------------

/**
 * Decide which sample indices to score. A turn is scored if it hits the
 * deterministic `sampleEvery` cadence OR passes the random `sampleRate` gate.
 * Index 0 is always scored (so short runs are never silently unscored).
 */
export function pickSamples(
  samples: RunSample[],
  opts: { sampleRate?: number; sampleEvery?: number; rng?: () => number } = {},
): number[] {
  const rate = opts.sampleRate == null ? 1 : clamp01(opts.sampleRate)
  const every = opts.sampleEvery && opts.sampleEvery > 0 ? Math.floor(opts.sampleEvery) : 0
  const rng = opts.rng ?? Math.random
  const out: number[] = []
  for (const s of samples) {
    const first = s.index === 0
    const cadence = every > 0 && s.index % every === 0
    const random = rate >= 1 ? true : rate <= 0 ? false : rng() < rate
    if (first || cadence || random) out.push(s.index)
  }
  return out
}

// ---- The scoring pass ---------------------------------------------------

function substitute(template: string, value: number, threshold: number): string {
  return template
    .replace(/\{value\}/g, formatNum(value))
    .replace(/\{threshold\}/g, formatNum(threshold))
}

function formatNum(n: number): string {
  if (!Number.isFinite(n)) return String(n)
  return Number.isInteger(n) ? String(n) : n.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}

/**
 * Run the full online-eval pass over a set of run samples.
 *
 * Pure aside from the injected scorers/judge/rng/clock. Deterministic given a
 * seeded rng and a fixed clock. Never throws for a misbehaving scorer — a
 * scorer that throws is treated as an abstention so one bad scorer can't sink
 * the whole eval on a live run.
 */
export async function runOnlineEval(samples: RunSample[], config: EvalConfig): Promise<EvalReport> {
  const now = config.now ?? Date.now
  const at = now()
  const sampledIndices = pickSamples(samples, {
    sampleRate: config.sampleRate,
    sampleEvery: config.sampleEvery,
    rng: config.rng,
  })
  const sampledSet = new Set(sampledIndices)
  const scored = samples.filter((s) => sampledSet.has(s.index))

  const scores: ScoreResult[] = []
  const sums = new Map<string, { sum: number; count: number; kind: 'quality' | 'cost' }>()

  for (const sample of scored) {
    for (const scorer of config.scorers) {
      let result: ScoreResult | null = null
      try {
        result = await scorer.score(sample)
      } catch {
        result = null // a throwing scorer abstains; never sinks the run
      }
      if (!result) continue
      const score = clamp01(Number.isFinite(result.score) ? result.score : 0)
      const normalized: ScoreResult = { ...result, score }
      scores.push(normalized)
      const agg = sums.get(scorer.id) ?? { sum: 0, count: 0, kind: scorer.kind }
      agg.sum += score
      agg.count += 1
      sums.set(scorer.id, agg)
    }
  }

  const aggregates: ScorerAggregate[] = config.scorers.map((scorer) => {
    const agg = sums.get(scorer.id)
    return {
      scorer: scorer.id,
      kind: scorer.kind,
      mean: agg && agg.count > 0 ? agg.sum / agg.count : null,
      count: agg?.count ?? 0,
    }
  })

  // Cost/token totals are over ALL samples, not just scored ones — spend is real
  // regardless of whether a turn was picked for quality scoring.
  let totalCostUsd = 0
  let totalTokens = 0
  for (const s of samples) {
    if (Number.isFinite(s.costUsd)) totalCostUsd += Math.max(0, s.costUsd as number)
    if (Number.isFinite(s.totalTokens)) totalTokens += Math.max(0, s.totalTokens as number)
  }

  const alerts: EvalAlert[] = []
  for (const rule of config.rules) {
    const value = resolveMetric(rule.metric, aggregates, totalCostUsd, totalTokens)
    if (value == null) continue // metric unavailable (e.g. no samples scored)
    const tripped = rule.op === 'below' ? value < rule.threshold : value > rule.threshold
    if (!tripped) continue
    alerts.push({
      kind: rule.kind,
      metric: rule.metric,
      severity: rule.severity,
      op: rule.op,
      value,
      threshold: rule.threshold,
      message: substitute(rule.message, value, rule.threshold),
      at,
    })
  }

  return { sampledIndices, scores, aggregates, totalCostUsd, totalTokens, alerts }
}

/** Resolve a rule's metric to a numeric value, or null if unavailable. */
function resolveMetric(
  metric: string,
  aggregates: ScorerAggregate[],
  totalCostUsd: number,
  totalTokens: number,
): number | null {
  if (metric === 'cost' || metric === 'total_cost') return totalCostUsd
  if (metric === 'tokens' || metric === 'total_tokens') return totalTokens
  const agg = aggregates.find((a) => a.scorer === metric)
  return agg ? agg.mean : null
}
