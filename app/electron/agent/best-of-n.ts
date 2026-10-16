// Best-of-N attempts + verifier auto-select — the pure orchestration core.
//
// The substrate already exists: rooms/parallel seats give us worktree-isolated agent
// runs, and delegate-task.ts is a concurrency pool that fans work out and collects
// results in order. What was missing is the OUTER loop that turns "run one agent" into
// "run N independent attempts at the same task, judge each, and pick the winner".
//
// This module is that judge-and-pick core, and NOTHING else. It is deliberately pure /
// dependency-injected (same shape as delegate-task.ts's runOneChild, pr-publish.ts's
// runner, ssh-connection's clientFactory):
//
//   • deps.runAttempt   — produces an AttemptOutcome for one descriptor (the real impl
//                          spins a worktree + agent run; here it's injected & faked).
//   • deps.verify       — scores one settled attempt (the real impl runs the test suite
//                          and/or asks an LLM judge; here it's injected & faked).
//
// Given those two injected effects, this file owns the ORCHESTRATION that is easy to get
// subtly wrong and therefore worth unit-testing exhaustively: bounded concurrency, score
// composition (tests-passed gate + verifier vote), deterministic tie-breaking, and picking
// a well-defined winner even when some attempts crash or fail their tests.
//
// The host mission runtime supplies the heavy effects: one isolated Git worktree and agent
// loop per attempt, followed by stop-condition verification and winner integration. Keeping
// this core dependency-injected makes its concurrency and deterministic ranking independently
// testable without Electron, Git, or a live model provider.

/** One thing to try. `id` is caller-stable so the winner can be mapped back to real state. */
export interface AttemptDescriptor {
  id: string
  /** The prompt/instructions/variant for this attempt (e.g. a different temperature or approach). */
  spec: string
  /** Optional free-form metadata carried through untouched (model, temperature, seed, worktree…). */
  meta?: Record<string, unknown>
}

/** What deps.runAttempt returns for a single descriptor: whether it produced a candidate at all. */
export interface AttemptOutcome {
  /** false => the attempt crashed / never produced a usable candidate; it is scored as a loss. */
  ok: boolean
  /** The produced artifact/diff/summary. Empty when !ok. */
  output: string
  /** Populated when !ok. */
  error?: string
}

/** The verifier's judgement of one settled attempt. All fields optional-by-default & clamped. */
export interface VerifierVerdict {
  /**
   * Hard gate: did the attempt's tests pass? An attempt with testsPassed === false can never
   * beat one with testsPassed === true, regardless of vote (a green build outranks a pretty
   * red one). `undefined` => "no test signal available" and is treated as neutral, NOT a fail.
   */
  testsPassed?: boolean
  /** Soft signal: the verifier's confidence/quality vote, clamped to [0,1]. Default 0. */
  vote?: number
  /** Optional human-readable justification, carried through for surfacing to the user. */
  rationale?: string
}

export interface ScoredAttempt {
  descriptor: AttemptDescriptor
  outcome: AttemptOutcome
  verdict: VerifierVerdict
  /**
   * The composite score used for ranking. Higher is better. Composed so that:
   *   ok && testsPassed(true)   dominates
   *   ok && testsPassed(undef)  next
   *   ok && testsPassed(false)  next
   *   !ok (crashed)             always last
   * with the clamped vote breaking ties WITHIN a tier.
   */
  score: number
}

export interface BestOfNResult {
  /** All attempts, sorted best-first (deterministic). Empty only if `descriptors` was empty. */
  ranked: ScoredAttempt[]
  /** The chosen attempt (ranked[0]), or null when there were no descriptors. */
  winner: ScoredAttempt | null
}

export interface BestOfNDeps {
  /** Run one attempt to a settled outcome. The heavy, injected effect (worktree + agent run). */
  runAttempt: (descriptor: AttemptDescriptor, index: number) => Promise<AttemptOutcome>
  /** Score one settled attempt. The heavy, injected effect (test run + LLM judge). */
  verify: (descriptor: AttemptDescriptor, outcome: AttemptOutcome, index: number) => Promise<VerifierVerdict>
  /** At most this many attempts run (and are verified) concurrently. Default 3, min 1. */
  maxConcurrent?: number
  /** Cooperative cancellation: once aborted, no new attempts launch. Optional. */
  signal?: AbortSignal
}

const DEFAULT_MAX_CONCURRENT = 3

// Tier bands keep testsPassed STRICTLY dominant over the vote. The vote lives in [0,1], so we
// space the tiers by 2 (not 1): even a max vote of 1.0 in a lower tier can never reach the
// floor of the tier above it. Ordering: green > unknown > red > crashed.
const TIER_TESTS_PASSED = 6
const TIER_TESTS_UNKNOWN = 4
const TIER_TESTS_FAILED = 2
const TIER_CRASHED = 0

function clamp01(n: unknown): number {
  const v = typeof n === 'number' && Number.isFinite(n) ? n : 0
  if (v < 0) return 0
  if (v > 1) return 1
  return v
}

/**
 * Compose a single comparable score from an outcome + verdict. Pure and total.
 *
 * A crashed attempt (!outcome.ok) is always the lowest tier no matter what the verifier said —
 * you cannot ship a diff that never built. Above that floor, the test-pass signal picks the
 * tier and the clamped vote (in [0,1]) orders attempts within a tier without ever letting a
 * loser jump a tier.
 */
export function scoreAttempt(outcome: AttemptOutcome, verdict: VerifierVerdict): number {
  const vote = clamp01(verdict?.vote)
  if (!outcome.ok) return TIER_CRASHED + vote
  let tier: number
  if (verdict?.testsPassed === true) tier = TIER_TESTS_PASSED
  else if (verdict?.testsPassed === false) tier = TIER_TESTS_FAILED
  else tier = TIER_TESTS_UNKNOWN
  return tier + vote
}

/**
 * Rank already-scored attempts best-first with a fully deterministic order. Ties on score
 * (e.g. two green builds with equal vote) break by ORIGINAL descriptor order via the injected
 * index, so the result never depends on which attempt happened to settle first. Pure; does not
 * mutate its input.
 */
export function rankScored(scored: ScoredAttempt[], orderOf: (s: ScoredAttempt, i: number) => number): ScoredAttempt[] {
  return scored
    .map((s, i) => ({ s, i }))
    .sort((a, b) => {
      if (b.s.score !== a.s.score) return b.s.score - a.s.score
      return orderOf(a.s, a.i) - orderOf(b.s, b.i)
    })
    .map((x) => x.s)
}

/**
 * Run every descriptor through deps.runAttempt (bounded to maxConcurrent in flight), verify each
 * settled attempt through deps.verify, compose scores, and return the attempts ranked best-first
 * with a single deterministic winner.
 *
 * Guarantees:
 *  - Every descriptor appears exactly once in `ranked` (even if it crashed or was never launched
 *    due to abort — such attempts are scored as the lowest tier, so they can only ever win when
 *    they are the ONLY attempts).
 *  - A thrown runAttempt or verify is caught per-attempt and turned into a loss, never a rejection
 *    of the whole batch (one bad attempt can't sink the run).
 *  - Ordering is deterministic and independent of settle order: ties break by original index.
 *  - `winner === ranked[0]` (or null iff there were no descriptors).
 */
export async function runBestOfN(
  descriptors: AttemptDescriptor[],
  deps: BestOfNDeps,
): Promise<BestOfNResult> {
  if (descriptors.length === 0) return { ranked: [], winner: null }

  const maxConcurrent = Math.max(1, deps.maxConcurrent ?? DEFAULT_MAX_CONCURRENT)
  const aborted = () => deps.signal?.aborted === true

  const scored: (ScoredAttempt | undefined)[] = descriptors.map(() => undefined)

  const settleOne = async (index: number): Promise<void> => {
    const descriptor = descriptors[index]

    // If we were aborted before this attempt got a turn, record it as a crashed/unlaunched loss
    // rather than launching real work.
    if (aborted()) {
      const outcome: AttemptOutcome = { ok: false, output: '', error: 'Aborted before this attempt started.' }
      scored[index] = { descriptor, outcome, verdict: {}, score: scoreAttempt(outcome, {}) }
      return
    }

    let outcome: AttemptOutcome
    try {
      outcome = await deps.runAttempt(descriptor, index)
    } catch (e: any) {
      outcome = { ok: false, output: '', error: e?.message ?? String(e) }
    }

    let verdict: VerifierVerdict = {}
    // Only bother verifying an attempt that actually produced something; a crashed attempt is a
    // loss regardless of any (missing) test/vote signal.
    if (outcome.ok) {
      try {
        verdict = (await deps.verify(descriptor, outcome, index)) ?? {}
      } catch (e: any) {
        // A verifier that itself crashes leaves the attempt with no positive signal: neutral
        // verdict, but we surface the failure in the rationale so it isn't silently swallowed.
        verdict = { rationale: `verifier failed: ${e?.message ?? String(e)}` }
      }
    }

    scored[index] = { descriptor, outcome, verdict, score: scoreAttempt(outcome, verdict) }
  }

  // Same bounded-semaphore shape as delegate-task.ts: exactly maxConcurrent workers pulling from
  // a shared cursor, so we keep N in flight (not "batches of N awaited together").
  let cursor = 0
  const worker = async (): Promise<void> => {
    while (true) {
      const index = cursor
      if (index >= descriptors.length) return
      cursor++
      await settleOne(index)
    }
  }
  const workerCount = Math.min(maxConcurrent, descriptors.length)
  await Promise.all(Array.from({ length: workerCount }, () => worker()))

  const complete: ScoredAttempt[] = scored.map((s, i) => {
    if (s) return s
    // Defensive: every slot is filled by settleOne, but never hand back an undefined.
    const outcome: AttemptOutcome = { ok: false, output: '', error: 'Attempt never settled.' }
    return { descriptor: descriptors[i], outcome, verdict: {}, score: scoreAttempt(outcome, {}) }
  })

  // Ranking ties break by original index. `complete` is already in descriptor order, so the
  // array position IS the original index.
  const ranked = rankScored(complete, (_s, i) => i)
  return { ranked, winner: ranked[0] ?? null }
}
