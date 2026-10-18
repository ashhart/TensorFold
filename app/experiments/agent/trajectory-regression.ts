// Golden trajectory regression tests — pure capture + comparison.
//
// A run's *trajectory* is the ordered path of tool selections the agent took to
// resolve a task, plus the step count. Two runs of the same task on the same input
// should follow (roughly) the same path; when they diverge — "used to take 4 steps,
// now takes 12", or "it used to pick `read_file` here and now picks `search`" — that
// is a behavioural regression a plain output-diff test won't catch.
//
// This module is the substrate for golden-trajectory regression tests. It sits on top
// of the orchestration/thread journal records (see orchestration-journal.ts): those
// hold what happened durably on disk; this distils the tool-selection path out of a
// run, freezes it as a *golden* record, and compares a later *candidate* run against
// that golden to flag regressions with a severity + human-readable reason.
//
// Pure + dependency-injected: capture takes an already-materialised list of steps (the
// caller projects them out of whatever event source it has — journal events, AgentEvent
// stream, room turns); nothing here touches fs, the clock, or the model. The optional
// `now` used to stamp a golden's captured time is injected so tests are deterministic.

// ---------------------------------------------------------------------------
// Trajectory model
// ---------------------------------------------------------------------------

/** One tool selection in a run, in execution order. Extra fields are ignored by the
 *  comparator — only `tool` (and optionally `phase`) shape the path. */
export interface TrajectoryStep {
  /** Name of the tool selected at this step, e.g. 'read_file', 'search', 'apply_patch'. */
  tool: string
  /** Optional coarse phase label ('explore' | 'edit' | 'verify' | …) if the caller tracks one. */
  phase?: string
  /** Optional free-form note (arg summary, target file). Never affects comparison; for debugging. */
  note?: string
}

/** A captured trajectory: the ordered tool path plus derived metrics. */
export interface Trajectory {
  /** Stable identifier of the task/scenario this trajectory belongs to. */
  taskId: string
  /** The ordered steps as captured. */
  steps: TrajectoryStep[]
  /** Number of steps (== steps.length). Precomputed so callers can compare cheaply. */
  stepCount: number
  /** The ordered tool-name path (steps.map(s => s.tool)). */
  toolPath: string[]
  /** Distinct tool names used, sorted, for cheap set-level comparison. */
  toolSet: string[]
}

/** A frozen golden trajectory: a captured trajectory plus provenance metadata. */
export interface GoldenTrajectory extends Trajectory {
  /** Wall-clock ms this golden was captured (injected via captureGolden's `now`). */
  capturedAt: number
  /** Optional label — a commit sha, model id, or run name — for provenance in reports. */
  label?: string
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/** Coerce a loosely-typed step (e.g. projected from a journal payload) into a
 *  TrajectoryStep, or null if it carries no usable tool name. */
function normaliseStep(raw: unknown): TrajectoryStep | null {
  if (!raw || typeof raw !== 'object') return null
  const obj = raw as Record<string, unknown>
  const tool = obj.tool
  if (typeof tool !== 'string' || !tool.trim()) return null
  const step: TrajectoryStep = { tool: tool.trim() }
  if (typeof obj.phase === 'string' && obj.phase.trim()) step.phase = obj.phase.trim()
  if (typeof obj.note === 'string' && obj.note) step.note = obj.note
  return step
}

/**
 * Capture a trajectory from an ordered list of steps. Steps that carry no tool name
 * are dropped (a run's tool path is the tool selections, not every event). Pure.
 */
export function captureTrajectory(taskId: string, rawSteps: readonly unknown[]): Trajectory {
  if (!taskId || !taskId.trim()) throw new Error('captureTrajectory: taskId is required')
  const steps: TrajectoryStep[] = []
  for (const raw of rawSteps) {
    const step = normaliseStep(raw)
    if (step) steps.push(step)
  }
  const toolPath = steps.map((s) => s.tool)
  const toolSet = [...new Set(toolPath)].sort()
  return { taskId: taskId.trim(), steps, stepCount: steps.length, toolPath, toolSet }
}

/** Freeze a captured trajectory into a golden record, stamping capture time + label. */
export function freezeGolden(trajectory: Trajectory, opts: { now: number; label?: string }): GoldenTrajectory {
  const golden: GoldenTrajectory = {
    ...trajectory,
    steps: trajectory.steps.map((s) => ({ ...s })),
    toolPath: [...trajectory.toolPath],
    toolSet: [...trajectory.toolSet],
    capturedAt: opts.now,
  }
  if (opts.label) golden.label = opts.label
  return golden
}

/** Convenience: capture + freeze in one call. */
export function captureGolden(
  taskId: string,
  rawSteps: readonly unknown[],
  opts: { now: number; label?: string },
): GoldenTrajectory {
  return freezeGolden(captureTrajectory(taskId, rawSteps), opts)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

export type RegressionKind =
  /** Candidate took materially more steps than the golden (path inflation / looping). */
  | 'step_inflation'
  /** Candidate uses a tool the golden never used, or dropped one the golden relied on. */
  | 'tool_set_changed'
  /** The ordered tool path diverges at some index — a different tool chosen at a step. */
  | 'path_divergence'
  /** Candidate is for a different taskId than the golden — comparison is meaningless. */
  | 'task_mismatch'

export type RegressionSeverity = 'info' | 'warning' | 'regression'

export interface RegressionFinding {
  kind: RegressionKind
  severity: RegressionSeverity
  /** Human-readable summary, e.g. "used to take 4 steps, now 12 (+200%)". */
  message: string
  /** For path_divergence: the 0-based step index where the paths first differ. */
  atIndex?: number
  /** Tool expected (golden) at the divergence / dropped tools. */
  expected?: string | string[]
  /** Tool actually seen (candidate) at the divergence / newly-introduced tools. */
  actual?: string | string[]
}

export interface CompareOptions {
  /**
   * Fractional step-count growth over golden that counts as a regression.
   * 0.5 => a candidate with >50% more steps than the golden is a regression.
   * Growth above `warnStepGrowth` but at/below this is a warning. Default 1.0 (2x).
   */
  regressionStepGrowth?: number
  /** Fractional growth that counts as a warning. Default 0.25 (25% more steps). */
  warnStepGrowth?: number
  /**
   * If true, path divergence is a 'regression'; if false, a 'warning' (the tool set /
   * step count is often the sharper signal, so callers can down-rank raw order). Default true.
   */
  pathDivergenceIsRegression?: boolean
}

export interface ComparisonResult {
  taskId: string
  /** True if any finding has severity 'regression'. The headline pass/fail. */
  regressed: boolean
  /** All findings, most-severe first. Empty when candidate matches golden closely. */
  findings: RegressionFinding[]
  /** golden.stepCount for at-a-glance reporting. */
  goldenSteps: number
  /** candidate.stepCount for at-a-glance reporting. */
  candidateSteps: number
}

const SEVERITY_RANK: Record<RegressionSeverity, number> = { regression: 0, warning: 1, info: 2 }

/** First index where two tool paths differ, or -1 if one is a prefix of the other. */
function firstDivergenceIndex(a: readonly string[], b: readonly string[]): number {
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) return i
  }
  return -1
}

function pct(growth: number): string {
  return `${growth >= 0 ? '+' : ''}${Math.round(growth * 100)}%`
}

/**
 * Compare a candidate trajectory against a golden and flag regressions.
 *
 * Signals, in the order they're evaluated:
 *  - task_mismatch (regression): different taskId — the comparison is moot.
 *  - step_inflation: candidate step count grew past the warn/regression thresholds.
 *  - tool_set_changed: candidate introduced a new tool or dropped one the golden used.
 *  - path_divergence: the ordered tool path first differs at some step index.
 *
 * Pure — no I/O, deterministic for the same inputs.
 */
export function compareTrajectory(
  golden: GoldenTrajectory,
  candidate: Trajectory,
  options: CompareOptions = {},
): ComparisonResult {
  const regressionStepGrowth = options.regressionStepGrowth ?? 1.0
  const warnStepGrowth = options.warnStepGrowth ?? 0.25
  const pathIsRegression = options.pathDivergenceIsRegression ?? true

  const findings: RegressionFinding[] = []

  // --- task mismatch: still report the other findings, but headline this loudly ---
  if (golden.taskId !== candidate.taskId) {
    findings.push({
      kind: 'task_mismatch',
      severity: 'regression',
      message: `candidate taskId ${JSON.stringify(candidate.taskId)} does not match golden ${JSON.stringify(golden.taskId)}`,
      expected: golden.taskId,
      actual: candidate.taskId,
    })
  }

  // --- step inflation ---
  const goldenSteps = golden.stepCount
  const candidateSteps = candidate.stepCount
  if (goldenSteps > 0) {
    const growth = (candidateSteps - goldenSteps) / goldenSteps
    if (growth > regressionStepGrowth) {
      findings.push({
        kind: 'step_inflation',
        severity: 'regression',
        message: `used to take ${goldenSteps} steps, now ${candidateSteps} (${pct(growth)})`,
      })
    } else if (growth > warnStepGrowth) {
      findings.push({
        kind: 'step_inflation',
        severity: 'warning',
        message: `step count grew from ${goldenSteps} to ${candidateSteps} (${pct(growth)})`,
      })
    }
  } else if (candidateSteps > 0) {
    // golden had no steps but candidate does — treat as a warning (new behaviour).
    findings.push({
      kind: 'step_inflation',
      severity: 'warning',
      message: `golden had 0 steps, candidate now takes ${candidateSteps}`,
    })
  }

  // --- tool set changed (introduced / dropped tools) ---
  const goldenSet = new Set(golden.toolSet)
  const candidateSet = new Set(candidate.toolSet)
  const introduced = candidate.toolSet.filter((t) => !goldenSet.has(t))
  const dropped = golden.toolSet.filter((t) => !candidateSet.has(t))
  if (introduced.length || dropped.length) {
    const parts: string[] = []
    if (introduced.length) parts.push(`now uses ${introduced.join(', ')}`)
    if (dropped.length) parts.push(`no longer uses ${dropped.join(', ')}`)
    findings.push({
      kind: 'tool_set_changed',
      // A newly-introduced tool (wrong-tool pick) is the sharper regression signal;
      // only dropping a tool (and using fewer) is a warning.
      severity: introduced.length ? 'regression' : 'warning',
      message: `tool set changed: ${parts.join('; ')}`,
      ...(introduced.length ? { actual: introduced } : {}),
      ...(dropped.length ? { expected: dropped } : {}),
    })
  }

  // --- ordered path divergence ---
  const divergeAt = firstDivergenceIndex(golden.toolPath, candidate.toolPath)
  if (divergeAt >= 0) {
    findings.push({
      kind: 'path_divergence',
      severity: pathIsRegression ? 'regression' : 'warning',
      message: `tool path diverges at step ${divergeAt}: expected ${golden.toolPath[divergeAt]}, got ${candidate.toolPath[divergeAt]}`,
      atIndex: divergeAt,
      expected: golden.toolPath[divergeAt],
      actual: candidate.toolPath[divergeAt],
    })
  }

  findings.sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity])
  const regressed = findings.some((f) => f.severity === 'regression')
  return { taskId: golden.taskId, regressed, findings, goldenSteps, candidateSteps }
}

/** True when a comparison shows no regression-level findings. Convenience for assertions. */
export function isTrajectoryStable(result: ComparisonResult): boolean {
  return !result.regressed
}

/** One-line human summary of a comparison, e.g. for a CI log line. */
export function formatComparison(result: ComparisonResult): string {
  const head = `[trajectory:${result.taskId}] ${result.regressed ? 'REGRESSED' : 'ok'} (${result.candidateSteps} vs ${result.goldenSteps} steps)`
  if (!result.findings.length) return head
  const lines = result.findings.map((f) => `  - ${f.severity}: ${f.message}`)
  return [head, ...lines].join('\n')
}
