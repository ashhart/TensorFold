// Loop Mode — the self-advancing engine state machine. Pure orchestration: all
// side effects (spawn a role, run verification, call the critic, checkpoint,
// rollback) are injected via LoopRuntime, so the control flow is fully
// unit-testable. The real LoopRuntime (spawn via runAgent/runRoom, real critic
// model call, git checkpoints, CDP real-surface QA) is wired in later phases.

import type { RunContract, LoopPhaseId, PhaseResult, Verdict, Blocker } from './loop-types'
import { classifyBlockers, hasUserBlocker } from './loop-blockers'
import { EVIDENCE_OPTIONAL_PHASES } from './loop-types'

export interface LoopRuntime {
  runPhase(phase: LoopPhaseId, contract: RunContract, cycle: number): Promise<PhaseResult>
  runVerification(contract: RunContract): Promise<string>
  critique(contract: RunContract, evidence: string): Promise<Verdict>
  checkpoint(cycle: number): Promise<string>
  rollback(): Promise<void>
  log(line: string): void
}

export type LoopStatus = 'shipped' | 'blocked' | 'capped' | 'failed'
export interface LoopResult {
  status: LoopStatus
  cycles: number
  checkpoints: number
  lastVerdict?: Verdict
  blockers: Blocker[]
}

const REGRESSION_RE = /regression|broke existing|made worse|revert|corrupted/i

export async function runLoop(contract: RunContract, phases: LoopPhaseId[], rt: LoopRuntime): Promise<LoopResult> {
  let cycles = 0
  let checkpoints = 0
  let lastVerdict: Verdict | undefined
  const implPhases = phases.filter((p) => p !== 'product' && p !== 'architect')

  for (let cycle = 0; cycle < contract.maxCycles; cycle++) {
    cycles++
    const stepPhases = cycle === 0 ? phases : implPhases
    const evidence: string[] = []
    const missingToolEvidence: string[] = []

    for (const phase of stepPhases) {
      let result = await rt.runPhase(phase, contract, cycle)
      // Tool-evidence guard: an implementation phase with no real tool activity
      // gets one stricter retry, then fails closed — no agent claims "done"
      // without doing anything. Planning roles (product/architect) are exempt:
      // a text-only plan is a valid result for them.
      // Guard on tool ATTEMPTS, not successes: a phase that ran tools which all failed
      // (a red build/test) did real work — let the critic judge it. Only a phase that
      // touched no tools at all is "no evidence" and gets the stricter retry / fail-closed.
      if (result.toolAttempts === 0 && !EVIDENCE_OPTIONAL_PHASES.includes(phase)) {
        rt.log(`${phase}: no tool evidence, stricter retry`)
        result = await rt.runPhase(phase, contract, cycle)
        if (result.toolAttempts === 0) {
          // A second text-only implementation response is evidence of a recoverable phase
          // failure, not a reason to crash the entire state machine. Carry it through
          // verification + critique as an explicit FIXABLE issue. This lets a later cycle
          // recover, while the forced failed verdict below prevents a critic from declaring
          // text-only implementation "shipped" without executable evidence.
          const issue = `${phase} produced text-only output twice without attempting a tool; implementation evidence is still missing.`
          missingToolEvidence.push(issue)
          evidence.push(`[${phase}] BLOCKED: ${issue} Last response: ${result.summary}`)
          rt.log(`${phase}: text-only retry reclassified as a fixable evidence blocker`)
          continue
        }
      }
      evidence.push(`[${phase}] ${result.summary}`)
    }

    const verifyOut = await rt.runVerification(contract)
    const criticVerdict = await rt.critique(contract, `${evidence.join('\n')}\n\nVERIFY:\n${verifyOut}`)
    const verdict: Verdict = missingToolEvidence.length
      ? {
          ...criticVerdict,
          passed: false,
          summary: criticVerdict.passed
            ? `Implementation evidence incomplete: ${missingToolEvidence.join(' ')}`
            : criticVerdict.summary,
          issues: [...criticVerdict.issues, ...missingToolEvidence],
        }
      : criticVerdict
    lastVerdict = verdict

    if (verdict.passed) {
      return { status: 'shipped', cycles, checkpoints, lastVerdict, blockers: [] }
    }

    const blockers = classifyBlockers(verdict.issues)
    if (hasUserBlocker(blockers)) {
      return { status: 'blocked', cycles, checkpoints, lastVerdict, blockers }
    }

    const regressed = REGRESSION_RE.test(verdict.summary) || verdict.issues.some((i) => REGRESSION_RE.test(i))
    if (regressed) {
      await rt.rollback()
    } else {
      await rt.checkpoint(cycle)
      checkpoints++
    }
  }

  return { status: 'capped', cycles, checkpoints, lastVerdict, blockers: [] }
}
