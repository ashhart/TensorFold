// Observed execution results and the independent critic must both permit completion.

import type { RunContract, LoopPhaseId, PhaseResult, Verdict, Blocker, VerificationReport } from './loop-types'
import { classifyBlockers, hasUserBlocker } from './loop-blockers'
import { EVIDENCE_OPTIONAL_PHASES } from './loop-types'
import { gateVerdict } from './loop-critic'
import { verificationConfigurationIssue, verificationEvidence, verificationIssues } from './loop-verification'

export interface LoopRuntime {
  runPhase(phase: LoopPhaseId, contract: RunContract, cycle: number): Promise<PhaseResult>
  runVerification(contract: RunContract): Promise<VerificationReport>
  critique(contract: RunContract, evidence: string): Promise<Verdict>
  checkpoint(cycle: number): Promise<string>
  rollback(): Promise<void>
  isCancelled?(): boolean
  log(line: string): void
}

export type LoopStatus = 'shipped' | 'blocked' | 'capped' | 'failed' | 'stopped'
export interface LoopResult {
  status: LoopStatus
  cycles: number
  checkpoints: number
  lastVerdict?: Verdict
  blockers: Blocker[]
}

const REGRESSION_RE = /regression|broke existing|made worse|revert|corrupted/i

export function loopConfigurationBlocker(contract: RunContract): LoopResult | null {
  const issue = verificationConfigurationIssue(contract)
  if (!issue) return null
  return {
    status: 'blocked', cycles: 0, checkpoints: 0,
    lastVerdict: { passed: false, confidence: 0, summary: issue, issues: [issue] },
    blockers: [{ kind: 'NEEDS_PRODUCT_DECISION', issue, action: 'Add the required verification commands and start again.' }],
  }
}

export async function runLoop(contract: RunContract, phases: LoopPhaseId[], rt: LoopRuntime): Promise<LoopResult> {
  const configuration = loopConfigurationBlocker(contract)
  if (configuration) return configuration
  const requiredCommands = [...contract.verificationCommands]
  let cycles = 0
  let checkpoints = 0
  let lastVerdict: Verdict | undefined
  const implPhases = phases.filter((p) => p !== 'product' && p !== 'architect')
  const stopped = (): LoopResult => ({
    status: 'stopped', cycles, checkpoints, blockers: [],
    lastVerdict: { passed: false, confidence: 0, summary: 'Loop stopped before completion.', issues: ['Loop was cancelled.'] },
  })

  for (let cycle = 0; cycle < contract.maxCycles; cycle++) {
    if (rt.isCancelled?.()) return stopped()
    cycles++
    const stepPhases = cycle === 0 ? phases : implPhases
    const evidence: string[] = []
    const missingToolEvidence: string[] = []

    for (const phase of stepPhases) {
      if (rt.isCancelled?.()) return stopped()
      let result = await rt.runPhase(phase, contract, cycle)
      if (rt.isCancelled?.()) return stopped()
      // Text-only implementation gets one retry; failed operations retain their evidence.
      if (result.toolAttempts === 0 && !EVIDENCE_OPTIONAL_PHASES.includes(phase)) {
        rt.log(`${phase}: no tool evidence, stricter retry`)
        result = await rt.runPhase(phase, contract, cycle)
        if (rt.isCancelled?.()) return stopped()
        if (result.toolAttempts === 0) {
          const issue = `${phase} produced text-only output twice without attempting a tool; implementation evidence is still missing.`
          missingToolEvidence.push(issue)
          evidence.push(`[${phase}] BLOCKED: ${issue} Last response: ${result.summary}`)
          rt.log(`${phase}: text-only retry reclassified as a fixable evidence blocker`)
          continue
        }
      }
      if (!EVIDENCE_OPTIONAL_PHASES.includes(phase) && (!result.ok || !(result.toolEvents > 0))) {
        missingToolEvidence.push(`${phase} has no successful implementation evidence; its operations must succeed.`)
      }
      evidence.push(`[${phase}] ${result.summary}`)
    }

    if (rt.isCancelled?.()) return stopped()
    let verification: VerificationReport
    const executionIssues: string[] = []
    try {
      verification = await rt.runVerification({ ...contract, verificationCommands: [...requiredCommands] })
    } catch (error) {
      verification = { checks: [], cancelled: error instanceof Error && error.name === 'AbortError' }
      executionIssues.push(`Verification could not complete: ${error instanceof Error ? error.message : String(error)}`)
    }
    if (rt.isCancelled?.() || verification?.cancelled) return stopped()
    const requiredIssues = [...missingToolEvidence, ...executionIssues, ...verificationIssues(requiredCommands, verification)]
    const criticVerdict = await rt.critique(contract, `${evidence.join('\n')}\n\nVERIFY:\n${verificationEvidence(verification)}`)
    if (rt.isCancelled?.()) return stopped()
    const verdict = gateVerdict(criticVerdict, requiredIssues)
    lastVerdict = verdict
    rt.log(verdict.passed ? 'Verified: all required commands passed.' : `Verification incomplete: ${verdict.summary}`)

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
