// Loop Mode — the independent verification critic. Builds the skeptical
// auditor prompt and parses the verdict FAIL-CLOSED. The model call itself is
// performed by the engine (so this module stays pure + unit-testable).

import type { RunContract, Verdict } from './loop-types'
import { CRITIC_PASS_CONFIDENCE } from './loop-types'

export function buildCriticPrompt(contract: RunContract, evidence: string, transcript: string): { system: string; user: string } {
  const system = `You are an independent, skeptical verification auditor — the final quality gate of an autonomous build pipeline. You did NOT do the work; judge whether it is genuinely complete and correct, do not be agreeable.
Be strict and fail-closed: default to NOT passing unless the goal and acceptance criteria are clearly met AND tests were actually EXECUTED (not merely written) and passed.
Respond with ONLY a single JSON object: {"passed":boolean,"confidence":number,"summary":string,"issues":string[]}`
  const user = `GOAL:\n${contract.goal}\n\nACCEPTANCE CRITERIA:\n${contract.acceptanceCriteria.join('\n')}\n\nFORBIDDEN SHORTCUTS:\n${contract.forbiddenShortcuts.join('\n')}\n\nEVIDENCE:\n${evidence || '(none)'}\n\nRECENT TRANSCRIPT:\n${transcript || '(none)'}\n\nJudge whether this is fully and correctly complete. Fail closed if tests were not actually run, or anything is unclear.`
  return { system, user }
}

/** Parse the critic's JSON; fail-closed on anything ambiguous. */
export function parseVerdict(raw: string): Verdict {
  try {
    const match = raw.match(/\{[\s\S]*\}/)
    if (match) {
      const data = JSON.parse(match[0]) as Partial<Verdict>
      const confidence = typeof data.confidence === 'number' ? data.confidence : 0
      const issues = Array.isArray(data.issues) ? data.issues.map(String) : []
      const passed = data.passed === true && confidence >= CRITIC_PASS_CONFIDENCE
      return {
        passed,
        confidence,
        summary: typeof data.summary === 'string' ? data.summary : passed ? 'Verified' : 'Not yet complete',
        issues: passed ? [] : issues.length ? issues : ['Verdict did not meet the confidence threshold to pass'],
      }
    }
  } catch {
    // fall through to fail-closed
  }
  return { passed: false, confidence: 0, summary: 'Could not parse verdict', issues: ['Critic returned an unparseable response'] }
}
