// The critic can reject completion; it cannot override observed verification failures.

import type { RunContract, Verdict } from './loop-types'
import { CRITIC_PASS_CONFIDENCE } from './loop-types'

export function buildCriticPrompt(contract: RunContract, evidence: string, transcript: string): { system: string; user: string } {
  const system = `You are an independent, skeptical verification auditor — the final quality gate of an autonomous build pipeline. You did NOT do the work; judge whether it is genuinely complete and correct, do not be agreeable.
Be strict and fail-closed: default to NOT passing unless the goal and acceptance criteria are clearly met AND tests were actually EXECUTED (not merely written) and passed.
Respond with ONLY a single JSON object: {"passed":boolean,"confidence":number,"summary":string,"issues":string[]}`
  const user = `GOAL:\n${contract.goal}\n\nACCEPTANCE CRITERIA:\n${contract.acceptanceCriteria.join('\n')}\n\nFORBIDDEN SHORTCUTS:\n${contract.forbiddenShortcuts.join('\n')}\n\nEVIDENCE:\n${evidence || '(none)'}\n\nRECENT TRANSCRIPT:\n${transcript || '(none)'}\n\nJudge whether this is fully and correctly complete. Fail closed if tests were not actually run, or anything is unclear.`
  return { system, user }
}

function invalidVerdict(): Verdict {
  return {
    passed: false, confidence: 0, summary: 'Could not parse verdict',
    issues: ['Critic returned an invalid or unparseable response'],
  }
}

function checkedVerdict(value: unknown): Verdict {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalidVerdict()
  const data = value as Partial<Verdict>
  if (typeof data.passed !== 'boolean' || typeof data.confidence !== 'number' ||
      !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1 ||
      typeof data.summary !== 'string' || !data.summary.trim() ||
      !Array.isArray(data.issues) || data.issues.some(issue => typeof issue !== 'string' || !issue.trim())) {
    return invalidVerdict()
  }
  const issues = data.issues.map(issue => issue.trim())
  const passed = data.passed && data.confidence >= CRITIC_PASS_CONFIDENCE && !issues.length
  return {
    passed, confidence: data.confidence, summary: data.summary,
    issues: passed ? [] : issues.length ? issues : ['Verdict did not meet the confidence threshold to pass'],
  }
}

export function gateVerdict(critic: Verdict, requiredIssues: readonly string[]): Verdict {
  const verdict = checkedVerdict(critic)
  if (!requiredIssues.length) return verdict
  return {
    ...verdict, passed: false,
    summary: `Required evidence incomplete: ${requiredIssues.join(' ')}`,
    issues: [...new Set([...verdict.issues, ...requiredIssues])],
  }
}

// Preserve compatible JSON-in-prose responses while validating the complete verdict shape.
export function parseVerdict(raw: string): Verdict {
  try {
    const match = raw.match(/\{[\s\S]*\}/)
    if (match) return checkedVerdict(JSON.parse(match[0]))
  } catch {}
  return invalidVerdict()
}
