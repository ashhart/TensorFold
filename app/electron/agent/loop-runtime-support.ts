// Loop Mode — pure helpers bridging the engine to real agent runs: build each
// role's system prompt from the contract, and derive a PhaseResult from the
// tool/assistant items a role's run produced. Kept pure for unit testing; the
// impure wiring (spawning runs) lives in the main-process loop runtime.

import type { LoopPhaseId, RunContract, PhaseResult } from './loop-types'

const READ_ONLY_ROLES = new Set<LoopPhaseId>(['product', 'architect'])

const ROLE_BRIEF: Record<LoopPhaseId, string> = {
  product: 'You are the PRODUCT role. Clarify requirements and acceptance criteria. Inspection-only: do not edit, modify, or write source.',
  architect: 'You are the ARCHITECT role. Design the approach and plan the implementation. Inspection-only (read-only): do not edit or write source.',
  dev: 'You are the DEV role. Implement the work to satisfy the goal and acceptance criteria. Edit/create files and run code as needed.',
  qa: 'You are the QA role. Write and actually RUN tests that exercise the acceptance criteria. Report real results, not intentions.',
  security: 'You are the SECURITY role. Review for vulnerabilities and unsafe patterns. Prefer read-only inspection and isolated temp-dir smoke checks; do not mutate source.',
  browser: 'You are the BROWSER role. Verify behavior through the real running app (UI/CLI/API). Capture concrete evidence (what you observed).',
  docs: 'You are the DOCS role. Verify README/docs/usage match the implementation. Inspection + runtime checks; flag stale docs rather than editing source.',
  review: 'You are the REVIEW role. Perform an evidence-driven code review against the contract. Flag issues with file references; do not fix them yourself.',
}

export function roleSystemPrompt(phase: LoopPhaseId, contract: RunContract): string {
  const parts: string[] = [ROLE_BRIEF[phase]]
  parts.push(`\nGOAL:\n${contract.goal}`)
  if (contract.acceptanceCriteria.length) parts.push(`\nACCEPTANCE CRITERIA:\n${contract.acceptanceCriteria.map((c) => `- ${c}`).join('\n')}`)
  if (contract.verificationCommands.length) parts.push(`\nVERIFICATION COMMANDS (these must pass):\n${contract.verificationCommands.map((c) => `- ${c}`).join('\n')}`)
  if (contract.forbiddenShortcuts.length) parts.push(`\nFORBIDDEN SHORTCUTS:\n${contract.forbiddenShortcuts.map((c) => `- ${c}`).join('\n')}`)
  if (READ_ONLY_ROLES.has(phase)) parts.push('\nYou are read-only this phase. Produce findings/plans as text; another role will implement.')
  return parts.join('\n')
}

interface TimelineLike { kind: string; text?: string; tool?: string; state?: string }

export function phaseResultFromTimeline(phase: LoopPhaseId, items: TimelineLike[]): PhaseResult {
  const attemptedToolItems = items.filter((i) => i.kind === 'tool')
  const toolItems = attemptedToolItems.filter((i) => i.state === 'ok')
  const lastAssistant = [...items].reverse().find((i) => i.kind === 'assistant' && (i.text ?? '').trim())
  const summary = (lastAssistant?.text ?? '').trim().slice(0, 500) || `${phase} produced no summary`
  const evidenceRefs = [...new Set(toolItems.map((i) => i.tool).filter((t): t is string => !!t))]
  return { phase, ok: toolItems.length > 0, toolEvents: toolItems.length, toolAttempts: attemptedToolItems.length, summary, evidenceRefs }
}
