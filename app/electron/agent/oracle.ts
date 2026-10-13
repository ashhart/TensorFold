// The Oracle tool: consult a deep-reasoning advisor on ONE hard subproblem while the main
// loop keeps driving. Adapted from Sourcegraph Amp's "Oracle" / Aider's architect split — the
// idea is a deliberate, high-reasoning-effort consult on a tough bug or design decision,
// isolated from the working context, returning a focused recommendation (not a survey).
//
// This module holds only the PURE, unit-testable parts (arg parsing + the consultant system
// prompt). The actual LLM-driving runOracle() lives in loop.ts next to runExplore (it needs
// the same streamChat/executeTool plumbing), so this file stays dependency-free.

export interface OracleQuery {
  question: string
  // Optional caller-supplied grounding (code snippets, constraints) folded into the user turn.
  context?: string
}

/**
 * Parse the oracle tool call's raw args into an OracleQuery, or null when there is no question
 * to ask. Accepts `question` (canonical) or `query` (alias the model sometimes emits). Pure.
 */
export function parseOracleArgs(args: Record<string, any>): OracleQuery | null {
  const question = String(args?.question ?? args?.query ?? '').trim()
  if (!question) return null
  const context = typeof args?.context === 'string' && args.context.trim() ? args.context.trim() : undefined
  return { question, ...(context ? { context } : {}) }
}

/** The user-turn text: the question, with any caller-supplied context appended. Pure. */
export function buildOracleUserMessage(query: OracleQuery): string {
  return query.context ? `${query.question}\n\nRelevant context:\n${query.context}` : query.question
}

/**
 * System prompt for the Oracle consult: a decisive senior engineer with read-only grounding,
 * asked for a concrete recommendation rather than a hedged list of options. Pure.
 */
export function buildOracleSystemPrompt(cwd: string): string {
  return [
    'You are the Oracle: a senior staff engineer consulted on ONE hard, isolated subproblem',
    `(a subtle bug, a tricky design/architecture decision, a confusing trade-off). Workspace: ${cwd}.`,
    'You have READ-ONLY access (list_dir/read_file/search/glob) to ground yourself in the real code —',
    'use it before answering. Reason carefully, then end with a CONCRETE, actionable recommendation:',
    'the decision or fix to make, the key trade-offs, and the main risk. Be decisive and specific;',
    'do not hedge with a survey of options. You cannot edit files or run commands.',
  ].join(' ')
}
