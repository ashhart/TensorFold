// Loop Mode — classifies critic/verifier issues so the engine knows when it can
// auto-retry vs. when it must pause for a human. Only non-FIXABLE blockers stop
// the loop. Pure; first-match wins.

import type { Blocker, BlockerKind } from './loop-types'

const RULES: Array<{ re: RegExp; kind: Exclude<BlockerKind, 'FIXABLE_BY_AGENT'>; action: string }> = [
  { re: /(secret|api[ _-]?key|access[ _-]?key|auth[ _-]?token|\btokens?\b|\bpasswords?\b|\bcredentials?\b|env var|\.env|_KEY\b)/i, kind: 'NEEDS_USER_SECRET', action: 'Ask the user to provide the secret.' },
  { re: /\b(should|which|prefer|decide|decision|product)\b[^?]*\?/i, kind: 'NEEDS_PRODUCT_DECISION', action: 'Ask the user to make the product decision.' },
  { re: /\b(network|registry|timeout|503|502|ECONN|offline|outage|unreachable)\b/i, kind: 'EXTERNAL_SERVICE_DOWN', action: 'Retry later or ask the user.' },
  { re: /(rm\s+-rf|\bsudo\b|destructive|irreversible|drop\s+database|force\s+push|\bwipe\b)/i, kind: 'UNSAFE_ACTION', action: 'Requires explicit user authorization.' },
]

export function classifyBlockers(issues: string[]): Blocker[] {
  return issues.filter(Boolean).map((issue) => {
    for (const r of RULES) if (r.re.test(issue)) return { kind: r.kind, issue, action: r.action }
    return { kind: 'FIXABLE_BY_AGENT', issue, action: 'Retry with a fix task.' }
  })
}

export function hasUserBlocker(blockers: Blocker[]): boolean {
  return blockers.some((b) => b.kind !== 'FIXABLE_BY_AGENT')
}
