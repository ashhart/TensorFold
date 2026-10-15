import { SECRET_MASK } from './security'
import { redactSecretValues } from './secret-detection'
import type { ExportRedaction, ExportRedactionKind } from './types'

// Value-level secret + PII scanning for exports. The rest of the codebase redacts
// by KEY name (connector-spine `safeKeys`) or blocks credential FILES
// (command-policy). This scans free-text VALUES — prompts, tool output, diffs —
// that may carry inlined secrets before a run ever leaves the machine.
interface Rule { kind: ExportRedactionKind; re: RegExp; mask?: (m: string) => string }

// Secret values are handled first by the shared scanner, then PII/anonymization rules run here.
const RULES: Rule[] = [
  { kind: 'email', re: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, mask: () => '<redacted-email>' },
  { kind: 'home_path', re: /(\/(?:Users|home)\/)[^/\s'":]+/g, mask: (m) => m.replace(/((?:Users|home)\/)[^/\s'":]+/, '$1<user>') },
]

// Only applied when anonymizing a share/eval packet — IPs are environment-revealing
// but legitimately appear in normal run output, so they survive a plain redaction.
const ANON_RULES: Rule[] = [
  { kind: 'ip', re: /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/g, mask: () => '<ip>' },
]

export interface RedactResult { text: string; hits: ExportRedactionKind[] }

export function redactText(input: string | undefined, anonymize = false): RedactResult {
  const hits: ExportRedactionKind[] = []
  if (!input) return { text: input ?? '', hits }
  const secretResult = redactSecretValues(input, SECRET_MASK)
  hits.push(...secretResult.hits)
  let text = secretResult.text
  const rules = anonymize ? [...RULES, ...ANON_RULES] : RULES
  for (const rule of rules) {
    text = text.replace(rule.re, (m) => { hits.push(rule.kind); return rule.mask ? rule.mask(m) : SECRET_MASK })
  }
  return { text, hits }
}

export function aggregate(hits: ExportRedactionKind[]): ExportRedaction[] {
  const counts = new Map<ExportRedactionKind, number>()
  for (const h of hits) counts.set(h, (counts.get(h) ?? 0) + 1)
  return [...counts.entries()]
    .map(([kind, count]) => ({ kind, count }))
    .sort((a, b) => b.count - a.count)
}
