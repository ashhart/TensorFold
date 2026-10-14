import { computeDiff } from './diff'
import type { DiffLine } from './types'

// Line-anchored diff review with user-defined checks.
//
// Given a multi-file diff and a set of user check rules (regex / markdown
// rubric / severity), produce per-file line-anchored findings ranked by
// severity. Pure + dependency-free: callers pass the before/after text (or a
// prebuilt DiffLine[]) and the rule set; no I/O, no globals, no clock.
//
// This is the *reviewer* layer that sits on top of diff.ts (which produces the
// hunks) — the semgrep-compatible scanner in code-intelligence.ts scans whole
// files, whereas this reviews only what a change touches and lets the user
// define the checks.

export type Severity = 'info' | 'warning' | 'error' | 'critical'

// Where a rule is allowed to fire relative to the diff.
//   'added'   — only on lines the change introduces (default; the common case)
//   'removed' — only on lines the change deletes
//   'context' — only on unchanged surrounding lines
//   'changed' — added OR removed (anything the diff touches)
//   'any'     — added, removed, or context
export type RuleScope = 'added' | 'removed' | 'context' | 'changed' | 'any'

export interface CheckRule {
  id: string
  // Human-readable message. May contain `$0`, `$1`… placeholders that are
  // substituted with the regex match / capture groups on the matched line.
  message: string
  severity: Severity
  // The matcher. Either a source string compiled to a RegExp, or a RegExp.
  // String patterns are compiled case-insensitively unless `flags` overrides.
  pattern: string | RegExp
  flags?: string
  scope?: RuleScope
  // Optional: only apply to files whose path matches this (string = substring,
  // RegExp = test). Omitted → applies to every file.
  pathInclude?: string | RegExp
  // Optional: skip files whose path matches this.
  pathExclude?: string | RegExp
  // A finding whose severity is at or above the run's blocking threshold is
  // marked `blocking`. Individual rules can force blocking regardless.
  forceBlocking?: boolean
}

export interface Finding {
  ruleId: string
  severity: Severity
  message: string
  path: string
  // 1-based line number in the NEW file for added/context lines, or the OLD
  // file for removed lines. Undefined only for degenerate lines (shouldn't
  // happen for real diff rows).
  line?: number
  side: 'added' | 'removed' | 'context'
  excerpt: string
  match: string
  blocking: boolean
}

export interface ReviewFile {
  path: string
  // Provide EITHER before/after (the reviewer diffs them) …
  before?: string
  after?: string
  // … OR a pre-computed diff (e.g. from computeDiff already run upstream).
  diff?: DiffLine[]
}

export interface ReviewOptions {
  // Findings at or above this severity are flagged `blocking`. Default 'error'.
  blockingThreshold?: Severity
  // Context lines when diffing before/after. Default 3.
  context?: number
  // Cap total findings (highest severity kept first). Default 500.
  maxFindings?: number
}

export interface ReviewResult {
  findings: Finding[]
  // Per-file counts, in the order files were supplied.
  files: Array<{ path: string; findings: number }>
  // Tally by severity across all findings.
  counts: Record<Severity, number>
  // True if any surviving finding is blocking.
  blocked: boolean
  rulesApplied: number
}

const SEVERITY_ORDER: Record<Severity, number> = {
  info: 0,
  warning: 1,
  error: 2,
  critical: 3,
}

const DEFAULT_MAX_FINDINGS = 500

// --- markdown rubric → CheckRule[] ---------------------------------------
//
// Users often express checks as a markdown rubric rather than raw rule
// objects. We support a compact, forgiving line format:
//
//   - [error] no-console: forbid raw console.* in shipped code /console\.\w+/
//   - [warning] todo-left: leftover TODO marker `TODO`
//   - critical  secret-key  possible hardcoded secret  /api[_-]?key/i
//
// Grammar (whitespace-tolerant, leading `-`/`*` bullet optional):
//   [<severity>]  <id>  :?  <message...>  <pattern>
// where <pattern> is the LAST token and is either /re/flags or `literal`
// (a backtick-quoted literal is matched as a plain substring, escaped).
// Lines that don't parse are ignored (so prose in the rubric is harmless).

const SEVERITY_WORDS = new Set(['info', 'warning', 'warn', 'error', 'critical', 'crit'])

function normSeverity(word: string): Severity {
  const w = word.toLowerCase()
  if (w === 'warn') return 'warning'
  if (w === 'crit') return 'critical'
  if (w === 'info' || w === 'warning' || w === 'error' || w === 'critical') return w
  return 'warning'
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// Extract a trailing pattern token: /re/flags or `literal`. Returns the
// pattern spec plus the remaining text with the pattern removed.
function extractTrailingPattern(text: string): { pattern: string | RegExp; flags?: string; rest: string } | null {
  const trimmed = text.trimEnd()
  // Backtick literal: `...`
  const backtick = trimmed.match(/`([^`]+)`\s*$/)
  if (backtick) {
    return { pattern: escapeRegExp(backtick[1]), flags: 'i', rest: trimmed.slice(0, backtick.index).trimEnd() }
  }
  // /re/flags — a trailing /.../<flags> allowing escaped slashes inside.
  const re = trimmed.match(/\/((?:\\.|[^/\\])+)\/([a-z]*)\s*$/)
  if (re) {
    return { pattern: re[1], flags: re[2] || 'i', rest: trimmed.slice(0, re.index).trimEnd() }
  }
  return null
}

export function parseRubric(markdown: string): CheckRule[] {
  const rules: CheckRule[] = []
  const seen = new Set<string>()
  const lines = markdown.split(/\r?\n/)
  for (const raw of lines) {
    let line = raw.trim()
    if (!line) continue
    // Strip a leading bullet marker.
    line = line.replace(/^[-*]\s+/, '')

    // Severity: either [sev] or a leading bare word.
    let severity: Severity | null = null
    const bracket = line.match(/^\[\s*([a-zA-Z]+)\s*\]\s*/)
    if (bracket) {
      severity = normSeverity(bracket[1])
      line = line.slice(bracket[0].length)
    } else {
      const firstWord = line.match(/^(\w+)\s+/)
      if (firstWord && SEVERITY_WORDS.has(firstWord[1].toLowerCase())) {
        severity = normSeverity(firstWord[1])
        line = line.slice(firstWord[0].length)
      }
    }
    if (!severity) continue

    const patternPart = extractTrailingPattern(line)
    if (!patternPart) continue
    let body = patternPart.rest

    // id: an optional `identifier :` or `identifier` before the message.
    let id = ''
    const idColon = body.match(/^([\w.-]+)\s*:\s*/)
    if (idColon) {
      id = idColon[1]
      body = body.slice(idColon[0].length)
    } else {
      const idBare = body.match(/^([\w.-]+)\s+/)
      if (idBare) {
        id = idBare[1]
        body = body.slice(idBare[0].length)
      }
    }
    const message = body.trim() || id || 'check failed'
    if (!id) {
      id = `rule-${rules.length + 1}`
    }
    // De-dupe ids so downstream keys stay unique.
    let uid = id
    let n = 2
    while (seen.has(uid)) uid = `${id}-${n++}`
    seen.add(uid)

    rules.push({ id: uid, message, severity, pattern: patternPart.pattern, flags: patternPart.flags })
  }
  return rules
}

// --- rule compilation ----------------------------------------------------

interface CompiledRule {
  rule: CheckRule
  re: RegExp
  scope: RuleScope
}

function compileRule(rule: CheckRule): CompiledRule {
  const flags = rule.flags ?? (typeof rule.pattern === 'string' ? 'i' : rule.pattern.flags)
  // Ensure global so we can iterate multiple matches per line without infinite
  // loops (we manage lastIndex ourselves).
  const src = typeof rule.pattern === 'string' ? rule.pattern : rule.pattern.source
  const withG = flags.includes('g') ? flags : flags + 'g'
  const re = new RegExp(src, withG)
  return { rule, re, scope: rule.scope ?? 'added' }
}

function pathMatches(matcher: string | RegExp | undefined, path: string): boolean {
  if (matcher == null) return true
  if (typeof matcher === 'string') return path.includes(matcher)
  return matcher.test(path)
}

function scopeAllows(scope: RuleScope, side: 'added' | 'removed' | 'context'): boolean {
  switch (scope) {
    case 'added':
      return side === 'added'
    case 'removed':
      return side === 'removed'
    case 'context':
      return side === 'context'
    case 'changed':
      return side === 'added' || side === 'removed'
    case 'any':
      return true
  }
}

function sideOf(type: DiffLine['type']): 'added' | 'removed' | 'context' | null {
  if (type === 'add') return 'added'
  if (type === 'del') return 'removed'
  if (type === 'ctx') return 'context'
  return null // 'hunk'
}

function fillMessage(template: string, match: RegExpExecArray): string {
  return template.replace(/\$(\d+)/g, (_, d: string) => {
    const idx = Number(d)
    const val = match[idx]
    return val == null ? '' : val
  })
}

// --- core reviewer -------------------------------------------------------

export function reviewDiff(files: ReviewFile[], rules: CheckRule[], options: ReviewOptions = {}): ReviewResult {
  const blockingThreshold = options.blockingThreshold ?? 'error'
  const context = options.context ?? 3
  const maxFindings = options.maxFindings ?? DEFAULT_MAX_FINDINGS
  const threshold = SEVERITY_ORDER[blockingThreshold]

  const compiled = rules.map(compileRule)
  const findings: Finding[] = []
  const perFile: Array<{ path: string; findings: number }> = []

  for (const file of files) {
    const diff = file.diff ?? computeDiff(file.before ?? '', file.after ?? '', context)
    let count = 0

    for (const row of diff) {
      const side = sideOf(row.type)
      if (!side) continue
      const line = side === 'removed' ? row.oldNo : row.newNo

      for (const { rule, re, scope } of compiled) {
        if (!scopeAllows(scope, side)) continue
        if (!pathMatches(rule.pathInclude, file.path)) continue
        if (rule.pathExclude != null && pathMatches(rule.pathExclude, file.path)) continue

        re.lastIndex = 0
        let m: RegExpExecArray | null
        // Multiple matches on one line each yield a finding.
        while ((m = re.exec(row.text)) != null) {
          const isBlocking = rule.forceBlocking === true || SEVERITY_ORDER[rule.severity] >= threshold
          findings.push({
            ruleId: rule.id,
            severity: rule.severity,
            message: fillMessage(rule.message, m),
            path: file.path,
            line,
            side,
            excerpt: row.text.trim().slice(0, 240),
            match: m[0],
            blocking: isBlocking,
          })
          count++
          // Guard against zero-width matches spinning forever.
          if (m.index === re.lastIndex) re.lastIndex++
        }
      }
    }

    perFile.push({ path: file.path, findings: count })
  }

  // Rank by severity (desc), then path, then line, then rule id — stable and
  // deterministic so callers can render/compare reliably.
  findings.sort((a, b) => {
    const s = SEVERITY_ORDER[b.severity] - SEVERITY_ORDER[a.severity]
    if (s !== 0) return s
    if (a.path !== b.path) return a.path < b.path ? -1 : 1
    const al = a.line ?? 0
    const bl = b.line ?? 0
    if (al !== bl) return al - bl
    return a.ruleId < b.ruleId ? -1 : a.ruleId > b.ruleId ? 1 : 0
  })

  const capped = findings.slice(0, maxFindings)

  const counts: Record<Severity, number> = { info: 0, warning: 0, error: 0, critical: 0 }
  let blocked = false
  for (const f of capped) {
    counts[f.severity]++
    if (f.blocking) blocked = true
  }

  return {
    findings: capped,
    files: perFile,
    counts,
    blocked,
    rulesApplied: rules.length,
  }
}

// Convenience: review with rules supplied as a markdown rubric.
export function reviewDiffWithRubric(files: ReviewFile[], rubric: string, options: ReviewOptions = {}): ReviewResult {
  return reviewDiff(files, parseRubric(rubric), options)
}
