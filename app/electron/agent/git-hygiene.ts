import { promises as fs } from 'node:fs'
import path from 'node:path'
import { exec } from 'node:child_process'
import type { GitHygieneFinding, GitHygieneKind, GitHygieneReport, GitHygieneSanitizeResult, GitHygieneSeverity, GitHygieneSource } from './types'

const SKIP_DIRS = new Set(['.git', 'node_modules', 'dist', 'dist-electron', '.next', 'build', '.cache', 'vendor', '.venv', '.turbo', 'coverage'])
const SKIP_FILES = new Set(['package-lock.json'])
const MAX_FILE_BYTES = 8 * 1024 * 1024
const MAX_FINDINGS = 600

const AI_ATTRIBUTION_RE = /^(?:co-authored-by|signed-off-by):\s*(?:.*(?:claude|clawed|codex|openai|anthropic|cursor|windsurf|copilot|zcode|z code|ai assistant).*)$/gim
const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi
const ABSOLUTE_PATH_RE = /(?:\/Users\/[A-Za-z0-9._-]+|\/Volumes\/[A-Za-z0-9._ -]+|\/home\/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)/g
const PRIVATE_IP_RE = /\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|169\.254\.\d{1,3}\.\d{1,3})\b/g
const PRIVATE_KEY_RE = /-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----/g
const ENV_SECRET_RE = /\b([A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE[_-]?KEY|CLIENT[_-]?SECRET)[A-Z0-9_]*)\s*([:=])\s*['"]?([^'"\s#,;)}\]]{8,})/gi
const SECRET_PATTERNS: Array<{ re: RegExp; label: string; severity: GitHygieneSeverity }> = [
  { re: /\bAKIA[0-9A-Z]{16}\b/g, label: 'AWS access key id', severity: 'critical' },
  { re: /\bgh[pousr]_[A-Za-z0-9_]{20,}\b/g, label: 'GitHub token', severity: 'critical' },
  { re: /\bsk-[A-Za-z0-9_-]{20,}\b/g, label: 'OpenAI-style API key', severity: 'critical' },
  { re: /\bsk-ant-[A-Za-z0-9_-]{20,}\b/g, label: 'Anthropic-style API key', severity: 'critical' },
  { re: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g, label: 'Slack token', severity: 'critical' },
  { re: /\b(?:eyJ[A-Za-z0-9_-]{16,})\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b/g, label: 'JWT-like token', severity: 'high' },
]
const SENSITIVE_FILENAME_RE = /(^|\/)(?:\.env(?:\.[\w.-]+)?|id_rsa|id_dsa|id_ecdsa|id_ed25519|\.netrc|credentials|secrets?\.(?:json|ya?ml|toml|ini|env)|.*(?:secret|credential|password).*)$/i
const RESERVED_EMAIL_DOMAINS = new Set(['example.com', 'example.org', 'example.net', 'example.test', 'localhost', 'test.invalid', 'invalid.test'])
const PLACEHOLDER_PATH_USERS = new Set(['example', 'demo', 'sample', 'placeholder', 'test'])

function shell(cmd: string, cwd: string, timeout = 30_000): Promise<string> {
  return new Promise((resolve) => exec(cmd, { cwd, timeout, maxBuffer: 40 * 1024 * 1024 }, (err, stdout) => resolve(err ? '' : (stdout || ''))))
}

function qq(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`
}

function idFor(source: GitHygieneSource, kind: GitHygieneKind, file: string | null, line: number | null, commit: string | null, match: string): string {
  const raw = `${source}:${kind}:${commit ?? ''}:${file ?? ''}:${line ?? ''}:${match}`
  let h = 2166136261
  for (let i = 0; i < raw.length; i++) {
    h ^= raw.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return `ghf_${(h >>> 0).toString(16)}`
}

function cleanExcerpt(line: string): string {
  return line.replace(/\s+/g, ' ').trim().slice(0, 240)
}

function redactMatch(value: string, kind: GitHygieneKind): string {
  if (kind === 'email') return value.replace(/^(.).+(@.+)$/, '$1***$2')
  if (kind === 'absolute_path') return value.split(/[\\/]/).slice(0, 3).join('/') + '/…'
  if (value.length <= 12) return value
  return `${value.slice(0, 5)}…${value.slice(-4)}`
}

function finding(input: Omit<GitHygieneFinding, 'id' | 'match'> & { match: string }): GitHygieneFinding {
  return {
    ...input,
    id: idFor(input.source, input.kind, input.file, input.line, input.commit, input.match),
    match: redactMatch(input.match, input.kind),
  }
}

function add(matches: GitHygieneFinding[], candidate: GitHygieneFinding) {
  if (matches.length >= MAX_FINDINGS) return
  if (matches.some((f) => f.id === candidate.id)) return
  matches.push(candidate)
}

function resetAndScan(re: RegExp, text: string): RegExpMatchArray[] {
  re.lastIndex = 0
  return [...text.matchAll(re)]
}

function isLikelyPatternDefinitionLine(text: string): boolean {
  const t = text.trim()
  return /^(?:export\s+)?const\s+[A-Z0-9_]*(?:RE|REGEX|PATTERN|PATTERNS)\b/.test(t)
    || /^\{\s*re:\s*\//.test(t)
    || /\bnew RegExp\s*\(/.test(t)
}

function isTestFixtureFile(file: string | null): boolean {
  return !!file && /\.(?:test|spec)\.(?:[cm]?[jt]sx?|mjs)$/i.test(file)
}

function isFixtureSecretLine(text: string, file: string | null): boolean {
  if (!isTestFixtureFile(file)) return false
  if (file === 'electron/agent/export.test.ts') return true
  const lower = text.toLowerCase()
  return /\b(fixture|sample|mock|test|fake|redact|redaction|do-not-forward|not-indexed)\b/.test(lower)
    || /\bsecret-[a-z0-9_-]+\b/.test(lower)
    || /\boauth-token\b/.test(lower)
    || /\bgateway-key\b/.test(lower)
    || /\b(?:supersecretvalue|hunter2|zzzzzz|abcdef123456)\b/.test(lower)
    || /begin (?:rsa |dsa |ec |openssh |pgp |encrypted )?private key/.test(lower)
}

function isReservedEmail(value: string): boolean {
  const domain = value.split('@').pop()?.toLowerCase() || ''
  return RESERVED_EMAIL_DOMAINS.has(domain)
}

function isEscapedFixtureMatch(text: string, match: RegExpMatchArray): boolean {
  const index = match.index ?? -1
  return index > 0 && text[index - 1] === '\\'
}

function isPlaceholderAbsolutePath(value: string): boolean {
  const normalized = value.replace(/\\/g, '/')
  const m = /^(?:\/Users\/([^/]+)|\/home\/([^/]+)|\/Volumes\/([^/]+)|[A-Za-z]:\/Users\/([^/]+))/.exec(normalized)
  const owner = (m?.[1] || m?.[2] || m?.[3] || m?.[4] || '').toLowerCase()
  return PLACEHOLDER_PATH_USERS.has(owner)
}

function isLikelyEnvSecret(match: RegExpMatchArray): boolean {
  const op = match[2] || ''
  const value = String(match[3] || '').trim()
  const lower = value.toLowerCase()
  if (!value) return false
  if (/^(?:true|false|null|undefined)$/i.test(value)) return false
  if (/^[•*]+$/.test(value) || value === '[set]') return false
  if (/^(?:provider|team|demo|test|sample|example|mock|local|lm-studio)(?:[-_](?:key|token|secret|password|value))?$/.test(lower)) return false
  if ((lower.startsWith('demo-') || lower.startsWith('example-') || lower.startsWith('test-')) && /-(?:key|token|secret|password)$/.test(lower)) return false
  if (lower.startsWith('safe:v')) return false
  if (lower.includes('placeholder')) return false
  if (/[().?${}]/.test(value)) return false
  if (!/^[A-Z0-9_]+$/.test(match[1] || '') && /^[A-Za-z_$][\w$]*$/.test(value)) return false
  if (op === ':' && /^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$/.test(value)) return false
  return true
}

function workspaceTerms(cwd: string): string[] {
  const homeUser = process.env.USER || process.env.LOGNAME || ''
  const segments = cwd.split(path.sep).filter(Boolean)
  const terms = new Set<string>()
  if (homeUser) terms.add(homeUser)
  for (let i = 0; i < segments.length; i++) {
    if (/^(Users|home|Volumes)$/i.test(segments[i]) && segments[i + 1]) terms.add(segments[i + 1])
  }
  for (const term of terms) {
    const clean = term.trim()
    if (!clean || clean.length < 3) continue
    if (/^(Users|Volumes|home|Documents|Projects|Personal|src|app|tmp|var)$/i.test(clean)) terms.delete(term)
  }
  return [...terms].slice(0, 12)
}

function termRegex(term: string): RegExp {
  return new RegExp(`(?<![A-Za-z0-9_])${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![A-Za-z0-9_])`, 'gi')
}

function scanLine(text: string, ctx: { source: GitHygieneSource; file: string | null; line: number | null; commit: string | null; cwd: string; terms: string[] }): GitHygieneFinding[] {
  const out: GitHygieneFinding[] = []
  const excerpt = cleanExcerpt(text)
  const isPatternDefinition = isLikelyPatternDefinitionLine(text)
  const isFixtureSecret = isFixtureSecretLine(text, ctx.file)
  if (!isFixtureSecret) {
    for (const m of resetAndScan(PRIVATE_KEY_RE, text)) add(out, finding({ source: ctx.source, kind: 'private_key', severity: 'critical', file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: 'Remove the private key and rotate it immediately.', autoFixable: false }))
  }
  if (!isPatternDefinition && !isFixtureSecret) {
    for (const m of resetAndScan(ENV_SECRET_RE, text)) {
      if (!isLikelyEnvSecret(m)) continue
      add(out, finding({ source: ctx.source, kind: 'env_secret', severity: 'critical', file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: 'Move this secret to a local .env or secret manager and commit only an example placeholder.', autoFixable: ctx.source === 'working_tree' }))
    }
    for (const pattern of SECRET_PATTERNS) for (const m of resetAndScan(pattern.re, text)) add(out, finding({ source: ctx.source, kind: 'secret', severity: pattern.severity, file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: `${pattern.label} detected. Remove it from Git and rotate it.`, autoFixable: false }))
  }
  for (const m of resetAndScan(EMAIL_RE, text)) {
    if (isFixtureSecret) continue
    if (isEscapedFixtureMatch(text, m)) continue
    if (isReservedEmail(m[0])) continue
    add(out, finding({ source: ctx.source, kind: 'email', severity: 'medium', file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: 'Replace personal email addresses with a role alias or documentation placeholder.', autoFixable: ctx.source === 'working_tree' }))
  }
  for (const m of resetAndScan(ABSOLUTE_PATH_RE, text)) {
    if (isPlaceholderAbsolutePath(m[0])) continue
    add(out, finding({ source: ctx.source, kind: 'absolute_path', severity: 'medium', file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: 'Replace absolute local paths with repo-relative paths or placeholders.', autoFixable: ctx.source === 'working_tree' }))
  }
  for (const m of resetAndScan(PRIVATE_IP_RE, text)) add(out, finding({ source: ctx.source, kind: 'private_network', severity: 'low', file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: 'Check whether this private network address should be documented or replaced with an example address.', autoFixable: false }))
  for (const m of resetAndScan(AI_ATTRIBUTION_RE, text)) add(out, finding({ source: ctx.source, kind: 'ai_attribution', severity: 'low', file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: 'Remove AI co-author / attribution trailers from committed metadata.', autoFixable: ctx.source !== 'git_history' }))
  for (const term of ctx.terms) {
    for (const m of resetAndScan(termRegex(term), text)) {
      add(out, finding({ source: ctx.source, kind: 'personal_term', severity: 'low', file: ctx.file, line: ctx.line, commit: ctx.commit, excerpt, match: m[0], suggestion: `Check whether "${term}" is a personal/company identifier that should be generalized.`, autoFixable: false }))
    }
  }
  return out
}

async function collectWorkspaceFiles(root: string): Promise<string[]> {
  const out: string[] = []
  const walk = async (dir: string) => {
    let entries: any[] = []
    try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
    for (const entry of entries) {
      if (entry.name.startsWith('.') && entry.name !== '.env' && entry.name !== '.env.example') {
        if (entry.name !== '.github') continue
      }
      if (SKIP_DIRS.has(entry.name)) continue
      const full = path.join(dir, entry.name)
      const rel = path.relative(root, full)
      if (entry.isDirectory()) await walk(full)
      else if (entry.isFile() && !SKIP_FILES.has(rel)) out.push(rel)
    }
  }
  await walk(root)
  return out
}

async function scanWorkingTree(root: string, terms: string[]): Promise<{ findings: GitHygieneFinding[]; files: number; bytes: number; notes: string[] }> {
  const findings: GitHygieneFinding[] = []
  const notes: string[] = []
  const files = await collectWorkspaceFiles(root)
  let scannedBytes = 0
  for (const rel of files) {
    if (SENSITIVE_FILENAME_RE.test(rel)) {
      add(findings, finding({ source: 'working_tree', kind: 'sensitive_filename', severity: 'high', file: rel, line: null, commit: null, excerpt: rel, match: rel, suggestion: 'Sensitive filenames should not be committed. Move secrets out of the repo or commit a sanitized example file.', autoFixable: false }))
    }
    const full = path.join(root, rel)
    const stat = await fs.stat(full).catch(() => null)
    if (!stat?.isFile()) continue
    scannedBytes += stat.size
    if (stat.size > MAX_FILE_BYTES) {
      add(findings, finding({ source: 'working_tree', kind: 'large_or_binary', severity: 'low', file: rel, line: null, commit: null, excerpt: `${rel} is ${stat.size} bytes`, match: rel, suggestion: 'Large files are flagged for manual review before pushing.', autoFixable: false }))
      continue
    }
    const buf = await fs.readFile(full).catch(() => null)
    if (!buf) continue
    if (buf.includes(0)) {
      if (SENSITIVE_FILENAME_RE.test(rel)) notes.push(`Binary sensitive-looking file: ${rel}`)
      continue
    }
    const text = buf.toString('utf8')
    const lines = text.split(/\r?\n/)
    for (let i = 0; i < lines.length; i++) {
      for (const hit of scanLine(lines[i], { source: 'working_tree', file: rel, line: i + 1, commit: null, cwd: root, terms })) add(findings, hit)
      if (findings.length >= MAX_FINDINGS) break
    }
    if (findings.length >= MAX_FINDINGS) break
  }
  return { findings, files: files.length, bytes: scannedBytes, notes }
}

const HISTORY_REGEX = [
  'Co-authored-by:',
  '[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}',
  '/Users/[A-Za-z0-9._-]+',
  '/Volumes/[A-Za-z0-9._ -]+',
  '/home/[A-Za-z0-9._-]+',
  'AKIA[0-9A-Z]{16}',
  'gh[pousr]_[A-Za-z0-9_]{20,}',
  'sk-[A-Za-z0-9_-]{20,}',
  'xox[baprs]-[A-Za-z0-9-]{10,}',
  'BEGIN (RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY',
  '[A-Z0-9_]*(API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE[_-]?KEY|CLIENT[_-]?SECRET)[A-Z0-9_]*\\s*[:=]',
].join('|')

async function scanGitHistory(root: string, terms: string[]): Promise<{ findings: GitHygieneFinding[]; commits: number; notes: string[] }> {
  const commitsRaw = await shell('git log --all --format=%H', root, 30_000)
  const commits = commitsRaw.split('\n').map((x) => x.trim()).filter(Boolean)
  if (!commits.length) return { findings: [], commits: 0, notes: ['No Git commits found; history scan skipped.'] }
  const findings: GitHygieneFinding[] = []
  for (let i = 0; i < commits.length; i += 80) {
    const batch = commits.slice(i, i + 80)
    const raw = await shell(`git grep -I -n -i -E ${qq(HISTORY_REGEX)} ${batch.map(qq).join(' ')} -- .`, root, 120_000)
    for (const line of raw.split('\n')) {
      if (!line.trim()) continue
      const m = /^([0-9a-f]{40}):(.+?):(\d+):(.*)$/.exec(line)
      if (!m) continue
      const [, commit, file, lineNo, text] = m
      if (file.split('/').some((part) => SKIP_DIRS.has(part))) continue
      for (const hit of scanLine(text, { source: 'git_history', file, line: Number(lineNo), commit, cwd: root, terms })) add(findings, hit)
      if (findings.length >= MAX_FINDINGS) break
    }
    if (findings.length >= MAX_FINDINGS) break
  }
  return { findings, commits: commits.length, notes: findings.length >= MAX_FINDINGS ? [`History scan stopped after ${MAX_FINDINGS} findings.`] : [] }
}

export function sanitizeCommitMessage(message: string): string {
  const attributionLine = /^(?:co-authored-by|signed-off-by):\s*(?:.*(?:claude|clawed|codex|openai|anthropic|cursor|windsurf|copilot|zcode|z code|ai assistant).*)$/i
  const clean = String(message || 'TensorFold changes')
    .split(/\r?\n/)
    .filter((line) => !attributionLine.test(line))
    .join('\n')
    .trim()
  return clean || 'TensorFold changes'
}

export async function scanGitHygiene(cwd: string, commitMessage = ''): Promise<GitHygieneReport> {
  const terms = workspaceTerms(cwd)
  const [working, history] = await Promise.all([scanWorkingTree(cwd, terms), scanGitHistory(cwd, terms)])
  const findings = [...working.findings, ...history.findings]
  for (const hit of scanLine(commitMessage, { source: 'commit_message', file: null, line: null, commit: null, cwd, terms })) add(findings, hit)
  const summary = {
    critical: findings.filter((f) => f.severity === 'critical').length,
    high: findings.filter((f) => f.severity === 'high').length,
    medium: findings.filter((f) => f.severity === 'medium').length,
    low: findings.filter((f) => f.severity === 'low').length,
  }
  const currentBlockers = findings.filter((f) => f.source !== 'git_history' && (f.severity === 'critical' || f.severity === 'high' || f.severity === 'medium'))
  const historyBlockers = findings.filter((f) => f.source === 'git_history' && (f.severity === 'critical' || f.severity === 'high' || f.severity === 'medium'))
  return {
    cwd,
    scannedAt: Date.now(),
    scannedFiles: working.files,
    scannedBytes: working.bytes,
    scannedCommits: history.commits,
    findings,
    blocked: currentBlockers.length > 0,
    summary,
    notes: [
      ...working.notes,
      ...history.notes,
      ...(historyBlockers.length > 0 && currentBlockers.length === 0
        ? [`Git history still has ${historyBlockers.length} sensitive finding${historyBlockers.length === 1 ? '' : 's'}. Local commits are allowed, but sanitize history before publishing or pushing.`]
        : []),
    ],
  }
}

function redactLine(line: string): string {
  return line
    .replace(ENV_SECRET_RE, (m) => {
      const eq = m.search(/[:=]/)
      return eq >= 0 ? `${m.slice(0, eq + 1)} [redacted]` : '[redacted]'
    })
    .replace(EMAIL_RE, '[redacted-email]')
    .replace(ABSOLUTE_PATH_RE, '[redacted-path]')
}

export async function sanitizeGitHygieneFindings(cwd: string, findings: GitHygieneFinding[]): Promise<GitHygieneSanitizeResult> {
  const changed = new Set<string>()
  const skipped: string[] = []
  const byFile = new Map<string, GitHygieneFinding[]>()
  for (const f of findings) {
    if (!f.autoFixable || f.source !== 'working_tree' || !f.file || !f.line) {
      skipped.push(f.id)
      continue
    }
    if (!byFile.has(f.file)) byFile.set(f.file, [])
    byFile.get(f.file)!.push(f)
  }
  for (const [rel, fileFindings] of byFile) {
    if (path.isAbsolute(rel) || rel.split(/[\\/]+/).includes('..')) {
      skipped.push(...fileFindings.map((f) => f.id))
      continue
    }
    const full = path.join(cwd, rel)
    const text = await fs.readFile(full, 'utf8').catch(() => null)
    if (text == null) {
      skipped.push(...fileFindings.map((f) => f.id))
      continue
    }
    const lines = text.split(/\r?\n/)
    let fileChanged = false
    for (const line of new Set(fileFindings.map((f) => f.line!).filter((n) => n > 0 && n <= lines.length))) {
      if (isLikelyPatternDefinitionLine(lines[line - 1])) {
        skipped.push(...fileFindings.filter((f) => f.line === line).map((f) => f.id))
        continue
      }
      const next = redactLine(lines[line - 1])
      if (next !== lines[line - 1]) {
        lines[line - 1] = next
        fileChanged = true
      } else {
        skipped.push(...fileFindings.filter((f) => f.line === line).map((f) => f.id))
      }
    }
    if (!fileChanged) continue
    await fs.writeFile(full, lines.join('\n'), 'utf8')
    changed.add(rel)
  }
  return { ok: true, changedFiles: [...changed], skippedFindings: skipped }
}
