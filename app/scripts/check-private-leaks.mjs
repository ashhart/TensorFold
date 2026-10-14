import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'

const root = fileURLToPath(new URL('..', import.meta.url))
const publicAttributions = JSON.parse(readFileSync(join(root, 'scripts/public-vendor-attributions.json'), 'utf8')).matches
const digest = (value) => createHash('sha256').update(value).digest('hex')

function isPublicAttribution(file, text, match, index) {
  const line = text.slice(text.lastIndexOf('\n', index - 1) + 1, text.indexOf('\n', index) < 0 ? text.length : text.indexOf('\n', index)).replace(/\r$/, '')
  const rel = relative(root, file).split('\\').join('/')
  return publicAttributions.some((entry) => entry.file === rel && entry.lineSha256 === digest(line) && entry.emailSha256 === digest(match))
}

const ignoredDirs = new Set([
  '.git',
  '.claude', // transient harness data — agent worktrees (with their own package-lock) live here, not repo source
  '.research',
  '.tools', // local vendored toolchain (for example a pinned Node runtime), not repo source
  '.vite',
  'build',
  'dist',
  'dist-electron',
  'node_modules',
  'release',
])

const ignoredFiles = new Set([
  'package-lock.json',
])

const allowedDomains = new Set([
  'example.com',
  'example.net',
  'example.org',
  'example.test',
  'localhost',
  'vontra.calendar', // product ICS UID domain (calendar-ics.ts), not a private email
])

const textFileRe = /\.(?:cjs|css|html|js|json|jsx|md|mjs|plist|py|sh|ts|tsx|txt|yml|yaml)$/i
const emailRe = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi

function walk(dir) {
  const out = []
  for (const name of readdirSync(dir)) {
    if (ignoredDirs.has(name)) continue
    const full = join(dir, name)
    // Expo regenerates these ignored native projects, including vendored Pods;
    // mobile source, configuration, modules and its lockfile stay scanned.
    const rel = relative(root, full).split('\\').join('/')
    if (rel === 'mobile/ios' || rel === 'mobile/android') continue
    const stat = statSync(full)
    if (stat.isDirectory()) {
      out.push(...walk(full))
    } else {
      if (!ignoredFiles.has(rel) && textFileRe.test(name)) out.push(full)
    }
  }
  return out
}

function lineAndColumn(text, index) {
  const before = text.slice(0, index)
  const lines = before.split(/\r?\n/)
  return { line: lines.length, column: lines.at(-1).length + 1 }
}

function isAllowedMatch(text, match, index) {
  const at = index + match.indexOf('@')
  const local = match.slice(0, match.indexOf('@')).toLowerCase()
  const domain = match.slice(match.indexOf('@') + 1).toLowerCase()
  if (allowedDomains.has(domain)) return true
  if (local === 'git' && domain === 'github.com') return true

  // URLs such as postgres://user:pass@host are credentials, not email
  // addresses. They are covered by export redaction tests separately.
  if (text[index - 1] === ':' && /[a-z][a-z0-9+.-]*:\/\/[^\s\/\"'<>@]+:$/i.test(text.slice(0, index))) return true

  // Claude/AGENTS import syntax after escaped newlines is a file reference,
  // not an email address.
  if (text.slice(at - 2, at) === '\\n' && /\.(?:json|md|mjs|ts|tsx)$/i.test(domain)) return true

  return false
}

const findings = []

for (const file of walk(root)) {
  const text = readFileSync(file, 'utf8')
  for (const match of text.matchAll(emailRe)) {
    if (isAllowedMatch(text, match[0], match.index ?? 0)) continue
    if (isPublicAttribution(file, text, match[0], match.index ?? 0)) continue
    const pos = lineAndColumn(text, match.index ?? 0)
    findings.push({
      file: relative(root, file).split('\\').join('/'),
      line: pos.line,
      column: pos.column,
      value: match[0],
    })
  }
}

if (findings.length) {
  console.error('Potential private email addresses found. Use example.com/example.test placeholders or redact the value:')
  for (const f of findings) console.error(`  - ${f.file}:${f.line}:${f.column} [redacted email]`)
  process.exit(1)
}

console.log('private leak scan passed')
