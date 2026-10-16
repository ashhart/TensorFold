import { isIPv6 } from 'node:net'
import { contentText } from './content.ts'
import { redactSecretValues } from './secret-detection'
import type { ApiMessage, ContentBlock } from './types'

// Outbound guard for external (planning-only) providers. Withholds tool traffic and images,
// blocks credential files, and swaps secrets, PII, the workspace path and the user's
// confidential terms for placeholders such as [EMAIL_1]. The placeholder map stays in this
// process; replies are rehydrated locally for display and for the local engine.

export type WithheldKind =
  | 'tool_output' | 'tool_call' | 'image' | 'credential' | 'secret' | 'project_context'
  | 'workspace' | 'home_path' | 'username' | 'hostname' | 'email' | 'phone' | 'ip'
  | 'card' | 'iban' | 'ssn' | 'nino' | 'term'

export type WithheldTally = Partial<Record<WithheldKind, number>>

export interface RedactionContext {
  workspace?: string | null
  homeDir?: string | null
  username?: string | null
  hostname?: string | null
  terms?: readonly string[]
}

const PLACEHOLDER_RE = /\[[A-Z]+_\d+\]/g
const PARTIAL_PLACEHOLDER_RE = /^\[[A-Z]*_?\d*$/
const MAX_PLACEHOLDER_CHARS = 24
const SECRET_SENTINEL = '\u0000SECRET\u0000'
const GENERIC_ACCOUNTS = new Set(['root', 'admin', 'user', 'users', 'guest', 'shared', 'dev', 'developer', 'test', 'runner', 'build', 'ubuntu', 'localhost'])
const CREDENTIAL_NOTE = '[Withheld: this part held a private key or credential file and was not sent.]'
const IMAGE_NOTE = '[Image withheld from the external model.]'

/** Stable per-thread placeholder map. Secrets never enter it, so they are never rehydrated. */
export class PlaceholderVault {
  private tokens = new Map<string, string>()
  private values = new Map<string, string>()
  private counters = new Map<string, number>()

  placeholder(label: string, value: string, key = value): string {
    const id = `${label}\u0000${key}`
    let token = this.tokens.get(id)
    if (!token) {
      const n = (this.counters.get(label) ?? 0) + 1
      this.counters.set(label, n)
      token = `[${label}_${n}]`
      this.tokens.set(id, token)
      this.values.set(token, value)
    }
    return token
  }

  rehydrate(text: string): string {
    return text.replace(PLACEHOLDER_RE, (token) => this.values.get(token) ?? token)
  }
}

interface Rule {
  kind: WithheldKind
  label: string
  re: RegExp
  key?: (match: string) => string
  // Returns the part of the match to replace (a prefix), or null to keep the match.
  accept?: (match: string) => string | null
}

const escapeRe = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
const PATH_END = String.raw`(?=$|[/\s'"\`)\]}>,;:]|\.(?:\s|$))`
const digits = (value: string) => value.replace(/\D/g, '')

function luhn(number: string): boolean {
  let sum = 0
  for (let i = number.length - 1, alt = false; i >= 0; i--, alt = !alt) {
    let n = Number(number[i])
    if (alt && (n *= 2) > 9) n -= 9
    sum += n
  }
  return sum % 10 === 0
}

function ibanValid(raw: string): boolean {
  const s = raw.replace(/\s/g, '')
  if (!/^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$/.test(s)) return false
  let rem = 0
  for (const ch of s.slice(4) + s.slice(0, 4)) {
    for (const c of /\d/.test(ch) ? ch : String(ch.charCodeAt(0) - 55)) rem = (rem * 10 + Number(c)) % 97
  }
  return rem === 1
}

function longestValidIban(match: string): string | null {
  const parts = match.split(' ')
  for (let k = parts.length; k >= 1; k--) {
    const candidate = parts.slice(0, k).join(' ')
    if (ibanValid(candidate)) return candidate
  }
  return null
}

function contextRules(ctx: RedactionContext): Rule[] {
  const rules: Rule[] = []
  const workspace = String(ctx.workspace ?? '').replace(/\/+$/, '')
  if (workspace.length > 1) rules.push({ kind: 'workspace', label: 'WORKSPACE', re: new RegExp(escapeRe(workspace) + PATH_END, 'g') })
  const home = String(ctx.homeDir ?? '').replace(/\/+$/, '')
  if (home.length > 1) rules.push({ kind: 'home_path', label: 'HOME', re: new RegExp(escapeRe(home) + PATH_END, 'g') })
  return rules
}

function wordRule(kind: WithheldKind, label: string, words: string[]): Rule | null {
  const list = [...new Set(words.map((w) => w.trim()).filter((w) => w.length >= 2))].sort((a, b) => b.length - a.length).slice(0, 200)
  if (!list.length) return null
  return { kind, label, re: new RegExp(`(?<![\\p{L}\\p{N}_])(?:${list.map(escapeRe).join('|')})(?![\\p{L}\\p{N}_])`, 'giu'), key: (m) => m.toLowerCase() }
}

function piiRules(ctx: RedactionContext): Rule[] {
  const host = String(ctx.hostname ?? '').trim()
  const shortHost = host.split('.')[0]
  const username = String(ctx.username ?? '').trim()
  const rules: Array<Rule | null> = [
    {
      // Other accounts' home folders keep their shape; only the account name is withheld.
      kind: 'home_path', label: 'USER', re: /(?<=\/(?:Users|home)\/)(?!Shared\b|Guest\b)[^/\s'"`:[\]]+/g, key: (m) => m.toLowerCase(),
    },
    { kind: 'email', label: 'EMAIL', re: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, key: (m) => m.toLowerCase() },
    wordRule('hostname', 'HOST', [host, shortHost].filter((h) => h && h.toLowerCase() !== 'localhost')),
    wordRule('term', 'TERM', [...(ctx.terms ?? [])]),
    {
      kind: 'ip', label: 'IP', re: /(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w]|\.\d)/g,
      accept: (m) => (m.startsWith('127.') || m === '0.0.0.0' || m === '255.255.255.255' ? null : m),
    },
    {
      kind: 'ip', label: 'IP', re: /(?<![\w:])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])/g,
      accept: (m) => (isIPv6(m) && m !== '::' && m !== '::1' ? m : null), key: (m) => m.toLowerCase(),
    },
    {
      kind: 'card', label: 'CARD', re: /(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])/g, key: digits,
      accept: (m) => { const d = digits(m); return /^(?:4|5[1-5]|2[2-7]|3[0-9]|6)/.test(d) && luhn(d) ? m : null },
    },
    { kind: 'iban', label: 'IBAN', re: /\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b/g, key: (m) => m.replace(/\s/g, ''), accept: longestValidIban },
    { kind: 'ssn', label: 'SSN', re: /\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b/g },
    { kind: 'nino', label: 'NINO', re: /\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b/g, key: (m) => m.replace(/\s/g, '') },
    {
      kind: 'phone', label: 'PHONE', key: digits,
      re: /(?<![\w+])\+\d[\d\s().-]{6,}\d(?!\w)|(?<![\w(])(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\w)|(?<!\w)0\d{2,4}[\s-]\d{3,4}[\s-]?\d{3,4}(?!\w)/g,
      accept: (m) => { const n = digits(m).length; return n >= 9 && n <= 15 ? m : null },
    },
    username.length >= 3 && !GENERIC_ACCOUNTS.has(username.toLowerCase()) ? wordRule('username', 'USER', [username]) : null,
  ]
  return rules.filter((rule): rule is Rule => !!rule)
}

function mapOutsidePlaceholders(text: string, fn: (segment: string) => string): string {
  let out = ''
  let last = 0
  for (const match of text.matchAll(PLACEHOLDER_RE)) {
    out += fn(text.slice(last, match.index)) + match[0]
    last = (match.index ?? 0) + match[0].length
  }
  return out + fn(text.slice(last))
}

function bump(tally: WithheldTally, kind: WithheldKind, by = 1): void {
  tally[kind] = (tally[kind] ?? 0) + by
}

const PRIVATE_KEY_RE = /-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----/
const CREDENTIAL_FILE_RES = [
  /aws_secret_access_key\s*=/i,
  /"type"\s*:\s*"service_account"/,
  /^\s*machine\s+\S+\s+login\s+\S+\s+password\s+\S+/m,
  /:_authToken\s*=/,
]

/** One redaction pass over a request. Secret numbering restarts per pass; the vault does not. */
export class RedactionRun {
  readonly tally: WithheldTally = {}
  private secrets = 0
  private rules: Rule[]

  constructor(readonly vault: PlaceholderVault, ctx: RedactionContext) {
    this.rules = [...contextRules(ctx), ...piiRules(ctx)]
  }

  text(input: string): string {
    if (!input) return input
    if (PRIVATE_KEY_RE.test(input) || CREDENTIAL_FILE_RES.some((re) => re.test(input))) {
      bump(this.tally, 'credential')
      return CREDENTIAL_NOTE
    }
    const scanned = redactSecretValues(input, SECRET_SENTINEL)
    // Three or more credentials in one part reads as a pasted .env or config file.
    if (scanned.hits.length >= 3) {
      bump(this.tally, 'credential')
      return CREDENTIAL_NOTE
    }
    let text = scanned.text.split(SECRET_SENTINEL).reduce((acc, part, i) => (i === 0 ? part : `${acc}[SECRET_${++this.secrets}]${part}`), '')
    bump(this.tally, 'secret', scanned.hits.length)
    for (const rule of this.rules) {
      text = mapOutsidePlaceholders(text, (segment) => segment.replace(rule.re, (match) => {
        const hit = rule.accept ? rule.accept(match) : match
        if (!hit) return match
        bump(this.tally, rule.kind)
        return this.vault.placeholder(rule.label, hit, rule.key ? rule.key(hit) : hit) + match.slice(hit.length)
      }))
    }
    return text
  }

  content(content: ApiMessage['content']): string {
    if (content == null || typeof content === 'string') return this.text(content ?? '')
    const parts = content.map((block: ContentBlock) => {
      if (block.type === 'text') return this.text(block.text)
      bump(this.tally, 'image')
      return IMAGE_NOTE
    })
    return parts.join('\n')
  }

  /** Withhold tool traffic, drop images, redact the rest. Output has no tool roles or tool_calls. */
  messages(messages: readonly ApiMessage[]): ApiMessage[] {
    const out: ApiMessage[] = []
    for (const message of messages) {
      if (message.role === 'tool') {
        bump(this.tally, 'tool_output')
        continue
      }
      let content = this.content(message.content)
      if (message.role === 'assistant' && message.tool_calls?.length) {
        bump(this.tally, 'tool_call', message.tool_calls.length)
        const counts = new Map<string, number>()
        for (const call of message.tool_calls) counts.set(call.function.name, (counts.get(call.function.name) ?? 0) + 1)
        const names = [...counts].map(([name, n]) => (n > 1 ? `${name} x${n}` : name)).join(', ')
        content = [content, this.text(`[Used tools: ${names}. Arguments and results were withheld from this external model.]`)].filter(Boolean).join('\n\n')
      }
      const previous = out[out.length - 1]
      if (message.role === 'assistant' && previous?.role === 'assistant') {
        previous.content = [previous.content, content].filter(Boolean).join('\n\n')
        continue
      }
      out.push({ role: message.role, content })
    }
    return out
  }
}

/** Streaming rehydration that holds back a placeholder split across chunks. */
export function createStreamRehydrator(vault: PlaceholderVault) {
  let pending = ''
  return {
    push(text: string): string {
      const all = pending + text
      const open = all.lastIndexOf('[')
      if (open >= 0 && all.length - open < MAX_PLACEHOLDER_CHARS && PARTIAL_PLACEHOLDER_RE.test(all.slice(open))) {
        pending = all.slice(open)
        return vault.rehydrate(all.slice(0, open))
      }
      pending = ''
      return vault.rehydrate(all)
    },
    flush(): string {
      const rest = pending
      pending = ''
      return vault.rehydrate(rest)
    },
  }
}

const KIND_NAMES: Record<WithheldKind, [string, string]> = {
  tool_output: ['tool result', 'tool results'],
  tool_call: ['tool call', 'tool calls'],
  image: ['image', 'images'],
  credential: ['credential file', 'credential files'],
  secret: ['secret', 'secrets'],
  project_context: ['project guidance or memory block', 'project guidance or memory blocks'],
  workspace: ['workspace path', 'workspace paths'],
  home_path: ['home path', 'home paths'],
  username: ['username', 'usernames'],
  hostname: ['hostname', 'hostnames'],
  email: ['email', 'emails'],
  phone: ['phone number', 'phone numbers'],
  ip: ['IP address', 'IP addresses'],
  card: ['card number', 'card numbers'],
  iban: ['IBAN', 'IBANs'],
  ssn: ['SSN', 'SSNs'],
  nino: ['NI number', 'NI numbers'],
  term: ['confidential term', 'confidential terms'],
}

export function withheldTotal(tally: WithheldTally): number {
  return Object.values(tally).reduce((sum, n) => sum + (n ?? 0), 0)
}

export function describeWithheld(tally: WithheldTally): string {
  return (Object.entries(tally) as Array<[WithheldKind, number]>)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([kind, n]) => `${n} ${KIND_NAMES[kind][n === 1 ? 0 : 1]}`)
    .join(', ')
}
