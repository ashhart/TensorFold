// Conservative value-level detector for keyword-less opaque credentials. It deliberately
// excludes identifiers that are commonly safe to share (UUIDs, Git/checksum hex, and path
// segments) and requires both high Shannon entropy and mixed character classes.

const CANDIDATE = /[A-Za-z0-9_-]{32,}={0,2}/g
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

export type SecretValueRedactionKind =
  | 'private_key' | 'jwt' | 'api_key' | 'aws_key' | 'github_token' | 'slack_token'
  | 'google_key' | 'bearer' | 'assignment' | 'opaque_secret'

interface SecretValueRule {
  kind: SecretValueRedactionKind
  re: RegExp
  mask?: (match: string, mask: string) => string
}

// Shared free-text value scanner used for model-facing command output, audit previews, and
// exports. Vendor formats are recognized even when their entropy is low; the final opaque rule
// catches keyword-less credentials while deliberately preserving UUIDs and hex hashes.
const SECRET_VALUE_RULES: SecretValueRule[] = [
  { kind: 'private_key', re: /-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----/g },
  { kind: 'jwt', re: /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g },
  { kind: 'aws_key', re: /\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}\b/g },
  { kind: 'github_token', re: /\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b/g },
  { kind: 'slack_token', re: /\bxox[baprs]-[A-Za-z0-9-]{8,}\b/g },
  { kind: 'google_key', re: /\bAIza[0-9A-Za-z_-]{30,}\b/g },
  { kind: 'api_key', re: /\b(?:sk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{20,}|(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}|npm_[A-Za-z0-9]{20,})\b/g },
  { kind: 'bearer', re: /\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*/g, mask: (_match, mask) => `Bearer ${mask}` },
  {
    kind: 'assignment',
    re: /(?:[A-Za-z0-9]*[_-])?(?:api[_-]?key|secret|token|password|passwd|pwd|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)\b\s*['"]?\s*[=:]\s*['"]?[^\s'"]{6,}/gi,
    mask: (match, mask) => match.replace(/([=:]\s*['"]?)[^\s'"]+$/, `$1${mask}`),
  },
  {
    kind: 'assignment',
    re: /\b[a-z][a-z0-9+.-]*:\/\/[^\s:@/]+:[^\s:@/]{3,}@/gi,
    mask: (match, mask) => match.replace(/:[^\s:@/]{3,}@$/, `:${mask}@`),
  },
]

export function shannonEntropy(value: string): number {
  const counts = new Map<string, number>()
  for (const char of value) counts.set(char, (counts.get(char) ?? 0) + 1)
  let entropy = 0
  for (const count of counts.values()) {
    const p = count / value.length
    entropy -= p * Math.log2(p)
  }
  return entropy
}

export function looksLikeOpaqueSecret(value: string): boolean {
  const candidate = value.replace(/={1,2}$/, '')
  if (candidate.length < 32 || candidate.length > 512) return false
  if (UUID.test(candidate)) return false
  // SHA/MD5/checksum and Git object ids are hex identifiers, not bearer credentials.
  if (/^[0-9a-f]{32,128}$/i.test(candidate)) return false
  const classes = [/[a-z]/.test(candidate), /[A-Z]/.test(candidate), /\d/.test(candidate), /[_-]/.test(candidate)].filter(Boolean).length
  if (classes < 3) return false
  if (new Set(candidate).size / candidate.length < 0.38) return false
  return shannonEntropy(candidate) >= 4.2
}

export function redactOpaqueSecrets(input: string, mask: string, onHit?: () => void): string {
  return input.replace(CANDIDATE, (candidate, offset: number, whole: string) => {
    // Do not redact a high-entropy-looking filename/path segment; path disclosure has its own
    // explicit anonymization policy and should not make ordinary diagnostic paths unreadable.
    const before = offset > 0 ? whole[offset - 1] : ''
    const after = offset + candidate.length < whole.length ? whole[offset + candidate.length] : ''
    const prefix = whole.slice(Math.max(0, whole.lastIndexOf(' ', offset - 1) + 1), offset)
    const inUrl = /(?:https?|wss?):\/\/\S*$/i.test(prefix)
    if (!inUrl && (before === '/' || before === '\\' || after === '/' || after === '\\')) return candidate
    if (!looksLikeOpaqueSecret(candidate)) return candidate
    onHit?.()
    return mask
  })
}

export function redactSecretValues(
  input: string | undefined,
  mask: string,
): { text: string; hits: SecretValueRedactionKind[] } {
  const hits: SecretValueRedactionKind[] = []
  if (!input) return { text: input ?? '', hits }
  let text = input
  for (const rule of SECRET_VALUE_RULES) {
    text = text.replace(rule.re, (match) => {
      hits.push(rule.kind)
      return rule.mask ? rule.mask(match, mask) : mask
    })
  }
  text = redactOpaqueSecrets(text, mask, () => hits.push('opaque_secret'))
  return { text, hits }
}
