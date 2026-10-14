// Scrub only the credential used for this request, including common wire escaping.
// Diagnostics and ordinary model text otherwise retain their content.
const MASK = '[redacted credential]'
function escapeRegex(value: string): string { return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') }
function normalizePercent(value: string): string { return value.replace(/%[0-9a-f]{2}/gi, (part) => part.toUpperCase()) }
function variants(credential?: string): string[] {
  if (!credential?.trim()) return []
  // Fetch trims trailing header whitespace, and Bearer parsers commonly trim the token.
  const credentials = [...new Set([credential, credential.trimEnd(), credential.trim()])]
  const values: string[] = []
  for (const value of credentials) {
    const json = JSON.stringify(value).slice(1, -1)
    values.push(value, json, json.replace(/\//g, '\\/'), json.replace(/[^\x20-\x7e]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`))
    try { values.push(encodeURIComponent(value), encodeURI(value), new URLSearchParams({ value }).toString().slice(6)) } catch { /* malformed Unicode cannot be URL encoded */ }
  }
  return [...new Set(values)].sort((a, b) => b.length - a.length)
}

function pattern(values: string[]): RegExp | null {
  if (!values.length) return null
  return new RegExp(values.map((value) => escapeRegex(value).replace(/%([0-9a-f])([0-9a-f])/gi, (_match, a, b) => `%[${a.toLowerCase()}${a.toUpperCase()}][${b.toLowerCase()}${b.toUpperCase()}]`)).join('|'), 'g')
}
export function scrubProviderCredential(text: string, credential?: string): string {
  const matcher = pattern(variants(credential))
  return matcher ? text.replace(matcher, MASK) : text
}

export type CredentialTextChunk = { kind: 'content' | 'reasoning'; text: string }

/** Keep original channel tags while buffering only a possible credential prefix. */
export function credentialChunkStream(credential?: string) {
  const values = variants(credential), matcher = pattern(values)
  const normalized = values.map(normalizePercent)
  const max = Math.max(0, ...values.map((value) => value.length - 1))
  let pending: CredentialTextChunk[] = []
  const slice = (chunks: CredentialTextChunk[], from: number, to: number): CredentialTextChunk[] => {
    const result: CredentialTextChunk[] = []
    let offset = 0
    for (const chunk of chunks) {
      const text = chunk.text.slice(Math.max(0, from - offset), Math.max(0, Math.min(chunk.text.length, to - offset)))
      if (text) result.push({ kind: chunk.kind, text })
      offset += chunk.text.length
      if (offset >= to) break
    }
    return result
  }
  const scrub = () => {
    if (!matcher) return
    const source = pending, text = source.map((chunk) => chunk.text).join('')
    const result: CredentialTextChunk[] = []
    let cursor = 0
    matcher.lastIndex = 0
    for (const match of text.matchAll(matcher)) {
      const at = match.index!
      result.push(...slice(source, cursor, at))
      const kind = slice(source, at, at + 1)[0].kind
      result.push({ kind, text: MASK })
      cursor = at + match[0].length
    }
    result.push(...slice(source, cursor, text.length))
    pending = result
  }
  return {
    push(kind: CredentialTextChunk['kind'], text: string): CredentialTextChunk[] {
      if (!matcher) return text ? [{ kind, text }] : []
      pending.push({ kind, text }); scrub()
      const full = pending.map((chunk) => chunk.text).join('')
      let keep = Math.min(max, full.length)
      while (keep && !normalized.some((value) => value.startsWith(normalizePercent(full.slice(-keep))))) keep--
      const output = slice(pending, 0, full.length - keep)
      pending = slice(pending, full.length - keep, full.length)
      return output
    },
    finish(): CredentialTextChunk[] { scrub(); const output = pending; pending = []; return output },
  }
}

export function credentialTextStream(credential?: string) {
  const stream = credentialChunkStream(credential)
  return {
    push: (text: string) => stream.push('content', text).map((chunk) => chunk.text).join(''),
    finish: () => stream.finish().map((chunk) => chunk.text).join(''),
  }
}

export function scrubProviderHeaders(headers: Headers, credential?: string): Headers {
  const clean = new Headers()
  headers.forEach((value, name) => {
    // Header names are case-normalized by Fetch; omit an echoed secret name rather
    // than putting an invalid masked header name into Headers.
    if (credential && variants(credential).some((entry) => name.includes(entry.toLowerCase()))) return
    clean.set(name, scrubProviderCredential(value, credential))
  })
  return clean
}

/** Preserve useful structured error fields, without retaining the raw error/cause. */
export function scrubProviderFailure(error: unknown, credential?: string): unknown {
  if (!credential?.trim()) return error
  const seen = new WeakSet<object>()
  const clean = (value: unknown, depth = 0): any => {
    if (typeof value === 'string') return scrubProviderCredential(value, credential)
    if (!value || typeof value !== 'object') return value
    if (value instanceof Headers) return scrubProviderHeaders(value, credential)
    if (seen.has(value) || depth > 6) return '[bounded error details]'
    seen.add(value)
    if (value instanceof Error) {
      const copy = new Error(scrubProviderCredential(value.message, credential))
      copy.name = scrubProviderCredential(value.name, credential)
      if (value.stack) copy.stack = scrubProviderCredential(value.stack, credential)
      for (const [key, entry] of Object.entries(value)) Object.defineProperty(copy, scrubProviderCredential(key, credential), { value: clean(entry, depth + 1), enumerable: true, writable: true, configurable: true })
      if ('cause' in value && !Object.prototype.hasOwnProperty.call(copy, 'cause')) (copy as any).cause = clean(value.cause, depth + 1)
      return copy
    }
    if (Array.isArray(value)) return value.map((entry) => clean(entry, depth + 1))
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [scrubProviderCredential(key, credential), clean(entry, depth + 1)]))
  }
  return clean(error)
}

/** Decode ordinary JSON wire escapes before scrubbing an HTTP error body. */
export function scrubProviderErrorBody(body: string, credential?: string): string {
  if (!credential?.trim()) return body
  try { return JSON.stringify(scrubProviderFailure(JSON.parse(body), credential)) }
  catch { return scrubProviderCredential(body, credential) }
}

/** Preserve research counters, while never turning a contaminated URL into a link. */
export function scrubProviderResearchMetadata(value: unknown, credential?: string): unknown {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return value
  const raw = value as Record<string, unknown>
  return {
    task: typeof raw.task === 'string' ? scrubProviderCredential(raw.task, credential) : raw.task,
    searches: raw.searches,
    pages_read: raw.pages_read,
    sources: Array.isArray(raw.sources) ? raw.sources.filter((source) => typeof source !== 'string' || !providerUrlHasCredential(source, credential)) : raw.sources,
  }
}

function providerUrlHasCredential(source: string, credential?: string): boolean {
  if (!credential?.trim() || source.length > 2048) return false // the report validator refuses oversized URLs
  if (scrubProviderCredential(source, credential) !== source) return true
  try {
    const url = new URL(source)
    const decode = (value: string) => { try { return decodeURIComponent(value) } catch { return value } }
    const components = [source, url.href, url.hostname, url.username, url.password, url.pathname, url.hash, ...url.searchParams.keys(), ...url.searchParams.values()]
    return components.some((part) => { const decoded = decode(part); return scrubProviderCredential(part, credential) !== part || scrubProviderCredential(decoded, credential) !== decoded })
  } catch { return false } // malformed URLs are omitted by readCortheonReport
}
