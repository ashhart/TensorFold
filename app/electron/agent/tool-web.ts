import { discardResponseBody, pinnedFetch, resolvePinnedHost, type PinnedHost, type PinnedHostResolver } from './pinned-fetch'
import { isGloballyRoutableIp, isPrivateHost } from './security'
import { readResponseTextPrefix } from './bounded-response'

const WEB_FETCH_LIMIT = 20_000

function decodeHtml(value: string): string {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&#(\d+);/g, (_m, n) => String.fromCharCode(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_m, n) => String.fromCharCode(parseInt(n, 16)))
}

export function stripHtml(value: string): string {
  return decodeHtml(value
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim()
}

export function safePublicWebUrl(raw: string): URL {
  const url = new URL(String(raw || '').trim())
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Web URLs must use http or https.')
  if (isPrivateHost(url.hostname)) throw new Error('Refusing to fetch local, private, link-local, or metadata hosts.')
  url.username = ''
  url.password = ''
  return url
}

export async function resolveFetchableUrl(raw: string, lookup?: PinnedHostResolver): Promise<PinnedHost> {
  const url = safePublicWebUrl(raw)
  return resolvePinnedHost(url, (address) => {
    if (!isGloballyRoutableIp(address)) throw new Error(`Refusing to fetch ${url.hostname}: it resolves to a private or non-public address (${address}).`)
  }, { lookup })
}

// Compatibility validator used by tool dispatch and config checks. Point-of-use fetchText
// resolves again and pins that exact result to the socket, so this can never be the sole guard.
export async function assertFetchableUrl(raw: string, lookup?: PinnedHostResolver): Promise<URL> {
  return (await resolveFetchableUrl(raw, lookup)).url
}

function duckDuckGoTarget(rawHref: string): string | null {
  const href = decodeHtml(rawHref)
  try {
    const url = new URL(href, 'https://duckduckgo.com')
    const target = url.searchParams.get('uddg')
    return target ? decodeURIComponent(target) : url.href
  } catch {
    return null
  }
}

export function parseDuckDuckGo(html: string, limit: number): Array<{ title: string; url: string; snippet: string }> {
  const results: Array<{ title: string; url: string; snippet: string }> = []
  const seen = new Set<string>()
  const blocks = html.split(/<div[^>]+class="[^"]*result[^"]*"[^>]*>/i).slice(1)
  for (const block of blocks) {
    if (results.length >= limit) break
    const link = /<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i.exec(block)
    if (!link) continue
    const url = duckDuckGoTarget(link[1])
    if (!url || seen.has(url)) continue
    const title = stripHtml(link[2])
    if (!title) continue
    const snippetMatch = /<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>|<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/div>/i.exec(block)
    const snippet = snippetMatch ? stripHtml(snippetMatch[1] || snippetMatch[2] || '') : ''
    seen.add(url)
    results.push({ title, url, snippet })
  }
  return results
}

// Follow redirects manually, re-validating EVERY hop against the SSRF guard before
// fetching it. `redirect: 'follow'` would happily chase a 302 from a public page to
// http://169.254.169.254/... ; this caps the chain and blocks any internal target.
const BROWSER_HEADERS = {
  // Many sites + search endpoints reject a non-browser agent (401/403/challenge),
  // which made web_fetch/web_search fail and agents loop. Present as a browser.
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
  'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
  'accept-language': 'en-US,en;q=0.9',
}
const MAX_REDIRECTS = 5

export async function fetchText(
  url: URL,
  timeoutMs: number,
  deps: { lookup?: PinnedHostResolver; fetchImpl?: typeof fetch } = {},
): Promise<{ status: number; statusText: string; contentType: string; text: string }> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  let current = url
  try {
    for (let hop = 0; ; hop++) {
      // Resolve immediately before this hop and constrain the socket to that exact answer.
      // DNS failures are fatal; an ordinary fetch fallback would reopen the rebinding race.
      const target = await resolveFetchableUrl(current.toString(), deps.lookup)
      const res = await pinnedFetch(target, { headers: BROWSER_HEADERS, redirect: 'manual', signal: controller.signal }, deps.fetchImpl ?? fetch)
      // Undici returns an opaque-redirect (type: 'opaqueredirect', status 0) when
      // redirect:'manual' meets a 3xx. Read the Location header directly instead.
      const location = res.headers.get('location')
      if (location && res.status >= 300 && res.status < 400 && res.status !== 304) {
        if (hop >= MAX_REDIRECTS) throw new Error(`Too many redirects fetching ${url.hostname}.`)
        // Re-validate the redirect target exactly like the original URL: string host
        // check + DNS resolution. Stops a public page redirecting to an internal host.
        current = safePublicWebUrl(new URL(location, current).toString())
        await discardResponseBody(res)
        continue
      }
      const contentType = res.headers.get('content-type') ?? ''
      const text = await readResponseTextPrefix(res, WEB_FETCH_LIMIT)
      return { status: res.status, statusText: res.statusText, contentType, text }
    }
  } finally {
    clearTimeout(timeout)
  }
}

function decodeXmlText(value: string): string {
  return value
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"')
    .replace(/&#0?39;|&apos;/g, "'").replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ').trim()
}

// Parse an RSS feed (e.g. Google News RSS) into search-result rows. DuckDuckGo
// HTML scraping is now blocked (403); Google News RSS is a reliable, key-free
// source that returns real article links + headlines for any query.
export function parseNewsRss(xml: string, limit: number): Array<{ title: string; url: string; snippet: string }> {
  const out: Array<{ title: string; url: string; snippet: string }> = []
  const itemRe = /<item\b[\s\S]*?<\/item>/gi
  let match: RegExpExecArray | null
  while ((match = itemRe.exec(xml)) && out.length < limit) {
    const block = match[0]
    const title = decodeXmlText(/<title\b[^>]*>([\s\S]*?)<\/title>/i.exec(block)?.[1] ?? '')
    const link = decodeXmlText(/<link\b[^>]*>([\s\S]*?)<\/link>/i.exec(block)?.[1] ?? '')
    const source = decodeXmlText(/<source\b[^>]*>([\s\S]*?)<\/source>/i.exec(block)?.[1] ?? '')
    const pub = decodeXmlText(/<pubDate\b[^>]*>([\s\S]*?)<\/pubDate>/i.exec(block)?.[1] ?? '')
    if (title && link) out.push({ title, url: link, snippet: [source, pub].filter(Boolean).join(' · ') })
  }
  return out
}
