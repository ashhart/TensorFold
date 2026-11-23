import { readBoundedResponseText } from './bounded-response'
import { assertFetchableUrl, safePublicWebUrl } from './tool-web'
import type { ToolContext, ToolResult } from './tools'

const API = 'https://api.keenable.ai/v1'
const APP_TITLE = 'TensorFold'
const USER_AGENT = 'TensorFold/1.0 (+https://github.com/ashhart/TensorFold)'
const TIMEOUT_MS = 20_000
const RESPONSE_LIMIT = 1024 * 1024
const SNIPPET_CHARS = 2000

async function callKeenable(url: URL, body: object | null, signal?: AbortSignal): Promise<any> {
  const controller = new AbortController()
  const cancel = () => controller.abort(signal?.reason)
  if (signal?.aborted) cancel()
  else signal?.addEventListener('abort', cancel, { once: true })
  const timer = setTimeout(() => controller.abort(new Error('Keenable request timed out.')), TIMEOUT_MS)
  try {
    const response = await fetch(url, {
      method: body ? 'POST' : 'GET', redirect: 'error', signal: controller.signal,
      headers: {
        Accept: 'application/json', 'User-Agent': USER_AGENT, 'X-Keenable-Title': APP_TITLE,
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    })
    const text = await readBoundedResponseText(response, 'Keenable', RESPONSE_LIMIT)
    if (!response.ok) throw new Error(`Keenable request failed: HTTP ${response.status}${response.status === 429 ? ' (rate limit reached)' : ''}.`)
    return JSON.parse(text)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
  }
}

async function fetchPage(args: Record<string, any>, signal?: AbortSignal): Promise<ToolResult> {
  const target = await assertFetchableUrl(String(args.url ?? ''))
  const endpoint = new URL(`${API}/fetch/public`)
  endpoint.searchParams.set('url', target.href)
  const data = await callKeenable(endpoint, null, signal)
  const content = typeof data?.content === 'string' ? data.content.trim() : ''
  if (!content) throw new Error('Keenable could not extract this page.')
  const rawMax = Number(args.max_chars ?? 6000)
  const maxChars = Math.max(500, Math.min(12_000, Number.isFinite(rawMax) ? Math.floor(rawMax) : 6000))
  const title = String(data.title || target.hostname)
  return { ok: true, output: [title, target.href, content].join('\n').slice(0, maxChars), title, path: target.href, meta: 'Keenable' }
}

export async function executeKeenableWebTool(name: 'web_search' | 'web_fetch', args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  if (ctx.networkCapabilityAllowed === false || ctx.disabledCapabilityIds?.includes('tool:web') || ctx.disabledToolNames?.includes(name)) {
    throw new Error('Public web access is disabled for this run.')
  }
  ctx.signal?.throwIfAborted()
  if (name === 'web_fetch') return await fetchPage(args, ctx.signal)
  const query = String(args.query ?? '').trim()
  if (!query) throw new Error('Search query is required.')
  const rawLimit = Number(args.limit ?? 5)
  const limit = Math.max(1, Math.min(8, Number.isFinite(rawLimit) ? Math.floor(rawLimit) : 5))
  const data = await callKeenable(new URL(`${API}/search/public`), { query, max_results: limit }, ctx.signal)
  if (!data || !Array.isArray(data.results)) throw new Error('Keenable returned an invalid search result.')
  // Snippets are page text of a few thousand characters; the cap keeps every requested result in the output.
  const rows = data.results.slice(0, limit).map((row: any) => {
    const target = safePublicWebUrl(String(row.url ?? ''))
    const snippet = typeof row.snippet === 'string' ? row.snippet.trim().slice(0, SNIPPET_CHARS) : ''
    return [String(row.title || target.hostname), target.href, snippet].filter(Boolean).join('\n')
  })
  return { ok: true, output: rows.join('\n\n').slice(0, 25_000) || 'No web results found.', title: query, meta: 'Keenable' }
}
