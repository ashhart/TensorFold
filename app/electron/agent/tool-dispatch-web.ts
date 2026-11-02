import path from 'node:path'
import { assertFetchableUrl, fetchText, parseNewsRss, stripHtml } from './tool-web'
import { executeParallelWebTool } from './tool-parallel-search'
import type { ToolContext, ToolResult } from './tool-context'


// Public web tools preserve URL validation and the selected search backend.
export async function executeWebTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  switch (name) {
    case 'web_search': {
      if (args.provider === 'parallel') return await executeParallelWebTool(name, args, ctx)
      if (args.provider != null && args.provider !== 'google_news') return { ok: false, output: 'Unknown web search provider.' }
      const query = String(args.query ?? '').trim()
      if (!query) return { ok: false, output: 'Search query is required.', title: 'Web search' }
      const rawLimit = Number(args.limit ?? 5)
      const limit = Math.max(1, Math.min(8, Number.isFinite(rawLimit) ? Math.floor(rawLimit) : 5))
      const url = new URL('https://news.google.com/rss/search')
      url.searchParams.set('q', query)
      url.searchParams.set('hl', 'en-US')
      url.searchParams.set('gl', 'US')
      url.searchParams.set('ceid', 'US:en')
      const fetched = await fetchText(url, 10_000, { signal: ctx.signal })
      if (fetched.status < 200 || fetched.status >= 300) {
        return { ok: false, output: `Search failed: HTTP ${fetched.status} ${fetched.statusText}`.trim(), title: query, meta: 'web' }
      }
      const results = parseNewsRss(fetched.text, limit)
      const output = results.map((result, index) => [
        `${index + 1}. ${result.title}`,
        `   ${result.url}`,
        result.snippet ? `   ${result.snippet}` : '',
      ].filter(Boolean).join('\n')).join('\n\n')
      return { ok: true, output: output || 'No web results found.', title: query, meta: `${results.length} results` }
    }

    case 'web_fetch': {
      if (args.provider === 'parallel') return await executeParallelWebTool(name, args, ctx)
      if (args.provider != null && args.provider !== 'direct') return { ok: false, output: 'Unknown web fetch provider.' }
      // assertFetchableUrl validates the host string AND DNS-resolves it, so a
      // public name that resolves to an internal/metadata IP is rejected before
      // we fetch. fetchText also re-checks every redirect hop.
      const url = await assertFetchableUrl(String(args.url ?? ''))
      const fetched = await fetchText(url, 10_000, { signal: ctx.signal })
      const rawMax = Number(args.max_chars ?? 6000)
      const maxChars = Math.max(500, Math.min(12_000, Number.isFinite(rawMax) ? Math.floor(rawMax) : 6000))
      const body = fetched.contentType.includes('html') ? stripHtml(fetched.text) : fetched.text.replace(/\s+/g, ' ').trim()
      const title = fetched.contentType.includes('html')
        ? stripHtml(/<title[^>]*>([\s\S]*?)<\/title>/i.exec(fetched.text)?.[1] ?? url.hostname)
        : url.hostname
      const output = [
        `URL: ${url.href}`,
        `Status: ${fetched.status} ${fetched.statusText}`.trim(),
        `Content-Type: ${fetched.contentType || 'unknown'}`,
        '',
        body.slice(0, maxChars) || '(no readable text)',
      ].join('\n')
      return { ok: fetched.status >= 200 && fetched.status < 300, output, title, path: url.href, meta: 'web' }
    }
    default: return { ok: false, output: `Unknown tool: ${name}` }
  }
}
