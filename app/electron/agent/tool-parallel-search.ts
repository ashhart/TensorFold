import { createHash, randomUUID } from 'node:crypto'
import { readBoundedResponseText } from './bounded-response'
import { assertFetchableUrl, safePublicWebUrl } from './tool-web'
import type { ToolContext, ToolResult } from './tools'

const ENDPOINT = 'https://search.parallel.ai/mcp'
const USER_AGENT = 'TensorFold/1.0 (+https://github.com/ashhart/TensorFold)'
const PROTOCOL = '2025-03-26'
const TIMEOUT_MS = 20_000
const RESPONSE_LIMIT = 1024 * 1024

function rpcResponse(text: string, mime: string, id: number): any {
  const messages = mime.includes('text/event-stream')
    ? text.split(/\r?\n\r?\n/).flatMap((event) => {
      const data = event.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n')
      return data ? [JSON.parse(data)] : []
    })
    : [JSON.parse(text)]
  const reply = messages.find((message) => message.jsonrpc === '2.0' && message.id === id)
  if (!reply) throw new Error('Parallel returned no matching MCP response.')
  if (reply.error) throw new Error(`Parallel MCP error: ${String(reply.error.message || reply.error.code).slice(0, 1000)}`)
  if (!reply.result || typeof reply.result !== 'object') throw new Error('Parallel returned an invalid MCP result.')
  return reply.result
}

async function callParallel(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<any> {
  const controller = new AbortController()
  const cancel = () => controller.abort(signal?.reason)
  if (signal?.aborted) cancel()
  else signal?.addEventListener('abort', cancel, { once: true })
  const timer = setTimeout(() => controller.abort(new Error('Parallel request timed out.')), TIMEOUT_MS)
  let session: string | null = null
  let nextId = 0
  const request = async (method: string, params: object, notification = false): Promise<any> => {
    controller.signal.throwIfAborted()
    const id = ++nextId
    const response = await fetch(ENDPOINT, {
      method: 'POST', redirect: 'error', signal: controller.signal,
      headers: {
        'Content-Type': 'application/json', Accept: 'application/json, text/event-stream',
        'User-Agent': USER_AGENT, 'MCP-Protocol-Version': PROTOCOL,
        ...(session ? { 'Mcp-Session-Id': session } : {}),
      },
      body: JSON.stringify({ jsonrpc: '2.0', ...(notification ? {} : { id }), method, params }),
    })
    const text = await readBoundedResponseText(response, 'Parallel MCP', RESPONSE_LIMIT)
    if (!response.ok) throw new Error(`Parallel request failed: HTTP ${response.status}${response.status === 429 ? ' (rate limit reached)' : ''}.`)
    if (notification) return
    const result = rpcResponse(text, response.headers.get('content-type') || '', id)
    if (method === 'initialize') session = response.headers.get('mcp-session-id')
    return result
  }
  try {
    const initialized = await request('initialize', { protocolVersion: PROTOCOL, capabilities: {}, clientInfo: { name: 'tensorfold', version: '1.0' } })
    if (initialized.protocolVersion !== PROTOCOL) throw new Error('Parallel negotiated an unsupported MCP version.')
    await request('notifications/initialized', {}, true)
    const discovery = await request('tools/list', {})
    if (!Array.isArray(discovery.tools) || !discovery.tools.some((tool: any) => tool.name === name)) throw new Error(`Parallel does not advertise ${name}.`)
    return await request('tools/call', { name, arguments: args })
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
  }
}

function resultData(result: any): any {
  const texts = Array.isArray(result.content) ? result.content.filter((item: any) => item.type === 'text' && typeof item.text === 'string').map((item: any) => item.text) : []
  if (result.isError) throw new Error(`Parallel: ${texts.join('\n').slice(0, 1000) || 'tool call failed'}`)
  const data = result.structuredContent ?? (texts.length ? JSON.parse(texts[0]) : null)
  if (!data || !Array.isArray(data.results)) throw new Error('Parallel returned an invalid web result.')
  return data
}

export async function executeParallelWebTool(name: 'web_search' | 'web_fetch', args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  if (ctx.networkCapabilityAllowed === false || ctx.disabledCapabilityIds?.includes('tool:web') || ctx.disabledToolNames?.includes(name)) {
    throw new Error('Public web access is disabled for this run.')
  }
  ctx.signal?.throwIfAborted()
  // Hash the host-owned random thread UUID to keep one opaque identity across turns and app restarts.
  const sessionId = ctx.currentThread
    ? createHash('sha256').update(`tensorfold-parallel:${ctx.currentThread.id}`).digest('hex')
    : randomUUID()
  const query = String(args.query ?? '').trim()
  if (name === 'web_search' && !query) throw new Error('Search query is required.')
  const url = name === 'web_fetch' ? await assertFetchableUrl(String(args.url ?? '')) : null
  const parameters = name === 'web_search'
    ? { objective: query, search_queries: [query], session_id: sessionId }
    : { urls: [url!.href], session_id: sessionId }
  const data = resultData(await callParallel(name, parameters, ctx.signal))
  const rawLimit = Number(args.limit ?? 5)
  const limit = Math.max(1, Math.min(8, Number.isFinite(rawLimit) ? Math.floor(rawLimit) : 5))
  const rows = (name === 'web_search' ? data.results.slice(0, limit) : data.results).map((row: any) => {
    const target = safePublicWebUrl(String(row.url ?? ''))
    const excerpts = Array.isArray(row.excerpts) ? row.excerpts.filter((excerpt: unknown) => typeof excerpt === 'string') : []
    return [String(row.title || target.hostname), target.href, ...excerpts].join('\n')
  })
  if (name === 'web_fetch' && !rows.length) throw new Error('Parallel could not extract this page.')
  const rawMax = Number(args.max_chars ?? 6000)
  const maxChars = name === 'web_fetch' ? Math.max(500, Math.min(12_000, Number.isFinite(rawMax) ? Math.floor(rawMax) : 6000)) : 25_000
  return { ok: true, output: rows.join('\n\n').slice(0, maxChars) || 'No web results found.', title: query || url!.hostname, meta: 'Parallel' }
}
