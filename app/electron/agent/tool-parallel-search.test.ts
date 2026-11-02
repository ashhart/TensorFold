import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { mock } from 'node:test'
import ts from 'typescript'
import { executeTool, TOOL_SCHEMAS } from './tools'
import { runAgent } from './loop'
import type { Thread } from './types'
import { installDnsFixture } from './test-fixtures/dns'

const endpoint = 'https://search.parallel.ai/mcp'
const publicPage = 'https://93.184.216.34/article'
const originalFetch = globalThis.fetch
const restoreDns = installDnsFixture({ 'news.google.com': '8.8.8.8' })
const requests: Array<{ body: any; headers: Headers }> = []
let fault = ''
let httpSessions = 0
let sse = false

// Execute the production creation handler with inert persistence and renderer collaborators.
const source = await readFile(new URL('../main/host/ipc/workspace-ipc.ts', import.meta.url), 'utf8')
const tree = ts.createSourceFile('workspace-ipc.ts', source, ts.ScriptTarget.ES2022, true)
let createExpression: ts.Node | undefined
function visit(node: ts.Node): void {
  if (ts.isCallExpression(node) && node.expression.getText(tree) === 'handle' && node.arguments[0]?.getText(tree) === "'threads:create'") createExpression = node.arguments[1]
  ts.forEachChild(node, visit)
}
visit(tree)
assert.ok(createExpression)
const handler = ts.createPrinter().printNode(ts.EmitHint.Expression, createExpression, tree)
const compiled = ts.transpileModule(`const create = ${handler};`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const store: any = { threads: [], projects: [], settings: { providers: [], activeProviderId: null, activeModel: null } }
const create = new Function('store', 'randomUUID', 'requireSecurityRole', 'spine', 'persistThreads', 'remoteServer', `${compiled}\nreturn create;`)(store, randomUUID, () => {}, () => ({ executeCommand() {} }), () => {}, { publish() {} })
const newChat = (): Promise<Thread> => create(null, { cwd: '', kind: 'chat' })

function rpc(body: any, result: any, headers: Record<string, string> = {}): Response {
  const message = JSON.stringify({ jsonrpc: '2.0', id: body.id, result })
  return new Response(sse ? `event: message\r\ndata: ${message}\r\n\r\n` : message, {
    headers: { 'Content-Type': sse ? 'text/event-stream' : 'application/json', ...headers },
  })
}

globalThis.fetch = async (input, init) => {
  if (String(input).startsWith('https://news.google.com/rss/search?')) return new Response('<rss><channel><item><title>Default news source</title><link>https://example.org/news</link></item></channel></rss>')
  if (String(input) === publicPage) return new Response('<title>Direct page</title><p>Direct page text.</p>', { headers: { 'Content-Type': 'text/html' } })
  if (String(input) !== endpoint) return originalFetch(input, init)
  const body = JSON.parse(String(init?.body))
  const headers = new Headers(init?.headers)
  requests.push({ body, headers })
  assert.equal(init?.redirect, 'error')
  assert.equal(headers.get('Authorization'), null)
  assert.equal(headers.get('x-api-key'), null)
  assert.equal(headers.get('User-Agent'), 'TensorFold/1.0 (+https://github.com/ashhart/TensorFold)')
  assert.equal(headers.get('MCP-Protocol-Version'), '2025-03-26')
  if (body.method !== 'initialize') assert.match(headers.get('Mcp-Session-Id') || '', /^transport-/)
  if (fault === `stall:${body.method}`) return new Promise((_resolve, reject) => {
    init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true })
  })
  if (body.method === 'initialize') return rpc(body, { protocolVersion: '2025-03-26' }, { 'Mcp-Session-Id': `transport-${++httpSessions}` })
  if (body.method === 'notifications/initialized') return new Response(null, { status: 202 })
  if (body.method === 'tools/list') return rpc(body, { tools: fault === 'missing' ? [] : [{ name: 'web_search' }, { name: 'web_fetch' }] })
  assert.equal(body.method, 'tools/call')
  if (fault === 'http') return new Response('quota', { status: 429 })
  if (fault === 'invalid') return new Response('invalid JSON', { headers: { 'Content-Type': 'application/json' } })
  if (fault === 'mismatch') return rpc({ id: -1 }, {})
  if (fault === 'rpc') return new Response(JSON.stringify({ jsonrpc: '2.0', id: body.id, error: { code: -32602, message: 'Invalid arguments' } }))
  if (fault === 'size') return new Response('x'.repeat(1024 * 1024 + 1))
  if (fault === 'tool') return rpc(body, { isError: true, content: [{ type: 'text', text: 'Extraction refused' }] })
  if (fault === 'shape') return rpc(body, { structuredContent: { results: 'invalid' } })
  const results = [{ title: 'Example source', url: publicPage, excerpts: ['Useful public excerpt.'] }, { title: 'Another source', url: 'https://example.org/', excerpts: ['Second excerpt.'] }]
  if (fault === 'private-result') results[0].url = 'http://127.0.0.1/secret'
  const data = { results: fault === 'empty' ? [] : results }
  return rpc(body, sse ? { content: [{ type: 'text', text: JSON.stringify(data) }] } : { structuredContent: data })
}

const search = (thread: Thread, extra: Record<string, unknown> = {}, context: any = {}) => executeTool('web_search', { query: 'public research question', provider: 'parallel', ...extra }, { cwd: '', currentThread: thread, ...context })
const calls = () => requests.filter(({ body }) => body.method === 'tools/call').map(({ body }) => body.params)
try {
  process.env.PARALLEL_API_KEY = 'synthetic-ignored-key'
  for (const name of ['web_search', 'web_fetch']) {
    const schema: any = TOOL_SCHEMAS.find((schema) => schema.function?.name === name)
    assert.ok(schema.function.parameters.properties.provider.enum.includes('parallel'), 'the maintained tool schema exposes Parallel')
  }
  const a = await newChat(), b = await newChat()
  for (const provider of [undefined, 'google_news']) {
    const news = await executeTool('web_search', { query: 'news', provider }, { cwd: '', currentThread: a })
    assert.equal(news.ok, true, news.output)
    assert.match(news.output, /Default news source/)
  }
  for (const provider of [undefined, 'direct']) {
    const page = await executeTool('web_fetch', { url: publicPage, provider }, { cwd: '', currentThread: a })
    assert.equal(page.ok, true, page.output)
    assert.match(page.output, /Direct page text/)
  }
  assert.equal(requests.length, 0, 'omitted and explicit existing providers never reach Parallel')
  const first = await search(a, { limit: 1 })
  assert.equal(first.ok, true, first.output)
  assert.match(first.output, /Example source\nhttps:\/\/93\.184\.216\.34\/article\nUseful public excerpt/)
  assert.doesNotMatch(first.output, /Another source/)
  assert.deepEqual(calls()[0].arguments.search_queries, ['public research question'])
  const fetched = await executeTool('web_fetch', { url: publicPage, provider: 'parallel', max_chars: 500 }, { cwd: '', currentThread: a })
  assert.equal(fetched.ok, true, fetched.output)
  assert.match(fetched.output, /Useful public excerpt/)
  assert.deepEqual(calls()[1].arguments.urls, [publicPage])
  sse = true
  await search(JSON.parse(JSON.stringify(a)))
  await Promise.all([search(a), search(b), search(a), search(b)])
  const ids = calls().map((call) => call.arguments.session_id)
  assert.match(ids[0], /^[a-f0-9]{64}$/)
  for (const index of [1, 2, 3, 5]) assert.equal(ids[index], ids[0])
  assert.equal(ids[4], ids[6])
  assert.notEqual(ids[0], ids[4])
  const reset = await newChat()
  await search(reset)
  assert.notEqual(calls().at(-1).arguments.session_id, ids[0])
  fault = 'http'
  assert.match((await search(a)).output, /HTTP 429/)
  fault = ''
  await search(a)
  assert.equal(calls().at(-1).arguments.session_id, ids[0], 'quota retry retains conversation identity')
  for (const name of ['web_search', 'web_fetch']) {
    const standalone = await executeTool(name, { provider: 'parallel', query: 'public question', url: publicPage }, { cwd: '' })
    assert.equal(standalone.ok, true, standalone.output)
    assert.equal('session_id' in calls().at(-1).arguments, false, 'standalone hosts have no conversation boundary to identify')
  }
  for (const [kind, message] of [['missing', /does not advertise/], ['invalid', /JSON|Unexpected/], ['mismatch', /matching/], ['rpc', /Invalid arguments/], ['size', /limit/], ['tool', /Extraction refused/], ['shape', /invalid web result/], ['private-result', /private|local/]]) {
    fault = kind as string
    assert.match((await search(a)).output, message as RegExp)
  }
  fault = 'empty'
  assert.equal((await search(a)).output, 'No web results found.')
  assert.equal((await executeTool('web_fetch', { provider: 'parallel', url: publicPage }, { cwd: '', currentThread: a })).ok, false)
  fault = ''
  const beforeBlocked = requests.length
  for (const ctx of [{ disabledToolNames: ['web_search'] }, { disabledCapabilityIds: ['tool:web'] }, { networkCapabilityAllowed: false }, { allowedToolNames: [] }, { phoneLocalOnly: true }]) {
    assert.equal((await search(a, {}, ctx)).ok, false)
  }
  assert.equal((await search({ ...a, googleLocalOnly: true })).ok, false)
  assert.equal((await search(a, { provider: 'unknown' })).ok, false)
  assert.equal((await search(a, { query: '' })).ok, false)
  for (const url of ['http://127.0.0.1/secret', 'http://169.254.169.254/latest', 'file:///etc/passwd']) {
    assert.equal((await executeTool('web_fetch', { url, provider: 'parallel' }, { cwd: '', currentThread: a })).ok, false)
  }
  assert.equal(requests.length, beforeBlocked, 'blocked calls never reach the provider')
  for (const method of ['initialize', 'tools/call']) {
    fault = `stall:${method}`
    const controller = new AbortController()
    const pending = search(a, {}, { signal: controller.signal })
    await new Promise<void>((resolve) => setImmediate(resolve))
    controller.abort(new Error('Synthetic cancellation'))
    assert.match((await pending).output, /Synthetic cancellation/)
    mock.timers.enable({ apis: ['setTimeout'] })
    try {
      const timeout = search(a)
      await new Promise<void>((resolve) => setImmediate(resolve))
      mock.timers.tick(20_000)
      assert.match((await timeout).output, /timed out/)
    } finally { mock.timers.reset() }
  }
  fault = ''

  // A scripted model requests the optional provider and consumes the real dispatcher's result on its next turn.
  const modelRequests: any[] = []
  const server = createServer((request, response) => {
    let raw = ''
    request.on('data', (chunk) => { raw += chunk })
    request.on('end', () => {
      const body = JSON.parse(raw)
      modelRequests.push(body)
      const delta = modelRequests.length === 1
        ? { tool_calls: [{ index: 0, id: 'search-call', type: 'function', function: { name: 'web_search', arguments: JSON.stringify({ query: 'public research question', provider: 'parallel' }) } }] }
        : { content: 'The public excerpt answers the question.' }
      response.writeHead(200, { 'Content-Type': 'text/event-stream' })
      response.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: modelRequests.length === 1 ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`)
    })
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  try {
    const thread = await newChat()
    thread.apiMessages.push({ role: 'user', content: 'Use Parallel to research this question.' })
    await runAgent({ thread, provider: { id: 'scripted', name: 'Scripted fixture', kind: 'local', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, apiKey: '', models: ['scripted'] }, model: 'scripted', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', signal: new AbortController().signal, persist() {}, autoCompact: false, maxSteps: 3, requestApproval: async () => 'deny', emit() {} })
    assert.equal(modelRequests.length, 2)
    assert.ok(modelRequests[1].messages.some((message: any) => message.role === 'tool' && message.content.includes('Useful public excerpt.')))
    assert.ok(thread.apiMessages.some((message) => message.role === 'assistant' && String(message.content).includes('answers the question')))
  } finally { await new Promise<void>((resolve) => server.close(() => resolve())) }
  console.log('Parallel caller checks passed: dispatch, scripted loop consumption, discovery, JSON/SSE, identity, permissions, errors, timeout and cancellation.')
} finally {
  globalThis.fetch = originalFetch
  restoreDns()
  delete process.env.PARALLEL_API_KEY
  mock.timers.reset()
}
