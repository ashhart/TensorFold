import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { mock } from 'node:test'
import ts from 'typescript'
import { executeTool, TOOL_SCHEMAS } from './tools'
import { runAgent } from './loop'
import type { Thread } from './types'

const searchEndpoint = 'https://api.keenable.ai/v1/search/public'
const fetchEndpoint = 'https://api.keenable.ai/v1/fetch/public'
const publicPage = 'https://93.184.216.34/article'
const originalFetch = globalThis.fetch
const requests: Array<{ url: URL; method: string; body: any; headers: Headers }> = []
let fault = ''

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

const json = (value: unknown): Response => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })

globalThis.fetch = async (input, init) => {
  if (String(input).startsWith('https://news.google.com/rss/search?')) return new Response('<rss><channel><item><title>Default news source</title><link>https://example.org/news</link></item></channel></rss>')
  if (String(input) === publicPage) return new Response('<title>Direct page</title><p>Direct page text.</p>', { headers: { 'Content-Type': 'text/html' } })
  const url = new URL(String(input))
  if (`${url.origin}${url.pathname}` !== searchEndpoint && `${url.origin}${url.pathname}` !== fetchEndpoint) return originalFetch(input, init)
  const headers = new Headers(init?.headers)
  const method = init?.method ?? 'GET'
  requests.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : null, headers })
  assert.equal(init?.redirect, 'error')
  assert.equal(headers.get('Authorization'), null)
  assert.equal(headers.get('X-API-Key'), null)
  assert.equal(headers.get('X-Keenable-Title'), 'TensorFold')
  assert.equal(headers.get('User-Agent'), 'TensorFold/1.0 (+https://github.com/ashhart/TensorFold)')
  if (fault === 'stall') return new Promise((_resolve, reject) => {
    init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true })
  })
  if (fault === 'http') return new Response('{"error":"rate limited"}', { status: 429 })
  if (fault === 'invalid') return new Response('invalid JSON', { headers: { 'Content-Type': 'application/json' } })
  if (fault === 'size') return new Response('x'.repeat(1024 * 1024 + 1))
  if (url.pathname.endsWith('/fetch/public')) {
    assert.equal(method, 'GET')
    if (fault === 'empty') return json({ url: url.searchParams.get('url'), title: '', content: '', description: '' })
    return json({ url: url.searchParams.get('url'), title: 'Fetched article', content: '# Fetched article\n\nThe page as Markdown.', description: '' })
  }
  assert.equal(method, 'POST')
  assert.equal(headers.get('Content-Type'), 'application/json')
  if (fault === 'shape') return json({ results: 'invalid' })
  const results = [
    { title: 'Example source', url: publicPage, description: '', snippet: `Useful public text.\n${'y'.repeat(5000)}`, acquired_at: '2026-11-01T00:00:00Z' },
    { title: 'Another source', url: 'https://example.org/', description: '', snippet: 'Second page text.', acquired_at: '2026-11-01T00:00:00Z' },
  ]
  if (fault === 'private-result') results[0].url = 'http://127.0.0.1/secret'
  return json({ query: requests.at(-1)!.body.query, results: fault === 'empty' ? [] : results })
}

const search = (thread: Thread, extra: Record<string, unknown> = {}, context: any = {}) => executeTool('web_search', { query: 'public research question', provider: 'keenable', ...extra }, { cwd: '', currentThread: thread, ...context })
try {
  process.env.KEENABLE_API_KEY = 'synthetic-ignored-key'
  for (const name of ['web_search', 'web_fetch']) {
    const schema: any = TOOL_SCHEMAS.find((schema) => schema.function?.name === name)
    assert.ok(schema.function.parameters.properties.provider.enum.includes('keenable'), 'the maintained tool schema exposes Keenable')
  }
  const a = await newChat()
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
  assert.equal(requests.length, 0, 'omitted and explicit existing providers never reach Keenable')

  const first = await search(a, { limit: 1 })
  assert.equal(first.ok, true, first.output)
  assert.equal(first.meta, 'Keenable')
  assert.match(first.output, /^Example source\nhttps:\/\/93\.184\.216\.34\/article\nUseful public text\./)
  assert.doesNotMatch(first.output, /Another source/)
  assert.deepEqual(requests.at(-1)!.body, { query: 'public research question', max_results: 1 })
  const both = await search(a, { limit: 50 })
  assert.match(both.output, /Another source\nhttps:\/\/example\.org\/\nSecond page text\./)
  assert.ok(both.output.length < 2000 + 200, 'long snippets are capped so every requested result fits')
  assert.equal(requests.at(-1)!.body.max_results, 8, 'the shared limit cap applies')

  const fetched = await executeTool('web_fetch', { url: publicPage, provider: 'keenable', max_chars: 500 }, { cwd: '', currentThread: a })
  assert.equal(fetched.ok, true, fetched.output)
  assert.equal(fetched.title, 'Fetched article')
  assert.match(fetched.output, /The page as Markdown\./)
  assert.equal(requests.at(-1)!.url.searchParams.get('url'), publicPage)

  fault = 'http'
  assert.match((await search(a)).output, /HTTP 429 \(rate limit reached\)/)
  for (const [kind, message] of [['invalid', /JSON|Unexpected/], ['size', /limit/], ['shape', /invalid search result/], ['private-result', /private|local/]]) {
    fault = kind as string
    assert.match((await search(a)).output, message as RegExp)
  }
  fault = 'empty'
  assert.equal((await search(a)).output, 'No web results found.')
  assert.match((await executeTool('web_fetch', { provider: 'keenable', url: publicPage }, { cwd: '', currentThread: a })).output, /could not extract/)
  fault = ''

  const beforeBlocked = requests.length
  for (const ctx of [{ disabledToolNames: ['web_search'] }, { disabledCapabilityIds: ['tool:web'] }, { networkCapabilityAllowed: false }, { allowedToolNames: [] }, { phoneLocalOnly: true }]) {
    assert.equal((await search(a, {}, ctx)).ok, false)
  }
  assert.equal((await search({ ...a, googleLocalOnly: true })).ok, false)
  assert.equal((await search(a, { provider: 'unknown' })).ok, false)
  assert.equal((await search(a, { query: '' })).ok, false)
  for (const url of ['http://127.0.0.1/secret', 'http://169.254.169.254/latest', 'file:///etc/passwd']) {
    assert.equal((await executeTool('web_fetch', { url, provider: 'keenable' }, { cwd: '', currentThread: a })).ok, false)
  }
  assert.equal(requests.length, beforeBlocked, 'blocked calls never reach the provider')

  fault = 'stall'
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
  fault = ''

  // A scripted model requests the optional provider and consumes the real dispatcher's result on its next turn.
  const modelRequests: any[] = []
  const server = createServer((request, response) => {
    let raw = ''
    request.on('data', (chunk) => { raw += chunk })
    request.on('end', () => {
      modelRequests.push(JSON.parse(raw))
      const delta = modelRequests.length === 1
        ? { tool_calls: [{ index: 0, id: 'search-call', type: 'function', function: { name: 'web_search', arguments: JSON.stringify({ query: 'public research question', provider: 'keenable' }) } }] }
        : { content: 'The public text answers the question.' }
      response.writeHead(200, { 'Content-Type': 'text/event-stream' })
      response.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: modelRequests.length === 1 ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`)
    })
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  try {
    const thread = await newChat()
    thread.apiMessages.push({ role: 'user', content: 'Use Keenable to research this question.' })
    await runAgent({ thread, provider: { id: 'scripted', name: 'Scripted fixture', kind: 'local', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, apiKey: '', models: ['scripted'] }, model: 'scripted', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', signal: new AbortController().signal, persist() {}, autoCompact: false, maxSteps: 3, requestApproval: async () => 'deny', emit() {} })
    assert.equal(modelRequests.length, 2)
    assert.ok(modelRequests[1].messages.some((message: any) => message.role === 'tool' && message.content.includes('Useful public text.')))
    assert.ok(thread.apiMessages.some((message) => message.role === 'assistant' && String(message.content).includes('answers the question')))
  } finally { await new Promise<void>((resolve) => server.close(() => resolve())) }
  console.log('Keenable caller checks passed: dispatch, scripted loop consumption, request shape, permissions, errors, timeout and cancellation.')
} finally {
  globalThis.fetch = originalFetch
  delete process.env.KEENABLE_API_KEY
  mock.timers.reset()
}
