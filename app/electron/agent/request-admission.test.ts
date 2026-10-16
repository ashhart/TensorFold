import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { streamChat, quickComplete } from './openai'
import { compactThread, runAgent } from './loop'
import { setEngineServeHook } from './engine/provider'
import { createLocalProviderFetch } from './secure-fetch'
import type { Provider, Thread } from './types'

let requests = 0, releases = 0, blocked = false, guards = 0, summary = false, summaryRequests = 0
const server = createServer((request, response) => {
  let raw = ''
  request.on('data', (chunk) => { raw += chunk })
  request.on('end', () => {
    requests++
    const body = JSON.parse(raw)
    if (!body.stream) { response.setHeader('content-type', 'application/json'); response.end(JSON.stringify({ choices: [{ message: { content: 'Synthetic title' }, finish_reason: 'stop' }] })); return }
    const content = summary ? ++summaryRequests % 2 ? 'Synthetic factual compaction summary.' : '{"valid":true}' : 'Synthetic admitted response.'
    response.setHeader('content-type', 'text/event-stream')
    response.end(`data: ${JSON.stringify({ choices: [{ delta: { content }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
const baseUrl = `http://127.0.0.1:${(server.address() as any).port}/v1`
const provider: Provider = { id: 'tensorfold', source: 'tensorfold', name: 'Synthetic engine route', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1/v1', apiKey: '', models: ['fixture'] }
const guard = () => { guards++; if (blocked) throw new Error('Synthetic canonical owner no longer permits this request.') }
const params = () => ({ provider, model: 'fixture', messages: [{ role: 'user' as const, content: 'OWNED_SYNTHETIC_REQUEST_AMBER' }], assertRequestAllowed: guard, signal: new AbortController().signal })
const consume = async () => { for await (const _chunk of streamChat(params())) { /* actual admitted stream */ } }
const make = (): Thread => ({ id: 'ordinary-helper', title: 'Synthetic ordinary history', kind: 'chat', cwd: '', providerId: provider.id, model: 'fixture', createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [{ role: 'system', content: 'Synthetic context.' }, ...Array.from({ length: 80 }, (_, n) => [{ role: 'user' as const, content: `Synthetic request ${n}. ` + 'Context. '.repeat(80) }, { role: 'assistant' as const, content: `Synthetic answer ${n}. ` + 'Response. '.repeat(80) }]).flat()] })
try {
  const flippingHook = async () => { await Promise.resolve(); blocked = true; return { model: 'fixture', baseUrl, apiKey: '', release: () => { releases++ }, assertRequestAllowed: () => {} } }
  setEngineServeHook(flippingHook)
  for (const invoke of [consume, () => quickComplete(params())]) {
    blocked = false
    const before = requests, beforeRelease = releases
    await assert.rejects(invoke, /canonical owner/)
    assert.equal(requests, before, 'the host guard is checked after awaited engine routing before any request')
    assert.equal(releases, beforeRelease + 1, 'a refused request releases its engine lease')
  }
  blocked = false
  const thread = make(), beforeCompact = requests
  await assert.rejects(() => compactThread(thread, provider, 'fixture', new AbortController().signal, 'manual', undefined, undefined, guard), /canonical owner/)
  assert.equal(requests, beforeCompact); assert.equal(thread.compactions?.length ?? 0, 0, 'helper privacy refusal is not swallowed into a successful model summary')
  blocked = false
  const events: any[] = [], primary = { ...make(), apiMessages: [{ role: 'user' as const, content: 'Synthetic ordinary assignment.' }] }
  await runAgent({ thread: primary, provider, model: 'fixture', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', autoCompact: false, maxSteps: 1, signal: new AbortController().signal, persist() {}, requestApproval: async () => 'deny', emit: (event) => events.push(event), assertRequestAllowed: guard })
  assert.equal(requests, beforeCompact); assert.ok(events.some((event) => event.type === 'error' && /canonical owner/.test(event.message)))
  blocked = false
  const recallEvents: any[] = []
  await runAgent({ thread: { ...primary, timeline: [], apiMessages: [{ role: 'user', content: 'Synthetic already returned recall.' }] }, provider, model: 'fixture', approvalMode: 'readonly', temperature: 0, reasoningEffort: 'low', autoCompact: false, maxSteps: 1, signal: new AbortController().signal, persist() {}, requestApproval: async () => 'deny', emit: (event) => recallEvents.push(event), toolContext: { cwd: '', assertRecallSourcesAllowed: guard } })
  assert.equal(requests, beforeCompact); assert.ok(recallEvents.some((event) => event.type === 'error' && /canonical owner/.test(event.message)), 'source-aware recall guard applies to ordinary primary requests too')
  blocked = false
  const localFetch = createLocalProviderFetch(baseUrl, undefined, () => { throw new Error('Synthetic late DNS admission refusal.') })
  await assert.rejects(() => localFetch(`${baseUrl}/chat/completions`, { method: 'POST', body: '{}' }), /late DNS admission/)
  assert.equal(requests, beforeCompact, 'secure fetch preserves pinning and checks admission after asynchronous address resolution')
  setEngineServeHook(async () => ({ model: 'fixture', baseUrl, apiKey: '', release: () => { releases++ } }))
  blocked = false
  const normal = requests
  await consume(); assert.equal(await quickComplete(params()), 'Synthetic title')
  assert.equal(requests, normal + 2, 'ordinary streaming and nonstreaming requests remain available under a valid host guard')
  summary = true
  const summaryBefore = requests
  assert.equal((await compactThread(make(), provider, 'fixture', new AbortController().signal, 'manual', undefined, undefined, guard))?.summaryMethod, 'verified-model')
  assert.equal(requests, summaryBefore + 2, 'ordinary guarded helper paths preserve both summary and verification requests')
  assert.ok(guards > 0)
  console.log('Actual transport admission passed: captured host/source-recall guards after engine await and DNS pinning, released leases, primary/quick/compaction refusal before requests, no swallowed helper refusal and ordinary model/helper positives.')
} finally { setEngineServeHook(null); server.closeAllConnections(); await new Promise<void>((resolve) => server.close(() => resolve())) }
