import assert from 'node:assert/strict'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { runAgent } from './loop'
import { PluginRuntime } from './plugin-runtime'
import type { Thread } from './types'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-private-hook-fixture-'))
const entry = path.join(home, 'fixture.cjs')
await writeFile(path.join(home, 'vontra-plugin.json'), JSON.stringify({ id: 'synthetic-hook', name: 'Synthetic hook fixture', version: '1', main: 'fixture.cjs', trustLevel: 'unknown' }))
// This is an authored inert test plugin; no third-party code or user plugin is loaded.
await writeFile(entry, `const seen=[]; module.exports.seen=seen; module.exports.register=ctx=>{ctx.onAgentEvent(event=>seen.push('event'));ctx.onSessionStart(id=>seen.push('start'));ctx.onSessionEnd(id=>seen.push('end'));ctx.onPreToolCall(async()=>{seen.push('pre')});ctx.onPostToolCall(async()=>{seen.push('post')});ctx.registerTool({type:'function',function:{name:'synthetic_probe',description:'Inert test probe',parameters:{type:'object',properties:{}}}},async()=>{seen.push('tool');return {ok:true,output:'Synthetic result'}})};`)
const plugin = new PluginRuntime()
const loaded = await plugin.loadPlugin(home)
assert.equal(loaded.ok, true, loaded.error)
const seen: string[] = createRequire(import.meta.url)(entry).seen
const originalFetch = globalThis.fetch
let requestNumber = 0
let harvested = 0
const provider = { id: 'synthetic', name: 'Synthetic local API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9898/v1', apiKey: '', models: ['synthetic'] }
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  assert.ok(String(input?.url ?? input).startsWith(provider.baseUrl), 'no unexpected network route')
  const body = JSON.parse(String(init?.body ?? '{}'))
  if (!body.tools?.length) return new Response(`data: ${JSON.stringify({choices:[{delta:{content:'Synthetic bounded history summary.'},finish_reason:'stop'}]})}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
  const delta = requestNumber++ === 0 ? { tool_calls: [{ index: 0, id: 'synthetic-call', type: 'function', function: { name: 'synthetic_probe', arguments: '{}' } }] } : { content: 'Synthetic run complete.' }
  return new Response(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: 'tool_calls' in delta ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch
async function scenario(incognito: boolean, compact = false) {
  seen.length = 0; requestNumber = 0; harvested = 0
  const thread: Thread = { id: incognito ? 'private' : 'normal', title: 'Synthetic fixture', kind: 'chat', cwd: home, incognito, providerId: provider.id, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', timeline: compact ? [{ id: 'authored-source', kind: 'user', memorySource: 'user', text: 'I always prefer lavenderwidget for synthetic demos.' }] : [], apiMessages: compact ? Array.from({length:48}, (_,i)=>({role:i%2?'assistant' as const:'user' as const,content:i===0?'I always prefer lavenderwidget for synthetic demos.':`Synthetic history ${i} ${'bounded context '.repeat(30)}`})) : [{ role: 'user', content: 'Inspect this synthetic fixture.' }] }
  await runAgent({ thread, provider, model: 'synthetic', approvalMode: 'auto', temperature: 0, reasoningEffort: 'low', signal: new AbortController().signal, persist() {}, emit() {}, requestApproval: async () => 'allow', autoCompact: compact, contextLimit:1500, onMemoryHarvest: (candidates) => { harvested += candidates.length }, maxSteps: 3, pluginRuntime: plugin })
  await new Promise((resolve) => setTimeout(resolve, 10))
  return [...seen]
}
try {
  assert.deepEqual(await scenario(true), [], 'Incognito must not invoke even an injected registered plugin/runtime hook or its executable tool')
  const ordinary = await scenario(false)
  for (const kind of ['start', 'event', 'pre', 'tool', 'post', 'end']) assert.ok(ordinary.includes(kind), `normal ${kind} plugin behavior remains intact`)
  assert.deepEqual(await scenario(true, true), [], 'private automatic compaction must also remain outside registered hooks')
  assert.equal(harvested, 0, 'an injected automatic memory-harvest callback cannot retain private history')
  await scenario(false, true)
  assert.ok(harvested > 0, 'ordinary compaction still proposes durable user statements')
  console.log('Actual registered plugin: Incognito refuses injected event/session/pre/post/tool hooks, while ordinary runs preserve all six positive controls.')
} finally {
  globalThis.fetch = originalFetch
  await rm(home, { recursive: true, force: true })
}
