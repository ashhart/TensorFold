import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { runAgent, compactThread } from './loop'
import { PluginRuntime } from './plugin-runtime'
import type { Thread } from './types'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-google-hooks-'))
const fixture = path.join(home, 'fixture.cjs')
await writeFile(path.join(home, 'vontra-plugin.json'), JSON.stringify({ id: 'synthetic-google-hook', name: 'Synthetic local hook', version: '1', main: 'fixture.cjs', trustLevel: 'unknown' }))
await writeFile(fixture, `const records=[];module.exports.records=records;module.exports.register=host=>{host.onAgentEvent(value=>records.push({kind:'event',value}));host.onSessionStart(value=>records.push({kind:'start',value}));host.onSessionEnd(value=>records.push({kind:'end',value}));host.onPreToolCall(async(name,args)=>records.push({kind:'pre',value:{name,args}}));host.onPostToolCall(async(name,args,result)=>records.push({kind:'post',value:{name,args,result}}))};`)
const pluginRuntime = new PluginRuntime()
assert.equal((await pluginRuntime.loadPlugin(home)).ok, true)
const records: any[] = createRequire(import.meta.url)(fixture).records
const originalFetch = globalThis.fetch
const provider = { id: 'synthetic-local', name: 'Synthetic local API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['synthetic'] }
const marker = 'SYNTHETIC_PRIVATE_GOOGLE_HOOK_AMBER'
let requests = 0; let harvests = 0
globalThis.fetch = (async () => {
  const delta = requests++ === 0 ? { tool_calls: [{ index: 0, id: 'synthetic-read', type: 'function', function: { name: 'gmail_read', arguments: JSON.stringify({ accountId: 'synthetic-google', id: 'synthetic-message' }) } }] } : { content: 'Synthetic Google review complete.' }
  return new Response(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: 'tool_calls' in delta ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch
try {
  const thread: Thread = { id: 'synthetic-owner', title: 'Existing synthetic chat', kind: 'chat', cwd: home, providerId: provider.id, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Read the explicitly connected synthetic message once.' }], timeline: [] }
  const ownEvents: unknown[] = [], observed: unknown[] = [], audited: unknown[] = []
  await runAgent({ thread, provider, model: 'synthetic', approvalMode: 'suggest', temperature: 0, reasoningEffort: 'low', autoCompact: false, maxSteps: 3, signal: new AbortController().signal, persist() {}, emit: value => ownEvents.push(value), observe: (kind, value) => observed.push({ kind, value }), auditToolCall: value => { audited.push(value) }, requestApproval: async () => 'deny', pluginRuntime, onMemoryHarvest: () => { harvests++ }, toolContext: {
    cwd: home, currentThread: thread, sensitiveRetrievalAllowed: true, googlePrivacyGuard: () => null,
    runGoogleWorkspaceTool: async () => { thread.googleLocalOnly = true; return { ok: true, output: marker } },
  } })
  await new Promise(resolve => setTimeout(resolve, 15))
  assert.equal(requests, 2)
  assert.ok(JSON.stringify(ownEvents).includes(marker), 'the owning local conversation receives its data')
  assert.ok(JSON.stringify(thread.apiMessages).includes(marker))
  assert.equal(JSON.stringify(records).includes(marker), false, 'registered automatic hooks cannot receive first Google result after the marker is set')
  assert.equal(JSON.stringify(observed).includes(marker), false)
  assert.equal(JSON.stringify(audited).includes(marker), false)
  const statement = 'I always prefer synthetic private Google context.'
  const compacted: Thread = { ...thread, timeline: [{ id: 'direct', kind: 'user', memorySource: 'user', text: statement }], apiMessages: Array.from({ length: 80 }, (_, index) => [{ role: 'user' as const, content: index ? `Synthetic ${index}. ` + 'Context. '.repeat(80) : statement }, { role: 'assistant' as const, content: 'Synthetic reply. '.repeat(80) }]).flat() }
  const before = requests
  assert.equal((await compactThread(compacted, provider, 'synthetic', new AbortController().signal, 'manual', () => { harvests++ }))?.summaryMethod, 'deterministic-fallback')
  assert.equal(requests, before, 'Google-derived summaries do not use auxiliary model calls')
  assert.equal(harvests, 0, 'Google-derived history cannot seed automatic memory harvesting')
} finally { globalThis.fetch = originalFetch; await rm(home, { recursive: true, force: true }) }
console.log('Actual Google tool loop keeps owned local output and suppresses automatic hooks, observers, audit content and compaction harvesting')
