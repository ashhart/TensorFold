import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { runAgent, compactThread } from '../loop'
import { PluginRuntime } from '../plugin-runtime'
import { PhoneService } from './service'
import type { Thread } from '../types'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-phone-hook-isolation-'))
const entry = path.join(home, 'fixture.cjs')
await writeFile(path.join(home, 'vontra-plugin.json'), JSON.stringify({ id: 'synthetic-phone-hook', name: 'Synthetic phone hook', version: '1', main: 'fixture.cjs', trustLevel: 'unknown' }))
await writeFile(entry, `const seen=[];module.exports.seen=seen;module.exports.register=ctx=>{ctx.onAgentEvent(value=>seen.push({kind:'event',value}));ctx.onSessionStart(value=>seen.push({kind:'start',value}));ctx.onSessionEnd(value=>seen.push({kind:'end',value}));ctx.onPreToolCall(async(name,args)=>{seen.push({kind:'pre',value:{name,args}})});ctx.onPostToolCall(async(name,args,result)=>{seen.push({kind:'post',value:{name,args,result}})})};`)
const runtime = new PluginRuntime(), loaded = await runtime.loadPlugin(home)
assert.equal(loaded.ok, true, loaded.error)
const seen: any[] = createRequire(import.meta.url)(entry).seen
const originalFetch = globalThis.fetch
const marker = 'OWNED_SYNTHETIC_PHONE_HOOK_AMBER'
const provider = { id: 'phone-hook-local', name: 'Synthetic local API', kind: 'local' as const, transport: 'chat-completions' as const, baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['fixture'] }
const device = { id: 'android:owned-fixture', name: 'Synthetic phone', platform: 'android' as const, transport: 'usb' as const, state: 'ready' as const, detail: null }
let request = 0, dynamic = false, observations = 0, harvests = 0
let activeThread!: Thread
const backend = { observe: async () => { observations++; if (dynamic) activeThread.phoneLocalOnly = true; return { width: 300, height: 600, image: null, app: 'example.synthetic', warnings: [], nodes: [{ text: marker, description: '', role: 'Button', bounds: { x: 0, y: 0, width: 100, height: 100 }, editable: false, secure: false, enabled: true, focused: false, interactive: true }] } }, act: async () => { throw new Error('No device action is requested.') }, disconnect: async () => {} }
const service = new PhoneService({ discover: async () => ({ adb: { available: false, path: null, version: null }, devices: [device], mirroring: { available: false, detail: null } }), connect: async () => backend, connectDeveloper: async () => ({ device, backend }), authorizeThread() {}, cancelThread() {}, emit() {} })
globalThis.fetch = (async (url: any) => {
  assert.ok(String(url?.url ?? url).startsWith(provider.baseUrl), 'all requests remain inert fixture requests')
  const delta = request++ === 0 ? { tool_calls: [{ index: 0, id: 'synthetic-phone-observe', type: 'function', function: { name: 'phone_observe', arguments: '{}' } }] } : { content: 'Synthetic phone observation complete.' }
  return new Response(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: 'tool_calls' in delta ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
}) as typeof fetch
async function scenario(phoneLocalOnly: boolean, flip = false, incognito = false) {
  seen.length = 0; request = 0; dynamic = flip
  const ownEvents: any[] = [], observed: any[] = [], audited: any[] = []
  activeThread = { id: 'synthetic-phone-owner', title: 'Synthetic existing chat', kind: 'chat', cwd: home, providerId: provider.id, model: 'fixture', phoneLocalOnly, incognito, createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [{ role: 'user', content: 'Inspect the explicitly granted synthetic phone once.' }] }
  await service.stop(); service.grant(activeThread.id)
  const before = observations
  await runAgent({ thread: activeThread, provider, model: 'fixture', approvalMode: 'suggest', temperature: 0, reasoningEffort: 'low', autoCompact: false, maxSteps: 3, signal: new AbortController().signal, persist() {}, emit: (event) => ownEvents.push(event), observe: (kind, value) => observed.push({ kind, value }), auditToolCall: (value) => { audited.push(value) }, requestApproval: async () => 'allow', pluginRuntime: runtime, onMemoryHarvest: () => { harvests++ }, toolContext: { cwd: home, backend: { kind: 'ssh' } as any, currentThread: activeThread, phoneLocalOnly, phoneControlAvailable: true, phonePrivacyGuard: () => null, runPhoneControl: (input) => service.tool(input, activeThread.id) } })
  await new Promise((resolve) => setTimeout(resolve, 15))
  assert.equal(observations, before + 1)
  assert.ok(JSON.stringify(ownEvents).includes(marker)); assert.ok(JSON.stringify(activeThread.apiMessages).includes(marker), 'own native timeline/model context remains usable')
  if (phoneLocalOnly || flip || incognito) {
    assert.equal(JSON.stringify(seen).includes(marker), false, 'automatic plugin hooks never receive phone-derived output, including a mid-run privacy transition')
    assert.equal(JSON.stringify(observed).includes(marker), false)
    assert.equal(JSON.stringify(audited).includes(marker), false)
    if (phoneLocalOnly || incognito) { assert.equal(seen.length, 0); assert.equal(observed.length, 0); assert.equal(audited.length, 0) }
  } else {
    assert.ok(JSON.stringify(seen).includes(marker), 'ordinary registered plugin positive remains active')
    assert.ok(JSON.stringify(audited).includes(marker))
    for (const kind of ['start', 'event', 'pre', 'post', 'end']) assert.ok(seen.some((item) => item.kind === kind))
  }
}
try {
  await service.refresh(); await service.connect({ deviceId: device.id })
  await scenario(true); await scenario(false, true); await scenario(true, false, true); await scenario(false)
  const statement = 'I always prefer synthetic phone context for this fixture.'
  const thread = { ...activeThread, phoneLocalOnly: true, timeline: [{ id: 'direct', kind: 'user', memorySource: 'user', text: statement }], apiMessages: Array.from({ length: 80 }, (_, n) => [{ role: 'user', content: n ? `Synthetic user ${n}. ` + 'Context. '.repeat(80) : statement }, { role: 'assistant', content: `Synthetic response ${n}. ` + 'Response. '.repeat(80) }]).flat() } as Thread
  const before = request
  assert.equal((await compactThread(thread, provider, 'fixture', new AbortController().signal, 'manual', () => { harvests++ }))?.summaryMethod, 'deterministic-fallback')
  assert.equal(harvests, 0); assert.equal(request, before, 'phone compaction neither exports a harvest callback nor makes auxiliary model requests')
  console.log('Actual PhoneService/registered plugins passed: initial/dynamic/Incognito phone hook isolation, observer/audit suppression, retained native history, no automatic harvest/model helper and ordinary hook positives.')
} finally { globalThis.fetch = originalFetch; await service.shutdown(); await rm(home, { recursive: true, force: true }) }
