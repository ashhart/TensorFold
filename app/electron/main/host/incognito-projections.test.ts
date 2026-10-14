import assert from 'node:assert/strict'
import { createServer } from 'node:net'
import { store } from '../state'
import { createEmitHelpers } from './emit-helpers'
import { isRemoteEventVisible, assertRemoteThreadVisible, assertRemoteApprovalVisible } from './remote-snapshot'
import { createRunAttention } from './run-attention'
import { compactWebhookData } from './webhooks'
import { createRemoteControlServer } from '../../agent/remote-control'

const marker = 'incognito-synthetic-projection-marker'
const privateThread: any = { id: 'private-thread', incognito: true, title: marker, timeline: [], apiMessages: [] }
store.threads = [privateThread, { id: 'normal-thread', title: 'Normal example', timeline: [], apiMessages: [] } as any]
store.jobs = [{ id: 'private-job', kind: 'thread', owner: privateThread.id, threadId: privateThread.id, incognito: true, pendingApprovals: [{ callId: 'private-call', resolvedAt: null }] } as any]
store.webhooks = []
const rendered: any[] = [], remote: any[] = [], observed: any[] = [], spined: any[] = [], attention: any[] = []
store.win = { isDestroyed: () => false, webContents: { isDestroyed: () => false, send: (_channel: string, payload: any) => rendered.push(payload) } } as any
const emitter = createEmitHelpers({ remoteServer: () => ({ publish: (event) => remote.push(event) }), spineHost: () => ({ record: (event: any) => spined.push(event) }) as any, observerHub: () => ({ emit: (...args: any[]) => observed.push(args) }) as any, persistMissions() {}, onAgentEvent: (event) => attention.push(event) })
const privateEvent: any = { type: 'item_add', threadId: privateThread.id, runId: 'run-example', item: { id: 'item-example', kind: 'tool', title: marker, path: marker, meta: marker, text: marker, args: { content: marker }, question: { answer: marker } } }
assert.match(JSON.stringify(compactWebhookData('agent', privateEvent)), new RegExp(marker), 'webhook metadata can contain derived private content even though transcript fields are compacted away')
emitter.emitToRenderer(privateEvent)
emitter.emitJob(store.jobs[0])
assert.equal(rendered.length, 2, 'private live state remains usable in the owner desktop window')
assert.deepEqual([remote.length, observed.length, spined.length, attention.length], [0, 0, 0, 0], 'private events do not reach remote, webhook/observer bridges, durable spine or native attention')
emitter.emitToRenderer({ ...privateEvent, threadId: 'normal-thread', type: 'done', item: undefined })
assert.equal(remote.length, 1, 'ordinary chat projections still work')
assert.equal(attention.length, 1)

const notices: any[] = [], badges: string[] = []
const native = createRunAttention({ isIncognito: (id) => id === privateThread.id, enabled: () => true, focused: () => false, supported: () => true, threadTitle: () => marker, notify: (notice) => notices.push(notice), setBadge: (text) => badges.push(text) })
native.approvalPending({ threadId: privateThread.id, tool: marker })
native.observe({ threadId: privateThread.id, runId: 'run', type: 'error', message: marker })
assert.deepEqual([notices, badges], [[], []], 'Incognito title/errors never enter the OS notification history')

assert.throws(() => assertRemoteThreadVisible(privateThread.id), (error: any) => error.code === 404)
assert.throws(() => assertRemoteApprovalVisible('private-call'), (error: any) => error.code === 404)
assert.doesNotThrow(() => assertRemoteThreadVisible('normal-thread'))

const port = await new Promise<number>((resolve, reject) => { const probe = createServer(); probe.once('error', reject); probe.listen(0, '127.0.0.1', () => { const address = probe.address(); probe.close(() => resolve(typeof address === 'object' && address ? address.port : 0)) }) })
let rejectVisibility = false
const published: string[] = []
let publicCalls = 0
const server = createRemoteControlServer({ getSettings: () => ({ enabled: true, bindHost: '127.0.0.1', port, authToken: 'example-test-token', serverName: 'Example host' }), getSnapshot: () => ({ threads: [] }), getArchives: () => ({ threads: [], projects: [], rooms: [], tickets: [] }), getPairedArchives: () => ({ threads: [], projects: [], rooms: [], tickets: [] }), startThread: async (payload: any) => { assertRemoteThreadVisible(payload.threadId); publicCalls++; return { runId: 'run' } }, isPublishVisible: (event: any) => { if (rejectVisibility) throw new Error('Example unavailable privacy state'); return isRemoteEventVisible(event) }, observePublish: (kind: string) => published.push(kind), onClientSeen() {}, onStarted() {}, onError() {}, audit() {}, heartbeatIntervalMs: 20 } as any)
const controller = new AbortController()
let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
try {
  await server.syncFromSettings()
  const headers = { Authorization: 'Bearer example-test-token', 'Content-Type': 'application/json' }
  const start = (threadId: string) => fetch(`${server.status().url}/v1/threads/${threadId}/start`, { method: 'POST', headers, body: JSON.stringify({ userText: 'Synthetic instruction' }) })
  assert.equal((await start(privateThread.id)).status, 404, 'actual authenticated remote route refuses Incognito IDs before execution')
  assert.equal((await start('normal-thread')).status, 202)
  assert.equal(publicCalls, 1)
  server.publish({ kind: 'agent', data: privateEvent })
  server.publish({ kind: 'agent', data: { threadId: 'normal-thread', text: 'public-replay-marker' } })
  rejectVisibility = true; server.publish({ kind: 'thread', data: { title: marker } }); rejectVisibility = false
  const response = await fetch(`${server.status().url}/v1/events?since=0`, { headers: { Authorization: 'Bearer example-test-token' }, signal: controller.signal })
  reader = response.body!.getReader()
  const decoder = new TextDecoder(); let wire = ''
  const readThrough = async (until: string) => {
    const timeout = setTimeout(() => controller.abort(), 2000)
    try { while (!wire.includes(until)) { const chunk = await reader!.read(); if (chunk.done) break; wire += decoder.decode(chunk.value, { stream: true }) } } finally { clearTimeout(timeout) }
  }
  await readThrough('public-replay-marker')
  server.publish({ kind: 'agent', data: privateEvent })
  server.publish({ kind: 'agent', data: { threadId: 'normal-thread', text: 'public-live-marker' } })
  await readThrough('public-live-marker')
  assert.doesNotMatch(wire, new RegExp(marker), 'actual authenticated owner SSE wire excludes private replay and live payloads')
  assert.equal(published.length, 2, 'privacy rejection happens before observers and replay allocation, including a thrown gate')
} finally { await reader?.cancel().catch(() => {}); controller.abort(); await server.stop(); store.win = null }
console.log('Incognito local-only live projections, remote snapshots/routes, notifications and actual SSE replay/live gate passed')
