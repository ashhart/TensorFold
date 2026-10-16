import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { fetchRemoteControlSse, readRemoteControlSse, REMOTE_SSE_FRAME_MAX_CHARS } from './remote-sse'

let connectSignal: AbortSignal | undefined
await assert.rejects(
  () => fetchRemoteControlSse((async (_input, init) => {
    connectSignal = init?.signal ?? undefined
    return await new Promise<Response>(() => {})
  }) as typeof fetch, 'https://orchestrator.example/v1/events', {}, 10),
  /did not connect within 10ms; reconnecting/,
)
assert.equal(connectSignal?.aborted, true, 'a pre-header timeout aborts the native fetch request')

const connectController = new AbortController()
const cancelledConnect = fetchRemoteControlSse((async () => await new Promise<Response>(() => {})) as typeof fetch,
  'https://orchestrator.example/v1/events', {}, 1_000, connectController.signal)
connectController.abort()
await assert.rejects(cancelledConnect, (error: any) => error?.name === 'AbortError')

const coreSource = readFileSync(fileURLToPath(new URL('../main/host/ipc/remote-control-ipc.ts', import.meta.url)), 'utf8')
const streamStart = coreSource.indexOf('function ensureRemoteEventStream')
const streamEnd = coreSource.indexOf('\nexport function syncRemoteEventStreams', streamStart)
const streamBody = coreSource.slice(streamStart, streamEnd)
assert.ok(streamStart >= 0 && streamEnd > streamStart, 'native remote-event connection path remains discoverable')
assert.doesNotMatch(streamBody, /await assertRemoteOrchestratorUrl/, 'no unbounded DNS lookup may precede the SSE connection deadline')
assert.ok(
  streamBody.indexOf('base = normalizeRemoteOrchestratorUrl') < streamBody.indexOf('const res = await fetchRemoteControlSse'),
  'lexical URL validation precedes the deadline that encloses secure DNS resolution and connect',
)

const events: any[] = []
await readRemoteControlSse(new Response([
  ': heartbeat',
  '',
  'event: thread',
  'data: {"id":"1","kind":"thread","createdAt":1,"data":{"id":"t"}}',
  '',
  'data: {"id":"2","kind":"job","createdAt":2,"data":{"id":"j"}}',
  '',
  '',
].join('\n'), { headers: { 'content-type': 'text/event-stream; charset=utf-8' } }), (event) => events.push(event))
assert.deepEqual(events.map((event) => event.id), ['1', '2'])

await assert.rejects(
  () => readRemoteControlSse(new Response('{}', { headers: { 'content-type': 'application/json' } }), () => {}),
  /content type/,
)

await assert.rejects(
  () => readRemoteControlSse(new Response(`data: ${'x'.repeat(REMOTE_SSE_FRAME_MAX_CHARS + 1)}`, { headers: { 'content-type': 'text/event-stream' } }), () => {}),
  /safety limit/,
)

const halfOpen = new Response(new ReadableStream<Uint8Array>({ start() {} }), {
  headers: { 'content-type': 'text/event-stream' },
})
await assert.rejects(
  () => readRemoteControlSse(halfOpen, () => {}, REMOTE_SSE_FRAME_MAX_CHARS, 10),
  /idle for 10ms; reconnecting/,
)

const controller = new AbortController()
const cancelled = new Response(new ReadableStream<Uint8Array>({ start() {} }), {
  headers: { 'content-type': 'text/event-stream' },
})
setTimeout(() => controller.abort(), 5)
await assert.rejects(
  () => readRemoteControlSse(cancelled, () => {}, REMOTE_SSE_FRAME_MAX_CHARS, 1_000, controller.signal),
  (error: any) => error?.name === 'AbortError',
)

console.log('remote SSE MIME, frame-bound, idle, and cancellation tests passed')
