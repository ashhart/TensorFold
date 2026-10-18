import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { NativeSession } from './native-session'
import { deferred, until, sleep } from './testing/fixture'

// Exercise the real fetch/stream/idle-ack methods against an HTTP peer, with inert process ownership.
const release = deferred()
let running = 0, disconnected = false, idleChecks = 0, stopped = false
const server = createServer((request, response) => {
  assert.equal(request.headers.authorization, 'Bearer ' + 'e'.repeat(64))
  if (request.url === '/stats') { idleChecks++; response.setHeader('content-type', 'application/json'); response.end(JSON.stringify({ running, waiting: 0 })); return }
  running = 1; request.resume(); response.writeHead(200, { 'content-type': 'text/event-stream' }); response.write('data: {"choices":[{"delta":{"content":"Synthetic"},"finish_reason":null}]}\n\n')
  response.on('close', () => { disconnected = true; void release.promise.then(() => { running = 0 }) })
})
await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
const port = (server.address() as { port: number }).port
const session: NativeSession = Object.assign(Object.create(NativeSession.prototype), { baseUrl: `http://127.0.0.1:${port}`, model: 'synthetic-model', apiKey: 'e'.repeat(64), alive: () => !stopped, stop: async () => { stopped = true } })
try {
  const controller = new AbortController(), stream = await session.request('/v1/chat/completions', { model: 'synthetic-model', messages: [] }, controller.signal)
  let settled = false
  const reading = (async () => { try { for await (const _part of stream.body) { controller.abort() } } finally { settled = true } })()
  void reading.catch(() => {})
  await until(() => disconnected && idleChecks > 0)
  await sleep(30); assert.equal(settled, false, 'A client disconnect cannot release ownership before the native engine reports idle')
  release.resolve(); await assert.rejects(reading, /abort/i)
  assert.equal(settled, true); assert.equal(stopped, false); assert.ok(idleChecks >= 2)
  console.log('PASS native session real HTTP methods: authenticated loopback request, stream abort and physical idle acknowledgement before ownership release, without a spawned engine or model')
} finally { release.resolve(); server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())) }
