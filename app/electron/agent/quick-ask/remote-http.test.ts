import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { once } from 'node:events'
import { handleQuickAskRemote, type QuickAskRemoteBackend } from './remote-http'

let authorized = true, cancelled = false, hold = false, requests = 0
const backend: QuickAskRemoteBackend = {
  routes: async () => [], load: async () => {}, handoff: async () => { throw new Error('not used') },
  generate: async function* (_input, signal) {
    requests++
    try {
      yield { type: 'content', text: 'First token' }
      if (hold) await new Promise<void>(resolve => signal.addEventListener('abort', () => { cancelled = true; resolve() }, { once: true }))
      signal.throwIfAborted()
      yield { type: 'finish' }
    } finally { if (signal.aborted) cancelled = true }
  },
}
const server = createServer((req, res) => {
  void handleQuickAskRemote(req, res, req.url!, req.headers.authorization === 'Bearer synthetic-owner', () => authorized, backend).catch(() => { if (!res.headersSent) res.writeHead(400); res.end() })
})
server.listen(0, '127.0.0.1'); await once(server, 'listening')
const base = `http://127.0.0.1:${(server.address() as any).port}`
try {
  let result = await fetch(`${base}/v1/ask/routes`)
  assert.equal(result.status, 403); await result.text()
  result = await fetch(`${base}/v1/ask/routes`, { headers: { authorization: 'Bearer synthetic-owner' } })
  assert.equal(result.status, 200); assert.deepEqual(await result.json(), [])
  result = await fetch(`${base}/v1/ask/stream`, { method: 'POST', headers: { authorization: 'Bearer synthetic-owner' }, body: '{}' })
  assert.match(result.headers.get('content-type')!, /ndjson/)
  const events = (await result.text()).trim().split('\n').map(line => JSON.parse(line))
  assert.equal(events[0].text, 'First token'); assert.equal(events.at(-1).type, 'finish')
  hold = true
  const controller = new AbortController()
  result = await fetch(`${base}/v1/ask/stream`, { method: 'POST', headers: { authorization: 'Bearer synthetic-owner' }, body: '{}', signal: controller.signal })
  await result.body!.getReader().read(); controller.abort()
  for (let i = 0; i < 100 && !cancelled; i++) await new Promise(resolve => setTimeout(resolve, 5))
  assert.equal(cancelled, true, 'Disconnecting a client cancels host generation')
  authorized = false
  result = await fetch(`${base}/v1/ask/stream`, { method: 'POST', headers: { authorization: 'Bearer synthetic-owner' }, body: '{}' })
  assert.equal(result.status, 401); await result.text(); assert.equal(requests, 2)
} finally { server.closeAllConnections(); server.close(); await once(server, 'close') }
console.log('Quick Ask remote owner authentication, streaming and disconnect cancellation checks passed')
