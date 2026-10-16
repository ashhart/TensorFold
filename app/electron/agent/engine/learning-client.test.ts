import assert from 'node:assert/strict'
import http from 'node:http'
import { parseLearningGraph, readLearningGraph, streamLesson } from './learning-client'
import type { LessonEvent } from './learning-client'

async function main() {
  let mode = 'graph'
  let requests = 0
  const server = http.createServer((req, res) => {
    assert.equal(req.headers.authorization, 'Bearer synthetic-local-key')
    requests++
    if (mode === 'graph') { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ next: 1, facts: [] })); return }
    if (mode === 'big-graph') { res.writeHead(200); res.end('x'.repeat(8 * 1024 ** 2 + 1)); return }
    if (mode === 'redirect') { res.writeHead(302, { Location: 'https://example.test/private' }); res.end(); return }
    if (mode === 'refused') { res.writeHead(503); res.end(); return }
    res.writeHead(200, { 'Content-Type': 'text/event-stream' })
    if (mode === 'ok') {
      const content = 'event: fact\r\ndata: {"id":1,"text":"The synthetic kite is violet."}\r\n\r\nevent: learning\ndata: {"id":1}\n\nevent: learned\ndata: {"id":1,"recalled":true}\n\ndata: [DONE]\n\n'
      for (let index = 0; index < content.length; index += 3) res.write(content.slice(index, index + 3))
      res.end(); return
    }
    if (mode === 'malformed') { res.end('event: fact\ndata: invalid\n\n'); return }
    if (mode === 'unknown') { res.end('event: unexpected\ndata: {}\n\n'); return }
    if (mode === 'big-event') { res.end(`event: fact\ndata: ${'x'.repeat(256 * 1024)}\n\n`); return }
    if (mode === 'incomplete') { res.end('event: learning\ndata: {"id":1}\n\n'); return }
    if (mode === 'cancel') { res.write('event: learning\ndata: {"id":1}\n\n'); req.on('close', () => res.destroy()); return }
    res.end()
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  const live = { model: 'synthetic/model', baseUrl: `http://127.0.0.1:${address.port}/v1`, apiKey: 'synthetic-local-key' }
  try {
    assert.deepEqual(await readLearningGraph(live), { next: 1, facts: [] })
    mode = 'big-graph'; await assert.rejects(readLearningGraph(live), /size limit/)
    mode = 'redirect'; await assert.rejects(readLearningGraph(live), /HTTP 302/)
    assert.equal(requests, 3, 'redirects cannot send the API key outside the local engine')
    mode = 'ok'
    const events: LessonEvent[] = []
    await streamLesson(live, 'The synthetic kite is violet.', new AbortController().signal, (event) => events.push(event))
    assert.deepEqual(events.map((event) => event.kind), ['fact', 'learning', 'learned'])
    for (const [next, message] of [['refused', /HTTP 503/], ['malformed', /malformed/], ['unknown', /unknown/], ['big-event', /size limit/], ['incomplete', /before completion/]] as const) {
      mode = next
      await assert.rejects(streamLesson(live, 'Synthetic fact.', new AbortController().signal, () => {}), message)
    }
    mode = 'cancel'
    const controller = new AbortController()
    await assert.rejects(streamLesson(live, 'Synthetic fact.', controller.signal, () => controller.abort()), /aborted|interrupted/i)
    await assert.rejects(streamLesson({ ...live, baseUrl: 'https://example.test/v1' }, 'Synthetic fact.', new AbortController().signal, () => {}), /local engine/)
  } finally { await new Promise<void>((resolve) => server.close(() => resolve())) }
  assert.throws(() => parseLearningGraph({ next: 1, facts: new Array(4097).fill({}) }), /oversized/)
  assert.throws(() => parseLearningGraph({ next: 1, facts: [{ id: 1, at: 1, text: 'Fact.', state: 'invented' }] }), /invalid learning fact/)
  const safe = parseLearningGraph({ next: 1, facts: [], apiKey: 'untrusted', model: { layers: 1, kinds: 'M', privatePath: 'untrusted', pocket: { neurons: 64, layers: [0], used: [0] }, shards: [] } })
  assert.ok(!JSON.stringify(safe).includes('untrusted'), 'only documented graph fields enter the renderer')
  console.log('Living Weights client: authenticated loopback-only requests, bounded JSON/SSE, split frames, malformed/unknown events, cancel and no credential-bearing redirects')
}
main().catch((error) => { console.error(error); process.exit(1) })
