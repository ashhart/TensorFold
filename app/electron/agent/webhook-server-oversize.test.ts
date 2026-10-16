// Run: jiti electron/agent/webhook-server-oversize.test.ts
//
// Integration test for the socket-layer fix in webhook-server.ts: an oversized POST body
// must return the documented HTTP 413 instead of resetting the socket (ECONNRESET). The
// bug was that req.destroy() in the 'data' handler suppressed the 'end' event, so the only
// code path that wrote the 413 response never ran. We start the real loopback server and POST
// a body past MAX_BODY_BYTES, asserting we get a 413 status line — not a bare connection reset.
import assert from 'node:assert/strict'
import net from 'node:net'
import { createWebhookServer } from './webhook-server.ts'
import type { WebhookEndpoint } from './types.ts'

const MAX_BODY_BYTES = 1_000_000 // mirrors webhook-server.ts

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function endpoint(): WebhookEndpoint {
  return { id: 'wh1', name: 'Test', direction: 'inbound', enabled: true, scope: {}, path: '/hooks/test', secretRef: 'secret', signatureMode: 'hmac-sha256', replayWindowSeconds: 300, capturePolicy: 'redacted_payload', triggerPolicy: 'notify', runAutomationId: null, createdAt: 0, updatedAt: 0 }
}

async function main() {
  console.log('webhook-server oversize tests:')

  const server = createWebhookServer({
    getEndpoints: () => [endpoint()],
    onConnectorEvent: () => {},
    runAutomationById: () => {},
    audit: () => {},
  })
  await server.syncFromEndpoints()
  const { port } = server.status()
  assert.ok(port, 'server is listening on a loopback port')

  // Drive a raw TCP socket so we read the server's response independently of any write-side
  // error. We POST a body just over MAX_BODY_BYTES (small enough to flush in one write, so the
  // client isn't blocked mid-upload when the response comes back). The server's size guard must
  // trip and write a real HTTP 413 status line (the bug: it called req.destroy() with no
  // response, so the client saw a bare socket reset / no status instead).
  const responseText = await new Promise<string>((resolve) => {
    const sock = net.connect(port!, '127.0.0.1')
    let buf = ''
    let settled = false
    const finish = () => { if (settled) return; settled = true; resolve(buf) }
    sock.on('error', () => { /* swallow EPIPE/ECONNRESET on the write side */ })
    sock.on('data', (d) => { buf += d.toString('utf8'); if (buf.includes('\r\n\r\n')) finish() })
    sock.on('close', finish)
    sock.on('connect', () => {
      const total = MAX_BODY_BYTES + 50_000 // just over the 1 MB cap
      sock.write(`POST /hooks/test HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: ${total}\r\nConnection: close\r\n\r\n`)
      sock.write(Buffer.alloc(total, 0x61))
    })
    setTimeout(finish, 3000)
  })

  await server.stop()

  assert.ok(!/^$/.test(responseText) && responseText.startsWith('HTTP/'), `server sent an HTTP response (got: ${JSON.stringify(responseText.slice(0, 40))})`)
  assert.ok(/^HTTP\/1\.\d 413\b/.test(responseText), `oversized body returns the documented HTTP 413, not a bare socket reset (got status line: ${JSON.stringify(responseText.split('\r\n')[0])})`)
  assert.ok(responseText.includes('payload_too_large'), 'the 413 carries the payload_too_large body')
  ok('oversized streamed body returns a real HTTP 413 (not ECONNRESET)')

  console.log(`\n${passed} checks passed.`)
}

main().catch((e) => { console.error(e); process.exit(1) })
