// Run: jiti electron/agent/mail-network.test.ts
import assert from 'node:assert/strict'
import { createServer } from 'node:net'
import type { AddressInfo, Socket } from 'node:net'
import { createPinnedSocketFactory, mailPinnedLookup, resolvePublicMailHost } from './mail-network.ts'
import { resolvePinnedHost } from './pinned-fetch.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }
console.log('mail-network tests:')

{
  const target = await resolvePublicMailHost('mail.example.test', 'IMAP', async () => [
    { address: '93.184.216.34', family: 4 },
    { address: '2606:2800:220:1:248:1893:25c8:1946', family: 6 },
  ])
  assert.equal(target.hostname, 'mail.example.test')
  await assert.rejects(
    () => resolvePublicMailHost('mail.example.test', 'SMTP', async () => [{ address: '198.18.0.1', family: 4 }]),
    /private or non-public/i,
  )
  await assert.rejects(
    () => resolvePublicMailHost('mail.example.test', 'SMTP', async () => [{ address: 'fd00:ec2::254', family: 6 }]),
    /private or non-public/i,
  )
  assert.equal(typeof mailPinnedLookup(target), 'function')
  ok('mail resolution vets every answer and rejects non-public/metadata targets')
}

{
  // The server ends the connection right after greeting; the client then destroys its
  // socket, which can surface an asynchronous ECONNRESET on either end. Attach no-op
  // 'error' handlers so the reset is observed rather than crashing the process as an
  // unhandled 'error' event.
  const server = createServer((socket) => { socket.on('error', () => {}); socket.end('hello') })
  server.on('connection', (s) => s.on('error', () => {}))
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const port = (server.address() as AddressInfo).port
  let resolverCalls = 0
  const target = await resolvePinnedHost(`https://mail-rebind.invalid:${port}/`, () => true, {
    lookup: async () => {
      resolverCalls++
      return resolverCalls === 1
        ? [{ address: '127.0.0.1', family: 4 }]
        : [{ address: '169.254.169.254', family: 4 }]
    },
  })
  try {
    const socket = await new Promise<Socket>((resolve, reject) => {
      createPinnedSocketFactory(target, port, 2_000)({}, (error, result) => {
        if (error) reject(error)
        else if (!result) reject(new Error('missing connected socket'))
        else resolve(result.connection)
      })
    })
    socket.on('error', () => {})
    socket.destroy()
    assert.equal(resolverCalls, 1, 'the raw socket used only the vetted resolution')
    ok('SMTP-style raw socket preconnection closes the DNS rebinding window')
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
  }
}

console.log(`mail-network: ${passed} checks passed`)
