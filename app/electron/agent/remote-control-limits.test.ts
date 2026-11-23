import assert from 'node:assert/strict'
import { connect } from 'node:net'
import { createRemoteControlServer } from './remote-control'
import { freePort, ok2, remoteDeps, settings } from './remote-control-test-fixture'

// --- unauthenticated request-body caps ---
await (async () => {
  let redeems = 0
  const s = createRemoteControlServer(remoteDeps({ ...settings, port: await freePort() }, {
    pairing: {
      issuePairingLink: async () => ({ credential: 'unused', expiresAt: 1 }),
      authorizeSessionToken: () => ({ ok: false }),
      redeemPairingLink: async () => { redeems += 1; throw new Error('must not redeem') },
      listSessions: () => [],
      revokeSession: async () => false,
    },
  }))
  await s.syncFromSettings()
  try {
    const atCap = await fetch(`${s.status().url}/v1/auth/pair`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: 'A'.repeat(64 * 1024),
    })
    assert.equal(atCap.status, 400, 'the exact cap reaches JSON parsing')
    for (const route of ['/v1/auth/pair', '/v1/auth/pair/session']) {
      const response = await fetch(`${s.status().url}${route}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: Buffer.alloc(20 * 1024 * 1024, 0x41),
      })
      assert.equal(response.status, 413)
      assert.match(await response.text(), /payload too large/i)
    }
    const socket = connect(s.status().port!, '127.0.0.1')
    const closed = new Promise<void>((resolve) => { socket.once('close', () => resolve()) })
    let data = ''
    socket.on('data', (chunk) => { data += chunk.toString('utf8') })
    socket.on('error', (error: NodeJS.ErrnoException) => {
      assert.ok(error.code === 'EPIPE' || error.code === 'ECONNRESET', `unexpected socket error: ${error}`)
    })
    socket.on('connect', () => {
      socket.write('POST /v1/auth/pair HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n')
      // Cross the 64 KiB cap without a 20 MiB client write race discarding the response.
      for (let i = 0; i < 3; i++) socket.write('8000\r\n' + 'A'.repeat(32768) + '\r\n')
      socket.write('0\r\n\r\n')
    })
    await closed
    assert.match(data, /^HTTP\/1\.1 413/)
    assert.equal(redeems, 0, 'oversized unauthenticated requests never reach pairing redemption')
    ok2('unauthenticated pairing bodies are capped before JSON parsing or token redemption')
  } finally {
    await s.stop()
  }
})()
