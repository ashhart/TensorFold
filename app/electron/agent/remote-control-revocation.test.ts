import assert from 'node:assert/strict'
import { createServer } from 'node:net'
import { createRemoteControlServer, type RemoteControlServerDeps } from './remote-control'
import { createPairingRegistry } from './pairing-sessions'
import type { RemoteControlSettings } from './types'

async function freePort(): Promise<number> {
  const probe = createServer()
  await new Promise<void>((resolve, reject) => { probe.once('error', reject); probe.listen(0, '127.0.0.1', resolve) })
  const port = (probe.address() as { port: number }).port
  await new Promise<void>((resolve) => probe.close(() => resolve()))
  return port
}

async function fixture(snapshot?: () => Promise<any>) {
  const settings: RemoteControlSettings = {
    enabled: true, serverName: 'Revocation fixture', bindHost: '127.0.0.1', allowLan: false,
    port: await freePort(), authToken: 'fixture-owner-token', lastStartedAt: null, lastClientAt: null, lastError: null,
  }
  const registry = createPairingRegistry()
  const link = registry.issuePairingLink('Fixture phone')
  const device = registry.redeemPairingLink(link.credential).session!
  const server = createRemoteControlServer({
    getSettings: () => settings, getSnapshot: snapshot ?? (() => Promise.resolve({ baseline: true })),
    getPairedSnapshot: snapshot ?? (() => Promise.resolve({ baseline: true })),
    heartbeatIntervalMs: 25, onClientSeen: () => {}, onStarted: () => {}, onError: () => {}, audit: () => {},
    pairing: {
      issuePairingLink: (label: string) => registry.issuePairingLink(label),
      authorizeSessionToken: (token: string) => registry.authorizeSessionToken(token),
      redeemPairingLink: (credential: string) => registry.redeemPairingLink(credential),
      listSessions: () => registry.listSessions(), revokeSession: (id: string) => registry.revokeSession(id),
    },
  } as unknown as RemoteControlServerDeps)
  await server.syncFromSettings()
  return { settings, registry, device, server, base: server.status().url! }
}

async function openStream(base: string, token: string) {
  const controller = new AbortController()
  const response = await fetch(`${base}/v1/events`, {
    headers: { Authorization: `Bearer ${token}` }, signal: controller.signal,
  })
  assert.equal(response.status, 200)
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let wire = ''
  return {
    controller,
    async baseline() {
      while (!wire.includes('event: snapshot')) {
        const next = await reader.read()
        assert.equal(next.done, false)
        wire += decoder.decode(next.value, { stream: true })
      }
    },
    async stoppedWithout(marker: string) {
      let stopped = false
      const timer = setTimeout(() => controller.abort(), 400)
      try {
        while (true) {
          const next = await reader.read()
          if (next.done) { stopped = true; break }
          wire += decoder.decode(next.value, { stream: true })
        }
      } catch {
        stopped = !controller.signal.aborted
      } finally {
        clearTimeout(timer)
        await reader.cancel().catch(() => {})
      }
      assert.ok(!wire.includes(marker), `A revoked stream received ${marker}`)
      assert.ok(stopped, 'A revoked or rotated stream must be disconnected promptly')
    },
  }
}

let failures = 0
const roleBefore = process.env.VONTRA_SECURITY_ROLE
process.env.VONTRA_SECURITY_ROLE = 'owner'

async function check(name: string, run: () => Promise<void>) {
  try { await run(); console.log(`PASS ${name}`) }
  catch (error) { failures++; console.error(`FAIL ${name}`, error) }
}

try {
  for (const route of ['local', 'http', 'registry'] as const) await check(`SSE device revocation via ${route}`, async () => {
    const f = await fixture()
    const stream = await openStream(f.base, f.device.token)
    try {
      await stream.baseline()
      if (route === 'local') assert.equal(await f.server.revokeDeviceSession(f.device.id), true)
      else if (route === 'registry') f.registry.clearAll()
      else {
        const response = await fetch(`${f.base}/v1/auth/sessions/revoke`, {
          method: 'POST', headers: { Authorization: `Bearer ${f.settings.authToken}`, 'content-type': 'application/json' },
          body: JSON.stringify({ sessionId: f.device.id }),
        })
        assert.equal(response.status, 200)
        await response.json()
      }
      f.server.publish({ kind: 'thread', data: { marker: 'after-device-revocation' } })
      await stream.stoppedWithout('after-device-revocation')
      const rejected = await fetch(`${f.base}/v1/snapshot`, { headers: { Authorization: `Bearer ${f.device.token}` } })
      assert.equal(rejected.status, 401)
      await rejected.json()
    } finally { stream.controller.abort(); await f.server.stop() }
  })

  await check('SSE owner token rotation', async () => {
    const f = await fixture()
    const stream = await openStream(f.base, f.settings.authToken)
    try {
      await stream.baseline()
      f.settings.authToken = 'fixture-rotated-owner-token'
      await f.server.syncFromSettings()
      f.server.publish({ kind: 'thread', data: { marker: 'after-owner-rotation' } })
      await stream.stoppedWithout('after-owner-rotation')
    } finally { stream.controller.abort(); await f.server.stop() }
  })

  await check('Revocation while an SSE baseline is being assembled', async () => {
    let finish: (snapshot: any) => void = () => {}
    const pending = new Promise<any>((resolve) => { finish = resolve })
    const f = await fixture(() => pending)
    const stream = await openStream(f.base, f.device.token)
    try {
      await f.server.revokeDeviceSession(f.device.id)
      finish({ marker: 'revoked-pending-baseline' })
      await stream.stoppedWithout('revoked-pending-baseline')
    } finally { finish({}); stream.controller.abort(); await f.server.stop() }
  })

  await check('Revocation while an HTTP snapshot is being assembled', async () => {
    let finish: (snapshot: any) => void = () => {}, entered: () => void = () => {}
    const pending = new Promise<any>((resolve) => { finish = resolve })
    const started = new Promise<void>((resolve) => { entered = resolve })
    const f = await fixture(() => { entered(); return pending })
    try {
      const request = fetch(`${f.base}/v1/snapshot`, { headers: { Authorization: `Bearer ${f.device.token}` } })
      await started
      await f.server.revokeDeviceSession(f.device.id)
      finish({ marker: 'revoked-http-snapshot' })
      const response = await request
      assert.equal(response.status, 401, 'An in-flight read must recheck authorization before releasing private data')
      assert.doesNotMatch(await response.text(), /revoked-http-snapshot/)
    } finally { finish({}); await f.server.stop() }
  })
} finally {
  if (roleBefore === undefined) delete process.env.VONTRA_SECURITY_ROLE
  else process.env.VONTRA_SECURITY_ROLE = roleBefore
}
assert.equal(failures, 0, `${failures} remote revocation regressions failed`)
