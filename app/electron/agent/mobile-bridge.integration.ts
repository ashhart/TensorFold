import assert from 'node:assert/strict'
import { request as httpsRequest } from 'node:https'
import { createServer as portProbe } from 'node:net'
import { checkServerIdentity } from 'node:tls'
import { createHash } from 'node:crypto'
import { mkdtemp, lstat, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createMobileBridge, isMobilePrivateAddress, mobileBridgeInterfaces, mobileBridgeRouteAllowed, validateMobileBridgeConfig } from './mobile-bridge'
import { loadMobileBridgeCertificate } from './mobile-bridge-certificate'
import { createRemoteControlServer, type RemoteControlServerDeps } from './remote-control'
import { createPairingRegistry } from './pairing-sessions'
import type { RemoteControlSettings } from './types'

for (const host of ['10.1.2.3', '172.16.0.2', '172.31.255.254', '192.168.50.2']) assert.equal(isMobilePrivateAddress(host), true)
for (const host of ['127.0.0.1', '0.0.0.0', '::1', '::', '8.8.8.8', '100.64.0.1', '169.254.2.3', '172.15.1.2', '172.32.1.2', '192.168.300.2', 'localhost', '192.168.1.2.example.test']) assert.equal(isMobilePrivateAddress(host), false)
const fictionalInterfaces = [{ name: 'fixture', address: '192.168.50.2' }]
validateMobileBridgeConfig({ version: 1, enabled: true, host: fictionalInterfaces[0].address, port: 8766 }, fictionalInterfaces)
for (const host of ['0.0.0.0', '127.0.0.1', '8.8.8.8', '192.168.50.3']) assert.throws(() => validateMobileBridgeConfig({ version: 1, enabled: true, host, port: 8766 }, fictionalInterfaces))
for (const port of [0, 443, 65536, NaN, 8766.5]) assert.throws(() => validateMobileBridgeConfig({ version: 1, enabled: true, host: fictionalInterfaces[0].address, port }, fictionalInterfaces))
assert(mobileBridgeRouteAllowed('GET', '/v1/snapshot'))
assert(mobileBridgeRouteAllowed('POST', '/v1/threads/task%2Fone/start'))
for (const route of ['/health', '/', '/v1/auth/sessions', '/v1/files', '/v1/terminals', '/v1/engine/start', '/v1/approve', '/v1/auth/issue', '/v1/threads/one/archive']) assert.equal(mobileBridgeRouteAllowed('GET', route) || mobileBridgeRouteAllowed('POST', route), false)

const address = mobileBridgeInterfaces()[0]?.address
if (!address) {
  throw new Error('LAN TLS integration requires an active private IPv4 interface; no network path was tested.')
} else {
  const profile = await mkdtemp(path.join(os.tmpdir(), 'studio-mobile-bridge-test-'))
  const previousRole = process.env.VONTRA_SECURITY_ROLE
  process.env.VONTRA_SECURITY_ROLE = 'owner'
  const registry = createPairingRegistry()
  const ownerToken = 'owner_' + 'a'.repeat(40)
  async function port(host: string) {
    return new Promise<number>(resolve => { const probe = portProbe(); probe.listen(0, host, () => { const value = (probe.address() as { port: number }).port; probe.close(() => resolve(value)) }) })
  }
  const corePort = await port('127.0.0.1'), mobilePort = await port(address)
  const settings: RemoteControlSettings = { enabled: true, serverName: 'Fixture Studio', bindHost: '127.0.0.1', port: corePort, allowLan: false, authToken: ownerToken, lastStartedAt: null, lastClientAt: null, lastError: null }
  const mutations: unknown[] = []
  const snapshot = { serverName: 'Fixture Studio', appVersion: '1.0.0', generatedAt: 1, activeThreadIds: [], threads: [], projects: [], work: { jobs: { running: 0, queued: 0 }, approvals: { pending: 0 } } }
  const core = createRemoteControlServer({
    getSettings: () => settings,
    getSnapshot: () => ({ ...snapshot, privateOwnerField: 'fixture-owner-secret' }),
    getPairedSnapshot: () => snapshot,
    startThread: async (input: unknown) => { mutations.push(input); return { runId: 'fixture-run' } },
    guideThread: async (input: unknown) => { mutations.push(input); return { queued: true, queueDepth: 1 } },
    cancelThread: async (id: string) => { mutations.push(id) },
    pairing: { authorizeSessionToken: (token: string) => registry.authorizeSessionToken(token), redeemPairingLink: (credential: string, clientInfo?: string | null) => registry.redeemPairingLink(credential, clientInfo), issuePairingLink: (label: string) => registry.issuePairingLink(label), listSessions: () => registry.listSessions(), revokeSession: (id: string) => registry.revokeSession(id) },
    onStarted: () => {}, onError: () => {}, onClientSeen: () => {}, audit: () => {},
  } as unknown as RemoteControlServerDeps)
  let badScope = false, ownerAllowed = true
  const bridge = createMobileBridge({
    requireOwner: () => { if (!ownerAllowed) throw new Error('owner_required') },
    loopbackUrl: () => core.status().url,
    ownerToken: () => ownerToken,
    authorizeSessionToken: token => badScope ? { ok: true, sessionId: 'fixture', scope: 'owner' } : registry.authorizeSessionToken(token),
    issuePairingLink: label => core.issuePairingLink(label),
    loadCertificate: host => loadMobileBridgeCertificate(profile, host),
  })
  let cert: Awaited<ReturnType<typeof loadMobileBridgeCertificate>>
  async function call(route: string, options: { method?: string; token?: string; body?: unknown; headers?: Record<string, string>; pin?: string } = {}): Promise<{ status: number; value: any }> {
    const bytes = options.body === undefined ? undefined : Buffer.from(typeof options.body === 'string' ? options.body : JSON.stringify(options.body))
    return new Promise((resolve, reject) => {
      const req = httpsRequest({ hostname: address, port: mobilePort, path: route, method: options.method || (bytes ? 'POST' : 'GET'), ca: cert.cert, agent: false,
        checkServerIdentity(host, peer) {
          const failed = checkServerIdentity(host, peer)
          if (failed) return failed
          const pin = createHash('sha256').update(peer.raw).digest('hex')
          if (pin !== (options.pin || cert.sha256)) return new Error('certificate_pin_mismatch')
        },
        headers: { ...(options.token ? { authorization: `Bearer ${options.token}` } : {}), ...(bytes ? { 'content-type': 'application/json', 'content-length': String(bytes.length) } : {}), ...options.headers } }, res => {
        const chunks: Buffer[] = []
        res.on('data', part => chunks.push(part))
        res.on('end', () => { try { resolve({ status: res.statusCode!, value: JSON.parse(Buffer.concat(chunks).toString('utf8')) }) } catch (error) { reject(error) } })
        res.on('error', reject)
      })
      req.setTimeout(3000, () => req.destroy(new Error('Private LAN TLS request timed out; check executable Local Network access.')))
      req.on('error', reject); req.end(bytes)
    })
  }
  try {
    assert.equal(bridge.status().listening, false, 'Private mobile access is disabled by default')
    ownerAllowed = false
    await assert.rejects(() => bridge.configure({ version: 1, enabled: true, host: address, port: mobilePort }), /owner_required/)
    ownerAllowed = true
    await core.syncFromSettings()
    const enabled = await bridge.configure({ version: 1, enabled: true, host: address, port: mobilePort })
    assert(enabled.listening); assert.equal(core.status().bindHost, '127.0.0.1')
    cert = await loadMobileBridgeCertificate(profile, address)
    assert.equal(cert.sha256, enabled.certificateSha256)
    assert.equal((await lstat(path.join(profile, 'mobile-bridge', 'private-key.pem'))).mode & 0o777, 0o600)
    assert.equal((await loadMobileBridgeCertificate(profile, address)).sha256, cert.sha256, 'Certificate stays stable for this private profile/interface')
    const qr = await bridge.issuePairing('Fixture phone'), parsed = new URL(qr.link), fragment = new URLSearchParams(parsed.hash.slice(1))
    assert.equal(fragment.get('pin'), cert.sha256); assert(parsed.protocol === 'https:'); assert(qr.expiresAt <= Date.now() + 10 * 60_000)
    assert(!qr.link.includes(ownerToken))
    await assert.rejects(() => call('/v1/snapshot', { pin: '0'.repeat(64) }), /certificate_pin_mismatch/)
    assert.equal((await call('/v1/snapshot', { token: ownerToken })).status, 403, 'Shared owner tokens cannot cross the bridge')
    assert.equal((await call('/v1/snapshot', { token: 'tok_' + 'z'.repeat(40) })).status, 401)
    assert.equal((await call('/health')).status, 404, 'Pair claim is the only unauthenticated route')
    const hidden = registry.issuePairingLink('Loopback-only code')
    assert.equal((await call('/v1/auth/pair', { body: { credential: hidden.credential, clientInfo: 'Fixture' } })).value.error, 'unknown_credential', 'Bridge accepts only its own issued nonces')
    const pair = await call('/v1/auth/pair', { body: { credential: fragment.get('pair'), clientInfo: 'Fixture phone' } })
    assert.equal(pair.status, 200)
    const token = pair.value.token as string
    assert.equal((await call('/v1/auth/pair', { body: { credential: fragment.get('pair') } })).value.error, 'already_consumed')
    assert.equal((await call('/v1/snapshot', { token, headers: { origin: 'https://other.example.test' } })).status, 403)
    assert.equal((await call('/v1/snapshot?token=fixture', { token })).status, 403)
    assert.equal((await call('/v1/auth/sessions', { token })).status, 404)
    const visible = await call('/v1/snapshot', { token })
    assert.equal(visible.status, 200); assert.doesNotMatch(JSON.stringify(visible.value), /fixture-owner-secret/)
    badScope = true
    assert.equal((await call('/v1/snapshot', { token })).status, 401)
    badScope = false
    assert.equal((await call('/v1/threads/task%2Fone/start', { token, body: { userText: 'Start this task.' } })).status, 202)
    assert.equal((await call('/v1/threads/task%2Fone/guide', { token, body: { userText: 'Check the docs.' } })).status, 202)
    assert.equal((await call('/v1/threads/task%2Fone/cancel', { token, body: {} })).status, 202)
    assert.equal(mutations.length, 3)
    assert.equal((await call('/v1/threads/one/start', { token, body: { userText: 'x', attachments: [] } })).status, 400)
    assert.equal((await call('/v1/threads/one/start', { token, body: JSON.stringify({ userText: 'x'.repeat(70 * 1024) }) })).status, 413)
    process.env.VONTRA_SECURITY_ROLE = 'viewer'
    assert.equal((await call('/v1/threads/one/start', { token, body: { userText: 'Cannot bypass role' } })).status, 403)
    assert.equal(mutations.length, 3)
    process.env.VONTRA_SECURITY_ROLE = 'owner'
    registry.revokeSession(pair.value.sessionId)
    assert.equal((await call('/v1/snapshot', { token })).status, 401)
    ownerAllowed = false
    await assert.rejects(() => bridge.issuePairing('Another phone'), /owner_required/)
    ownerAllowed = true
    await bridge.configure({ version: 1, enabled: false, host: address, port: mobilePort })
    assert.equal(bridge.status().listening, false)
    await assert.rejects(() => call('/v1/snapshot', { token }))
    await assert.rejects(() => bridge.issuePairing('Disabled'), /Enable the private mobile/)
  } finally {
    await bridge.stop(); await core.stop(); await rm(profile, { recursive: true, force: true })
    if (previousRole === undefined) delete process.env.VONTRA_SECURITY_ROLE; else process.env.VONTRA_SECURITY_ROLE = previousRole
  }
  console.log('mobile-bridge.integration.ts passed, including real pinned TLS, private binds, nonce/session/owner boundaries, body limits, host role, revocation and shutdown.')
}
