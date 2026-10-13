import assert from 'node:assert/strict'
import { createServer } from 'node:net'
import { createRemoteControlServer, type RemoteControlServerDeps } from '../../electron/agent/remote-control'
import { createPairingRegistry } from '../../electron/agent/pairing-sessions'
import type { EngineStatus, RemoteControlSettings } from '../../electron/agent/types'
import { remoteDeviceStatus } from '../../electron/agent/device-status'
import { HostError, parsePairingInput, RESTRICTED_CAPABILITIES, StudioClient, validateHost } from '../../mobile/src/lib/client'

const registry = createPairingRegistry()
const port = await new Promise<number>((resolve) => { const probe = createServer(); probe.listen(0, '127.0.0.1', () => { const address = probe.address() as { port: number }; probe.close(() => resolve(address.port)) }) })
const settings: RemoteControlSettings = { enabled: true, serverName: 'Fixture Studio', bindHost: '127.0.0.1', allowLan: false, port, authToken: 'fixture-owner-token', lastStartedAt: null, lastClientAt: null, lastError: null }
const engine: EngineStatus = { available: true, version: '1.0.7', binaryPath: '/private/fixture-secret', state: 'ready', model: 'example/local-model', drafter: null, port: 1234, startedAt: 1, readyAt: 2, message: 'fixture-secret', logTail: ['fixture-secret'], totalMemoryBytes: 16 * 1024 ** 3, memoryLimitBytes: 11 * 1024 ** 3, contextTokens: 32768, hardware: { platform: 'darwin', architecture: 'arm64', chip: 'Apple M3' } }
const snapshot = { serverName: 'Fixture Studio', appVersion: '1.0.0', generatedAt: 1, activeThreadIds: [], threads: [{ id: 'task/ one', title: 'Update docs', kind: 'task', status: 'idle', providerId: 'tensorfold', model: engine.model, updatedAt: 1, timeline: [] }], projects: [], work: { approvals: { pending: 0 }, jobs: { running: 0, queued: 0 } } }
const started: unknown[] = []; const guided: unknown[] = []; const stopped: string[] = []
const deps = {
  getSettings: () => settings, getSnapshot: () => snapshot, getPairedSnapshot: () => snapshot,
  getDeviceEngine: async () => ({ status: { ...engine, apiKey: 'fixture-secret' }, performance: null }),
  startThread: async (input: unknown) => { started.push(input); return { runId: 'fixture-run' } },
  guideThread: async (input: unknown) => { guided.push(input); return { queued: true, queueDepth: 1 } },
  cancelThread: async (id: string) => { stopped.push(id) },
  pairing: { authorizeSessionToken: (token: string) => registry.authorizeSessionToken(token), redeemPairingLink: (credential: string, clientInfo?: string | null) => registry.redeemPairingLink(credential, clientInfo), issuePairingLink: (label: string) => registry.issuePairingLink(label), listSessions: () => registry.listSessions(), revokeSession: (id: string) => registry.revokeSession(id) },
  audit: () => {}, onStarted: () => {}, onClientSeen: () => {}, onError: () => {},
} as unknown as RemoteControlServerDeps
const server = createRemoteControlServer(deps)
const before = process.env.VONTRA_SECURITY_ROLE
process.env.VONTRA_SECURITY_ROLE = 'operator'

try {
  await server.syncFromSettings()
  const base = server.status().url!
  const credential = registry.issuePairingLink('Fixture phone').credential
  const parsed = parsePairingInput(`${base}/#pair=${encodeURIComponent(credential)}`)
  assert.equal(parsed.baseUrl, base)
  assert.equal(parsed.credential, credential)
  assert.ok(!parsed.baseUrl.includes(credential), 'One-time credentials never remain in the saved host URL')
  const timeout = AbortSignal.timeout
  Object.defineProperty(AbortSignal, 'timeout', { value: undefined, configurable: true })
  let session
  try { session = await StudioClient.pair(parsed.baseUrl, credential, 'Fixture iOS') }
  finally { Object.defineProperty(AbortSignal, 'timeout', { value: timeout, configurable: true }) }
  const client = new StudioClient(session)
  const host = await client.host()
  assert.equal(host.snapshot.serverName, 'Fixture Studio')
  assert.equal(host.status?.scope, 'paired')
  assert.equal(host.status?.engine?.model, engine.model)
  assert.equal(host.status?.capabilities.taskStart, true)
  assert.equal(host.status?.capabilities.changesReview, false)
  assert.equal(host.status?.capabilities.approvalResolve, false)
  assert.doesNotMatch(JSON.stringify(host.status), /fixture-secret|binaryPath|logTail|apiKey|\"port\"/, 'Device-status payload picks operational fields without paths, keys, logs or endpoint ports')
  await client.send('task/ one', 'Update the README.', false, host.status!.capabilities)
  await client.send('task/ one', 'Check the examples too.', true, host.status!.capabilities)
  await client.stop('task/ one', host.status!.capabilities)
  assert.equal(started.length, 1)
  assert.equal(guided.length, 1)
  assert.deepEqual(stopped, ['task/ one'], 'Encoded IDs round-trip through actual authenticated host REST routes')
  await assert.rejects(() => client.approve('private-call', 'allow', host.status!.capabilities), /desktop/)
  await assert.rejects(() => client.send('task/ one', 'Forbidden', false, RESTRICTED_CAPABILITIES), /disabled/)
  assert.equal(started.length, 1)

  process.env.VONTRA_SECURITY_ROLE = 'viewer'
  const viewer = await client.host()
  assert.equal(viewer.status?.capabilities.taskStart, false)
  await assert.rejects(() => client.send('task/ one', 'Cannot bypass the viewer host', false, host.status!.capabilities), (error: unknown) => error instanceof HostError && error.status === 403)
  assert.equal(started.length, 1, 'The host independently enforces a role change, despite stale client capabilities')
  process.env.VONTRA_SECURITY_ROLE = 'operator'
  registry.revokeSession(session.sessionId)
  await assert.rejects(() => client.host(), (error: unknown) => error instanceof HostError && error.status === 401)
  await assert.rejects(() => StudioClient.pair(base, credential, 'Second device'), /already used/)
  const unauthorized = await fetch(`${base}/v1/device-status`)
  assert.equal(unauthorized.status, 401, 'Device status remains behind the paired/owner bearer gate')

  const owner = remoteDeviceStatus('owner', 'operator', engine)
  assert.equal(owner.capabilities.changesReview, true)
  assert.equal(owner.capabilities.approvalResolve, true)
  assert.equal(owner.capabilities.engineControl, false)
  const viewerCaps = remoteDeviceStatus('paired', 'viewer', engine).capabilities
  assert.ok(!viewerCaps.taskStart && !viewerCaps.taskGuide && !viewerCaps.taskStop && !viewerCaps.approvalResolve)
} finally {
  await server.stop()
  if (before === undefined) delete process.env.VONTRA_SECURITY_ROLE; else process.env.VONTRA_SECURITY_ROLE = before
}

for (const url of ['http://192.168.1.10:8765', 'http://studio.example.test', 'https://token:secret@example.test', 'https://example.test?token=secret', 'file:///tmp', 'https://example.test/#pair=secret']) assert.throws(() => validateHost(url))
assert.equal(validateHost('https://studio.example.test/private/'), 'https://studio.example.test/private')
assert.throws(() => validateHost('http://10.0.2.2:8765'))
assert.equal(validateHost('http://10.0.2.2:8765', true), 'http://10.0.2.2:8765')
const pin = 'ab'.repeat(32)
assert.deepEqual(parsePairingInput(`https://192.168.1.10:8766/#pair=fixture-credential&pin=${pin.toUpperCase()}`), { baseUrl: 'https://192.168.1.10:8766', credential: 'fixture-credential', certificatePin: pin })
assert.deepEqual(parsePairingInput('fixture-credential', `https://192.168.1.10:8766/#pin=${pin}`), { baseUrl: 'https://192.168.1.10:8766', credential: 'fixture-credential', certificatePin: pin })
assert.throws(() => parsePairingInput(`http://127.0.0.1:8766/#pair=fixture-credential&pin=${pin}`), /HTTPS certificate pin/)
assert.throws(() => parsePairingInput('https://192.168.1.10:8766/#pair=fixture-credential&pin=bad'), /certificate pin/)
assert.throws(() => parsePairingInput(`https://192.168.1.10:8766/#pair=fixture-credential&pin=${pin}&cert=${'ff'.repeat(32)}`), /conflicting/)
assert.throws(() => parsePairingInput(`https://192.168.1.10:8766/#pair=fixture-credential&pin=${pin}&expires=1&v=1`), /expired/)
console.log('Native mobile client pairs over real host REST, uses revocable header-only tokens, starts/guides/stops tasks, respects effective roles, and preserves paired content redaction.')
