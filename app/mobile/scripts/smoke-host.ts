// Disposable protocol fixture for native UI QA; this never runs inference or edits a real project.
import { createServer } from 'node:net'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { createRemoteControlServer, type RemoteControlServerDeps } from '../../electron/agent/remote-control'
import { createPairingRegistry } from '../../electron/agent/pairing-sessions'
import { createMobileBridge, mobileBridgeInterfaces } from '../../electron/agent/mobile-bridge'
import { loadMobileBridgeCertificate } from '../../electron/agent/mobile-bridge-certificate'
import type { EngineStatus, RemoteControlSettings } from '../../electron/agent/types'

const root = path.resolve('.expo/native-smoke')
await fs.mkdir(root, { recursive: true, mode: 0o700 })
const registry = createPairingRegistry()
const port = await new Promise<number>((resolve) => { const probe = createServer(); probe.listen(0, '127.0.0.1', () => { const address = probe.address() as { port: number }; probe.close(() => resolve(address.port)) }) })
const settings: RemoteControlSettings = { enabled: true, serverName: 'Native QA Studio', bindHost: '127.0.0.1', allowLan: false, port, authToken: crypto.randomUUID(), lastStartedAt: null, lastClientAt: null, lastError: null }
const engine: EngineStatus = { available: true, version: 'fixture', binaryPath: null, state: 'ready', model: 'example/native-qa-model', drafter: null, port: null, startedAt: 1, readyAt: 1, message: null, logTail: [], totalMemoryBytes: 16 * 1024 ** 3, memoryLimitBytes: Math.floor(16 * 1024 ** 3 * 0.7), contextTokens: 32768, hardware: { platform: 'darwin', architecture: 'arm64', chip: 'Fixture Apple silicon profile' } }
const task = { id: 'native-qa-worker', title: 'Review repository documentation', kind: 'task', providerId: 'tensorfold', model: engine.model, status: 'idle', updatedAt: Date.now(), timeline: [] as { id: string; kind: string; tool?: string; state?: string }[] }
const snapshot = { serverName: settings.serverName, appVersion: 'fixture', generatedAt: Date.now(), activeThreadIds: [] as string[], threads: [task], projects: [], work: { approvals: { pending: 0 }, jobs: { running: 0, queued: 0 } } }
const actions: { type: string; threadId: string; text?: string }[] = []
const save = () => fs.writeFile(path.join(root, 'actions.json'), JSON.stringify({ actions, engineState: engine.state, role: process.env.VONTRA_SECURITY_ROLE, active: snapshot.activeThreadIds }, null, 2), { mode: 0o600 })
const server = createRemoteControlServer({
  getSettings: () => settings, getSnapshot: () => snapshot, getPairedSnapshot: () => snapshot,
  getDeviceEngine: async () => ({ status: engine, performance: null }),
  pairing: { issuePairingLink: (label: string) => registry.issuePairingLink(label), redeemPairingLink: (credential: string, info?: string | null) => registry.redeemPairingLink(credential, info), authorizeSessionToken: (token: string) => registry.authorizeSessionToken(token), listSessions: () => registry.listSessions(), revokeSession: (id: string) => registry.revokeSession(id) },
  startThread: async ({ threadId, userText }: { threadId: string; userText: string }) => {
    if (engine.state !== 'ready') throw Object.assign(new Error('fixture_model_not_ready'), { code: 409 })
    actions.push({ type: 'start', threadId, text: userText }); task.status = 'running'; task.updatedAt = Date.now(); snapshot.activeThreadIds = [threadId]
    task.timeline.push({ id: crypto.randomUUID(), kind: 'tool', tool: 'read_file', state: 'running' }); await save(); return { runId: 'fixture-run' }
  },
  guideThread: async ({ threadId, userText }: { threadId: string; userText: string }) => { actions.push({ type: 'guide', threadId, text: userText }); await save(); return { queued: true, queueDepth: 1 } },
  cancelThread: async (threadId: string) => { actions.push({ type: 'stop', threadId }); task.status = 'idle'; snapshot.activeThreadIds = []; task.timeline.forEach((item) => { item.state = 'cancelled' }); if (process.env.TENSORFOLD_QA_REVOKE_AFTER_STOP === '1') registry.listSessions().forEach(session => registry.revokeSession(session.id)); await save() },
  audit: () => {}, onStarted: () => {}, onClientSeen: () => {}, onError: () => {},
} as unknown as RemoteControlServerDeps)
process.env.VONTRA_SECURITY_ROLE = 'operator'
await server.syncFromSettings()
// The private listener is opt-in and uses the unchanged production bridge policy.
const bridge = createMobileBridge({
  requireOwner: () => {}, loopbackUrl: () => server.status().url, ownerToken: () => settings.authToken,
  authorizeSessionToken: token => registry.authorizeSessionToken(token),
  issuePairingLink: async label => registry.issuePairingLink(label),
  loadCertificate: host => loadMobileBridgeCertificate(path.join(root, 'profile'), host),
})
if (process.env.TENSORFOLD_QA_PRIVATE_LAN === '1') {
  const iface = mobileBridgeInterfaces()[0]
  if (!iface) throw new Error('Native QA needs an available private interface.')
  const tlsPort = await new Promise<number>(resolve => { const probe = createServer(); probe.listen(0, iface.address, () => { const address = probe.address() as { port: number }; probe.close(() => resolve(address.port)) }) })
  await bridge.configure({ version: 1, enabled: true, host: iface.address, port: tlsPort })
}
const saveLink = async () => {
  const tls = bridge.status()
  const credential = tls.listening ? null : registry.issuePairingLink('Native simulator fixture').credential
  const pairingLink = tls.listening ? (await bridge.issuePairing('Native simulator fixture')).link : `${server.status().url}/#pair=${encodeURIComponent(credential!)}`
  await fs.writeFile(path.join(root, 'connection.json'), JSON.stringify({ baseUrl: tls.url ?? server.status().url, pairingLink }, null, 2), { mode: 0o600 })
}
await saveLink()
await save()
console.log('Native UI fixture ready; pairing data is in .expo/native-smoke/connection.json, never a real profile.')
process.stdin.on('data', (data) => {
  const command = String(data).trim()
  if (command === 'new-link') {
    void saveLink()
  }
  if (command === 'model-stopped') engine.state = 'stopped'
  if (command === 'model-ready') engine.state = 'ready'
  if (command === 'viewer' || command === 'operator') process.env.VONTRA_SECURITY_ROLE = command
  if (command === 'revoke') registry.listSessions().forEach((session) => registry.revokeSession(session.id))
  void save()
})
const stop = () => { void bridge.stop().then(() => server.stop()).then(() => process.exit(0)) }
process.on('SIGTERM', stop)
process.on('SIGINT', stop)
