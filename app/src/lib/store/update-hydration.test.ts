import assert from 'node:assert/strict'
import type { AppUpdateStatus, Settings } from '../types'

const globals = globalThis as any
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = () => {}; globals.removeEventListener = () => {}; globals.window = globals
const { api } = await import('../ipc')
const { useStore } = await import('./index')

const settings: Settings = await api.getSettings()
const unchecked: AppUpdateStatus = {
  state: 'not_checked', currentVersion: '1.0.0-beta.5', latestVersion: null, releaseName: null,
  releaseNotes: null, downloadUrl: null, checksum: null, signature: null, signatureVerified: false,
  integrity: 'not_checked', feedUrl: 'https://example.com/update.json', checkedAt: null, error: null,
  automaticUpdate: null, automaticInstall: { supported: false, reason: 'Check first.' }, downloadProgress: null,
}
const available: AppUpdateStatus = {
  ...unchecked, state: 'available', latestVersion: '1.0.0-beta.6', signatureVerified: true, signature: 'synthetic',
  integrity: 'signature_verified', checkedAt: 100, downloadUrl: 'https://example.com/installer.dmg',
  checksum: { algorithm: 'sha256', value: 'a'.repeat(64) }, automaticInstall: { supported: true, reason: null },
}
const ready: AppUpdateStatus = { ...available, state: 'ready', integrity: 'artifact_verified', downloadProgress: { receivedBytes: 100, totalBytes: 100 } }
const progress: AppUpdateStatus = { ...available, state: 'downloading', downloadProgress: { receivedBytes: 42, totalBytes: 100 } }
let listener: ((status: AppUpdateStatus) => void) | undefined
let subscriptionCount = 0
for (const name of ['onAgentEvent', 'onProjectUpdate', 'onThreadUpdate', 'onRoomUpdate', 'onRoomEvent', 'onTicketUpdate', 'onJobUpdate', 'onAutomationUpdate', 'onOpenThread', 'onWhatsAppStatus', 'onPersistenceFailure', 'onSshStatus', 'onSshSecretPrompt', 'onSshTrustPrompt', 'onTerminalSessions', 'onTerminalStatus']) (api as any)[name] = () => () => {}
api.onUpdateStatus = (callback) => { listener = callback; subscriptionCount++; return () => {} }
api.getThreads = async () => []; api.getProjects = async () => []; api.getRooms = async () => []
api.getTickets = async () => []; api.getJobs = async () => []; api.getAutomations = async () => []
api.getArchives = async () => ({ projects: [], threads: [], rooms: [], tickets: [] })
api.listSshHosts = async () => []
api.spineReadSince = async () => { throw new Error('Synthetic spine unavailable; no background data.') }

let runs = 0
async function hydrate(name: string, input: { initial?: AppUpdateStatus | null; snapshot?: AppUpdateStatus; events?: AppUpdateStatus[]; snapshotFailure?: boolean; settingsFailure?: boolean }) {
  useStore.setState({ updateStatus: input.initial ?? null, notice: null })
  let releaseSettings!: () => void
  api.getSettings = () => new Promise<Settings>((resolve, reject) => { releaseSettings = () => input.settingsFailure ? reject(new Error('Synthetic settings unavailable.')) : resolve(settings) })
  let snapshotRead = false
  api.getUpdateStatus = async () => { snapshotRead = true; if (input.snapshotFailure) throw new Error('Synthetic snapshot unavailable.'); return input.snapshot ?? unchecked }
  const pending = useStore.getState().init(runs++ ? { retry: true } : undefined)
  assert.equal(snapshotRead, true, 'The old update snapshot is captured while settings remains pending.')
  assert.equal(useStore.getState().ready, false)
  await Promise.resolve(); await Promise.resolve()
  assert.ok(listener, 'The real init registers the IPC update listener before primary loads finish.')
  for (const status of input.events ?? []) { listener(status); assert.equal(useStore.getState().updateStatus, status, 'Real listener applies the live push before final hydration.') }
  releaseSettings(); await pending
  assert.equal(useStore.getState().ready, true)
  const expected = input.events?.at(-1) ?? (input.snapshotFailure ? input.initial ?? null : input.snapshot ?? unchecked)
  assert.equal(useStore.getState().updateStatus, expected, `${name}: final hydration must retain the newest live push, or use the snapshot when no push arrived.`)
  if (input.snapshotFailure || input.settingsFailure) assert.equal(useStore.getState().notice?.kind, 'error')
  assert.equal(subscriptionCount, 1, 'Retry uses the existing update subscription, without duplicates.')
  console.log(`PASS ${name}`)
}
await hydrate('available notice survives stale unchecked snapshot', { snapshot: unchecked, events: [available] })
await hydrate('ready installer survives stale checking snapshot', { snapshot: { ...unchecked, state: 'checking' }, events: [ready] })
await hydrate('download progress survives stale available snapshot', { snapshot: available, events: [progress] })
await hydrate('last of multiple live pushes wins', { snapshot: unchecked, events: [available, progress, ready] })
await hydrate('fresh snapshot replaces existing state when no live event', { initial: ready, snapshot: { ...available, latestVersion: '1.0.0-beta.7' } })
await hydrate('failed snapshot retains live ready state', { snapshotFailure: true, events: [ready] })
await hydrate('failed snapshot without push retains existing state', { initial: available, snapshotFailure: true })
await hydrate('other failed loader does not overwrite live notice', { snapshot: unchecked, events: [available], settingsFailure: true })
await hydrate('live error supersedes stale available snapshot', { snapshot: available, events: [{ ...unchecked, state: 'error', error: 'Synthetic newer check failed.', checkedAt: 200 }] })
console.log('Actual store.init hydration preserves live update notice/ready/progress/error pushes, uses fresh snapshots without pushes, and keeps failure/retry controls.')
