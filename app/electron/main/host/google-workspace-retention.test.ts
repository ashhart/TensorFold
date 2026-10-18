import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { store } from '../state'
import { createThreadForkFactory } from './thread-revert-fork'
import { assertRemoteThreadVisible, isRemoteEventVisible, remoteArchiveSnapshot, remoteControlSnapshot } from './remote-snapshot'
import { createWatchCommandProjection } from './watch-command-projection'
import { createWatchSpeechRuntime } from './watch-speech-runtime'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-google-retention-'))
process.env.VONTRA_HOME = home
const { saveThreads, loadThreads, defaultSettings, closeSessionSearchIndexes } = await import('../../store')
const saved = { ...store }
const marker = 'SYNTHETIC_PRIVATE_GOOGLE_RETAINED_AMBER'
const source: any = { id: 'synthetic-google-source', title: marker, googleLocalOnly: true, kind: 'task', cwd: home, providerId: 'synthetic-local', model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', timeline: [{ id: 'synthetic-assistant', kind: 'assistant', text: marker }], apiMessages: [{ role: 'tool', name: 'gmail_read', content: marker }] }
let workers = 0
const speech = createWatchSpeechRuntime(() => { workers++; return {} as any }, () => true)
try {
  store.settings = defaultSettings(); store.settings.providers = []; store.threads = [source]; store.jobs = []; store.projects = []; store.rooms = []; store.tickets = []; store.missions = []; store.automations = []; store.connectorEvents = []; store.cliSessions = []
  await saveThreads(store.threads)
  assert.equal((await loadThreads())[0].googleLocalOnly, true, 'Google context marker survives protected-store restart')
  const fork = createThreadForkFactory({ persistThreads() {}, remoteServer: () => ({ publish() {} }), spineHost: () => null })(source.id, { turn: 1, branch: 'synthetic', conversation: { apiMessages: source.apiMessages, timeline: source.timeline } } as any)!
  assert.equal(fork.googleLocalOnly, true, 'forked history inherits its local-only boundary')
  const snapshot = remoteControlSnapshot({ appVersion: 'synthetic', headless: false, activeThreadIds: [source.id, fork.id], activeRoomIds: [], activeMissionIds: [] })
  assert.equal(snapshot.threads.length, 0)
  assert.equal(snapshot.activeThreadIds.length, 0)
  assert.equal(JSON.stringify(snapshot).includes(marker), false)
  assert.equal(remoteArchiveSnapshot({ projects: [], rooms: [], tickets: [], threads: [source] } as any).threads.length, 0)
  assert.equal(isRemoteEventVisible({ kind: 'agent', data: { threadId: source.id, output: marker } }), false)
  assert.equal(isRemoteEventVisible({ kind: 'thread', data: source }), false)
  assert.throws(() => assertRemoteThreadVisible(source.id), /not_found/)
  const watch = createWatchCommandProjection({ now: () => 100, environment: () => ({}) })({ version: 1, session: 'synthetic-paired', generatedAt: 1, emittedAt: 1, connected: true, serverName: 'Synthetic', jobs: [{ id: source.id, title: marker, state: 'finished', updatedAt: 1, canReply: true, latestReply: marker, latestReplyId: 'synthetic-assistant' }], running: 0, queued: 0, approvals: 0, transcriptReview: true, error: null }, 'synthetic-paired')
  assert.equal(watch.jobs.length, 0)
  await assert.rejects(speech.start('synthetic-paired', { threadId: source.id, itemId: 'synthetic-assistant' }), /not_found|Mac/)
  assert.equal(workers, 0, 'watch speech refuses Google content before starting a model or worker')
} finally { speech.stop(); closeSessionSearchIndexes(); Object.assign(store, saved); await rm(home, { recursive: true, force: true }) }
console.log('Google history marker survives protected persistence and forks while remote/watch snapshots, events and speech stay private')
