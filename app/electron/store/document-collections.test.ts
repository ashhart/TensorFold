import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import type { Thread } from '../agent/types'

const home = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-document-policy-'))
process.env.VONTRA_HOME = home
const { saveThreads, loadThreads, writeProtectedJson, threadsPath, closeSessionSearchIndexes, defaultSettings } = await import('../store')
const { createThreadDocumentPolicyHost } = await import('../main/host/thread-document-policy-host')
const { WorkspaceMutationGuard } = await import('../main/host/workspace-mutation-guard')
const thread: Thread = { id: 'document-owner', title: 'Synthetic document policy chat', kind: 'chat', cwd: '', status: 'idle', createdAt: 1, updatedAt: 1, providerId: 'local', model: 'synthetic', apiMessages: [{ role: 'user', content: 'PRESERVED-SYNTHETIC-HISTORY' }], timeline: [] }
let threads = [thread]
const settings = defaultSettings()
settings.activeProviderId = 'local'
settings.providers = [{ id: 'local', name: 'Synthetic local', kind: 'local', baseUrl: 'http://127.0.0.1:9896/v1', models: ['synthetic'], apiKey: '' } as any]
settings.documents.sources = [{ id: 'one', label: 'First collection', path: '/synthetic/first', kind: 'dir', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null }, { id: 'two', label: 'Second collection', path: '/synthetic/second', kind: 'dir', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null }]
const aborters = new Map<string, AbortController>(), guard = new WorkspaceMutationGuard()
let authorized = true
const change = createThreadDocumentPolicyHost({ threads: () => threads, settings: () => settings, projects: () => [], aborters, guard, authorize: () => { if (!authorized) throw new Error('Synthetic role revoked') } })
const rename = fs.rename
try {
  await saveThreads(threads)
  assert.equal((await loadThreads())[0].documentSourceIds, undefined, 'Legacy selection stays explicit legacy behavior rather than becoming [] or all IDs')
  let entered!: () => void, release!: () => void, blocked = true
  const gate = new Promise<void>(done => { release = done }), started = new Promise<void>(done => { entered = done })
  fs.rename = async (from, to) => { if (String(to) === threadsPath() && blocked) { blocked = false; entered(); await gate }; return rename(from, to) }
  const pending = change(thread.id, ['one'])
  await started
  assert.equal(thread.documentSourceIds, undefined, 'The live chat must not claim acceptance before atomic durable rename')
  assert.throws(() => guard.reserveStart({ threadId: thread.id }), /starting|restoration/)
  const genericSave = saveThreads(threads)
  release(); await pending; await genericSave
  assert.deepEqual(thread.documentSourceIds, ['one']); assert.deepEqual((await loadThreads())[0].documentSourceIds, ['one'])
  assert.equal((await loadThreads())[0].apiMessages[0].content, 'PRESERVED-SYNTHETIC-HISTORY')
  assert.doesNotMatch(await fs.readFile(threadsPath(), 'utf8'), /PRESERVED-SYNTHETIC-HISTORY|documentSourceIds/)
  await change(thread.id, [])
  assert.deepEqual((await loadThreads())[0].documentSourceIds, [], 'An empty selection survives restart without a fallback to enabled sources')
  const beforeFailure = await fs.readFile(threadsPath())
  fs.rename = async (from, to) => { if (String(to) === threadsPath()) throw new Error('Synthetic durable failure'); return rename(from, to) }
  await assert.rejects(change(thread.id, ['two']), /durable failure/)
  assert.deepEqual(thread.documentSourceIds, []); assert.deepEqual(await fs.readFile(threadsPath()), beforeFailure)
  fs.rename = rename
  for (const value of [undefined, null, ['unknown'], ['one', 'one'], [''], ['bad\nname']]) await assert.rejects(change(thread.id, value))
  settings.documents.sources[0].enabled = false
  await assert.rejects(change(thread.id, ['one']), /enabled document collections/); settings.documents.sources[0].enabled = true
  thread.status = 'running'; await assert.rejects(change(thread.id, ['one']), /Stop this run/); thread.status = 'idle'
  aborters.set(thread.id, new AbortController()); await assert.rejects(change(thread.id, ['one']), /Stop this run/); aborters.clear()
  for (const patch of [{ orchestratorHostId: 'synthetic-remote' }, { workspace: { kind: 'ssh', hostId: 'synthetic-host', path: '/workspace' } }, { bot: { presetId: 'synthetic-bot' } }, { phoneLocalOnly: true }, { googleLocalOnly: true }]) {
    Object.assign(thread, patch); await assert.rejects(change(thread.id, ['one']), /eligible local chat/)
    for (const key of Object.keys(patch)) delete (thread as any)[key]
  }
  settings.providers[0].baseUrl = 'http://192.0.2.1:9896/v1'; await assert.rejects(change(thread.id, ['one']), /eligible local chat/); settings.providers[0].baseUrl = 'http://127.0.0.1:9896/v1'
  const privateThread = { ...thread, id: 'private-document-owner', incognito: true }; threads.push(privateThread)
  const beforePrivate = await fs.readFile(threadsPath()); await change(privateThread.id, ['two']); assert.deepEqual(privateThread.documentSourceIds, ['two']); assert.deepEqual(await fs.readFile(threadsPath()), beforePrivate)
  let entered2!: () => void, release2!: () => void, block2 = true
  const gate2 = new Promise<void>(done => { release2 = done }), started2 = new Promise<void>(done => { entered2 = done })
  fs.rename = async (from, to) => { if (String(to) === threadsPath() && block2) { block2 = false; entered2(); await gate2 }; return rename(from, to) }
  const blocker = writeProtectedJson(threadsPath(), threads); await started2
  const stale = change(thread.id, ['one']); authorized = false; release2(); await blocker
  await assert.rejects(stale, /role revoked/); assert.deepEqual(thread.documentSourceIds, [])
  fs.rename = rename; authorized = true
  await writeProtectedJson(threadsPath(), [{ ...thread, documentSourceIds: 'malformed legacy policy' }])
  assert.deepEqual((await loadThreads())[0].documentSourceIds, [], 'Damaged saved allowlists fail closed instead of granting legacy enabled collections')
  await assert.rejects(saveThreads([{ ...thread, documentSourceIds: ['one', 'one'] }]))
  console.log('PASS durable chat document choices: staged protected save, no optimistic publication, queued generic writes, rollback, restart, [] semantics, active/remote/bot/phone/Google/cloud/unknown-source guards, incognito and serialized-role revocation')
} finally { fs.rename = rename; closeSessionSearchIndexes(); await fs.rm(home, { recursive: true, force: true }) }
