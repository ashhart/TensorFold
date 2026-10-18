import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { assertGoogleContextTransport } from '../../shared/google-workspace-privacy'
import { assertThreadChildSupported } from '../../shared/thread-child-policy'
import { runAgent, compactThread } from './loop'
import { executeTool } from './tools'
import { createEditedTaskBranch } from './message-edit'
import { canRecallThread, readSessionThread, searchSessionThreads } from './session-recall'
import { buildRetrievalIndex, buildWorkspaceRetrievalIndex } from './retrieval-index'
import { closeSessionSearchIndexes, searchSessionIndex, syncSessionSearchIndex } from '../store/session-search-index'
import { createThreadTitleGenerator } from '../main/host/thread-title'

const local: any = { id: 'tensorfold', source: 'tensorfold', kind: 'local', transport: 'chat-completions' }
const source: any = { id: 'synthetic-google-source', kind: 'task', googleLocalOnly: true, title: 'Synthetic Google context', apiMessages: [{ role: 'tool', name: 'gmail_read', content: 'SYNTHETIC_PRIVATE_GOOGLE_AMBER' }], timeline: [], providerId: local.id, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle' }
assert.doesNotThrow(() => assertGoogleContextTransport(source, local))
for (const provider of [undefined, { id: 'cloud', kind: 'cloud', baseUrl: 'https://model.example.com/v1' }, { id: 'lan', kind: 'local', baseUrl: 'http://192.0.2.20:8080/v1' }, { ...local, transport: 'codex-cli' }, { ...local, baseUrl: 'https://model.example.com/v1' }]) {
  assert.throws(() => assertGoogleContextTransport(source, provider), /private Google account data/)
  await assert.rejects(runAgent({ thread: source, provider } as any), /private Google account data/)
}
await assert.rejects(runAgent({ thread: source, provider: local } as any), /Google account privacy guard is unavailable/)
await assert.rejects(runAgent({ thread: source, provider: local, toolContext: { googlePrivacyGuard: () => 'External research must be off.' } } as any), /External research/)
await assert.rejects(compactThread(source, { ...local, transport: 'claude-cli' }, 'synthetic', new AbortController().signal), /private Google/)
assert.throws(() => assertThreadChildSupported(source), /Google account conversations/)
const edited = createEditedTaskBranch(source, {}, 'synthetic-user-message')
assert.equal(edited.googleLocalOnly, true)
for (const name of ['web_search', 'web_fetch']) assert.equal((await executeTool(name, { query: 'Synthetic', url: 'https://example.com' }, { cwd: '/synthetic', currentThread: source } as any)).ok, false)
let titleCalls = 0
const title = createThreadTitleGenerator({ completeTitle: async () => { titleCalls++; return 'Should not run' }, reserveOperation: () => ({ release() {} }), isCurrentThread: () => true, persistThreads() {}, emitToRenderer() {} })
await title({ ...source, title: 'New task' }, local, 'synthetic', 'private context')
assert.equal(titleCalls, 0)

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-google-privacy-'))
const reader: any = { ...source, id: 'synthetic-reader', googleLocalOnly: false, cwd: home, apiMessages: [] }
source.cwd = home
const threads = [source, reader]
const blind = (text: string) => createHash('sha256').update(`synthetic:${text}`).digest('hex')
try {
  assert.equal(canRecallThread(source, reader.id), false)
  assert.equal(canRecallThread(source, source.id), true)
  assert.equal(readSessionThread(threads, { threadId: source.id, currentThreadId: reader.id }).thread, null)
  assert.equal(searchSessionThreads(threads, { query: 'AMBER', currentThreadId: reader.id }).matches.length, 0)
  assert.equal(searchSessionThreads(threads, { query: 'AMBER', currentThreadId: source.id, includeCurrent: true }).matches.length, 1)
  const index = path.join(home, 'synthetic-index.sqlite')
  syncSessionSearchIndex(index, [{ ...source, googleLocalOnly: false }, reader], blind)
  assert.equal(searchSessionIndex(index, threads, { query: 'AMBER', currentThreadId: reader.id }, blind).matches.length, 0, 'stale blind index cannot bypass live Google source privacy')
  syncSessionSearchIndex(index, threads, blind)
  assert.equal(searchSessionIndex(index, threads, { query: 'AMBER', currentThreadId: source.id, includeCurrent: true }, blind).matches.length, 1)
  const retrieval = buildRetrievalIndex({ currentThreadId: reader.id, threads, records: [{ id: 'artifact:old-google', kind: 'artifact', text: 'SYNTHETIC_PRIVATE_GOOGLE_AMBER', sourceIds: [`thread:${source.id}:message:0`] }] })
  assert.equal(retrieval.records.length, 0)
  const pending = buildWorkspaceRetrievalIndex({ cwd: home, kinds: ['chat'], currentThreadId: reader.id, threads: [{ ...source, googleLocalOnly: false }], recallThreads: () => threads })
  assert.equal((await pending).records.some(record => record.id.startsWith(`thread:${source.id}:`)), false, 'async collection checks latest Google source privacy')
} finally { closeSessionSearchIndexes(); await rm(home, { recursive: true, force: true }) }
console.log('Google retained history pins inference, helper models, branches, web requests and cross-chat recall to the owning local context')
