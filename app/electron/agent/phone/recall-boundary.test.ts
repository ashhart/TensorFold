import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp, mkdir, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { executeTool } from '../tools'
import { readSessionThread, searchSessionThreads } from '../session-recall'
import { buildRetrievalIndex, buildWorkspaceRetrievalIndex } from '../retrieval-index'
import { closeSessionSearchIndexes, searchSessionIndex, syncSessionSearchIndex } from '../../store/session-search-index'
import type { Thread } from '../types'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-phone-recall-boundary-'))
const marker = 'SYNTHETIC_PHONE_CONTEXT_AMBER'
const thread = (id: string, phoneLocalOnly = false): Thread => ({ id, title: 'Synthetic source', kind: 'chat', cwd: home, providerId: null, model: null, createdAt: 1, updatedAt: 1, status: 'idle', phoneLocalOnly, apiMessages: [{ role: 'user', content: marker }], timeline: [] })
const source = thread('phone-source', true), ordinary = thread('ordinary-source'), reader = thread('ordinary-reader')
const threads = [source, ordinary, reader]
const blind = (token: string) => createHash('sha256').update(`synthetic-test:${token}`).digest('hex')
try {
  const context: any = { cwd: home, currentThread: reader, threads }
  const direct = await executeTool('session_read', { threadId: source.id, currentThreadId: source.id }, context)
  assert.equal(direct.ok, false, 'raw session_read cannot copy phone-local history into an unmarked recipient, even with a forged reader id')
  assert.doesNotMatch(direct.output, /SYNTHETIC_PHONE_CONTEXT_AMBER/)
  const search = await executeTool('session_search', { query: marker, currentThreadId: source.id, includeCurrent: true }, context)
  assert.equal(search.ok, true); assert.doesNotMatch(search.output, /thread: phone-source/)
  assert.equal((await executeTool('session_read', { threadId: ordinary.id }, context)).ok, true)
  const ownContext = { ...context, currentThread: source }
  assert.equal((await executeTool('session_read', { threadId: source.id }, ownContext)).ok, true, 'same phone-local owner retains its own context under the run privacy guard')
  assert.equal(readSessionThread(threads, { threadId: source.id }).thread, null, 'portable helper fails closed without a recipient')
  assert.equal(readSessionThread(threads, { threadId: source.id, currentThreadId: source.id }).thread?.id, source.id)
  assert.equal(searchSessionThreads(threads, { query: marker, currentThreadId: reader.id }).matches.some((m) => m.threadId === source.id), false)
  const index = path.join(home, 'index.sqlite')
  syncSessionSearchIndex(index, [{ ...source, phoneLocalOnly: false }, ordinary, reader], blind)
  assert.equal(searchSessionIndex(index, threads, { query: marker, currentThreadId: reader.id }, blind).matches.some((m) => m.threadId === source.id), false, 'stale ordinary index entry cannot expose a now phone-local source')
  syncSessionSearchIndex(index, threads, blind)
  assert.equal(searchSessionIndex(index, threads, { query: marker, currentThreadId: source.id, includeCurrent: true }, blind).matches.some((m) => m.threadId === source.id), true)
  const broken = path.join(home, 'not-a-database'); await mkdir(broken)
  assert.equal(searchSessionIndex(broken, threads, { query: marker, currentThreadId: reader.id }, blind).matches.some((m) => m.threadId === source.id), false, 'broken-index fallback preserves source protection')
  const prebuilt: any[] = [{ id: 'artifact:old-phone', sourceIds: [`thread:${source.id}:timeline:old`], kind: 'artifact', text: marker }, { id: 'thread:unknown-source:message:0', kind: 'chat', text: marker }]
  const retrieval = buildRetrievalIndex({ currentThreadId: reader.id, threads, records: prebuilt })
  assert.equal(retrieval.records.some((record) => record.sourceIds.some((id) => id.startsWith(`thread:${source.id}:`))), false)
  assert.equal(retrieval.records.some((record) => record.id.includes('unknown-source')), false, 'prebuilt thread records require live visible provenance')
  assert.equal(buildRetrievalIndex({ currentThreadId: source.id, threads }).records.some((record) => record.id.startsWith(`thread:${source.id}:`)), true)
  const refreshed = await buildWorkspaceRetrievalIndex({ cwd: home, kinds: ['chat'], currentThreadId: reader.id, threads: [{ ...source, phoneLocalOnly: false }], recallThreads: () => threads })
  assert.equal(refreshed.records.some((record) => record.id.startsWith(`thread:${source.id}:`)), false, 'async collection checks current canonical source policy before exposing records')
  const dynamic = { ...context, threads: [{ ...source, phoneLocalOnly: false }], recallThreads: () => threads }
  assert.equal((await executeTool('session_read', { threadId: source.id }, dynamic)).ok, false, 'host callback rereads canonical source privacy rather than a captured permissive list')
  for (const tool of ['session_search', 'retrieval_search']) {
    let live = [{ ...source, phoneLocalOnly: false }, reader]
    const ctx = { ...context, threads: live, recallThreads: () => live, searchSessions: async (options: any) => searchSessionThreads(live, options) }
    const pending = executeTool(tool, { query: 'AMBER', kinds: ['chat'], includeCurrent: true }, ctx)
    live = [source, reader]
    const result = await pending
    assert.equal(result.output.includes(`thread:${source.id}:`), false, `${tool} revalidates after its last awaited collection/search before formatting`)
  }
  const linked: any = { id: 'private-linked-mission', threadId: source.id, title: 'Synthetic linked mission', checkpoints: [], scratchMemory: [{ id: 'note', kind: 'fact', text: 'PHONE-LINKED-MISSION-AMBER' }] }
  assert.equal(buildRetrievalIndex({ currentThreadId: reader.id, threads, missions: [linked] }).records.some((record) => record.text.includes('PHONE-LINKED-MISSION-AMBER')), false)
  assert.equal(buildRetrievalIndex({ currentThreadId: source.id, threads, missions: [linked] }).records.some((record) => record.text.includes('PHONE-LINKED-MISSION-AMBER')), true)
  const child: any = { id: 'private-child', threadId: source.id, parent: { threadId: source.id }, updatedAt: 1, messages: [], name: 'Synthetic child', status: 'succeeded', task: 'PHONE-LINKED-CHILD-AMBER', result: 'PHONE-LINKED-CHILD-AMBER' }
  assert.equal((await executeTool('subagent_status', { agent_ids: [child.id] }, { ...context, listSubAgents: () => [child] })).output.includes('PHONE-LINKED-CHILD-AMBER'), false)
  assert.equal((await executeTool('subagent_status', { agent_ids: [child.id] }, { ...ownContext, listSubAgents: () => [child] })).output.includes('PHONE-LINKED-CHILD-AMBER'), true)
  console.log('Phone-local recall: direct/forged reader, ordinary/same-owner positives, stale and broken index, prebuilt provenance and live host source protection passed')
} finally { closeSessionSearchIndexes(); await rm(home, { recursive: true, force: true }) }
