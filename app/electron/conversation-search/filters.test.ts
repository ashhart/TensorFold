import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { Thread } from '../agent/types'
import { conversationHasAttachments, conversationSearchDateBound, normalizeConversationSearchFilters } from '../../shared/conversation-search-filters'
import { conversationSearchMatchIsCurrent, createConversationSearchService, validateConversationSearchRequest } from './service'
import { registerConversationSearchIpc } from './ipc'
import { closeSessionSearchIndexes, selectSessionSearchCandidates, syncSessionSearchIndex } from '../store/session-search-index'

const day = new Date(2026, 9, 10).getTime()
const fixture = (id: string, change: Partial<Thread> = {}): Thread => ({ id, title: 'Synthetic needle', kind: 'chat', cwd: '', providerId: 'provider-a', model: 'model-a', status: 'idle', createdAt: day - 100, updatedAt: day,
  apiMessages: [], timeline: [{ id: `${id}-message`, kind: 'user', text: 'Synthetic needle' }], ...change })
const withFile = fixture('file', { timeline: [{ id: 'file-message', kind: 'user', text: 'Synthetic needle', attachments: [{ name: 'synthetic.txt' } as any] }] })
const legacyImage = fixture('image', { apiMessages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,synthetic' } }] }] })
let live = [withFile, legacyImage, fixture('plain'), fixture('before', { updatedAt: day - 1 }), fixture('after', { updatedAt: day + 86_400_000 }), fixture('prefix', { model: 'model-a-extra' }), fixture('provider', { providerId: 'provider-b' }),
  fixture('private', { incognito: true }), fixture('phone', { phoneLocalOnly: true }), fixture('google', { googleLocalOnly: true }), fixture('bot', { bot: {} as any }), fixture('archive', { archivedAt: day }), fixture('project', { projectId: 'project-a' })]
const original = structuredClone(live)
const root = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-search-filter-'))
const file = path.join(root, 'search.sqlite')
const blind = (value: string) => createHash('sha256').update(`synthetic:${value}`).digest('hex')
const service = createConversationSearchService({ getThreads: () => live, findCandidates: (threads, options) => selectSessionSearchCandidates(file, threads, options, blind) })
const ids = (matches: { threadId: string }[]) => [...new Set(matches.map(match => match.threadId))].sort()
try {
  syncSessionSearchIndex(file, live, blind)
  const filters = { updatedFrom: day, updatedBefore: day + 86_400_000, providerId: 'provider-a', model: 'model-a', hasAttachments: true }
  const exact = await service.conversationSearch({ scope: 'all', query: 'needle', filters })
  assert.deepEqual(ids(exact.matches), ['file', 'image'])
  assert.equal(conversationHasAttachments(fixture('tool-only', { timeline: [{ id: 'tool', kind: 'tool', attachments: [{} as any] }] })), false, 'Tool artifacts do not pretend to be user-selected attachments')
  assert.equal(conversationHasAttachments(fixture('system-only', { apiMessages: [{ role: 'system', content: [{ type: 'image_url', image_url: { url: 'synthetic' } }] }] })), false)
  assert.deepEqual(ids((await service.conversationSearch({ scope: 'all', query: 'needle', projectId: 'project-a', filters: { model: 'model-a' } })).matches), ['project'])
  assert.ok(ids((await service.conversationSearch({ scope: 'all', query: 'needle', includeArchived: true, filters: { model: 'model-a' } })).matches).includes('archive'))
  for (const id of ['private', 'phone', 'google', 'bot']) {
    const result = await service.conversationSearch({ scope: 'thread', threadId: id, query: 'needle', filters: { model: 'model-a' } }, { allowedThreadId: id })
    assert.deepEqual(ids(result.matches), [id])
  }
  assert.deepEqual(live, original, 'Filtering never changes threads, attachment references or model selections')
  assert.equal((await service.conversationSearch({ scope: 'thread', threadId: 'file', query: 'needle', filters: { model: 'model-b' } }, { allowedThreadId: 'file' })).matches.length, 0)
  assert.throws(() => validateConversationSearchRequest({ scope: 'all', query: 'needle', filters: { model: '' } }), /model filter/)
  for (const filters of [null, [], { extra: true }, { updatedFrom: NaN }, { updatedFrom: -1 }, { updatedBefore: 0, updatedFrom: 0 }, { updatedBefore: day, updatedFrom: day + 1 }, { hasAttachments: 'true' }, { providerId: 1 }, { model: 'x\u0000y' }]) {
    assert.throws(() => normalizeConversationSearchFilters(filters), /search|start date/)
  }
  assert.equal(conversationSearchDateBound('2026-10-10'), day)
  assert.equal(conversationSearchDateBound('2026-10-10', true), new Date(2026, 9, 11).getTime())
  for (const value of ['2026-02-30', '2026-13-01', '1969-01-01', 'not-a-date']) assert.throws(() => conversationSearchDateBound(value), /valid calendar/)
  const marker = exact.matches.find(match => match.threadId === 'file')!
  withFile.model = 'changed-model'
  assert.equal(conversationSearchMatchIsCurrent(marker, { scope: 'all', query: 'needle', filters }, live), false, 'Late model changes invalidate computed filtered excerpts')
  withFile.model = 'model-a'; withFile.timeline[0].attachments = []
  assert.equal(conversationSearchMatchIsCurrent(marker, { scope: 'all', query: 'needle', filters }, live), false, 'Late attachment removal invalidates a filtered result')
  withFile.timeline[0].attachments = original[0].timeline[0].attachments
  withFile.updatedAt = day - 1
  assert.equal(conversationSearchMatchIsCurrent(marker, { scope: 'all', query: 'needle', filters }, live), false)
  // Hundreds of unrelated index hits must not consume the candidate budget before filters apply.
  live = [...Array.from({ length: 230 }, (_, index) => fixture(`unrelated-${index}`, { model: 'other' })), fixture('filtered-tail', { model: 'wanted' })]
  syncSessionSearchIndex(file, live, blind)
  const narrow = await service.conversationSearch({ scope: 'all', query: 'needle', filters: { model: 'wanted' } })
  assert.deepEqual(ids(narrow.matches), ['filtered-tail']); assert.equal(narrow.truncated, false)

  const handlers = new Map<string, any>()
  let release: () => void = () => undefined, auth = 0
  registerConversationSearchIpc({ handle: (channel, callback) => handlers.set(channel, callback), getThreads: () => live, getProfileKey: () => 'synthetic',
    authorize: async () => { if (++auth === 2) await new Promise<void>(resolve => { release = resolve }) } })
  const pending = handlers.get('conversation:search')({ sender: {} }, { scope: 'all', query: 'needle', filters: { model: 'wanted' } })
  await new Promise<void>(resolve => setImmediate(resolve))
  live.at(-1)!.model = 'different'
  release()
  await assert.rejects(pending, /Conversation changed/, 'IPC revalidates filter membership after its second authorization await')
  console.log('Search filters pass exact identity, local inclusive dates, file/legacy image detection, restricted scopes, index budget, invalid input and authority-race checks')
} finally { closeSessionSearchIndexes(); rmSync(root, { recursive: true, force: true }) }
