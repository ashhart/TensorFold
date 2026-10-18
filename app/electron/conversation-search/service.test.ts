import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { Thread } from '../agent/types'
import { closeSessionSearchIndexes, selectSessionSearchCandidates, syncSessionSearchIndex } from '../store/session-search-index'
import { createConversationSearchService } from './service'

function thread(id: string, text = 'Synthetic needle in a message'): Thread {
  return { id, title: 'Synthetic conversation', kind: 'chat', cwd: '', providerId: 'local', model: 'synthetic',
    createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [{ role: 'user', content: text }],
    timeline: [{ id: `${id}-message`, kind: 'user', text }] }
}

const root = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-conversation-search-'))
const file = path.join(root, 'profile-one.sqlite')
const blind = (token: string) => createHash('sha256').update(`synthetic-secret:${token}`).digest('hex')
let live: Thread[] = [thread('visible'),
  { ...thread('private'), incognito: true },
  { ...thread('phone'), phoneLocalOnly: true },
  { ...thread('google'), googleLocalOnly: true },
  { ...thread('bot'), bot: {} as any },
  { ...thread('remote'), orchestratorHostId: 'synthetic-remote' },
  { ...thread('archive'), archivedAt: 3 },
  { ...thread('project'), projectId: 'synthetic-project' },
]
const original = structuredClone(live)
const service = createConversationSearchService({ getThreads: () => live,
  findCandidates: (threads, options) => selectSessionSearchCandidates(file, threads, options, blind) })

try {
  syncSessionSearchIndex(file, live, blind)
  const result = await service.conversationSearch({ scope: 'all', query: 'needle' })
  assert.deepEqual(result.matches.map((match) => match.threadId).sort(), ['project', 'visible'])
  assert.equal(result.matches[0].source, 'timeline')
  assert.ok(result.matches.every((match) => match.itemId === `${match.threadId}-message`))
  assert.ok(result.matches.every((match) => match.highlights.some((range) => match.excerpt.slice(range.start, range.end) === 'needle')))
  assert.equal((await service.conversationSearch({ scope: 'all', query: 'needle', includeArchived: true })).matches.length, 3)
  assert.deepEqual((await service.conversationSearch({ scope: 'all', query: 'needle', projectId: 'synthetic-project' })).matches.map((match) => match.threadId), ['project'])
  assert.deepEqual(live, original, 'search leaves live messages, generation status and other state untouched')

  await assert.rejects(service.conversationSearch({ scope: 'thread', query: 'needle', threadId: 'private' }), /not authorized/)
  await assert.rejects(service.conversationSearch({ scope: 'thread', query: 'needle', threadId: 'private' }, { allowedThreadId: 'visible' }), /not authorized/)
  for (const id of ['private', 'phone', 'google', 'bot']) {
    const own = await service.conversationSearch({ scope: 'thread', query: 'needle', threadId: id }, { allowedThreadId: id })
    assert.equal(own.matches[0]?.threadId, id, 'bound current chat is scanned without persisting restricted text')
  }
  await assert.rejects(service.conversationSearch({ scope: 'thread', query: 'needle', threadId: 'remote' }, { allowedThreadId: 'remote' }), /no longer available/)

  const multilingual = thread('unicode', '😀 Un cafe\u0301 à Montréal; ПРИВЕТ МИР; 合成東京検索テスト中文资料')
  multilingual.status = 'running'
  live = [multilingual]
  syncSessionSearchIndex(file, live, blind)
  for (const query of ['café montréal', 'привет', '東京検索', '京']) {
    const found = await service.conversationSearch({ scope: 'all', query })
    assert.equal(found.matches[0]?.itemId, 'unicode-message', `indexed multilingual timeline match ${query}`)
    assert.ok(found.matches[0].highlights.every((range) => range.start >= 0 && range.end <= found.matches[0].excerpt.length && range.start < range.end))
  }
  const accent = await service.conversationSearch({ scope: 'all', query: 'cafe' })
  assert.equal(accent.matches[0].excerpt.slice(accent.matches[0].highlights[0].start, accent.matches[0].highlights[0].end), 'cafe\u0301')
  closeSessionSearchIndexes()
  assert.equal((await service.conversationSearch({ scope: 'all', query: 'привет' })).matches[0]?.threadId, 'unicode', 'indexed results survive reopening')
  const bytes = readFileSync(file)
  for (const secret of ['Montréal', 'ПРИВЕТ', '東京検索', 'cafe']) assert.equal(bytes.includes(Buffer.from(secret)), false, 'derived index stores hashes, not display text')

  multilingual.timeline[0].text = 'Replacement current message'
  multilingual.apiMessages[0].content = 'Replacement current message'
  multilingual.messageEditRevisions = [{ id: 'old', itemId: 'unicode-message', editedAt: 4, conversationRevision: 0,
    timeline: [{ id: 'unicode-message', kind: 'user', text: 'Superseded needle' }], apiMessages: [{ role: 'user', content: 'Superseded needle' }] }]
  assert.equal((await service.conversationSearch({ scope: 'all', query: 'привет' })).matches.length, 0, 'stale index candidate cannot report edited text')
  assert.equal((await service.conversationSearch({ scope: 'all', query: 'Replacement current' })).matches[0]?.itemId, 'unicode-message', 'new live text remains searchable before index sync')
  assert.equal((await service.conversationSearch({ scope: 'thread', threadId: 'unicode', query: 'needle' }, { allowedThreadId: 'unicode' })).matches.length, 0, 'superseded histories never attach to an active message')
  multilingual.timeline.push({ id: 'streaming', kind: 'assistant', text: 'Unpersisted stream marker' })
  assert.equal((await service.conversationSearch({ scope: 'thread', threadId: 'unicode', query: 'stream marker' }, { allowedThreadId: 'unicode' })).matches[0]?.itemId, 'streaming', 'current-chat search sees live generation before index persistence')

  const legacy = thread('legacy', 'Legacy saved history marker')
  legacy.timeline = []
  legacy.apiMessages.push({ role: 'system', content: 'System-only confidentialterm' }, { role: 'tool', content: 'Tool-only confidentialterm' })
  live = [legacy]
  syncSessionSearchIndex(file, live, blind)
  const history = await service.conversationSearch({ scope: 'all', query: 'history marker' })
  assert.equal(history.matches[0]?.source, 'history')
  assert.equal(history.matches[0]?.itemId, null)
  assert.equal(history.matches[0]?.messageIndex, 0)
  assert.equal((await service.conversationSearch({ scope: 'all', query: 'confidentialterm' })).matches.length, 0)

  const capped = thread('capped', '')
  capped.apiMessages = []
  capped.timeline = [{ id: 'old-evidence', kind: 'user', text: 'zzoldbodyneedle' },
    { id: 'recent-filler', kind: 'assistant', text: Array.from({ length: 3_000 }, (_, index) => `${index.toString(36).padStart(5, '0')}syntheticfiller`).join(' ') }]
  live = [capped]
  syncSessionSearchIndex(file, live, blind)
  assert.equal(selectSessionSearchCandidates(file, live, { query: 'zzoldbodyneedle' }, blind, 200, false).threads.length, 0, 'fixture puts older body content beyond the hashed token cap')
  const rescued = await service.conversationSearch({ scope: 'all', query: 'zzoldbodyneedle' })
  assert.equal(rescued.matches[0]?.itemId, 'old-evidence', 'bounded live supplement recovers content omitted by token cap')
  live = Array.from({ length: 205 }, (_, index) => thread(`candidate-${index}`))
  syncSessionSearchIndex(file, live, blind)
  const candidateCap = await service.conversationSearch({ scope: 'all', query: 'needle' })
  assert.equal(candidateCap.truncated, true, 'candidate cap cannot masquerade as a complete result set')

  // Profiles can reuse ids; the live owner is authoritative over stale derived rows.
  live = [thread('unicode', 'Second profile unrelated text')]
  assert.equal((await service.conversationSearch({ scope: 'all', query: 'привет' })).matches.length, 0)
  const malicious = createConversationSearchService({ getThreads: () => [thread('allowed')],
    findCandidates: () => ({ threads: [thread('unpermitted')], truncated: true }) })
  assert.deepEqual((await malicious.conversationSearch({ scope: 'all', query: 'needle' })).matches, [])

  const many = thread('many', '')
  many.timeline = Array.from({ length: 10_005 }, (_, index) => ({ id: `row-${index}`, kind: 'assistant', text: 'needle' }))
  many.apiMessages = []
  const bounded = createConversationSearchService({ getThreads: () => [many] })
  const partial = await bounded.conversationSearch({ scope: 'thread', threadId: 'many', query: 'needle', limit: 1000 }, { allowedThreadId: 'many' })
  assert.equal(partial.matches.length, 100)
  assert.equal(partial.omitted, 9_900)
  assert.equal(partial.truncated, true)
  assert.equal(partial.matches[0].itemId, 'row-0', 'find navigation preserves timeline order')
  assert.equal((await bounded.conversationSearch({ scope: 'all', query: '   ' })).matches.length, 0)
  await assert.rejects(bounded.conversationSearch({ scope: 'all', query: 'x'.repeat(513) }), /at most 512/)
  await assert.rejects(bounded.conversationSearch({ scope: 'all', query: 'needle', threadId: 'many' }), /explicit current/)
  await assert.rejects(bounded.conversationSearch({ scope: 'all', query: 'needle', limit: NaN }), /limit/)
  console.log('Conversation search service policy, Unicode, targets, restart and bounded-state tests passed')
} finally {
  closeSessionSearchIndexes()
  rmSync(root, { recursive: true, force: true })
}
