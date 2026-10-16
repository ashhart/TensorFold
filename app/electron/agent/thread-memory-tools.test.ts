import assert from 'node:assert/strict'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { buildRetrievalIndex } from './retrieval-index'
import { executeTool } from './tools'
import { authoredMemoryMessages, harvestPostRunMemoryCandidates } from './memory-harvest'
import { auxiliaryToolContext } from './loop'
import type { Thread } from './types'
const dir = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-memory-tools-'))
const current: Thread = { id: 'current', providerId: null, model: null, title: 'Current history', cwd: dir, kind: 'chat', status: 'idle', createdAt: 1, updatedAt: 1, apiMessages: [{ role: 'user', content: 'CURRENT-SEEN-FACT' }], timeline: [] }
const previous: Thread = { ...current, id: 'previous', title: 'Saved prior chat', apiMessages: [{ role: 'user', content: 'SAVED-CROSS-CHAT-AMBER' }], compactions: [{ id: 'compact', summaryExcerpt: 'SAVED-COMPACTION-AMBER' } as any] }
const room: any = { id: 'room', name: 'Room', memorySummary: 'SAVED-ROOM-AMBER', memory: [{ id: 'learned', kind: 'fact', text: 'SAVED-ROOM-FACT-AMBER' }], sharedMemory: { blocks: { decisions: { content: 'SAVED-SHARED-ROOM-AMBER' } } }, finalText: 'SAVED-ROOM-FINAL-AMBER', events: [{ id: 'status', seat: 'lead', kind: 'status', text: 'LIVE-ROOM-STATUS' }, { id: 'prior-chat', seat: 'lead', kind: 'speak', text: 'SAVED-ROOM-CHAT-AMBER' }, { id: 'prior-final', seat: 'lead', kind: 'final', text: 'SAVED-ROOM-FINAL-AMBER' }, { id: 'prior-tool', seat: 'lead', kind: 'tool', tool: 'memory_read', text: 'SAVED-ROOM-TOOL-AMBER' }] }
const memory: any = { id: 'saved', kind: 'fact', projectId: 'project', text: 'SAVED-TYPED-AMBER', sourceIds: ['memory:saved'] }
try {
  await writeFile(path.join(dir, 'explicit.txt'), 'EXPLICIT-FILE-AMBER')
  for (const useExisting of [false, true]) for (const contribute of [false, true]) {
    let searchCalls = 0, filed = 0
    const policy = { useExisting, contribute }
    const context: any = { cwd: dir, currentThread: { ...current, memoryPolicy: policy }, memoryPolicy: policy, threads: [previous, current], projectMemory: [memory], rooms: [room], searchSessions: () => { searchCalls++; return { matches: [], omitted: 0, query: '' } }, fileSuggestion: () => { filed++ } }
    const read = await executeTool('session_read', { threadId: previous.id }, context)
    assert.equal(read.ok, useExisting)
    assert.equal(read.output.includes('SAVED-CROSS-CHAT-AMBER'), useExisting)
    const search = await executeTool('session_search', { query: 'AMBER', currentThreadId: 'forged-other-owner' }, context)
    assert.equal(search.ok, useExisting)
    assert.equal(searchCalls, useExisting ? 1 : 0, 'portable gate precedes injected host search callback')
    const suggestion = await executeTool('suggest_skill', { name: 'synthetic', body: 'Synthetic reviewed steps.' }, context)
    assert.equal(suggestion.ok, contribute)
    assert.equal(filed, contribute ? 1 : 0)
    const retrieval = await executeTool('retrieval_search', { query: 'AMBER', kinds: ['chat', 'summary', 'artifact'] }, context)
    assert.equal(retrieval.ok, true)
    assert.equal(retrieval.output.includes('SAVED-CROSS-CHAT-AMBER'), useExisting)
    assert.equal(retrieval.output.includes('SAVED-TYPED-AMBER'), useExisting)
    assert.equal(retrieval.output.includes('SAVED-ROOM-AMBER'), useExisting)
    assert.equal(retrieval.output.includes('SAVED-ROOM-CHAT-AMBER'), useExisting, 'prior room conversation cannot reintroduce a saved fact when recall is off')
    assert.equal(retrieval.output.includes('SAVED-ROOM-TOOL-AMBER'), useExisting)
    const explicit = await executeTool('read_file', { path: 'explicit.txt' }, context)
    assert.equal(explicit.ok, true); assert.match(explicit.output, /EXPLICIT-FILE-AMBER/)
    const child = auxiliaryToolContext(dir, context, new AbortController().signal)
    assert.equal((await executeTool('session_read', { threadId: previous.id }, child)).ok, useExisting, 'core read-only child retains the parent restriction')
    assert.equal((await executeTool('suggest_skill', { name: 'child', body: 'Synthetic.' }, child)).ok, contribute)
  }
  const index = buildRetrievalIndex({ savedMemoryAllowed: false, currentThreadId: current.id, threads: [current, previous], rooms: [room], projectMemory: [memory], records: [
    { id: 'connector:calendar:live', kind: 'summary', text: 'LIVE-CALENDAR-EVENT', title: 'Calendar' },
    { id: 'room:room:event:prebuilt-status', kind: 'summary', text: 'LIVE-PREBUILT-ROOM-STATUS', title: 'Current status', metadata: { roomEventKind: 'status' } },
    { id: 'room:room:event:prebuilt', kind: 'chat', text: 'PREBUILT-ROOM-CHAT-LEAK', title: 'Prior room conversation' },
    { id: 'memory:prebuilt', kind: 'artifact', text: 'PREBUILT-MEMORY-LEAK', title: 'Prior memory' },
    { id: 'artifact:disguised', sourceIds: ['thread:previous:timeline:old'], kind: 'artifact', text: 'CROSS-CHAT-ARTIFACT-LEAK', title: 'Old output' },
  ] })
  const indexed = JSON.stringify(index)
  assert.doesNotMatch(indexed, /SAVED-|PREBUILT-MEMORY-LEAK|PREBUILT-ROOM-CHAT-LEAK|CROSS-CHAT-ARTIFACT-LEAK/)
  assert.match(indexed, /CURRENT-SEEN-FACT/)
  assert.match(indexed, /LIVE-CALENDAR-EVENT/)
  assert.match(indexed, /LIVE-ROOM-STATUS/)
  assert.match(indexed, /LIVE-PREBUILT-ROOM-STATUS/)
  current.timeline = [
    { id: 'direct', kind: 'user', text: 'I always prefer lavenderwidget for synthetic demos.', memorySource: 'user' },
    { id: 'connector', kind: 'user', text: 'I always prefer externalwidget for synthetic demos.', memorySource: 'external' },
    { id: 'quoted', kind: 'user', text: '> I always prefer quotedwidget for synthetic demos.', memorySource: 'user' },
    { id: 'legacy', kind: 'user', text: 'I always prefer legacywidget for synthetic demos.' },
    { id: 'attached', kind: 'user', text: 'I always prefer attachedwidget for synthetic demos.', attachments: [{ id: 'attachment' } as any], memorySource: 'user' },
  ]
  assert.deepEqual(harvestPostRunMemoryCandidates(authoredMemoryMessages(current)), ['I always prefer lavenderwidget for synthetic demos.'])
  console.log('Four memory combinations: actual core tools/direct callbacks/child context/explicit files, source-aware retrieval and conservative authored proposal provenance passed')
} finally { await rm(dir, { recursive: true, force: true }) }
