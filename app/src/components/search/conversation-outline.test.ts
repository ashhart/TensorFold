import assert from 'node:assert/strict'
import { buildConversationOutline, outlinePage, OUTLINE_PAGE_SIZE, OUTLINE_SCAN_LIMIT } from './conversation-outline'
import { buildTurnRows } from '../../lib/turn-blocks'
import { transcriptSegments, includeTranscriptRange } from '../../lib/transcript-window'
import type { TimelineItem } from '../../lib/types'
const timeline: TimelineItem[] = [
  { id: 'request', kind: 'user', text: 'Original request 日本語' },
  { id: 'interim', kind: 'assistant', text: 'Intermediate work' },
  { id: 'read', kind: 'tool', tool: 'read_file', state: 'ok', output: 'Original document' },
  { id: 'changed', kind: 'tool', title: 'Changed a file', state: 'ok', changedPaths: ['synthetic.txt'] },
  { id: 'answer', kind: 'assistant', text: 'Final original answer' },
  { id: 'second', kind: 'user', text: 'Second original request' },
  { id: 'failed', kind: 'tool', title: 'Failed operation', state: 'error' },
  { id: 'denied', kind: 'tool', state: 'denied' },
  { id: 'trailing', kind: 'assistant', text: 'Not final because work follows' },
  { id: 'running', kind: 'tool', state: 'running' },
]
const annotations = { pinned: [{ id: 'pin', itemId: 'interim', label: 'Original decision label', done: false, createdAt: 1 }, { id: 'stale', itemId: 'removed', label: 'Old pin', done: false, createdAt: 1 }], notes: '' }
const original = structuredClone({ timeline, annotations })
const outline = buildConversationOutline(timeline, annotations)
assert.deepEqual(outline.entries.map(entry => [entry.itemId, entry.kind]), [['request', 'message'], ['interim', 'pin'], ['changed', 'outcome'], ['answer', 'answer'], ['second', 'message'], ['failed', 'outcome'], ['denied', 'outcome']])
assert.equal(outline.entries[1].label, 'Original decision label')
assert.deepEqual({ timeline, annotations }, original, 'Outline does not write annotations, history or messages')
assert.equal(outlinePage(outline.entries, { kind: 'pin', query: '', page: 100 }).entries[0]?.itemId, 'interim')
assert.equal(outlinePage(outline.entries, { kind: 'all', query: '日本語', page: 0 }).entries[0]?.itemId, 'request')
const streaming = [...timeline.slice(0, 6), { id: 'streaming-answer', kind: 'assistant' as const, text: 'Still generating' }]
assert.equal(buildConversationOutline(streaming, annotations, { activeAssistantId: 'streaming-answer' }).entries.some(entry => entry.itemId === 'streaming-answer'), false, 'An active streamed answer is not labeled final')
assert.equal(buildConversationOutline(streaming, annotations).entries.some(entry => entry.itemId === 'streaming-answer' && entry.kind === 'answer'), true, 'Completed final answers remain in the outline')
assert.equal(buildConversationOutline(streaming, { ...annotations, pinned: [...annotations.pinned, { id: 'live-pin', itemId: 'streaming-answer', label: 'Explicit user pin', createdAt: 1, done: false }] }, { activeAssistantId: 'streaming-answer' }).entries.find(entry => entry.itemId === 'streaming-answer')?.kind, 'pin', 'An explicit active-message pin is preserved without labeling it a final answer')
const many: TimelineItem[] = Array.from({ length: 12_000 }, (_, index) => ({ id: `message-${index}`, kind: index % 2 ? 'assistant' : 'user', text: `Synthetic item ${index}` }))
const large = buildConversationOutline(many, { pinned: [], notes: '' })
assert.equal(large.entries.length, 12_000)
for (const page of [0, 1, 99, 999]) assert.ok(outlinePage(large.entries, { kind: 'all', query: '', page }).entries.length <= OUTLINE_PAGE_SIZE, 'DOM pages remain bounded in long conversations')
const clipped = buildConversationOutline(Array.from({ length: OUTLINE_SCAN_LIMIT + 5 }, (_, index) => ({ id: `capped-${index}`, kind: 'user', text: 'Synthetic' })), { pinned: [], notes: '' })
assert.equal(clipped.truncated, true); assert.equal(clipped.entries.length, OUTLINE_SCAN_LIMIT)
const rows = buildTurnRows(timeline)
const workIndex = rows.findIndex(row => row.type === 'work' && row.turn.work.some(item => item.id === 'changed'))
assert.ok(workIndex >= 0, 'A tool outcome is a real exact target inside the collapsed activity row')
const segment = transcriptSegments(rows.length, includeTranscriptRange([], workIndex, workIndex + 1, rows.length))
assert.ok(segment.some(value => value.kind === 'visible' && value.start <= workIndex && value.end > workIndex))
console.log('Conversation outline passes original data, exact active IDs, pinned labels, answer/outcome classification, large page bounds, scan disclosure and collapsed work targets')
