import assert from 'node:assert/strict'
import { buildBackupArchive, encodeArchive, parseArchive } from './archive'
import { validateTimeline } from './record-validation'
import { buildUserContent } from '../agent/content'
import { prepareThreadMessageEdit, apiUserMessageRef } from '../agent/message-edit'
import { prepareMessageQuote } from '../agent/conversation-quotation'
import type { Thread } from '../agent/types'
const source: any = { id: 'synthetic-quoted-chat', title: 'Synthetic quotation backup', kind: 'chat', cwd: '', status: 'idle', providerId: 'local', model: 'synthetic', createdAt: 1, updatedAt: 2, apiMessages: [], timeline: [{ id: 'answer', kind: 'assistant', text: 'Original amber evidence.', createdAt: 1 }] }
const quote = { ...prepareMessageQuote(source, { threadId: source.id, itemId: 'answer' }), text: 'Edited amber evidence.' }
const content = buildUserContent('Explain the evidence', undefined, [quote])
source.timeline.push({ id: 'user', kind: 'user', text: 'Explain the evidence', quotedMessages: [quote], apiMessageRef: apiUserMessageRef([], content, 0), createdAt: 2 })
source.apiMessages.push({ role: 'user', content })
const archive = buildBackupArchive({ revision: 'synthetic', blobs: new Map(), data: { settings: {}, conversations: [source], projects: [], bots: [], drafts: [], attachments: [] } }, { conversationIds: [source.id] }, 3)
const restored = parseArchive(encodeArchive(archive)).archive.data.conversations[0] as unknown as Thread
assert.deepEqual(restored.timeline.find(row => row.id === 'user')!.quotedMessages, [quote], 'portable normalization cannot strip the quotation that remains in model history')
assert.deepEqual(restored.apiMessages[0].content, content)
assert.equal(prepareThreadMessageEdit(restored, 'user').expectedMessageText, 'Explain the evidence', 'restored quotation content and visible history remain editable together')
for (const item of [{ ...source.timeline[1], quotedMessages: [{ ...quote, text: '' }] }, { ...source.timeline[1], kind: 'assistant' }, { ...source.timeline[1], quotedMessages: [{ ...quote, arbitrary: true }] }]) assert.throws(() => validateTimeline([item]), /Invalid|empty|Unsupported/)
console.log('Portable backup round-trip preserves sent quotation provenance, edited markers and matching model context; malformed/non-user quote metadata refuses')
