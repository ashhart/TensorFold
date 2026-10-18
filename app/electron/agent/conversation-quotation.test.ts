import assert from 'node:assert/strict'
import { prepareMessageQuote, resolveMessageQuotes } from './conversation-quotation'
import { buildUserContent, contentText } from './content'
import { apiUserMessageRef, prepareThreadMessageEdit, stageThreadMessageEdit } from './message-edit'
import { validateMessageQuotes } from '../../shared/message-quotes'
import type { Thread } from './types'
const thread = { id: 'synthetic-chat', kind: 'chat', status: 'idle', apiMessages: [], timeline: [
  { id: 'answer', kind: 'assistant', text: 'An amber passage with a second line.', createdAt: 1 },
  { id: 'private-reasoning', kind: 'reasoning', text: 'Never quote this internal row.' },
] } as unknown as Thread
const input = { threadId: thread.id, itemId: 'answer', selection: 'amber passage' }
const quote = prepareMessageQuote(thread, input)
assert.equal(quote.sourceText, 'amber passage')
assert.deepEqual(resolveMessageQuotes(thread, [quote]), [quote])
const edited = { ...quote, text: 'My edited passage' }
assert.deepEqual(resolveMessageQuotes(thread, [edited]), [edited], 'user edits retain source provenance')
for (const bad of [{ ...quote, sourceRevision: 'b'.repeat(64) }, { ...quote, sourceText: 'not in that message' }, { ...quote, sourceItemId: 'missing' }]) assert.throws(() => resolveMessageQuotes(thread, [bad]), /changed|available/)
for (const bad of [{ ...input, threadId: 'other' }, { ...input, itemId: 'private-reasoning' }, { ...input, selection: 'not present' }, { ...input, extra: 'ignored?' }]) assert.throws(() => prepareMessageQuote(thread, bad), /Choose|available|Select/)
assert.throws(() => prepareMessageQuote({ ...thread, orchestratorHostId: 'another-host' }, input), /owning host/)
assert.throws(() => validateMessageQuotes([quote, quote]), /Duplicate/)
assert.throws(() => validateMessageQuotes([{ ...quote, text: '' }]), /empty/)
assert.throws(() => validateMessageQuotes([{ ...quote, text: 'a'.repeat(8001) }]), /large/)
assert.throws(() => validateMessageQuotes([{ ...quote, arbitrary: true }]), /Unsupported/)
const attachment = { id: 'file', name: 'synthetic.txt', kind: 'document' as const, mime: 'text/plain', text: 'Attached evidence' }
const content = buildUserContent('Explain this', [attachment], [edited])
assert.match(contentText(content), /"editedByUser":true/)
assert.match(contentText(content), /My edited passage[\s\S]*User message:\nExplain this[\s\S]*Attached evidence/)
assert.equal(buildUserContent('Plain old behavior', undefined), 'Plain old behavior')
thread.apiMessages = [{ role: 'assistant', content: thread.timeline[0].text! }, { role: 'user', content }]
thread.timeline.push({ id: 'reply', kind: 'user', text: 'Explain this', attachments: [attachment], quotedMessages: [edited], memorySource: 'user', apiMessageRef: apiUserMessageRef(thread.apiMessages.slice(0, 1), content, 0) })
const prep = prepareThreadMessageEdit(thread, 'reply')
const patch = stageThreadMessageEdit(thread, { ...prep, threadId: thread.id, text: 'Explain differently' }, 3)
assert.deepEqual(patch.timeline!.find(row => row.id === 'reply')!.quotedMessages, [edited])
assert.match(contentText(patch.apiMessages![1].content), /My edited passage[\s\S]*Explain differently/)
thread.timeline[0].text = 'Source changed'
assert.throws(() => resolveMessageQuotes(thread, [edited]), /changed/)
thread.timeline = []
assert.throws(() => resolveMessageQuotes(thread, [edited]), /available/)
console.log('Quotation snapshots validate same-chat provenance, bounded edited context, stale sources and matching sent-message history edits')
