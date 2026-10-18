import assert from 'node:assert/strict'
import { acceptAttachmentExtraction, reviewAttachmentExtraction } from '../../src/lib/attachments'
import { normalizeAttachments } from './attachment-policy'
import { buildUserContent } from './content'
import { apiUserMessageRef, prepareThreadMessageEdit, stageThreadMessageEdit } from './message-edit'
import type { Attachment, Thread } from './types'

// Cross the real extraction-review -> trusted host normalization -> API content
// -> sent-history edit seams. Synthetic data only; no OCR helper or app launches.
const image: Attachment = { id: 'handwriting-fixture', kind: 'image', name: 'sample-note.png', mime: 'image/png', status: 'ready', dataUrl: 'data:image/png;base64,SYNTHETIC_PRIVATE_PREVIEW' }
const review = reviewAttachmentExtraction(image, { ok: true, text: 'Meet at 7?', confidence: 0.42, warnings: ['Synthetic uncertain digit'], pageCount: 1, processedPages: 1, truncated: false })
assert.throws(() => normalizeAttachments([review]), /not ready|not been accepted/)
assert.throws(() => buildUserContent('Read this note', [review]), /not ready/)
const accepted = acceptAttachmentExtraction(review, 'Meet at [unclear: 7 or 9]')
const normalized = normalizeAttachments([accepted])!
assert.equal(normalized[0].kind, 'document')
assert.equal(normalized[0].dataUrl, undefined)
assert.equal(normalized[0].previewDataUrl, undefined, 'private source image preview is never retained in model/timeline history after transcript acceptance')
assert.equal(normalized[0].extraction?.state, 'accepted')
assert.equal(normalized[0].extraction?.text, 'Meet at [unclear: 7 or 9]')
assert.equal(normalized[0].extraction?.confidence, 0.42)
const content = buildUserContent('Read this note', normalized)
assert.match(String(content), /sample-note\.png[\s\S]*User-reviewed OCR transcript[\s\S]*Meet at \[unclear: 7 or 9\]/)
assert.ok(!String(content).includes('SYNTHETIC_PRIVATE_PREVIEW'))
const ref = apiUserMessageRef([], content, 0)
const thread: Thread = { id: 'ocr-history-fixture', title: 'Synthetic', kind: 'chat', cwd: '', providerId: null, model: null, status: 'idle', createdAt: 1, updatedAt: 1, apiMessages: [{ role: 'user', content }, { role: 'assistant', content: 'Earlier answer' }], timeline: [{ id: 'u', kind: 'user', text: 'Read this note', attachments: normalized, apiMessageRef: ref }, { id: 'a', kind: 'assistant', text: 'Earlier answer' }] }
const patch = stageThreadMessageEdit(thread, { threadId: thread.id, ...prepareThreadMessageEdit(thread, 'u'), text: 'Summarize the uncertain note' })
assert.match(String(patch.apiMessages![0].content), /^Summarize the uncertain note[\s\S]*Meet at \[unclear: 7 or 9\]/)
assert.equal(patch.timeline![0].attachments?.[0].extraction?.text, normalized[0].extraction?.text)
assert.equal(patch.messageEditRevisions?.[0].timeline[0].attachments?.[0].extraction?.text, normalized[0].extraction?.text)
assert.throws(() => normalizeAttachments([{ ...accepted, text: accepted.text!.replace('7 or 9', '3 or 4') }]), /does not match/, 'host rejects edited document text whose claimed accepted transcript differs')
console.log('Accepted OCR cross-seam: unreviewed send refused, exact corrected/uncertain transcript and filename reach API/timeline, raw preview omitted, edited sent history remains matched and forged acceptance mismatch rejects')
