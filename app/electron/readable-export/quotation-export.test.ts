import assert from 'node:assert/strict'
import { buildReadableExport, READABLE_EXPORT_LIMITS } from './format'
import { createReadableExportService } from './service'
import type { Thread } from '../agent/types'
import type { ReadableExportRequest } from '../../shared/readable-export'

const secret = 'sk-test-' + '0Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St'
const address = ['reader', 'example.invalid'].join('@')
const quote = { id: 'quoted', sourceItemId: 'source-message', sourceRevision: 'a'.repeat(64), sourceText: 'UNSHOWN_ORIGINAL_TEXT', text: `VISIBLE_QUOTATION PrivateTerm ${address} password=${secret} </pre><script>throw 1</script>` }
const original = { ...quote, id: 'original', sourceItemId: 'other-source', sourceText: 'Original visible passage', text: 'Original visible passage' }
const thread = { id: 'synthetic-export', title: 'Synthetic quoted conversation', updatedAt: 1, timeline: [{ id: 'user', kind: 'user', text: 'Explain these passages.', quotedMessages: [quote, original] }] } as unknown as Thread
Object.defineProperty(thread, 'apiMessages', { get() { throw new Error('Never read model history for readable export') } })
const request: ReadableExportRequest = { threadId: thread.id, scope: 'selected', itemIds: ['user'], format: 'text', branch: 'active' }
for (const format of ['text', 'markdown', 'html'] as const) {
  const preview = buildReadableExport(thread, { ...request, format }, { terms: ['PrivateTerm'] })
  assert.match(preview.body, /VISIBLE_QUOTATION/)
  assert.match(preview.body, /source message: source-message/)
  assert.match(preview.body, /User-edited quotation/)
  assert.match(preview.body, /Original quotation/)
  assert.match(preview.body, /Original visible passage/)
  assert.match(preview.body, /Explain these passages/)
  for (const hidden of [secret, address, 'PrivateTerm', 'UNSHOWN_ORIGINAL_TEXT']) assert.equal(preview.body.includes(hidden), false, 'The same redaction applies to visible quotes without exporting hidden originals')
  if (format === 'html') { assert.match(preview.body, /&lt;script&gt;/); assert.doesNotMatch(preview.body, /<(?:script|img|iframe|a|link)\b/i) }
  assert.equal(preview.bytes, Buffer.byteLength(preview.body))
}
let serial = 0
const service = createReadableExportService({ readThread: () => thread, scopeKey: () => 'synthetic-profile', newId: () => `quote-approval-${++serial}` })
for (const key of ['text', 'sourceText', 'sourceItemId', 'sourceRevision'] as const) {
  const snapshot = service.preview(request, 'owner'), before = quote[key]
  quote[key] = key === 'sourceRevision' ? 'b'.repeat(64) : `${quote[key]} changed`
  assert.throws(() => service.claim({ approvalId: snapshot.approvalId }, 'owner'), /changed/, `A changed quote ${key} invalidates a preview even when conversation text/time is unchanged`)
  quote[key] = before
}
const claim = service.claim({ approvalId: service.preview(request, 'owner').approvalId }, 'owner')
const packet = service.finalize(claim.claimId, 'owner')
quote.text += ' changed before atomic rename'
assert.throws(() => packet.assertCurrent(), /changed/, 'Quote text is checked again before the atomic destination rename')
quote.text = 'VISIBLE_QUOTATION'
const excessive = { ...thread, timeline: [{ ...thread.timeline[0], text: 'x'.repeat(READABLE_EXPORT_LIMITS.sourceCharacters - 100) }] } as Thread
assert.throws(() => buildReadableExport(excessive, request), /limit/, 'Quoted blocks and provenance labels count towards the source-size limit')
assert.throws(() => buildReadableExport({ ...thread, timeline: [{ ...thread.timeline[0], quotedMessages: [{ ...quote, text: 'x'.repeat(8001) }] }] }, request), /large/)
assert.throws(() => buildReadableExport({ ...thread, timeline: [{ ...thread.timeline[0], kind: 'assistant' }] }, request), /Only user/)
console.log('Actual readable exports retain bounded visible original/edited quotes and source IDs in all formats, redact/escape them, omit hidden originals, and invalidate stale preview/atomic-save approvals without reading raw history')
