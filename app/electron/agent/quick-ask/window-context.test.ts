import assert from 'node:assert/strict'
import { capturedWindowAttachments, inferWindowKind } from './window-context'
import { qualificationImage } from '../model-qualification/fixture-image'

const png = qualificationImage().split(',')[1]
const context = { app: 'TextEdit', title: 'Local sample.txt', kind: 'document', text: 'Visible sample text', png }
const items = capturedWindowAttachments(context)
assert.equal(items.length, 2)
assert.equal(items[0].kind, 'image')
assert.equal(items[1].kind, 'document')
assert.equal(items[0].id.replace(/-image$/, ''), items[1].id.replace(/-text$/, ''))
assert.match(items[1].text!, /OCR can omit or misread content/)
assert.match(items[1].text!, /Visible sample text/)
assert.equal(capturedWindowAttachments({ ...context, text: '' }).length, 1)
assert.ok(capturedWindowAttachments({ ...context, title: 'x'.repeat(520) }).every(item => item.name.length <= 200))
for (const value of [null, {}, { ...context, kind: 'display' }, { ...context, text: 'x'.repeat(32_001) },
  { ...context, png: png + '\n' }, { ...context, png: Buffer.from('<svg/>').toString('base64') },
  { ...context, png: Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString('base64') }]) {
  assert.throws(() => capturedWindowAttachments(value))
}
const oversized = Buffer.from(png, 'base64'); oversized.writeUInt32BE(1801, 16)
assert.throws(() => capturedWindowAttachments({ ...context, png: oversized.toString('base64') }), /PNG/)
assert.equal(inferWindowKind('Sample - Safari'), 'web')
assert.equal(inferWindowKind('Sample.png'), 'image')
assert.equal(inferWindowKind('Sample.txt - TextEdit'), 'document')
assert.equal(inferWindowKind('Untitled ordinary window'), 'window')
console.log('Selected-window context rejects malformed, oversized and non-PNG data; pairs bounded visible text with one image.')
