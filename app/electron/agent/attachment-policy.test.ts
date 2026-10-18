import assert from 'node:assert/strict'
import {
  HOST_MAX_ATTACHMENT_FILES,
  HOST_MAX_DOCUMENT_CHARS,
  HOST_MAX_IMAGE_BYTES,
  HOST_MAX_IMAGE_PIXELS,
  normalizeAttachments,
} from './attachment-policy'

const png = Buffer.alloc(24)
png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
png.writeUInt32BE(1, 16)
png.writeUInt32BE(1, 20)
const dataUrl = `data:image/png;base64,${png.toString('base64')}`

const valid = normalizeAttachments([
  { id: 'doc', kind: 'document', name: '../../notes\n.txt', mime: ' NOT A MIME ', text: 'safe text', bytes: 999999 },
  { id: 'img', kind: 'image', name: 'pixel.png', mime: 'image/png', dataUrl },
])!
assert.equal(valid.length, 2)
assert.equal(valid[0].bytes, Buffer.byteLength('safe text'), 'untrusted size metadata is recomputed')
assert.equal(valid[0].name, 'notes .txt', 'filenames are reduced to a safe display-only leaf')
assert.equal(valid[0].mime, 'text/plain', 'invalid document MIME metadata is normalized')
assert.equal(valid[1].dataUrl, dataUrl)
assert.equal(valid[1].bytes, png.length, 'base64 padding is excluded from recomputed decoded bytes')

for (const url of [
  'http://127.0.0.1/metadata',
  'http://169.254.169.254/latest/meta-data',
  'https://example.com/image.png',
  'file:///etc/passwd',
  'private://secrets/image',
  'blob:https://example.com/id',
  'data:image/svg+xml;base64,PHN2Zz4=',
]) {
  assert.throws(() => normalizeAttachments([{ kind: 'image', name: 'bad', dataUrl: url }]), /supported base64 data URL|missing or oversized/)
}
assert.throws(() => normalizeAttachments('not-an-array'), /must be an array/)
assert.throws(() => normalizeAttachments(null), /must be an array/)
assert.throws(() => normalizeAttachments(Array.from({ length: HOST_MAX_ATTACHMENT_FILES + 1 }, () => ({ kind: 'document', text: 'x' }))), /file host limit/)
assert.throws(() => normalizeAttachments([{ kind: 'document', text: 'x'.repeat(HOST_MAX_DOCUMENT_CHARS + 1) }]), /oversized/)
const rendererSizedUnicode = 'ࠀ'.repeat(HOST_MAX_DOCUMENT_CHARS)
const unicodeDocument = normalizeAttachments([{ kind: 'document', text: rendererSizedUnicode }])![0]
assert.equal(unicodeDocument.bytes, Buffer.byteLength(rendererSizedUnicode), 'renderer-sized non-ASCII text remains host-accepted')

const bomb = Buffer.from(png)
bomb.writeUInt32BE(HOST_MAX_IMAGE_PIXELS, 16)
bomb.writeUInt32BE(2, 20)
assert.throws(() => normalizeAttachments([{ kind: 'image', dataUrl: `data:image/png;base64,${bomb.toString('base64')}` }]), /dimensions exceed/)

const mislabeled = `data:image/jpeg;base64,${png.toString('base64')}`
assert.throws(() => normalizeAttachments([{ kind: 'image', dataUrl: mislabeled }]), /do not match/)

const atImageLimit = Buffer.alloc(HOST_MAX_IMAGE_BYTES)
png.copy(atImageLimit)
const atImageLimitUrl = `data:image/png;base64,${atImageLimit.toString('base64')}`
assert.equal(normalizeAttachments([{ kind: 'image', dataUrl: atImageLimitUrl }])![0].bytes, HOST_MAX_IMAGE_BYTES)
const overImageLimit = Buffer.alloc(HOST_MAX_IMAGE_BYTES + 1)
png.copy(overImageLimit)
assert.throws(
  () => normalizeAttachments([{ kind: 'image', dataUrl: `data:image/png;base64,${overImageLimit.toString('base64')}` }]),
  /decoded bytes exceed/,
)
assert.throws(
  () => normalizeAttachments(Array.from({ length: 4 }, (_, index) => ({ kind: 'image', name: `${index}.png`, dataUrl: atImageLimitUrl }))),
  /aggregate host payload limit/,
)


for (const status of ['processing', 'error', 'needs-reselection', 'review', 'unknown']) {
  assert.throws(() => normalizeAttachments([{ kind: 'document', text: 'not ready', status }]), /not ready/, `the trusted host must reject ${status} rows`)
}
assert.throws(() => normalizeAttachments([{ kind: 'document', text: 'unreviewed OCR', extraction: { state: 'review', text: 'unreviewed OCR' } }]), /not been accepted/)
assert.throws(() => normalizeAttachments([{ kind: 'image', dataUrl, extraction: { state: 'accepted', text: 'OCR' } }]), /must be attached as a document/)
assert.equal(normalizeAttachments([{ kind: 'document', status: 'ready', text: '[User-reviewed OCR transcript. Recognition may contain errors.]\ncorrected handwriting', extraction: { state: 'accepted', method: 'apple-vision', text: 'corrected handwriting' } }])![0].text?.includes('corrected handwriting'), true)

console.log('host attachment policy tests passed')
