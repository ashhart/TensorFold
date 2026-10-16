// Run: jiti src/lib/attachments.test.ts
import assert from 'node:assert/strict'
import {
  fileToAttachment,
  filesToAttachments,
  appendAttachments,
  imageDimensionsFromHeader,
  MAX_ATTACHMENT_FILES,
  MAX_DOC_CHARS,
  MAX_DOC_READ_BYTES,
  MAX_IMAGE_DECODED_PIXELS,
  MAX_IMAGE_RAW_BYTES,
  MAX_IMAGE_SOURCE_BYTES,
} from './attachments'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function fakeSize(file: File, size: number): File {
  Object.defineProperty(file, 'size', { configurable: true, value: size })
  return file
}

console.log('attachment ingestion safety tests:')

{
  const file = new File(['x'.repeat(MAX_DOC_READ_BYTES + 100)], 'large.txt', { type: 'text/plain' })
  const originalSlice = file.slice.bind(file)
  let sliceEnd = -1
  Object.defineProperty(file, 'slice', {
    configurable: true,
    value: (start?: number, end?: number, type?: string) => {
      sliceEnd = Number(end)
      return originalSlice(start, end, type)
    },
  })
  Object.defineProperty(file, 'text', { configurable: true, value: async () => { throw new Error('whole file read') } })
  const attachment = await fileToAttachment(file)
  assert.equal(sliceEnd, MAX_DOC_READ_BYTES)
  assert.equal(attachment.kind, 'document')
  assert.ok((attachment.text?.length ?? 0) <= MAX_DOC_CHARS + 20)
  assert.match(attachment.text ?? '', /\[truncated\]$/)
  ok('large text files are sliced before decoding and character-capped')
}

{
  let bitmapCalls = 0
  let rawReads = 0
  const file = fakeSize(new File(['tiny'], 'huge.png', { type: 'image/png' }), MAX_IMAGE_SOURCE_BYTES + 1)
  const attachment = await fileToAttachment(file, {
    createBitmap: async () => { bitmapCalls++; throw new Error('must not decode') },
    readDataUrl: async () => { rawReads++; return 'data:image/png;base64,AA' },
  })
  assert.equal(attachment.kind, 'document')
  assert.match(attachment.text ?? '', /source exceeds/)
  assert.equal(bitmapCalls, 0)
  assert.equal(rawReads, 0)
  ok('oversized encoded images are rejected before decode or whole-file fallback')
}

{
  const png = new Uint8Array(24)
  png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const view = new DataView(png.buffer)
  view.setUint32(16, 10_000)
  view.setUint32(20, 10_000)
  const file = new File([png], 'bomb.png', { type: 'image/png' })
  assert.deepEqual(await imageDimensionsFromHeader(file), { width: 10_000, height: 10_000 })
  let bitmapCalls = 0
  const attachment = await fileToAttachment(file, {
    createBitmap: async () => { bitmapCalls++; return { width: 1, height: 1 } },
  })
  assert.equal(attachment.kind, 'document')
  assert.match(attachment.text ?? '', new RegExp(`${MAX_IMAGE_DECODED_PIXELS}-pixel`))
  assert.equal(bitmapCalls, 0)
  ok('PNG dimension bombs are rejected from a bounded header before browser decode')
}

{
  let bitmapCalls = 0
  let rawReads = 0
  const file = new File(['not a real image'], 'unknown.heic', { type: 'image/heic' })
  const attachment = await fileToAttachment(file, {
    createBitmap: async () => { bitmapCalls++; return { width: 1, height: 1 } },
    readDataUrl: async () => { rawReads++; return 'data:image/png;base64,AA' },
  })
  assert.equal(attachment.kind, 'document')
  assert.match(attachment.text ?? '', /could not be safely validated/)
  assert.equal(bitmapCalls, 0, 'unknown compressed formats are never decoded before dimensions can be bounded')
  assert.equal(rawReads, 0)
  ok('unparsed image formats fail closed before browser decode')
}

{
  let rawReads = 0
  let closed = false
  const png = new Uint8Array(24)
  png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const view = new DataView(png.buffer)
  view.setUint32(16, 1)
  view.setUint32(20, 1)
  const file = new File([png], 'decoded-bomb.png', { type: 'image/png' })
  const attachment = await fileToAttachment(file, {
    createBitmap: async () => ({ width: 10_000, height: 10_000, close: () => { closed = true } }),
    readDataUrl: async () => { rawReads++; return 'data:image/png;base64,AA' },
  })
  assert.equal(attachment.kind, 'document')
  assert.match(attachment.text ?? '', /pixel safety limit/)
  assert.equal(rawReads, 0, 'decoded-pixel rejection must not fall back to the original')
  assert.equal(closed, true)
  ok('post-decode pixel limits close the bitmap and cannot be bypassed by raw fallback')
}

{
  let rawReads = 0
  const png = new Uint8Array(24)
  png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  const view = new DataView(png.buffer)
  view.setUint32(16, 1)
  view.setUint32(20, 1)
  const file = fakeSize(new File([png], 'decode-fails.png', { type: 'image/png' }), MAX_IMAGE_RAW_BYTES + 1)
  const attachment = await fileToAttachment(file, {
    createBitmap: async () => { throw new Error('decode failed') },
    readDataUrl: async () => { rawReads++; return 'data:image/png;base64,AA' },
  })
  assert.equal(attachment.kind, 'document')
  assert.match(attachment.text ?? '', /decode failed/)
  assert.equal(rawReads, 0)
  ok('decode failures only use the original-image fallback below its independent byte cap')
}

{
  const files = Array.from({ length: MAX_ATTACHMENT_FILES + 3 }, (_, index) =>
    new File(['x'], `file-${index}.bin`, { type: 'application/octet-stream' }),
  )
  const attachments = await filesToAttachments(files)
  assert.equal(attachments.length, MAX_ATTACHMENT_FILES)
  assert.match(attachments.at(-1)?.text ?? '', /4 attachments not attached/)
  ok('batch ingestion caps file count without materializing or processing the overflow')
}

{
  const make = (index: number) => ({ id: `a${index}`, kind: 'document' as const, name: `a${index}`, mime: 'text/plain', text: 'x' })
  const current = Array.from({ length: 15 }, (_, index) => make(index))
  const next = Array.from({ length: 15 }, (_, index) => make(index + 15))
  const merged = appendAttachments(current, next)
  assert.equal(merged.length, MAX_ATTACHMENT_FILES)
  assert.deepEqual(merged.slice(0, current.length), current)
  ok('repeated picker operations cannot bypass the cumulative attachment-count limit')
}

console.log(`\n${passed} passed`)
