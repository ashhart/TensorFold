// Run: jiti src/lib/attachments.test.ts
import assert from 'node:assert/strict'
import {
  fileToAttachment,
  filesToAttachments,
  appendAttachments,
  attachmentsForSend,
  replaceAttachment,
  reviewAttachmentExtraction,
  acceptAttachmentExtraction,
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
  const first = new File(['first'], 'first.txt', { type: 'text/plain' })
  const broken = new File(['bad'], 'unreadable.txt', { type: 'text/plain' })
  Object.defineProperty(broken, 'slice', { value: () => { throw new Error('selected file is unreadable') } })
  const third = new File(['third'], 'third.txt', { type: 'text/plain' })
  const attachments = await filesToAttachments([first, broken, third])
  assert.deepEqual(attachments.map((item) => item.name), ['first.txt', 'unreadable.txt', 'third.txt'], 'every selected file needs its own visible row')
  assert.equal((attachments[1] as any).status, 'error')
  assert.match((attachments[1] as any).error, /unreadable/)
  assert.equal(attachments[0].text, 'first')
  assert.equal(attachments[2].text, 'third')
  ok('one unreadable selection stays visible and cannot discard successful siblings')
}


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
    new File(['x'], `file-${index}.txt`, { type: 'text/plain' }),
  )
  const attachments = await filesToAttachments(files)
  assert.equal(attachments.length, files.length)
  assert.deepEqual(attachments.map((item) => item.name), files.map((file) => file.name))
  assert.equal(attachmentsForSend(attachments).length, MAX_ATTACHMENT_FILES)
  assert.ok(attachments.slice(MAX_ATTACHMENT_FILES).every((item) => item.status === 'error' && /file limit/.test(item.error ?? '')))
  ok('batch ingestion bounds file reads while keeping each overflow filename and error visible')
}

{
  const make = (index: number) => ({ id: `a${index}`, kind: 'document' as const, name: `a${index}`, mime: 'text/plain', text: 'x' })
  const current = Array.from({ length: 15 }, (_, index) => make(index))
  const next = Array.from({ length: 15 }, (_, index) => make(index + 15))
  const merged = appendAttachments(current, next)
  assert.equal(merged.length, current.length + next.length)
  assert.equal(attachmentsForSend(merged).length, MAX_ATTACHMENT_FILES)
  assert.deepEqual(merged.slice(0, current.length), current)
  assert.ok(merged.slice(MAX_ATTACHMENT_FILES).every((item) => item.status === 'error'))
  ok('repeated selections preserve every filename while enforcing the cumulative send limit')
}


{
  const states: Array<{ id: string; name: string; status: string | undefined }> = []
  const selected = [new File(['one'], 'one.txt', { type: 'text/plain' }), new File(['two'], 'two.txt', { type: 'text/plain' }), new File(['three'], 'three.txt', { type: 'text/plain' })]
  const attachments = await filesToAttachments(selected, { onUpdate: (item) => states.push({ id: item.id, name: item.name, status: item.status }) })
  assert.deepEqual(states.slice(0, 3).map((item) => item.status), ['processing', 'processing', 'processing'])
  for (const item of attachments) assert.equal(states.find((state) => state.name === item.name)?.id, item.id)
  assert.deepEqual(attachments.map((item) => item.text), ['one', 'two', 'three'])
  ok('all selected rows are published before reads with stable IDs across completion')
}

{
  const image = { id: 'ocr', name: 'handwriting.png', mime: 'image/png', kind: 'image' as const, dataUrl: 'data:image/png;base64,AA', status: 'ready' as const }
  const review = reviewAttachmentExtraction(image, { ok: true, text: 'meet on Fridoy', confidence: 0.81, warnings: ['Review handwriting against the original.'], pageCount: 1, processedPages: 1 })
  assert.equal(review.status, 'review')
  assert.equal(attachmentsForSend([review]).length, 0, 'unaccepted OCR never reaches model content')
  assert.equal(review.text, undefined)
  assert.throws(() => acceptAttachmentExtraction(review, '  '), /No readable text/)
  const accepted = acceptAttachmentExtraction(review, 'meet on Friday')
  assert.equal(accepted.kind, 'document')
  assert.equal(accepted.dataUrl, undefined, 'accepted transcript is usable by text-only transports')
  assert.equal(accepted.extraction?.text, 'meet on Friday')
  assert.match(accepted.text ?? '', /User-reviewed OCR transcript/)
  assert.match(accepted.text ?? '', /meet on Friday/)
  assert.doesNotMatch(accepted.text ?? '', /Fridoy/)
  assert.equal(attachmentsForSend([accepted]).length, 1)
  ok('OCR remains editable review-only until explicitly accepted and is labelled uncertain in model history')
}

{
  const abort = new AbortController()
  abort.abort()
  const files = [new File(['first'], 'first.txt', { type: 'text/plain' }), new File(['second'], 'second.txt', { type: 'text/plain' })]
  const attachments = await filesToAttachments(files, { signal: abort.signal })
  assert.equal(attachments.length, 2)
  assert.ok(attachments.every((item) => item.status === 'error' && /cancelled/.test(item.error ?? '')))
  ok('cancelled ingestion reports individual named failures instead of dropping them')
}

{
  const current = Array.from({ length: MAX_ATTACHMENT_FILES }, (_, index) => ({ id: `ready-${index}`, kind: 'document' as const, name: `ready-${index}.txt`, mime: 'text/plain', text: 'ready', status: 'ready' as const }))
  const processing = { id: 'pending', kind: 'document' as const, name: 'pending.txt', mime: 'text/plain', status: 'processing' as const }
  const all = appendAttachments(current, [processing])
  const resolved = replaceAttachment(all, { ...processing, status: 'ready', text: 'later' })
  assert.equal(resolved.length, MAX_ATTACHMENT_FILES + 1)
  assert.equal(resolved.at(-1)?.status, 'error')
  assert.equal(attachmentsForSend(resolved).length, MAX_ATTACHMENT_FILES)
  assert.equal(replaceAttachment(current, { ...processing, status: 'ready', text: 'removed' }), current, 'a late result cannot recreate a removed row')
  ok('late processing completions enforce draft limits and cannot recreate removed files')
}

{
  const largeImage = (id: string) => ({ id, kind: 'image' as const, name: `${id}.png`, mime: 'image/png', dataUrl: `data:image/png;base64,${'A'.repeat(Math.ceil(4 * 1024 * 1024 / 3) * 4)}`, status: 'ready' as const })
  const largeUnicodeDoc = (index: number) => ({ id: `unicode-${index}`, kind: 'document' as const, name: `unicode-${index}.txt`, mime: 'text/plain', text: '界'.repeat(200_000), status: 'ready' as const })
  const merged = appendAttachments([], [largeImage('one'), largeImage('two'), ...Array.from({ length: 18 }, (_, index) => largeUnicodeDoc(index))])
  assert.ok(attachmentsForSend(merged).length < 20, 'mixed images and UTF-8 documents must obey the host byte limit, not only a character count')
  assert.equal(merged.length, 20, 'excess payloads stay individually visible')
  assert.equal(merged.at(-1)?.status, 'error')
  ok('mixed image and non-ASCII payloads are rejected per file before the host aggregate limit')
}

console.log(`\n${passed} passed`)
