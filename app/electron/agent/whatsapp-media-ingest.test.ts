// Run: jiti electron/agent/whatsapp-media-ingest.test.ts
//
// The PURE half of WhatsApp incoming-media ingest: reading a Baileys message's content to decide
// whether it carries media and, if so, what kind (image/video/audio/document) plus the caption /
// filename / mimetype the media cache needs to classify + name the decrypted bytes. The actual
// download+decrypt+write is I/O (covered by the crypto unit test for the decrypt core, and by
// Baileys for the real download); this proves the message-shape parsing deterministically.
import assert from 'node:assert/strict'
import { whatsappMediaDescriptor, whatsappMediaCaption } from './whatsapp-media-ingest.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('whatsapp-media-ingest tests:')

{
  // An image message -> image descriptor with its mimetype + caption.
  const d = whatsappMediaDescriptor({ imageMessage: { mimetype: 'image/jpeg', caption: 'sunset', mediaKey: new Uint8Array([1, 2]) } })
  assert.ok(d)
  assert.equal(d!.mediaType, 'image')
  assert.equal(d!.mimetype, 'image/jpeg')
  assert.equal(d!.caption, 'sunset')
  ok('whatsappMediaDescriptor: detects an imageMessage (type + mimetype + caption)')
}

{
  // Video / audio / document map to their types; documents surface the fileName.
  assert.equal(whatsappMediaDescriptor({ videoMessage: { mimetype: 'video/mp4', mediaKey: new Uint8Array([1]) } })!.mediaType, 'video')
  assert.equal(whatsappMediaDescriptor({ audioMessage: { mimetype: 'audio/ogg', mediaKey: new Uint8Array([1]) } })!.mediaType, 'audio')
  const doc = whatsappMediaDescriptor({ documentMessage: { mimetype: 'application/pdf', fileName: 'invoice.pdf', mediaKey: new Uint8Array([1]) } })
  assert.equal(doc!.mediaType, 'document')
  assert.equal(doc!.fileName, 'invoice.pdf')
  ok('whatsappMediaDescriptor: maps video/audio/document and surfaces a document fileName')
}

{
  // Stickers are images (WhatsApp encrypts them with the Image keys). PTT (voice notes) are audio.
  assert.equal(whatsappMediaDescriptor({ stickerMessage: { mimetype: 'image/webp', mediaKey: new Uint8Array([1]) } })!.mediaType, 'image')
  assert.equal(whatsappMediaDescriptor({ audioMessage: { mimetype: 'audio/ogg; codecs=opus', ptt: true, mediaKey: new Uint8Array([1]) } })!.mediaType, 'audio')
  ok('whatsappMediaDescriptor: stickers => image keys, voice notes (ptt) => audio')
}

{
  // A document wrapped in documentWithCaptionMessage (newer WhatsApp) is unwrapped to the document.
  const d = whatsappMediaDescriptor({ documentWithCaptionMessage: { message: { documentMessage: { mimetype: 'application/pdf', fileName: 'report.pdf', caption: 'Q3', mediaKey: new Uint8Array([1]) } } } })
  assert.ok(d)
  assert.equal(d!.mediaType, 'document')
  assert.equal(d!.fileName, 'report.pdf')
  assert.equal(d!.caption, 'Q3')
  ok('whatsappMediaDescriptor: unwraps documentWithCaptionMessage to the inner document')
}

{
  // A view-once image (viewOnceMessage / viewOnceMessageV2 wrapper) is unwrapped too.
  const d = whatsappMediaDescriptor({ viewOnceMessageV2: { message: { imageMessage: { mimetype: 'image/jpeg', mediaKey: new Uint8Array([1]) } } } })
  assert.equal(d!.mediaType, 'image')
  ok('whatsappMediaDescriptor: unwraps a view-once wrapper to the inner media')
}

{
  // Non-media messages (plain text, no media key) yield null — nothing to download.
  assert.equal(whatsappMediaDescriptor({ conversation: 'hi there' }), null)
  assert.equal(whatsappMediaDescriptor({ extendedTextMessage: { text: 'link' } }), null)
  assert.equal(whatsappMediaDescriptor(undefined), null)
  assert.equal(whatsappMediaDescriptor(null), null)
  // A media-shaped content with NO mediaKey can't be decrypted, so it is not a download candidate.
  assert.equal(whatsappMediaDescriptor({ imageMessage: { mimetype: 'image/jpeg' } }), null)
  ok('whatsappMediaDescriptor: returns null for text / empty / keyless content')
}

{
  // whatsappMediaCaption pulls the caption/filename for the event summary (caption wins, then
  // a document filename, else ''). Used so a received photo shows "[image] sunset" not a blank.
  assert.equal(whatsappMediaCaption({ imageMessage: { mimetype: 'image/jpeg', caption: 'sunset', mediaKey: new Uint8Array([1]) } }), 'sunset')
  assert.equal(whatsappMediaCaption({ documentMessage: { mimetype: 'application/pdf', fileName: 'invoice.pdf', mediaKey: new Uint8Array([1]) } }), 'invoice.pdf')
  assert.equal(whatsappMediaCaption({ audioMessage: { mimetype: 'audio/ogg', mediaKey: new Uint8Array([1]) } }), '')
  assert.equal(whatsappMediaCaption({ conversation: 'hi' }), '')
  ok('whatsappMediaCaption: caption first, then document fileName, else empty')
}

console.log(`\n${passed} checks passed.`)
