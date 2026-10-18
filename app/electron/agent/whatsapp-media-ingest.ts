// WhatsApp incoming-media ingest: turn an ENCRYPTED Baileys media message into a LOCAL cached
// file the vision tool can read by path. Two halves:
//
//   • PURE (whatsappMediaDescriptor / whatsappMediaCaption) — read a message's content to decide
//     whether it carries media, its kind (image/video/audio/document → which HKDF keys), and the
//     caption/filename/mimetype the cache needs. Unwraps the view-once / documentWithCaption /
//     edited wrappers WhatsApp nests media in. Deterministic + unit-tested (no Baileys import).
//   • I/O (cacheWhatsAppMediaMessage) — get the DECRYPTED bytes and write them to the shared media
//     cache (~/.vontra/cache/...) via cacheBytes, returning the local path(s). The decrypted bytes
//     come from Baileys' own downloadMediaMessage when available (it streams+decrypts the real CDN
//     blob); the pure decryptWhatsAppMedia primitive (whatsapp-media.ts) is the tested crypto core
//     and the documented fallback when a raw encrypted buffer + mediaKey is in hand instead.
//
// HONESTY: a real device→encrypted-blob→decrypted-image E2E can't be produced without a paired
// account, so the live download path is exercised only against Baileys at runtime; the crypto and
// the message-shape parsing here are what's deterministically verified.

import { cacheBytes, type MediaCacheResult } from './media-cache.ts'
import { decryptWhatsAppMedia, type WhatsAppMediaType } from './whatsapp-media.ts'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MessageContent = any

export interface WhatsAppMediaDescriptor {
  /** Which media type's HKDF keys decrypt this (image|video|audio|document). */
  mediaType: WhatsAppMediaType
  /** The proto content key the media lives under (e.g. 'imageMessage'). */
  contentKey: string
  /** MIME type from the media node, used to classify the cache bucket + extension. */
  mimetype: string | null
  /** A document's original filename, if any (drives the cached file's extension). */
  fileName: string | null
  /** The caption (image/video) so the event summary isn't blank. */
  caption: string | null
}

// The proto content keys that carry encrypted media, mapped to the HKDF key family they use.
// Stickers/voice-notes reuse the Image/Audio keys respectively (matches WhatsApp/Baileys).
const MEDIA_CONTENT_KEYS: Record<string, WhatsAppMediaType> = {
  imageMessage: 'image',
  stickerMessage: 'image',
  videoMessage: 'video',
  audioMessage: 'audio', // includes ptt (voice notes)
  documentMessage: 'document',
}

// Wrapper messages that nest the real content one level down (view-once, document-with-caption,
// edited, ephemeral). We unwrap these before looking for a media node.
const WRAPPER_KEYS = [
  'viewOnceMessage',
  'viewOnceMessageV2',
  'viewOnceMessageV2Extension',
  'documentWithCaptionMessage',
  'editedMessage',
  'ephemeralMessage',
]

/** Unwrap nested wrapper messages to the innermost content. Pure; bounded depth. */
function unwrapContent(content: MessageContent): MessageContent {
  let cur = content
  for (let depth = 0; depth < 6 && cur && typeof cur === 'object'; depth++) {
    const wrapper = WRAPPER_KEYS.find((k) => cur[k]?.message)
    if (!wrapper) break
    cur = cur[wrapper].message
  }
  return cur
}

/**
 * Inspect a Baileys message content; return a descriptor when it carries DECRYPTABLE media
 * (a known media node WITH a mediaKey), else null. Unwraps view-once / document-with-caption /
 * edited / ephemeral wrappers first. Pure (no I/O, no Baileys import).
 */
export function whatsappMediaDescriptor(content: MessageContent): WhatsAppMediaDescriptor | null {
  const inner = unwrapContent(content)
  if (!inner || typeof inner !== 'object') return null
  for (const [contentKey, mediaType] of Object.entries(MEDIA_CONTENT_KEYS)) {
    const node = inner[contentKey]
    if (!node || typeof node !== 'object') continue
    // Without a mediaKey there is nothing to decrypt — not a download candidate.
    if (node.mediaKey == null || (typeof node.mediaKey !== 'string' && (node.mediaKey as { length?: number }).length === 0)) continue
    return {
      mediaType,
      contentKey,
      mimetype: typeof node.mimetype === 'string' ? node.mimetype : null,
      fileName: typeof node.fileName === 'string' ? node.fileName : null,
      caption: typeof node.caption === 'string' ? node.caption : null,
    }
  }
  return null
}

/** Caption for the event summary: a caption wins, then a document filename, else ''. Pure. */
export function whatsappMediaCaption(content: MessageContent): string {
  const d = whatsappMediaDescriptor(content)
  if (!d) return ''
  return d.caption || d.fileName || ''
}

// ---- I/O: download (decrypt) + write into the media cache ----------------------

export interface CacheWhatsAppMediaOptions {
  cacheRoot: string
  capBytes?: number
  /**
   * Resolve the DECRYPTED media bytes for a message. In the host this is Baileys'
   * downloadMediaMessage(message, 'buffer', {}) — which streams the real CDN blob and decrypts it
   * internally. Injectable so the wiring is testable and so a future caller can swap in the pure
   * primitive path (downloadViaPrimitive) when it holds a raw encrypted buffer instead.
   */
  fetchDecryptedBytes: () => Promise<Uint8Array>
  /** Recheck current account/scope ownership after asynchronous decryption and before cache admission. */
  shouldCache?: () => boolean
}

/**
 * Decrypt + cache one WhatsApp media message, returning the local cache path (or null). The
 * decrypted bytes come from opts.fetchDecryptedBytes (Baileys' downloader in the host); they are
 * written to the shared media cache via cacheBytes, classified by the message's mimetype/filename
 * so the vision tool reads an image by path. Never throws — a failed download/decrypt yields null
 * so a media message still ingests as text. I/O; the crypto core is unit-tested separately.
 */
export async function cacheWhatsAppMediaMessage(
  descriptor: WhatsAppMediaDescriptor,
  opts: CacheWhatsAppMediaOptions,
): Promise<string | null> {
  try {
    if (opts.shouldCache && !opts.shouldCache()) return null
    const bytes = await opts.fetchDecryptedBytes()
    if (opts.shouldCache && !opts.shouldCache()) return null
    if (!bytes || bytes.byteLength === 0) return null
    const result: MediaCacheResult = await cacheBytes(
      { data: bytes, filename: descriptor.fileName ?? null, contentType: descriptor.mimetype ?? null },
      { cacheRoot: opts.cacheRoot, capBytes: opts.capBytes },
    )
    return result.ok ? result.path : null
  } catch {
    return null
  }
}

/**
 * Pure-primitive download path: given a raw ENCRYPTED buffer (ciphertext ‖ 10-byte MAC) and the
 * message's mediaKey, decrypt with the tested whatsapp-media primitive. This is the dependency-free
 * fallback for when the encrypted bytes are already in hand (no Baileys downloader) — it is the
 * SAME crypto the unit test pins. Returns the plaintext bytes; throws on a MAC failure.
 */
export function decryptViaPrimitive(encrypted: Uint8Array, mediaKey: Uint8Array | string, mediaType: WhatsAppMediaType): Buffer {
  return decryptWhatsAppMedia({ encrypted, mediaKey, mediaType })
}
