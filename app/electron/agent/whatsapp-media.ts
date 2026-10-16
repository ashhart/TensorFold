// WhatsApp (Baileys) incoming-media decryption — a PURE, dependency-free crypto primitive
// (Node `crypto` only). WhatsApp media is delivered as an ENCRYPTED blob: the message carries a
// base64 `mediaKey`, and the downloaded file is `ciphertext || mac` where `mac` is the last 10
// bytes. Keys are derived from the mediaKey with HKDF-SHA256 expanded to 112 bytes (info = a
// per-media-type app-info string), split into iv(16) | cipherKey(32) | macKey(32) | refKey(32):
//
//   1. HKDF-SHA256(mediaKey, 112, info)  ->  iv ‖ cipherKey ‖ macKey ‖ refKey
//   2. verify  HMAC-SHA256(macKey, iv ‖ ciphertext)[0:10] === mac   (else throw — integrity)
//   3. AES-256-CBC decrypt(cipherKey, iv, ciphertext)  ->  plaintext
//
// This is the well-established WhatsApp media scheme (matches Baileys' getMediaKeys / the
// reference "WhatsApp <Type> Keys" HKDF info + the iv/cipherKey/macKey split). It is kept here,
// pure and side-effect-free, so it is unit-testable against a self-constructed known vector and
// usable as a fallback when Baileys' own downloader isn't available. No network, no fs.

import { createDecipheriv, createHmac, hkdfSync, timingSafeEqual } from 'node:crypto'

// The media kinds that carry a mediaKey + per-type HKDF info. These are the four BASE app-info
// strings; thumbnails/stickers/etc. reuse one of these in practice.
export type WhatsAppMediaType = 'image' | 'video' | 'audio' | 'document'

const MEDIA_INFO: Record<WhatsAppMediaType, string> = {
  image: 'WhatsApp Image Keys',
  video: 'WhatsApp Video Keys',
  audio: 'WhatsApp Audio Keys',
  document: 'WhatsApp Document Keys',
}

/** The WhatsApp HKDF app-info string for a media type (e.g. 'WhatsApp Image Keys'). Pure. */
export function whatsappMediaInfo(mediaType: WhatsAppMediaType): string {
  const info = MEDIA_INFO[mediaType]
  if (!info) throw new Error(`Unknown WhatsApp media type: ${String(mediaType)}`)
  return info
}

/**
 * HKDF-SHA256 expand of `key` to `length` bytes with the given `info` (empty salt — Node uses a
 * hashLen-zero salt internally, which is what WhatsApp/Baileys rely on). Returns a Buffer.
 * Pure + deterministic so the unit test can pin it against an independent HKDF reference.
 */
export function hkdfExpand(key: Buffer | Uint8Array, length: number, info: string): Buffer {
  if (!key || key.length === 0) throw new Error('hkdfExpand: media key is empty.')
  if (!Number.isInteger(length) || length <= 0) throw new Error('hkdfExpand: length must be a positive integer.')
  // hkdfSync returns an ArrayBuffer; wrap it as a Buffer for ergonomic slicing.
  return Buffer.from(hkdfSync('sha256', key, Buffer.alloc(0), Buffer.from(info, 'utf8'), length))
}

export interface ExpandedMediaKeys {
  iv: Buffer // 16 bytes — AES-CBC IV
  cipherKey: Buffer // 32 bytes — AES-256 key
  macKey: Buffer // 32 bytes — HMAC-SHA256 key
  refKey: Buffer // 32 bytes — reference key (unused for decrypt; kept for completeness)
}

/**
 * Derive the per-message media keys from the (base64 or raw) mediaKey: HKDF-expand to 112 bytes
 * with the per-type app-info, then split iv(16)|cipherKey(32)|macKey(32)|refKey(32). Pure.
 */
export function deriveWhatsAppMediaKeys(mediaKey: Buffer | Uint8Array | string, mediaType: WhatsAppMediaType): ExpandedMediaKeys {
  const keyBuf = normalizeMediaKey(mediaKey)
  if (keyBuf.length === 0) throw new Error('deriveWhatsAppMediaKeys: media key is empty.')
  const expanded = hkdfExpand(keyBuf, 112, whatsappMediaInfo(mediaType))
  return {
    iv: expanded.subarray(0, 16),
    cipherKey: expanded.subarray(16, 48),
    macKey: expanded.subarray(48, 80),
    refKey: expanded.subarray(80, 112),
  }
}

export interface DecryptWhatsAppMediaInput {
  /** The message's mediaKey — base64 string (as Baileys delivers it) or raw bytes. */
  mediaKey: Buffer | Uint8Array | string
  /** Which media type's HKDF info to use. */
  mediaType: WhatsAppMediaType
  /** The downloaded encrypted file: ciphertext followed by a 10-byte MAC. */
  encrypted: Buffer | Uint8Array
}

const MAC_LENGTH = 10

/**
 * Decrypt a WhatsApp media blob. Splits off the trailing 10-byte MAC, verifies
 * HMAC-SHA256(macKey, iv ‖ ciphertext)[0:10] in constant time (THROWS on mismatch — a tampered
 * blob or wrong key must never yield silent garbage), then AES-256-CBC decrypts with the derived
 * cipherKey/iv and returns the plaintext Buffer. Pure (no I/O); the only inputs are the mediaKey,
 * the type, and the bytes — so it is exhaustively testable against a constructed known vector.
 */
export function decryptWhatsAppMedia(input: DecryptWhatsAppMediaInput): Buffer {
  const file = toBuffer(input.encrypted)
  if (file.length <= MAC_LENGTH) {
    throw new Error('decryptWhatsAppMedia: encrypted blob is too short to contain a MAC.')
  }
  const { iv, cipherKey, macKey } = deriveWhatsAppMediaKeys(input.mediaKey, input.mediaType)

  const ciphertext = file.subarray(0, file.length - MAC_LENGTH)
  const mac = file.subarray(file.length - MAC_LENGTH)

  // Integrity: HMAC-SHA256(macKey, iv || ciphertext), compare the first 10 bytes in constant time.
  const expectedMac = createHmac('sha256', macKey).update(iv).update(ciphertext).digest().subarray(0, MAC_LENGTH)
  if (mac.length !== expectedMac.length || !timingSafeEqual(mac, expectedMac)) {
    throw new Error('decryptWhatsAppMedia: MAC verification failed (tampered media or wrong key).')
  }

  const decipher = createDecipheriv('aes-256-cbc', cipherKey, iv)
  return Buffer.concat([decipher.update(ciphertext), decipher.final()])
}

// ---- internals ----------------------------------------------------------------

// A mediaKey may arrive as a base64 string (Baileys' on-the-wire form, optionally with a
// `data:;base64,` prefix), a Buffer, or a Uint8Array. Normalize to a Buffer.
function normalizeMediaKey(mediaKey: Buffer | Uint8Array | string): Buffer {
  if (typeof mediaKey === 'string') {
    return Buffer.from(mediaKey.replace(/^data:;base64,/, ''), 'base64')
  }
  return toBuffer(mediaKey)
}

function toBuffer(data: Buffer | Uint8Array): Buffer {
  return Buffer.isBuffer(data) ? data : Buffer.from(data)
}
