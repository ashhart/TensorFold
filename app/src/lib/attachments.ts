import type { Attachment } from './types'

// Turn picked/pasted/dropped Files into Attachments for the composer.
//   images    → downscaled JPEG data URL (vision transports see them).
//   text-like → extracted text (works on every transport, incl. the CLI ones).
//   other     → acknowledged as a document with a "not extracted" note.
const MAX_IMAGE_DIM = 1568
const IMAGE_QUALITY = 0.85
export const MAX_DOC_CHARS = 200_000
export const MAX_DOC_READ_BYTES = 512 * 1024
export const MAX_IMAGE_SOURCE_BYTES = 20 * 1024 * 1024
export const MAX_IMAGE_RAW_BYTES = 4 * 1024 * 1024
export const MAX_IMAGE_DECODED_PIXELS = 24 * 1024 * 1024
export const MAX_IMAGE_DATA_URL_CHARS = 8 * 1024 * 1024
export const MAX_ATTACHMENT_FILES = 20
export const MAX_ATTACHMENT_BATCH_READ_BYTES = 40 * 1024 * 1024
export const MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS = 24 * 1024 * 1024
const MAX_IMAGE_HEADER_BYTES = 512 * 1024

interface BitmapLike {
  width: number
  height: number
  close?: () => void
}

class AttachmentSafetyError extends Error {}

export interface AttachmentIngestionDeps {
  createBitmap?: (file: Blob) => Promise<BitmapLike>
  readDataUrl?: (file: Blob) => Promise<string>
  encodeBitmap?: (bitmap: BitmapLike, width: number, height: number) => string
}

let _seq = 0
const aid = () => `att_${++_seq}_${Math.random().toString(36).slice(2, 8)}`

const TEXT_EXT = /\.(txt|md|markdown|mdx|csv|tsv|json|ya?ml|toml|ini|cfg|conf|xml|html?|css|scss|less|js|jsx|ts|tsx|mjs|cjs|py|rb|go|rs|java|kt|kts|c|h|cpp|cc|hpp|cs|php|sh|bash|zsh|fish|sql|swift|mm?|r|lua|pl|pm|dart|ex|exs|erl|clj|vue|svelte|astro|gradle|properties|dockerfile|env|log|diff|patch|tex|rst|org)$/i

function isTextLike(file: File): boolean {
  if (file.type.startsWith('text/')) return true
  if (/^application\/(json|xml|x-yaml|yaml|javascript|x-sh|toml|x-toml)/i.test(file.type)) return true
  return TEXT_EXT.test(file.name)
}

async function readDataUrl(file: Blob): Promise<string> {
  return await new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(String(r.result))
    r.onerror = () => reject(r.error)
    r.readAsDataURL(file)
  })
}

function u16be(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] << 8) | bytes[offset + 1]
}

function u16le(bytes: Uint8Array, offset: number): number {
  return bytes[offset] | (bytes[offset + 1] << 8)
}

function u32be(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset] * 0x1000000) + (bytes[offset + 1] << 16) + (bytes[offset + 2] << 8) + bytes[offset + 3]) >>> 0
}

function u32le(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] + (bytes[offset + 1] << 8) + (bytes[offset + 2] << 16) + (bytes[offset + 3] * 0x1000000)) >>> 0
}

/** Read only a bounded header and reject common image bombs before asking the browser to decode. */
export async function imageDimensionsFromHeader(file: Blob): Promise<{ width: number; height: number } | null> {
  const bytes = new Uint8Array(await file.slice(0, MAX_IMAGE_HEADER_BYTES).arrayBuffer())
  // PNG signature + IHDR.
  if (bytes.length >= 24 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return { width: u32be(bytes, 16), height: u32be(bytes, 20) }
  }
  // GIF logical screen dimensions.
  if (bytes.length >= 10 && String.fromCharCode(...bytes.subarray(0, 6)).startsWith('GIF8')) {
    return { width: u16le(bytes, 6), height: u16le(bytes, 8) }
  }
  // JPEG SOF marker. APP/EXIF segments can be large, hence the bounded 512 KiB header window.
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2
    while (offset + 4 <= bytes.length) {
      while (offset < bytes.length && bytes[offset] !== 0xff) offset++
      while (offset < bytes.length && bytes[offset] === 0xff) offset++
      if (offset >= bytes.length) break
      const marker = bytes[offset++]
      if (marker === 0xd8 || marker === 0x01) continue
      if (marker === 0xd9 || marker === 0xda) break
      if (offset + 2 > bytes.length) break
      const segmentLength = u16be(bytes, offset)
      if (segmentLength < 2 || offset + segmentLength > bytes.length) break
      const isSof = (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker))
      if (isSof && segmentLength >= 7) {
        return { width: u16be(bytes, offset + 5), height: u16be(bytes, offset + 3) }
      }
      offset += segmentLength
    }
  }
  // WebP extended header stores 24-bit little-endian dimensions minus one.
  if (bytes.length >= 30 && String.fromCharCode(...bytes.subarray(0, 4)) === 'RIFF' && String.fromCharCode(...bytes.subarray(8, 12)) === 'WEBP') {
    const chunk = String.fromCharCode(...bytes.subarray(12, 16))
    if (chunk === 'VP8X') {
      const width = 1 + bytes[24] + (bytes[25] << 8) + (bytes[26] << 16)
      const height = 1 + bytes[27] + (bytes[28] << 8) + (bytes[29] << 16)
      return { width, height }
    }
    if (chunk === 'VP8L' && bytes[20] === 0x2f && bytes.length >= 25) {
      const width = 1 + bytes[21] + ((bytes[22] & 0x3f) << 8)
      const height = 1 + (bytes[22] >> 6) + (bytes[23] << 2) + ((bytes[24] & 0x0f) << 10)
      return { width, height }
    }
    if (chunk === 'VP8 ' && bytes[23] === 0x9d && bytes[24] === 0x01 && bytes[25] === 0x2a) {
      return { width: u16le(bytes, 26) & 0x3fff, height: u16le(bytes, 28) & 0x3fff }
    }
  }
  // BMP DIB width/height. Negative height means top-down and has the same pixel count.
  if (bytes.length >= 26 && bytes[0] === 0x42 && bytes[1] === 0x4d) {
    const width = u32le(bytes, 18) | 0
    const rawHeight = u32le(bytes, 22) | 0
    return { width: Math.abs(width), height: Math.abs(rawHeight) }
  }
  return null
}

function assertSafeImageDimensions(width: number, height: number): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new AttachmentSafetyError('image dimensions are invalid')
  }
  if (width * height > MAX_IMAGE_DECODED_PIXELS) {
    throw new AttachmentSafetyError(`decoded image exceeds the ${MAX_IMAGE_DECODED_PIXELS}-pixel safety limit`)
  }
}

function assertSafeDataUrl(dataUrl: string): string {
  if (dataUrl.length > MAX_IMAGE_DATA_URL_CHARS) {
    throw new AttachmentSafetyError('encoded image exceeds the data URL safety limit')
  }
  return dataUrl
}

function encodeBitmap(bitmap: BitmapLike, width: number, height: number): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas decoding is unavailable')
  ctx.drawImage(bitmap as CanvasImageSource, 0, 0, width, height)
  return canvas.toDataURL('image/jpeg', IMAGE_QUALITY)
}

// Downscale large images so a data URL doesn't bloat the request. A decode/canvas failure may
// fall back to the original only while that original remains under the separate raw-byte cap.
async function imageDataUrl(file: File, deps: AttachmentIngestionDeps = {}): Promise<string> {
  const dimensions = await imageDimensionsFromHeader(file)
  const mime = file.type.toLowerCase()
  if (mime === 'image/svg+xml') throw new AttachmentSafetyError('SVG images are not accepted as model attachments')
  // Fail closed for every format whose dimensions cannot be established from a bounded header.
  // Decoding AVIF/HEIC/other browser-supported formats first and checking bitmap.width AFTERWARD
  // is too late: a compressed image bomb may already have allocated its decoded pixel surface.
  if (!dimensions) throw new AttachmentSafetyError('image format or dimensions could not be safely validated within the header limit')
  assertSafeImageDimensions(dimensions.width, dimensions.height)
  const createBitmap = deps.createBitmap ?? ((blob: Blob) => createImageBitmap(blob))
  const rawDataUrl = deps.readDataUrl ?? readDataUrl
  const encode = deps.encodeBitmap ?? encodeBitmap
  let bitmap: BitmapLike | null = null
  try {
    bitmap = await createBitmap(file)
  } catch (error) {
    if (file.size <= MAX_IMAGE_RAW_BYTES) {
      return assertSafeDataUrl(await rawDataUrl(file))
    }
    throw error
  }
  try {
    assertSafeImageDimensions(bitmap.width, bitmap.height)
    const scale = Math.min(1, MAX_IMAGE_DIM / Math.max(bitmap.width, bitmap.height))
    if (scale >= 1 && file.size <= MAX_IMAGE_RAW_BYTES) return assertSafeDataUrl(await rawDataUrl(file))
    const width = Math.max(1, Math.round(bitmap.width * scale))
    const height = Math.max(1, Math.round(bitmap.height * scale))
    try {
      return assertSafeDataUrl(encode(bitmap, width, height))
    } catch (error) {
      if (!(error instanceof AttachmentSafetyError) && file.size <= MAX_IMAGE_RAW_BYTES) {
        return assertSafeDataUrl(await rawDataUrl(file))
      }
      throw error
    }
  } finally {
    bitmap.close?.()
  }
}

function unavailableImage(base: Omit<Attachment, 'kind'>, reason: string): Attachment {
  return { ...base, kind: 'document', text: `[image ${base.name} not attached — ${reason}]` }
}

export async function fileToAttachment(file: File, deps: AttachmentIngestionDeps = {}): Promise<Attachment> {
  const base = { id: aid(), name: file.name || 'file', mime: file.type || 'application/octet-stream', bytes: file.size }
  if (file.type.startsWith('image/')) {
    if (file.size > MAX_IMAGE_SOURCE_BYTES) {
      return unavailableImage(base, `source exceeds the ${MAX_IMAGE_SOURCE_BYTES}-byte safety limit`)
    }
    try {
      return { ...base, kind: 'image', dataUrl: await imageDataUrl(file, deps) }
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'image decoding failed'
      return unavailableImage(base, reason)
    }
  }
  if (isTextLike(file)) {
    const truncatedBytes = file.size > MAX_DOC_READ_BYTES
    let text = await file.slice(0, MAX_DOC_READ_BYTES).text()
    const truncatedChars = text.length > MAX_DOC_CHARS
    if (truncatedChars) text = text.slice(0, MAX_DOC_CHARS)
    if (truncatedBytes || truncatedChars) text = `${text}\n…[truncated]`
    return { ...base, kind: 'document', text }
  }
  return { ...base, kind: 'document', text: `[attached ${base.name} (${base.mime}) — binary file, text not extracted]` }
}

export async function filesToAttachments(files: FileList | File[]): Promise<Attachment[]> {
  const out: Attachment[] = []
  let readBudget = 0
  let payloadChars = 0
  const hasOverflow = files.length > MAX_ATTACHMENT_FILES
  const count = Math.min(files.length, hasOverflow ? MAX_ATTACHMENT_FILES - 1 : MAX_ATTACHMENT_FILES)
  for (let index = 0; index < count; index++) {
    const f = files[index]
    if (!f) continue
    const plannedRead = f.type.startsWith('image/')
      ? (f.size <= MAX_IMAGE_SOURCE_BYTES ? f.size : 0)
      : isTextLike(f) ? Math.min(f.size, MAX_DOC_READ_BYTES) : 0
    if (readBudget + plannedRead > MAX_ATTACHMENT_BATCH_READ_BYTES) {
      out.push({
        id: aid(), kind: 'document', name: f.name || 'file', mime: f.type || 'application/octet-stream', bytes: f.size,
        text: `[attachment ${f.name || 'file'} not attached — selection exceeds the ${MAX_ATTACHMENT_BATCH_READ_BYTES}-byte batch-read safety limit]`,
      })
      continue
    }
    readBudget += plannedRead
    try {
      const attachment = await fileToAttachment(f)
      const size = attachment.dataUrl?.length ?? attachment.text?.length ?? 0
      if (payloadChars + size > MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS) {
        out.push({
          id: aid(), kind: 'document', name: f.name || 'file', mime: f.type || 'application/octet-stream', bytes: f.size,
          text: `[attachment ${f.name || 'file'} not attached — selection exceeds the ${MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS}-character payload safety limit]`,
        })
        continue
      }
      payloadChars += size
      out.push(attachment)
    } catch { /* skip unreadable file */ }
  }
  if (hasOverflow) {
    const skipped = files.length - count
    out.push({
      id: aid(), kind: 'document', name: `${skipped} skipped attachment${skipped === 1 ? '' : 's'}`, mime: 'text/plain', bytes: 0,
      text: `[${skipped} attachment${skipped === 1 ? '' : 's'} not attached — selection exceeds the ${MAX_ATTACHMENT_FILES}-file safety limit]`,
    })
  }
  return out
}

/** Enforce the same count/payload ceilings across repeated picker operations. */
export function appendAttachments(current: Attachment[], next: Attachment[]): Attachment[] {
  const out = current.slice(0, MAX_ATTACHMENT_FILES)
  let payloadChars = out.reduce((sum, attachment) => sum + (attachment.dataUrl?.length ?? attachment.text?.length ?? 0), 0)
  for (const attachment of next) {
    if (out.length >= MAX_ATTACHMENT_FILES) break
    const size = attachment.dataUrl?.length ?? attachment.text?.length ?? 0
    if (payloadChars + size > MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS) continue
    payloadChars += size
    out.push(attachment)
  }
  return out
}

export const isAttachableImage = (file: File) => file.type.startsWith('image/')
