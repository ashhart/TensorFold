import { randomUUID } from 'node:crypto'
import type { Attachment, AttachmentTextExtraction } from './types'

export const HOST_MAX_ATTACHMENT_FILES = 20
export const HOST_MAX_DOCUMENT_CHARS = 200_020
// Blob.text() caps renderer documents by JS characters after its 512 KiB source read. Keep
// enough UTF-8 headroom for every renderer-accepted code unit, including non-ASCII text.
export const HOST_MAX_DOCUMENT_BYTES = HOST_MAX_DOCUMENT_CHARS * 4
// Renderer output is capped at 8 MiB of data-URL characters (about 6 MiB decoded). The
// trusted host allows only the tiny scheme/MIME prefix beyond that renderer ceiling, then
// independently enforces the exact decoded-byte limit below.
export const HOST_MAX_IMAGE_DATA_URL_CHARS = (8 * 1024 * 1024) + 64
export const HOST_MAX_IMAGE_BYTES = 6 * 1024 * 1024
export const HOST_MAX_ATTACHMENT_PAYLOAD_BYTES = 18 * 1024 * 1024
export const HOST_MAX_IMAGE_PIXELS = 24 * 1024 * 1024
const MAX_IMAGE_HEADER_BYTES = 512 * 1024

/** Input-policy failures are client errors on the paired REST surface, not host faults. */
export class AttachmentPolicyError extends Error {
  readonly code = 400
  constructor(message: string) {
    super(message)
    this.name = 'AttachmentPolicyError'
  }
}

function rejectAttachment(message: string): never {
  throw new AttachmentPolicyError(message)
}

function u16be(bytes: Buffer, offset: number): number { return bytes.readUInt16BE(offset) }
function u16le(bytes: Buffer, offset: number): number { return bytes.readUInt16LE(offset) }
function u32be(bytes: Buffer, offset: number): number { return bytes.readUInt32BE(offset) }
function u32le(bytes: Buffer, offset: number): number { return bytes.readUInt32LE(offset) }

type ImageFormat = 'jpeg' | 'png' | 'gif' | 'webp' | 'bmp'

function imageHeader(bytes: Buffer): { format: ImageFormat; width: number; height: number } | null {
  if (bytes.length >= 24 && bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
    return { format: 'png', width: u32be(bytes, 16), height: u32be(bytes, 20) }
  }
  if (bytes.length >= 10 && ['GIF87a', 'GIF89a'].includes(bytes.subarray(0, 6).toString('ascii'))) {
    return { format: 'gif', width: u16le(bytes, 6), height: u16le(bytes, 8) }
  }
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2
    while (offset + 4 <= bytes.length) {
      while (offset < bytes.length && bytes[offset] !== 0xff) offset++
      while (offset < bytes.length && bytes[offset] === 0xff) offset++
      if (offset >= bytes.length) break
      const marker = bytes[offset++]
      if (marker === 0xd8 || marker === 0x01) continue
      if (marker === 0xd9 || marker === 0xda || offset + 2 > bytes.length) break
      const length = u16be(bytes, offset)
      if (length < 2 || offset + length > bytes.length) break
      if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker) && length >= 7) {
        return { format: 'jpeg', width: u16be(bytes, offset + 5), height: u16be(bytes, offset + 3) }
      }
      offset += length
    }
  }
  if (bytes.length >= 30 && bytes.subarray(0, 4).toString('ascii') === 'RIFF' && bytes.subarray(8, 12).toString('ascii') === 'WEBP') {
    const chunk = bytes.subarray(12, 16).toString('ascii')
    if (chunk === 'VP8X') return { format: 'webp', width: 1 + bytes[24] + (bytes[25] << 8) + (bytes[26] << 16), height: 1 + bytes[27] + (bytes[28] << 8) + (bytes[29] << 16) }
    if (chunk === 'VP8L' && bytes[20] === 0x2f && bytes.length >= 25) return { format: 'webp', width: 1 + bytes[21] + ((bytes[22] & 0x3f) << 8), height: 1 + (bytes[22] >> 6) + (bytes[23] << 2) + ((bytes[24] & 0x0f) << 10) }
    if (chunk === 'VP8 ' && bytes[23] === 0x9d && bytes[24] === 0x01 && bytes[25] === 0x2a) return { format: 'webp', width: u16le(bytes, 26) & 0x3fff, height: u16le(bytes, 28) & 0x3fff }
  }
  if (bytes.length >= 26 && bytes[0] === 0x42 && bytes[1] === 0x4d) {
    return { format: 'bmp', width: Math.abs(u32le(bytes, 18) | 0), height: Math.abs(u32le(bytes, 22) | 0) }
  }
  return null
}

function cleanString(value: unknown, max: number): string {
  return typeof value === 'string' ? value.replace(/[\r\n\0]/g, ' ').trim().slice(0, max) : ''
}

function cleanFilename(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback
  const leaf = value.normalize('NFKC').split(/[\\/]/).at(-1) ?? ''
  const cleaned = leaf
    .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069:]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 240)
  return cleaned && cleaned !== '.' && cleaned !== '..' ? cleaned : fallback
}

function cleanDocumentMime(value: unknown): string {
  const mime = cleanString(value, 129).toLowerCase()
  return /^[a-z0-9][a-z0-9!#$&^_.+*-]{0,63}\/[a-z0-9][a-z0-9!#$&^_.+*-]{0,63}$/.test(mime)
    ? mime
    : 'text/plain'
}

function normalizeImageDataUrl(raw: unknown): { dataUrl: string; mime: string; bytes: number } {
  if (typeof raw !== 'string' || raw.length > HOST_MAX_IMAGE_DATA_URL_CHARS) rejectAttachment('image data URL is missing or oversized')
  const match = /^data:image\/(jpeg|jpg|png|gif|webp|bmp|x-ms-bmp);base64,([A-Za-z0-9+/]*={0,2})$/i.exec(raw)
  if (!match || match[2].length % 4 !== 0) rejectAttachment('image URL must be a supported base64 data URL')
  const declared = match[1].toLowerCase().replace('jpg', 'jpeg').replace('x-ms-bmp', 'bmp') as ImageFormat
  // Decode only the bounded header first. The full base64 string remains bounded separately.
  const headerChars = Math.min(match[2].length, Math.ceil(MAX_IMAGE_HEADER_BYTES / 3) * 4)
  const header = imageHeader(Buffer.from(match[2].slice(0, headerChars), 'base64'))
  if (!header || header.format !== declared) rejectAttachment('image bytes do not match the declared supported format')
  if (!Number.isSafeInteger(header.width) || !Number.isSafeInteger(header.height) || header.width < 1 || header.height < 1 || header.width * header.height > HOST_MAX_IMAGE_PIXELS) {
    rejectAttachment('image decoded dimensions exceed the host safety limit')
  }
  const canonicalMime = `image/${declared}`
  const padding = match[2].endsWith('==') ? 2 : match[2].endsWith('=') ? 1 : 0
  const bytes = (match[2].length * 3 / 4) - padding
  if (bytes > HOST_MAX_IMAGE_BYTES) rejectAttachment('image decoded bytes exceed the host safety limit')
  return { dataUrl: `data:${canonicalMime};base64,${match[2]}`, mime: canonicalMime, bytes }
}

function normalizeAcceptedExtraction(raw: unknown, text: string): AttachmentTextExtraction | undefined {
  if (raw === undefined) return undefined
  const value = raw as Record<string, unknown>
  if (value.method !== 'apple-vision' || value.state !== 'accepted' || typeof value.text !== 'string' || !value.text.trim() || value.text.length > 199_800 || !text.endsWith(`\n${value.text}`)) rejectAttachment('The accepted OCR transcript does not match the attached document text.')
  const confidence = typeof value.confidence === 'number' && Number.isFinite(value.confidence) && value.confidence >= 0 && value.confidence <= 1 ? value.confidence : null
  const pageCount = Number.isSafeInteger(value.pageCount) && Number(value.pageCount) >= 1 ? Number(value.pageCount) : 1
  const processedPages = Number.isSafeInteger(value.processedPages) && Number(value.processedPages) >= 1 && Number(value.processedPages) <= Math.min(8, pageCount) ? Number(value.processedPages) : 1
  const warnings = Array.isArray(value.warnings) ? value.warnings.filter((item): item is string => typeof item === 'string').slice(0, 20).map((item) => item.slice(0, 500)) : []
  return { method: 'apple-vision', state: 'accepted', text: value.text, confidence, warnings, pageCount, processedPages, truncated: value.truncated === true }
}

/** Fail-closed normalization at the trusted host boundary (IPC, REST, and internal callers). */
export function normalizeAttachments(input: unknown): Attachment[] | undefined {
  if (input === undefined) return undefined
  if (!Array.isArray(input)) rejectAttachment('Attachments must be an array.')
  if (input.length > HOST_MAX_ATTACHMENT_FILES) rejectAttachment(`Attachments exceed the ${HOST_MAX_ATTACHMENT_FILES}-file host limit.`)
  const output: Attachment[] = []
  let payloadBytes = 0
  for (let index = 0; index < input.length; index++) {
    const value = input[index]
    if (!value || typeof value !== 'object' || Array.isArray(value)) rejectAttachment(`Attachment ${index + 1} is invalid.`)
    const item = value as Record<string, unknown>
    const kind = item.kind
    if (item.status !== undefined && item.status !== 'ready') rejectAttachment(`Attachment ${index + 1} is not ready; finish processing, review it, select it again or remove it.`)
    if (item.extraction !== undefined) {
      if (!item.extraction || typeof item.extraction !== 'object' || Array.isArray(item.extraction) || (item.extraction as Record<string, unknown>).state !== 'accepted') rejectAttachment(`Attachment ${index + 1} extracted text has not been accepted.`)
      if (kind !== 'document') rejectAttachment(`Attachment ${index + 1} accepted text must be attached as a document.`)
    }
    const name = cleanFilename(item.name, `attachment-${index + 1}`)
    const id = cleanString(item.id, 160) || `att_host_${randomUUID()}`
    if (kind === 'document') {
      if (typeof item.text !== 'string' || item.text.length > HOST_MAX_DOCUMENT_CHARS) rejectAttachment(`Attachment ${index + 1} document text is missing or oversized.`)
      const bytes = Buffer.byteLength(item.text)
      if (bytes > HOST_MAX_DOCUMENT_BYTES) rejectAttachment(`Attachment ${index + 1} document bytes exceed the host safety limit.`)
      payloadBytes += bytes
      if (payloadBytes > HOST_MAX_ATTACHMENT_PAYLOAD_BYTES) rejectAttachment('Attachments exceed the aggregate host payload limit.')
      const extraction = normalizeAcceptedExtraction(item.extraction, item.text)
      const warnings = Array.isArray(item.warnings) ? item.warnings.filter((value): value is string => typeof value === 'string').slice(0, 20).map((value) => value.slice(0, 500)) : undefined
      output.push({ id, kind: 'document', name, mime: cleanDocumentMime(item.mime), text: item.text, bytes, ...(extraction ? { extraction } : {}), ...(warnings?.length ? { warnings } : {}) })
      continue
    }
    if (kind === 'image') {
      const image = normalizeImageDataUrl(item.dataUrl)
      payloadBytes += image.bytes
      if (payloadBytes > HOST_MAX_ATTACHMENT_PAYLOAD_BYTES) rejectAttachment('Attachments exceed the aggregate host payload limit.')
      output.push({ id, kind: 'image', name, mime: image.mime, dataUrl: image.dataUrl, bytes: image.bytes })
      continue
    }
    rejectAttachment(`Attachment ${index + 1} has an unsupported kind.`)
  }
  return output
}
