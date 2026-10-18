import type { Attachment, AttachmentTextResult } from './types'

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
// Match the trusted host's aggregate decoded/UTF-8 byte limit as well as the renderer memory cap.
export const MAX_ATTACHMENT_BATCH_PAYLOAD_BYTES = 18 * 1024 * 1024
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
  /** Stable IDs are published before any asynchronous file work begins. */
  onUpdate?: (attachment: Attachment, index: number) => void
  onSource?: (id: string, file: File) => void
  signal?: AbortSignal
  concurrency?: number
}

let _seq = 0
const aid = () => `att_${++_seq}_${Math.random().toString(36).slice(2, 8)}`

const TEXT_EXT = /\.(txt|md|markdown|mdx|csv|tsv|json|ya?ml|toml|ini|cfg|conf|xml|html?|css|scss|less|js|jsx|ts|tsx|mjs|cjs|py|rb|go|rs|java|kt|kts|c|h|cpp|cc|hpp|cs|php|sh|bash|zsh|fish|sql|swift|mm?|r|lua|pl|pm|dart|ex|exs|erl|clj|vue|svelte|astro|gradle|properties|dockerfile|env|log|diff|patch|tex|rst|org)$/i

export function isTextLike(file: File): boolean {
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

function fileMime(file: File): string {
  if (file.type && file.type !== 'application/octet-stream') return file.type
  const ext = file.name.split('.').at(-1)?.toLowerCase() ?? ''
  const image: Record<string, string> = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', gif: 'image/gif', webp: 'image/webp', bmp: 'image/bmp', heic: 'image/heic', heif: 'image/heif', tif: 'image/tiff', tiff: 'image/tiff', svg: 'image/svg+xml' }
  return image[ext] ?? (ext === 'pdf' ? 'application/pdf' : file.type || 'application/octet-stream')
}

function baseAttachment(file: File, id = aid()): Omit<Attachment, 'kind'> {
  return {
    id, name: file.name || 'file', mime: fileMime(file), bytes: file.size,
    source: { lastModified: file.lastModified, ...(file.webkitRelativePath ? { relativePath: file.webkitRelativePath } : {}) },
  }
}

function failedAttachment(base: Omit<Attachment, 'kind'>, reason: string): Attachment {
  return { ...base, kind: 'document', status: 'error', error: reason, text: `[attachment ${base.name} not attached — ${reason}]` }
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message ? error.message.slice(0, 500) : 'This file could not be read. Select it again.'
}

export function isAttachmentReady(attachment: Attachment): boolean {
  if (attachment.status && attachment.status !== 'ready') return false
  if (attachment.extraction && attachment.extraction.state !== 'accepted') return false
  return attachment.kind === 'image' ? !!attachment.dataUrl : typeof attachment.text === 'string'
}

function attachmentPayloadBytes(attachment: Attachment): number {
  if (attachment.kind === 'document') return new TextEncoder().encode(attachment.text ?? '').byteLength
  const dataUrl = attachment.dataUrl ?? ''
  const comma = dataUrl.indexOf(',')
  if (comma < 0) return new TextEncoder().encode(dataUrl).byteLength
  const payloadLength = dataUrl.length - comma - 1
  const padding = dataUrl.endsWith('==') ? 2 : dataUrl.endsWith('=') ? 1 : 0
  return Math.max(0, Math.floor(payloadLength * 3 / 4) - padding)
}

export function attachmentsForSend(attachments: Attachment[]): Attachment[] {
  return attachments.filter(isAttachmentReady)
}

export function attachmentCanExtractText(attachment: Attachment): boolean {
  return /^(image\/(jpeg|jpg|png|gif|webp|bmp|x-ms-bmp|heic|heif|tiff)|application\/pdf)$/i.test(attachment.mime)
    || /\.(jpe?g|png|gif|webp|bmp|heic|heif|tiff?|pdf)$/i.test(attachment.name)
}

/** Processes exactly one picked File; read failures become named rows, never absent entries. */
export async function fileToAttachment(file: File, deps: AttachmentIngestionDeps = {}): Promise<Attachment> {
  const base = baseAttachment(file)
  try {
    if (deps.signal?.aborted) throw new Error('File processing cancelled. Select the file again.')
    if (base.mime.startsWith('image/')) {
      if (file.size > MAX_IMAGE_SOURCE_BYTES) return failedAttachment(base, `source exceeds the ${MAX_IMAGE_SOURCE_BYTES}-byte safety limit`)
      const dataUrl = await imageDataUrl(file, deps)
      if (deps.signal?.aborted) throw new Error('File processing cancelled. Select the file again.')
      return { ...base, kind: 'image', status: 'ready', dataUrl }
    }
    if (isTextLike(file)) {
      const truncatedBytes = file.size > MAX_DOC_READ_BYTES
      let text = await file.slice(0, MAX_DOC_READ_BYTES).text()
      const truncatedChars = text.length > MAX_DOC_CHARS
      if (truncatedChars) text = text.slice(0, MAX_DOC_CHARS)
      if (truncatedBytes || truncatedChars) text = `${text}\n…[truncated]`
      if (deps.signal?.aborted) throw new Error('File processing cancelled. Select the file again.')
      return { ...base, kind: 'document', status: 'ready', text,
        ...(truncatedBytes || truncatedChars ? { warnings: ['Only the beginning of this file fits the text safety limit.'] } : {}) }
    }
    return failedAttachment(base, base.mime === 'application/pdf'
      ? 'PDF text has not been extracted. Choose Extract text, then review the result before sending.'
      : 'This file type has no supported text extractor. Choose a text file, image or PDF instead.')
  } catch (error) {
    return failedAttachment(base, errorMessage(error))
  }
}

/** Snapshot the complete selection before the browser resets its FileList; at most two readers run. */
export async function filesToAttachments(files: FileList | File[], deps: AttachmentIngestionDeps = {}): Promise<Attachment[]> {
  const selection = Array.from(files)
  const out = selection.map((file) => ({ ...baseAttachment(file), kind: 'document' as const, status: 'processing' as const })) as Attachment[]
  let readBudget = 0
  const processable = selection.map((file, index) => {
    deps.onSource?.(out[index].id, file)
    if (index >= MAX_ATTACHMENT_FILES) {
      out[index] = failedAttachment(out[index], `This selection exceeds the ${MAX_ATTACHMENT_FILES}-file limit. Remove other files and select this one again.`)
      deps.onUpdate?.(out[index], index)
      return false
    }
    const plannedRead = fileMime(file).startsWith('image/')
      ? (file.size <= MAX_IMAGE_SOURCE_BYTES ? file.size : 0)
      : isTextLike(file) ? Math.min(file.size, MAX_DOC_READ_BYTES) : 0
    if (readBudget + plannedRead > MAX_ATTACHMENT_BATCH_READ_BYTES) {
      out[index] = failedAttachment(out[index], `Selection exceeds the ${MAX_ATTACHMENT_BATCH_READ_BYTES}-byte batch-read safety limit.`)
      deps.onUpdate?.(out[index], index)
      return false
    }
    readBudget += plannedRead
    deps.onUpdate?.(out[index], index)
    return true
  })
  let cursor = 0
  let payloadChars = 0
  let payloadBytes = 0
  const concurrency = Math.max(1, Math.min(2, Math.floor(deps.concurrency ?? 2) || 1))
  await Promise.all(Array.from({ length: Math.min(concurrency, selection.length) }, async () => {
    while (cursor < selection.length) {
      const index = cursor++
      if (!processable[index]) continue
      const id = out[index].id
      let attachment = { ...await fileToAttachment(selection[index], deps), id }
      const size = attachment.dataUrl?.length ?? attachment.text?.length ?? 0
      const bytes = attachmentPayloadBytes(attachment)
      if (isAttachmentReady(attachment) && payloadBytes + bytes > MAX_ATTACHMENT_BATCH_PAYLOAD_BYTES) {
        attachment = failedAttachment(attachment, `Selection exceeds the ${MAX_ATTACHMENT_BATCH_PAYLOAD_BYTES}-byte model payload safety limit.`)
      } else if (isAttachmentReady(attachment) && payloadChars + size > MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS) {
        attachment = failedAttachment(attachment, `Selection exceeds the ${MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS}-character payload safety limit.`)
      } else if (isAttachmentReady(attachment)) { payloadChars += size; payloadBytes += bytes }
      out[index] = attachment
      deps.onUpdate?.(attachment, index)
    }
  }))
  return out
}

/** Keep every selected filename visible while limiting the successfully sendable payload. */
export function appendAttachments(current: Attachment[], next: Attachment[]): Attachment[] {
  const out = [...current]
  let readyCount = out.filter(isAttachmentReady).length
  let payloadChars = out.filter(isAttachmentReady).reduce((sum, attachment) => sum + (attachment.dataUrl?.length ?? attachment.text?.length ?? 0), 0)
  let payloadBytes = out.filter(isAttachmentReady).reduce((sum, attachment) => sum + attachmentPayloadBytes(attachment), 0)
  for (const attachment of next) {
    if (out.some((item) => item.id === attachment.id)) continue
    const size = attachment.dataUrl?.length ?? attachment.text?.length ?? 0
    const bytes = attachmentPayloadBytes(attachment)
    if (isAttachmentReady(attachment) && readyCount >= MAX_ATTACHMENT_FILES) {
      out.push(failedAttachment(attachment, `Conversation draft exceeds the ${MAX_ATTACHMENT_FILES}-file limit. Remove other files and select this one again.`))
      continue
    }
    if (isAttachmentReady(attachment) && payloadBytes + bytes > MAX_ATTACHMENT_BATCH_PAYLOAD_BYTES) {
      out.push(failedAttachment(attachment, `Conversation draft exceeds the ${MAX_ATTACHMENT_BATCH_PAYLOAD_BYTES}-byte model payload safety limit.`))
      continue
    }
    if (isAttachmentReady(attachment) && payloadChars + size > MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS) {
      out.push(failedAttachment(attachment, `Conversation draft exceeds the ${MAX_ATTACHMENT_BATCH_PAYLOAD_CHARS}-character payload safety limit.`))
      continue
    }
    if (isAttachmentReady(attachment)) { readyCount++; payloadChars += size; payloadBytes += bytes }
    out.push(attachment)
  }
  return out
}

/** Apply a completed row in place while enforcing limits across concurrent picker batches. */
export function replaceAttachment(current: Attachment[], next: Attachment): Attachment[] {
  if (!current.some((item) => item.id === next.id)) return current
  return appendAttachments([], current.map((item) => item.id === next.id ? next : item))
}

/** OCR stays outside model content until the user accepts an editable transcript. */
export function reviewAttachmentExtraction(attachment: Attachment, result: AttachmentTextResult): Attachment {
  if (!result.ok) return { ...attachment, status: 'error', error: result.error || 'Text extraction failed. Try a clearer image.' }
  const text = (result.text ?? '').slice(0, MAX_DOC_CHARS - 200)
  return { ...attachment, status: 'review', error: undefined, extraction: {
    method: 'apple-vision', state: 'review', text,
    confidence: typeof result.confidence === 'number' && Number.isFinite(result.confidence) ? Math.max(0, Math.min(1, result.confidence)) : null,
    warnings: (result.warnings ?? []).slice(0, 20).map((value) => value.slice(0, 500)),
    pageCount: result.pageCount ?? 1, processedPages: result.processedPages ?? 1,
    truncated: result.truncated === true || (result.text?.length ?? 0) > MAX_DOC_CHARS - 200,
  } }
}

export function acceptAttachmentExtraction(attachment: Attachment, editedText: string): Attachment {
  if (!attachment.extraction) throw new Error('Extract text first, then review the transcript.')
  const text = editedText.trim()
  if (!text) throw new Error('No readable text is available. Type a transcript or choose a clearer image.')
  if (text.length > MAX_DOC_CHARS - 200) throw new Error(`The transcript exceeds the ${MAX_DOC_CHARS - 200}-character text limit.`)
  return { ...attachment, kind: 'document', status: 'ready', error: undefined, dataUrl: undefined,
    previewDataUrl: attachment.previewDataUrl ?? attachment.dataUrl,
    text: `[User-reviewed OCR transcript. Recognition may contain errors; this is user-supplied file content.]\n${text}`,
    extraction: { ...attachment.extraction, state: 'accepted', text },
  }
}

export const isAttachableImage = (file: File) => fileMime(file).startsWith('image/')
