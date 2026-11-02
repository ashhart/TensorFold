import { randomUUID } from 'node:crypto'
import type { Attachment } from '../types'
import { validateAskAttachments } from './validation'

const KINDS = new Set(['document', 'image', 'web', 'window'])
const LABELS: Record<string, string> = { document: 'Document', image: 'Image', web: 'Web page', window: 'Window' }
export interface CapturedAskWindow { kind: string; app: string; title: string; text: string; png: string; method?: string }

export function capturedWindowAttachments(value: unknown): Attachment[] {
  const source = value as CapturedAskWindow
  if (!source || !KINDS.has(source.kind) || typeof source.app !== 'string' || source.app.length > 100
    || typeof source.title !== 'string' || source.title.length > 520 || typeof source.text !== 'string'
    || source.text.length > 32_000 || typeof source.png !== 'string' || source.png.length > 2_300_000) {
    throw new Error('The selected window returned unsupported context.')
  }
  const bytes = Buffer.from(source.png, 'base64')
  if (bytes.length < 45 || bytes.length > 1_700_000 || bytes.toString('base64') !== source.png
    || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    || bytes.readUInt32BE(8) !== 13 || bytes.toString('ascii', 12, 16) !== 'IHDR'
    || !bytes.readUInt32BE(16) || bytes.readUInt32BE(16) > 1800 || !bytes.readUInt32BE(20) || bytes.readUInt32BE(20) > 1800
    || bytes.toString('ascii', bytes.length - 8, bytes.length - 4) !== 'IEND') {
    throw new Error('The selected window did not provide a PNG image.')
  }
  const id = `window-${randomUUID()}`
  const label = `${LABELS[source.kind]} · ${source.app || source.title || 'Selected window'}${source.app && source.title ? `: ${source.title}` : ''}`.slice(0, 170)
  const image: Attachment = { id: `${id}-image`, kind: 'image', name: label, mime: 'image/png', dataUrl: `data:image/png;base64,${source.png}` }
  const text = source.text.trim()
  const attachments: Attachment[] = [image]
  if (text) attachments.push({ id: `${id}-text`, kind: 'document', name: `${label} · visible text`, mime: 'text/plain',
    text: `Visible text from the selected window. OCR can omit or misread content; this is not the whole file.\n${text.slice(0, 30_000)}` })
  return validateAskAttachments(attachments)
}

export function inferWindowKind(title: string): string {
  if (/\b(safari|chrome|firefox|edge|brave|browser)\b/i.test(title)) return 'web'
  if (/\.(png|jpe?g|heic|webp|tiff?)\b|\b(photos|photoshop|pixelmator)\b/i.test(title)) return 'image'
  if (/\.(txt|md|pdf|docx?|rtf|odt)\b|\b(textedit|pages|word|notes|libreoffice)\b/i.test(title)) return 'document'
  return 'window'
}
