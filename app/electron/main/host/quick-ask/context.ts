import { clipboard, desktopCapturer, screen, systemPreferences } from 'electron'
import { randomUUID } from 'node:crypto'
import type { Attachment } from '../../../agent/types'
import { validateAskAttachments, ASK_TEXT_LIMIT } from '../../../agent/quick-ask/validation'

export function askClipboard(): Attachment[] {
  const image = clipboard.readImage()
  if (!image.isEmpty()) {
    const size = image.getSize(), scale = Math.min(1, 1568 / Math.max(size.width, size.height))
    const bytes = image.resize({ width: Math.max(1, Math.round(size.width * scale)), height: Math.max(1, Math.round(size.height * scale)) }).toJPEG(85)
    return validateAskAttachments([{ id: randomUUID(), name: 'Clipboard image', kind: 'image', dataUrl: `data:image/jpeg;base64,${bytes.toString('base64')}` }])
  }
  const text = clipboard.readText()
  if (!text.trim()) throw new Error('The clipboard has no text or image to attach.')
  if (text.length > ASK_TEXT_LIMIT) throw new Error('The clipboard is too long. Paste a shorter selection.')
  return validateAskAttachments([{ id: randomUUID(), name: 'Clipboard text', kind: 'document', text }])
}
export function askDisplays() { return screen.getAllDisplays().map((display, index) => ({ id: String(display.id), label: display.label || `Display ${index + 1}` })) }
export async function captureAskDisplay(displayId: string): Promise<Attachment> {
  if (typeof displayId !== 'string' || !askDisplays().some(display => display.id === displayId)) throw new Error('Choose a connected display.')
  if (process.platform === 'darwin' && systemPreferences.getMediaAccessStatus('screen') === 'denied') throw new Error('Allow TensorFold in System Settings > Privacy & Security > Screen Recording, then restart TensorFold.')
  const sources = await desktopCapturer.getSources({ types: ['screen'], thumbnailSize: { width: 1600, height: 1000 }, fetchWindowIcons: false })
  const source = sources.find(source => source.display_id === displayId) ?? (sources.length === 1 ? sources[0] : undefined)
  if (!source || source.thumbnail.isEmpty()) throw new Error('Screen capture was declined or unavailable. You can paste a screenshot instead.')
  const bytes = source.thumbnail.toJPEG(85)
  return validateAskAttachments([{ id: randomUUID(), name: 'Screen capture', kind: 'image', dataUrl: `data:image/jpeg;base64,${bytes.toString('base64')}` }])[0]
}
