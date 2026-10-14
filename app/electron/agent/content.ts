import type { Attachment, ContentBlock } from './types/base'

// Plain-text view of (possibly multimodal) message content. Every consumer that used
// to treat ApiMessage.content as a string routes through this so arrays don't break
// token counting, transcripts, recall, redaction, or the CLI text-flattening path.
// Image parts render as a short placeholder (the filename, when present, lives in the
// adjacent text block from buildUserContent).
export function contentText(content: string | ContentBlock[] | null | undefined): string {
  if (content == null) return ''
  if (typeof content === 'string') return content
  return content.map((b) => (b.type === 'text' ? b.text : '[image]')).join('\n')
}

// Assemble a user message's content from typed text + attachments.
//   documents → folded into the text as labelled blocks (work on every transport).
//   images    → image_url blocks for vision transports (chat-completions); the text
//               also names them so CLI/non-vision seats still know an image was sent.
// Returns a plain string when there are no images (back-compat with the string path).
export function buildUserContent(text: string, attachments: Attachment[] | undefined): string | ContentBlock[] {
  const list = attachments ?? []
  if (!list.length) return text
  const docs = list.filter((a) => a.kind === 'document' && a.text && a.text.trim())
  const images = list.filter((a) => a.kind === 'image')
  const docText = docs.map((d) => `\n\n<attached file: ${d.name}>\n${d.text}\n</attached file>`).join('')
  const imageNotes = images.map((i) => `\n[attached image: ${i.name}]`).join('')
  const baseText = `${text}${docText}${imageNotes}`
  if (!images.length) return baseText
  const blocks: ContentBlock[] = [{ type: 'text', text: baseText }]
  for (const img of images) if (img.dataUrl) blocks.push({ type: 'image_url', image_url: { url: img.dataUrl } })
  return blocks
}
