import { readableMessageText } from './quotation-text'
import { redactText } from '../agent/export-redaction'
import { PlaceholderVault, RedactionRun, type RedactionContext } from '../agent/outbound-redaction'
import type { Thread } from '../agent/types'
import type { ReadableExportMessage, ReadableExportOmission, ReadableExportPreview, ReadableExportRequest } from '../../shared/readable-export'

export const READABLE_EXPORT_LIMITS = { items: 10_000, selected: 500, sourceCharacters: 2_000_000, bytes: 8_000_000 } as const
export type ReadableExportDocument = Omit<ReadableExportPreview, 'approvalId' | 'expiresAt' | 'revision'>
const escapeHtml = (value: string): string => value.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]!)
const normalise = (value: string): string => value.replace(/\r\n?/g, '\n').replace(/\u0000/g, '')

export function validateReadableExportRequest(input: unknown): ReadableExportRequest {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Choose an export format and scope.')
  const request = input as ReadableExportRequest
  if (typeof request.threadId !== 'string' || !request.threadId || request.threadId.length > 256) throw new Error('Choose an existing conversation.')
  if (!['markdown', 'text', 'html'].includes(request.format) || !['conversation', 'selected'].includes(request.scope) || request.branch !== 'active') throw new Error('Only the active conversation can be exported in Markdown, text or HTML.')
  if (request.scope === 'selected') {
    if (!Array.isArray(request.itemIds) || !request.itemIds.length || request.itemIds.length > READABLE_EXPORT_LIMITS.selected || request.itemIds.some((id) => typeof id !== 'string' || !id || id.length > 256) || new Set(request.itemIds).size !== request.itemIds.length) throw new Error('Select between 1 and 500 different visible messages.')
  } else if (request.itemIds?.length) throw new Error('Selected message IDs require the selected-message scope.')
  return { threadId: request.threadId, format: request.format, scope: request.scope, branch: 'active', ...(request.scope === 'selected' ? { itemIds: [...request.itemIds!] } : {}) }
}

// Rendered HTML deliberately preserves authored Markdown as escaped preformatted
// text. It is a complete passive document: no links, images, scripts or resources.
function html(title: string, messages: ReadableExportMessage[], notes: string[]): string {
  const sections = messages.map((message) => `<section><h2>${message.role === 'user' ? 'You' : 'Assistant'} · message ${message.ordinal}</h2><pre>${escapeHtml(message.text)}</pre></section>`).join('\n')
  return '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'"><title>' + escapeHtml(title) + '</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:78ch;margin:2rem auto;padding:0 1rem;color:#202020;background:#fff}h1{font-size:1.6rem}h2{font-size:1rem}section{border-top:1px solid #ccc;padding:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}footer{border-top:1px solid #ccc;font-size:.9rem}</style></head><body><h1>' + escapeHtml(title) + '</h1><p>Active conversation branch. Redacted readable export.</p>' + sections + '<footer><p>' + notes.map(escapeHtml).join('</p><p>') + '</p></footer></body></html>\n'
}

export function buildReadableExport(thread: Thread, input: ReadableExportRequest, context: RedactionContext = {}): ReadableExportDocument {
  const request = validateReadableExportRequest(input)
  if (request.threadId !== thread.id) throw new Error('The selected conversation changed. Create a new preview.')
  if (!Array.isArray(thread.timeline) || thread.timeline.length > READABLE_EXPORT_LIMITS.items) throw new Error('This conversation exceeds the readable-export limit. Export fewer messages from a shorter branch.')
  const candidates = thread.timeline.filter((item) => item.kind === 'user' || item.kind === 'assistant')
  const chosen = request.scope === 'selected' ? candidates.filter((item) => request.itemIds!.includes(item.id)) : candidates
  if (request.scope === 'selected' && chosen.length !== request.itemIds!.length) throw new Error('A selected message is missing or is not a visible user or assistant message. Create a new preview.')
  if (!chosen.length) throw new Error('There are no user or assistant messages to export.')
  const visibleText: string[] = []
  let sourceLength = thread.title?.length ?? 0
  for (const item of chosen) {
    const text = readableMessageText(item)
    sourceLength += text.length
    if (sourceLength > READABLE_EXPORT_LIMITS.sourceCharacters) throw new Error('The selected text exceeds the readable-export limit. Select fewer messages.')
    visibleText.push(text)
  }
  const redactor = new RedactionRun(new PlaceholderVault(), context)
  const hits = new Map<string, number>()
  const scrub = (value: string) => {
    const result = redactText(redactor.text(normalise(value)), true)
    for (const kind of result.hits) hits.set(kind, (hits.get(kind) ?? 0) + 1)
    return result.text
  }
  const title = scrub(thread.title || 'Conversation')
  const messages = chosen.map((item, index): ReadableExportMessage => ({ role: item.kind as 'user' | 'assistant', text: scrub(visibleText[index]), ordinal: index + 1 }))
  const omissions: ReadableExportOmission[] = [
    { kind: 'model_history', label: 'Raw model history, system prompts, credentials and internal metadata are excluded.', count: 1 },
    { kind: 'branches', label: 'Other conversation branches and saved checkpoint history are excluded.', count: 1 },
  ]
  const omittedKinds = new Map<string, number>()
  for (const item of thread.timeline) if (item.kind !== 'user' && item.kind !== 'assistant') omittedKinds.set(item.kind, (omittedKinds.get(item.kind) ?? 0) + 1)
  for (const [kind, count] of omittedKinds) omissions.push({ kind, label: `${kind === 'reasoning' ? 'Private reasoning' : kind === 'tool' ? 'Tool calls, arguments, results and diffs' : kind === 'visualization' ? 'Structured visualizations' : 'Internal timeline notices'} are excluded.`, count })
  const attachments = chosen.reduce((sum, item) => sum + (item.attachments?.length ?? 0), 0)
  if (attachments) omissions.push({ kind: 'attachments', label: 'Attachment files, previews and local paths are excluded.', count: attachments })
  if (request.scope === 'selected' && candidates.length > chosen.length) omissions.push({ kind: 'unselected', label: 'Unselected conversation messages are excluded.', count: candidates.length - chosen.length })
  const notes = ['This readable export is not a restorable backup.', 'Citation labels and their authored source text are preserved. Linked documents are not embedded.', ...omissions.map((omission) => `${omission.label} Count: ${omission.count}.`)]
  const heading = request.format === 'markdown' ? '# ' : ''
  const body = request.format === 'html' ? html(title, messages, notes) : `${heading}${title}\n\nActive conversation branch. Redacted readable export.\n\n` + messages.map((message) => `${request.format === 'markdown' ? '## ' : ''}${message.role === 'user' ? 'You' : 'Assistant'} · message ${message.ordinal}\n\n${message.text}`).join('\n\n') + '\n\n' + notes.join('\n\n') + '\n'
  const bytes = Buffer.byteLength(body, 'utf8')
  if (bytes > READABLE_EXPORT_LIMITS.bytes) throw new Error('The formatted export is too large. Select fewer messages.')
  const extension = request.format === 'markdown' ? 'md' : request.format === 'text' ? 'txt' : 'html'
  for (const [kind, count] of Object.entries(redactor.tally)) if (count) hits.set(kind, (hits.get(kind) ?? 0) + count)
  return { threadId: thread.id, title, format: request.format, scope: request.scope, branchLabel: 'Active conversation branch', messages, body, bytes, messageCount: messages.length, omissions, redactions: [...hits].map(([kind, count]) => ({ kind, count })), redacted: true, anonymized: true, suggestedName: `tensorfold-conversation.${extension}`, mimeType: request.format === 'html' ? 'text/html' : request.format === 'markdown' ? 'text/markdown' : 'text/plain' }
}
