import { readableQuotations } from './quotation-text'
import { createHash, randomUUID } from 'node:crypto'
import type { Thread } from '../agent/types'
import type { RedactionContext } from '../agent/outbound-redaction'
import type { ReadableExportPreview, ReadableExportRequest, ReadableExportSaveRequest } from '../../shared/readable-export'
import { buildReadableExport, READABLE_EXPORT_LIMITS, validateReadableExportRequest } from './format'

export interface ReadableExportServiceDeps {
  readThread(threadId: string): Thread | null | undefined
  /** Current profile/storage namespace, captured again before save. */
  scopeKey(): string
  redactionContext?(thread: Thread): RedactionContext
  now?(): number
  newId?(): string
}
interface Pending { owner: string; scope: string; request: ReadableExportRequest; preview: ReadableExportPreview; revision: string }
export interface ReadableExportPacket { body: string; bytes: number; mimeType: string; suggestedName: string; format: ReadableExportRequest['format']; assertCurrent(): void }

export function readableExportRevision(thread: Thread, context: RedactionContext = {}): string {
  const tooLarge = () => { throw new Error('The active conversation exceeds the revision-check limit. Use a shorter branch.') }
  if (!Array.isArray(thread.timeline) || thread.timeline.length > READABLE_EXPORT_LIMITS.items) tooLarge()
  let characters = 0
  const timeline = thread.timeline.map(item => {
    const displayed = item.kind === 'user' || item.kind === 'assistant'
    const quotes = displayed ? readableQuotations(item) : []
    characters += displayed ? (item.text?.length ?? 0) + quotes.reduce((sum, quote) => sum + quote.sourceText.length + quote.text.length + quote.sourceItemId.length, 0) : 0
    if (characters > READABLE_EXPORT_LIMITS.bytes) tooLarge()
    return { id: item.id, kind: item.kind, text: displayed ? item.text : undefined, attachments: item.attachments?.length ?? 0, ...(quotes.length ? { quotedMessages: quotes } : {}) }
  })
  // Only displayed history is inspected. Raw API/model history never enters the export service.
  return createHash('sha256').update(JSON.stringify({ id: thread.id, title: thread.title, updatedAt: thread.updatedAt, revision: thread.conversationRevision ?? 0, context, timeline })).digest('hex')
}

/** No filesystem/Electron dependency: trusted IPC owns sender checks, chooser and atomic save. */
export function createReadableExportService(deps: ReadableExportServiceDeps) {
  const now = deps.now ?? Date.now, newId = deps.newId ?? randomUUID
  const previews = new Map<string, Pending>(), claims = new Map<string, Pending>()
  function current(record: Pending, owner: string): void {
    if (!owner || owner !== record.owner || deps.scopeKey() !== record.scope || now() >= record.preview.expiresAt) throw new Error('This export approval expired or belongs to another session. Create a new preview.')
    const thread = deps.readThread(record.request.threadId)
    if (!thread || readableExportRevision(thread, deps.redactionContext?.(thread) ?? {}) !== record.revision) throw new Error('The conversation or redaction settings changed. Review a new preview before saving.')
  }
  function prune(): void {
    for (const map of [previews, claims]) {
      for (const [id, record] of map) if (now() >= record.preview.expiresAt) map.delete(id)
      while (map.size >= 24) map.delete(map.keys().next().value!)
    }
  }
  return {
    preview(input: ReadableExportRequest, owner: string): ReadableExportPreview {
      if (!owner) throw new Error('An authenticated local export owner is required.')
      prune()
      const request = validateReadableExportRequest(input), thread = deps.readThread(request.threadId)
      if (!thread) throw new Error('Conversation not found.')
      const context = deps.redactionContext?.(thread) ?? {}
      const document = buildReadableExport(thread, request, context), revision = readableExportRevision(thread, context)
      const preview = { ...document, approvalId: newId(), expiresAt: now() + 5 * 60_000, revision }
      previews.set(preview.approvalId, { owner, scope: deps.scopeKey(), request, preview: structuredClone(preview), revision })
      return preview
    },
    claim(input: ReadableExportSaveRequest, owner: string): { claimId: string; suggestedName: string; mimeType: string; format: ReadableExportRequest['format'] } {
      if (!input || typeof input.approvalId !== 'string') throw new Error('Review the export preview before saving.')
      const record = previews.get(input.approvalId)
      if (!record) throw new Error('This export approval was already used or has expired. Create a new preview.')
      // The first attempt consumes approval even if the native chooser is canceled.
      previews.delete(input.approvalId); current(record, owner)
      const claimId = newId(); claims.set(claimId, record)
      return { claimId, suggestedName: record.preview.suggestedName, mimeType: record.preview.mimeType, format: record.preview.format }
    },
    finalize(claimId: string, owner: string): ReadableExportPacket {
      const record = claims.get(claimId)
      claims.delete(claimId)
      if (!record) throw new Error('This export save attempt was already completed or canceled.')
      current(record, owner)
      return { body: record.preview.body, bytes: record.preview.bytes, mimeType: record.preview.mimeType, suggestedName: record.preview.suggestedName, format: record.preview.format, assertCurrent: () => current(record, owner) }
    },
    cancel(claimId: string): void { claims.delete(claimId) },
    clear(): void { previews.clear(); claims.clear() },
  }
}
