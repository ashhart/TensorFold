import { validateMessageQuotes } from './message-quotes'
import type { MessageQuote } from './message-quotes'
/** Draft identity deliberately excludes the selected model/provider account. */
export interface ComposerDraftScope {
  surface: 'chat' | 'assistant' | 'home'
  conversationId: string | null
  hostId: string | null
  incognito: boolean
}

/** Pending file payloads are not copied to disk. The user must select the file again. */
export interface DraftAttachmentReference {
  id: string
  name: string
  kind: 'image' | 'document'
  mime: string
  bytes?: number
  status: 'needs-reselection'
  error?: string
  /** Only text the user explicitly accepted in the extraction editor is retained. */
  acceptedText?: string
}

export interface ComposerDraftSnapshot {
  revision: number
  text: string
  attachments: DraftAttachmentReference[]
  quotes?: MessageQuote[]
  updatedAt: number
}

export interface ComposerDraftWrite {
  scope: ComposerDraftScope
  expectedRevision: number
  text: string
  attachments: DraftAttachmentReference[]
  quotes?: MessageQuote[]
}

export type ComposerDraftWriteResult = { saved: boolean; draft: ComposerDraftSnapshot }

export interface ComposerDraftPort {
  getComposerDraft(scope: ComposerDraftScope): Promise<ComposerDraftSnapshot>
  saveComposerDraft(input: ComposerDraftWrite): Promise<ComposerDraftWriteResult>
  clearComposerDraft(input: { scope: ComposerDraftScope; expectedRevision: number }): Promise<ComposerDraftWriteResult>
}

export const MAX_DRAFT_TEXT_CHARS = 1_000_000
// Includes failed/overflow file rows. The separate ingestion payload limit remains 20.
export const MAX_DRAFT_ATTACHMENTS = 200
export const MAX_DRAFT_ACCEPTED_TEXT_CHARS = 200_000

function object(value: unknown, description: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`Invalid ${description}.`)
  return value as Record<string, unknown>
}

function fields(value: Record<string, unknown>, allowed: string[], description: string): void {
  if (Object.keys(value).some((key) => !allowed.includes(key))) throw new Error(`Unsupported ${description} field.`)
}

function string(value: unknown, limit: number, description: string, allowEmpty = true): string {
  if (typeof value !== 'string' || value.length > limit || (!allowEmpty && !value.trim())) throw new Error(`Invalid ${description}.`)
  return value
}

export function validateComposerDraftScope(value: unknown): ComposerDraftScope {
  const raw = object(value, 'draft scope')
  fields(raw, ['surface', 'conversationId', 'hostId', 'incognito'], 'draft scope')
  if (!['chat', 'assistant', 'home'].includes(String(raw.surface)) || typeof raw.incognito !== 'boolean') throw new Error('Invalid draft scope.')
  const conversationId = raw.conversationId === null ? null : string(raw.conversationId, 16_384, 'conversation identity', false)
  const hostId = raw.hostId === null ? null : string(raw.hostId, 320, 'host identity', false)
  if (raw.surface === 'home' && conversationId !== null) throw new Error('Home drafts cannot belong to a conversation.')
  return { surface: raw.surface as ComposerDraftScope['surface'], conversationId, hostId, incognito: raw.incognito }
}

/** Collision-free in-memory identity. Disk filenames separately hash this value. */
export function composerDraftKey(scope: ComposerDraftScope): string {
  const valid = validateComposerDraftScope(scope)
  return JSON.stringify([valid.hostId, valid.conversationId, valid.conversationId === null ? valid.surface : 'conversation', valid.incognito])
}

export function validateDraftAttachmentReferences(value: unknown): DraftAttachmentReference[] {
  if (!Array.isArray(value) || value.length > MAX_DRAFT_ATTACHMENTS) throw new Error('Invalid draft attachments.')
  const seen = new Set<string>()
  return value.map((entry) => {
    const raw = object(entry, 'draft attachment')
    fields(raw, ['id', 'name', 'kind', 'mime', 'bytes', 'status', 'error', 'acceptedText'], 'draft attachment')
    const id = string(raw.id, 160, 'attachment identity', false)
    if (seen.has(id)) throw new Error('Duplicate draft attachment identity.')
    seen.add(id)
    if (!['image', 'document'].includes(String(raw.kind)) || raw.status !== 'needs-reselection') throw new Error('Invalid draft attachment status.')
    if (raw.bytes !== undefined && (!Number.isSafeInteger(raw.bytes) || Number(raw.bytes) < 0)) throw new Error('Invalid draft attachment size.')
    return {
      id, name: string(raw.name, 512, 'attachment filename', false), kind: raw.kind as DraftAttachmentReference['kind'],
      mime: string(raw.mime, 160, 'attachment media type'), status: 'needs-reselection',
      ...(raw.bytes === undefined ? {} : { bytes: Number(raw.bytes) }),
      ...(raw.error === undefined ? {} : { error: string(raw.error, 1_000, 'attachment error') }),
      ...(raw.acceptedText === undefined ? {} : { acceptedText: string(raw.acceptedText, MAX_DRAFT_ACCEPTED_TEXT_CHARS, 'accepted extraction text') }),
    }
  })
}

export function validateComposerDraftSnapshot(value: unknown): ComposerDraftSnapshot {
  const raw = object(value, 'saved draft')
  fields(raw, ['revision', 'text', 'attachments', 'quotes', 'updatedAt'], 'saved draft')
  if (!Number.isSafeInteger(raw.revision) || Number(raw.revision) < 0 || !Number.isSafeInteger(raw.updatedAt) || Number(raw.updatedAt) < 0) throw new Error('Invalid saved draft revision.')
  const quotes = validateMessageQuotes(raw.quotes)
  return { revision: Number(raw.revision), text: string(raw.text, MAX_DRAFT_TEXT_CHARS, 'draft text'), attachments: validateDraftAttachmentReferences(raw.attachments), ...(quotes.length ? { quotes } : {}), updatedAt: Number(raw.updatedAt) }
}

export function emptyComposerDraft(): ComposerDraftSnapshot {
  return { revision: 0, text: '', attachments: [], updatedAt: 0 }
}

export function validateComposerDraftWrite(value: unknown): ComposerDraftWrite {
  const raw = object(value, 'draft write')
  fields(raw, ['scope', 'expectedRevision', 'text', 'attachments', 'quotes'], 'draft write')
  if (!Number.isSafeInteger(raw.expectedRevision) || Number(raw.expectedRevision) < 0) throw new Error('Invalid expected draft revision.')
  const scope = validateComposerDraftScope(raw.scope), quotes = validateMessageQuotes(raw.quotes)
  if (quotes.length && (scope.surface !== 'chat' || !scope.conversationId || scope.hostId !== null)) throw new Error('Quotes belong to a local conversation draft.')
  return { scope, expectedRevision: Number(raw.expectedRevision), text: string(raw.text, MAX_DRAFT_TEXT_CHARS, 'draft text'), attachments: validateDraftAttachmentReferences(raw.attachments), ...(quotes.length ? { quotes } : {}) }
}
