import { validateMessageQuotes } from '../../shared/message-quotes'
import type { MessageQuote } from '../../shared/message-quotes'
import { composerDraftKey, validateComposerDraftScope, validateComposerDraftSnapshot, MAX_DRAFT_ACCEPTED_TEXT_CHARS, MAX_DRAFT_ATTACHMENTS } from '../../shared/composer-drafts'
import type { ComposerDraftPort, ComposerDraftScope, ComposerDraftSnapshot, DraftAttachmentReference } from '../../shared/composer-drafts'
import type { Attachment } from './types'

export interface ConversationDraftView {
  text: string
  attachments: Attachment[]
  quotes?: MessageQuote[]
  ready: boolean
  saving: boolean
  dirty: boolean
  error: string | null
  version: number
}

export interface ConversationDraftSendToken {
  key: string
  version: number
  text: string
  attachments: Attachment[]
  quotes?: MessageQuote[]
}

interface Entry {
  scope: ComposerDraftScope
  view: ConversationDraftView
  listeners: Set<() => void>
  serverRevision: number
  savedVersion: number
  hydrate: Promise<void> | null
  chain: Promise<void>
  timer: ReturnType<typeof setTimeout> | null
  readFailed: boolean
  textTouched: boolean
  attachmentsTouched: boolean
  quotesTouched: boolean
  generation: number
}

const RESELECT = 'Select this file again after restarting TensorFold; its contents were not stored with the draft.'
export const SESSION_ONLY_DRAFT_NOTICE = 'Drafts are kept in this tab; persistent drafts require the local desktop app.'

/** Strictly omit payloads, paths and source objects from durable references. */
export function draftAttachmentReferences(attachments: Attachment[]): DraftAttachmentReference[] {
  if (attachments.length > MAX_DRAFT_ATTACHMENTS) throw new Error(`Draft has more than ${MAX_DRAFT_ATTACHMENTS} file references; remove some files before restarting.`)
  return attachments.map((attachment) => {
    const extraction = (attachment as Attachment & { extraction?: { state: string; text: string } }).extraction
    if (extraction?.state === 'accepted' && extraction.text.length > MAX_DRAFT_ACCEPTED_TEXT_CHARS) throw new Error('Accepted extracted text exceeds the draft storage limit; copy it to a document before restarting.')
    const rawError = (attachment as Attachment & { error?: string }).error
    return {
      id: attachment.id, name: attachment.name, kind: attachment.kind, mime: attachment.mime,
      ...(attachment.bytes === undefined ? {} : { bytes: attachment.bytes }),
      status: 'needs-reselection', error: rawError ? rawError.slice(0, 1_000) : RESELECT,
      ...(extraction?.state === 'accepted' && typeof extraction.text === 'string' && extraction.text.length <= MAX_DRAFT_ACCEPTED_TEXT_CHARS ? { acceptedText: extraction.text } : {}),
    }
  })
}

export function restoreDraftAttachments(references: DraftAttachmentReference[]): Attachment[] {
  return references.map((reference) => ({
    id: reference.id, name: reference.name, kind: reference.kind, mime: reference.mime,
    ...(reference.bytes === undefined ? {} : { bytes: reference.bytes }),
    status: 'needs-reselection', error: reference.error && reference.error !== RESELECT ? `${RESELECT} Previous file error: ${reference.error}` : RESELECT,
    ...(reference.acceptedText === undefined ? {} : {
      text: reference.acceptedText,
      extraction: { method: 'apple-vision', state: 'accepted', text: reference.acceptedText, confidence: null, warnings: ['Original file must be selected again before sending. Previously accepted extracted text was retained.'], pageCount: 0, processedPages: 0, truncated: false },
    }),
  }))
}

function detached(attachments: Attachment[]): Attachment[] {
  // Attachment payloads are immutable values; clone the nested review object so an editor
  // cannot mutate another conversation or a captured send token through a shared reference.
  return attachments.map((attachment) => ({ ...attachment, ...('extraction' in attachment && attachment.extraction ? { extraction: structuredClone(attachment.extraction) } : {}) }))
}

/** One coordinator per transport/profile. Entries survive component unmounts and navigation. */
export class ConversationDrafts {
  private entries = new Map<string, Entry>()

  constructor(private port: ComposerDraftPort | null, private debounceMs = 180) {}

  private entry(scope: ComposerDraftScope): Entry {
    const key = composerDraftKey(scope)
    const known = this.entries.get(key)
    if (known) return known
    const valid = validateComposerDraftScope(scope)
    const entry: Entry = {
      scope: valid, listeners: new Set(), serverRevision: 0, savedVersion: 0, hydrate: null,
      chain: Promise.resolve(), timer: null, readFailed: false, textTouched: false, attachmentsTouched: false, quotesTouched: false, generation: 0,
      view: { text: '', attachments: [], quotes: [], ready: valid.incognito || !this.port, saving: false, dirty: false, error: !this.port && !valid.incognito ? SESSION_ONLY_DRAFT_NOTICE : null, version: 0 },
    }
    this.entries.set(key, entry)
    if (!entry.view.ready) entry.hydrate = this.hydrate(entry)
    return entry
  }

  private publish(entry: Entry, patch: Partial<ConversationDraftView>): void {
    entry.view = { ...entry.view, ...patch }
    for (const listener of entry.listeners) listener()
  }

  private async hydrate(entry: Entry): Promise<void> {
    const generation = entry.generation
    try {
      const saved = validateComposerDraftSnapshot(await this.port!.getComposerDraft(entry.scope))
      if (entry.generation !== generation) return
      entry.serverRevision = saved.revision
      entry.readFailed = false
      // New input always wins over late hydration, but its save uses the loaded CAS revision.
      this.publish(entry, {
        ...(!entry.textTouched ? { text: saved.text } : {}),
        ...(!entry.attachmentsTouched ? { attachments: restoreDraftAttachments(saved.attachments) } : {}),
        ...(!entry.quotesTouched ? { quotes: validateMessageQuotes(saved.quotes) } : {}),
        ready: true, error: null,
      })
      if (entry.view.dirty) this.schedule(entry)
    } catch {
      if (entry.generation !== generation) return
      entry.readFailed = true
      this.publish(entry, { ready: true, error: 'Could not load this draft; your new text stays in this window and will not overwrite saved work.' })
    }
  }

  get(scope: ComposerDraftScope): ConversationDraftView { return this.entry(scope).view }

  subscribe(scope: ComposerDraftScope, listener: () => void): () => void {
    const entry = this.entry(scope)
    entry.listeners.add(listener)
    return () => { entry.listeners.delete(listener) }
  }

  private schedule(entry: Entry): void {
    if (!this.port || entry.scope.incognito || entry.readFailed) return
    if (entry.timer) clearTimeout(entry.timer)
    entry.timer = setTimeout(() => {
      entry.timer = null
      void this.flush(entry.scope).catch(() => {})
    }, this.debounceMs)
  }

  update(scope: ComposerDraftScope, change: { text?: string; attachments?: Attachment[]; quotes?: MessageQuote[] }): void {
    const entry = this.entry(scope)
    const version = entry.view.version + 1
    const persistent = !!this.port && !entry.scope.incognito
    if (change.text !== undefined) entry.textTouched = true
    if (change.attachments !== undefined) entry.attachmentsTouched = true
    const quotes = change.quotes === undefined ? undefined : validateMessageQuotes(change.quotes)
    if (quotes?.length && (entry.scope.surface !== 'chat' || !entry.scope.conversationId || entry.scope.hostId !== null)) throw new Error('Quotes belong to a local conversation draft.')
    if (change.quotes !== undefined) entry.quotesTouched = true
    this.publish(entry, {
      ...(change.text === undefined ? {} : { text: change.text }),
      ...(change.attachments === undefined ? {} : { attachments: detached(change.attachments) }),
      ...(quotes === undefined ? {} : { quotes }),
      version, dirty: persistent,
      ...(!persistent || entry.readFailed ? {} : { error: null }),
    })
    if (!persistent) entry.savedVersion = version
    this.schedule(entry)
  }

  capture(scope: ComposerDraftScope): ConversationDraftSendToken {
    const entry = this.entry(scope)
    return { key: composerDraftKey(scope), version: entry.view.version, text: entry.view.text, attachments: detached(entry.view.attachments), ...(entry.view.quotes?.length ? { quotes: validateMessageQuotes(entry.view.quotes) } : {}) }
  }

  /** Accepted send clears only that exact input; later typing and other chats are untouched. */
  async clearAccepted(token: ConversationDraftSendToken): Promise<boolean> {
    const entry = this.entries.get(token.key)
    if (!entry || entry.view.version !== token.version) return false
    this.update(entry.scope, { text: '', attachments: [], quotes: [] })
    await this.flush(entry.scope)
    return true
  }

  flush(scope: ComposerDraftScope): Promise<void> {
    const entry = this.entry(scope)
    if (entry.timer) { clearTimeout(entry.timer); entry.timer = null }
    if (!this.port || entry.scope.incognito) return Promise.resolve()
    const next = entry.chain.catch(() => {}).then(async () => {
      const generation = entry.generation
      if (entry.hydrate) await entry.hydrate
      if (entry.generation !== generation) return
      // A damaged old draft must not trap an otherwise untouched window open. Only
      // unsaved local changes require a successful load/CAS before Quit can proceed.
      if (entry.savedVersion === entry.view.version && !entry.view.dirty) return
      if (entry.readFailed) throw new Error(entry.view.error ?? 'Draft could not be loaded.')
      while (entry.savedVersion !== entry.view.version) {
        const version = entry.view.version
        this.publish(entry, { saving: true })
        try {
          const snapshot = { text: entry.view.text, attachments: draftAttachmentReferences(entry.view.attachments), ...(entry.view.quotes?.length ? { quotes: validateMessageQuotes(entry.view.quotes) } : {}) }
          const result = await this.port!.saveComposerDraft({ scope: entry.scope, expectedRevision: entry.serverRevision, ...snapshot })
          if (entry.generation !== generation) return
          const saved = validateComposerDraftSnapshot(result.draft)
          if (!result.saved) throw new Error('Draft changed in another window; copy your text before reopening this conversation.')
          entry.serverRevision = saved.revision
          entry.savedVersion = version
          this.publish(entry, { saving: false, dirty: entry.savedVersion !== entry.view.version, error: null })
        } catch (error) {
          if (entry.generation !== generation) return
          // Keep local input and the old expected revision. A retry cannot silently clobber
          // another writer, and a failed save never marks the draft as safely persisted.
          const reportable = error instanceof Error && /^(Draft changed in another window|Draft has more than \d+ file references|Accepted extracted text exceeds the draft storage limit)/.test(error.message)
          this.publish(entry, { saving: false, dirty: true, error: reportable ? error.message : 'Draft could not be saved; keep this window open and retry.' })
          throw new Error(entry.view.error!)
        }
      }
    })
    entry.chain = next
    return next
  }

  async retry(scope: ComposerDraftScope): Promise<void> {
    const entry = this.entry(scope)
    if (entry.readFailed) entry.hydrate = this.hydrate(entry)
    await this.flush(scope)
  }

  async flushAll(): Promise<void> {
    await Promise.all([...this.entries.values()].map((entry) => this.flush(entry.scope)))
  }

  pendingReason(): string | null {
    for (const entry of this.entries.values()) {
      if (!this.port || entry.scope.incognito) continue
      if (entry.view.dirty || entry.view.saving) return entry.view.error ?? 'Saving your unfinished message before restarting TensorFold.'
    }
    return null
  }

  /** Called only after the host's explicit Forget/purge transaction commits. */
  forget(scope?: ComposerDraftScope | null): void {
    const key = scope ? composerDraftKey(scope) : null
    for (const [entryKey, entry] of this.entries) {
      if (key !== null && key !== entryKey) continue
      entry.generation++
      if (entry.timer) clearTimeout(entry.timer)
      entry.timer = null
      entry.serverRevision = 0
      entry.savedVersion = entry.view.version + 1
      entry.textTouched = false
      entry.attachmentsTouched = false
      entry.quotesTouched = false
      entry.readFailed = false
      entry.hydrate = null
      this.publish(entry, { text: '', attachments: [], quotes: [], ready: true, dirty: false, saving: false, error: null, version: entry.savedVersion })
    }
  }

  /** A first-turn privacy transition erases disk state but keeps unfinished input in RAM. */
  makeIncognito(scope: ComposerDraftScope): void {
    const privateScope = { ...scope, incognito: true }
    const normalScope = { ...scope, incognito: false }
    const previous = this.entries.get(composerDraftKey(normalScope))
    const unfinished = previous ? {
      text: previous.view.text,
      ...(previous.view.quotes?.length ? { quotes: validateMessageQuotes(previous.view.quotes) } : {}),
      attachments: detached(previous.view.attachments).map((attachment): Attachment => (attachment.status === 'processing' || attachment.status === 'review') && attachment.extraction?.state !== 'accepted'
        ? { ...attachment, status: 'needs-reselection', error: 'File processing was interrupted when you chose Incognito; select this file again in this private conversation.' }
        : attachment),
    } : null
    this.forget(normalScope)
    if (unfinished) this.update(privateScope, unfinished)
  }

  dispose(): void {
    for (const entry of this.entries.values()) if (entry.timer) clearTimeout(entry.timer)
  }
}

const transports = new WeakMap<ComposerDraftPort, ConversationDrafts>()
const memoryOnlyDrafts = new ConversationDrafts(null)

export function conversationDraftsFor(port: ComposerDraftPort | null): ConversationDrafts {
  if (!port) return memoryOnlyDrafts
  let drafts = transports.get(port)
  if (!drafts) { drafts = new ConversationDrafts(port); transports.set(port, drafts) }
  return drafts
}
