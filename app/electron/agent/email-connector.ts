import { connectorSourceBinding } from './connector-source-binding'
import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type {
  ConnectorEvent,
  ConnectorSourceRef,
  EmailAttachment,
  EmailConnectorSettings,
  EmailDraftInput,
  EmailListOptions,
  EmailMessage,
  EmailReadResult,
  EmailSummaryResult,
  EmailSourceConfig,
  EmailSyncResult,
  EmailThreadResult,
  ConnectorReplyContext,
  SyncCursor,
} from './types'
import { ensureSummaryCitations } from './summary-citations'
import { fetchImapMessages, isImapSource, readImapMessage, validateImapSource } from './imap-fetch'
import { isOutlookSource, outlookReadResult, scanOutlookMessages, withDetectedOutlookSource } from './outlook-connector.ts'
import { advanceSyncCursor, imapUidFloor, injectReplyContext, isNewerThanCursor, makeReplyContext } from './connector-sync'

const MAX_FILE_BYTES = 2_000_000
const MAX_SCAN_FILES = 1_500
const EMAIL_EXTENSIONS = new Set(['.eml', '.emlx'])

export function makeEmailSource(dir: string, now = Date.now()): EmailSourceConfig {
  const kind = looksLikeMaildirPath(dir) ? 'maildir' : 'eml_dir'
  return {
    id: `email_${randomUUID().slice(0, 8)}`,
    label: path.basename(dir) || 'Email',
    path: dir,
    kind,
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
  }
}

// Validates that `dir` exists and is a folder before building the source config, so
// users get immediate feedback when adding a bad path instead of a silent failure that
// only surfaces on the next sync (#59).
export async function validateAndMakeEmailSource(dir: string, now = Date.now()): Promise<EmailSourceConfig> {
  let stat
  try {
    stat = await fs.stat(dir)
  } catch {
    throw new Error('Email source folder does not exist or is not accessible.')
  }
  if (!stat.isDirectory()) throw new Error('Email source folder does not exist or is not accessible.')
  return makeEmailSource(dir, now)
}

export async function scanEmailMessages(settings: EmailConnectorSettings, opts: EmailListOptions = {}, scanOpts: { incremental?: boolean; withAttachments?: boolean } = {}): Promise<{ messages: EmailMessage[]; sources: EmailSourceConfig[] }> {
  const now = Date.now()
  // Outlook-local auto-pickup is opt-in (detectOutlook): probing Outlook's Group Container raises a
  // macOS "data from other apps" prompt and stalls the scan, so a plain folder source never does.
  const detected = settings.detectOutlook ? await withDetectedOutlookSource(settings.sources, now) : settings.sources
  const sources = detected.map((source) => ({ ...source }))
  const messages: EmailMessage[] = []
  const maxMessages = Math.max(25, Math.min(settings.maxMessages || 300, 2_000))

  for (const source of sources) {
    if (!source.enabled) continue
    try {
      if (isImapSource(source)) {
        // Multi-account IMAP fetch: one account per source. `incremental` (sync only)
        // fetches just UID > cursor; read paths fetch the full recent window.
        const bad = validateImapSource(source)
        if (bad) throw new Error(bad)
        const imapMessages = await fetchImapMessages(source, { incremental: scanOpts.incremental, withAttachments: scanOpts.withAttachments })
        for (const message of imapMessages) messages.push(message)
      } else if (isOutlookSource(source)) {
        // Outlook for Mac local store: read its on-disk SQLite DB read-only (NO IMAP/SMTP). The
        // scanner degrades to [] when the DB is absent, so the source is inert until Outlook is
        // installed with a local profile. Messages come back already in the EmailMessage shape.
        const outlookMessages = await scanOutlookMessages(source, { limit: maxMessages })
        for (const message of outlookMessages) messages.push(message)
      } else {
        const files = await collectEmailFiles(source.path, source.kind)
        for (const file of files.slice(0, MAX_SCAN_FILES)) {
          try {
            const message = await readEmailFile(source, file)
            if (message) messages.push(message)
          } catch {
            // Ignore individual malformed messages; source-level errors are for unreadable roots.
          }
        }
      }
      source.lastSyncAt = now
      source.lastError = null
      source.updatedAt = now
    } catch (e: any) {
      source.lastError = String(e?.message ?? e)
      source.updatedAt = now
    }
  }

  let filtered = messages.sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
  if (opts.unreadOnly) filtered = filtered.filter((message) => message.unread)
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    filtered = filtered.filter((message) => [message.subject, message.from, message.to, message.snippet].some((value) => value.toLowerCase().includes(q)))
  }
  const limit = Math.max(1, Math.min(opts.limit ?? maxMessages, maxMessages))
  return { messages: filtered.slice(0, limit), sources }
}

export async function readEmailMessage(settings: EmailConnectorSettings, id: string): Promise<EmailReadResult> {
  const targetId = normalizeEmailId(id)
  for (const source of settings.sources.filter((s) => s.enabled)) {
    // IMAP account: fetch the single message by mailbox:uid over a fresh connection.
    if (isImapSource(source)) {
      try {
        const message = await readImapMessage(source, targetId)
        if (message) return message
      } catch {
        // fall through to "not found" below
      }
      continue
    }
    if (isOutlookSource(source)) {
      // Outlook local store: scan and match by id (no per-message DB round-trip needed — the
      // preview is the only body we surface). Keeps collectEmailFiles from seeing the .sqlite path.
      const scanned = await scanOutlookMessages(source, { limit: settings.maxMessages || 300 })
      const hit = scanned.find((m) => m.id === targetId)
      if (hit) return outlookReadResult(hit)
      continue
    }
    const files = await collectEmailFiles(source.path, source.kind)
    for (const file of files.slice(0, MAX_SCAN_FILES)) {
      const parsed = await readEmailFile(source, file, true)
      if (parsed?.id === targetId) return parsed
    }
  }
  throw new Error('Email message not found. Sync the connector and try again.')
}

export async function readEmailThread(settings: EmailConnectorSettings, id: string): Promise<EmailThreadResult> {
  const root = await readEmailMessage(settings, id)
  const subject = normalizeSubject(root.subject)
  const messageIds = new Set([
    normalizeMessageHeaderId(root.headers['message-id']),
    ...messageHeaderIds(root.headers.references),
    ...messageHeaderIds(root.headers['in-reply-to']),
  ].filter(Boolean))
  const participants = new Set([emailAddress(root.from), emailAddress(root.to)].filter(Boolean))
  const messages: EmailReadResult[] = []

  for (const source of settings.sources.filter((s) => s.enabled)) {
    if (isImapSource(source)) {
      // Gather thread messages from the IMAP account: fetch recent and filter to the thread.
      try {
        const recent = await fetchImapMessages(source)
        for (const parsed of recent) {
          if (parsed.id === root.id || messageBelongsToThread(parsed, subject, messageIds, participants)) {
            messages.push(parsed)
          }
        }
      } catch {
        // account unreachable during thread read — skip it
      }
      continue
    }
    if (isOutlookSource(source)) {
      // Gather thread members from the Outlook local store: scan recent and filter to the thread.
      const scanned = await scanOutlookMessages(source, { limit: settings.maxMessages || 300 })
      for (const message of scanned) {
        const parsed = outlookReadResult(message)
        if (parsed.id === root.id || messageBelongsToThread(parsed, subject, messageIds, participants)) messages.push(parsed)
      }
      continue
    }
    const files = await collectEmailFiles(source.path, source.kind)
    for (const file of files.slice(0, MAX_SCAN_FILES)) {
      const parsed = await readEmailFile(source, file, true)
      if (!parsed) continue
      if (parsed.id === root.id || messageBelongsToThread(parsed, subject, messageIds, participants)) messages.push(parsed)
    }
  }

  const unique = [...new Map(messages.map((message) => [message.id, message])).values()]
    .sort((a, b) => (a.date ?? 0) - (b.date ?? 0))
    .slice(-25)
  const unread = unique.filter((message) => message.unread).length
  const summary = `${unique.length} message${unique.length === 1 ? '' : 's'} in thread, ${unread} unread. Latest: ${unique.at(-1)?.subject || root.subject || '(no subject)'}`
  return { root, messages: unique.length ? unique : [root], summary }
}

export function emailSourceId(message: Pick<EmailMessage, 'sourceId' | 'id'>): string {
  return `email:${message.sourceId}:${message.id}`
}

export function emailSourceRef(message: EmailMessage): ConnectorSourceRef {
  return {
    id: emailSourceId(message),
    connectorId: 'email',
    sourceId: message.sourceId,
    itemId: message.id,
    title: message.subject || '(no subject)',
    path: message.path,
    excerpt: message.snippet,
    createdAt: message.date,
  }
}

export function formatEmailSourceRefs(messages: EmailMessage[]): string {
  const refs = messages.map(emailSourceRef)
  if (!refs.length) return 'Sources: none.'
  return [
    'Sources:',
    ...refs.map((ref) => `- ${ref.id} | ${ref.title}${ref.excerpt ? ` | ${ref.excerpt}` : ''}`),
  ].join('\n')
}

export async function summarizeEmail(settings: EmailConnectorSettings, opts: EmailListOptions = {}): Promise<EmailSummaryResult> {
  const { messages } = await scanEmailMessages(settings, { ...opts, limit: opts.limit ?? 50 })
  if (!messages.length) return ensureSummaryCitations({ summary: 'No email messages found for the current filters.\n\nSources: none.', sources: [] })
  const unread = messages.filter((m) => m.unread).length
  const senders = new Map<string, number>()
  for (const message of messages) senders.set(message.from || 'Unknown', (senders.get(message.from || 'Unknown') ?? 0) + 1)
  const topSenders = [...senders.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([sender, count]) => `${sender} (${count})`).join(', ')
  const recentMessages = messages.slice(0, 8)
  const recent = recentMessages.map((message) => `- [${emailSourceId(message)}] ${message.subject || '(no subject)'} from ${message.from || 'Unknown'}: ${message.snippet}`).join('\n')
  const sources = recentMessages.map(emailSourceRef)
  return ensureSummaryCitations({
    summary: `${messages.length} message${messages.length === 1 ? '' : 's'} scanned, ${unread} unread. Top senders: ${topSenders || 'none'}.\n\nRecent:\n${recent}\n\n${formatEmailSourceRefs(recentMessages)}`,
    sources,
  })
}

export async function syncEmailConnector(settings: EmailConnectorSettings, existingEvents: ConnectorEvent[]): Promise<EmailSyncResult> {
  const syncedAt = Date.now()
  // Sync uses incremental IMAP fetch (UID > cursor); the date-based cursor filter below
  // then covers file sources (and double-guards IMAP). Read paths stay full-window.
  // withAttachments pulls each message's decoded MIME attachment bytes (sync only) so the host
  // can cache them and stamp mediaPaths; read paths never download attachment bytes.
  const { messages, sources } = await scanEmailMessages(settings, { limit: settings.maxMessages }, { incremental: true, withAttachments: true })
  const known = new Set(existingEvents.map((event) => event.threadKey))
  // Index scanned messages by their RFC5322 Message-ID so an inbound reply can resolve
  // the quoted parent's author + snippet when both sit in the same scan batch.
  const byMessageId = new Map<string, EmailMessage>()
  for (const message of messages) {
    if (message.messageId) byMessageId.set(message.messageId, message)
  }
  const cursorBySource = new Map(sources.map((s) => [s.id, s.cursor]))
  // Group scanned messages per source so each source's cursor advances to its own newest.
  const bySource = new Map<string, EmailMessage[]>()
  for (const message of messages) {
    const list = bySource.get(message.sourceId) ?? []
    list.push(message)
    bySource.set(message.sourceId, list)
  }
  const events: ConnectorEvent[] = []
  // Decoded attachment bytes for the emitted events, keyed by the freshly-built event id, so the
  // host can write them to the media cache + stamp mediaPaths without re-opening IMAP.
  const attachmentsByEvent: Record<string, EmailAttachment[]> = {}
  // INCREMENTAL: only emit messages NEWER than each source's cursor. For IMAP, newness is
  // the UID (the fetch already pulled only UID > floor), NOT the sender-controlled Date
  // header: a genuinely new message whose Date is older than the date high-water mark
  // (delayed/queued delivery, a moved/copied message, clock skew, or a forged old Date:)
  // would otherwise be filtered out here and then permanently lost, since its UID is past
  // the next fetch floor. File sources (no UID) keep date-based gating.
  let emitted = 0
  for (const message of messages) {
    if (emitted >= 50) break
    const cursor = cursorBySource.get(message.sourceId)
    const imapUid = imapUidOf(message)
    const floor = imapUid != null ? imapUidFloor(cursor) : 0
    if (imapUid != null && floor > 0) {
      if (imapUid <= floor) continue // IMAP: gate by UID newness, not the sender's Date
    } else if (!isNewerThanCursor({ position: message.date, id: message.id }, cursor)) {
      continue
    }
    const threadKey = `email:${message.id}`
    if (!cursor && known.has(threadKey)) continue // full-resync fallback de-dupes by threadKey
    const event = emailEventFromMessage(message, syncedAt, resolveEmailReply(message, byMessageId))
    const selectedSource = sources.find(source => source.id === message.sourceId)
    if (selectedSource) event.sources = event.sources?.map(ref => ({ ...ref, sourceRevision: connectorSourceBinding(selectedSource) }))
    const attachments = (message as EmailReadResult).attachments
    if (attachments?.length) attachmentsByEvent[event.id] = attachments
    events.push(event)
    emitted++
  }
  // Advance + persist each source's cursor to the newest message it returned (the IMAP
  // UID floor is stamped too so the next IMAP fetch keeps using UID-range incremental fetch).
  const advancedSources = sources.map((source) => {
    const sourceMessages = bySource.get(source.id) ?? []
    if (!sourceMessages.length) return source
    const advanced = advanceSyncCursor(source.cursor, sourceMessages.map((m) => ({ position: m.date, id: m.id })), syncedAt) ?? undefined
    return { ...source, cursor: withImapUidFloor(advanced, sourceMessages) }
  })
  // Persist cursors for CONFIGURED sources only. An auto-detected Outlook-local source (prepended
  // ephemerally by scanEmailMessages when the DB exists) must NOT be written into settings — it is
  // re-detected fresh each sync, so it relies on the threadKey de-dup fallback above rather than a
  // persisted cursor. Without this filter the ephemeral source would leak into saved settings.
  const configuredIds = new Set(settings.sources.map((s) => s.id))
  const persistedSources = advancedSources.filter((s) => configuredIds.has(s.id))
  return {
    events,
    messages: messages.length,
    sources: persistedSources,
    syncedAt,
    ...(Object.keys(attachmentsByEvent).length ? { attachmentsByEvent } : {}),
  }
}

// Resolve the quoted author/text for a reply by matching its In-Reply-To Message-ID
// against the scanned batch. Returns undefined when the message isn't a reply or its
// parent isn't in the batch — best-effort, since a parent older than the scan window
// won't be present. Pure over an in-memory index.
function resolveEmailReply(message: EmailMessage, byMessageId: Map<string, EmailMessage>) {
  if (!message.inReplyTo) return undefined
  const parent = byMessageId.get(message.inReplyTo)
  if (!parent) return undefined
  return makeReplyContext({ id: parent.messageId ?? message.inReplyTo, author: parent.from, text: parent.snippet }) ?? undefined
}

// The IMAP UID for a message, parsed from its `<mailbox>:<uid>` id. Returns null for
// non-IMAP (file-source) messages whose ids aren't `<x>:<uid>`, so callers can tell
// IMAP messages (gated by UID) apart from file messages (gated by date).
export function imapUidOf(message: Pick<EmailMessage, 'path' | 'id'>): number | null {
  if (!message.path.startsWith('imap://')) return null
  const sep = message.id.lastIndexOf(':')
  if (sep < 0) return null
  const uid = Number(message.id.slice(sep + 1))
  return Number.isFinite(uid) ? uid : null
}

// IMAP cursors must carry the UID floor (lastImapUid) for the next incremental fetch.
// advanceSyncCursor only tracks the date position, so stamp the max UID parsed from the
// batch's IMAP ids (`<mailbox>:<uid>`). File-source ids aren't `<x>:<uid>`, so this is a
// no-op for them.
function withImapUidFloor(cursor: SyncCursor | undefined, messages: EmailMessage[]): SyncCursor | undefined {
  if (!cursor) return cursor
  let maxUid = cursor.lastImapUid ?? 0
  for (const m of messages) {
    const uid = imapUidOf(m)
    if (uid != null && uid > maxUid) maxUid = uid
  }
  return maxUid > 0 ? { ...cursor, lastImapUid: maxUid } : cursor
}

export interface EmailTransportResult {
  transport: 'local_outbox'
  eventId: string
  approvalId: string | null
  path: string
  deliveredAt: number
}

export function emailDraftEvent(input: EmailDraftInput, now = Date.now(), sources: ConnectorSourceRef[] = []): ConnectorEvent {
  const wantsSend = input.mode === 'send'
  const wantsReply = Boolean(input.inReplyTo || input.threadKey)
  const action = wantsSend
    ? wantsReply ? 'reply_requested' : 'send_requested'
    : wantsReply ? 'reply_draft_created' : 'draft_created'
  const label = wantsSend
    ? wantsReply ? 'Reply send request' : 'Send request'
    : wantsReply ? 'Reply draft' : 'Draft'
  return {
    id: randomUUID(),
    connectorId: 'email',
    pluginName: 'Email',
    action,
    direction: 'outbound',
    classification: 'approval',
    priority: wantsSend ? 'high' : 'normal',
    trustLevel: wantsSend ? 'approve' : 'suggest',
    requiresApproval: wantsSend,
    approvalId: wantsSend ? `approval-${randomUUID().slice(0, 8)}` : null,
    approvalStatus: wantsSend ? 'pending' : null,
    resolvedAt: null,
    resolvedBy: null,
    title: input.subject || label,
    summary: `${label} to ${input.to}: ${snippet(input.body, 220)}`,
    actor: input.to,
    threadKey: input.threadKey || input.inReplyTo || `email:draft:${randomUUID().slice(0, 8)}`,
    payloadKeys: ['to', 'subject', 'body', ...(input.threadKey ? ['thread_key'] : []), ...(input.inReplyTo ? ['in_reply_to'] : [])],
    payload: {
      mode: input.mode ?? 'draft',
      ...(input.accountId ? { accountId: input.accountId } : {}),
      to: input.to,
      subject: input.subject,
      body: input.body,
      ...(input.threadKey ? { threadKey: input.threadKey } : {}),
      ...(input.inReplyTo ? { inReplyTo: input.inReplyTo } : {}),
    },
    sources: sources.length ? sources : undefined,
    createdAt: now,
  }
}

export async function writeEmailOutboxMessage(event: ConnectorEvent, outboxDir: string, now = Date.now()): Promise<EmailTransportResult> {
  if (event.requiresApproval || event.approvalStatus !== 'approved') throw new Error('Email send events must be approved before transport.')
  if (event.connectorId !== 'email' || !['send_requested', 'reply_requested'].includes(event.action)) throw new Error('Connector event is not an email send request.')
  const input = emailDraftInputFromEvent(event)
  await fs.mkdir(outboxDir, { recursive: true, mode: 0o700 })
  // mkdir's mode does not tighten an already-existing directory. Outbox messages contain full
  // recipients, subjects, and bodies, so keep both the directory and each message owner-only.
  await fs.chmod(outboxDir, 0o700)
  const file = path.join(outboxDir, `${new Date(now).toISOString().replace(/[:.]/g, '-')}-${safeFilePart(event.id)}.eml`)
  await fs.writeFile(file, renderOutboxEmail(event, input, now), { encoding: 'utf8', mode: 0o600 })
  await fs.chmod(file, 0o600)
  return { transport: 'local_outbox', eventId: event.id, approvalId: event.approvalId, path: file, deliveredAt: now }
}

function emailDraftInputFromEvent(event: ConnectorEvent): EmailDraftInput {
  const payload = event.payload ?? {}
  const to = stringPayload(payload.to)
  const subject = stringPayload(payload.subject)
  const body = stringPayload(payload.body)
  if (!to || !subject || !body) throw new Error('Email send event is missing draft payload.')
  return {
    mode: payload.mode === 'send' ? 'send' : 'draft',
    to,
    subject,
    body,
    threadKey: stringPayload(payload.threadKey) || undefined,
    inReplyTo: stringPayload(payload.inReplyTo) || undefined,
  }
}

function renderOutboxEmail(event: ConnectorEvent, input: EmailDraftInput, now: number): string {
  const sourceIds = (event.sources ?? []).map((source) => source.id).join(', ')
  const headers = [
    ['To', input.to],
    ['Subject', input.subject],
    ['Date', new Date(now).toUTCString()],
    ['Message-ID', `<${event.id}@vontra.local>`],
    ['X-TensorFold-Studio-Transport', 'local_outbox'],
    ['X-TensorFold-Studio-Connector-Event', event.id],
    ['X-TensorFold-Studio-Approval-ID', event.approvalId ?? ''],
    ['X-TensorFold-Studio-Thread-Key', event.threadKey],
    ['X-TensorFold-Studio-Sources', sourceIds],
    ...(input.inReplyTo ? [['In-Reply-To', input.inReplyTo] as [string, string]] : []),
    ['Content-Type', 'text/plain; charset=utf-8'],
    ['Content-Transfer-Encoding', '8bit'],
  ] satisfies Array<[string, string]>
  return `${headers.map(([key, value]) => `${key}: ${headerValue(value)}`).join('\r\n')}\r\n\r\n${normalizeBody(input.body)}\r\n`
}

function stringPayload(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function headerValue(value: string): string {
  return String(value || '').replace(/[\r\n]+/g, ' ').trim()
}

function normalizeBody(value: string): string {
  return String(value || '').replace(/\r\n?/g, '\n').replace(/\n/g, '\r\n')
}

function safeFilePart(value: string): string {
  return String(value || '').replace(/[^A-Za-z0-9_.-]+/g, '-').slice(0, 80) || randomUUID()
}

export function emailThreadResolvedEvent(input: { id: string; note?: string }, thread: EmailThreadResult, now = Date.now()): ConnectorEvent {
  const sources = thread.messages.map(emailSourceRef)
  return {
    id: randomUUID(),
    connectorId: 'email',
    pluginName: 'Email',
    action: 'thread_resolved',
    direction: 'internal',
    classification: 'communication',
    priority: 'normal',
    trustLevel: 'auto_logged',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: now,
    resolvedBy: 'user',
    title: thread.root.subject || 'Email thread resolved',
    summary: `${input.note?.trim() || 'Marked resolved locally.'} ${thread.summary}`,
    actor: thread.root.from || 'Email',
    threadKey: `email:${thread.root.id}`,
    payloadKeys: ['message_id', ...(input.note ? ['note'] : [])],
    sources,
    createdAt: now,
  }
}

function emailEventFromMessage(message: EmailMessage, now: number, reply?: ConnectorReplyContext): ConnectorEvent {
  const baseSummary = message.snippet || 'Email has no readable plain-text preview.'
  return {
    id: randomUUID(),
    connectorId: 'email',
    pluginName: 'Email',
    action: 'message_received',
    direction: 'inbound',
    classification: 'communication',
    priority: message.unread ? 'high' : 'normal',
    trustLevel: 'suggest',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: null,
    resolvedBy: null,
    title: message.subject || '(no subject)',
    // Prepend the quoted parent so the agent's context shows what is being replied to.
    summary: reply ? injectReplyContext(baseSummary, reply) : baseSummary,
    actor: message.from || 'Unknown sender',
    threadKey: `email:${message.id}`,
    payloadKeys: ['message_id', 'source', 'subject', 'from', 'date', ...(reply ? ['reply'] : [])],
    // The raw From header rides in the payload so the host can learn an address ->
    // display-name contact (contact-directory) for "reply to <name>" resolution.
    payload: { from: message.from, subject: message.subject, messageId: message.messageId ?? message.id },
    ...(reply ? { reply } : {}),
    createdAt: message.date ?? now,
  }
}

function normalizeEmailId(id: string): string {
  return String(id || '').trim().replace(/^email:/, '')
}

function normalizeSubject(value: string): string {
  return String(value || '')
    .replace(/^\s*(re|fwd?|aw|sv):\s*/ig, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function normalizeMessageHeaderId(value: string | undefined): string {
  return String(value || '').trim().replace(/^<|>$/g, '').toLowerCase()
}

function messageHeaderIds(value: string | undefined): string[] {
  return String(value || '')
    .split(/\s+/)
    .map(normalizeMessageHeaderId)
    .filter(Boolean)
}

function emailAddress(value: string): string {
  const raw = String(value || '').toLowerCase()
  const match = raw.match(/<([^>]+)>/)
  return (match?.[1] || raw).replace(/\s+/g, '').trim()
}

function messageBelongsToThread(message: EmailReadResult, subject: string, messageIds: Set<string>, participants: Set<string>): boolean {
  const headerIds = [
    normalizeMessageHeaderId(message.headers['message-id']),
    ...messageHeaderIds(message.headers.references),
    ...messageHeaderIds(message.headers['in-reply-to']),
  ]
  if (headerIds.some((id) => messageIds.has(id))) return true
  if (!subject || normalizeSubject(message.subject) !== subject) return false
  const from = emailAddress(message.from)
  const to = emailAddress(message.to)
  return !participants.size || participants.has(from) || participants.has(to)
}

async function collectEmailFiles(root: string, kind: EmailSourceConfig['kind']): Promise<string[]> {
  const stat = await fs.stat(root)
  if (!stat.isDirectory()) throw new Error('Email source is not a folder.')
  const files: string[] = []
  const roots = kind === 'maildir'
    ? [path.join(root, 'new'), path.join(root, 'cur'), root]
    : [root]

  const seen = new Set<string>()
  const walk = async (dir: string, depth: number) => {
    if (depth > 8 || files.length >= MAX_SCAN_FILES) return
    let entries: any[] = []
    try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (files.length >= MAX_SCAN_FILES) break
      const full = path.join(dir, entry.name)
      if (seen.has(full)) continue
      seen.add(full)
      if (entry.isDirectory()) {
        if (entry.name === 'tmp' || entry.name === '.git' || entry.name === 'node_modules') continue
        await walk(full, depth + 1)
        continue
      }
      if (!entry.isFile()) continue
      const ext = path.extname(entry.name).toLowerCase()
      const inMaildirBucket = /[/\\](cur|new)[/\\]/.test(full)
      if (EMAIL_EXTENSIONS.has(ext) || inMaildirBucket) files.push(full)
    }
  }

  for (const rootDir of roots) await walk(rootDir, 0)
  return [...new Set(files)]
}

async function readEmailFile(source: EmailSourceConfig, file: string, includeBody = false): Promise<EmailReadResult | null> {
  const stat = await fs.stat(file)
  if (!stat.isFile() || stat.size > MAX_FILE_BYTES) return null
  let raw = await fs.readFile(file, 'utf8')
  raw = normalizeRawEmail(raw, file)
  const parsed = parseEmail(raw)
  const messageId = parsed.headers['message-id'] || `${file}:${stat.mtimeMs}:${stat.size}`
  const id = createHash('sha1').update(`${source.id}:${messageId}:${file}`).digest('hex').slice(0, 20)
  const date = parsed.headers.date ? Date.parse(parsed.headers.date) : stat.mtimeMs
  const message: EmailReadResult = {
    id,
    sourceId: source.id,
    sourceLabel: source.label,
    path: file,
    subject: decodeHeader(parsed.headers.subject || ''),
    from: decodeHeader(parsed.headers.from || ''),
    to: decodeHeader(parsed.headers.to || ''),
    date: Number.isFinite(date) ? date : stat.mtimeMs,
    unread: isUnread(file),
    snippet: snippet(parsed.body, 280),
    size: stat.size,
    // Message-ID + In-Reply-To are cheap to read (headers are already parsed) and let
    // sync resolve and inject reply-to context without loading every body.
    messageId: normalizeMessageHeaderId(parsed.headers['message-id']) || null,
    inReplyTo: normalizeMessageHeaderId(parsed.headers['in-reply-to']) || null,
    headers: includeBody ? parsed.headers : {},
    body: includeBody ? parsed.body.slice(0, 20_000) : '',
  }
  return message
}

function parseEmail(raw: string): { headers: Record<string, string>; body: string } {
  const normalized = raw.replace(/\r\n/g, '\n')
  const split = normalized.search(/\n\s*\n/)
  const headerText = split >= 0 ? normalized.slice(0, split) : ''
  const bodyText = split >= 0 ? normalized.slice(split).trim() : normalized
  const headers = parseHeaders(headerText)
  return { headers, body: extractReadableBody(headers, bodyText) }
}

function parseHeaders(headerText: string): Record<string, string> {
  const headers: Record<string, string> = {}
  const unfolded = headerText.replace(/\n[ \t]+/g, ' ')
  for (const line of unfolded.split('\n')) {
    const idx = line.indexOf(':')
    if (idx <= 0) continue
    const key = line.slice(0, idx).trim().toLowerCase()
    const value = line.slice(idx + 1).trim()
    headers[key] = headers[key] ? `${headers[key]}, ${value}` : value
  }
  return headers
}

// One MIME leaf we might surface as the readable body: its declared content-type,
// transfer-encoding, and the (still-encoded) text between its header block and its end.
interface MimeLeaf { contentType: string; encoding: string; body: string }

// Recursively walk a multipart container, descending into nested multiparts on each
// part's OWN boundary, and return the best text leaf: text/plain wins over text/html.
// This replaces the previous flat `split('--' + boundary)` which could not isolate a
// text/plain part nested inside a multipart/alternative (different inner boundary) and
// would substring-match a "Content-Type: text/plain" that appeared in quoted content.
function selectTextLeaf(contentType: string, bodyText: string): MimeLeaf | null {
  const boundary = contentType.match(/boundary="?([^";]+)"?/i)?.[1]
  if (!boundary) return null
  let plain: MimeLeaf | null = null
  let html: MimeLeaf | null = null
  // Split on the part delimiter; the preamble (before the first `--boundary`) and the
  // closing `--boundary--` epilogue are not parts and are ignored by the header parse.
  for (const segment of bodyText.split(`--${boundary}`)) {
    const part = segment.replace(/^--/, '') // tolerate the closing delimiter's trailing --
    const split = part.search(/\n\s*\n/)
    if (split < 0) continue
    const partHeaders = parseHeaders(part.slice(0, split))
    const partType = partHeaders['content-type'] || ''
    if (!partType) continue
    if (/^multipart\//i.test(partType)) {
      const nested = selectTextLeaf(partType, part.slice(split))
      if (nested) {
        if (/text\/plain/i.test(nested.contentType) && !plain) plain = nested
        else if (/text\/html/i.test(nested.contentType) && !html) html = nested
      }
      continue
    }
    const leaf: MimeLeaf = { contentType: partType, encoding: partHeaders['content-transfer-encoding'] || '', body: part.slice(split).trim() }
    if (/text\/plain/i.test(partType) && !plain) plain = leaf
    else if (/text\/html/i.test(partType) && !html) html = leaf
  }
  return plain ?? html
}

function extractReadableBody(headers: Record<string, string>, bodyText: string): string {
  const contentType = headers['content-type'] || ''
  let body = bodyText
  let partContentType = contentType
  // The matched part's own Content-Transfer-Encoding (if it declares one) takes priority over
  // the outer message's, since a part is commonly encoded independently of its container.
  let partEncoding: string | undefined
  const leaf = selectTextLeaf(contentType, bodyText)
  if (leaf) {
    body = leaf.body
    partEncoding = leaf.encoding
    partContentType = leaf.contentType
  }
  body = decodeTransfer(body, partEncoding || headers['content-transfer-encoding'] || '', charsetOf(partContentType))
  if (/text\/html/i.test(partContentType) || /<html|<body|<p[>\s]/i.test(body)) body = stripHtml(body)
  return body.replace(/\n{3,}/g, '\n\n').trim()
}

// The charset= parameter of a Content-Type header (defaulting to utf-8). Used so a
// quoted-printable/base64 part's bytes are decoded with the charset the sender declared.
function charsetOf(contentType: string): string {
  return contentType.match(/charset="?([^";]+)"?/i)?.[1]?.trim() || 'utf-8'
}

// Decode a byte buffer with a named charset, falling back to utf-8 then latin1 for
// charsets the runtime's TextDecoder doesn't know.
function decodeBytes(bytes: Buffer, charset: string): string {
  const cs = (charset || 'utf-8').toLowerCase().trim()
  if (cs === 'utf-8' || cs === 'utf8' || cs === 'us-ascii' || cs === 'ascii') return bytes.toString('utf8')
  try {
    return new TextDecoder(cs).decode(bytes)
  } catch {
    try { return bytes.toString('utf8') } catch { return bytes.toString('latin1') }
  }
}

function decodeTransfer(value: string, encoding: string, charset = 'utf-8'): string {
  if (/base64/i.test(encoding)) {
    try { return decodeBytes(Buffer.from(value.replace(/\s+/g, ''), 'base64'), charset) } catch { return value }
  }
  if (/quoted-printable/i.test(encoding) || /=\r?\n|=[0-9A-F]{2}/i.test(value)) return decodeQuotedPrintable(value, charset)
  return value
}

// Decode quoted-printable into text. Soft line-breaks (=<CRLF>) are removed, then the
// encoded `=XX` bytes AND any literal characters are accumulated into a byte buffer and
// decoded ONCE with the declared charset — so a multibyte UTF-8 sequence like
// "caf=C3=A9" reassembles to "café" instead of the per-byte mojibake ("cafÃ©") that
// String.fromCharCode(parseInt(...)) produced.
export function decodeQuotedPrintable(value: string, charset = 'utf-8'): string {
  const text = value.replace(/=\r?\n/g, '')
  const bytes: number[] = []
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '=' && /^[0-9A-Fa-f]{2}$/.test(text.slice(i + 1, i + 3))) {
      bytes.push(parseInt(text.slice(i + 1, i + 3), 16))
      i += 2
    } else {
      // A literal (already-decoded) character: append its UTF-8 bytes so any non-ASCII
      // literal in the source is preserved through the single charset decode below.
      for (const b of Buffer.from(ch, 'utf8')) bytes.push(b)
    }
  }
  return decodeBytes(Buffer.from(bytes), charset)
}

function decodeHeader(value: string): string {
  return value.replace(/=\?([^?]+)\?([BQbq])\?([^?]+)\?=/g, (_m, charset, enc, data) => {
    try {
      if (/b/i.test(enc)) return decodeBytes(Buffer.from(data, 'base64'), charset)
      return decodeQuotedPrintable(data.replace(/_/g, ' '), charset)
    } catch {
      return data
    }
  }).replace(/\s+/g, ' ').trim()
}

function stripHtml(value: string): string {
  return value
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

function normalizeRawEmail(raw: string, file: string): string {
  if (path.extname(file).toLowerCase() !== '.emlx') return raw
  const firstLine = raw.indexOf('\n')
  if (firstLine > 0 && /^\d+$/.test(raw.slice(0, firstLine).trim())) raw = raw.slice(firstLine + 1)
  const plist = raw.indexOf('<?xml')
  return plist > 0 ? raw.slice(0, plist) : raw
}

function snippet(value: string, max: number): string {
  return value.replace(/\s+/g, ' ').trim().slice(0, max)
}

function looksLikeMaildirPath(dir: string): boolean {
  return /[/\\](cur|new)$/.test(dir) || /Maildir/i.test(dir)
}

function isUnread(file: string): boolean {
  if (/[/\\]new[/\\]/.test(file)) return true
  const flags = file.match(/:2,([A-Z]*)$/)?.[1]
  return flags ? !flags.includes('S') : false
}
