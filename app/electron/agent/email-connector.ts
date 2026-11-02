import { emailSourceId, emailSourceRef, formatEmailSourceRefs } from './email-sources'
import { microsoftMailRuntime } from './email-provider-runtime'
import { connectorSourceBinding } from './connector-source-binding'
import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { ConnectorEvent, EmailAttachment, EmailConnectorSettings, EmailListOptions, EmailMessage, EmailReadResult, EmailSummaryResult, EmailSourceConfig, EmailSyncResult, EmailThreadResult, ConnectorReplyContext, SyncCursor } from './types'
import { ensureSummaryCitations } from './summary-citations'
import { fetchImapMessages, isImapSource, readImapMessage, validateImapSource } from './imap-fetch'
import { isOutlookSource, outlookReadResult, scanOutlookMessages, withDetectedOutlookSource } from './outlook-connector.ts'
import { advanceSyncCursor, imapUidFloor, injectReplyContext, isNewerThanCursor, makeReplyContext } from './connector-sync'
import { looksLikeMaildirPath, collectEmailFiles, MAX_SCAN_FILES, readEmailFile, normalizeMessageHeaderId } from './email-files'

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

  const detected = settings.detectOutlook ? await withDetectedOutlookSource(settings.sources, now) : settings.sources
  const sources = detected.map((source) => ({ ...source }))
  const messages: EmailMessage[] = []
  const maxMessages = Math.max(25, Math.min(settings.maxMessages || 300, 2_000))

  for (const source of sources) {
    if (!source.enabled) continue
    try {
      if (source.kind === 'microsoft_graph') {
        messages.push(...await microsoftMailRuntime().list(source, maxMessages))
      } else if (isImapSource(source)) {

        const bad = validateImapSource(source)
        if (bad) throw new Error(bad)
        const imapMessages = await fetchImapMessages(source, { incremental: scanOpts.incremental, withAttachments: scanOpts.withAttachments })
        for (const message of imapMessages) messages.push(message)
      } else if (isOutlookSource(source)) {

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
    if (source.kind === 'microsoft_graph') {
      const message = await microsoftMailRuntime().read(source, targetId)
      if (message) return message
      continue
    }
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

  for (const source of settings.sources.filter((s) => s.enabled && s.id === root.sourceId)) {
    if (source.kind === 'microsoft_graph') {
      const recent = await microsoftMailRuntime().list(source, settings.maxMessages || 300)
      for (const parsed of recent) {
        if (parsed.id === root.id || messageBelongsToThread(parsed, subject, messageIds, participants)) messages.push(parsed)
      }
      continue
    }
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

  const { messages, sources } = await scanEmailMessages(settings, { limit: settings.maxMessages }, { incremental: true, withAttachments: true })
  const known = new Set(existingEvents.map((event) => event.threadKey))

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

  const attachmentsByEvent: Record<string, EmailAttachment[]> = {}

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

  const advancedSources = sources.map((source) => {
    const sourceMessages = bySource.get(source.id) ?? []
    if (!sourceMessages.length) return source
    const advanced = advanceSyncCursor(source.cursor, sourceMessages.map((m) => ({ position: m.date, id: m.id })), syncedAt) ?? undefined
    return { ...source, cursor: withImapUidFloor(advanced, sourceMessages) }
  })

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

export

  function resolveEmailReply(message: EmailMessage, byMessageId: Map<string, EmailMessage>) {
  if (!message.inReplyTo) return undefined
  const parent = byMessageId.get(message.inReplyTo)
  if (!parent) return undefined
  return makeReplyContext({ id: parent.messageId ?? message.inReplyTo, author: parent.from, text: parent.snippet }) ?? undefined
}

export function imapUidOf(message: Pick<EmailMessage, 'path' | 'id'>): number | null {
  if (!message.path.startsWith('imap://')) return null
  const sep = message.id.lastIndexOf(':')
  if (sep < 0) return null
  const uid = Number(message.id.slice(sep + 1))
  return Number.isFinite(uid) ? uid : null
}

export

  function withImapUidFloor(cursor: SyncCursor | undefined, messages: EmailMessage[]): SyncCursor | undefined {
  if (!cursor) return cursor
  let maxUid = cursor.lastImapUid ?? 0
  for (const m of messages) {
    const uid = imapUidOf(m)
    if (uid != null && uid > maxUid) maxUid = uid
  }
  return maxUid > 0 ? { ...cursor, lastImapUid: maxUid } : cursor
}

export function emailEventFromMessage(message: EmailMessage, now: number, reply?: ConnectorReplyContext): ConnectorEvent {
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

    payload: { from: message.from, subject: message.subject, messageId: message.messageId ?? message.id },
    ...(reply ? { reply } : {}),
    createdAt: message.date ?? now,
  }
}

export function normalizeEmailId(id: string): string {
  return String(id || '').trim().replace(/^email:/, '')
}

export function normalizeSubject(value: string): string {
  return String(value || '')
    .replace(/^\s*(re|fwd?|aw|sv):\s*/ig, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

export function messageHeaderIds(value: string | undefined): string[] {
  return String(value || '')
    .split(/\s+/)
    .map(normalizeMessageHeaderId)
    .filter(Boolean)
}

export function emailAddress(value: string): string {
  const raw = String(value || '').toLowerCase()
  const match = raw.match(/<([^>]+)>/)
  return (match?.[1] || raw).replace(/\s+/g, '').trim()
}

export function messageBelongsToThread(message: EmailReadResult, subject: string, messageIds: Set<string>, participants: Set<string>): boolean {
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
export { type EmailTransportResult } from './email-actions'
export { emailDraftEvent } from './email-actions'
export { writeEmailOutboxMessage } from './email-actions'
export { emailThreadResolvedEvent } from './email-actions'
export { decodeQuotedPrintable } from './email-files'

export { emailSourceId, emailSourceRef, formatEmailSourceRefs } from './email-sources'
