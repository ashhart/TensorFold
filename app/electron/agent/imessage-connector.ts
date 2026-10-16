import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import type { ConnectorEvent, ConnectorReplyContext, IMessageItem, IMessageConnectorSettings, IMessageReadResult, IMessageSyncResult, SyncCursor } from './types'
import { advanceSyncCursor, injectReplyContext, isNewerThanCursor } from './connector-sync'

const execFileP = promisify(execFile)

const exec = (cmd: string, args: string[]) =>
  execFileP(cmd, args, { maxBuffer: 8_000_000, timeout: 15_000 }).then(({ stdout }) => stdout)

// Resolve iMessage handles (phone numbers / emails) to contact names via the macOS
// Contacts framework (AppleScript). This is what makes a bubble read "Maya Chen"
// instead of "+15551234567" (Bug 2). Runs once per scan over the unique handles;
// failures (no Contacts access, unknown handle) fall back to the raw handle.
// The lookup uses `name of first person whose value contains <handle>` — one
// osascript call handles a batch of handles via tab-delimited output.
async function resolveContactNames(handles: string[]): Promise<Map<string, string>> {
  const out = new Map<string, string>()
  const unique = [...new Set(handles.filter(Boolean))]
  if (!unique.length || process.platform !== 'darwin') return out
  // Batch up to ~80 handles per osascript call (Contacts lookups are slow; keep it bounded).
  const batch = unique.slice(0, 80)
  const lines = batch.map((h) => escapeAppleScript(h)).join(',')
  const script = [
    'use framework "Foundation"',
    'set AppleScript\'s text item delimiters to linefeed',
    'set out to {}',
    `set handles to {${lines}}`,
    'repeat with h in handles',
    '  try',
    '    set p to first person whose (value of every phone of every person contains h)',
    '    set end of out to (name of p) & ""',
    '  on error',
    '    try',
    '      set p to first person whose (value of every email of every person contains h)',
    '      set end of out to (name of p) & ""',
    '    on error',
    '      set end of out to h & ""',
    '    end try',
    '  end try',
    'end repeat',
    'set AppleScript\'s text item delimiters to ""',
    'return (out as string)',
  ].join('\n')
  try {
    const raw = await exec('/usr/bin/osascript', ['-e', `tell application "Contacts" to run "${script.replace(/"/g, '\\"')}"`])
    const names = raw.split(/\r?\n/)
    batch.forEach((h, i) => { if (names[i] && names[i] !== h) out.set(h, names[i]) })
  } catch {
    // Contacts access denied or not configured — fall back to raw handles.
  }
  return out
}

function escapeAppleScript(s: string): string {
  return '"' + s.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"'
}

// A fully-escaped AppleScript string literal. Escapes backslash, quote, and control chars so a
// recipient handle or message body can NEVER break out of the string (osascript injection defense).
function appleScriptString(s: string): string {
  return '"' + s
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t')
    + '"'
}

/**
 * Messages chat.db uses `chat…` identifiers for group conversations. Those are
 * existing text chats, not participant handles: asking Messages.app for a
 * participant named `chat…` makes every group reply fail. Keep this classifier
 * narrow so ordinary phone/email handles continue through the buddy path, while
 * also accepting the fully-qualified Messages.app chat id shape.
 */
export function isIMessageGroupConversation(target: string): boolean {
  const value = target.trim()
  return /^chat[\w.-]+$/i.test(value) || /^(?:iMessage|SMS);[^;]*;chat[\w.-]+$/i.test(value)
}

/** Build the AppleScript that sends `text` to a participant or existing group chat. Pure — unit-tested for escaping. */
export function buildIMessageSendScript(target: string, text: string): string {
  if (isIMessageGroupConversation(target)) {
    return [
      'tell application "Messages"',
      `  set targetChat to text chat id ${appleScriptString(target)}`,
      `  send ${appleScriptString(text)} to targetChat`,
      'end tell',
    ].join('\n')
  }
  return [
    'tell application "Messages"',
    '  set targetService to 1st account whose service type = iMessage',
    `  set targetBuddy to participant ${appleScriptString(target)} of targetService`,
    `  send ${appleScriptString(text)} to targetBuddy`,
    'end tell',
  ].join('\n')
}

/** Send an iMessage reply via Messages.app (macOS only). Outbound — callers approval-gate it. */
export async function sendIMessage(target: string, text: string): Promise<{ ok: boolean }> {
  if (process.platform !== 'darwin') throw new Error('iMessage sending is only available on macOS.')
  const to = target.trim()
  if (!to) throw new Error('iMessage recipient or conversation is required.')
  if (!text.trim()) throw new Error('Message body is required.')
  try {
    await exec('/usr/bin/osascript', ['-e', buildIMessageSendScript(to, text)])
    return { ok: true }
  } catch (e: any) {
    const detail = String(e?.stderr || e?.message || e)
    if (/not authori[sz]ed|-1743|automation|assistive access|Application isn.t running/i.test(detail)) {
      throw new Error('TensorFold needs Automation permission for Messages (System Settings → Privacy & Security → Automation), and Messages must be signed in.')
    }
    throw new Error(`iMessage send failed: ${detail.replace(/\s+/g, ' ').slice(0, 200)}`)
  }
}

// chat.db lives under the protected ~/Library/Messages directory. Reading it
// requires Full Disk Access granted to the app at runtime (System Settings ->
// Privacy & Security). No entitlement pre-grants it; we detect the resulting
// permission error and surface it so the UI can redirect the user.
export function imessageDbPath(): string {
  return path.join(os.homedir(), 'Library', 'Messages', 'chat.db')
}

// The system `sqlite3` CLI is always present on macOS and needs no native module
// (npmRebuild is off, so better-sqlite3 would break in the packaged app). We open
// the database read-only so we can never mutate the user's iMessage store.
const SQLITE3 = 'sqlite3'

// Detect whether TensorFold can read chat.db. Returns the kind of permission state so
// the UI can show a precise "grant Full Disk Access" redirect vs a real error.
export type IMessageAccessState = 'ok' | 'needs_full_disk_access' | 'no_macos' | 'missing_db' | 'error'
export async function imessageAccessState(): Promise<{ state: IMessageAccessState; detail: string }> {
  if (process.platform !== 'darwin') return { state: 'no_macos', detail: 'iMessage is only available on macOS.' }
  try {
    // A trivial read-only query proves both that the DB exists and that we have FDA.
    await exec(SQLITE3, ['-readonly', imessageDbPath(), 'SELECT count(*) FROM message LIMIT 1;'])
    return { state: 'ok', detail: 'iMessage local bridge ready.' }
  } catch (e: any) {
    const msg = String(e?.stderr || e?.message || e)
    if (/authorization denied|unable to open database|operation not permitted|restricted/i.test(msg)) {
      return { state: 'needs_full_disk_access', detail: 'Grant TensorFold Full Disk Access in System Settings → Privacy & Security to read iMessage.' }
    }
    if (/no such table|database disk image is malformed|no such file/i.test(msg)) {
      return { state: 'missing_db', detail: 'iMessage database was not found or is unreadable.' }
    }
    return { state: 'error', detail: msg.slice(0, 200) }
  }
}

// Fetch recent messages from chat.db, joined to handles for sender identity. Each
// row becomes an IMessageItem where isFromMe drives the bubble direction (blue out
// vs gray in). Limited to the most recent N across all chats by default.
export async function scanIMessageMessages(settings: IMessageConnectorSettings): Promise<IMessageItem[]> {
  if (process.platform !== 'darwin') throw new Error('iMessage is only available on macOS.')
  const limit = Math.max(25, Math.min(settings.maxMessages || 200, 1_000))
  const allowed = parseAllowedChats(settings.allowedChatIds)
  // Query recent messages with sender handle; is_from_me=1 means you sent it.
  // chat_identifier + message.date (CFAbsoluteTime epoch 2001-01-01) per Apple.
  const sql = [
    'SELECT m.ROWID, m.handle_id, m.is_from_me, m.text, m.date, m.cache_has_attachments,',
    'm.guid, m.associated_message_guid,',
    'h.id AS handle, c.chat_identifier, c.display_name',
    'FROM message m',
    'LEFT JOIN handle h ON h.ROWID = m.handle_id',
    'LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID',
    'LEFT JOIN chat c ON c.ROWID = cmj.chat_id',
    "WHERE m.text IS NOT NULL AND m.text <> ''",
    'ORDER BY m.date DESC',
    `LIMIT ${limit * 4};`,
  ].join(' ')
  const raw = await exec(SQLITE3, ['-readonly', '-json', imessageDbPath(), sql])
  const rows: any[] = raw.trim() ? JSON.parse(raw) : []
  let items = rows.map(rowToIMessage).filter((m): m is IMessageItem => !!m)
  // Resolve raw handles (phone/email) to contact names so bubbles read "Maya Chen"
  // instead of "+15551234567" (Bug 2). Falls back to the handle if Contacts is unavailable.
  const handles = items.map((it) => it.handle).filter((h, i) => h && !items[i].isFromMe)
  const names = await resolveContactNames(handles)
  if (names.size) {
    items = items.map((m) => {
      const name = m.handle ? names.get(m.handle) : undefined
      return name ? { ...m, chatName: m.chatName || name } : m
    })
  }
  if (allowed.size) items = items.filter((m) => isAllowedMessage(m, allowed))
  return items.slice(0, limit).sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
}

export async function readIMessageMessage(id: string): Promise<IMessageReadResult | null> {
  if (process.platform !== 'darwin') throw new Error('iMessage is only available on macOS.')
  const rowid = Number(id)
  if (!Number.isFinite(rowid)) return null
  const sql = [
    'SELECT m.ROWID, m.handle_id, m.is_from_me, m.text, m.date, m.cache_has_attachments,',
    'm.guid, m.associated_message_guid,',
    'h.id AS handle, c.chat_identifier, c.display_name',
    'FROM message m',
    'LEFT JOIN handle h ON h.ROWID = m.handle_id',
    'LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID',
    'LEFT JOIN chat c ON c.ROWID = cmj.chat_id',
    `WHERE m.ROWID = ${rowid};`,
  ].join(' ')
  const raw = await exec(SQLITE3, ['-readonly', '-json', imessageDbPath(), sql])
  const rows: any[] = raw.trim() ? JSON.parse(raw) : []
  if (!rows.length) return null
  return rowToIMessage(rows[0]) ?? null
}

// Emit connector events for newly-seen messages so the Assistant surfaces them.
// `cursor` is the incremental high-water mark (newest message date+ROWID already
// ingested): a sync skips anything at/below it before the per-message dedup, so an
// ever-growing chat.db isn't fully re-walked into events each poll. A null cursor (or
// the runtime omitting it) falls back to the dedup-only full scan. The advanced cursor
// is returned for the host to persist.
export async function syncIMessageConnector(settings: IMessageConnectorSettings, known: Set<string>, cursor?: SyncCursor | null): Promise<IMessageSyncResult> {
  const messages = await scanIMessageMessages(settings)
  const events: ConnectorEvent[] = []
  // Dedup by MESSAGE identity, not conversation. `known` is the set of per-message
  // dedup keys the runtime rebuilds via imessageEventDedupKey (payload.messageId), so
  // a conversation that has already surfaced one event still ingests its later
  // messages — one event per new message — while the event's threadKey stays
  // conversation-scoped for grouping/replies.
  const seen = new Set(known)
  // Resolve quoted-reply context within the batch: chat.db threads a reply via the
  // associated-message GUID (IMessageItem.replyToId), matched against a sibling's guid.
  const byGuid = new Map<string, IMessageItem>()
  for (const m of messages) {
    if (m.guid) byGuid.set(m.guid, m)
  }
  for (const message of messages) {
    // INCREMENTAL: drop anything not strictly newer than the cursor (position = date,
    // id = ROWID) before the dedup work.
    if (!isNewerThanCursor({ position: message.date, id: message.id }, cursor)) continue
    const messageKey = imessageMessageKey(message)
    if (seen.has(messageKey)) continue
    seen.add(messageKey)
    events.push(imessageEventFromMessage(message, Date.now(), resolveIMessageReply(message, byGuid)))
  }
  const advanced = advanceSyncCursor(cursor, messages.map((m) => ({ position: m.date, id: m.id })), Date.now()) ?? undefined
  return { events, messages: messages.length, syncedAt: Date.now(), cursor: advanced }
}

// Resolve a reply's quoted message from the batch via the associated-message GUID
// chat.db stores on a reply. Returns undefined when the message isn't a reply or its
// parent isn't in the batch. Pure over the in-memory index.
function resolveIMessageReply(message: IMessageItem, byGuid: Map<string, IMessageItem>): ConnectorReplyContext | undefined {
  if (!message.replyToId) return undefined
  const parent = byGuid.get(message.replyToId)
  if (!parent) return undefined
  const author = parent.isFromMe ? 'You' : (parent.chatName || parent.handle || 'Unknown')
  const text = parent.text?.trim()
  if (!text && !author) return undefined
  return { id: message.replyToId, author: author || null, text: text ? text.slice(0, 2_000) : null }
}

// Stable per-message dedup/identity key. Must be unique per message (not per
// conversation) or an ongoing chat collapses to a single ever-emitted event.
export function imessageMessageKey(message: Pick<IMessageItem, 'id'>): string {
  return `imessage:msg:${message.id}`
}

// Conversation-scoped key. This is the event's threadKey: the Assistant groups a
// conversation's events by threadKey (Assistant.tsx) and replies target it, so it
// MUST stay per-conversation — dedup uses imessageMessageKey, never this.
export function imessageConversationKey(message: Pick<IMessageItem, 'chatId' | 'handle'>): string {
  return `imessage:${message.chatId || message.handle}`
}

// Per-message dedup key for a STORED event, used by the runtime to rebuild `known`.
// Prefers the persisted source message id (payload.messageId); falls back to the
// (now conversation-scoped) threadKey only for pre-upgrade events that lack it.
export function imessageEventDedupKey(event: { payload?: Record<string, unknown> | null; threadKey?: string }): string {
  const id = event.payload && typeof event.payload === 'object' ? (event.payload as { messageId?: unknown }).messageId : undefined
  return (typeof id === 'string' || typeof id === 'number') ? `imessage:msg:${id}` : (event.threadKey ?? '')
}

// The event's createdAt MUST be the real message time, not the ingest time, so the
// Assistant's thread list (sorted by createdAt) shows the newest CONVERSATION at the top.
// A whole sync batch is remembered with one Date.now(), so using ingest time would make a
// fresh resync of historical chats order arbitrarily; the message's own timestamp keeps
// "newest message at the top" stable regardless of when it was scanned. Falls back to `now`
// when chat.db gave no/garbage date, and clamps a future timestamp to now so a bad clock on
// one row can't pin it to the top forever. Pure + unit-tested.
export function imessageEventTimestamp(message: Pick<IMessageItem, 'date'>, now = Date.now()): number {
  const date = message.date
  if (typeof date !== 'number' || !Number.isFinite(date) || date <= 0) return now
  return Math.min(date, now)
}

export function imessageEventFromMessage(message: IMessageItem, now = Date.now(), reply?: ConnectorReplyContext): ConnectorEvent {
  const isFromMe = message.isFromMe
  const baseSummary = message.text.slice(0, 220)
  return {
    id: randomUUID(),
    connectorId: 'imessage',
    pluginName: 'iMessage',
    action: isFromMe ? 'message_sent' : 'message_received',
    direction: isFromMe ? 'outbound' : 'inbound',
    classification: 'communication',
    priority: 'normal',
    trustLevel: 'notify',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: null,
    resolvedBy: null,
    title: message.chatName || message.handle || 'iMessage',
    // Prepend the quoted parent so the agent's context shows what is being replied to.
    summary: reply ? injectReplyContext(baseSummary, reply) : baseSummary,
    actor: message.chatName || message.handle || (isFromMe ? 'You' : 'Unknown'),
    // Conversation-scoped so the Assistant groups a chat's messages together and
    // replies target the conversation. Per-message dedup is keyed off payload.messageId
    // (see imessageEventDedupKey), not this.
    threadKey: imessageConversationKey(message),
    payloadKeys: ['text', 'handle', 'chat', ...(reply ? ['reply'] : [])],
    payload: { text: message.text, handle: message.handle, chat: message.chatId, messageId: message.id },
    ...(reply ? { reply } : {}),
    // Real message time (not ingest time) so the thread list orders newest-conversation-first.
    createdAt: imessageEventTimestamp(message, now),
  }
}

function parseAllowedChats(raw: string | undefined): Set<string> {
  if (!raw) return new Set()
  return new Set(raw.split(/[\s,;]+/).map((s) => s.trim()).filter(Boolean))
}

// Allowlist gate. A message is allowed only if its chat_identifier OR its sender
// handle is explicitly listed. An empty chatId must NOT auto-pass: chat_message_join
// is a LEFT JOIN, so messages with no chat row resolve chatId to '' and would
// otherwise slip past the allowlist entirely. Matching the handle too lets users
// gate by the contact (phone/email) they actually configured, not just chat ids.
export function isAllowedMessage(message: Pick<IMessageItem, 'chatId' | 'handle'>, allowed: Set<string>): boolean {
  if (!allowed.size) return true
  return (!!message.chatId && allowed.has(message.chatId)) || (!!message.handle && allowed.has(message.handle))
}

// Apple's `date` column is CFAbsoluteTime since 2001-01-01 UTC. Older databases
// store seconds; modern Messages databases commonly store nanoseconds. Detect the
// magnitude before converting or a modern row lands millions of years in the future.
const APPLE_EPOCH_OFFSET_MS = 978_307_200_000
export function appleDateToMs(date: number | null | undefined): number | null {
  if (date == null || !Number.isFinite(date)) return null
  const sinceAppleEpochMs = Math.abs(date) >= 1_000_000_000_000_000
    ? date / 1_000_000 // modern chat.db: nanoseconds
    : date * 1000 // legacy chat.db: seconds
  return Math.round(sinceAppleEpochMs) + APPLE_EPOCH_OFFSET_MS
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function rowToIMessage(row: any): IMessageItem | null {
  if (!row) return null
  const text = typeof row.text === 'string' ? row.text : String(row.text ?? '')
  if (!text.trim()) return null
  return {
    id: String(row.ROWID),
    handle: row.handle || '',
    isFromMe: row.is_from_me === 1 || row.is_from_me === '1' || row.is_from_me === true,
    text,
    date: appleDateToMs(Number(row.date)),
    hasAttachments: Boolean(row.cache_has_attachments),
    chatId: row.chat_identifier || '',
    chatName: row.display_name || row.chat_identifier || '',
    guid: typeof row.guid === 'string' ? row.guid : null,
    replyToId: parseAssociatedGuid(row.associated_message_guid),
  }
}

// chat.db stores a reply's parent in associated_message_guid as `bp:<guid>`, `p:0/<guid>`,
// or a bare guid. Tapbacks (likes/emphasis) use the same column, so only treat it as a
// THREADED reply target when it parses to a guid; the bare guid is what message.guid holds.
export function parseAssociatedGuid(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const match = value.match(/([0-9A-Fa-f-]{36})/)
  return match ? match[1] : null
}
