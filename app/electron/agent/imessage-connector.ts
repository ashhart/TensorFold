import { blueBubblesAccess, blueBubblesIdentity, scanBlueBubbles, sendBlueBubbles } from './bluebubbles-connector'
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

function appleScriptString(s: string): string {
  return '"' + s
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t')
    + '"'
}

export function isIMessageGroupConversation(target: string): boolean {
  const value = target.trim()
  return /^chat[\w.-]+$/i.test(value) || /^(?:iMessage|SMS);[^;]*;chat[\w.-]+$/i.test(value)
}

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

// Send an iMessage reply via Messages.app (macOS only). Outbound — callers approval-gate it.
export async function sendIMessage(target: string, text: string, settings?: IMessageConnectorSettings, assertCurrent?: () => void): Promise<{ ok: boolean }> {
  assertCurrent?.()
  if (settings?.transport === 'bluebubbles') return sendBlueBubbles(settings, target, text, { assertCurrent })
  if (settings && (!settings.enabled || !isAllowedMessage({ chatId: target, handle: target } as IMessageItem, parseAllowedChats(settings.allowedChatIds)))) throw new Error('Choose an iMessage contact in the current connection scope before sending.')
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

export function imessageDbPath(): string {
  return path.join(os.homedir(), 'Library', 'Messages', 'chat.db')
}

const SQLITE3 = 'sqlite3'

export type IMessageAccessState = 'ok' | 'needs_full_disk_access' | 'no_macos' | 'missing_db' | 'error'
export async function imessageAccessState(settings?: IMessageConnectorSettings): Promise<{ state: IMessageAccessState; detail: string }> {
  if (settings?.transport === 'bluebubbles') {
    try { return await blueBubblesAccess(settings) } catch (error) { return { state: 'error', detail: error instanceof Error ? error.message : 'BlueBubbles connection failed.' } }
  }
  if (process.platform !== 'darwin') return { state: 'no_macos', detail: 'Use BlueBubbles to connect iMessage through a Mac.' }
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

export async function scanIMessageMessages(settings: IMessageConnectorSettings): Promise<IMessageItem[]> {
  if (!settings.enabled) throw new Error('Enable iMessage before reading selected conversations.')
  if (settings.transport === 'bluebubbles') return scanBlueBubbles(settings)
  if (process.platform !== 'darwin') throw new Error('iMessage is only available on macOS.')
  const limit = Math.max(25, Math.min(settings.maxMessages || 200, 1_000))
  const allowed = parseAllowedChats(settings.allowedChatIds)

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

export async function readIMessageMessage(id: string, settings?: IMessageConnectorSettings): Promise<IMessageReadResult | null> {
  if (settings) return (await scanIMessageMessages(settings)).find(item => item.id === id) ?? null
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

export async function syncIMessageConnector(settings: IMessageConnectorSettings, known: Set<string>, cursor?: SyncCursor | null): Promise<IMessageSyncResult> {
  const bridge = settings.transport === 'bluebubbles'
  if (bridge && settings.bridgeCursorIdentity !== blueBubblesIdentity(settings)) cursor = null
  const messages = await scanIMessageMessages(settings)
  const events: ConnectorEvent[] = []

  const seen = new Set(known)

  const byGuid = new Map<string, IMessageItem>()
  for (const m of messages) {
    if (m.guid) byGuid.set(m.guid, m)
  }
  for (const message of messages) {

    if (!isNewerThanCursor({ position: message.date, id: message.id }, cursor)) continue
    const messageKey = imessageMessageKey(message)
    if (seen.has(messageKey)) continue
    seen.add(messageKey)
    const event = imessageEventFromMessage(message, Date.now(), resolveIMessageReply(message, byGuid))
    if (bridge) event.threadKey = `imessage:bb-${blueBubblesIdentity(settings)}:${message.chatId}`
    events.push(event)
  }
  const advanced = advanceSyncCursor(cursor, messages.map((m) => ({ position: m.date, id: m.id })), Date.now()) ?? undefined
  return { events, messages: messages.length, syncedAt: Date.now(), cursor: advanced }
}

function resolveIMessageReply(message: IMessageItem, byGuid: Map<string, IMessageItem>): ConnectorReplyContext | undefined {
  if (!message.replyToId) return undefined
  const parent = byGuid.get(message.replyToId)
  if (!parent) return undefined
  const author = parent.isFromMe ? 'You' : (parent.chatName || parent.handle || 'Unknown')
  const text = parent.text?.trim()
  if (!text && !author) return undefined
  return { id: message.replyToId, author: author || null, text: text ? text.slice(0, 2_000) : null }
}

export function imessageMessageKey(message: Pick<IMessageItem, 'id'>): string {
  return `imessage:msg:${message.id}`
}

export function imessageConversationKey(message: Pick<IMessageItem, 'chatId' | 'handle'>): string {
  return `imessage:${message.chatId || message.handle}`
}

export function imessageEventDedupKey(event: { payload?: Record<string, unknown> | null; threadKey?: string }): string {
  const id = event.payload && typeof event.payload === 'object' ? (event.payload as { messageId?: unknown }).messageId : undefined
  return (typeof id === 'string' || typeof id === 'number') ? `imessage:msg:${id}` : (event.threadKey ?? '')
}

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

export function isAllowedMessage(message: Pick<IMessageItem, 'chatId' | 'handle'>, allowed: Set<string>): boolean {
  if (!allowed.size) return true
  return (!!message.chatId && allowed.has(message.chatId)) || (!!message.handle && allowed.has(message.handle))
}

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

export function parseAssociatedGuid(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const match = value.match(/([0-9A-Fa-f-]{36})/)
  return match ? match[1] : null
}
