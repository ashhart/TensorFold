import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type {
  ConnectorEvent,
  ConnectorSourceRef,
  SlackConnectorSettings,
  SlackListOptions,
  SlackMessage,
  SlackReadResult,
  SlackSourceConfig,
  SlackSummaryResult,
  SlackSyncResult,
  ConnectorReplyContext,
} from './types'
import { ensureSummaryCitations } from './summary-citations'
import { fetchSlackMessages, isLiveSlackSource, postSlackMessage, readSlackThread, validateLiveSlackSource } from './slack-live'
import { advanceSyncCursor, injectReplyContext, isNewerThanCursor, makeReplyContext } from './connector-sync'

const MAX_SCAN_FILES = 2_000
const MAX_FILE_BYTES = 4_000_000

interface SlackExportMetadata {
  channels: Map<string, string>
  users: Map<string, string>
}

export function makeSlackSource(dir: string, now = Date.now()): SlackSourceConfig {
  return {
    id: `slack_${randomUUID().slice(0, 8)}`,
    label: path.basename(dir) || 'Slack',
    path: dir,
    kind: 'export_dir',
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
  }
}

export async function scanSlackMessages(settings: SlackConnectorSettings, opts: SlackListOptions = {}): Promise<{ messages: SlackMessage[]; sources: SlackSourceConfig[] }> {
  const now = Date.now()
  const sources = settings.sources.map((source) => ({ ...source }))
  const maxMessages = Math.max(25, Math.min(settings.maxMessages || 300, 2_000))
  const messages: SlackMessage[] = []

  for (const source of sources) {
    if (!source.enabled) continue
    try {
      if (isLiveSlackSource(source)) {
        // Live workspace token: fetch channel history + threads via the Web API.
        const bad = validateLiveSlackSource(source)
        if (bad) throw new Error(bad)
        messages.push(...await fetchSlackMessages(source))
      } else {
        const metadata = await readSlackMetadata(source.path)
        const files = await collectSlackMessageFiles(source.path)
        for (const file of files.slice(0, MAX_SCAN_FILES)) {
          try {
            messages.push(...(await readSlackFile(source, file, metadata)).map(publicSlackMessage))
          } catch {
            // Ignore malformed message files; source errors are for unreadable export roots.
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
  if (opts.channel?.trim()) {
    const channel = normalizeChannel(opts.channel)
    filtered = filtered.filter((message) => normalizeChannel(message.channelName) === channel || message.channelId.toLowerCase() === channel)
  }
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    filtered = filtered.filter((message) => [message.text, message.channelName, message.senderName, message.snippet].some((value) => value.toLowerCase().includes(q)))
  }
  const limit = Math.max(1, Math.min(Number(opts.limit) || maxMessages, maxMessages))
  return { messages: filtered.slice(0, limit), sources }
}

export async function readSlackMessage(settings: SlackConnectorSettings, id: string): Promise<SlackReadResult> {
  const target = normalizeSlackId(id)
  for (const source of settings.sources.filter((s) => s.enabled)) {
    if (isLiveSlackSource(source)) {
      // Live: id is "<channel>:<ts>" or "<channel>/<ts>". Read the thread.
      const { channel, ts } = parseLiveSlackId(id)
      if (channel && ts) {
        const thread = await readSlackThread(source, channel, ts)
        const found = thread.find((m) => m.ts === ts)
        if (found) return found
      }
      continue
    }
    const metadata = await readSlackMetadata(source.path)
    const files = await collectSlackMessageFiles(source.path)
    for (const file of files.slice(0, MAX_SCAN_FILES)) {
      const messages = await readSlackFile(source, file, metadata)
      const found = messages.find((message) => message.id === target || message.ts === id)
      if (found) return found
    }
  }
  throw new Error('Slack message not found. Sync the connector and try again.')
}

// Live message ids carry extra prefixes (e.g. "slack:<sourceId>:<sourceId>:<channel>:<ts>"
// from slackSourceId + toSlackMessage), so parse from the RIGHT: the ts is the trailing
// "<digits>.<digits>" segment, and the channel is the C/G-prefixed segment just before it.
export function parseLiveSlackId(id: string): { channel: string; ts: string } {
  // Split on both delimiters used across the id schemes ("/" and ":").
  const parts = id.split(/[:/]/).filter(Boolean)
  // Find the last ts-shaped segment so a leading numeric source token can't be mistaken for it.
  let tsIndex = -1
  for (let i = parts.length - 1; i >= 0; i--) {
    if (/^\d+\.\d+$/.test(parts[i])) { tsIndex = i; break }
  }
  if (tsIndex < 0) return { channel: '', ts: '' }
  const ts = parts[tsIndex]
  // The channel id is the segment immediately preceding the ts in every id scheme here
  // ("<channel>:<ts>", "<channel>/<ts>", "slack:<sourceId>:<sourceId>:<channel>:<ts>").
  const channel = parts[tsIndex - 1] ?? ''
  return { channel, ts }
}

// Transport an approved Slack reply over the Web API. Only call after approval.
export async function transportSlackReply(source: SlackSourceConfig, channel: string, text: string, threadTs?: string): Promise<{ ok: boolean; ts?: string }> {
  if (!isLiveSlackSource(source)) throw new Error('Only live (api_token) Slack sources can send messages.')
  return postSlackMessage(source, channel, text, threadTs)
}

export async function summarizeSlack(settings: SlackConnectorSettings, opts: SlackListOptions = {}): Promise<SlackSummaryResult> {
  const { messages } = await scanSlackMessages(settings, { ...opts, limit: opts.limit ?? 50 })
  if (!messages.length) return ensureSummaryCitations({ summary: 'No Slack messages found for the current filters.\n\nSources: none.', sources: [] })
  const channels = new Map<string, number>()
  for (const message of messages) channels.set(message.channelName, (channels.get(message.channelName) ?? 0) + 1)
  const topChannels = [...channels.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([channel, count]) => `#${channel} (${count})`).join(', ')
  const recentMessages = messages.slice(0, 8)
  const recent = recentMessages.map((message) => `- [${slackSourceId(message)}] #${message.channelName} ${message.senderName || 'Unknown'}: ${message.snippet}`).join('\n')
  const sources = recentMessages.map(slackSourceRef)
  return ensureSummaryCitations({
    summary: `${messages.length} Slack message${messages.length === 1 ? '' : 's'} scanned. Top channels: ${topChannels || 'none'}.\n\nRecent:\n${recent}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}${source.excerpt ? ` | ${source.excerpt}` : ''}`).join('\n')}`,
    sources,
  })
}

export async function syncSlackConnector(settings: SlackConnectorSettings, existingEvents: ConnectorEvent[]): Promise<SlackSyncResult> {
  const syncedAt = Date.now()
  const { messages, sources } = await scanSlackMessages(settings, { limit: settings.maxMessages })
  const known = new Set(existingEvents.map((event) => event.threadKey))
  // Index messages by ts so a threaded reply (threadTs -> root ts) can quote its root.
  const byTs = new Map(messages.map((message) => [message.ts, message]))
  const cursorBySource = new Map(sources.map((s) => [s.id, s.cursor]))
  // Group per source so each source's cursor advances to its own newest message.
  const bySource = new Map<string, SlackMessage[]>()
  for (const message of messages) {
    const list = bySource.get(message.sourceId) ?? []
    list.push(message)
    bySource.set(message.sourceId, list)
  }
  const events: ConnectorEvent[] = []
  // INCREMENTAL: only emit messages newer than each source's cursor (position = date).
  let emitted = 0
  for (const message of messages) {
    if (emitted >= 50) break
    const cursor = cursorBySource.get(message.sourceId)
    if (!isNewerThanCursor({ position: message.date, id: message.id }, cursor)) continue
    const threadKey = slackThreadKey(message)
    if (!cursor && known.has(threadKey)) continue // full-resync fallback de-dupes by threadKey
    events.push(slackEventFromMessage(message, syncedAt, resolveSlackReply(message, byTs)))
    emitted++
  }
  const advancedSources = sources.map((source) => {
    const sourceMessages = bySource.get(source.id) ?? []
    if (!sourceMessages.length) return source
    const cursor = advanceSyncCursor(source.cursor, sourceMessages.map((m) => ({ position: m.date, id: m.id })), syncedAt) ?? undefined
    return { ...source, cursor }
  })
  return { events, messages: messages.length, sources: advancedSources, syncedAt }
}

// Resolve a threaded reply's quoted root (threadTs points at the root ts). Skips the
// self-reference a root has to itself (threadTs === ts). Returns undefined when the
// root isn't in the batch. Pure over the in-memory index.
function resolveSlackReply(message: SlackMessage, byTs: Map<string, SlackMessage>): ConnectorReplyContext | undefined {
  const rootTs = message.threadTs
  if (!rootTs || rootTs === message.ts) return undefined
  const root = byTs.get(rootTs)
  if (!root) return undefined
  return makeReplyContext({ id: root.id, author: root.senderName, text: root.snippet || root.text }) ?? undefined
}

export function slackSourceId(message: Pick<SlackMessage, 'sourceId' | 'id'>): string {
  return `slack:${message.sourceId}:${message.id}`
}

// The routing threadKey (parsed by parseThreadAddress → account/conversation/message). Unlike
// slackSourceId (a citation id whose middle segment is the opaque message id), the threadKey
// MUST place the real channel id in the conversation segment and the ts in the message segment,
// so the channel adapter posts replies/drafts/approvals to the right channel and burst-coalescing
// groups per channel. Shape: `slack:<sourceId>:<channelId>:<ts>`.
export function slackThreadKey(message: Pick<SlackMessage, 'sourceId' | 'channelId' | 'ts'>): string {
  return `slack:${message.sourceId}:${message.channelId}:${message.ts}`
}

export function slackSourceRef(message: SlackMessage): ConnectorSourceRef {
  return {
    id: slackSourceId(message),
    connectorId: 'slack',
    sourceId: message.sourceId,
    itemId: message.id,
    title: `#${message.channelName} ${message.senderName || 'Unknown'}`,
    path: message.path,
    excerpt: message.snippet,
    createdAt: message.date,
  }
}

function publicSlackMessage(message: SlackReadResult): SlackMessage {
  const { raw: _raw, ...rest } = message
  return rest
}

function slackEventFromMessage(message: SlackMessage, now: number, reply?: ConnectorReplyContext): ConnectorEvent {
  const baseSummary = message.snippet || 'Slack message has no readable preview.'
  return {
    id: randomUUID(),
    connectorId: 'slack',
    pluginName: 'Slack',
    action: 'message_received',
    direction: 'inbound',
    classification: 'communication',
    priority: 'normal',
    trustLevel: 'notify',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: null,
    resolvedBy: null,
    title: `#${message.channelName}: ${message.senderName || 'Unknown'}`,
    // Prepend the quoted thread root so the agent's context shows what is being replied to.
    summary: reply ? injectReplyContext(baseSummary, reply) : baseSummary,
    actor: `#${message.channelName}`,
    threadKey: slackThreadKey(message),
    payloadKeys: ['message_id', 'channel', 'sender', 'sender_id', 'text', 'ts', ...(reply ? ['reply'] : []), ...(message.files?.length ? ['slack_files'] : [])],
    // sender + senderId ride in the payload so the host can learn a Slack id -> display-name
    // contact (contact-directory) for "reply to <name>" resolution, mirroring how Discord
    // carries its message id. The actor stays the channel name (not the sender) so the
    // feed reads "#channel", and enrichContacts never rewrites it.
    // slackFiles carries the message's url_private file refs so the host can download them with
    // the bot token (auth-aware media cache) and stamp mediaPaths during slack:sync.
    payload: { messageId: message.id, channel: message.channelId, sender: message.senderName, senderId: message.senderId, ts: message.ts, ...(message.files?.length ? { slackFiles: message.files } : {}) },
    ...(reply ? { reply } : {}),
    sources: [slackSourceRef(message)],
    createdAt: message.date ?? now,
  }
}

async function collectSlackMessageFiles(root: string): Promise<string[]> {
  const stat = await fs.stat(root)
  if (!stat.isDirectory()) throw new Error('Slack source is not an export folder.')
  const files: string[] = []
  const walk = async (dir: string, depth: number) => {
    if (depth > 3 || files.length >= MAX_SCAN_FILES) return
    const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (files.length >= MAX_SCAN_FILES) break
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        if (entry.name === '.git' || entry.name === 'node_modules') continue
        await walk(full, depth + 1)
      } else if (entry.isFile() && entry.name.endsWith('.json') && !isSlackMetadataFile(entry.name)) {
        files.push(full)
      }
    }
  }
  await walk(root, 0)
  return files
}

async function readSlackMetadata(root: string): Promise<SlackExportMetadata> {
  return {
    channels: await readSlackNameMap(path.join(root, 'channels.json'), 'name'),
    users: await readSlackNameMap(path.join(root, 'users.json'), 'real_name', 'name'),
  }
}

async function readSlackNameMap(file: string, ...fields: string[]): Promise<Map<string, string>> {
  const map = new Map<string, string>()
  try {
    const raw = JSON.parse(await fs.readFile(file, 'utf8'))
    if (!Array.isArray(raw)) return map
    for (const item of raw) {
      const id = typeof item?.id === 'string' ? item.id : ''
      if (!id) continue
      const name = fields.map((field) => item?.[field]).find((value) => typeof value === 'string' && value.trim())
      if (name) map.set(id, String(name).trim())
    }
  } catch {
    // Slack exports can omit metadata files; fall back to ids and folder names.
  }
  return map
}

async function readSlackFile(source: SlackSourceConfig, file: string, metadata: SlackExportMetadata): Promise<SlackReadResult[]> {
  const stat = await fs.stat(file)
  if (!stat.isFile() || stat.size > MAX_FILE_BYTES) return []
  const raw = JSON.parse(await fs.readFile(file, 'utf8'))
  if (!Array.isArray(raw)) return []
  const channelName = channelNameFromPath(source.path, file)
  const channelId = [...metadata.channels.entries()].find(([_id, name]) => normalizeChannel(name) === normalizeChannel(channelName))?.[0] ?? channelName
  const messages: SlackReadResult[] = []
  for (const item of raw) {
    const parsed = slackMessageFromRaw(item, source, file, channelId, metadata.channels.get(channelId) || channelName, metadata)
    if (parsed) messages.push(parsed)
  }
  return messages
}

function slackMessageFromRaw(raw: any, source: SlackSourceConfig, file: string, channelId: string, channelName: string, metadata: SlackExportMetadata): SlackReadResult | null {
  if (!raw || raw.type !== 'message') return null
  if (raw.subtype && raw.subtype !== 'bot_message' && raw.subtype !== 'thread_broadcast') return null
  const ts = typeof raw.ts === 'string' ? raw.ts : ''
  const text = slackText(String(raw.text ?? ''), metadata)
  if (!ts || !text.trim()) return null
  const senderId = String(raw.user || raw.bot_id || raw.username || '')
  const senderName = senderId ? metadata.users.get(senderId) || String(raw.username || senderId) : String(raw.username || 'Unknown')
  return {
    id: stableSlackId(source.id, channelId, ts),
    sourceId: source.id,
    sourceLabel: source.label,
    path: file,
    channelId,
    channelName,
    senderId,
    senderName,
    ts,
    threadTs: typeof raw.thread_ts === 'string' ? raw.thread_ts : null,
    date: slackTsToMs(ts),
    text,
    snippet: snippet(text),
    raw,
  }
}

function slackText(value: string, metadata: SlackExportMetadata): string {
  return decodeSlackEntities(value)
    .replace(/<@([A-Z0-9]+)>/g, (_m, id) => metadata.users.get(id) ? `@${metadata.users.get(id)}` : `@${id}`)
    .replace(/<#([A-Z0-9]+)\|([^>]+)>/g, (_m, _id, name) => `#${name}`)
    .replace(/<([^|>]+)\|([^>]+)>/g, '$2')
    .replace(/<([^>]+)>/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()
}

function decodeSlackEntities(value: string): string {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}

function channelNameFromPath(root: string, file: string): string {
  const rel = path.relative(root, file)
  const first = rel.split(path.sep)[0]
  return path.basename(first || path.dirname(file)).replace(/\.json$/i, '')
}

function isSlackMetadataFile(name: string): boolean {
  return ['channels.json', 'users.json', 'integration_logs.json', 'dms.json', 'groups.json', 'mpims.json'].includes(name)
}

function normalizeSlackId(id: string): string {
  return String(id || '').trim().split(':').pop() || ''
}

function normalizeChannel(value: string): string {
  return String(value || '').trim().replace(/^#/, '').toLowerCase()
}

function slackTsToMs(ts: string): number | null {
  const n = Number(ts)
  return Number.isFinite(n) ? Math.floor(n * 1000) : null
}

function stableSlackId(sourceId: string, channelId: string, ts: string): string {
  return createHash('sha256').update(`${sourceId}|${channelId}|${ts}`).digest('hex').slice(0, 16)
}

function snippet(value: string, max = 260): string {
  return value.replace(/\s+/g, ' ').trim().slice(0, max)
}
