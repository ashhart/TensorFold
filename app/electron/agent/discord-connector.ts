import { randomUUID } from 'node:crypto'
import type { ConnectorEvent, DiscordChannel, DiscordConnectorSettings, DiscordMessage, DiscordReadResult } from './types'
import { injectReplyContext, makeReplyContext } from './connector-sync'
import { fetchConnectorJson, type ConnectorHttpDeps } from './connector-http'

// Live Discord connector. Authenticates with a bot token (discord.com/developers),
// reads channel history + threads via the REST API, and sends messages/replies
// via REST. Real-time inbound would use the Gateway websocket; for the v1 connector
// we poll history on sync (matching the email/Slack model) — a Gateway listener can
// be layered on later. The token is a protected secret.

const DISCORD_API = 'https://discord.com/api/v10'

export function makeDiscordSource(label: string, now = Date.now()): DiscordConnectorSettings['sources'][number] {
  return {
    id: `discord_${randomUUID().slice(0, 8)}`,
    label: label || 'Discord server',
    token: '',
    allowedChannels: '',
    enabled: false,
    lastSyncAt: null,
    lastError: null,
    createdAt: now,
    updatedAt: now,
  }
}

export function validateDiscordSource(source: DiscordConnectorSettings['sources'][number]): string | null {
  if (!source.token?.trim()) return 'Discord bot token is required.'
  return null
}

// Verify the token + fetch the bot's guilds (servers).
export async function verifyDiscordToken(token: string, deps: ConnectorHttpDeps = {}): Promise<{ ok: boolean; bot?: string; guilds?: { id: string; name: string }[]; error?: string }> {
  const { response: res, json: user } = await discordJson(`${DISCORD_API}/users/@me`, { headers: authHeaders(token) }, deps)
  if (!res.ok) {
    return { ok: false, error: user?.message || `Discord rejected the token (${res.status}).` }
  }
  const { response: guildsRes, json: guildData } = await discordJson(`${DISCORD_API}/users/@me/guilds`, { headers: authHeaders(token) }, deps)
  const guilds = guildsRes.ok && Array.isArray(guildData) ? guildData.map((g: any) => ({ id: g.id, name: g.name })) : []
  return { ok: true, bot: user.username, guilds }
}

// List text channels + active threads the bot can see across its guilds.
export async function listDiscordChannels(token: string, deps: ConnectorHttpDeps = {}): Promise<DiscordChannel[]> {
  const out: DiscordChannel[] = []
  const { response: guildsRes, json: guildData } = await discordJson(`${DISCORD_API}/users/@me/guilds`, { headers: authHeaders(token) }, deps)
  if (!guildsRes.ok) throw new Error(`Discord could not list servers (${guildsRes.status}); check the bot token and permissions.`)
  const guilds = Array.isArray(guildData) ? guildData : []
  for (const guild of guilds) {
    const { response: chRes, json: channelData } = await discordJson(`${DISCORD_API}/guilds/${guild.id}/channels`, { headers: authHeaders(token) }, deps)
    if (!chRes.ok) throw new Error(`Discord could not list server channels (${chRes.status}); check View Channels permission.`)
    const channels = Array.isArray(channelData) ? channelData : []
    // Type 0 = text channel. Threads are fetched separately via active threads.
    for (const ch of channels) {
      if (ch.type === 0) out.push({ id: ch.id, name: ch.name, guildId: guild.id, guildName: guild.name, threadId: null })
    }
    // Active threads in the guild.
    const { response: threadsRes, json: data } = await discordJson(`${DISCORD_API}/guilds/${guild.id}/threads/active`, { headers: authHeaders(token) }, deps)
    if (threadsRes.ok) {
      for (const t of data.threads ?? []) {
        out.push({ id: t.id, name: t.name, guildId: guild.id, guildName: guild.name, threadId: t.id, parentId: t.parent_id })
      }
    }
  }
  return out
}

// Fetch recent messages across the allowed channels/threads of one bot token. When the
// source carries a prior cursor (a Discord snowflake id from the last sync), messages are
// paginated forward from that point with after=<id> so a burst of activity while offline
// (more than 50 messages) is retrieved in full rather than truncated to the newest 50 — the
// cursor can only dedupe what WAS fetched, it can't recover messages Discord never returned.
export async function fetchDiscordMessages(source: DiscordConnectorSettings['sources'][number], deps: ConnectorHttpDeps = {}): Promise<DiscordReadResult[]> {
  if (!source.enabled) throw new Error('Discord connection is disconnected.')
  const validation = validateDiscordSource(source)
  if (validation) throw new Error(validation)
  if (!source.allowedChannels?.trim()) return []
  const token = source.token!.trim()
  const channels = await listDiscordChannels(token, deps)
  const visible = filterChannels(channels, source.allowedChannels)
  if (source.allowedChannels.trim() !== '*' && !visible.length) throw new Error('None of the selected Discord channels are available; check the bot installation and channel permissions.')
  const out: DiscordReadResult[] = []
  for (const channel of visible) {
    const target = channel.threadId ?? channel.id
    const messages = await channelMessagesSince(token, target, source.cursor?.lastId ?? null, deps)
    for (const m of messages) {
      // Skip bot-authored messages (including our own posts) — otherwise sync re-ingests the bot's
      // replies as inbound events, polluting the feed and risking reply-to-self loops.
      if (m.author?.bot) continue
      out.push(toDiscordMessage(source, channel, m))
    }
  }
  return out.sort((a, b) => b.date - a.date)
}

// Read one message (+ its thread context if it's in a thread).
export async function readDiscordMessage(source: DiscordConnectorSettings['sources'][number], channelId: string, messageId: string, deps: ConnectorHttpDeps = {}): Promise<DiscordReadResult | null> {
  await ensureDiscordChannelAllowed(source, channelId, deps)
  const token = source.token!.trim()
  const { response: res, json: m } = await discordJson(`${DISCORD_API}/channels/${channelId}/messages/${messageId}`, { headers: authHeaders(token) }, deps)
  if (!res.ok) return null
  const channels = await listDiscordChannels(token, deps)
  const channel = channels.find((c) => c.id === channelId || c.threadId === channelId) ?? { id: channelId, name: channelId, guildId: '', guildName: '', threadId: null }
  return toDiscordMessage(source, channel, m)
}

// Send a message to a channel, or reply to a specific message (Discord reply).
// The allowedChannels allowlist scopes which channels this source may touch. It was enforced
// on reads (filterChannels) but NOT on sends, so a send/reply could target ANY channel.
// Synchronous fast-path: true with an explicit * or when the channel id is explicitly
// listed. Anything else (incl. a name-based allowlist) returns false and the caller must
// resolve names -> ids before allowing — a name-only allowlist must NOT be permissive, or the
// agent can harvest a restricted channel's id and post to it.
export function isDiscordChannelAllowed(allowed: string | undefined, channelId: string): boolean {
  const raw = allowed?.trim()
  if (raw === '*') return true
  if (!raw) return false
  const entries = raw.split(/[\s,;]+/).map((s) => s.replace(/^#/, '').trim().toLowerCase()).filter(Boolean)
  return entries.includes(channelId.trim().toLowerCase())
}

// Enforce the allowlist on the WRITE path, resolving channel NAMES to ids when needed.
async function ensureDiscordChannelAllowed(source: DiscordConnectorSettings['sources'][number], channelId: string, deps: ConnectorHttpDeps = {}): Promise<void> {
  if (!source.enabled) throw new Error('Discord connection is disconnected.')
  if (!source.allowedChannels?.trim()) throw new Error('Choose allowed Discord channels before reading or sending messages.')
  if (isDiscordChannelAllowed(source.allowedChannels, channelId)) return
  // Resolve the allowed set (handles name-based allowlists; filterChannels matches id OR name)
  // and require membership. Fail closed: if the channel list can't be fetched, reject.
  const channels = await listDiscordChannels(source.token!.trim(), deps)
  const allowedIds = new Set(filterChannels(channels, source.allowedChannels).map((c) => c.id.toLowerCase()))
  if (!allowedIds.has(channelId.trim().toLowerCase())) {
    throw new Error(`Discord channel ${channelId} is not in this source's allowed channels.`)
  }
}

export async function postDiscordMessage(source: DiscordConnectorSettings['sources'][number], channelId: string, text: string, replyTo?: string, components?: unknown[], deps: ConnectorHttpDeps = {}): Promise<{ ok: boolean; id?: string }> {
  await ensureDiscordChannelAllowed(source, channelId, deps)
  const token = source.token!.trim()
  const body: Record<string, unknown> = { content: text }
  if (replyTo) body.message_reference = { message_id: replyTo }
  if (components?.length) body.components = components
  const { response: res, json: data } = await discordJson(`${DISCORD_API}/channels/${channelId}/messages`, {
    method: 'POST',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, deps)
  if (!res.ok) throw new Error(data?.message || `Discord send failed (${res.status}).`)
  return { ok: true, id: data.id }
}

// Edit a message the bot previously sent (PATCH /channels/{channelId}/messages/{messageId}) — the
// Discord side of draft streaming. `messageId` is the id returned by postDiscordMessage. Re-checks
// the channel allowlist on the write path (an edit is still a write). Message edits share the
// channel's rate-limit bucket, so intermediate frames must be spaced (see DRAFT_EDIT_INTERVALS_MS);
// the caller swallows a failed in-flight edit and lets finalizeDraft commit the full text.
export async function editDiscordMessage(source: DiscordConnectorSettings['sources'][number], channelId: string, messageId: string, text: string, deps: ConnectorHttpDeps = {}): Promise<{ ok: boolean; id?: string }> {
  await ensureDiscordChannelAllowed(source, channelId, deps)
  const token = source.token!.trim()
  const { response: res, json: data } = await discordJson(`${DISCORD_API}/channels/${channelId}/messages/${messageId}`, {
    method: 'PATCH',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: text }),
  }, deps)
  if (!res.ok) throw new Error(data?.message || `Discord edit failed (${res.status}).`)
  return { ok: true, id: data?.id ?? messageId }
}

// ---- internals ----

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bot ${token.trim()}` }
}

function filterChannels(channels: DiscordChannel[], allowed: string | undefined): DiscordChannel[] {
  const raw = allowed?.trim()
  if (raw === '*') return channels
  if (!raw) return []
  const allow = new Set(raw.split(/[\s,;]+/).map((s) => s.replace(/^#/, '').trim().toLowerCase()).filter(Boolean))
  return channels.filter((c) => allow.has(c.id.toLowerCase()) || allow.has(c.name.toLowerCase()))
}

const MESSAGES_PAGE_SIZE = 50
// Cap on how many pages a single sync will follow per channel (10 * 50 = 500 messages),
// so a pathological backlog can't make one sync run unbounded — the cursor only advances to
// the last message actually retrieved, so a backlog beyond the cap is picked up by later syncs.
const MAX_PAGES_PER_SYNC = 10

// Fetch one page of messages, newest-first (descending), optionally after a given snowflake id.
async function channelMessages(token: string, channelId: string, limit: number, after: string | null | undefined, deps: ConnectorHttpDeps): Promise<any[]> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (after) params.set('after', after)
  const { response: res, json } = await discordJson(`${DISCORD_API}/channels/${channelId}/messages?${params.toString()}`, { headers: authHeaders(token) }, deps)
  if (!res.ok) throw new Error(`Discord could not read channel history (${res.status}); check View Channels, Read Message History and Message Content Intent.`)
  return Array.isArray(json) ? json : []
}

// Fetch all messages newer than `after` (a Discord snowflake id), paginating forward with
// after=<greatest seen id> until a page returns fewer than MESSAGES_PAGE_SIZE messages (exhausted)
// or MAX_PAGES_PER_SYNC is hit. With no `after`, this is a single bounded fetch of the newest
// MESSAGES_PAGE_SIZE messages — the existing full-resync behaviour is unchanged.
async function channelMessagesSince(token: string, channelId: string, after: string | null, deps: ConnectorHttpDeps): Promise<any[]> {
  if (!after) return channelMessages(token, channelId, MESSAGES_PAGE_SIZE, null, deps)
  if (!/^\d{1,20}$/.test(after)) throw new Error('Discord history cursor is invalid; reconnect this source before reading history.')
  const out: any[] = []
  let cursor = after
  for (let page = 0; page < MAX_PAGES_PER_SYNC; page++) {
    // Discord returns newest-to-oldest, including after= pages; never advance to the
    // last/oldest item or coerce a snowflake to Number, which loses integer precision.
    const batch = await channelMessages(token, channelId, MESSAGES_PAGE_SIZE, cursor, deps)
    if (!batch.length) break
    const fresh = batch.filter((message) => {
      if (typeof message?.id !== 'string' || !/^\d{1,20}$/.test(message.id)) throw new Error('Discord returned an invalid message ID; history was not advanced.')
      return BigInt(message.id) > BigInt(cursor)
    })
    if (!fresh.length) break
    out.push(...fresh)
    cursor = fresh.reduce((greatest, message) => BigInt(message.id) > BigInt(greatest) ? message.id : greatest, cursor)
    if (batch.length < MESSAGES_PAGE_SIZE) break // fewer than a full page => no more messages
  }
  return out
}

function discordJson(input: Parameters<typeof fetch>[0], init: Parameters<typeof fetch>[1], deps: ConnectorHttpDeps) {
  return fetchConnectorJson(DISCORD_API, 'Discord API', input, init, deps)
}

function toDiscordMessage(source: DiscordConnectorSettings['sources'][number], channel: DiscordChannel, m: any): DiscordReadResult {
  const text = typeof m.content === 'string' ? m.content : ''
  const author = m.author?.username || m.author?.global_name || 'unknown'
  const threadKey = channel.threadId ? `discord:${source.id}:${channel.threadId}` : `discord:${source.id}:${channel.id}`
  return {
    id: m.id,
    sourceId: source.id,
    sourceLabel: source.label,
    channelId: channel.id,
    channelName: channel.name,
    threadId: channel.threadId ?? null,
    threadName: channel.threadId ? channel.name : null,
    guildName: channel.guildName,
    senderId: m.author?.id || 'unknown',
    senderName: author,
    isBot: Boolean(m.author?.bot),
    date: m.timestamp ? new Date(m.timestamp).getTime() : Date.now(),
    text,
    snippet: text.replace(/\s+/g, ' ').trim().slice(0, 200),
    replyTo: m.referenced_message?.id ?? null,
    // Discord returns the replied-to message inline as referenced_message; capture its
    // author + content so the agent's context shows what is being replied to.
    replyToAuthor: m.referenced_message ? (m.referenced_message.author?.global_name || m.referenced_message.author?.username || null) : null,
    replyToText: typeof m.referenced_message?.content === 'string' ? m.referenced_message.content : null,
    threadKey,
    raw: m as Record<string, unknown>,
  }
}

// Pull the remote media URLs off a raw Discord message (attachments + image embeds) so
// the media cache can download them to local paths. Discord returns a stable `url` plus a
// CDN `proxy_url` per attachment; we prefer the canonical `url`. Image-bearing embeds
// (link unfurls) also carry a thumbnail/image url. De-duplicated, https-only is enforced
// downstream by the cache's SSRF gate.
export function extractDiscordMediaUrls(raw: Record<string, unknown> | null | undefined): string[] {
  const out: string[] = []
  const attachments = Array.isArray((raw as any)?.attachments) ? (raw as any).attachments : []
  for (const a of attachments) {
    const url = typeof a?.url === 'string' ? a.url : typeof a?.proxy_url === 'string' ? a.proxy_url : null
    if (url) out.push(url)
  }
  const embeds = Array.isArray((raw as any)?.embeds) ? (raw as any).embeds : []
  for (const e of embeds) {
    const url = typeof e?.image?.url === 'string' ? e.image.url : typeof e?.thumbnail?.url === 'string' ? e.thumbnail.url : null
    if (url) out.push(url)
  }
  return [...new Set(out)]
}

// Build a connector event for a Discord message (used by sync).
export function discordEventFromMessage(message: DiscordReadResult, now = Date.now()): ConnectorEvent {
  const baseSummary = message.text.slice(0, 220)
  // referenced_message gives the quoted author + content inline; fold it into the
  // summary so the agent sees what is being replied to.
  const reply = message.replyTo
    ? makeReplyContext({ id: message.replyTo, author: message.replyToAuthor ?? null, text: message.replyToText ?? null }) ?? undefined
    : undefined
  return {
    id: randomUUID(),
    connectorId: 'discord',
    pluginName: 'Discord',
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
    title: `${message.senderName} in #${message.channelName}`,
    summary: reply ? injectReplyContext(baseSummary, reply) : baseSummary,
    actor: message.senderName,
    threadKey: message.threadKey,
    payloadKeys: ['text', 'channel', ...(reply ? ['reply'] : [])],
    payload: { text: message.text, channel: message.channelId, thread: message.threadId },
    ...(reply ? { reply } : {}),
    createdAt: now,
  }
}
