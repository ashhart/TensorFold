import type { SlackFileRef, SlackMessage, SlackReadResult, SlackSourceConfig } from './types'
import { fetchConnectorJson, type ConnectorHttpDeps } from './connector-http'

// Live Slack Web API client for `kind: 'api_token'` sources. Reads channel
// history + threads (conversations.history / conversations.replies) and sends
// replies (chat.postMessage), all behind the connector's approval gate. Each
// source is one Slack workspace token (xoxb-...). The token is a protected
// secret resolved in the main host before this module sees it.

const SLACK_API = 'https://slack.com/api'
const MAX_PER_CHANNEL = 50

export function isLiveSlackSource(source: SlackSourceConfig): boolean {
  return source.kind === 'api_token'
}

export function validateLiveSlackSource(source: SlackSourceConfig): string | null {
  if (!source.token?.trim()) return 'Slack bot token is required (xoxb-...).'
  if (!/xox[bpras]-/.test(source.token.trim())) return 'Token does not look like a Slack bot/user token.'
  return null
}

// An empty list grants no channel access; only a standalone * explicitly grants all.
function allowedChannelFilter(source: SlackSourceConfig): ((id: string, name: string) => boolean) | null {
  const raw = source.channels?.trim()
  if (raw === '*') return null
  if (!raw) return () => false
  const allowed = new Set(raw.split(/[\s,;]+/).map((s) => s.replace(/^#/, '').trim().toLowerCase()).filter(Boolean))
  return (id, name) => allowed.has(id.toLowerCase()) || allowed.has(name.toLowerCase())
}

// The synchronous fast-path: allowed only with explicit * or when the id is explicitly
// listed. A name-based allowlist returns false here so the caller resolves names -> ids before
// allowing — it must NOT be permissive at send time, or the agent could harvest a restricted
// channel's id (e.g. from a prompt-injected message) and post to it. Mirrors
// isDiscordChannelAllowed / isMattermostChannelAllowed.
export function isSlackChannelAllowed(allowed: string | undefined, channelId: string): boolean {
  const raw = allowed?.trim()
  if (raw === '*') return true
  if (!raw) return false
  const entries = raw.split(/[\s,;]+/).map((s) => s.replace(/^#/, '').trim().toLowerCase()).filter(Boolean)
  return entries.includes(channelId.trim().toLowerCase())
}

// Enforce the allowlist on the WRITE path, resolving channel NAMES to ids when needed. Fail
// closed: if the channel list can't be fetched, reject. Mirrors ensureDiscordChannelAllowed /
// ensureMattermostChannelAllowed — the allowlist was enforced on reads (allowedChannelFilter)
// but not on sends, so a prompt-injected reply could target any channel the bot token can reach.
async function ensureSlackChannelAllowed(source: SlackSourceConfig, channelId: string, deps: ConnectorHttpDeps = {}): Promise<void> {
  if (!source.enabled) throw new Error('Slack connection is disconnected.')
  if (!source.channels?.trim()) throw new Error('Choose allowed Slack channels before reading or sending messages.')
  if (isSlackChannelAllowed(source.channels, channelId)) return
  const filter = allowedChannelFilter(source)
  const channels = await listSlackChannels(source.token!.trim(), deps)
  const allowedIds = new Set((filter ? channels.filter((c) => filter(c.id, c.name)) : channels).map((c) => c.id.toLowerCase()))
  if (!allowedIds.has(channelId.trim().toLowerCase())) {
    throw new Error(`Slack channel ${channelId} is not in this source's allowed channels.`)
  }
}

// Verify the token works + the bot is installed. Returns the workspace name on success.
export async function verifySlackToken(token: string, deps: ConnectorHttpDeps = {}): Promise<{ ok: boolean; team?: string; bot?: string; error?: string }> {
  const { json: data } = await slackJson(`${SLACK_API}/auth.test`, { headers: { Authorization: `Bearer ${token}` } }, deps)
  if (!data?.ok) return { ok: false, error: data?.error || 'Slack rejected the token.' }
  return { ok: true, team: data.team, bot: data.user }
}

// List the channels the bot can see (it must be invited to private channels).
export async function listSlackChannels(token: string, deps: ConnectorHttpDeps = {}): Promise<{ id: string; name: string; isPrivate: boolean }[]> {
  const out: { id: string; name: string; isPrivate: boolean }[] = []
  let cursor: string | undefined
  do {
    const url = new URL(`${SLACK_API}/conversations.list`)
    url.searchParams.set('limit', '200')
    url.searchParams.set('types', 'public_channel,private_channel')
    if (cursor) url.searchParams.set('cursor', cursor)
    const { json: data } = await slackJson(url, { headers: { Authorization: `Bearer ${token}` } }, deps)
    if (!data?.ok) throw new Error(data?.error || 'conversations.list failed')
    for (const ch of data.channels ?? []) out.push({ id: ch.id, name: ch.name, isPrivate: ch.is_private })
    cursor = data.response_metadata?.next_cursor || undefined
  } while (cursor)
  return out
}

// Fetch recent messages across the allowed channels of one workspace. Each top-level
// message and each threaded reply becomes a SlackMessage; threadTs links replies to roots.
export async function fetchSlackMessages(source: SlackSourceConfig, deps: ConnectorHttpDeps = {}): Promise<SlackReadResult[]> {
  if (!source.enabled) throw new Error('Slack connection is disconnected.')
  const validation = validateLiveSlackSource(source)
  if (validation) throw new Error(validation)
  if (!source.channels?.trim()) return []
  const token = source.token!.trim()
  const filter = allowedChannelFilter(source)
  const channels = await listSlackChannels(token, deps)
  const visible = filter ? channels.filter((c) => filter(c.id, c.name)) : channels
  if (filter && !visible.length) throw new Error('None of the selected Slack channels are available; invite the bot and refresh channel permissions.')
  const out: SlackReadResult[] = []
  let failedChannels = 0
  // User cache for readable names (best-effort; failures fall back to IDs).
  const users = new Map<string, string>()

  for (const channel of visible) {
    // Isolate per-channel failures: a channel that stays rate-limited (or otherwise errors)
    // after fetchWithRetry's bounded backoff must NOT abort the whole workspace sync — the
    // remaining channels still get ingested this cycle. (fetchWithRetry already absorbs the
    // transient 429/5xx; this catch only fires on a persistent failure.)
    try {
      const recent = await channelHistory(token, channel.id, MAX_PER_CHANNEL, undefined, deps)
      for (const m of recent) {
        out.push(toSlackMessage(source, channel, m, users, token, deps))
        // Fetch thread replies for parent messages that have them.
        // Slack bot tokens cannot read public/private channel threads through replies.
        // The setup UI states this capability limit; supported user-token sources retain it.
        if (!token.startsWith('xoxb-') && m.thread_ts && m.thread_ts === m.ts && m.reply_count) {
          const replies = await threadReplies(token, channel.id, m.thread_ts, deps)
          for (const r of replies) out.push(toSlackMessage(source, channel, r, users, token, deps))
        }
      }
    } catch {
      failedChannels++
    }
  }
  if (failedChannels) throw new Error(`Slack could not read ${failedChannels} selected channel(s); check channel membership, history scopes and service rate limits, then retry.`)
  return out.sort((a, b) => Number(b.ts) - Number(a.ts))
}

// Read one thread (a root + its replies) by its threadTs.
export async function readSlackThread(source: SlackSourceConfig, channel: string, threadTs: string, deps: ConnectorHttpDeps = {}): Promise<SlackReadResult[]> {
  await ensureSlackChannelAllowed(source, channel, deps)
  const token = source.token!.trim()
  const users = new Map<string, string>()
  const root = await channelHistory(token, channel, 1, threadTs, deps)
  const chName = await resolveChannelName(token, channel, deps)
  const ch = { id: channel, name: chName, isPrivate: false }
  const out = root.map((m) => toSlackMessage(source, ch, m, users, token, deps))
  const replies = token.startsWith('xoxb-') ? [] : await threadReplies(token, channel, threadTs, deps)
  for (const r of replies) out.push(toSlackMessage(source, ch, r, users, token, deps))
  return out.sort((a, b) => Number(a.ts) - Number(b.ts))
}

// Send a message to a channel (top-level) or as a threaded reply (threadTs set).
// Only call after approval. `blocks` attaches Block Kit interactive controls (e.g. the
// in-channel approval buttons); `text` stays as the notification/fallback string.
export async function postSlackMessage(source: SlackSourceConfig, channel: string, text: string, threadTs?: string, blocks?: unknown[], deps: ConnectorHttpDeps = {}): Promise<{ ok: boolean; ts?: string; threadTs?: string }> {
  await ensureSlackChannelAllowed(source, channel, deps)
  const token = source.token!.trim()
  const body: Record<string, unknown> = { channel, text }
  if (threadTs) body.thread_ts = threadTs
  if (blocks?.length) body.blocks = blocks
  const { json: data } = await slackJson(`${SLACK_API}/chat.postMessage`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify(body),
  }, deps)
  if (!data?.ok) throw new Error(data?.error || 'chat.postMessage failed')
  return { ok: true, ts: data.ts, threadTs: data.thread_ts ?? threadTs }
}

// Edit a previously-sent message in place (chat.update) — the Slack side of draft streaming.
// `ts` is the message timestamp returned by postSlackMessage. Best-effort by caller: an edit to
// the identical text is a no-op Slack may reject, so the draft loop swallows failures and lets the
// final commit land the full text. chat.update shares the Tier-3 method rate limit, so intermediate
// edits must be spaced (see DRAFT_EDIT_INTERVALS_MS).
export async function editSlackMessage(source: SlackSourceConfig, channel: string, ts: string, text: string, deps: ConnectorHttpDeps = {}): Promise<{ ok: boolean; ts?: string }> {
  await ensureSlackChannelAllowed(source, channel, deps)
  const token = source.token!.trim()
  const { json: data } = await slackJson(`${SLACK_API}/chat.update`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify({ channel, ts, text }),
  }, deps)
  if (!data?.ok) throw new Error(data?.error || 'chat.update failed')
  return { ok: true, ts: data.ts ?? ts }
}

// ---- internals ----

async function channelHistory(token: string, channel: string, limit: number, latest: string | undefined, deps: ConnectorHttpDeps): Promise<any[]> {
  const url = new URL(`${SLACK_API}/conversations.history`)
  url.searchParams.set('channel', channel)
  url.searchParams.set('limit', String(limit))
  if (latest) url.searchParams.set('latest', latest)
  url.searchParams.set('inclusive', 'true')
  const { json: data } = await slackJson(url, { headers: { Authorization: `Bearer ${token}` } }, deps)
  if (!data?.ok) throw new Error(data?.error || 'conversations.history failed')
  return data.messages ?? []
}

async function threadReplies(token: string, channel: string, threadTs: string, deps: ConnectorHttpDeps): Promise<any[]> {
  const url = new URL(`${SLACK_API}/conversations.replies`)
  url.searchParams.set('channel', channel)
  url.searchParams.set('ts', threadTs)
  const { json: data } = await slackJson(url, { headers: { Authorization: `Bearer ${token}` } }, deps)
  if (!data?.ok) return []
  // conversations.replies returns the root + all replies; drop the root.
  return (data.messages ?? []).filter((m: any) => m.ts !== threadTs)
}

async function resolveChannelName(token: string, channel: string, deps: ConnectorHttpDeps): Promise<string> {
  const url = new URL(`${SLACK_API}/conversations.info`)
  url.searchParams.set('channel', channel)
  const { json: data } = await slackJson(url, { headers: { Authorization: `Bearer ${token}` } }, deps)
  return data?.channel?.name || channel
}

async function resolveUserName(token: string, userId: string, cache: Map<string, string>, deps: ConnectorHttpDeps): Promise<string> {
  if (cache.has(userId)) return cache.get(userId)!
  try {
    const url = new URL(`${SLACK_API}/users.info`)
    url.searchParams.set('user', userId)
    const { json: data } = await slackJson(url, { headers: { Authorization: `Bearer ${token}` } }, deps)
    const name = data?.user?.real_name || data?.user?.name || userId
    cache.set(userId, name)
    return name
  } catch {
    return userId
  }
}

// Pull file attachment refs off a raw Slack message. The live API attaches a `files` array where
// each file exposes `url_private` (an AUTHENTICATED URL needing the bot token as a Bearer header)
// plus name + mimetype. We carry url_private (NOT the thumbnails) so the host can download the
// original via the auth-aware media cache. Files that are deleted/hidden (no url_private) or
// tombstoned (`mode: 'tombstone'`) are skipped. De-duplicated by url_private. Pure.
export function extractSlackFileRefs(raw: Record<string, unknown> | null | undefined): SlackFileRef[] {
  const files = Array.isArray((raw as any)?.files) ? (raw as any).files : []
  const out: SlackFileRef[] = []
  const seen = new Set<string>()
  for (const f of files) {
    if (f?.mode === 'tombstone' || f?.mode === 'hidden_by_limit') continue
    const urlPrivate = typeof f?.url_private_download === 'string' ? f.url_private_download : typeof f?.url_private === 'string' ? f.url_private : null
    if (!urlPrivate || seen.has(urlPrivate)) continue
    seen.add(urlPrivate)
    out.push({
      urlPrivate,
      name: typeof f?.name === 'string' ? f.name : (typeof f?.title === 'string' ? f.title : null),
      mimetype: typeof f?.mimetype === 'string' ? f.mimetype : null,
    })
  }
  return out
}

function toSlackMessage(source: SlackSourceConfig, channel: { id: string; name: string }, m: any, users: Map<string, string>, token: string, deps: ConnectorHttpDeps): SlackReadResult {
  const text = typeof m.text === 'string' ? m.text : String(m.text ?? '')
  const senderId = m.user || m.bot_id || 'unknown'
  // Best-effort synchronous name; we resolve from cache, falling back to the id.
  const senderName = (m.user && users.get(m.user)) || senderId
  if (m.user && !users.has(m.user)) void resolveUserName(token, m.user, users, deps).then((name) => users.set(m.user, name))
  const ts = String(m.ts ?? '')
  const files = extractSlackFileRefs(m)
  return {
    id: `${source.id}:${channel.id}:${ts}`,
    sourceId: source.id,
    sourceLabel: source.label,
    path: `slack:///${channel.id}/${ts}`,
    channelId: channel.id,
    channelName: channel.name,
    senderId,
    senderName,
    ts,
    threadTs: m.thread_ts && m.thread_ts !== m.ts ? m.thread_ts : (m.thread_ts && m.thread_ts === m.ts ? m.thread_ts : null),
    date: ts ? new Date(Number(ts.split('.')[0]) * 1000).getTime() : null,
    text,
    snippet: text.replace(/\s+/g, ' ').trim().slice(0, 200),
    ...(files.length ? { files } : {}),
    // Synthesized raw envelope for SlackReadResult consumers (the live API doesn't
    // return export JSON, but readers expect a raw payload to exist). The raw `files` array is
    // preserved so a downstream consumer can re-extract url_private if needed.
    raw: { channel: channel.id, ts, user: senderId, text, thread_ts: m.thread_ts ?? null, type: m.type ?? 'message', ...(Array.isArray(m.files) ? { files: m.files } : {}) } as Record<string, unknown>,
  }
}

function slackJson(input: Parameters<typeof fetch>[0], init: Parameters<typeof fetch>[1], deps: ConnectorHttpDeps) {
  return fetchConnectorJson(SLACK_API, 'Slack API', input, init, deps)
}
