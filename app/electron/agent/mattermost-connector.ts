import { randomUUID } from 'node:crypto'
import type {
  ConnectorEvent,
  MattermostChannel,
  MattermostReadResult,
  MattermostSource,
  SyncCursor,
} from './types'
import { advanceSyncCursor, injectReplyContext, isNewerThanCursor, makeReplyContext } from './connector-sync'
import { assertConnectorBaseUrl } from './security'
import { fetchWithRetry } from './connector-retry.ts'
import { createConnectorFetch } from './secure-fetch'
import { readBoundedConnectorJson } from './connector-http'

export type { MattermostChannel, MattermostConnectorSettings, MattermostMessage, MattermostReadResult, MattermostSource } from './types'

// Live Mattermost connector. Mattermost is a self-hosted Slack-alike with a REST API v4.
// One source = one server (base URL) + a Personal Access Token / bot token sent as
// `Authorization: Bearer <token>`. We poll channel history on sync (matching the
// email/Slack/Discord model) — a websocket listener (GET /api/v4/websocket) can be layered
// on later. The token is a protected secret; the base URL is per-source (not a fixed host),
// so every request is built off `source.baseUrl`.

const API_SUFFIX = '/api/v4'

export function makeMattermostSource(label: string, now = Date.now()): MattermostSource {
  return {
    id: `mattermost_${randomUUID().slice(0, 8)}`,
    label: label || 'Mattermost server',
    baseUrl: '',
    token: '',
    allowPrivateNetwork: false,
    allowedChannels: '',
    enabled: true,
    lastSyncAt: null,
    lastError: null,
    createdAt: now,
    updatedAt: now,
  }
}

export function validateMattermostSource(source: MattermostSource): string | null {
  if (!source.baseUrl?.trim()) return 'Mattermost server URL is required.'
  if (!/^https?:\/\//i.test(source.baseUrl.trim())) return 'Mattermost server URL must start with http(s)://.'
  try {
    assertConnectorBaseUrl(source.baseUrl, 'Mattermost server', { allowPrivateNetwork: source.allowPrivateNetwork })
  } catch (e: any) {
    return e?.message || 'Mattermost server URL cannot point at a local/private/link-local address.'
  }
  if (!source.token?.trim()) return 'Mattermost access token is required.'
  return null
}

// Verify the token + base URL by fetching the bot's own identity (GET /users/me).
export async function verifyMattermostConnection(
  baseUrl: string,
  token: string,
  allowPrivateNetwork = false,
): Promise<{ ok: boolean; user?: string; userId?: string; error?: string }> {
  const root = apiRoot(baseUrl)
  if (!root) return { ok: false, error: 'Mattermost server URL must start with http(s)://.' }
  const res = await mattermostFetch(baseUrl, allowPrivateNetwork, `${root}/users/me`, { headers: authHeaders(token) })
  if (!res.ok) {
    return { ok: false, error: await errorBody(res, 'Mattermost rejected the token') }
  }
  const me = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  if (!me?.id) return { ok: false, error: 'Mattermost returned no user identity for this token.' }
  return { ok: true, user: me.username || me.id, userId: me.id }
}

// List channels the bot can see across all its teams.
//   GET /api/v4/users/me/teams  ->  GET /api/v4/users/me/teams/{team_id}/channels
export async function listMattermostChannels(baseUrl: string, token: string, allowPrivateNetwork = false): Promise<MattermostChannel[]> {
  const root = apiRoot(baseUrl)
  const out: MattermostChannel[] = []
  if (!root) return out
  const teamsRes = await mattermostFetch(baseUrl, allowPrivateNetwork, `${root}/users/me/teams`, { headers: authHeaders(token) })
  if (!teamsRes.ok) return out
  const teams = await readBoundedConnectorJson(teamsRes, 'Mattermost API').catch(() => null)
  if (!Array.isArray(teams)) return out
  for (const team of teams) {
    const chRes = await mattermostFetch(baseUrl, allowPrivateNetwork, `${root}/users/me/teams/${team.id}/channels`, { headers: authHeaders(token) })
    if (!chRes.ok) continue
    const channels = await readBoundedConnectorJson(chRes, 'Mattermost API').catch(() => null)
    if (!Array.isArray(channels)) continue
    for (const ch of channels) {
      out.push({
        id: ch.id,
        name: ch.name || ch.id,
        displayName: ch.display_name || ch.name || ch.id,
        type: ch.type || 'O',
        teamId: team.id,
        teamName: team.display_name || team.name || team.id,
      })
    }
  }
  return out
}

// Fetch recent posts across the allowed channels of one source.
//   GET /api/v4/channels/{channel_id}/posts?per_page=50  ->  { order:[ids], posts:{id:post} }
export async function fetchMattermostMessages(source: MattermostSource): Promise<MattermostReadResult[]> {
  const validation = validateMattermostSource(source)
  if (validation) throw new Error(validation)
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  // Resolve the bot's own user id so we can skip its own posts (reply-to-self guard).
  const me = await verifyMattermostConnection(baseUrl, token, source.allowPrivateNetwork)
  if (!me.ok) throw new Error(me.error || 'Mattermost authentication failed.')
  const botUserId = me.userId ?? ''
  const channels = await listMattermostChannels(baseUrl, token, source.allowPrivateNetwork)
  const visible = filterChannels(channels, source.allowedChannels)
  const out: MattermostReadResult[] = []
  for (const channel of visible) {
    const { posts } = await channelPosts(baseUrl, token, channel.id, 50, source.allowPrivateNetwork)
    // Resolve user_id -> username for every author in the batch (senders AND reply-to roots) so
    // display names render instead of cryptic ids. One bulk POST /users/ids per channel.
    const userIds = new Set<string>()
    for (const p of posts) if (typeof p?.user_id === 'string' && p.user_id) userIds.add(p.user_id)
    const users = await fetchUsersByIds(baseUrl, token, [...userIds], source.allowPrivateNetwork)
    for (const p of posts) {
      // Skip system posts (joins/leaves/etc carry a non-empty `type`) and the bot's own
      // posts — otherwise sync re-ingests our own replies as inbound events, polluting the
      // feed and risking reply-to-self loops (mirrors discord's m.author.bot skip).
      if (p.type) continue
      if (botUserId && p.user_id === botUserId) continue
      out.push(toMattermostMessage(source, channel, p, botUserId, posts, users))
    }
  }
  return out.sort((a, b) => b.date - a.date)
}

// Read one post (+ resolve its channel + reply context).
export async function readMattermostMessage(
  source: MattermostSource,
  channelId: string,
  postId: string,
): Promise<MattermostReadResult | null> {
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const root = apiRoot(baseUrl)
  if (!root) return null
  const res = await mattermostFetch(baseUrl, source.allowPrivateNetwork, `${root}/posts/${postId}`, { headers: authHeaders(token) })
  if (!res.ok) return null
  const post = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  if (!post?.id) return null
  const channels = await listMattermostChannels(baseUrl, token, source.allowPrivateNetwork)
  const channel =
    channels.find((c) => c.id === channelId || c.id === post.channel_id) ??
    ({ id: post.channel_id || channelId, name: post.channel_id || channelId, displayName: post.channel_id || channelId, type: 'O', teamId: '', teamName: '' } as MattermostChannel)
  const users = typeof post.user_id === 'string' && post.user_id ? await fetchUsersByIds(baseUrl, token, [post.user_id], source.allowPrivateNetwork) : {}
  return toMattermostMessage(source, channel, post, '', [], users)
}

// The synchronous fast-path: allowed only when there's no allowlist or the id is
// explicitly listed. A name/slug-based allowlist returns false here so the caller resolves
// names->ids before allowing — it must NOT be permissive at send time, or the agent could
// harvest a restricted channel's id and post to it.
export function isMattermostChannelAllowed(allowed: string | undefined, channelId: string): boolean {
  const raw = allowed?.trim()
  if (!raw) return true
  const entries = parseAllowlist(raw)
  return entries.includes(channelId.trim().toLowerCase())
}

// Enforce the allowlist on the WRITE path, resolving channel names/slugs to ids when needed.
// Fail closed: if the channel list can't be fetched, reject.
async function ensureMattermostChannelAllowed(source: MattermostSource, channelId: string): Promise<void> {
  if (isMattermostChannelAllowed(source.allowedChannels, channelId)) return // no allowlist or id listed
  const channels = await listMattermostChannels(source.baseUrl.trim(), source.token.trim(), source.allowPrivateNetwork)
  const allowedIds = new Set(filterChannels(channels, source.allowedChannels).map((c) => c.id.toLowerCase()))
  if (!allowedIds.has(channelId.trim().toLowerCase())) {
    throw new Error(`Mattermost channel ${channelId} is not in this source's allowed channels.`)
  }
}

// Send a post to a channel, or reply in a thread (root_id is the THREAD ROOT post id).
// Mattermost requires root_id to be the root of the thread — a reply's own id is NOT a
// valid root, so when `replyTo` is itself a reply we resolve it to its root_id first.
export async function postMattermostMessage(
  source: MattermostSource,
  channelId: string,
  text: string,
  replyTo?: string,
): Promise<{ ok: boolean; id?: string }> {
  await ensureMattermostChannelAllowed(source, channelId)
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const root = apiRoot(baseUrl)
  if (!root) throw new Error('Mattermost server URL must start with http(s)://.')
  const body: Record<string, unknown> = { channel_id: channelId, message: text }
  if (replyTo) {
    const rootId = await resolveRootId(baseUrl, token, replyTo, source.allowPrivateNetwork)
    if (rootId) body.root_id = rootId
  }
  const res = await mattermostFetch(baseUrl, source.allowPrivateNetwork, `${root}/posts`, {
    method: 'POST',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  if (!res.ok) throw new Error(messageOf(data) || `Mattermost send failed (${res.status}).`)
  return { ok: true, id: data?.id }
}

// Edit a post the bot previously sent (PUT /posts/{post_id}/patch) — the Mattermost side of
// draft streaming. `postId` is the id returned by postMattermostMessage. Re-checks the
// channel allowlist on the write path (an edit is still a write).
export async function editMattermostMessage(
  source: MattermostSource,
  channelId: string,
  postId: string,
  text: string,
): Promise<{ ok: boolean; id?: string }> {
  await ensureMattermostChannelAllowed(source, channelId)
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const root = apiRoot(baseUrl)
  if (!root) throw new Error('Mattermost server URL must start with http(s)://.')
  const res = await mattermostFetch(baseUrl, source.allowPrivateNetwork, `${root}/posts/${postId}/patch`, {
    method: 'PUT',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text }),
  })
  const data = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  if (!res.ok) throw new Error(messageOf(data) || `Mattermost edit failed (${res.status}).`)
  return { ok: true, id: data?.id ?? postId }
}

// ---- internals ----

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token.trim()}` }
}

function mattermostFetch(baseUrl: string, allowPrivateNetwork: boolean | undefined, url: string, init: RequestInit = {}): Promise<Response> {
  return fetchWithRetry(url, init, {
    fetchImpl: createConnectorFetch(baseUrl, 'Mattermost server', { allowPrivateNetwork: allowPrivateNetwork === true }),
  })
}

// Build the `${origin}/api/v4` root from a per-source base URL. Returns '' for a
// non-http(s) URL so callers fail closed instead of issuing a request to a bad origin.
function apiRoot(baseUrl: string | undefined): string {
  const raw = (baseUrl ?? '').trim().replace(/\/+$/, '')
  if (!/^https?:\/\//i.test(raw)) return ''
  return `${raw}${API_SUFFIX}`
}

function parseAllowlist(raw: string): string[] {
  return raw
    .split(/[\s,;]+/)
    .map((s) => s.replace(/^[#~]/, '').trim().toLowerCase())
    .filter(Boolean)
}

function filterChannels(channels: MattermostChannel[], allowed: string | undefined): MattermostChannel[] {
  const raw = allowed?.trim()
  if (!raw) return channels
  const allow = new Set(parseAllowlist(raw))
  return channels.filter(
    (c) => allow.has(c.id.toLowerCase()) || allow.has(c.name.toLowerCase()) || allow.has(c.displayName.toLowerCase()),
  )
}

// Fetch a channel's recent posts. Returns the bundled users map too so we can resolve
// sender + reply-to author names without an extra round-trip per post.
async function channelPosts(
  baseUrl: string,
  token: string,
  channelId: string,
  perPage: number,
  allowPrivateNetwork: boolean | undefined,
): Promise<{ posts: any[]; byId: Record<string, any> }> {
  const root = apiRoot(baseUrl)
  if (!root) return { posts: [], byId: {} }
  const res = await mattermostFetch(baseUrl, allowPrivateNetwork, `${root}/channels/${channelId}/posts?per_page=${perPage}`, { headers: authHeaders(token) })
  if (!res.ok) return { posts: [], byId: {} }
  const data = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  const byId: Record<string, any> = data?.posts && typeof data.posts === 'object' ? data.posts : {}
  // `order` is newest-first; map it through the posts dict to keep ordering + completeness.
  const order: string[] = Array.isArray(data?.order) ? data.order : Object.keys(byId)
  const posts = order.map((id) => byId[id]).filter(Boolean)
  return { posts, byId }
}

// Resolve a batch of user_ids to a user_id -> username map (POST /api/v4/users/ids). Best-effort:
// returns {} on any failure (empty input, non-http base, error response) so name resolution
// degrades to the raw id rather than throwing the whole sync. The username is the canonical
// display handle; callers fall back to the id when a user isn't in the map.
async function fetchUsersByIds(baseUrl: string, token: string, ids: string[], allowPrivateNetwork: boolean | undefined): Promise<Record<string, string>> {
  const out: Record<string, string> = {}
  const root = apiRoot(baseUrl)
  if (!root || ids.length === 0) return out
  const res = await mattermostFetch(baseUrl, allowPrivateNetwork, `${root}/users/ids`, {
    method: 'POST',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(ids),
  })
  if (!res.ok) return out
  const data = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  if (!Array.isArray(data)) return out
  for (const u of data) {
    if (u && typeof u.id === 'string') {
      const name = u.nickname || u.username || [u.first_name, u.last_name].filter(Boolean).join(' ').trim() || u.id
      out[u.id] = String(name)
    }
  }
  return out
}

// Resolve a post id to its THREAD ROOT id. Mattermost rejects a reply's own id used as a
// root_id ("Invalid RootId parameter"); a reply carries the real root in its own root_id.
async function resolveRootId(baseUrl: string, token: string, postId: string, allowPrivateNetwork: boolean | undefined): Promise<string> {
  const root = apiRoot(baseUrl)
  if (!root || !postId) return postId
  const res = await mattermostFetch(baseUrl, allowPrivateNetwork, `${root}/posts/${postId}`, { headers: authHeaders(token) })
  if (!res.ok) return postId
  const data = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  return data?.root_id ? String(data.root_id) : postId
}

// `users` maps user_id -> username (from fetchUsersByIds). senderNameFor prefers an explicit
// override_username, then the resolved username, and only falls back to the raw user_id when no
// name is known — so inbound events and reply attributions show readable names, not ids.
function senderNameFor(post: any, users: Record<string, string>): { id: string; name: string; isBot: boolean } {
  const id = post?.user_id || 'unknown'
  const props = post?.props || {}
  const name =
    props.override_username ||
    (typeof post?.user_id === 'string' ? users[post.user_id] : undefined) ||
    (typeof post?.user_id === 'string' ? post.user_id : 'unknown')
  const isBot = Boolean(post?.props?.from_bot === 'true' || post?.props?.from_bot === true)
  return { id, name, isBot }
}

function toMattermostMessage(
  source: MattermostSource,
  channel: MattermostChannel,
  post: any,
  _botUserId: string,
  allPosts: any[],
  users: Record<string, string> = {},
): MattermostReadResult {
  const text = typeof post?.message === 'string' ? post.message : ''
  const sender = senderNameFor(post, users)
  const rootId = post?.root_id ? String(post.root_id) : null
  // Thread key: a threaded reply keys to its root post so the whole thread coalesces; a
  // top-level post keys to itself within the channel.
  const threadKey = rootId
    ? `mattermost:${source.id}:${channel.id}:${rootId}`
    : `mattermost:${source.id}:${channel.id}`
  // Resolve the replied-to (root) post's author + text from the same batch when present so
  // the agent's context shows what's being replied to (no extra round-trip).
  const parent = rootId ? allPosts.find((p) => p?.id === rootId) : null
  const replyToAuthor = parent ? senderNameFor(parent, users).name : null
  const replyToText = typeof parent?.message === 'string' ? parent.message : null
  return {
    id: post?.id,
    sourceId: source.id,
    sourceLabel: source.label,
    channelId: channel.id,
    channelName: channel.displayName || channel.name || channel.id,
    teamName: channel.teamName,
    senderId: sender.id,
    senderName: sender.name,
    isBot: sender.isBot,
    date: typeof post?.create_at === 'number' ? post.create_at : Date.now(),
    text,
    snippet: text.replace(/\s+/g, ' ').trim().slice(0, 200),
    rootId,
    replyTo: rootId,
    replyToAuthor,
    replyToText,
    threadKey,
    raw: post as Record<string, unknown>,
  }
}

// Pull remote media URLs off a raw Mattermost post. Mattermost file attachments are NOT
// public URLs — each file_id is downloaded from GET /api/v4/files/{id} WITH the bot token —
// so we surface authenticated download URLs built off the post's base server. The media
// cache's auth-aware path supplies the Bearer header; https-only/SSRF is enforced
// downstream. De-duplicated. Returns [] when the post has no file_ids.
export function extractMattermostMediaUrls(
  raw: Record<string, unknown> | null | undefined,
  baseUrl?: string,
): string[] {
  const root = apiRoot(baseUrl)
  if (!root) return []
  const fileIds = Array.isArray((raw as any)?.file_ids) ? (raw as any).file_ids : []
  const out: string[] = []
  for (const fid of fileIds) {
    if (typeof fid === 'string' && fid) out.push(`${root}/files/${fid}`)
  }
  return [...new Set(out)]
}

// Build a connector event for a Mattermost post (used by sync).
export function mattermostEventFromMessage(message: MattermostReadResult, now = Date.now()): ConnectorEvent {
  const baseSummary = message.text.slice(0, 220)
  const reply = message.replyTo
    ? makeReplyContext({ id: message.replyTo, author: message.replyToAuthor ?? null, text: message.replyToText ?? null }) ?? undefined
    : undefined
  return {
    id: randomUUID(),
    connectorId: 'mattermost',
    pluginName: 'Mattermost',
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
    title: `${message.senderName} in ${message.channelName}`,
    summary: reply ? injectReplyContext(baseSummary, reply) : baseSummary,
    actor: message.senderName,
    threadKey: message.threadKey,
    // messageId is the source post id — carry it so insert-time dedup (rememberConnectorEvents)
    // recognises an already-ingested post. It MUST come from the message it was built from, not
    // be re-derived later by matching threadKey+text (two distinct posts with identical text in a
    // channel would collide and the second would be deduped away).
    payloadKeys: ['text', 'channel', ...(message.id ? ['messageId'] : []), ...(reply ? ['reply'] : [])],
    payload: { text: message.text, channel: message.channelId, thread: message.rootId, ...(message.id ? { messageId: message.id } : {}) },
    ...(reply ? { reply } : {}),
    createdAt: now,
  }
}

// Build the inbound events for a sync batch, honouring the incremental cursor: only posts
// strictly newer than the source's high-water mark become events, then the cursor advances
// to the newest post in the batch. Pure given `messages`; the host persists the returned
// cursor back onto the source. Mirrors the email/slack/discord sync-cursor wiring.
export function buildMattermostSync(
  source: MattermostSource,
  messages: MattermostReadResult[],
  now = Date.now(),
): { events: ConnectorEvent[]; cursor: SyncCursor | null } {
  const fresh = messages.filter((m) => isNewerThanCursor({ position: m.date, id: m.id }, source.cursor))
  const events = fresh.map((m) => mattermostEventFromMessage(m, now))
  const cursor = advanceSyncCursor(
    source.cursor,
    messages.map((m) => ({ position: m.date, id: m.id })),
    now,
  )
  return { events, cursor }
}

function messageOf(data: any): string | null {
  if (data && typeof data === 'object' && typeof data.message === 'string' && data.message.trim()) return data.message
  return null
}

async function errorBody(res: Response, fallback: string): Promise<string> {
  const data = await readBoundedConnectorJson(res, 'Mattermost API').catch(() => null)
  return messageOf(data) || `${fallback} (${res.status}).`
}
