import { randomUUID } from 'node:crypto'
import type {
  ConnectorEvent,
  MatrixReadResult,
  MatrixRoom,
  MatrixSource,
  MatrixSyncResult,
} from './types'
import { advanceSyncCursor, injectReplyContext, isNewerThanCursor, makeReplyContext } from './connector-sync'
import { assertConnectorBaseUrl } from './security'
import { fetchWithRetry } from './connector-retry.ts'
import { createConnectorFetch } from './secure-fetch'
import { readBoundedConnectorJson } from './connector-http'

export type { MatrixConnectorSettings, MatrixMessage, MatrixReadResult, MatrixRoom, MatrixSource } from './types'

// Live Matrix connector. Authenticates against any homeserver (Synapse, Dendrite,
// matrix.org, …) with a long-lived ACCESS TOKEN sent as `Authorization: Bearer <token>`
// over the client-server API. Reads room timelines via a one-shot `/sync?timeout=0`
// and sends messages/threaded replies via `PUT …/send/m.room.message/{txnId}`. The
// allowlist here is ROOM IDs/aliases (allowedRooms), not channels. Real-time inbound
// would hold a long-poll `/sync` loop; for the v1 connector we poll on sync (matching
// the Discord/Slack model) — a streaming listener can be layered on later. The access
// token is a protected secret.

const CS_API = '/_matrix/client/v3'

export function makeMatrixSource(label: string, now = Date.now()): MatrixSource {
  return {
    id: `matrix_${randomUUID().slice(0, 8)}`,
    label: label || 'Matrix account',
    baseUrl: 'https://matrix.org',
    token: '',
    allowPrivateNetwork: false,
    allowedRooms: '',
    enabled: true,
    lastSyncAt: null,
    lastError: null,
    createdAt: now,
    updatedAt: now,
  }
}

export function validateMatrixSource(source: MatrixSource): string | null {
  if (!source.baseUrl?.trim()) return 'Matrix homeserver URL is required.'
  if (!/^https?:\/\//i.test(source.baseUrl.trim())) return 'Matrix homeserver URL must start with http(s)://.'
  try {
    assertConnectorBaseUrl(source.baseUrl, 'Matrix homeserver', { allowPrivateNetwork: source.allowPrivateNetwork })
  } catch (e: any) {
    return e?.message || 'Matrix homeserver URL cannot point at a local/private/link-local address.'
  }
  if (!source.token?.trim()) return 'Matrix access token is required.'
  return null
}

// Verify the access token + homeserver by resolving the account's own user_id.
export async function verifyMatrixConnection(baseUrl: string, token: string, allowPrivateNetwork = false): Promise<{ ok: boolean; userId?: string; error?: string }> {
  const res = await matrixFetch(baseUrl, allowPrivateNetwork, `${apiBase(baseUrl)}/account/whoami`, { headers: authHeaders(token) })
  if (!res.ok) {
    const body = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
    return { ok: false, error: errMessage(body) || `Matrix rejected the token (${res.status}).` }
  }
  const data = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
  const userId = typeof data?.user_id === 'string' ? data.user_id : undefined
  if (!userId) return { ok: false, error: 'Matrix whoami returned no user_id.' }
  return { ok: true, userId }
}

// List the rooms the account is joined to (listChannels analog). Display names + aliases
// are resolved best-effort per room; failures degrade to the room id.
export async function listMatrixChannels(source: MatrixSource): Promise<MatrixRoom[]> {
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const res = await matrixFetch(baseUrl, source.allowPrivateNetwork, `${apiBase(baseUrl)}/joined_rooms`, { headers: authHeaders(token) })
  if (!res.ok) return []
  const data = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
  const ids: string[] = Array.isArray(data?.joined_rooms) ? data.joined_rooms.filter((r: unknown) => typeof r === 'string') : []
  const out: MatrixRoom[] = []
  for (const roomId of ids) {
    out.push(await resolveRoom(baseUrl, token, roomId, source.allowPrivateNetwork))
  }
  return out
}

// Fetch recent messages across the allowed rooms of one account, via a one-shot sync.
export async function fetchMatrixMessages(source: MatrixSource): Promise<MatrixReadResult[]> {
  const validation = validateMatrixSource(source)
  if (validation) throw new Error(validation)
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const me = await verifyMatrixConnection(baseUrl, token, source.allowPrivateNetwork)
  if (!me.ok) throw new Error(me.error || 'Matrix authentication failed.')
  const selfId = me.userId ?? null

  const res = await matrixFetch(baseUrl, source.allowPrivateNetwork, `${apiBase(baseUrl)}/sync?timeout=0`, { headers: authHeaders(token) })
  if (!res.ok) {
    const body = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
    throw new Error(errMessage(body) || `Matrix sync failed (${res.status}).`)
  }
  const data = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
  const join = (data?.rooms?.join ?? {}) as Record<string, any>

  // Resolve the rooms once so each event gets a display name + alias, and so the
  // allowlist can match by alias as well as id.
  const allRooms = await listMatrixChannels(source)
  const roomById = new Map(allRooms.map((r) => [r.roomId, r]))
  const visibleIds = new Set(filterRooms(allRooms, source.allowedRooms).map((r) => r.roomId))

  // First pass: index every timeline event by id so a reply can resolve its quoted parent
  // within this batch (Matrix does not inline the replied-to event).
  const byEventId = new Map<string, { sender: string; body: string }>()
  for (const roomId of Object.keys(join)) {
    for (const ev of timelineEvents(join[roomId])) {
      if (typeof ev?.event_id === 'string') {
        byEventId.set(ev.event_id, { sender: ev.sender, body: textBody(ev) })
      }
    }
  }

  const out: MatrixReadResult[] = []
  for (const roomId of Object.keys(join)) {
    if (!visibleIds.has(roomId)) continue
    const room = roomById.get(roomId) ?? { roomId, alias: null, name: roomId }
    for (const ev of timelineEvents(join[roomId])) {
      if (!isTextMessage(ev)) continue
      // Skip our own events — otherwise sync re-ingests the account's own replies as
      // inbound events, polluting the feed and risking reply-to-self loops.
      if (selfId && ev.sender === selfId) continue
      out.push(await toMatrixMessage(source, room, ev, selfId, byEventId, baseUrl, token))
    }
  }
  return out.sort((a, b) => b.date - a.date)
}

// Read one event from a room (best-effort reply-parent resolution via a fresh GET).
export async function readMatrixMessage(source: MatrixSource, roomId: string, eventId: string): Promise<MatrixReadResult | null> {
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const res = await matrixFetch(baseUrl, source.allowPrivateNetwork, `${apiBase(baseUrl)}/rooms/${encodeURIComponent(roomId)}/event/${encodeURIComponent(eventId)}`, { headers: authHeaders(token) })
  if (!res.ok) return null
  const ev = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
  if (!ev || !isTextMessage(ev)) return null
  const me = await verifyMatrixConnection(baseUrl, token, source.allowPrivateNetwork)
  const room = await resolveRoom(baseUrl, token, roomId, source.allowPrivateNetwork)
  return toMatrixMessage(source, room, ev, me.ok ? me.userId ?? null : null, new Map(), baseUrl, token)
}

// Synchronous fast-path: allowed only when there's no allowlist or the room id/alias is
// explicitly listed. Mirrors Discord's gate — a name/alias-only allowlist returns true
// ONLY for a literal alias match; an unknown id falls through to ensureMatrixRoomAllowed,
// which resolves aliases→ids before allowing, so the gate never goes permissive.
export function isMatrixRoomAllowed(allowed: string | undefined, roomIdOrAlias: string): boolean {
  const raw = allowed?.trim()
  if (!raw) return true
  const entries = allowEntries(raw)
  return entries.includes(roomIdOrAlias.trim().toLowerCase())
}

// Enforce the allowlist on the WRITE path, resolving room ALIASES to ids when needed.
// Fail closed: if the joined-room list can't be fetched, reject.
async function ensureMatrixRoomAllowed(source: MatrixSource, roomId: string): Promise<void> {
  if (isMatrixRoomAllowed(source.allowedRooms, roomId)) return // no allowlist or id/alias listed
  const rooms = await listMatrixChannels(source)
  const allowedIds = new Set(filterRooms(rooms, source.allowedRooms).map((r) => r.roomId.toLowerCase()))
  if (!allowedIds.has(roomId.trim().toLowerCase())) {
    throw new Error(`Matrix room ${roomId} is not in this source's allowed rooms.`)
  }
}

// Send a message to a room, or post a threaded reply. `threadRootId` attaches the
// message to a thread (m.relates_to rel_type 'm.thread'); `replyTo` quotes a specific
// event (m.in_reply_to). A unique txnId makes the PUT idempotent.
export async function postMatrixMessage(
  source: MatrixSource,
  roomId: string,
  text: string,
  opts?: { threadRootId?: string | null; replyTo?: string | null },
): Promise<{ ok: boolean; id?: string }> {
  await ensureMatrixRoomAllowed(source, roomId)
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const content = buildTextContent(text, opts?.threadRootId ?? null, opts?.replyTo ?? null)
  const txnId = `vontra_${Date.now()}_${randomUUID().slice(0, 8)}`
  const res = await matrixFetch(baseUrl, source.allowPrivateNetwork, `${apiBase(baseUrl)}/rooms/${encodeURIComponent(roomId)}/send/m.room.message/${encodeURIComponent(txnId)}`, {
    method: 'PUT',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(content),
  })
  const data = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
  if (!res.ok) throw new Error(errMessage(data) || `Matrix send failed (${res.status}).`)
  return { ok: true, id: typeof data?.event_id === 'string' ? data.event_id : undefined }
}

// Edit a message the account previously sent (m.replace) — the Matrix side of draft
// streaming. `eventId` is the id returned by postMatrixMessage. Re-checks the room
// allowlist on the write path (an edit is still a write). A Matrix edit is a NEW
// m.room.message carrying `m.new_content` + a `m.relates_to` rel_type 'm.replace'
// pointing at the original event.
export async function editMatrixMessage(source: MatrixSource, roomId: string, eventId: string, text: string): Promise<{ ok: boolean; id?: string }> {
  await ensureMatrixRoomAllowed(source, roomId)
  const baseUrl = source.baseUrl.trim()
  const token = source.token.trim()
  const content = {
    msgtype: 'm.text',
    body: `* ${text}`,
    'm.new_content': { msgtype: 'm.text', body: text },
    'm.relates_to': { rel_type: 'm.replace', event_id: eventId },
  }
  const txnId = `vontra_edit_${Date.now()}_${randomUUID().slice(0, 8)}`
  const res = await matrixFetch(baseUrl, source.allowPrivateNetwork, `${apiBase(baseUrl)}/rooms/${encodeURIComponent(roomId)}/send/m.room.message/${encodeURIComponent(txnId)}`, {
    method: 'PUT',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(content),
  })
  const data = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
  if (!res.ok) throw new Error(errMessage(data) || `Matrix edit failed (${res.status}).`)
  return { ok: true, id: typeof data?.event_id === 'string' ? data.event_id : eventId }
}

// ---- internals ----

function apiBase(baseUrl: string): string {
  return `${baseUrl.trim().replace(/\/+$/, '')}${CS_API}`
}

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token.trim()}` }
}

function matrixFetch(baseUrl: string, allowPrivateNetwork: boolean | undefined, url: string, init: RequestInit = {}): Promise<Response> {
  return fetchWithRetry(url, init, {
    fetchImpl: createConnectorFetch(baseUrl, 'Matrix homeserver', { allowPrivateNetwork: allowPrivateNetwork === true }),
  })
}

// Matrix error bodies look like { errcode: 'M_FORBIDDEN', error: 'message' }.
function errMessage(body: any): string | null {
  if (!body || typeof body !== 'object') return null
  if (typeof body.error === 'string') return body.error
  if (typeof body.errcode === 'string') return body.errcode
  return null
}

function allowEntries(raw: string): string[] {
  return raw.split(/[\s,;]+/).map((s) => s.trim().toLowerCase()).filter(Boolean)
}

function filterRooms(rooms: MatrixRoom[], allowed: string | undefined): MatrixRoom[] {
  const raw = allowed?.trim()
  if (!raw) return rooms
  const allow = new Set(allowEntries(raw))
  return rooms.filter((r) => allow.has(r.roomId.toLowerCase()) || (r.alias != null && allow.has(r.alias.toLowerCase())))
}

// A room's timeline events live at rooms.join[roomId].timeline.events.
function timelineEvents(roomJoin: any): any[] {
  const events = roomJoin?.timeline?.events
  return Array.isArray(events) ? events : []
}

function isTextMessage(ev: any): boolean {
  if (ev?.type !== 'm.room.message') return false
  const msgtype = ev?.content?.msgtype
  return msgtype === 'm.text'
}

function textBody(ev: any): string {
  const body = ev?.content?.body
  return typeof body === 'string' ? body : ''
}

function relatesTo(ev: any): any {
  return ev?.content?.['m.relates_to'] ?? null
}

// Resolve a room's display name + canonical alias best-effort (state lookups may 404 for
// rooms the homeserver hides state on — that's fine, we degrade to the id).
async function resolveRoom(baseUrl: string, token: string, roomId: string, allowPrivateNetwork: boolean | undefined): Promise<MatrixRoom> {
  let name = roomId
  let alias: string | null = null
  const nameRes = await matrixFetch(baseUrl, allowPrivateNetwork, `${apiBase(baseUrl)}/rooms/${encodeURIComponent(roomId)}/state/m.room.name/`, { headers: authHeaders(token) }).catch(() => null)
  if (nameRes && nameRes.ok) {
    const data = await readBoundedConnectorJson(nameRes, 'Matrix API').catch(() => null)
    if (typeof data?.name === 'string' && data.name.trim()) name = data.name.trim()
  }
  const aliasRes = await matrixFetch(baseUrl, allowPrivateNetwork, `${apiBase(baseUrl)}/rooms/${encodeURIComponent(roomId)}/state/m.room.canonical_alias/`, { headers: authHeaders(token) }).catch(() => null)
  if (aliasRes && aliasRes.ok) {
    const data = await readBoundedConnectorJson(aliasRes, 'Matrix API').catch(() => null)
    if (typeof data?.alias === 'string' && data.alias.trim()) {
      const trimmed = data.alias.trim()
      alias = trimmed
      if (name === roomId) name = trimmed
    }
  }
  return { roomId, alias, name }
}

// Resolve a user's display name best-effort via the profile endpoint.
async function resolveDisplayName(baseUrl: string, token: string, userId: string, allowPrivateNetwork: boolean | undefined): Promise<string> {
  const res = await matrixFetch(baseUrl, allowPrivateNetwork, `${apiBase(baseUrl)}/profile/${encodeURIComponent(userId)}/displayname`, { headers: authHeaders(token) }).catch(() => null)
  if (res && res.ok) {
    const data = await readBoundedConnectorJson(res, 'Matrix API').catch(() => null)
    if (typeof data?.displayname === 'string' && data.displayname.trim()) return data.displayname.trim()
  }
  return userId
}

async function toMatrixMessage(
  source: MatrixSource,
  room: MatrixRoom,
  ev: any,
  selfId: string | null,
  byEventId: Map<string, { sender: string; body: string }>,
  baseUrl: string,
  token: string,
): Promise<MatrixReadResult> {
  const text = textBody(ev)
  const senderId = typeof ev?.sender === 'string' ? ev.sender : 'unknown'
  const senderName = await resolveDisplayName(baseUrl, token, senderId, source.allowPrivateNetwork)
  const rel = relatesTo(ev)
  const threadRootId = rel?.rel_type === 'm.thread' && typeof rel?.event_id === 'string' ? rel.event_id : null
  const replyTo = typeof rel?.['m.in_reply_to']?.event_id === 'string' ? rel['m.in_reply_to'].event_id : null
  // Thread root anchors the threadKey when present so a threaded conversation collapses
  // to one key; otherwise the room is the thread.
  const threadAnchor = threadRootId ?? room.roomId
  const threadKey = `matrix:${source.id}:${threadAnchor}`

  let replyToAuthor: string | null = null
  let replyToText: string | null = null
  if (replyTo) {
    const parent = byEventId.get(replyTo)
    if (parent) {
      replyToText = parent.body || null
      replyToAuthor = parent.sender ? await resolveDisplayName(baseUrl, token, parent.sender, source.allowPrivateNetwork) : null
    }
  }

  return {
    id: typeof ev?.event_id === 'string' ? ev.event_id : 'unknown',
    sourceId: source.id,
    sourceLabel: source.label,
    roomId: room.roomId,
    roomName: room.name,
    roomAlias: room.alias,
    senderId,
    senderName,
    isSelf: !!selfId && senderId === selfId,
    date: typeof ev?.origin_server_ts === 'number' ? ev.origin_server_ts : Date.now(),
    text,
    snippet: text.replace(/\s+/g, ' ').trim().slice(0, 200),
    threadRootId,
    replyTo,
    replyToAuthor,
    replyToText,
    threadKey,
    raw: ev as Record<string, unknown>,
  }
}

// Build outbound m.room.message text content, attaching thread/reply relations. A
// threaded reply uses rel_type 'm.thread' with is_falling_back so non-threaded clients
// still render it as a reply; a plain reply uses m.in_reply_to.
function buildTextContent(text: string, threadRootId: string | null, replyTo: string | null): Record<string, unknown> {
  const content: Record<string, unknown> = { msgtype: 'm.text', body: text }
  if (threadRootId) {
    content['m.relates_to'] = {
      rel_type: 'm.thread',
      event_id: threadRootId,
      is_falling_back: true,
      'm.in_reply_to': { event_id: replyTo || threadRootId },
    }
  } else if (replyTo) {
    content['m.relates_to'] = { 'm.in_reply_to': { event_id: replyTo } }
  }
  return content
}

// Pull remote media URLs off a raw Matrix message event. Matrix media events
// (m.image/m.file/m.video/m.audio) carry an `mxc://server/mediaId` URI in content.url
// (and content.info.thumbnail_url); those are NOT directly fetchable, so we convert each
// to the homeserver's authenticated media-download URL. `baseUrl` is the source's
// homeserver. De-duplicated. A plain text event yields nothing.
export function extractMatrixMediaUrls(raw: Record<string, unknown> | null | undefined, baseUrl: string): string[] {
  const content = (raw as any)?.content
  if (!content || typeof content !== 'object') return []
  const out: string[] = []
  const push = (mxc: unknown) => {
    if (typeof mxc !== 'string' || !mxc.startsWith('mxc://')) return
    const http = mxcToHttp(mxc, baseUrl)
    if (http) out.push(http)
  }
  push(content.url)
  push(content?.info?.thumbnail_url)
  return [...new Set(out)]
}

// Convert mxc://server/mediaId to the homeserver's media-download URL.
function mxcToHttp(mxc: string, baseUrl: string): string | null {
  const parts = mxc.slice('mxc://'.length)
  if (!parts.includes('/')) return null
  const [server, ...rest] = parts.split('/')
  const mediaId = rest.join('/')
  if (!server || !mediaId) return null
  return `${baseUrl.trim().replace(/\/+$/, '')}/_matrix/client/v1/media/download/${encodeURIComponent(server)}/${encodeURIComponent(mediaId)}`
}

// Build a connector event for a Matrix message (used by sync). The threadKey collapses a
// threaded conversation onto its root event; reply context (resolved within the batch) is
// folded into the summary so the assistant sees what's being replied to.
export function matrixEventFromMessage(message: MatrixReadResult, now = Date.now()): ConnectorEvent {
  const baseSummary = message.text.slice(0, 220)
  const reply = message.replyTo
    ? makeReplyContext({ id: message.replyTo, author: message.replyToAuthor ?? null, text: message.replyToText ?? null }) ?? undefined
    : undefined
  const where = message.roomAlias || message.roomName || message.roomId
  return {
    id: randomUUID(),
    connectorId: 'matrix',
    pluginName: 'Matrix',
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
    title: `${message.senderName} in ${where}`,
    summary: reply ? injectReplyContext(baseSummary, reply) : baseSummary,
    actor: message.senderName,
    threadKey: message.threadKey,
    payloadKeys: ['text', 'room', ...(message.threadRootId ? ['thread'] : []), ...(reply ? ['reply'] : [])],
    payload: { text: message.text, room: message.roomId, thread: message.threadRootId },
    ...(reply ? { reply } : {}),
    createdAt: now,
  }
}

// Sync convenience: normalize → drop self/duplicates via the cursor → emit events and an
// advanced cursor to persist. Mirrors the incremental-sync shape the other connectors use.
// The cursor position is the event origin_server_ts in ms.
export async function syncMatrixSource(source: MatrixSource, now = Date.now()): Promise<MatrixSyncResult> {
  const messages = await fetchMatrixMessages(source)
  const cursored = messages.map((m) => ({ position: m.date, id: m.id }))
  const fresh = messages.filter((m) => isNewerThanCursor({ position: m.date, id: m.id }, source.cursor))
  const events = fresh.map((m) => matrixEventFromMessage(m, now))
  const cursor = advanceSyncCursor(source.cursor, cursored, now) ?? source.cursor
  const updated: MatrixSource = { ...source, cursor: cursor ?? undefined, lastSyncAt: now, lastError: null }
  return { events, messages: fresh.length, sources: [updated], syncedAt: now }
}
