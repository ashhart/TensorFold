import { randomUUID } from 'node:crypto'
import type {
  ConnectorEvent,
  ConnectorReplyContext,
  GoogleChatListOptions,
  GoogleChatReadResult,
  GoogleChatSource,
  GoogleChatSpace,
  GoogleChatSyncResult,
} from './types'
import { advanceSyncCursor, injectReplyContext, isNewerThanCursor, makeReplyContext } from './connector-sync'
import { fetchWithRetry } from './connector-retry.ts'
import { createConnectorFetch } from './secure-fetch'
import { fetchConnectorResponse, readBoundedConnectorJson } from './connector-http'

export type { GoogleChatConnectorSettings, GoogleChatMessage, GoogleChatReadResult, GoogleChatSource, GoogleChatSpace } from './types'

// Live Google Chat connector (chat.googleapis.com REST). v1 ships OPTION A — an
// incoming-webhook URL per space — which is the simplest viable model: send-first.
//
//   • SEND  : POST the webhook URL with { text } (+ { thread: { threadKey } } and a
//             ?messageReplyOption= param for in-thread replies). Works with only the
//             webhook URL the user pastes from a space's "Apps & integrations".
//   • VERIFY: validate the URL shape (host + /v1/spaces/.../messages + key & token query
//             params). A live probe POST is intentionally deferred so verify never posts a
//             stray "test" message into the user's space.
//   • READ  : NOT possible over a webhook. fetch/read return [] / null and stamp lastError
//             explaining that a service-account / OAuth2 bearer token is required for inbound.
//
// OPTION B (full inbound, drop-in later): set source.accessToken to a service-account
// OAuth2 bearer token (token MINTING is out of scope for v1 — the user supplies a short-
// lived access token, or a future seam mints one). Then listSpaces() GETs /v1/spaces and
// fetchGoogleChatMessages() GETs /v1/{space}/messages. Everything below the GOOGLE_CHAT_API
// constant is structured so Option B slots in without touching the Option A path: the read
// functions already branch on `hasAccessToken(source)` and fall back to the webhook-only
// notice. Search "OPTION B" for the seam.
//
// A "channel" here is a Google Chat SPACE, named `spaces/XXXXXXX`. The allowlist
// (allowedChannels, csv) is matched against space names (and bare space ids).

const GOOGLE_CHAT_API = 'https://chat.googleapis.com'
// A read needs Option B (a bearer token); webhook URLs can only send. Surfaced as lastError.
const READ_REQUIRES_TOKEN =
  'Reading Google Chat messages requires a service-account / OAuth2 access token (set accessToken on the source). An incoming-webhook URL can only send.'

export function makeGoogleChatSource(label: string, now = Date.now()): GoogleChatSource {
  return {
    id: `googlechat_${randomUUID().slice(0, 8)}`,
    label: label || 'Google Chat space',
    webhookUrl: '',
    accessToken: '',
    baseUrl: '',
    allowedChannels: '',
    enabled: true,
    lastSyncAt: null,
    lastError: null,
    createdAt: now,
    updatedAt: now,
  }
}

// A source is usable as soon as it can SEND: either an incoming-webhook URL (Option A) or a
// bearer access token (Option B). Validate the webhook URL shape when it's the only credential.
export function validateGoogleChatSource(source: GoogleChatSource): string | null {
  const hasWebhook = !!source.webhookUrl?.trim()
  const hasToken = !!source.accessToken?.trim()
  if (!hasWebhook && !hasToken) {
    return 'A Google Chat incoming-webhook URL (or a service-account access token) is required.'
  }
  if (hasWebhook) {
    const bad = webhookUrlError(source.webhookUrl!.trim())
    if (bad) return bad
  }
  return null
}

// Verify the source WITHOUT posting: validate the webhook URL shape (Option A), or confirm a
// token is present (Option B). A live probe POST is deferred so verify never injects a stray
// message into the user's space. Returns the resolved space id parsed from the webhook path.
export async function verifyGoogleChatConnection(
  source: GoogleChatSource,
): Promise<{ ok: boolean; mode: 'webhook' | 'token'; space?: string | null; error?: string }> {
  const validation = validateGoogleChatSource(source)
  if (validation) return { ok: false, mode: 'webhook', error: validation }
  if (source.webhookUrl?.trim()) {
    const space = spaceFromWebhookUrl(source.webhookUrl.trim())
    return { ok: true, mode: 'webhook', space }
  }
  // OPTION B: a token is present but no webhook — a real probe would GET /v1/spaces here.
  return { ok: true, mode: 'token', space: null }
}

// List spaces the source can see. OPTION A (webhook-only): the single space encoded in the
// webhook URL is the only thing we can name, so return just that. OPTION B (token): GET
// /v1/spaces. Either way the result is filtered by the allowlist.
export async function listGoogleChatChannels(source: GoogleChatSource): Promise<GoogleChatSpace[]> {
  let spaces: GoogleChatSpace[] = []
  if (hasAccessToken(source)) {
    // OPTION B: enumerate spaces via the REST API.
    const res = await googleChatApiFetch(source, `${apiBase(source)}/v1/spaces`, { headers: bearerHeaders(source.accessToken!) })
    if (res.ok) {
      const data = await readBoundedConnectorJson(res, 'Google Chat API').catch(() => null)
      for (const s of (data?.spaces ?? []) as any[]) {
        if (typeof s?.name === 'string') {
          spaces.push({ name: s.name, displayName: s.displayName || s.name, spaceType: s.spaceType || s.type || null })
        }
      }
    }
  }
  if (spaces.length === 0 && source.webhookUrl?.trim()) {
    // OPTION A: the only nameable space is the one baked into the webhook URL.
    const name = spaceFromWebhookUrl(source.webhookUrl.trim())
    if (name) spaces.push({ name, displayName: name, spaceType: null })
  }
  return filterSpaces(spaces, source.allowedChannels)
}

// Fetch recent messages across the allowed spaces. OPTION A: returns [] (webhooks can't read)
// — the caller stamps READ_REQUIRES_TOKEN onto lastError. OPTION B: GET /v1/{space}/messages
// per allowed space, skipping bot-authored messages (our own outbound included).
export async function fetchGoogleChatMessages(source: GoogleChatSource, opts: GoogleChatListOptions = {}): Promise<GoogleChatReadResult[]> {
  const validation = validateGoogleChatSource(source)
  if (validation) throw new Error(validation)
  if (!hasAccessToken(source)) {
    // OPTION A is send-only. Signal "no inbound" rather than throwing so a webhook-only
    // source still syncs cleanly (zero messages) with an explanatory lastError upstream.
    throw new Error(READ_REQUIRES_TOKEN)
  }
  // OPTION B below.
  const spaces = await listGoogleChatChannels(source)
  const wanted = opts.space ? spaces.filter((s) => sameSpace(s.name, opts.space!)) : spaces
  const limit = Math.max(1, Math.min(opts.limit ?? 50, 200))
  const out: GoogleChatReadResult[] = []
  for (const space of wanted) {
    const url = `${apiBase(source)}/v1/${space.name}/messages?pageSize=${limit}`
    const res = await googleChatApiFetch(source, url, { headers: bearerHeaders(source.accessToken!) })
    if (!res.ok) continue
    const data = await readBoundedConnectorJson(res, 'Google Chat API').catch(() => null)
    for (const m of (data?.messages ?? []) as any[]) {
      // Skip app/bot-authored messages (including our own posts) — otherwise sync re-ingests
      // the bot's replies as inbound events, risking reply-to-self loops (mirror discord).
      if ((m?.sender?.type || '').toUpperCase() === 'BOT') continue
      out.push(toGoogleChatMessage(source, space, m))
    }
  }
  return out.sort((a, b) => b.date - a.date)
}

// Read one message by its resource name. OPTION B only (needs a bearer token).
export async function readGoogleChatMessage(source: GoogleChatSource, messageName: string): Promise<GoogleChatReadResult | null> {
  if (!hasAccessToken(source)) return null
  const name = messageName.trim()
  const res = await googleChatApiFetch(source, `${apiBase(source)}/v1/${name}`, { headers: bearerHeaders(source.accessToken!) })
  if (!res.ok) return null
  const m = await readBoundedConnectorJson(res, 'Google Chat API').catch(() => null)
  if (!m) return null
  const spaceName = spaceFromMessageName(name) ?? (m.space?.name as string) ?? ''
  const space: GoogleChatSpace = { name: spaceName, displayName: m.space?.displayName || spaceName, spaceType: m.space?.type || null }
  return toGoogleChatMessage(source, space, m)
}

// Synchronous allowlist fast-path for the WRITE/READ guard. Allowed when there's no
// allowlist or the space is explicitly listed. Matches on the full `spaces/ID` name OR the
// bare id (a leading "spaces/" and case are normalized). Like discord, this is the cheap
// gate; the caller (ensureGoogleChatSpaceAllowed) does the fail-closed check around it.
export function isGoogleChatChannelAllowed(allowed: string | undefined, spaceName: string): boolean {
  const raw = allowed?.trim()
  if (!raw) return true
  const target = normalizeSpace(spaceName)
  return allowedSpaceSet(raw).has(target)
}

// Enforce the allowlist on BOTH the read and the write path. Fail closed: a non-empty
// allowlist that doesn't contain this space rejects (no remote name resolution is needed —
// Google Chat spaces are addressed by their stable resource name, so the synchronous check
// is authoritative, unlike discord's id-vs-name ambiguity).
async function ensureGoogleChatSpaceAllowed(source: GoogleChatSource, spaceName: string): Promise<void> {
  if (isGoogleChatChannelAllowed(source.allowedChannels, spaceName)) return
  throw new Error(`Google Chat space ${spaceName} is not in this source's allowed channels.`)
}

// Post a message to a space, or reply within a thread. OPTION A: POST the incoming-webhook
// URL (threading via { thread: { threadKey } } + ?messageReplyOption=). OPTION B (token):
// POST /v1/{space}/messages with a bearer token (threading via { thread: { name } }). The
// allowlist is enforced before any send.
export async function postGoogleChatMessage(
  source: GoogleChatSource,
  spaceName: string,
  text: string,
  replyTo?: { threadKey?: string | null; threadName?: string | null } | string | null,
): Promise<{ ok: boolean; id?: string; threadName?: string | null }> {
  await ensureGoogleChatSpaceAllowed(source, spaceName)
  const reply = typeof replyTo === 'string' ? { threadKey: replyTo } : replyTo ?? undefined

  if (hasAccessToken(source)) {
    // OPTION B: REST send with a bearer token. Thread by resource name.
    const body: Record<string, unknown> = { text }
    const params = new URLSearchParams()
    if (reply?.threadName) {
      body.thread = { name: reply.threadName }
      params.set('messageReplyOption', 'REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD')
    } else if (reply?.threadKey) {
      body.thread = { threadKey: reply.threadKey }
      params.set('messageReplyOption', 'REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD')
    }
    const qs = params.toString()
    const url = `${apiBase(source)}/v1/${normalizeSpace(spaceName)}/messages${qs ? `?${qs}` : ''}`
    const res = await googleChatApiFetch(source, url, {
      method: 'POST',
      headers: { ...bearerHeaders(source.accessToken!), 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await readBoundedConnectorJson(res, 'Google Chat API').catch(() => null)
    if (!res.ok) throw new Error(errorMessage(data) || `Google Chat send failed (${res.status}).`)
    return { ok: true, id: data?.name, threadName: data?.thread?.name ?? null }
  }

  // OPTION A: incoming-webhook send. The webhook URL already targets a fixed space, so the
  // space must match the one the webhook posts to (the allowlist already approved it).
  const webhook = source.webhookUrl?.trim()
  if (!webhook) throw new Error('Google Chat source has no webhook URL or access token to send with.')
  const body: Record<string, unknown> = { text }
  let url = webhook
  if (reply?.threadKey) {
    // threadKey is an arbitrary caller-chosen key that groups a thread; FALLBACK_TO_NEW_THREAD
    // starts a new thread if it can't route to an existing one.
    body.thread = { threadKey: reply.threadKey }
    url = appendQueryParam(webhook, 'messageReplyOption', 'REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD')
  }
  const res = await fetchWithRetry(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    redirect: 'error',
  }, {
    fetchImpl: createConnectorFetch(webhook, 'Google Chat webhook'),
  })
  const data = await readBoundedConnectorJson(res, 'Google Chat API').catch(() => null)
  if (!res.ok) throw new Error(errorMessage(data) || `Google Chat webhook send failed (${res.status}).`)
  return { ok: true, id: data?.name, threadName: data?.thread?.name ?? null }
}

// Build the inbound connector events for a sync over one source: fetch newer-than-cursor
// messages and map them, advancing the cursor. OPTION A sources yield zero events with an
// explanatory lastError; the work doesn't throw so a mixed batch still syncs.
export async function syncGoogleChatSource(source: GoogleChatSource, now = Date.now()): Promise<GoogleChatSyncResult> {
  const base: GoogleChatSyncResult = { events: [], messages: 0, sources: [source], syncedAt: now, cursor: source.cursor }
  if (!source.enabled) return base
  let messages: GoogleChatReadResult[]
  try {
    messages = await fetchGoogleChatMessages(source)
  } catch (err) {
    const lastError = err instanceof Error ? err.message : String(err)
    return { ...base, sources: [{ ...source, lastError, lastSyncAt: now }] }
  }
  const fresh = messages.filter((m) => isNewerThanCursor({ position: m.date, id: m.id }, source.cursor))
  const events = fresh.map((m) => googlechatEventFromMessage(m, now))
  const cursor = advanceSyncCursor(source.cursor, fresh.map((m) => ({ position: m.date, id: m.id })), now) ?? undefined
  return {
    events,
    messages: fresh.length,
    sources: [{ ...source, cursor, lastError: null, lastSyncAt: now }],
    syncedAt: now,
    cursor,
  }
}

// Pull remote media URLs (attachment download URIs) off a raw Google Chat message so the
// media cache can fetch them. Google Chat returns attachments as `attachment[]`, each with a
// `downloadUri`/`thumbnailUri` (drive files carry `driveDataRef` instead and aren't a plain
// URL — skipped here). De-duplicated; https-only is enforced downstream by the cache SSRF gate.
export function extractGoogleChatMediaUrls(raw: Record<string, unknown> | null | undefined): string[] {
  const out: string[] = []
  const attachments = Array.isArray((raw as any)?.attachment) ? (raw as any).attachment : []
  for (const a of attachments) {
    const url = typeof a?.downloadUri === 'string' ? a.downloadUri : typeof a?.thumbnailUri === 'string' ? a.thumbnailUri : null
    if (url) out.push(url)
  }
  return [...new Set(out)]
}

// Build a connector event for a Google Chat message (used by sync).
export function googlechatEventFromMessage(message: GoogleChatReadResult, now = Date.now()): ConnectorEvent {
  const baseSummary = message.text.slice(0, 220)
  const reply: ConnectorReplyContext | undefined = message.replyTo
    ? makeReplyContext({ id: message.replyTo, author: message.replyToAuthor ?? null, text: message.replyToText ?? null }) ?? undefined
    : undefined
  const where = message.spaceDisplayName || message.spaceName
  return {
    id: randomUUID(),
    connectorId: 'googlechat',
    pluginName: 'Google Chat',
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
    payloadKeys: ['text', 'space', ...(message.threadName ? ['thread'] : []), ...(reply ? ['reply'] : [])],
    payload: { text: message.text, space: message.spaceName, thread: message.threadName },
    ...(reply ? { reply } : {}),
    createdAt: now,
  }
}

// ---- internals ----

function hasAccessToken(source: GoogleChatSource): boolean {
  return !!source.accessToken?.trim()
}

function apiBase(source: GoogleChatSource): string {
  const raw = source.baseUrl?.trim()
  return (raw || GOOGLE_CHAT_API).replace(/\/+$/, '')
}

function googleChatApiFetch(source: GoogleChatSource, url: string, init: RequestInit = {}): Promise<Response> {
  const base = apiBase(source)
  return fetchConnectorResponse(base, 'Google Chat server', url, init)
}

function bearerHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token.trim()}` }
}

// Validate an incoming-webhook URL: it must be an https chat.googleapis.com (or the source's
// baseUrl host) `/v1/spaces/<id>/messages` URL carrying the `key` and `token` query params
// Google generates. Returns an error string, or null when the shape is acceptable.
function webhookUrlError(webhookUrl: string): string | null {
  let u: URL
  try {
    u = new URL(webhookUrl)
  } catch {
    return 'The Google Chat webhook URL is not a valid URL.'
  }
  if (u.protocol !== 'https:') return 'The Google Chat webhook URL must be https.'
  if (!/(^|\.)googleapis\.com$/i.test(u.hostname)) {
    return 'The Google Chat webhook URL must be a chat.googleapis.com URL.'
  }
  if (!/\/v1\/spaces\/[^/]+\/messages\/?$/.test(u.pathname)) {
    return 'The Google Chat webhook URL must look like /v1/spaces/<id>/messages.'
  }
  if (!u.searchParams.get('key') || !u.searchParams.get('token')) {
    return 'The Google Chat webhook URL is missing its key/token query parameters.'
  }
  return null
}

// Parse the `spaces/<id>` name out of a webhook URL path (…/v1/spaces/<id>/messages).
function spaceFromWebhookUrl(webhookUrl: string): string | null {
  try {
    const u = new URL(webhookUrl)
    const m = u.pathname.match(/\/v1\/(spaces\/[^/]+)\/messages/)
    return m ? m[1] : null
  } catch {
    return null
  }
}

// Parse the `spaces/<id>` prefix out of a message resource name (spaces/<id>/messages/<mid>).
function spaceFromMessageName(name: string): string | null {
  const m = name.match(/^(spaces\/[^/]+)\//)
  return m ? m[1] : null
}

// Normalize a space reference to its bare comparable id: strip a leading "spaces/" and
// lower-case. So "spaces/AAAA", "AAAA", and "  spaces/aaaa " all compare equal.
function normalizeSpaceId(ref: string): string {
  return ref.trim().replace(/^spaces\//i, '').trim().toLowerCase()
}

// Normalize a space reference to a canonical `spaces/<id>` (lower-cased) for matching.
function normalizeSpace(ref: string): string {
  const id = normalizeSpaceId(ref)
  return id ? `spaces/${id}` : ''
}

function sameSpace(a: string, b: string): boolean {
  return normalizeSpaceId(a) === normalizeSpaceId(b)
}

// Build the set of allowed canonical space names from the csv allowlist. Accepts both bare
// ids and full "spaces/<id>" entries; commas/semicolons/whitespace separate.
function allowedSpaceSet(raw: string): Set<string> {
  const set = new Set<string>()
  for (const part of raw.split(/[\s,;]+/)) {
    const canon = normalizeSpace(part)
    if (canon) set.add(canon)
  }
  return set
}

function filterSpaces(spaces: GoogleChatSpace[], allowed: string | undefined): GoogleChatSpace[] {
  const raw = allowed?.trim()
  if (!raw) return spaces
  const set = allowedSpaceSet(raw)
  return spaces.filter((s) => set.has(normalizeSpace(s.name)))
}

// Add/replace a query parameter on a URL string without disturbing the existing ones.
function appendQueryParam(urlStr: string, key: string, value: string): string {
  try {
    const u = new URL(urlStr)
    u.searchParams.set(key, value)
    return u.toString()
  } catch {
    const sep = urlStr.includes('?') ? '&' : '?'
    return `${urlStr}${sep}${encodeURIComponent(key)}=${encodeURIComponent(value)}`
  }
}

// Surface the platform's error body message: Google APIs nest it at error.message.
function errorMessage(data: any): string | null {
  if (!data) return null
  if (typeof data.error === 'string') return data.error
  if (data.error && typeof data.error.message === 'string') return data.error.message
  if (typeof data.message === 'string') return data.message
  return null
}

function toGoogleChatMessage(source: GoogleChatSource, space: GoogleChatSpace, m: any): GoogleChatReadResult {
  // Google Chat surfaces user-facing text as `argumentText` (mention-stripped) when the bot
  // is @-mentioned, else `text`. Prefer argumentText so a "@Bot do X" reads as "do X".
  const text = typeof m?.argumentText === 'string' && m.argumentText.length > 0
    ? m.argumentText
    : typeof m?.text === 'string'
      ? m.text
      : ''
  const sender = m?.sender ?? {}
  const senderName = sender.displayName || sender.email || sender.name || 'unknown'
  const threadName = (m?.thread?.name as string) || null
  // Key the thread by its Google Chat thread name when present, else by the space — so a
  // top-level message and its thread fold into distinct conversations like discord's channels.
  const threadKey = threadName ? `googlechat:${source.id}:${threadName}` : `googlechat:${source.id}:${space.name}`
  const date = parseCreateTime(m?.createTime)
  return {
    id: m?.name || `${space.name}/messages/${randomUUID()}`,
    sourceId: source.id,
    sourceLabel: source.label,
    spaceName: space.name,
    spaceDisplayName: space.displayName || space.name,
    threadName,
    senderId: sender.name || sender.email || 'unknown',
    senderName,
    senderEmail: sender.email || null,
    isBot: (sender.type || '').toUpperCase() === 'BOT',
    date,
    text,
    snippet: text.replace(/\s+/g, ' ').trim().slice(0, 200),
    replyTo: null,
    replyToAuthor: null,
    replyToText: null,
    threadKey,
    raw: (m ?? {}) as Record<string, unknown>,
  }
}

// Google Chat `createTime` is an RFC3339 timestamp string; fall back to now on a bad value.
function parseCreateTime(createTime: unknown): number {
  if (typeof createTime === 'string') {
    const t = new Date(createTime).getTime()
    if (Number.isFinite(t)) return t
  }
  return Date.now()
}
