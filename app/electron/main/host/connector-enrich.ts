// Host-side enrichment applied to a freshly-synced batch of connector events, threading
// two cross-cutting features into the existing sync path WITHOUT each connector
// re-implementing them:
//
//   • CONTACT NAMES (contact-directory.ts) — learn handle -> display-name pairs from what
//     the sync already saw, persist them, and rewrite an event's actor/title to the
//     resolved NAME when the raw value was a phone number / opaque id. So the feed shows
//     "Maya Chen" instead of "+447700900123", and "reply to Maya" can later resolve back.
//   • MEDIA CACHE (media-cache.ts) — download each event's inbound attachment URLs to local
//     files under ~/.vontra/cache and stamp the local paths onto event.mediaPaths, so the
//     vision tool can read the image by path after the remote CDN link has expired.
//
// The contact half is pure (returns the next directory + rewritten events); the media half
// does I/O. The Discord sync wires this in; other connectors can adopt it incrementally.

import { cacheAuthedMedia, cacheInboundMedia, cacheInboundMediaBytes, MEDIA_SIZE_CAP_BYTES, type InboundMediaBytes } from '../../agent/media-cache'
import {
  displayFor,
  displayNameFromEmailFrom,
  emailAddressFromFrom,
  learnContacts,
  looksLikePhone,
  type ContactDirectory,
  type ContactObservation,
  type ContactPlatform,
} from '../../agent/contact-directory'
import type { ConnectorEvent } from '../../agent/types'
import { vontraPath } from '../../store'

/** The on-disk root for the media cache (~/.vontra/cache). */
export function mediaCacheRoot(): string {
  return vontraPath('cache')
}

// Map a connector id to the contact-directory platform bucket. iMessage/WhatsApp handles
// are phone numbers; slack/discord are id namespaces; email is addresses.
export function contactPlatformFor(connectorId: ConnectorEvent['connectorId']): ContactPlatform {
  switch (connectorId) {
    case 'imessage':
    case 'whatsapp':
      return 'phone'
    case 'slack':
      return 'slack'
    case 'discord':
      return 'discord'
    case 'email':
      return 'email'
    case 'telegram':
      return 'telegram'
    default:
      return 'other'
  }
}

export interface ContactEnrichResult {
  directory: ContactDirectory
  events: ConnectorEvent[]
}

/**
 * Learn the supplied observations into the directory, then rewrite each event so a
 * name-shaped field that currently shows a raw handle shows the resolved name instead.
 * Pure (no I/O); returns the next directory + the rewritten events. An event is rewritten
 * only when (a) we can read its handle (from payload.handle/sender for messages) and (b) a
 * name is known — otherwise it is passed through untouched. Returns the SAME events array
 * contents when nothing resolved, so callers can persist conditionally.
 */
export function enrichContacts(
  dir: ContactDirectory,
  events: ConnectorEvent[],
  observations: ContactObservation[],
  now = Date.now(),
): ContactEnrichResult {
  const directory = learnContacts(dir, observations, now)
  const rewritten = events.map((event) => rewriteEventActor(directory, event))
  return { directory, events: rewritten }
}

// Rewrite an event's actor/title to the resolved display name when the actor is a raw
// handle (a phone number or an opaque id) and a name is known. The original handle is
// preserved on the payload (payload.handle) so reply routing still has it.
function rewriteEventActor(dir: ContactDirectory, event: ConnectorEvent): ConnectorEvent {
  const platform = contactPlatformFor(event.connectorId)
  const handle = handleForEvent(event)
  if (!handle) return event
  const name = displayFor(dir, platform, handle)
  if (!name || name === handle) return event
  // Only replace the actor when it currently *is* the raw handle (a number/id), so we never
  // clobber a channel name like "#launch-room" or an already-resolved name.
  const actorIsHandle = event.actor === handle || looksLikePhone(event.actor) || event.actor === String(event.payload?.sender ?? '')
  const nextActor = actorIsHandle ? name : event.actor
  const nextTitle = event.title === handle || looksLikePhone(event.title) ? name : event.title
  if (nextActor === event.actor && nextTitle === event.title) return event
  return { ...event, actor: nextActor, title: nextTitle }
}

// The raw handle an event represents, read from its payload (set by the connector event
// builders: imessage payload.handle, discord/slack payload.sender, etc.).
function handleForEvent(event: ConnectorEvent): string {
  const p = event.payload ?? {}
  const candidate = p.handle ?? p.sender ?? p.from
  const handle = typeof candidate === 'string' ? candidate.trim() : ''
  return handle
}

/**
 * Download every event's inbound attachment URLs to the media cache and stamp the local
 * paths onto event.mediaPaths. Mutates+returns the events (in place) for the host's
 * convenience; failures are swallowed per-URL so a bad attachment never blocks the message.
 * `mediaUrlsFor` maps an event to its candidate URLs (the connector knows where they live —
 * e.g. Discord attachments off the raw message). Only inbound events are processed.
 */
export async function attachMediaPaths(
  events: ConnectorEvent[],
  mediaUrlsFor: (event: ConnectorEvent) => string[],
  opts: { capBytes?: number; cacheRoot?: string } = {},
): Promise<ConnectorEvent[]> {
  const cacheRoot = opts.cacheRoot ?? mediaCacheRoot()
  for (const event of events) {
    if (event.direction === 'outbound') continue
    const urls = mediaUrlsFor(event) ?? []
    if (!urls.length) continue
    const paths = await cacheInboundMedia(urls, { cacheRoot, capBytes: opts.capBytes ?? MEDIA_SIZE_CAP_BYTES })
    if (paths.length) {
      event.mediaPaths = paths
      if (!event.payloadKeys.includes('mediaPaths')) event.payloadKeys = [...event.payloadKeys, 'mediaPaths']
    }
  }
  return events
}

// Stamp mediaPaths on an event (merging with any already present) and register the payload key.
function stampMediaPaths(event: ConnectorEvent, paths: string[]): void {
  if (!paths.length) return
  const merged = [...new Set([...(event.mediaPaths ?? []), ...paths])]
  event.mediaPaths = merged
  if (!event.payloadKeys.includes('mediaPaths')) event.payloadKeys = [...event.payloadKeys, 'mediaPaths']
}

/**
 * AUTH-aware sibling of attachMediaPaths for media whose download needs an Authorization header
 * (Slack url_private requires the workspace bot token as a Bearer). `mediaItemsFor` maps an event
 * to its `{ url, headers }` items (the connector knows where the auth'd URLs live and which token
 * to attach). Reuses the same SSRF + size-cap contract via cacheAuthedMedia; failures are
 * swallowed per item so a bad/expired attachment never blocks ingest. Only inbound events.
 */
export async function attachAuthedMediaPaths(
  events: ConnectorEvent[],
  mediaItemsFor: (event: ConnectorEvent) => {
    url: string
    headers?: Record<string, string>
    endpoint?: { baseUrl: string; label: string; options?: { allowPrivateNetwork?: boolean } }
  }[],
  opts: { capBytes?: number; cacheRoot?: string } = {},
): Promise<ConnectorEvent[]> {
  const cacheRoot = opts.cacheRoot ?? mediaCacheRoot()
  for (const event of events) {
    if (event.direction === 'outbound') continue
    const items = mediaItemsFor(event) ?? []
    if (!items.length) continue
    const paths = await cacheAuthedMedia(items, { cacheRoot, capBytes: opts.capBytes ?? MEDIA_SIZE_CAP_BYTES })
    stampMediaPaths(event, paths)
  }
  return events
}

/**
 * Stamp mediaPaths from already-in-memory attachment BYTES (email MIME parts have no URL to
 * fetch). `bytesFor` maps an event to its decoded attachment parts; each is written to the media
 * cache (classified + size-capped) and the resulting local paths are stamped. Per-item failures
 * are swallowed so a bad attachment never blocks ingest. Only inbound events.
 */
export async function attachMediaBytesPaths(
  events: ConnectorEvent[],
  bytesFor: (event: ConnectorEvent) => InboundMediaBytes[],
  opts: { capBytes?: number; cacheRoot?: string } = {},
): Promise<ConnectorEvent[]> {
  const cacheRoot = opts.cacheRoot ?? mediaCacheRoot()
  for (const event of events) {
    if (event.direction === 'outbound') continue
    const items = bytesFor(event) ?? []
    if (!items.length) continue
    const paths = await cacheInboundMediaBytes(items, { cacheRoot, capBytes: opts.capBytes ?? MEDIA_SIZE_CAP_BYTES })
    stampMediaPaths(event, paths)
  }
  return events
}

// ---- Per-connector contact observation builders (pure) ------------------------
// Each maps a batch of a connector's freshly-synced events to the (platform, handle, name)
// observations the contact-directory should learn — the connector-specific half of the
// enrichment that the Discord/iMessage sites inline. Kept here (pure, no I/O) so the runtime
// call sites stay one-liners and the handle/name extraction is unit-tested. Only INBOUND
// events teach contacts (an outbound "You" send must never overwrite a real contact's name).

/**
 * Slack: learn a Slack user id -> display-name from each inbound message's payload
 * (sender + senderId, stamped by slackEventFromMessage). The actor is the channel name, so
 * enrichContacts won't rewrite it; this only grows the directory so "reply to <name>" on
 * Slack resolves. Skips bot/unknown ids and the placeholder where the name IS the id (the
 * live API falls back senderName=senderId until users.info resolves) — isUsableName drops
 * those downstream, but filtering here keeps the observation list tight.
 */
export function slackContactObservations(events: ConnectorEvent[]): ContactObservation[] {
  const out: ContactObservation[] = []
  for (const event of events) {
    if (event.connectorId !== 'slack' || event.direction !== 'inbound') continue
    const handle = stringField(event.payload?.senderId)
    const name = stringField(event.payload?.sender)
    if (!handle || !name || name === handle) continue
    out.push({ platform: 'slack', handle, name, source: 'slack' })
  }
  return out
}

/**
 * Email: learn an address -> display-name from each inbound message's From header (stamped
 * onto payload.from by emailEventFromMessage). A bare address with no display name yields no
 * observation (displayNameFromEmailFrom returns '' — nothing to learn). The actor already
 * carries the From header, so this only grows the directory for "reply to <name>".
 */
export function emailContactObservations(events: ConnectorEvent[]): ContactObservation[] {
  const out: ContactObservation[] = []
  for (const event of events) {
    if (event.connectorId !== 'email' || event.direction !== 'inbound') continue
    const from = stringField(event.payload?.from) || stringField(event.actor)
    const address = emailAddressFromFrom(from)
    const name = displayNameFromEmailFrom(from)
    if (!address || !name) continue
    out.push({ platform: 'email', handle: address, name, source: 'email' })
  }
  return out
}

/**
 * WhatsApp: learn a phone (from the conversation JID) -> pushName from each inbound message.
 * The bridge already resolves the sender's WhatsApp display name (pushName) into actor; we
 * key on the JID's number (normalizeHandle reduces "<number>@s.whatsapp.net" to E.164) so a
 * later "reply to <name>" can find the number to send to. Group messages (@g.us) are skipped:
 * the conversation JID is the GROUP, not the sender, so the number wouldn't be the author's.
 */
export function whatsappContactObservations(events: ConnectorEvent[]): ContactObservation[] {
  const out: ContactObservation[] = []
  for (const event of events) {
    if (event.connectorId !== 'whatsapp' || event.direction !== 'inbound') continue
    const jid = stringField(event.payload?.jid)
    if (!jid || jid.endsWith('@g.us')) continue
    const name = stringField(event.actor)
    if (!name) continue
    out.push({ platform: 'phone', handle: jid, name, source: 'whatsapp' })
  }
  return out
}

function stringField(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}
