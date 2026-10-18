// One contract for sending a reply to any connector, so the host has a SINGLE reply
// path instead of five bespoke `<platform>:send` handlers. Routing keys off the typed
// ThreadAddress (thread-address.ts): parse the threadKey, pick the adapter by platform,
// send. Adding a channel = implement one adapter + register it. Pure router + injected
// adapters (testable with fakes); concrete adapters wrap the existing send fns.

import { limitForPlatform, splitForPlatform } from './channel-split'
import { makeDraftEditThrottle, type DraftEditThrottle } from './draft-throttle'
import { parseThreadAddress, type ThreadAddress } from './thread-address'
import type { ChannelApprovalRequest, ChannelApprovalResult, ConnectorApprovalDecision } from './types/connectors'

export type { ChannelApprovalRequest, ChannelApprovalResult } from './types/connectors'
export { PLATFORM_MESSAGE_LIMITS, limitForPlatform, splitForPlatform, splitMessage } from './channel-split'

export interface OutboundReply {
  text: string
  // Anchor the reply to a specific source message where the channel supports it
  // (slack thread_ts, discord reply-to, email In-Reply-To). Optional.
  inReplyTo?: string | null
}

export interface SendResult {
  ok: boolean
  error?: string
  // Platform message id of the sent reply, when the channel returns one. For a reply split
  // into several messages this is the id of the FIRST message (continuationIds carries the rest).
  sentId?: string
  // When a long reply was split across multiple messages, the platform ids of the
  // continuation messages (2nd..Nth), in order. Absent/undefined for a single-message send,
  // so existing callers that only read `ok`/`sentId` are unaffected (additive).
  continuationIds?: string[]
  // Total number of messages the reply was sent as (1 when not split). Absent for callers
  // that don't need it; present whenever sendViaChannel chunked a reply.
  chunkCount?: number
}

// ---- Draft streaming (optional capability) ----
// A live preview the agent can animate WHILE composing a reply: the first partial is sent
// (or, on Telegram, the message is created once and edited in place), each subsequent partial
// edits the same message, and a final commit replaces it with the finished text. Adapters that
// can't edit a sent message simply don't implement this and the router sends once at the end.
export interface DraftHandle {
  // Platform message id of the in-flight draft message, used to edit/finalize it. Adapters
  // that can't return an id (or chose to no-op) leave it null and updates become no-ops.
  messageId: string | null
  // The adapter that owns this draft, so update/finalize route back to the same channel.
  platform: string
}
export interface DraftResult {
  ok: boolean
  error?: string
  handle?: DraftHandle
}

export interface ChannelAdapter {
  platform: string
  send(address: ThreadAddress, reply: OutboundReply): Promise<SendResult>
  // Capability flag: true when this channel can render inline approve/deny controls
  // (Slack Block Kit buttons, Discord buttons, Telegram inline keyboard) and resolve the
  // user's decision in-channel. Plain-text channels leave this false/undefined.
  supportsInlineApproval?: boolean
  // Prompt the user for a connector-action approval IN the channel and resolve with their
  // decision. Only adapters with supportsInlineApproval implement this; the router falls
  // back to a plain-text prompt (and the in-app gate) for the rest.
  requestApproval?(address: ThreadAddress, request: ChannelApprovalRequest): Promise<ConnectorApprovalDecision>
  // Capability flag: true when this channel can render a live, editable preview while the
  // agent composes (edit-in-place on Telegram; send-then-edit elsewhere). Adapters leave it
  // false/undefined and the router falls back to a single send() at the end.
  supportsDraftStreaming?: boolean
  // Begin a streaming draft with the first partial text, returning a DraftHandle the router
  // uses to update/finalize it. Only adapters with supportsDraftStreaming implement this.
  sendDraft?(address: ThreadAddress, text: string): Promise<DraftResult>
  // Edit the in-flight draft message to the new partial text (animation frame). No-ops safely
  // when the handle has no message id. Only implemented alongside sendDraft.
  updateDraft?(address: ThreadAddress, handle: DraftHandle, text: string): Promise<DraftResult>
  // Commit the draft: replace its content with the finished reply and return the final send
  // result (so the caller learns the message id, as if it had called send()).
  finalizeDraft?(address: ThreadAddress, handle: DraftHandle, reply: OutboundReply): Promise<SendResult>
}

/** Pick the adapter that owns a parsed address's platform. */
export function resolveAdapter(adapters: ChannelAdapter[], address: ThreadAddress | null): ChannelAdapter | null {
  if (!address) return null
  return adapters.find((a) => a.platform === address.platform) ?? null
}

/**
 * Unified outbound send: parse the threadKey → route to the platform's adapter → send.
 * Folds in the outbound SILENCE GUARD (an empty/whitespace reply is never sent — a
 * channel chokepoint so the agent can't post a blank message). Returns a normalized
 * SendResult so callers (approval flow, IPC) handle every channel the same way.
 */
export async function sendViaChannel(adapters: ChannelAdapter[], threadKey: string, reply: OutboundReply, options?: { preserveText?: boolean }): Promise<SendResult> {
  if (!reply || !reply.text || !reply.text.trim()) return { ok: false, error: 'Refusing to send an empty reply.' }
  const address = parseThreadAddress(threadKey)
  if (!address) return { ok: false, error: 'Unrecognized conversation.' }
  const adapter = resolveAdapter(adapters, address)
  if (!adapter) return { ok: false, error: `No channel adapter for "${address.platform}".` }
  // Split a long reply at safe boundaries under the platform's per-message cap. Short replies
  // (the common case) yield a single chunk and send exactly as before.
  const chunks = options?.preserveText ? exactMessageChunks(address.platform, reply.text) : splitForPlatform(address.platform, reply.text)
  try {
    if (chunks.length <= 1) {
      return await adapter.send(address, { text: chunks[0] ?? reply.text.trim(), inReplyTo: reply.inReplyTo ?? null })
    }
    // Anchor only the FIRST chunk to inReplyTo; continuations chain naturally after it. Stop on
    // the first failure so we don't keep posting after the channel has started rejecting.
    const continuationIds: string[] = []
    let firstId: string | undefined
    let sent = 0
    for (let i = 0; i < chunks.length; i++) {
      const part = await adapter.send(address, { text: chunks[i], inReplyTo: i === 0 ? (reply.inReplyTo ?? null) : null })
      if (!part.ok) {
        return { ok: false, error: part.error ?? `Failed sending message ${i + 1} of ${chunks.length}.`, sentId: firstId, continuationIds, chunkCount: sent }
      }
      sent++
      if (i === 0) firstId = part.sentId
      else if (part.sentId) continuationIds.push(part.sentId)
    }
    return { ok: true, sentId: firstId, continuationIds, chunkCount: sent }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) }
  }
}

// Reviewed sends retain every character. Ordinary conversational replies keep the
// existing whitespace-aware splitting behavior; no transport fallback retries a chunk.
function exactMessageChunks(platform: string, text: string): string[] {
  const max = limitForPlatform(platform).maxUnits
  if (!max || text.length <= max) return [text]
  const chunks: string[] = []
  for (let start = 0; start < text.length;) {
    let end = Math.min(start + max, text.length)
    if (end < text.length && /[\uD800-\uDBFF]/.test(text[end - 1]) && /[\uDC00-\uDFFF]/.test(text[end])) end--
    chunks.push(text.slice(start, end)); start = end
  }
  return chunks
}

/** Platforms a set of adapters can currently send to (for capability checks / UI). */
export function sendablePlatforms(adapters: ChannelAdapter[]): string[] {
  return [...new Set(adapters.map((a) => a.platform))]
}

/** Platforms whose adapters can resolve an approve/deny decision in-channel. */
export function inlineApprovalPlatforms(adapters: ChannelAdapter[]): string[] {
  return [...new Set(adapters.filter((a) => a.supportsInlineApproval && a.requestApproval).map((a) => a.platform))]
}

/** Platforms whose adapters can animate a live draft preview (sendDraft + finalizeDraft). */
export function draftStreamingPlatforms(adapters: ChannelAdapter[]): string[] {
  return [...new Set(adapters.filter((a) => a.supportsDraftStreaming && a.sendDraft && a.finalizeDraft).map((a) => a.platform))]
}

// The placeholder shown as the live draft's first frame when a reply has no incremental source
// (a connector reply arrives as one finished string). It animates a brief "composing" state that
// finalizeDraft then replaces with the real message — so even a non-streamed reply gets the
// live-preview affordance on a capable channel. Never sent as the actual message (the silence
// guard + the authoritative `final` text own that).
export const COMPOSING_INDICATOR = '…'

/**
 * Build a DraftStream for a reply that is only available as a FINAL string (no token stream): show
 * a brief composing placeholder, then commit the real text. On a draft-capable channel this opens a
 * placeholder message and finalizes it into `text`; on a non-capable channel streamDraftViaChannel
 * drains it and sends `text` once. Pure — yields the placeholder then the final text, with `final`
 * pinned to the real reply so finalize always lands the true message regardless of throttling.
 */
export function composingDraftStream(text: string, inReplyTo?: string | null): DraftStream {
  const finalText = String(text ?? '')
  return {
    partials: (function* () {
      // Animate the placeholder ONLY when there's a real message coming; otherwise yield nothing so
      // the silence guard refuses an empty reply (no lonely "…" gets committed). The real text is
      // carried by `final` below — finalizeDraft commits it, so we don't yield it as a partial.
      if (finalText.trim()) yield COMPOSING_INDICATOR
    })(),
    final: { text: finalText, inReplyTo: inReplyTo ?? null },
  }
}

// What a streaming reply produces as it composes: a sequence of partial texts, then the final
// reply. The router edits a live preview from the partials (when the channel can) and commits
// the final text. `final` is the authoritative reply text the user keeps.
export interface DraftStream {
  // Successive partial snapshots of the reply as the agent writes it (each REPLACES the last).
  // The router throttles/edits at its discretion; an empty/whitespace partial is skipped.
  partials: AsyncIterable<string> | Iterable<string>
  // The finished reply. If omitted, the last non-empty partial is used as the final text.
  final?: OutboundReply
}

/**
 * Stream a live draft preview to a channel, then finalize it. If the channel supports draft
 * streaming (supportsDraftStreaming + sendDraft/finalizeDraft), the first partial creates an
 * editable message, each later partial edits it in place (Telegram) or via a send-then-edit
 * cycle (others' adapters implement that internally), and the final text commits the message.
 *
 * GRACEFUL FALLBACK: adapters without the capability never see a partial — the helper just
 * drains the stream to learn the final text and sends it ONCE via the normal (split-aware)
 * sendViaChannel path. So a caller can always opt into streaming and let unsupported channels
 * degrade to a single send. Returns the same SendResult shape as sendViaChannel.
 *
 * RATE LIMIT: intermediate animation frames (updateDraft) are gated by a min-interval throttle so
 * a fast token stream can't trip the channel's edit rate limit (Telegram's editMessageText). The
 * throttle only drops INTERMEDIATE frames — the opening sendDraft and the closing finalizeDraft
 * always go through, so the user still sees a draft appear and always ends on the full message. Pass
 * your own throttle (e.g. a permissive one in tests) to override the default 1.5s spacing.
 */
export async function streamDraftViaChannel(
  adapters: ChannelAdapter[],
  threadKey: string,
  stream: DraftStream,
  throttle: DraftEditThrottle = makeDraftEditThrottle(),
): Promise<SendResult> {
  const address = parseThreadAddress(threadKey)
  if (!address) return { ok: false, error: 'Unrecognized conversation.' }
  const adapter = resolveAdapter(adapters, address)
  if (!adapter) return { ok: false, error: `No channel adapter for "${address.platform}".` }
  const canStream = Boolean(adapter.supportsDraftStreaming && adapter.sendDraft && adapter.finalizeDraft)
  let lastPartial = ''
  // Fallback path: drain partials only to capture the latest non-empty text, then send once.
  if (!canStream) {
    try {
      for await (const partial of toAsync(stream.partials)) {
        if (partial && partial.trim()) lastPartial = partial
      }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) }
    }
    const finalReply = stream.final ?? { text: lastPartial }
    if (!finalReply.text || !finalReply.text.trim()) return { ok: false, error: 'Refusing to send an empty reply.' }
    return sendViaChannel(adapters, threadKey, finalReply)
  }
  // Streaming path: open the draft on the first partial, edit on each subsequent one.
  let handle: DraftHandle | null = null
  try {
    for await (const partial of toAsync(stream.partials)) {
      if (!partial || !partial.trim()) continue
      lastPartial = partial
      if (!handle) {
        const started = await adapter.sendDraft!(address, partial)
        if (!started.ok || !started.handle) {
          // Couldn't open a preview — bail to a single final send so the reply still goes out.
          const finalReply = stream.final ?? { text: lastPartial }
          return sendViaChannel(adapters, threadKey, finalReply)
        }
        handle = started.handle
        // The opening frame is the draft itself (sendDraft) — count it against the throttle so the
        // very next token doesn't immediately fire an edit on top of it.
        throttle.markFinal(Date.now())
      } else if (throttle.shouldEmit(Date.now())) {
        // A failed in-flight edit is non-fatal: keep the latest text and let finalize commit it.
        // Throttled: frames arriving inside the min-interval are dropped (the next allowed frame
        // carries the newest text), so a rapid stream can't exceed the channel's edit rate limit.
        await adapter.updateDraft?.(address, handle, partial)
      }
    }
    const finalReply = stream.final ?? { text: lastPartial }
    if (!finalReply.text || !finalReply.text.trim()) return { ok: false, error: 'Refusing to send an empty reply.' }
    if (!handle) {
      // No partial ever arrived (empty stream) — just send the final text once.
      return sendViaChannel(adapters, threadKey, finalReply)
    }
    return await adapter.finalizeDraft!(address, handle, { text: finalReply.text.trim(), inReplyTo: finalReply.inReplyTo ?? null })
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) }
  }
}

/** Normalize a sync or async iterable of partials into an async iterable. */
async function* toAsync(it: AsyncIterable<string> | Iterable<string>): AsyncIterable<string> {
  if (Symbol.asyncIterator in Object(it)) {
    for await (const v of it as AsyncIterable<string>) yield v
  } else {
    for (const v of it as Iterable<string>) yield v
  }
}

/**
 * Parse an inbound channel reply into an approval decision. Accepts the inline-control
 * payloads ("approve"/"deny", and the token-suffixed "approve:<id>"/"deny:<id>" a button
 * carries) AND the plain-text fallback ("/approve", "/deny", "yes"/"no", "y"/"n"). Returns
 * null when the text is not a recognizable decision — so an ordinary chat message is never
 * misread as an approval. Pure + tested.
 */
export function parseApprovalReply(text: string | null | undefined): ConnectorApprovalDecision | null {
  const raw = String(text ?? '').trim().toLowerCase()
  if (!raw) return null
  // First token, after stripping a leading slash and any token suffix ("/approve:ab12" ->
  // "approve"). The slash/keyword forms are the unambiguous inline-control + command payloads.
  const body = raw.replace(/^\//, '')
  const head = body.split(/[\s:@]/, 1)[0]
  const wasCommand = raw.startsWith('/')
  // The reply is a DEDICATED decision payload when it is the lone keyword, optionally carrying a
  // button's token suffix ("approve" or "approve:ab12") — i.e. nothing follows but a ":<token>".
  // It must NOT match when the keyword is merely the first word of a longer sentence
  // ("approve the budget tomorrow", "cancel that"), which would otherwise silently resolve a
  // parked approval the user never meant to act on.
  const rest = body.slice(head.length) // '' for a lone keyword, ':<token>' for a button payload
  const dedicated = rest === '' || /^:[^\s:@]+$/.test(rest)
  // Emoji reactions are an unambiguous in-channel decision on their own.
  if (head === '👍' || head === '✅') return 'approve'
  if (head === '👎' || head === '❌') return 'deny'
  // Every keyword (hard decisions AND soft affirmations) counts only as the WHOLE reply, a
  // token-suffixed button payload, or a slash command.
  if (wasCommand || dedicated) {
    if (['approve', 'approved', 'allow'].includes(head)) return 'approve'
    if (['deny', 'denied', 'reject', 'rejected', 'cancel'].includes(head)) return 'deny'
    if (['ok', 'okay', 'yes', 'y'].includes(head)) return 'approve'
    if (['no', 'n', 'nope', 'stop'].includes(head)) return 'deny'
  }
  return null
}

/**
 * Format the plain-text approval prompt for channels without inline controls. The user
 * replies "/approve" or "/deny" (parsed by parseApprovalReply) and the in-app gate stays
 * the source of truth for the actual decision.
 */
export function plainTextApprovalPrompt(request: ChannelApprovalRequest): string {
  const lines = [`Approval needed: ${request.title}`.trim()]
  if (request.summary?.trim()) lines.push(request.summary.trim())
  if (request.command?.trim()) lines.push(`Action: ${request.command.trim()}`)
  lines.push('Reply /approve to allow or /deny to decline.')
  return lines.join('\n')
}

/**
 * Unified in-channel approval: parse the threadKey -> route to the platform's adapter. If
 * the adapter supports inline controls, prompt and block until the user answers in-channel.
 * Otherwise post a PLAIN-TEXT prompt via the normal send path and return fallbackToApp so
 * the caller keeps the in-app approval gate authoritative. Never rejects — every channel
 * resolves to a normalized ChannelApprovalResult the caller can branch on.
 */
export async function requestApprovalViaChannel(
  adapters: ChannelAdapter[],
  threadKey: string,
  request: ChannelApprovalRequest,
): Promise<ChannelApprovalResult> {
  const address = parseThreadAddress(threadKey)
  if (!address) return { ok: false, error: 'Unrecognized conversation.', fallbackToApp: true }
  const adapter = resolveAdapter(adapters, address)
  if (!adapter) return { ok: false, error: `No channel adapter for "${address.platform}".`, fallbackToApp: true }
  if (adapter.supportsInlineApproval && adapter.requestApproval) {
    try {
      const decision = await adapter.requestApproval(address, request)
      return { ok: true, decision }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e), fallbackToApp: true }
    }
  }
  // Plain-text fallback: post the prompt so the user sees it in-channel, then defer the
  // actual decision to the in-app gate (the channel can't block synchronously here).
  const posted = await adapter.send(address, { text: plainTextApprovalPrompt(request), inReplyTo: address.message ?? null })
  return { ok: false, error: posted.ok ? undefined : posted.error, fallbackToApp: true }
}
