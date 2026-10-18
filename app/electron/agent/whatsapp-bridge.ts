import { randomUUID, randomBytes } from 'node:crypto'
import path from 'node:path'
import type { ConnectorEvent, WhatsAppBridgeConnectorSettings, WhatsAppBridgeSource, WhatsAppBridgeRuntimeSnapshot } from './types'
import { normalizeWhatsAppConversationJid, normalizeWhatsAppJid } from '../../shared/whatsapp-event-scope'
export { normalizeWhatsAppConversationJid, normalizeWhatsAppJid } from '../../shared/whatsapp-event-scope'
import { cacheWhatsAppMediaMessage, whatsappMediaCaption, whatsappMediaDescriptor } from './whatsapp-media-ingest.ts'

// Multi-account WhatsApp bridge via the WhatsApp multidevice protocol (the
// standard "Linked Devices" QR flow). Each account is one source with its own
// persisted auth credentials (a folder under TensorFold state), so you can pair
// several numbers. Inbound messages become connector events; outbound sends are
// approval-gated by the existing connector approval path.
//
// This uses the *unofficial* multidevice protocol (Baileys). Bridging your own
// accounts on your own machine is an established legitimate use, but it is
// technically against WhatsApp's ToS and carries a small account-flagging risk.
// The connector is OFF by default and clearly labeled experimental.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type BaileysModule = any

// Resolve the usable baileys API object out of whatever an ESM dynamic import() hands back.
//
// baileys is pure ESM (`"type":"module"`, default export = makeWASocket, plus a flat set of
// NAMED exports: useMultiFileAuthState / DisconnectReason / fetchLatestBaileysVersion / …).
// When the Electron main (bundled to CJS) does `await import('@whiskeysockets/baileys')`, Node
// returns the Module *namespace*: the named exports live on the namespace itself, while
// `namespace.default` is the makeWASocket function ALONE (it has no `.useMultiFileAuthState`).
//
// The previous `m.default ?? m` picked that bare function, so the very next line —
// `typeof mod.makeWASocket` — was undefined and pairing threw "library failed to load" even
// though baileys had loaded fine. We must keep the NAMESPACE (named exports + the makeWASocket
// the bridge calls). Only fall through to `.default` for the unlikely CJS-interop double-wrap
// where a real namespace isn't present but `.default` carries the whole API. Exported for tests.
export function resolveBaileysModule(m: unknown): BaileysModule {
  const ns = m as { default?: BaileysModule; makeWASocket?: unknown } | undefined
  // The namespace already exposes makeWASocket as a named export → it carries the full API.
  if (ns && typeof ns.makeWASocket === 'function') return ns
  // Fallback: an interop wrapper where the real module sits under .default.
  const def = ns?.default as { makeWASocket?: unknown } | undefined
  if (def && typeof def.makeWASocket === 'function') return def
  return ns
}

let baileysPromise: Promise<BaileysModule> | null = null
async function loadBaileys(): Promise<BaileysModule> {
  if (!baileysPromise) {
    // NOTE: @whiskeysockets/baileys is marked EXTERNAL in the electron-main rollup config
    // (vite.config.ts) so this stays a genuine ESM `import()` of the real package from
    // node_modules at runtime. Inlining it (rollup tried to and emitted a 110B empty stub)
    // is what previously made loadBaileys() resolve to {} and pairing silently do nothing.
    baileysPromise = import('@whiskeysockets/baileys')
      .then((m) => {
        const mod = resolveBaileysModule(m)
        // Guard against a bundler stub (an empty module) shipping instead of the real
        // package — that's what made pairing silently do nothing. Fail loud + clear so
        // the error surfaces in the UI instead of a later "x is not a function".
        if (!mod || typeof mod.makeWASocket !== 'function') {
          throw new Error('WhatsApp bridge library failed to load (not available in this build).')
        }
        return mod
      })
      .catch((e) => {
        baileysPromise = null // allow a later retry instead of caching the failure forever
        throw e instanceof Error ? e : new Error('WhatsApp bridge library is not available.')
      })
  }
  return baileysPromise
}

export type { WhatsAppBridgeSource } from './types'
export type WhatsAppBridgeSettings = WhatsAppBridgeConnectorSettings

export function makeWhatsAppBridgeSource(label: string, now = Date.now()): WhatsAppBridgeSource {
  return {
    id: `wa_${randomUUID().slice(0, 8)}`,
    label: label || 'WhatsApp account',
    enabled: true,
    allowedChatJids: [],
    allowSelfChat: false,
    phoneNumber: null,
    paired: false,
    lastQrAt: null,
    lastError: null,
    lastEventAt: null,
    credentialRotatedAt: null,
    createdAt: now,
    updatedAt: now,
  }
}

// Ephemeral lifecycle belongs to the host, not the durable `paired` credential hint.
interface BridgeSession {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sock: any
  sourceId: string
  seen: Set<string>
  outbound: Set<string>
  ownJids: Set<string>
  runtime: WhatsAppBridgeRuntimeSnapshot
  reconnectTimer: ReturnType<typeof setTimeout> | null
  qrTimer: ReturnType<typeof setTimeout> | null
  onRuntime: (snapshot: WhatsAppBridgeRuntimeSnapshot) => void
  getSource: () => WhatsAppBridgeSource | undefined
  cancelSchedule: (timer: ReturnType<typeof setTimeout>) => void
}
const SEEN_CAP = 5000
const MAX_RECONNECTS = 5
const QR_LIFETIME_MS = 60_000
const sessions = new Map<string, BridgeSession>()
const runtimeStates = new Map<string, WhatsAppBridgeRuntimeSnapshot>()
const sourceLifecycleGenerations = new Map<string, number>()
let allBridgesGeneration = 0
/** Startup preflight uses the same stop/purge generation as socket initialization. */
export function getWhatsAppBridgeLifecycleEpoch(): number { return allBridgesGeneration }
function sourceLifecycleGeneration(sourceId: string): number { return sourceLifecycleGenerations.get(sourceId) ?? 0 }
function assertBridgeStartActive(sourceId: string, sourceGeneration: number, globalGeneration: number): void {
  if (sourceLifecycleGeneration(sourceId) !== sourceGeneration || allBridgesGeneration !== globalGeneration) {
    throw new Error('WhatsApp bridge start was cancelled because its credential session was stopped or purged.')
  }
}
function publishRuntime(session: BridgeSession, patch: Partial<WhatsAppBridgeRuntimeSnapshot>): void {
  session.runtime = { ...session.runtime, ...patch, updatedAt: Math.max(Date.now(), session.runtime.updatedAt + 1) }
  runtimeStates.set(session.sourceId, session.runtime)
  session.onRuntime({ ...session.runtime })
}
export function getWhatsAppBridgeRuntime(sourceId?: string): WhatsAppBridgeRuntimeSnapshot[] {
  return [...runtimeStates.values()].filter((s) => !sourceId || s.sourceId === sourceId).map((s) => ({
    ...s,
    ...(s.qrExpiresAt != null && s.qrExpiresAt <= Date.now() ? { qr: null, qrExpiresAt: null } : {}),
  }))
}
export function validateWhatsAppSourcePatch(patch: unknown): Pick<WhatsAppBridgeSource, 'label' | 'enabled' | 'allowedChatJids' | 'allowSelfChat'> | Partial<Pick<WhatsAppBridgeSource, 'label' | 'enabled' | 'allowedChatJids' | 'allowSelfChat'>> {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new Error('Invalid WhatsApp account settings.')
  const input = patch as Record<string, unknown>
  const result: Partial<Pick<WhatsAppBridgeSource, 'label' | 'enabled' | 'allowedChatJids' | 'allowSelfChat'>> = {}
  if ('label' in input) {
    if (typeof input.label !== 'string' || !input.label.trim() || input.label.length > 120) throw new Error('Enter an account label of 1–120 characters.')
    result.label = input.label.trim()
  }
  for (const key of ['enabled', 'allowSelfChat'] as const) {
    if (key in input) {
      if (typeof input[key] !== 'boolean') throw new Error(`Invalid WhatsApp ${key} setting.`)
      result[key] = input[key]
    }
  }
  if ('allowedChatJids' in input) {
    if (!Array.isArray(input.allowedChatJids) || input.allowedChatJids.length > 100) throw new Error('Select at most 100 exact WhatsApp contacts.')
    const normalized = input.allowedChatJids.map(normalizeWhatsAppConversationJid)
    if (normalized.some((jid) => !jid)) throw new Error('Use exact international phone numbers or WhatsApp contact/group JIDs; display names and wildcards are not accepted.')
    result.allowedChatJids = [...new Set(normalized as string[])]
  }
  return result
}
function selectedConversation(source: WhatsAppBridgeSource | undefined, session: BridgeSession, jid: string, fromMe: boolean): boolean {
  if (!source?.enabled) return false
  const normalized = normalizeWhatsAppConversationJid(jid)
  if (!normalized) return false
  const self = session.ownJids.has(normalized)
  if (fromMe) return self && source.allowSelfChat === true
  if (self) return source.allowSelfChat === true
  return (source.allowedChatJids ?? []).some((allowed) => normalizeWhatsAppConversationJid(allowed) === normalized)
}
// Returns true if this id is new (and records it); false if already seen this session.
export function markWhatsAppSeen(session: { seen: Set<string> }, messageId: string): boolean {
  if (!messageId) return true // no id to dedup on — let the store-level guard handle it
  if (session.seen.has(messageId)) return false
  session.seen.add(messageId)
  if (session.seen.size > SEEN_CAP) {
    // Trim oldest (insertion order) to bound memory; replays cluster around reconnect time.
    for (const id of session.seen) { session.seen.delete(id); if (session.seen.size <= SEEN_CAP) break }
  }
  return true
}

// A minimal pino-shaped no-op logger for makeWASocket. baileys (>=7) doesn't just read
// `logger.level` — it calls `logger.child(...)` and `logger.{trace,debug,info,warn,error,fatal}(...)`
// internally. A bare `{ level: 'silent' }` therefore throws "logger.child is not a function" the
// moment makeWASocket runs, which aborts pairing BEFORE any QR is emitted. This stand-in
// implements the full surface (child returns itself; every level method is a no-op) so the socket
// constructs and the QR flows — without pulling pino into the CJS main bundle. Pure + tested.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function silentBaileysLogger(): any {
  const noop = () => {}
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const logger: any = {
    level: 'silent',
    trace: noop,
    debug: noop,
    info: noop,
    warn: noop,
    error: noop,
    fatal: noop,
    // baileys passes a binding object to child(); we ignore it and reuse this same silent logger.
    child: () => logger,
  }
  return logger
}

export interface WhatsAppAuthStateHandle {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  state: any
  saveCreds(): Promise<number | null>
  credentialUpdatedAt(): number | null
}

export type WhatsAppAuthStateFactory = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  baileys: any,
  baseDir: string,
  sourceId: string,
) => Promise<WhatsAppAuthStateHandle>

/** Production keeps all I/O in Baileys and the protected auth adapter; tests inject only protocol/timing seams. */
export interface WhatsAppBridgeRuntimeDeps {
  loadBaileys?: () => Promise<BaileysModule>
  getSource?: () => WhatsAppBridgeSource | undefined
  onRuntime?: (snapshot: WhatsAppBridgeRuntimeSnapshot) => void
  schedule?: (callback: () => void, delay: number) => ReturnType<typeof setTimeout>
  cancelSchedule?: (timer: ReturnType<typeof setTimeout>) => void
  /** Auto-resume must never create credentials or start QR pairing. */
  requireRegistered?: boolean
}
export async function startWhatsAppBridge(
  source: WhatsAppBridgeSource,
  baseDir: string,
  onQr: (qr: string) => void,
  onEvent: (event: ConnectorEvent) => void,
  onStatus: (patch: Partial<WhatsAppBridgeSource>) => void,
  authStateFactory?: WhatsAppAuthStateFactory,
  deps: WhatsAppBridgeRuntimeDeps = {},
): Promise<WhatsAppBridgeSource> {
  if (!authStateFactory) throw new Error('WhatsApp QR pairing is disabled because a protected credential store is unavailable. Use the signed WhatsApp Cloud connector instead.')
  if (!source.enabled) throw new Error('Enable this WhatsApp account before connecting.')
  if (sessions.has(source.id)) return { ...source }
  const sourceGeneration = sourceLifecycleGeneration(source.id)
  const globalGeneration = allBridgesGeneration
  const schedule = deps.schedule ?? setTimeout
  const cancelSchedule = deps.cancelSchedule ?? clearTimeout
  const currentSource = () => deps.getSource ? deps.getSource() : source
  const session: BridgeSession = {
    sourceId: source.id, sock: null, seen: new Set(), outbound: new Set(), ownJids: new Set(),
    reconnectTimer: null, qrTimer: null, onRuntime: deps.onRuntime ?? (() => {}), getSource: currentSource, cancelSchedule,
    runtime: { sourceId: source.id, phase: 'connecting', qr: null, qrExpiresAt: null, phoneNumber: null, reconnectAttempt: 0, nextReconnectAt: null, lastError: null, updatedAt: Math.max(Date.now(), (runtimeStates.get(source.id)?.updatedAt ?? 0) + 1) },
  }
  sessions.set(source.id, session)
  publishRuntime(session, {})
  const active = () => sessions.get(source.id) === session && sourceLifecycleGeneration(source.id) === sourceGeneration && allBridgesGeneration === globalGeneration
  const assertActive = () => {
    assertBridgeStartActive(source.id, sourceGeneration, globalGeneration)
    if (!active() || !currentSource()?.enabled) throw new Error('WhatsApp connection was stopped or disabled.')
  }
  const clearQr = () => {
    if (session.qrTimer) cancelSchedule(session.qrTimer)
    session.qrTimer = null
    publishRuntime(session, { qr: null, qrExpiresAt: null })
  }
  const fail = (error: unknown) => {
    if (!active()) return
    const message = whatsappPairingErrorPatch(error).lastError
    publishRuntime(session, { phase: 'error', qr: null, qrExpiresAt: null, lastError: message ?? null, nextReconnectAt: null })
    onStatus({ lastError: message })
  }
  const connect = async (): Promise<void> => {
    assertActive()
    const B = await (deps.loadBaileys ?? loadBaileys)()
    assertActive()
    const auth = await authStateFactory(B, baseDir, source.id)
    assertActive()
    if (deps.requireRegistered && auth.state?.creds?.registered !== true) throw new Error('Saved WhatsApp credentials are not registered; link this phone again.')
    const { version } = await B.fetchLatestBaileysVersion()
    assertActive()
    const sock = B.makeWASocket({ version, auth: auth.state, printQRInTerminal: false, logger: silentBaileysLogger() })
    session.sock = sock
    const ownsSocket = () => active() && session.sock === sock
    sock.ev.on('creds.update', () => {
      if (!ownsSocket()) return
      void auth.saveCreds().then((credentialRotatedAt) => {
        if (ownsSocket() && credentialRotatedAt != null) onStatus({ credentialRotatedAt })
      }).catch((error: unknown) => {
        if (!ownsSocket()) return
        session.sock = null
        try { sock.end?.() } catch { /* best effort */ }
        fail(new Error(`WhatsApp credential update could not be protected on disk: ${String(error instanceof Error ? error.message : error).slice(0, 220)}`))
      })
    })
    const existingCredentialTime = auth.credentialUpdatedAt()
    if (existingCredentialTime != null && existingCredentialTime !== currentSource()?.credentialRotatedAt) onStatus({ credentialRotatedAt: existingCredentialTime })
    sock.ev.on('connection.update', (update: { qr?: string; connection?: string; lastDisconnect?: { error?: { output?: { statusCode?: number } } } }) => {
      if (!ownsSocket()) return
      if (update.qr) {
        clearQr()
        if (deps.requireRegistered) {
          session.sock = null
          try { sock.end?.() } catch { /* best effort */ }
          fail(new Error('Saved WhatsApp credentials need a new phone link.'))
          return
        }
        const qrExpiresAt = Date.now() + QR_LIFETIME_MS
        publishRuntime(session, { phase: 'pairing', qr: update.qr, qrExpiresAt, lastError: null })
        onQr(update.qr)
        onStatus({ lastQrAt: Date.now() })
        session.qrTimer = schedule(() => { if (ownsSocket()) clearQr() }, QR_LIFETIME_MS)
        session.qrTimer.unref?.()
      }
      if (update.connection === 'open') {
        clearQr()
        const phoneJid = normalizeWhatsAppJid(sock.user?.id)
        const lid = normalizeWhatsAppJid(sock.user?.lid)
        session.ownJids = new Set([phoneJid, lid].filter((jid): jid is string => Boolean(jid)))
        if (!session.ownJids.size) {
          session.sock = null
          try { sock.end?.() } catch { /* best effort */ }
          fail(new Error('WhatsApp did not provide an authenticated linked-account identity.'))
          return
        }
        const phoneNumber = phoneJid?.endsWith('@s.whatsapp.net') ? `+${phoneJid.split('@')[0]}` : null
        publishRuntime(session, { phase: 'connected', phoneNumber, lastError: null, nextReconnectAt: null })
        onStatus({ paired: true, phoneNumber, lastError: null, lastEventAt: Date.now() })
      }
      if (update.connection === 'close') {
        session.sock = null
        clearQr()
        const code = update.lastDisconnect?.error?.output?.statusCode
        if (code === B.DisconnectReason?.loggedOut || code === 401 || code === 403) {
          publishRuntime(session, { phase: 'disconnected', lastError: 'Logged out — link the phone again.', nextReconnectAt: null })
          onStatus({ paired: false, lastError: 'Logged out — link the phone again.' })
        } else if ([515, 410, 408, 428, 503].includes(code ?? -1) && session.runtime.reconnectAttempt < MAX_RECONNECTS && currentSource()?.enabled) {
          const attempt = session.runtime.reconnectAttempt + 1
          const delay = Math.min(30_000, 1000 * 2 ** (attempt - 1))
          publishRuntime(session, { phase: 'reconnecting', reconnectAttempt: attempt, nextReconnectAt: Date.now() + delay, lastError: 'Connection interrupted; reconnecting.' })
          onStatus({ lastError: 'Connection interrupted; reconnecting.' })
          session.reconnectTimer = schedule(() => {
            session.reconnectTimer = null
            if (!active()) return
            void connect().catch(fail)
          }, delay)
          session.reconnectTimer.unref?.()
        } else {
          fail(new Error(session.runtime.reconnectAttempt >= MAX_RECONNECTS ? 'Automatic reconnect limit reached; choose Reconnect to try again.' : `Connection closed (code ${code ?? 'unknown'}). Choose Reconnect.`))
        }
      }
    })
    const ingest = (messages: unknown[], historical: boolean) => {
      if (!ownsSocket() || session.runtime.phase !== 'connected') return
      // A linked-device replay is history, never a remote command or authorization.
      for (const raw of messages.slice(historical ? -300 : -500)) {
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const m = raw as any
          const jid = normalizeWhatsAppConversationJid(m?.key?.remoteJid)
          const fromMe = m?.key?.fromMe === true
          if (!jid || !selectedConversation(currentSource(), session, jid, fromMe)) continue
          const media = whatsappMediaDescriptor(m.message)
          const text = m.message?.conversation ?? m.message?.extendedTextMessage?.text ?? (media ? whatsappMediaCaption(m.message) : '')
          if (typeof text !== 'string' || (!text && !media)) continue
          const dedupId = m.key.id || `syn:${jid}:${m.messageTimestamp ?? ''}:${(text || media?.contentKey || '').length}`
          if (session.outbound.has(dedupId) || !markWhatsAppSeen(session, dedupId)) continue
          const base: WhatsAppMessageInput = {
            // Own PN/LID aliases have already passed selectedConversation against
            // sock.user identity. Store one canonical self-chat, never a raw LID flag.
            sourceId: source.id, jid: session.ownJids.has(jid) ? normalizeWhatsAppJid(sock.user?.id) ?? jid : jid, fromMe: false, text: text || '[media]',
            sender: fromMe ? 'You (linked phone)' : whatsappSenderFor(jid, m.pushName, m.key.participant),
            ts: Number(m.messageTimestamp) ? Number(m.messageTimestamp) * 1000 : Date.now(),
            messageId: dedupId, historical, selfChat: session.ownJids.has(jid),
            authenticatedAccountJid: normalizeWhatsAppJid(sock.user?.id) ?? undefined,
          }
          const stillAllowed = () => ownsSocket() && session.runtime.phase === 'connected' && selectedConversation(currentSource(), session, jid, fromMe)
          if (!media || historical) {
            if (stillAllowed()) onEvent(whatsappEventFromMessage(base))
          } else {
            void cacheWhatsAppMediaMessage(media, {
              cacheRoot: path.join(baseDir, 'cache'),
              shouldCache: stillAllowed,
              fetchDecryptedBytes: async () => {
                if (!stillAllowed()) throw new Error('WhatsApp contact scope changed.')
                return B.downloadMediaMessage(m, 'buffer', {})
              },
            }).then((mediaPath) => {
              if (stillAllowed()) onEvent(whatsappEventFromMessage({ ...base, mediaPaths: mediaPath ? [mediaPath] : [] }))
            }).catch(() => { if (stillAllowed()) onEvent(whatsappEventFromMessage(base)) })
          }
        } catch { /* malformed input is untrusted and cannot crash ingestion */ }
      }
    }
    sock.ev.on('messages.upsert', (payload: { messages?: unknown[]; type?: string }) => ingest(payload?.messages ?? [], payload?.type !== 'notify'))
    sock.ev.on('messaging-history.set', (payload: { messages?: unknown[] }) => ingest(payload?.messages ?? [], true))
  }
  try { await connect() } catch (error) { fail(error); throw error }
  return { ...source, updatedAt: Date.now() }
}

// Normalize a pairing failure into a renderer-facing status patch. The bridge starts
// fire-and-forget, so a failure (e.g. the bundled main can't dynamically import baileys)
// must be PUSHED to the renderer as a status — writing it only to settings leaves the live
// draft showing nothing, which is the "Start pairing does nothing" symptom. Pure + tested.
export function whatsappPairingErrorPatch(error: unknown): Partial<WhatsAppBridgeSource> {
  const raw = error instanceof Error ? error.message : String(error ?? '')
  const message = raw.trim() || 'WhatsApp pairing failed to start.'
  return { paired: false, lastError: message.slice(0, 300) }
}

export function stopWhatsAppBridge(sourceId: string): void {
  sourceLifecycleGenerations.set(sourceId, sourceLifecycleGeneration(sourceId) + 1)
  const session = sessions.get(sourceId)
  if (session) {
    sessions.delete(sourceId)
    if (session.reconnectTimer) session.cancelSchedule(session.reconnectTimer)
    if (session.qrTimer) session.cancelSchedule(session.qrTimer)
    const sock = session.sock
    session.sock = null
    publishRuntime(session, { phase: 'disconnected', qr: null, qrExpiresAt: null, nextReconnectAt: null, lastError: null })
    try { sock?.end?.() } catch { /* best effort */ }
  } else {
    const runtime = runtimeStates.get(sourceId)
    if (runtime) runtimeStates.set(sourceId, { ...runtime, phase: 'disconnected', qr: null, qrExpiresAt: null, nextReconnectAt: null, updatedAt: Math.max(Date.now(), runtime.updatedAt + 1) })
  }
}
export function stopAllWhatsAppBridges(): void {
  allBridgesGeneration++
  for (const id of [...sessions.keys()]) stopWhatsAppBridge(id)
}
/** Remove an account's non-secret ephemeral metadata after credential removal. */
export function forgetWhatsAppBridgeRuntime(sourceId: string): void { stopWhatsAppBridge(sourceId); runtimeStates.delete(sourceId) }
export async function sendWhatsAppMessage(sourceId: string, to: string, text: string): Promise<{ ok: boolean; sentId?: string; authenticatedAccountJid?: string }> {
  const session = sessions.get(sourceId)
  if (!session || session.runtime.phase !== 'connected' || !session.sock) throw new Error('WhatsApp bridge is not connected for this account.')
  const jid = normalizeWhatsAppConversationJid(to)
  if (!jid || typeof text !== 'string' || !text.trim() || text.length > 65_536) throw new Error('A valid WhatsApp recipient and bounded message text are required.')
  if (!selectedConversation(session.getSource(), session, jid, false)) throw new Error('This WhatsApp recipient is outside the account’s selected contacts or self-chat scope.')
  // Reserve the protocol id before sendMessage can echo, including synchronous self-chat echoes.
  const authenticatedAccountJid = normalizeWhatsAppJid(session.runtime.phoneNumber) ?? undefined
  const messageId = randomBytes(16).toString('hex').toUpperCase()
  markWhatsAppSeen({ seen: session.outbound }, messageId)
  const sent = await session.sock.sendMessage(jid, { text }, { messageId })
  if (sent?.key?.id) markWhatsAppSeen({ seen: session.outbound }, sent.key.id)
  return { ok: true, authenticatedAccountJid, ...(sent?.key?.id ? { sentId: sent.key.id } : {}) }
}
export function isWhatsAppBridgeReady(sourceId: string): boolean {
  const session = sessions.get(sourceId)
  return Boolean(session?.sock && session.runtime.phase === 'connected' && session.getSource()?.enabled)
}

export interface WhatsAppMessageInput {
  sourceId: string
  historical?: boolean
  selfChat?: boolean
  authenticatedAccountJid?: string
  jid: string
  fromMe: boolean
  text: string
  sender: string
  ts: number
  messageId?: string
  // Local cache paths of decrypted inbound media (image/document/audio/video) for this message,
  // so the vision tool can read an attachment by path. Empty/absent for a text-only message.
  mediaPaths?: string[]
}

// Attribute a message to its author. In a group (@g.us) the conversation JID is the GROUP,
// so the author must come from the participant (display name first, then the participant
// JID) — never the group JID, which would mislabel every member's message as the group.
export function whatsappSenderFor(jid: string, pushName: string | undefined, participant: string | undefined): string {
  if (jid.endsWith('@g.us')) return pushName || participant || jid
  return pushName || jid
}

export function whatsappEventFromMessage(input: WhatsAppMessageInput, now = Date.now()): ConnectorEvent {
  const messageId = input.messageId?.trim() || ''
  const mediaPaths = (input.mediaPaths ?? []).filter((p) => typeof p === 'string' && p.length > 0)
  // payloadKeys advertises which payload fields are present (messageId for dedup, mediaPaths so
  // the assistant/vision tool knows a decrypted local attachment is attached).
  const payloadKeys = ['text', 'jid']
  if (messageId) payloadKeys.push('messageId')
  if (mediaPaths.length) payloadKeys.push('mediaPaths')
  return {
    id: randomUUID(),
    connectorId: 'whatsapp',
    pluginName: 'WhatsApp',
    action: input.fromMe ? 'message_sent' : 'message_received',
    direction: input.fromMe ? 'outbound' : 'inbound',
    classification: 'communication',
    priority: 'normal',
    trustLevel: 'notify',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: null,
    resolvedBy: null,
    title: input.sender,
    summary: input.text.slice(0, 220),
    actor: input.sender,
    threadKey: `whatsapp:${input.sourceId}:${input.jid}`,
    // messageId rides in the payload so rememberConnectorEvents can dedup replays by it.
    payloadKeys,
    payload: { text: input.text, jid: input.jid, fromMe: input.fromMe, historical: input.historical === true, selfChat: input.selfChat === true, remoteCommandEligible: false, ...(input.authenticatedAccountJid ? { authenticatedAccountJid: input.authenticatedAccountJid } : {}), ...(messageId ? { messageId } : {}) },
    // Decrypted inbound media lands here so the vision tool reads the image/PDF by local path.
    ...(mediaPaths.length ? { mediaPaths } : {}),
    sources: [{ id: `whatsapp:${input.sourceId}:${input.jid}`, connectorId: 'whatsapp', title: input.sender, createdAt: input.ts }],
    createdAt: now,
  }
}
