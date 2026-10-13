import { randomUUID } from 'node:crypto'
import path from 'node:path'
import type { ConnectorEvent } from './types'
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

export interface WhatsAppBridgeSource {
  id: string
  label: string
  enabled: boolean
  /** Phone number (jid), filled in after first pairing. */
  phoneNumber: string | null
  /** Whether the bridge has completed QR pairing and holds valid credentials. */
  paired: boolean
  lastQrAt: number | null
  lastEventAt: number | null
  lastError: string | null
  /** Server-authored age marker for the durable linked-device credential. */
  credentialRotatedAt?: number | null
  createdAt: number
  updatedAt: number
}

export interface WhatsAppBridgeSettings {
  sources: WhatsAppBridgeSource[]
  lastSyncAt: number | null
}

export function makeWhatsAppBridgeSource(label: string, now = Date.now()): WhatsAppBridgeSource {
  return {
    id: `wa_${randomUUID().slice(0, 8)}`,
    label: label || 'WhatsApp account',
    enabled: true,
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

// A live bridge session: one per paired account. Held in-process only.
interface BridgeSession {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sock: any
  sourceId: string
  // Per-session seen message ids: baileys re-delivers messages.upsert on reconnect, and the
  // store-level dedup only holds within a rolling 500-event window — so a replay of an older
  // message would slip through. This unbounded-but-trimmed set dedups at the source.
  seen: Set<string>
}

const SEEN_CAP = 5000

const sessions = new Map<string, BridgeSession>()
const sourceLifecycleGenerations = new Map<string, number>()
let allBridgesGeneration = 0

function sourceLifecycleGeneration(sourceId: string): number {
  return sourceLifecycleGenerations.get(sourceId) ?? 0
}

function assertBridgeStartActive(sourceId: string, sourceGeneration: number, globalGeneration: number): void {
  if (sourceLifecycleGeneration(sourceId) !== sourceGeneration || allBridgesGeneration !== globalGeneration) {
    throw new Error('WhatsApp bridge start was cancelled because its credential session was stopped or purged.')
  }
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

// Start (or resume) a bridge for one account. Emits a QR code string for pairing
// (the renderer shows it as a scannable code), and `onEvent` for each inbound
// message. Returns the current source so callers can persist status.
export async function startWhatsAppBridge(
  source: WhatsAppBridgeSource,
  baseDir: string,
  onQr: (qr: string) => void,
  onEvent: (event: ConnectorEvent) => void,
  onStatus: (patch: Partial<WhatsAppBridgeSource>) => void,
  authStateFactory?: WhatsAppAuthStateFactory,
): Promise<WhatsAppBridgeSource> {
  if (!authStateFactory) {
    throw new Error('WhatsApp QR pairing is disabled because a protected credential store is unavailable. Use the signed WhatsApp Cloud connector instead.')
  }
  const sourceGeneration = sourceLifecycleGeneration(source.id)
  const globalGeneration = allBridgesGeneration
  const B = await loadBaileys()
  assertBridgeStartActive(source.id, sourceGeneration, globalGeneration)
  const { DisconnectReason, makeWASocket, fetchLatestBaileysVersion } = B
  const auth = await authStateFactory(B, baseDir, source.id)
  assertBridgeStartActive(source.id, sourceGeneration, globalGeneration)
  const { state, saveCreds } = auth
  const { version } = await fetchLatestBaileysVersion()
  assertBridgeStartActive(source.id, sourceGeneration, globalGeneration)

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sock = makeWASocket({
    version,
    auth: state,
    printQRInTerminal: false,
    // Full pino-shaped silent logger — baileys calls logger.child()/level methods, so a bare
    // `{ level: 'silent' }` throws "logger.child is not a function" and kills pairing pre-QR.
    logger: silentBaileysLogger(),
  })
  // Reuse the existing seen-set across a reconnect (515/410 re-invokes this function): a
  // fresh Set() would defeat dedup for exactly the reconnect-replay case it exists for.
  sessions.set(source.id, { sock, sourceId: source.id, seen: sessions.get(source.id)?.seen ?? new Set<string>() })

  // Baileys mutates `state.creds` before this event. Persist that real credential transition
  // first, then stamp its server-authored age marker. Status-only edits never call this path.
  sock.ev.on('creds.update', () => {
    if (sessions.get(source.id)?.sock !== sock) return
    void saveCreds()
      .then((credentialRotatedAt) => {
        if (credentialRotatedAt != null) onStatus({ credentialRotatedAt })
      })
      .catch((error) => {
        try { sock.end?.() } catch { /* best effort */ }
        onStatus({
          paired: false,
          lastError: `WhatsApp credential update could not be protected on disk: ${String(error instanceof Error ? error.message : error).slice(0, 220)}`,
        })
      })
  })

  // Recover the age marker from protected state or a safely-migrated legacy creds mtime.
  const existingCredentialTime = auth.credentialUpdatedAt()
  if (existingCredentialTime != null && existingCredentialTime !== source.credentialRotatedAt) {
    onStatus({ credentialRotatedAt: existingCredentialTime })
  }

  sock.ev.on('connection.update', (update: { qr?: string; connection?: string; lastDisconnect?: { error?: { output?: { statusCode?: number } } } }) => {
    // A stopped/replaced socket may emit one final close/update event. It no longer owns this
    // source and must not publish status or launch a reconnect that recreates wiped auth state.
    if (sessions.get(source.id)?.sock !== sock) return
    if (update.qr) {
      onQr(update.qr)
      onStatus({ lastQrAt: Date.now() })
    }
    if (update.connection === 'open') {
      onStatus({ paired: true, lastError: null, lastEventAt: Date.now() })
    }
    if (update.connection === 'close') {
      const code = update.lastDisconnect?.error?.output?.statusCode
      // 515 / 410 are reconnectable; 401/403 mean the auth is invalid (logged out).
      if (code === DisconnectReason.loggedOut) {
        onStatus({ paired: false, lastError: 'Logged out — re-pair by scanning the QR again.' })
      } else if (code === 515 || code === 410) {
        // Reconnect transiently.
        sessions.delete(source.id)
        void startWhatsAppBridge(source, baseDir, onQr, onEvent, onStatus, authStateFactory).catch(() => {})
      } else {
        onStatus({ lastError: `Connection closed (code ${code ?? 'unknown'}).` })
      }
    }
  })

  // ~/.vontra/cache — the shared media cache the vision tool reads decrypted attachments from.
  const mediaCacheRoot = path.join(baseDir, 'cache')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sock.ev.on('messages.upsert', (payload: { messages: Array<any> }) => {
    const session = sessions.get(source.id)
    if (!session || session.sock !== sock) return
    for (const m of payload.messages ?? []) {
      if (m.key.fromMe) continue // skip our own outbound messages — don't re-ingest them as inbound
      // Media (image/document/audio/video) arrives ENCRYPTED; detect it so a caption-less photo
      // still ingests (the old `if (!text) continue` dropped media-only messages entirely).
      const media = whatsappMediaDescriptor(m.message)
      const text = m.message?.conversation ?? m.message?.extendedTextMessage?.text ?? (media ? whatsappMediaCaption(m.message) : '')
      if (!text && !media) continue
      const jid = m.key.remoteJid
      // Stable dedup id: prefer the WhatsApp message id; synthesize a deterministic one when
      // it's absent (rare) so a no-id message still dedups across replays at both layers.
      const dedupId = m.key.id || `syn:${jid}:${m.messageTimestamp ?? ''}:${(text || media?.contentKey || '').length}`
      // Drop replays at the source (reconnect re-delivers), regardless of the store cap.
      if (session && !markWhatsAppSeen(session, dedupId)) continue
      const base = {
        sourceId: source.id,
        jid,
        fromMe: Boolean(m.key.fromMe),
        text,
        sender: whatsappSenderFor(jid, m.pushName, m.key.participant),
        ts: Number(m.messageTimestamp) ? Number(m.messageTimestamp) * 1000 : Date.now(),
        messageId: dedupId,
      }
      if (!media) {
        onEvent(whatsappEventFromMessage(base))
        continue
      }
      // Media message: download (Baileys decrypts the CDN blob internally via downloadMediaMessage)
      // and write the decrypted bytes to the media cache, then emit with the local path attached.
      // Best-effort + async so the download never blocks the socket; a failed decrypt still emits
      // the message as text (the caption) so nothing is silently dropped.
      void cacheWhatsAppMediaMessage(media, {
        cacheRoot: mediaCacheRoot,
        fetchDecryptedBytes: async () => B.downloadMediaMessage(m, 'buffer', {}),
      })
        .then((mediaPath) => {
          onEvent(whatsappEventFromMessage({ ...base, mediaPaths: mediaPath ? [mediaPath] : [] }))
        })
        .catch(() => {
          onEvent(whatsappEventFromMessage(base)) // download/decrypt failed → still ingest the text
        })
    }
  })

  // HISTORY BACKFILL: on link, WhatsApp delivers a recent-chats snapshot. The live messages.upsert
  // above only captures messages that arrive AFTER pairing, so without this a freshly-paired account
  // shows zero threads until someone messages you. Backfill inbound text messages from the snapshot
  // through the same dedup + event path (text-only — no media-download burst). Fully guarded so a
  // malformed history payload can never crash the bridge; the per-message dedup set prevents doubles
  // when a backfilled message is later re-delivered live.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sock.ev.on('messaging-history.set', (payload: { messages?: Array<any> }) => {
    const session = sessions.get(source.id)
    if (!session || session.sock !== sock) return
    for (const m of (payload?.messages ?? []).slice(-300)) {
      try {
        if (!m?.key || m.key.fromMe) continue
        const media = whatsappMediaDescriptor(m.message)
        const text = m.message?.conversation ?? m.message?.extendedTextMessage?.text ?? (media ? whatsappMediaCaption(m.message) : '')
        if (!text && !media) continue
        const jid = m.key.remoteJid
        const dedupId = m.key.id || `syn:${jid}:${m.messageTimestamp ?? ''}:${(text || media?.contentKey || '').length}`
        if (session && !markWhatsAppSeen(session, dedupId)) continue
        onEvent(whatsappEventFromMessage({
          sourceId: source.id, jid, fromMe: false, text: text || '[media]',
          sender: whatsappSenderFor(jid, m.pushName, m.key.participant),
          ts: Number(m.messageTimestamp) ? Number(m.messageTimestamp) * 1000 : Date.now(),
          messageId: dedupId,
        }))
      } catch { /* skip a malformed history item */ }
    }
  })

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
    // Revoke ownership before end(): some socket implementations synchronously emit a final
    // connection.update from end(), and that callback must see itself as stale.
    sessions.delete(sourceId)
    try { session.sock.end?.() } catch { /* best effort */ }
  }
}

export function stopAllWhatsAppBridges(): void {
  allBridgesGeneration++
  for (const id of [...sessions.keys()]) stopWhatsAppBridge(id)
}

// Send a text message on a paired account. Only call after approval.
export async function sendWhatsAppMessage(sourceId: string, to: string, text: string): Promise<{ ok: boolean }> {
  const session = sessions.get(sourceId)
  if (!session) throw new Error('WhatsApp bridge is not connected for this account.')
  const jid = to.includes('@') ? to : `${to}@s.whatsapp.net`
  await session.sock.sendMessage(jid, { text })
  return { ok: true }
}

export function isWhatsAppBridgeReady(sourceId: string): boolean {
  return sessions.has(sourceId)
}

export interface WhatsAppMessageInput {
  sourceId: string
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
    payload: { text: input.text, jid: input.jid, fromMe: input.fromMe, ...(messageId ? { messageId } : {}) },
    // Decrypted inbound media lands here so the vision tool reads the image/PDF by local path.
    ...(mediaPaths.length ? { mediaPaths } : {}),
    sources: [{ id: `whatsapp:${input.sourceId}:${input.jid}`, connectorId: 'whatsapp', title: input.sender, createdAt: input.ts }],
    createdAt: now,
  }
}
