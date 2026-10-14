import { ImapFlow } from 'imapflow'
import type { EmailAttachment, EmailReadResult, EmailSourceConfig } from './types'
import { imapUidFloor } from './connector-sync'
import { mailPinnedLookup, resolvePublicMailHost } from './mail-network'

// Cap how many attachment parts we pull per message (a mailbomb with 200 parts shouldn't
// trigger 200 downloads) and the per-part byte ceiling we ask the server to stream.
const MAX_ATTACHMENTS_PER_MESSAGE = 10
const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

// Multi-account IMAP fetch. Each `kind: 'imap'` email source is one account
// (host/port/user/password); we open a short-lived connection, fetch recent
// envelopes + bodies, and close. No state is held between syncs.

const MAX_PER_MAILBOX = 200

export function isImapSource(source: EmailSourceConfig): boolean {
  return source.kind === 'imap'
}

export function validateImapSource(source: EmailSourceConfig): string | null {
  if (!source.host?.trim()) return 'IMAP host is required.'
  if (!source.user?.trim()) return 'IMAP username is required.'
  if (!source.password) return 'IMAP password is required.'
  return null
}

async function createPinnedImapClient(source: EmailSourceConfig): Promise<ImapFlow> {
  const host = source.host!.trim()
  const secure = source.tls !== false
  const target = await resolvePublicMailHost(host, 'IMAP')
  return new ImapFlow({
    host: target.hostname,
    port: source.port ?? (secure ? 993 : 143),
    secure,
    // ImapFlow merges this object into both net.connect and tls.connect. The lookup can
    // return only the addresses vetted above, so authentication cannot race a second DNS
    // answer. STARTTLS reuses the same already-pinned socket.
    tls: { lookup: mailPinnedLookup(target) as any },
    // A password must never cross a plaintext IMAP connection. When implicit TLS is off,
    // require STARTTLS instead of accepting an opportunistic downgrade.
    doSTARTTLS: secure ? undefined : true,
    auth: { user: source.user!, pass: source.password! },
    logger: false,
    emitLogs: false,
  })
}

// Fetch messages from one IMAP account into the shared EmailMessage shape.
//
// `incremental` opts into INCREMENTAL fetch: only messages with UID strictly greater
// than the source cursor's lastImapUid are transferred (UID range `floor+1:*`). This is
// for the SYNC path, which advances the cursor each poll. Read paths (list/summary/thread)
// pass it off (the default) so they always see the full recent window, independent of how
// far sync has advanced. With no cursor (or after a reset) both modes fall back to the
// most-recent-N window.
export async function fetchImapMessages(source: EmailSourceConfig, opts: { incremental?: boolean; withAttachments?: boolean } = {}): Promise<EmailReadResult[]> {
  const validation = validateImapSource(source)
  if (validation) throw new Error(validation)
  const client = await createPinnedImapClient(source)
  const out: EmailReadResult[] = []
  try {
    await client.connect()
    const mailbox = source.mailbox?.trim() || 'INBOX'
    const lock = await client.getMailboxLock(mailbox)
    try {
      // INCREMENTAL (sync only): when the source carries a UID high-water mark, fetch ONLY
      // messages with UID strictly greater than it (UID range `floor+1:*`, addressed by
      // UID), so a sync transfers just the new mail instead of the whole recent window
      // every poll. Read paths leave incremental off so they always see the recent window.
      // Without a cursor (or after a reset) fall back to the most-recent-N sequence range.
      const floor = opts.incremental ? imapUidFloor(source.cursor) : 0
      if (floor > 0) {
        for await (const msg of client.fetch(`${floor + 1}:*`, { uid: true, envelope: true, internalDate: true, flags: true, source: true, headers: true, bodyStructure: true }, { uid: true })) {
          // A `floor+1:*` UID range can echo the boundary message when nothing is newer;
          // drop anything at/below the floor so the cursor never re-ingests it.
          if (typeof msg.uid === 'number' && msg.uid <= floor) continue
          const message = imapMessageToEmailMessage(source, msg)
          if (message) {
            if (opts.withAttachments) message.attachments = await fetchMessageAttachments(client, msg)
            out.push(message)
          }
        }
      } else {
        // Full-resync fallback: most recent N by sequence range (newest first).
        const status = await client.status(mailbox, { messages: true })
        const total = status.messages ?? 0
        if (total === 0) return []
        const from = Math.max(1, total - MAX_PER_MAILBOX + 1)
        for await (const msg of client.fetch(`${from}:${total}`, { uid: true, envelope: true, internalDate: true, flags: true, source: true, headers: true, bodyStructure: true })) {
          const message = imapMessageToEmailMessage(source, msg)
          if (message) {
            if (opts.withAttachments) message.attachments = await fetchMessageAttachments(client, msg)
            out.push(message)
          }
        }
      }
    } finally {
      lock.release()
    }
  } finally {
    await client.logout().catch(() => {})
  }
  // Newest first, matching the file-source ordering the rest of the connector expects.
  return out.sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
}

// Read a single message by its IMAP UID/path id. id is `<mailbox>:<uid>` set in
// fetchImapMessages; we reconnect, open the mailbox, and fetch that UID.
export async function readImapMessage(source: EmailSourceConfig, id: string): Promise<EmailReadResult | null> {
  const { mailbox, uid } = parseImapId(id)
  if (uid == null) return null
  const client = await createPinnedImapClient(source)
  try {
    await client.connect()
    const lock = await client.getMailboxLock(mailbox || 'INBOX')
    try {
      // Fetch by IMAP UID (not sequence number): ids are built as `<mailbox>:<uid>`
      // from msg.uid, and uid != seq in any non-trivial mailbox. Without { uid: true }
      // imapflow defaults to sequence-number addressing and returns the wrong message.
      for await (const msg of client.fetch(uid.toString(), { envelope: true, internalDate: true, flags: true, source: true, headers: true, bodyStructure: true }, { uid: true })) {
        return imapMessageToEmailMessage(source, msg) ?? null
      }
    } finally {
      lock.release()
    }
  } finally {
    await client.logout().catch(() => {})
  }
  return null
}

export function parseImapId(id: string): { mailbox: string; uid: number | null } {
  const sep = id.lastIndexOf(':')
  if (sep < 0) return { mailbox: 'INBOX', uid: null }
  const mailbox = id.slice(0, sep)
  const uid = Number(id.slice(sep + 1))
  return { mailbox, uid: Number.isFinite(uid) ? uid : null }
}

// ---- Attachment MIME-part extraction (pure) -----------------------------------

// The metadata of one attachment part to download (no bytes yet) — what extractAttachmentParts
// returns from a BODYSTRUCTURE so the host knows which parts to fetch.
export interface AttachmentPartRef {
  part: string
  filename: string | null
  contentType: string | null
  size: number | null
}

// A minimal shape of imapflow's MessageStructureObject (BODYSTRUCTURE node) — only the fields
// the extractor reads. Kept local + loose so the pure walk is unit-testable without imapflow.
export interface BodyStructureNode {
  part?: string
  type?: string
  encoding?: string
  size?: number
  id?: string
  parameters?: Record<string, unknown>
  disposition?: string
  dispositionParameters?: Record<string, unknown>
  childNodes?: BodyStructureNode[]
}

/**
 * Walk a parsed BODYSTRUCTURE and return the real attachment leaf parts to download. PURE (no
 * I/O), so it is the unit-tested core of the email-attachment path. The rules, in order:
 *
 *   • Multipart container nodes (type `multipart/...`, or any node carrying childNodes) are not
 *     downloadable themselves — recurse into their children instead.
 *   • A node with `disposition: attachment` is always an attachment.
 *   • A node with a filename (Content-Disposition filename or Content-Type name) is an attachment
 *     even if the disposition header was absent/inline (many clients send inline images this way).
 *   • Otherwise an INLINE BODY part — `text/plain` / `text/html` with no filename — is SKIPPED:
 *     that is the message body the connector already renders, not an attachment.
 *   • A node WITHOUT a `part` id can't be fetched, so it is skipped.
 *
 * Parts are returned in document order and capped (maxParts) so a part-bomb can't fan out.
 */
export function extractAttachmentParts(node: BodyStructureNode | null | undefined, maxParts = MAX_ATTACHMENTS_PER_MESSAGE): AttachmentPartRef[] {
  const out: AttachmentPartRef[] = []
  const walk = (n: BodyStructureNode | null | undefined): void => {
    if (!n || out.length >= maxParts) return
    const children = Array.isArray(n.childNodes) ? n.childNodes : null
    const type = String(n.type ?? '').toLowerCase()
    if (children || type.startsWith('multipart/')) {
      for (const child of children ?? []) walk(child)
      return
    }
    const filename = attachmentFilename(n)
    const disposition = String(n.disposition ?? '').toLowerCase().trim()
    const isAttachment = disposition === 'attachment' || Boolean(filename)
    if (!isAttachment) return // inline body part (text/plain, text/html, …) — not an attachment
    if (!n.part) return // unfetchable without a part id
    out.push({ part: n.part, filename, contentType: type || null, size: typeof n.size === 'number' ? n.size : null })
  }
  walk(node)
  return out
}

// Filename from a BODYSTRUCTURE node: Content-Disposition `filename` wins, else Content-Type
// `name`. Returns null when neither is present. Pure.
function attachmentFilename(n: BodyStructureNode): string | null {
  const fromDisp = stringParam(n.dispositionParameters, 'filename')
  if (fromDisp) return fromDisp
  const fromType = stringParam(n.parameters, 'name')
  if (fromType) return fromType
  return null
}

function stringParam(params: Record<string, unknown> | undefined, key: string): string | null {
  if (!params) return null
  const v = params[key]
  return typeof v === 'string' && v.trim() ? v.trim() : null
}

// imapflow returns message flags as a Set<string>, but other shapes (array, missing) can
// appear. has() exists only on Set and includes() only on arrays, so probe by shape rather
// than calling one blindly — calling .includes() on a Set threw and aborted the whole fetch
// on the first unread message.
export function isUnreadFromFlags(flags: unknown): boolean {
  if (flags instanceof Set) return !flags.has('\\Seen')
  if (Array.isArray(flags)) return !flags.includes('\\Seen')
  return true // no/unknown flags => treat as unread
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function imapMessageToEmailMessage(source: EmailSourceConfig, msg: any): EmailReadResult | null {
  if (!msg) return null
  const env = msg.envelope
  const from = env?.from?.[0] ? addressToString(env.from[0]) : ''
  const to = (env?.to ?? []).map(addressToString).join(', ')
  const subject = env?.subject ?? '(no subject)'
  const date = env?.date ? new Date(env.date).getTime() : (msg.internalDate ? new Date(msg.internalDate).getTime() : Date.now())
  const unread = isUnreadFromFlags(msg.flags)
  const uid = msg.uid
  const mailbox = source.mailbox?.trim() || 'INBOX'
  // Body: prefer the parsed text part if available; otherwise decode the raw source.
  let body = ''
  let size = 0
  if (typeof msg.source === 'object' && msg.source) {
    try { body = msg.source.toString('utf8'); size = msg.source.length } catch { body = '' }
  }
  const headers = msg.headers ? headersToObject(msg.headers) : {}
  if (env?.messageId) headers['message-id'] = String(env.messageId)
  const inReplyToRaw = env?.inReplyTo ?? headers['in-reply-to'] ?? ''
  return {
    id: `${mailbox}:${uid}`,
    sourceId: source.id,
    sourceLabel: source.label,
    path: `imap://${source.host}/${mailbox}/${uid}`,
    subject,
    from,
    to,
    date,
    unread,
    snippet: body.replace(/\s+/g, ' ').trim().slice(0, 200),
    size,
    messageId: env?.messageId ? String(env.messageId).trim().replace(/^<|>$/g, '').toLowerCase() : (headers['message-id'] ? String(headers['message-id']).trim().replace(/^<|>$/g, '').toLowerCase() : null),
    inReplyTo: String(inReplyToRaw).trim().replace(/^<|>$/g, '').toLowerCase() || null,
    body: body.slice(0, 20_000),
    headers,
  }
}

// Download the attachment parts of one already-fetched message over the OPEN connection (the
// mailbox lock is held by the caller). Best-effort: any failure (a single bad part or the whole
// download) yields [] so a bad attachment never blocks ingesting the message. The bytes come
// back already transfer-decoded (imapflow decodes base64/quoted-printable in downloadMany).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function fetchMessageAttachments(client: any, msg: any): Promise<EmailAttachment[]> {
  try {
    const refs = extractAttachmentParts(msg?.bodyStructure as BodyStructureNode | undefined)
    if (!refs.length || typeof msg?.uid !== 'number') return []
    // downloadMany(range, parts, { uid, maxBytes }) → { [part]: { content: Buffer, meta } }.
    const data = await client.downloadMany(msg.uid, refs.map((r) => r.part), { uid: true, maxBytes: MAX_ATTACHMENT_BYTES }).catch(() => null)
    if (!data || data.response === false) return []
    const out: EmailAttachment[] = []
    for (const ref of refs) {
      const entry = data[ref.part]
      const content = entry?.content
      if (!content || !(content instanceof Uint8Array) || !content.byteLength) continue
      out.push({
        part: ref.part,
        // Prefer the decoded filename/content-type from the part's MIME headers when present.
        filename: (typeof entry?.meta?.filename === 'string' && entry.meta.filename) || ref.filename,
        contentType: (typeof entry?.meta?.contentType === 'string' && entry.meta.contentType) || ref.contentType,
        size: ref.size,
        data: content,
      })
    }
    return out
  } catch {
    return []
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function addressToString(addr: any): string {
  if (!addr) return ''
  if (typeof addr === 'string') return addr
  const name = [addr.name, addr.surname].filter(Boolean).join(' ')
  const email = addr.address || ''
  return name ? `${name} <${email}>` : email
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function headersToObject(headers: any): Record<string, string> {
  const out: Record<string, string> = {}
  if (!headers) return out
  // With `headers: true` imapflow returns the raw header block as a Buffer
  // (FetchMessageObject.headers?: Buffer). Treating it as a plain object yields
  // byte-index keys, so parse it as RFC 5322 text: unfold continuations and key
  // by lowercased header name. A string is parsed the same way.
  if (Buffer.isBuffer(headers) || typeof headers === 'string') {
    const text = Buffer.isBuffer(headers) ? headers.toString('utf8') : headers
    // Unfold folded headers: a line starting with SP/TAB continues the previous one.
    const unfolded = text.replace(/\r\n/g, '\n').replace(/\n[ \t]+/g, ' ')
    for (const line of unfolded.split('\n')) {
      const sep = line.indexOf(':')
      if (sep <= 0) continue
      const key = line.slice(0, sep).trim().toLowerCase()
      const value = line.slice(sep + 1).trim()
      if (!key) continue
      // Preserve repeated headers (e.g. Received) by joining rather than overwriting.
      out[key] = key in out ? `${out[key]}, ${value}` : value
    }
    return out
  }
  // A Map-like object keyed by header name.
  if (typeof headers.get === 'function') {
    for (const key of Object.keys(headers)) out[key.toLowerCase()] = String(headers[key])
    return out
  }
  for (const [key, value] of Object.entries(headers)) out[key.toLowerCase()] = String(value)
  return out
}
