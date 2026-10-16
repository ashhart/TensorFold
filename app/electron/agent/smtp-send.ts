import nodemailer from 'nodemailer'
import type { ConnectorEvent, EmailSourceConfig } from './types'
import { createPinnedSocketFactory, resolvePublicMailHost } from './mail-network'

// SMTP send for approved email drafts. When an approved send/reply event was
// composed against an IMAP account (kind:'imap'), we send it over SMTP using
// that account's credentials — so a reply actually leaves the inbox, not just
// the local outbox. Falls back to the existing local-outbox path for file sources.

export interface SmtpSendResult {
  transport: 'smtp'
  messageId: string
  envelope: { to: string; from: string }
  deliveredAt: number
}

// Resolve the SMTP submission port + TLS mode from an email account config.
// The account's `port` field is populated by the UI for IMAP (default 993, the
// implicit-TLS IMAPS port; 143 is plaintext IMAP). Those ports do NOT speak SMTP,
// so reusing them silently hangs every send. We therefore treat the known IMAP
// ports as "no SMTP port set" and fall back to 587 (STARTTLS submission). Only a
// genuinely SMTP-shaped port (anything that isn't an IMAP port) is honored as an
// explicit override. `secure` follows the actual SMTP port: 465 => implicit TLS,
// 587/25 => STARTTLS (secure:false, upgraded after EHLO).
export function resolveSmtpTransport(account: Pick<EmailSourceConfig, 'port'>): { port: number; secure: boolean } {
  const IMAP_PORTS = new Set([993, 143])
  const configured = account.port
  const port = configured == null || IMAP_PORTS.has(configured) ? 587 : configured
  return { port, secure: port === 465 }
}

// Looks-like-an-email check (intentionally loose): a single '@' with a non-empty
// local part and a dotted domain. Used to decide whether the login `user` is itself
// usable as a From address.
function looksLikeEmail(value: string | null | undefined): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(value ?? '').trim())
}

// Resolve the From / envelope-from address for an outgoing message. Prefer the
// account's explicit `fromAddress`; otherwise fall back to the login `user` only when
// it itself parses as an email address. A login that is NOT an email (e.g. 'maya' or a
// domain\login) would produce an invalid envelope the MTA rejects, so we surface that
// as a clear error instead of sending a broken From.
export function resolveFromAddress(account: Pick<EmailSourceConfig, 'fromAddress' | 'user'>): string {
  const explicit = account.fromAddress?.trim()
  if (explicit && looksLikeEmail(explicit)) return explicit
  if (explicit) {
    throw new Error(`Configured email From address "${explicit}" is not a valid email address.`)
  }
  const user = account.user?.trim()
  if (user && looksLikeEmail(user)) return user
  throw new Error('Cannot determine a valid From address for this account: set the account email (fromAddress).')
}

// Resolve the References header value. `inReplyTo` (when present) is a real
// Message-ID resolved from the original message we're replying to — that's the
// correct value for References (same as In-Reply-To). `threadKey` is only an
// internal connector navigational key (e.g. 'email:abc123'), never a Message-ID,
// so it must never be emitted as an RFC 5322 References value. If no real source
// Message-ID was captured, omit the header.
export function resolveReferences(inReplyTo: string | undefined, _threadKey: string | undefined): string | undefined {
  return inReplyTo
}

// Send an approved email event via the IMAP account's SMTP server. Throws on
// failure so the host can record the error. Only call after approvalStatus==='approved'.
export async function sendEmailOverSmtp(event: ConnectorEvent, account: EmailSourceConfig): Promise<SmtpSendResult> {
  if (event.approvalStatus !== 'approved') throw new Error('Email send events must be approved before SMTP transport.')
  if (!account.host || !account.user || !account.password) throw new Error('IMAP account is missing SMTP credentials.')
  const host = account.host.trim()
  // Resolve the SMTP port + TLS mode. The account's `port` is the IMAP port (993/143
  // for standard providers); those can't carry SMTP, so resolveSmtpTransport falls
  // back to 587 rather than hanging on an IMAPS socket. We resolve + reject private hosts.
  const { port, secure } = resolveSmtpTransport(account)
  // Resolve a valid From/envelope-from. A login username that isn't a full email
  // address would produce a broken envelope, so this throws rather than send one.
  const from = resolveFromAddress(account)
  const target = await resolvePublicMailHost(host, 'SMTP')

  const payload = (event.payload ?? {}) as Record<string, unknown>
  const to = String(payload.to ?? '')
  const subject = String(payload.subject ?? '')
  const body = String(payload.body ?? '')
  const inReplyTo = payload.inReplyTo ? String(payload.inReplyTo) : undefined
  const threadKey = payload.threadKey ? String(payload.threadKey) : undefined
  if (!to || !subject || !body) throw new Error('Approved email event is missing to/subject/body.')

  const transporter = nodemailer.createTransport({
    host: target.hostname,
    port,
    secure,
    // Nodemailer runs its own resolve4/resolve6 pipeline before connecting, so a tls.lookup
    // option alone is insufficient. Preconnect through our vetted lookup and hand that exact
    // socket to the transport; implicit TLS/STARTTLS then retains the original hostname for
    // certificate verification and SNI.
    getSocket: createPinnedSocketFactory(target, port) as any,
    requireTLS: !secure,
    tls: { servername: target.hostname },
    auth: { user: account.user, pass: account.password },
  })

  try {
    const info = await transporter.sendMail({
      from,
      to,
      subject,
      text: body,
      inReplyTo,
      references: resolveReferences(inReplyTo, threadKey),
    })
    return {
      transport: 'smtp',
      messageId: info.messageId,
      envelope: { to, from },
      deliveredAt: Date.now(),
    }
  } finally {
    transporter.close()
  }
}
