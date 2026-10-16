// Pairing + device sessions (VST-035): turn the Orchestrator's single shared bearer token
// into per-device paired sessions, so "local becomes a special case of remote" — a laptop,
// phone, or browser each pairs once and gets its own revocable session token: a one-time
// pairing credential becomes a durable per-device session that can be revoked.
//
// Pure + dependency-injected: the clock and id/token generators are passed in, so tests are
// deterministic and no crypto/time is read at module scope. Production defaults use Node's
// CSPRNG; injected generators keep tests deterministic. The registry snapshots to/from plain
// state for persistence.

import { randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'

export interface PairingLink {
  // The one-time credential a client presents to redeem a session. Never persisted to the
  // renderer/list views.
  credential: string
  label: string
  createdAt: number
  expiresAt: number
  consumedAt: number | null
}

export interface DeviceSession {
  id: string
  label: string
  // The bearer token this device sends on every request. Secret — never in a list view.
  token: string
  createdAt: number
  lastSeenAt: number
  revokedAt: number | null
  clientInfo?: string | null
}

// Renderer/API-safe projection: identifies a session without exposing its token.
export interface DeviceSessionView {
  id: string
  label: string
  createdAt: number
  lastSeenAt: number
  revoked: boolean
  clientInfo?: string | null
}

export interface PairingRegistryState {
  links: PairingLink[]
  sessions: DeviceSession[]
}

export interface PairingRegistryOptions {
  now?: () => number
  newId?: () => string
  newToken?: () => string
  // Pairing links are short-lived (default 10 min): a credential unused past this is dead.
  linkTtlMs?: number
}

export interface RedeemResult {
  ok: boolean
  session?: DeviceSession
  error?: 'unknown_credential' | 'already_consumed' | 'expired'
}

function safeEqual(a: string, b: string): boolean {
  if (!a || !b) return false
  const ba = Buffer.from(a)
  const bb = Buffer.from(b)
  if (ba.length !== bb.length || ba.length === 0) return false
  return timingSafeEqual(ba, bb)
}

function randomDeviceId(): string {
  return `dev_${randomUUID()}`
}

function randomBearerToken(): string {
  // 256 bits of CSPRNG output. base64url stays header/QR-safe without reducing entropy.
  return `tok_${randomBytes(32).toString('base64url')}`
}

export class PairingRegistry {
  private links: PairingLink[] = []
  private sessions: DeviceSession[] = []
  private readonly now: () => number
  private readonly newId: () => string
  private readonly newToken: () => string
  private readonly linkTtlMs: number

  constructor(options: PairingRegistryOptions = {}, restore: Partial<PairingRegistryState> = {}) {
    this.now = options.now ?? (() => Date.now())
    this.newId = options.newId ?? randomDeviceId
    this.newToken = options.newToken ?? randomBearerToken
    this.linkTtlMs = options.linkTtlMs ?? 10 * 60_000
    // Deep-copy so a restored registry owns independent records — mutating it (revoke, touch
    // lastSeen) must never reach back into the caller's snapshot/state object.
    this.links = (restore.links ?? []).map((link) => ({ ...link }))
    this.sessions = (restore.sessions ?? []).map((session) => ({ ...session }))
  }

  // Owner action: mint a one-time pairing credential for a new device. The credential is
  // shown once (QR/link); redeeming it yields the device's durable session token.
  issuePairingLink(label: string): { credential: string; expiresAt: number } {
    const at = this.now()
    const link: PairingLink = {
      credential: this.newToken(),
      label: label?.trim() || 'Paired device',
      createdAt: at,
      expiresAt: at + this.linkTtlMs,
      consumedAt: null,
    }
    this.links.push(link)
    return { credential: link.credential, expiresAt: link.expiresAt }
  }

  // Redeem a pairing credential for a device session. One-time: a consumed or expired
  // credential is refused (no token minted). Constant-time credential match.
  redeemPairingLink(credential: string, clientInfo?: string | null): RedeemResult {
    const at = this.now()
    const link = this.links.find((candidate) => safeEqual(candidate.credential, credential))
    if (!link) return { ok: false, error: 'unknown_credential' }
    if (link.consumedAt != null) return { ok: false, error: 'already_consumed' }
    if (at > link.expiresAt) return { ok: false, error: 'expired' }
    link.consumedAt = at
    const session: DeviceSession = {
      id: this.newId(),
      label: link.label,
      token: this.newToken(),
      createdAt: at,
      lastSeenAt: at,
      revokedAt: null,
      clientInfo: clientInfo ?? null,
    }
    this.sessions.push(session)
    return { ok: true, session }
  }

  // Gate check: is this bearer token an ACTIVE (non-revoked) device session? Touches
  // lastSeenAt on a hit so the sessions list shows recency. Returns the session id on match.
  authorizeSessionToken(token: string): { ok: boolean; sessionId?: string } {
    if (!token) return { ok: false }
    for (const session of this.sessions) {
      if (session.revokedAt != null) continue
      if (safeEqual(session.token, token)) {
        session.lastSeenAt = this.now()
        return { ok: true, sessionId: session.id }
      }
    }
    return { ok: false }
  }

  listSessions(): DeviceSessionView[] {
    return this.sessions.map((session) => ({
      id: session.id,
      label: session.label,
      createdAt: session.createdAt,
      lastSeenAt: session.lastSeenAt,
      revoked: session.revokedAt != null,
      clientInfo: session.clientInfo ?? null,
    }))
  }

  // Revoke one device (a lost phone). Idempotent; returns true if a live session flipped.
  revokeSession(id: string): boolean {
    const session = this.sessions.find((candidate) => candidate.id === id)
    if (!session || session.revokedAt != null) return false
    session.revokedAt = this.now()
    return true
  }

  // "Sign out everywhere else" — revoke every live session except the caller's. Returns the
  // count revoked.
  revokeAllexcept(keepId: string): number {
    let count = 0
    for (const session of this.sessions) {
      if (session.id === keepId || session.revokedAt != null) continue
      session.revokedAt = this.now()
      count++
    }
    return count
  }

  // Security purge/sign-out boundary: synchronously remove every credential and device
  // session from the live registry. Callers that acknowledge a purge must persist the
  // resulting snapshot before returning success.
  clearAll(): number {
    const removed = this.links.length + this.sessions.length
    this.links = []
    this.sessions = []
    return removed
  }

  // Housekeeping: drop expired-and-unconsumed links (they can never be redeemed) and
  // consumed links older than the TTL (their session already exists). Keeps state bounded.
  pruneExpiredLinks(): void {
    const at = this.now()
    this.links = this.links.filter((link) => {
      if (link.consumedAt != null) return at - link.consumedAt <= this.linkTtlMs
      return at <= link.expiresAt
    })
  }

  snapshot(): PairingRegistryState {
    return { links: this.links.map((link) => ({ ...link })), sessions: this.sessions.map((session) => ({ ...session })) }
  }

  // Revive a persisted snapshot at boot (VST-035): merge by identity with LIVE entries
  // winning, so a restore can never resurrect a session revoked since the save or clobber
  // one paired during startup. Deep-copied like the constructor path. Returns how many
  // sessions were added.
  restore(state: Partial<PairingRegistryState> | null | undefined): number {
    if (!state || typeof state !== 'object') return 0
    const liveSessions = new Set(this.sessions.map((session) => session.id))
    let added = 0
    for (const raw of Array.isArray(state.sessions) ? state.sessions : []) {
      if (!raw || typeof raw.id !== 'string' || typeof raw.token !== 'string' || !raw.id || !raw.token) continue
      if (liveSessions.has(raw.id)) continue
      this.sessions.push({ ...raw })
      added++
    }
    const liveLinks = new Set(this.links.map((link) => link.credential))
    for (const raw of Array.isArray(state.links) ? state.links : []) {
      if (!raw || typeof raw.credential !== 'string' || !raw.credential) continue
      if (liveLinks.has(raw.credential)) continue
      this.links.push({ ...raw })
    }
    return added
  }
}

export function createPairingRegistry(options?: PairingRegistryOptions, restore?: Partial<PairingRegistryState>): PairingRegistry {
  return new PairingRegistry(options, restore)
}
