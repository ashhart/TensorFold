import type { IncomingMessage } from 'node:http'
import { timingSafeEqual } from 'node:crypto'
import { currentSecurityRoleFromEnv } from './security'
import type { SecurityRole } from './types'

const SECURITY_ROLE_RANK: Record<SecurityRole, number> = { viewer: 0, operator: 1, owner: 2 }

interface PairingAttemptState {
  failures: number
  windowStartedAt: number
  lastSeenAt: number
}

export interface PairingAttemptLimiterOptions {
  now?: () => number
  maxFailures?: number
  windowMs?: number
  maxEntries?: number
}

export class PairingAttemptLimiter {
  private readonly attempts = new Map<string, PairingAttemptState>()
  private readonly now: () => number
  private readonly maxFailures: number
  private readonly windowMs: number
  private readonly maxEntries: number

  constructor(options: PairingAttemptLimiterOptions = {}) {
    this.now = options.now ?? (() => Date.now())
    this.maxFailures = Math.max(1, Math.floor(options.maxFailures ?? 5))
    this.windowMs = Math.max(1_000, Math.floor(options.windowMs ?? 60_000))
    this.maxEntries = Math.max(1, Math.floor(options.maxEntries ?? 1_024))
  }

  check(key: string): { blocked: boolean; retryAfterMs: number } {
    const at = this.now()
    this.prune(at)
    const state = this.attempts.get(key)
    if (!state || state.failures < this.maxFailures) return { blocked: false, retryAfterMs: 0 }
    return { blocked: true, retryAfterMs: Math.max(1, state.windowStartedAt + this.windowMs - at) }
  }

  recordFailure(key: string): void {
    const at = this.now()
    this.prune(at)
    const current = this.attempts.get(key)
    if (current) {
      current.failures++
      current.lastSeenAt = at
      return
    }
    while (this.attempts.size >= this.maxEntries) {
      const oldest = this.attempts.keys().next().value as string | undefined
      if (oldest == null) break
      this.attempts.delete(oldest)
    }
    this.attempts.set(key, { failures: 1, windowStartedAt: at, lastSeenAt: at })
  }

  clear(key: string): void {
    this.attempts.delete(key)
  }

  trackedSourceCount(): number {
    this.prune(this.now())
    return this.attempts.size
  }

  private prune(at: number): void {
    for (const [key, state] of this.attempts) {
      if (at - state.windowStartedAt >= this.windowMs) this.attempts.delete(key)
    }
  }
}

export function authToken(req: IncomingMessage): string {
  const header = req.headers.authorization
  if (typeof header === 'string' && /^bearer\s+/i.test(header)) return header.replace(/^bearer\s+/i, '').trim()
  const token = req.headers['x-vontra-token']
  return Array.isArray(token) ? token[0] : (token ?? '')
}

export function tokenMatches(expected: string, actual: string): boolean {
  if (!expected || !actual) return false
  const a = Buffer.from(expected)
  const b = Buffer.from(actual)
  if (a.length !== b.length || a.length === 0) return false
  return timingSafeEqual(a, b)
}

export function securityRoleAllows(required: SecurityRole): boolean {
  return SECURITY_ROLE_RANK[currentSecurityRoleFromEnv(process.env)] >= SECURITY_ROLE_RANK[required]
}

export function requiredRoleForRemoteMutation(pathname: string): SecurityRole {
  if (pathname === '/v1/auth/sessions/revoke' || pathname === '/v1/terminals' || /^\/v1\/terminals\/[^/]+\/(?:write|resize|rename|kill)$/.test(pathname)) {
    return 'owner'
  }
  return 'operator'
}
