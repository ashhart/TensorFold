import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { vontraPath } from '../../store'
import { auditPreview, currentSecurityRoleFromEnv } from '../../agent/security'
import type { SecurityRole } from '../../agent/types'

// Security roles, the RBAC gate, and the tamper-evident audit log (hash-chained,
// owner-only 0o600 file). Pure host infrastructure — depends on env/os/fs only,
// never on the in-memory `store`.

export const SECURITY_ROLE_RANK: Record<SecurityRole, number> = { viewer: 0, operator: 1, owner: 2 }

let lastAuditError: string | null = null
export function getLastAuditError(): string | null { return lastAuditError }

export function envEnabled(name: string): boolean {
  const value = process.env[name]?.toLowerCase()
  return value === '1' || value === 'true' || value === 'yes'
}

export function currentSecurityRole(): SecurityRole {
  return currentSecurityRoleFromEnv(process.env)
}

export function auditActor(): string {
  const configured = process.env.VONTRA_AUDIT_ACTOR?.trim()
  if (configured) return configured
  try {
    const username = os.userInfo().username || 'local'
    const hostname = os.hostname() || 'machine'
    return `${username}@${hostname}`
  } catch {
    return 'local'
  }
}

export function requireSecurityRole(required: SecurityRole, action: string): void {
  const role = currentSecurityRole()
  if (SECURITY_ROLE_RANK[role] >= SECURITY_ROLE_RANK[required]) return
  void auditEvent('security.rbac_denied', { action, requiredRole: required })
  throw new Error(`Current TensorFold role "${role}" cannot ${action}; requires ${required}.`)
}

// Serialize all appends through a single promise chain. auditEvent does an unserialized
// read-last-hash → append; two concurrent calls (most call sites fire `void auditEvent(...)`) would
// otherwise read the same prevHash and FORK the tamper-evident chain. Capturing actor/role/ts at
// call time keeps the event's metadata accurate even though the write is queued.
let auditTail: Promise<void> = Promise.resolve()

export function auditEvent(kind: string, data: Record<string, unknown> = {}): Promise<void> {
  // Treat every caller as potentially carrying user/model/provider text. Most call sites already
  // use auditPreview explicitly, but a new nested field (approval labels/explanations were a real
  // example) must not silently reintroduce raw credentials into the plaintext audit journal.
  // Central sanitization is the final invariant; numbers/booleans remain machine-queryable.
  const sanitized = sanitizeAuditRecord(data)
  const event = { ...sanitized, kind, actor: auditPreview(auditActor(), 200), role: currentSecurityRole(), ts: Date.now(), chainVersion: 1 }
  const run = auditTail.then(() => appendAuditEvent(event))
  auditTail = run.catch(() => {}) // keep the chain alive even if one append throws
  return run
}

function sanitizeAuditRecord(input: Record<string, unknown>): Record<string, unknown> {
  const seen = new WeakSet<object>()
  const sanitize = (value: unknown, depth: number): unknown => {
    if (typeof value === 'string') return auditPreview(value)
    if (value == null || typeof value === 'number' || typeof value === 'boolean') return value
    if (typeof value === 'bigint') return value.toString()
    if (typeof value !== 'object') return auditPreview(String(value))
    if (seen.has(value)) return '[circular]'
    if (depth >= 6) {
      try { return auditPreview(JSON.stringify(value)) } catch { return '[unserializable]' }
    }
    seen.add(value)
    try {
      if (Array.isArray(value)) {
        const bounded = value.slice(0, 100).map((item) => sanitize(item, depth + 1))
        if (value.length > bounded.length) bounded.push(`[${value.length - bounded.length} more items]`)
        return bounded
      }
      const entries = Object.entries(value).slice(0, 100)
      const out = Object.fromEntries(entries.map(([key, item]) => [key, sanitize(item, depth + 1)]))
      if (Object.keys(value).length > entries.length) out._truncatedFields = Object.keys(value).length - entries.length
      return out
    } finally {
      seen.delete(value)
    }
  }
  return sanitize(input, 0) as Record<string, unknown>
}

/** Await every audit event captured before this call (used before securely erasing the log). */
export async function flushAuditEvents(): Promise<void> {
  await auditTail
}

/**
 * Serialize an operation behind every already-captured audit append and ahead of every
 * append captured afterwards. The purge control uses this barrier to erase the old log
 * without a queued pre-purge event racing back onto disk after deletion.
 */
export function runAuditBarrier(task: () => Promise<void>): Promise<void> {
  const run = auditTail.then(task)
  auditTail = run.catch(() => {})
  return run
}

async function appendAuditEvent(base: Record<string, unknown>): Promise<void> {
  try {
    const file = vontraPath('security-events.jsonl')
    const anchorFile = vontraPath('security-events.anchor.json')
    await fs.mkdir(path.dirname(file), { recursive: true })
    // The log alone can't detect deletion/truncation — a wiped file just looks like a fresh
    // chain. A separate anchor file persists the last hash OUTSIDE the log, so a missing or
    // emptied log is recorded as `missing:<lastHash>` / `truncated:<lastHash>` instead of
    // silently restarting at genesis. Genuine first-ever write (no log AND no anchor) is null.
    const tail = await readLogTail(file)
    const anchorHash = await readAnchorHash(anchorFile)
    let prevHash: string | null
    if (tail.unreadable) prevHash = anchorHash ? `unreadable:${anchorHash}` : 'unreadable'
    else if (tail.hash) prevHash = anchorHash && anchorHash !== tail.hash ? `rewound:${anchorHash}` : tail.hash
    else if (!tail.exists) prevHash = anchorHash ? `missing:${anchorHash}` : null
    else prevHash = anchorHash ? `truncated:${anchorHash}` : null
    const event = { ...base, prevHash }
    const hash = hashAuditEvent(event)
    await fs.appendFile(file, JSON.stringify({ ...event, hash }) + '\n', { encoding: 'utf8', mode: 0o600 })
    await fs.chmod(file, 0o600).catch(() => {})
    await writeAnchorHash(anchorFile, hash)
    lastAuditError = null
  } catch (error: any) {
    lastAuditError = String(error?.message ?? error)
    console.warn(`[vontra] audit logging failed: ${lastAuditError}`)
    if (envEnabled('VONTRA_STRICT_AUDIT')) throw error
  }
}

interface LogTail { exists: boolean; hash: string | null; unreadable: boolean }

async function readLogTail(file: string): Promise<LogTail> {
  let raw: string
  try {
    raw = await fs.readFile(file, 'utf8')
  } catch (e: any) {
    if (e?.code === 'ENOENT') return { exists: false, hash: null, unreadable: false } // no log
    return { exists: true, hash: null, unreadable: true } // exists but unreadable
  }
  const last = raw.trimEnd().split('\n').filter(Boolean).pop()
  if (!last) return { exists: true, hash: null, unreadable: false } // empty/truncated
  try {
    const parsed = JSON.parse(last)
    const hash = typeof parsed?.hash === 'string' && parsed.hash ? parsed.hash : hashAuditEvent(parsed)
    return { exists: true, hash, unreadable: false }
  } catch {
    return { exists: true, hash: null, unreadable: true } // corrupt last line
  }
}

async function readAnchorHash(anchorFile: string): Promise<string | null> {
  try {
    const parsed = JSON.parse(await fs.readFile(anchorFile, 'utf8'))
    if (typeof parsed?.lastHash === 'string' && parsed.lastHash) return parsed.lastHash
    return typeof parsed?.hash === 'string' && parsed.hash ? parsed.hash : null
  } catch {
    return null
  }
}

async function writeAnchorHash(anchorFile: string, lastHash: string): Promise<void> {
  // The anchor is the continuity checkpoint of the tamper-evident hash chain, so it must be
  // durable: stage to a temp file, fdatasync it, then atomically rename (mirrors atomicWriteJson
  // in store.ts). Without the sync, a crash after rename "succeeds" could lose the still-cached
  // bytes and reopen the chain at a stale anchor on reboot.
  const tmp = `${anchorFile}.tmp-${process.pid}-${Date.now()}`
  const payload = JSON.stringify({ lastHash, updatedAt: Date.now() }, null, 2)
  let handle: import('node:fs/promises').FileHandle | null = null
  try {
    handle = await fs.open(tmp, 'w', 0o600)
    await handle.writeFile(payload, 'utf8')
    // Flush before the rename; a flush failure aborts (catch unlinks the temp, preserving the
    // prior anchor) instead of renaming possibly-unflushed bytes over the good checkpoint.
    await handle.datasync()
    await handle.close()
    handle = null
    await fs.rename(tmp, anchorFile)
    await fs.chmod(anchorFile, 0o600).catch(() => {})
  } catch (e) {
    if (handle) await handle.close().catch(() => {})
    await fs.unlink(tmp).catch(() => {}) // don't leave a stray temp on failure
    throw e
  }
}

function hashAuditEvent(event: Record<string, unknown>): string {
  const { hash: _hash, ...withoutHash } = event as Record<string, unknown>
  return createHash('sha256').update(JSON.stringify(withoutHash)).digest('hex')
}
