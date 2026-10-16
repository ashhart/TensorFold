// Observer hooks (VST-012): a read-only, fail-open observation contract adapted from
// Hermes — session, turn, API request/response, tool, approval, subagent, and remote
// events flow to registered observers as SANITIZED, BOUNDED payloads. Observers can never
// steer or break a run: emission is fire-and-forget, exceptions and rejections are
// swallowed (counted, never propagated), and a hook that keeps failing is auto-disabled
// with its status visible — broken observation must not look healthy, and observation
// must never become interference.

import { redactText } from './export-redaction'
import type { AgentEvent } from './types'

export type ObserverEventKind =
  | 'session.start'
  | 'session.end'
  | 'turn.start'
  | 'turn.end'
  | 'api.request'
  | 'api.response'
  | 'tool.start'
  | 'tool.end'
  | 'approval.requested'
  | 'approval.resolved'
  | 'subagent.start'
  | 'subagent.end'
  | 'remote.event'

export interface ObserverEvent {
  kind: ObserverEventKind
  at: number
  threadId?: string | null
  runId?: string | null
  payload: Record<string, unknown>
}

export type ObserverHandler = (event: ObserverEvent) => void | Promise<void>

export interface ObserverHookStatus {
  name: string
  delivered: number
  failed: number
  consecutiveFailures: number
  disabled: boolean
}

// Sanitization bounds: observers get enough to correlate and measure, never transcripts.
const MAX_STRING = 300
const MAX_DEPTH = 2
const MAX_ARRAY = 10
const MAX_KEYS = 24
// Key names that never cross the boundary, regardless of value.
const BLOCKED_KEYS = /(api[_-]?key|secret|token|password|passwd|credential|authorization|cookie)/i
// Consecutive failures before a hook is auto-disabled (fail-open, visibly).
export const OBSERVER_DISABLE_AFTER = 20

export function sanitizeObserverPayload(raw: unknown, depth = 0): Record<string, unknown> {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {}
  const out: Record<string, unknown> = {}
  let keys = 0
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (keys >= MAX_KEYS) break
    if (BLOCKED_KEYS.test(key)) continue
    const clean = sanitizeValue(value, depth)
    if (clean !== undefined) {
      out[key] = clean
      keys++
    }
  }
  return out
}

function sanitizeValue(value: unknown, depth: number): unknown {
  if (value == null) return value
  if (typeof value === 'string') {
    // Redact secret/PII shapes, then bound the length — observers get labels, not bodies.
    const redacted = redactText(value).text
    return redacted.length > MAX_STRING ? `${redacted.slice(0, MAX_STRING)}…` : redacted
  }
  if (typeof value === 'number' || typeof value === 'boolean') return value
  if (Array.isArray(value)) {
    if (depth >= MAX_DEPTH) return `[array:${value.length}]`
    return value.slice(0, MAX_ARRAY).map((item) => sanitizeValue(item, depth + 1))
  }
  if (typeof value === 'object') {
    if (depth >= MAX_DEPTH) return '[object]'
    return sanitizeObserverPayload(value, depth + 1)
  }
  // Functions/symbols never cross.
  return undefined
}

export interface ObserverHub {
  register(name: string, handler: ObserverHandler): () => void
  emit(kind: ObserverEventKind, input: { threadId?: string | null; runId?: string | null; payload?: unknown }): void
  status(): ObserverHookStatus[]
  // Bounded ring of recent events (doctor/debug visibility without an external sink).
  recent(limit?: number): ObserverEvent[]
}

export function createObserverHub(options: { now?: () => number; ringSize?: number } = {}): ObserverHub {
  const now = options.now ?? (() => Date.now())
  const ringSize = options.ringSize ?? 200
  interface HookEntry { name: string; handler: ObserverHandler; delivered: number; failed: number; consecutive: number; disabled: boolean }
  const hooks: HookEntry[] = []
  const ring: ObserverEvent[] = []

  return {
    register(name, handler) {
      const entry: HookEntry = { name: name.trim() || 'observer', handler, delivered: 0, failed: 0, consecutive: 0, disabled: false }
      hooks.push(entry)
      return () => {
        const at = hooks.indexOf(entry)
        if (at >= 0) hooks.splice(at, 1)
      }
    },

    emit(kind, input) {
      const event: ObserverEvent = {
        kind,
        at: now(),
        threadId: input.threadId ?? null,
        runId: input.runId ?? null,
        payload: sanitizeObserverPayload(input.payload),
      }
      ring.push(event)
      if (ring.length > ringSize) ring.splice(0, ring.length - ringSize)
      for (const hook of hooks) {
        if (hook.disabled) continue
        const fail = () => {
          hook.failed++
          hook.consecutive++
          if (hook.consecutive >= OBSERVER_DISABLE_AFTER) hook.disabled = true
        }
        try {
          const result = hook.handler(event)
          // Async failures are swallowed too — a rejecting observer can't surface as an
          // unhandled rejection inside a run.
          if (result && typeof (result as Promise<void>).then === 'function') {
            void (result as Promise<void>).then(
              () => { hook.delivered++; hook.consecutive = 0 },
              () => fail(),
            )
          } else {
            hook.delivered++
            hook.consecutive = 0
          }
        } catch {
          fail()
        }
      }
    },

    status() {
      return hooks.map((hook) => ({
        name: hook.name,
        delivered: hook.delivered,
        failed: hook.failed,
        consecutiveFailures: hook.consecutive,
        disabled: hook.disabled,
      }))
    },

    recent(limit = 50) {
      return ring.slice(-Math.max(1, limit))
    },
  }
}

/**
 * Map a runtime AgentEvent onto the observer contract (VST-012), mirroring the spine's
 * whitelist approach: structural facts only — per-token deltas and free text never cross.
 * Returns null for events observers have no business seeing.
 */
export function observerEventForAgentEvent(event: AgentEvent): { kind: ObserverEventKind; payload: Record<string, unknown> } | null {
  switch (event.type) {
    case 'status':
      if (event.status === 'thinking') return { kind: 'turn.start', payload: { status: event.status } }
      return null
    case 'item_add':
      if (event.item.kind === 'tool') return { kind: 'tool.start', payload: { itemId: event.item.id, tool: event.item.tool ?? null } }
      return null
    case 'item_update': {
      if (!event.patch.state || !['ok', 'error', 'denied'].includes(event.patch.state)) return null
      return { kind: 'tool.end', payload: { itemId: event.id, state: event.patch.state } }
    }
    case 'approval_request':
      return { kind: 'approval.requested', payload: { callId: event.callId, itemId: event.itemId } }
    case 'done':
      return { kind: 'turn.end', payload: { ok: true } }
    case 'error':
      return { kind: 'turn.end', payload: { ok: false, message: event.message } }
    default:
      return null
  }
}
