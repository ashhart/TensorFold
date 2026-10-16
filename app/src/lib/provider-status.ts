// Provider status views (VST-037) — the RENDERER side. Translates the raw codex rate-limit
// snapshot the host persisted (electron/agent/provider-status.ts) into UsageWindow pace/ETA
// views for the model picker and the readiness doctor.
//
// codex's primary window is the short/session limit, secondary the longer weekly one, each
// reported as `usedPercent` + optional `resetsAt`/`windowDurationMins` (installed codex
// 0.142.5 protocol). Percent maps to our count-based UsageWindow as used/limit=100 so
// usage-windows.ts renders pace + ETA unchanged; startedAt = resetsAt − windowDuration gives
// the even-pace reference line.

import type { CodexRateLimitSnapshot, CodexRateLimitWindow, PersistedProviderStatus, ProviderStatusSnapshot } from './types'
import type { UsageWindow, UsageWindowView, UsageTone } from './usage-windows'
import { orderUsageWindows, usageWindowViews, worstUsageTone } from './usage-windows'

export type { CodexRateLimitSnapshot, PersistedProviderStatus, ProviderStatusSnapshot } from './types'

// Mirror of the host key builder (electron/agent/provider-status.ts) — a one-liner
// duplicated so the renderer never imports main-side runtime across the type boundary.
export function providerStatusKey(providerId: string, accountId: string | null | undefined): string {
  return `${providerId}:${accountId ?? 'default'}`
}

export function codexWindowToUsageWindow(
  window: CodexRateLimitWindow | null | undefined,
  id: string,
  label: string,
  kind: UsageWindow['kind'],
): UsageWindow | null {
  if (!window || typeof window.usedPercent !== 'number' || !Number.isFinite(window.usedPercent)) return null
  const resetsAt = typeof window.resetsAt === 'number' && Number.isFinite(window.resetsAt) ? window.resetsAt : null
  const durationMs = typeof window.windowDurationMins === 'number' && window.windowDurationMins > 0 ? window.windowDurationMins * 60_000 : null
  const startedAt = resetsAt != null && durationMs != null ? resetsAt - durationMs : null
  return {
    id,
    label,
    kind,
    used: Math.max(0, Math.min(100, Math.round(window.usedPercent))),
    limit: 100,
    resetsAt,
    startedAt,
  }
}

// A raw codex account rate-limit snapshot → ordered UsageWindows (session first, then weekly).
export function codexRateLimitsToWindows(snapshot: CodexRateLimitSnapshot | null | undefined): UsageWindow[] {
  if (!snapshot) return []
  const windows: UsageWindow[] = []
  const primary = codexWindowToUsageWindow(snapshot.primary, 'codex:session', 'Session', 'session')
  if (primary) windows.push(primary)
  const secondary = codexWindowToUsageWindow(snapshot.secondary, 'codex:weekly', 'Weekly', 'weekly')
  if (secondary) windows.push(secondary)
  return orderUsageWindows(windows)
}

export interface ProviderStatusView {
  providerId: string
  accountId: string | null
  planType: string | null
  windows: UsageWindowView[]
  tone: UsageTone
  updatedAt: number
}

// The picker/doctor view: window views + a rolled-up tone. A snapshot older than
// staleAfterMs (default 6h) is dropped — a day-old rate-limit read is not current truth.
export function providerStatusView(
  snapshot: ProviderStatusSnapshot | null | undefined,
  now = Date.now(),
  staleAfterMs = 6 * 60 * 60_000,
): ProviderStatusView | null {
  if (!snapshot || !snapshot.raw) return null
  if (now - snapshot.updatedAt > staleAfterMs) return null
  const views = usageWindowViews(codexRateLimitsToWindows(snapshot.raw), now)
  if (!views.length) return null
  return {
    providerId: snapshot.providerId,
    accountId: snapshot.accountId,
    planType: snapshot.raw.planType ?? null,
    windows: views,
    tone: worstUsageTone(views),
    updatedAt: snapshot.updatedAt,
  }
}

// Pull a provider/account's status view from the persisted map, if fresh and usable.
export function providerStatusViewFor(
  persisted: PersistedProviderStatus | null | undefined,
  providerId: string,
  accountId: string | null | undefined,
  now = Date.now(),
): ProviderStatusView | null {
  if (!persisted || persisted.version !== 1 || !persisted.statuses) return null
  return providerStatusView(persisted.statuses[providerStatusKey(providerId, accountId)], now)
}
