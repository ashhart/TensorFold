// Provider status persistence (VST-037) — the MAIN-side pure core, deliberately free of any
// renderer/UsageWindow dependency so it can live host-side and cross the type boundary
// cleanly (the renderer translates the RAW codex snapshot into pace/ETA views on read;
// see src/lib/provider-status.ts).
//
// Rate-limit shapes are from the INSTALLED codex 0.142.5 app-server protocol
// (`account/rateLimits/updated` → RateLimitSnapshot { primary, secondary, planType,
// limitName }; each window { usedPercent, resetsAt?, windowDurationMins? }).

export interface CodexRateLimitWindow {
  usedPercent: number
  resetsAt?: number | null
  windowDurationMins?: number | null
}

export interface CodexRateLimitSnapshot {
  primary?: CodexRateLimitWindow | null
  secondary?: CodexRateLimitWindow | null
  planType?: string | null
  limitName?: string | null
}

// One provider/account's last-known rate-limit snapshot. `raw` is the codex data verbatim
// (percent/reset/duration); the renderer derives windows + pace from it.
export interface ProviderStatusSnapshot {
  providerId: string
  accountId: string | null
  raw: CodexRateLimitSnapshot
  updatedAt: number
}

export interface PersistedProviderStatus {
  version: 1
  statuses: Record<string, ProviderStatusSnapshot>
}

export function providerStatusKey(providerId: string, accountId: string | null | undefined): string {
  return `${providerId}:${accountId ?? 'default'}`
}

// True when a codex snapshot carries at least one usable window — guards against persisting
// (and later rendering as "current") an empty rolling update that clears nothing.
export function codexSnapshotHasWindows(snapshot: CodexRateLimitSnapshot | null | undefined): boolean {
  if (!snapshot) return false
  const usable = (w: CodexRateLimitWindow | null | undefined) => !!w && typeof w.usedPercent === 'number' && Number.isFinite(w.usedPercent)
  return usable(snapshot.primary) || usable(snapshot.secondary)
}

// Coerce an untrusted wire snapshot (from the loop dep) into the stored shape, keeping only
// the fields we persist — a crafted payload can't smuggle extra keys onto disk.
export function normalizeCodexSnapshot(input: unknown): CodexRateLimitSnapshot | null {
  if (!input || typeof input !== 'object') return null
  const raw = input as Record<string, unknown>
  const win = (value: unknown): CodexRateLimitWindow | null => {
    if (!value || typeof value !== 'object') return null
    const w = value as Record<string, unknown>
    if (typeof w.usedPercent !== 'number' || !Number.isFinite(w.usedPercent)) return null
    return {
      usedPercent: w.usedPercent,
      resetsAt: typeof w.resetsAt === 'number' ? w.resetsAt : null,
      windowDurationMins: typeof w.windowDurationMins === 'number' ? w.windowDurationMins : null,
    }
  }
  const snapshot: CodexRateLimitSnapshot = {
    primary: win(raw.primary),
    secondary: win(raw.secondary),
    planType: typeof raw.planType === 'string' ? raw.planType : null,
    limitName: typeof raw.limitName === 'string' ? raw.limitName : null,
  }
  return codexSnapshotHasWindows(snapshot) ? snapshot : null
}

// Merge a fresh snapshot into the persisted map: a new snapshot REPLACES the old one for its
// provider/account (rate limits are point-in-time). An empty snapshot is refused so a
// transient blank read can't erase a good last-known status.
export function mergeProviderStatus(
  persisted: PersistedProviderStatus | null | undefined,
  snapshot: ProviderStatusSnapshot,
): PersistedProviderStatus {
  const base: PersistedProviderStatus = persisted && persisted.version === 1 && persisted.statuses ? { version: 1, statuses: { ...persisted.statuses } } : { version: 1, statuses: {} }
  if (!codexSnapshotHasWindows(snapshot.raw)) return base
  base.statuses[providerStatusKey(snapshot.providerId, snapshot.accountId)] = snapshot
  return base
}

export function providerStatusSnapshotFor(
  persisted: PersistedProviderStatus | null | undefined,
  providerId: string,
  accountId: string | null | undefined,
): ProviderStatusSnapshot | null {
  if (!persisted || persisted.version !== 1 || !persisted.statuses) return null
  return persisted.statuses[providerStatusKey(providerId, accountId)] ?? null
}
