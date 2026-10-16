// Pure core for provider rate-limit windows (VST-037).
// A window is a rolling budget (a 5-hour session cap, a weekly cap, a per-model cap). Given
// where we are in the window and how much has been consumed, compute a progress view: percent
// used, the "expected" consumption if spending evenly (a marker on the bar),
// whether we're ahead of / on / behind that pace, an ETA to reset, and a tone for the UI.

export type UsageWindowKind = 'session' | 'weekly' | 'model' | 'generic'
export type UsagePace = 'ahead' | 'on_track' | 'behind' | 'exhausted' | 'unknown'
export type UsageTone = 'good' | 'normal' | 'warn' | 'critical' | 'muted'

// Raw window input, matching what a provider usage snapshot carries per window.
export interface UsageWindow {
  id: string
  label: string
  kind: UsageWindowKind
  used: number
  limit: number | null
  // When this window's budget resets (epoch ms). null = no reset (a lifetime/among-others cap).
  resetsAt: number | null
  // When this window's counting STARTED (epoch ms), so pace can compare elapsed vs consumed.
  // null → pace is unknown (we can still show percent used).
  startedAt?: number | null
}

export interface UsageWindowView {
  id: string
  label: string
  kind: UsageWindowKind
  used: number
  limit: number | null
  usedPercent: number | null      // 0..100, null when no limit
  expectedPercent: number | null  // 0..100 pace marker, null when pace unknown
  pace: UsagePace
  paceDetail: string
  etaText: string | null          // human "resets in 2h", null when no reset
  tone: UsageTone
  remaining: number | null
}

const MINUTE = 60_000
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

// How far off the even-spend line counts as ahead/behind rather than on-track.
const PACE_TOLERANCE = 0.08

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(100, value))
}

export function formatUsageEta(resetsAt: number | null, now: number): string | null {
  if (resetsAt == null) return null
  const delta = resetsAt - now
  if (delta <= 0) return 'resetting'
  if (delta < MINUTE) return 'resets in <1m'
  if (delta < HOUR) return `resets in ${Math.round(delta / MINUTE)}m`
  if (delta < DAY) return `resets in ${Math.round(delta / HOUR)}h`
  return `resets in ${Math.round(delta / DAY)}d`
}

// Fraction of the window elapsed (0..1), or null when we can't tell (no start, or no reset to
// bound the window). Used to place the even-spend pace marker.
function elapsedFraction(window: UsageWindow, now: number): number | null {
  if (window.startedAt == null || window.resetsAt == null) return null
  const span = window.resetsAt - window.startedAt
  if (span <= 0) return null
  return Math.max(0, Math.min(1, (now - window.startedAt) / span))
}

export function usageWindowView(window: UsageWindow, now = Date.now()): UsageWindowView {
  const hasLimit = window.limit != null && window.limit > 0
  const usedPercent = hasLimit ? clampPercent((window.used / (window.limit as number)) * 100) : null
  const remaining = hasLimit ? Math.max(0, (window.limit as number) - window.used) : null
  const etaText = formatUsageEta(window.resetsAt, now)
  const elapsed = elapsedFraction(window, now)
  const expectedPercent = elapsed == null ? null : clampPercent(elapsed * 100)

  let pace: UsagePace = 'unknown'
  let paceDetail = 'Usage tracked; pace unknown.'
  let tone: UsageTone = 'muted'

  if (hasLimit && usedPercent != null && usedPercent >= 100) {
    pace = 'exhausted'
    paceDetail = etaText ? `Limit reached — ${etaText}.` : 'Limit reached.'
    tone = 'critical'
  } else if (hasLimit && usedPercent != null && expectedPercent != null) {
    const usedFraction = usedPercent / 100
    const expectedFraction = expectedPercent / 100
    const drift = usedFraction - expectedFraction
    if (drift > PACE_TOLERANCE) {
      pace = 'behind' // spending FASTER than the even line → will run out early
      const overshoot = Math.round(drift * 100)
      paceDetail = `${overshoot}% ahead of an even pace — may exhaust before reset.`
      tone = usedPercent >= 90 ? 'critical' : 'warn'
    } else if (drift < -PACE_TOLERANCE) {
      pace = 'ahead' // spending SLOWER → comfortable headroom
      paceDetail = `${Math.round(-drift * 100)}% under an even pace — comfortable headroom.`
      tone = 'good'
    } else {
      pace = 'on_track'
      paceDetail = 'Tracking an even pace.'
      tone = usedPercent >= 90 ? 'warn' : 'normal'
    }
  } else if (hasLimit && usedPercent != null) {
    // Have a limit but no pace reference (no start/reset): tone purely on how full it is.
    pace = 'unknown'
    paceDetail = `${Math.round(usedPercent)}% of the limit used.`
    tone = usedPercent >= 90 ? 'critical' : usedPercent >= 75 ? 'warn' : 'normal'
  } else {
    // No limit: a pure usage counter.
    pace = 'unknown'
    paceDetail = 'No published limit for this window.'
    tone = 'muted'
  }

  return {
    id: window.id,
    label: window.label,
    kind: window.kind,
    used: window.used,
    limit: window.limit,
    usedPercent,
    expectedPercent,
    pace,
    paceDetail,
    etaText,
    tone,
    remaining,
  }
}

// Windows sort session → weekly → model-specific → generic, so the
// most actionable (soonest-resetting) window reads first.
const KIND_ORDER: Record<UsageWindowKind, number> = { session: 0, weekly: 1, model: 2, generic: 3 }

export function orderUsageWindows(windows: UsageWindow[]): UsageWindow[] {
  return [...windows].sort((a, b) => {
    const byKind = KIND_ORDER[a.kind] - KIND_ORDER[b.kind]
    if (byKind !== 0) return byKind
    // Within a kind, soonest reset first.
    return (a.resetsAt ?? Infinity) - (b.resetsAt ?? Infinity)
  })
}

export function usageWindowViews(windows: UsageWindow[], now = Date.now()): UsageWindowView[] {
  return orderUsageWindows(windows).map((window) => usageWindowView(window, now))
}

// The single most-constrained window drives an at-a-glance provider badge: the worst tone,
// then the highest percent used.
export function worstUsageTone(views: UsageWindowView[]): UsageTone {
  const rank: Record<UsageTone, number> = { critical: 4, warn: 3, normal: 2, good: 1, muted: 0 }
  let worst: UsageTone = 'muted'
  for (const view of views) if (rank[view.tone] > rank[worst]) worst = view.tone
  return worst
}
