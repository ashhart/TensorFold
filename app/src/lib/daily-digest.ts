// Daily digest formatter: renders the away-brief items (buildAwayBriefItems) into a readable,
// deterministic briefing a scheduled 'daily' automation can deliver — no model call needed, so
// it's cheap and always consistent. The scheduler ('daily' scheduleKind + computeNextRun) and
// the item composer (away-brief.ts) already exist; this is the delivery-shaping piece that was
// missing between them.

import type { AwayBriefItem, AwayBriefTone } from './away-brief'

export interface DailyDigestOptions {
  title?: string // default 'Daily brief'
  dateLabel?: string // e.g. 'Wed 1 Jul'; if omitted, derived from `now`
  now?: number // used to derive dateLabel when one isn't given
  maxItems?: number // cap total listed items (default 40)
}

// Tone → tier. critical/high demand action; normal is progress; low/good is ambient.
const TIER: Record<AwayBriefTone, 'attention' | 'updates' | 'fyi'> = {
  critical: 'attention', high: 'attention', normal: 'updates', low: 'fyi', good: 'fyi',
}
const TIER_ORDER: Array<{ key: 'attention' | 'updates' | 'fyi'; heading: string }> = [
  { key: 'attention', heading: '## ⚠ Needs attention' },
  { key: 'updates', heading: '## Updates' },
  { key: 'fyi', heading: '## FYI' },
]

function defaultDateLabel(now: number): string {
  try {
    return new Date(now).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })
  } catch {
    return new Date(now).toISOString().slice(0, 10)
  }
}

function line(it: AwayBriefItem): string {
  const detail = it.detail?.trim() ? ` — ${it.detail.trim()}` : ''
  return `- [${it.kind}] ${it.title}${detail}`
}

/**
 * Render a briefing: a dated header, a one-line headline (how many need you vs other), then the
 * items grouped into attention → updates → FYI tiers, newest-first within each. Pure and
 * deterministic given (items, options). Caps the list at maxItems and notes any remainder.
 */
export function formatDailyDigest(items: AwayBriefItem[], opts: DailyDigestOptions = {}): string {
  const title = opts.title || 'Daily brief'
  const dateLabel = opts.dateLabel || defaultDateLabel(opts.now ?? 0)
  const header = `# ${title} — ${dateLabel}`

  if (!items.length) return `${header}\n\nNo activity to report.`

  const maxItems = Math.max(1, opts.maxItems ?? 40)
  const sorted = [...items].sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0))
  const shown = sorted.slice(0, maxItems)
  const truncated = sorted.length - shown.length

  const attention = items.filter((i) => TIER[i.tone] === 'attention').length
  const others = items.length - attention
  const headlineParts: string[] = []
  if (attention) headlineParts.push(`${attention} need you`)
  if (others) headlineParts.push(`${others} update${others === 1 ? '' : 's'}`)
  const headline = headlineParts.join(' · ') || `${items.length} items`

  const sections: string[] = []
  for (const tier of TIER_ORDER) {
    const inTier = shown.filter((i) => TIER[i.tone] === tier.key)
    if (!inTier.length) continue
    sections.push(`${tier.heading}\n${inTier.map(line).join('\n')}`)
  }

  const footer = truncated > 0 ? `\n\n…and +${truncated} more.` : ''
  return `${header}\n\n${headline}\n\n${sections.join('\n\n')}${footer}`
}
