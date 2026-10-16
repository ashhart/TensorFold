import clsx from 'clsx'
import type { UsageWindowView } from '../../lib/usage-windows'

// A single rate-limit window (VST-037): a labelled progress bar with percent-used, pace
// tone, and a reset ETA — the real "session 62% · resets in 1h" line codex reports.
export function RateLimitBar({ view }: { view: UsageWindowView }) {
  const pct = view.usedPercent ?? 0
  const barTone = view.tone === 'critical' ? 'bg-danger' : view.tone === 'warn' ? 'bg-warn' : view.tone === 'good' ? 'bg-positive' : 'bg-ink/50'
  const textTone = view.tone === 'critical' ? 'text-danger' : view.tone === 'warn' ? 'text-warn' : 'text-ink-3'
  return (
    <div className="rounded-lg border border-line bg-card-2/70 px-2.5 py-1.5" title={view.paceDetail}>
      <div className="mb-1 flex items-center justify-between gap-2 text-[12px]">
        <span className="font-medium text-ink-3">{view.label}</span>
        <span className={clsx('tabular-nums', textTone)}>{Math.round(pct)}%{view.etaText ? ` · ${view.etaText}` : ''}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-5">
        <div className={clsx('h-full rounded-full transition-all', barTone)} style={{ width: `${Math.max(2, pct)}%` }} />
      </div>
    </div>
  )
}
