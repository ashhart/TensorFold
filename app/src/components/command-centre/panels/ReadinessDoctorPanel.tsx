import clsx from 'clsx'
import type { DoctorCheck, DoctorSummary } from '../../../lib/readiness-doctor'
import { Button } from '../../ui'
import { EmptyState, Panel, RunStat, StatusIcon } from '../ui'
import { cap, type Tone } from '../model'
import { toneBadge } from '../styles'

interface ReadinessDoctorPanelProps {
  readinessChecks: DoctorCheck[]
  readinessSummary: DoctorSummary
  readinessTone: Tone
  refreshReadiness: () => void
  openReadinessSettings: () => void
}

export function ReadinessDoctorPanel(props: ReadinessDoctorPanelProps) {
  const { readinessChecks, readinessSummary, readinessTone, refreshReadiness, openReadinessSettings } = props

  const readinessIssues = readinessChecks
    .filter((check) => check.status !== 'ok')
    .sort((a, b) => {
      const rank = { error: 0, warning: 1, checking: 2, ok: 3 }
      return rank[a.status] - rank[b.status] || a.area.localeCompare(b.area) || a.title.localeCompare(b.title)
    })
    .slice(0, 6)
  const readinessLabel = readinessSummary.error
    ? `${readinessSummary.error} blocked`
    : readinessSummary.warning
      ? `${readinessSummary.warning} review`
      : readinessSummary.checking
        ? `${readinessSummary.checking} checking`
        : 'ready'
  const readinessCheckTone = (status: DoctorCheck['status']): Tone => {
    if (status === 'error') return 'critical'
    if (status === 'warning') return 'high'
    if (status === 'checking') return 'low'
    return 'good'
  }
  const readinessStatusLabel = (status: DoctorCheck['status']) => status === 'error' ? 'Blocked' : status === 'warning' ? 'Review' : status === 'checking' ? 'Checking' : 'Ready'

  return (
    <Panel title="Readiness Doctor" icon="check" action={readinessLabel}>
      <div className="space-y-3 px-3 py-3">
        <div className="grid grid-cols-4 gap-px overflow-hidden rounded-lg border border-line bg-line">
          <RunStat label="Ready" value={readinessSummary.ok} tone="good" />
          <RunStat label="Review" value={readinessSummary.warning} tone={readinessSummary.warning ? 'high' : 'good'} />
          <RunStat label="Blocked" value={readinessSummary.error} tone={readinessSummary.error ? 'critical' : 'good'} />
          <RunStat label="Checking" value={readinessSummary.checking} tone={readinessSummary.checking ? 'low' : 'good'} />
        </div>

        {readinessIssues.length === 0 ? (
          <EmptyState icon="check" title="All systems ready" detail="Providers, connectors, remote control, terminals, and security checks are passing." />
        ) : (
          <div className="divide-y divide-line rounded-lg border border-line">
            {readinessIssues.map((check) => {
              const tone = readinessCheckTone(check.status)
              return (
                <div key={check.id} className="px-2.5 py-2.5">
                  <div className="flex items-start gap-2.5">
                    <StatusIcon icon={check.status === 'error' ? 'warning' : check.status === 'checking' ? 'clock' : 'shield'} tone={tone} pulse={check.status === 'checking'} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <div className="min-w-0 flex-1 truncate text-[13px] font-semibold text-ink">{check.title}</div>
                        <span className={clsx('shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium', toneBadge(tone))}>{readinessStatusLabel(check.status)}</span>
                      </div>
                      <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{check.detail}</div>
                      <div className="mt-1 flex items-center gap-2">
                        <span className="shrink-0 rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-4">{cap(check.area)}</span>
                        <span className="min-w-0 flex-1" />
                        {check.action && <Button variant={tone === 'critical' ? 'default' : 'ghost'} size="xs" icon="arrowRight" onClick={openReadinessSettings}>{check.action}</Button>}
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        <div className="flex items-center gap-2">
          <span className="min-w-0 flex-1 truncate text-[13px] text-ink-4">{readinessSummary.total} checks tracked</span>
          <Button variant="ghost" size="xs" icon="refresh" onClick={refreshReadiness}>Refresh</Button>
          <Button variant={readinessTone === 'critical' || readinessTone === 'high' ? 'default' : 'ghost'} size="xs" icon="settings" onClick={openReadinessSettings}>Full doctor</Button>
        </div>
      </div>
    </Panel>
  )
}
