import clsx from 'clsx'
import { formatDuration } from '../../../lib/format'
import type { Settings } from '../../../lib/types'
import { Icon } from '../../../lib/icons'
import { Button } from '../../ui'
import { EmptyState, Panel, RunFact, StatusIcon } from '../ui'
import { cap, liveJobIsActive, type CommandCentreSetView, type LiveJob, type PowerGuardSignal, type RunningWorkSummary } from '../model'
import { toneBadge } from '../styles'
import { jobTitle, leaseSummaryText, leaseSummaryTone, wakeTime } from '../work'

interface PowerGuardPanelProps {
  powerGuard: PowerGuardSignal
  settings: Settings | null
  longWorkCount: number
  runningWorkSummary: RunningWorkSummary
  setView: CommandCentreSetView
}

export function PowerGuardPanel(props: PowerGuardPanelProps) {
  const { powerGuard, settings, longWorkCount, runningWorkSummary, setView } = props

  return (
    <Panel title="Power Guard" icon="coffee" action={powerGuard.label}>
      <div className="space-y-2 px-3 py-3">
        <div className="flex items-start gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
          <StatusIcon icon="coffee" tone={powerGuard.tone} pulse={settings?.power?.keepComputerAlive && longWorkCount > 0} />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <div className="truncate text-[13px] font-semibold text-ink">{powerGuard.label}</div>
              <span className={clsx('ml-auto shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium', toneBadge(powerGuard.tone))}>{longWorkCount} long work</span>
            </div>
            <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{powerGuard.detail}</div>
            <div className="mt-2 grid grid-cols-2 gap-1.5">
              <RunFact icon="clock" label="Next Wake" value={runningWorkSummary.nextWake ? wakeTime(runningWorkSummary.nextWake.at) : 'None'} detail={runningWorkSummary.nextWake ? jobTitle(runningWorkSummary.nextWake.job) : 'No scheduled resume'} tone={runningWorkSummary.nextWake && runningWorkSummary.nextWake.at <= Date.now() ? 'high' : 'low'} />
              <RunFact icon="shield" label="Lease Risk" value={leaseSummaryText(runningWorkSummary) || 'Clear'} detail={`${runningWorkSummary.leases.active} active leases`} tone={leaseSummaryTone(runningWorkSummary)} />
            </div>
            <div className="mt-2 flex items-center gap-2">
              <span className="min-w-0 flex-1 truncate text-[13px] text-ink-4">Next: {powerGuard.nextAction}</span>
              <Button variant={powerGuard.tone === 'high' || powerGuard.tone === 'critical' ? 'default' : 'ghost'} size="xs" icon="settings" onClick={() => setView({ name: 'settings' })}>Settings</Button>
            </div>
          </div>
        </div>
      </div>
    </Panel>
  )
}

interface LiveJobsPanelProps {
  liveJobs: LiveJob[]
}

export function LiveJobsPanel(props: LiveJobsPanelProps) {
  const { liveJobs } = props

  return (
    <Panel title="Live Jobs" icon="terminal" action={liveJobs.length ? `${liveJobs.filter((job) => liveJobIsActive(job.status)).length} active` : 'idle'}>
      <div className="divide-y divide-line">
        {liveJobs.length === 0 ? <EmptyState icon="check" title="No active runs" detail="Threads, tickets, rooms, and automations will appear here while running." /> : liveJobs.map((job) => (
          <div key={job.id} className="px-3 py-3">
            <div className="flex items-start gap-3">
              <StatusIcon icon={job.icon} tone={job.tone} pulse={liveJobIsActive(job.status)} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <div className="truncate text-[13px] font-semibold text-ink">{job.title}</div>
                  {job.remote && (
                    <span title={job.remoteLabel ? `Running on ${job.remoteLabel}` : 'Running on Orchestrator'} className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-accent/12 text-accent">
                      <Icon name="globe" size={11} />
                    </span>
                  )}
                  <span className="shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{job.status}</span>
                </div>
                <div className="mt-0.5 truncate text-[13px] text-ink-3">{job.detail}</div>
                {job.footnote && <div className="mt-1 truncate font-mono text-[12px] text-ink-4">{job.footnote}</div>}
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-[12px] text-ink-4">{liveJobIsActive(job.status) ? 'Active' : cap(job.status)} {formatDuration(Date.now() - job.startedAt)}</span>
                  <span className="min-w-0 flex-1" />
                  {job.onResume && <Button variant="default" size="xs" icon="play" onClick={job.onResume}>Resume</Button>}
                  {job.onRetry && <Button variant="ghost" size="xs" icon="clock" onClick={job.onRetry}>Retry</Button>}
                  {job.onDeadLetter && <Button variant="danger" size="xs" icon="archive" onClick={job.onDeadLetter}>Archive</Button>}
                  {job.onCancel && <Button variant="ghost" size="xs" icon="stop" onClick={job.onCancel}>Cancel</Button>}
                  {job.onOpen && <Button variant="default" size="xs" icon="arrowRight" onClick={job.onOpen}>Inspect</Button>}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}
