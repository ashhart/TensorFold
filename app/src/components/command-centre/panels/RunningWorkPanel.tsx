import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import type { Job, Project, Settings } from '../../../lib/types'
import { Icon } from '../../../lib/icons'
import { Button } from '../../ui'
import { Panel, RunFact, RunStat, StatusIcon } from '../ui'
import { cap, type RunningWorkSummary, type WorkLaneSummary } from '../model'
import { toneBadge, toneText } from '../styles'
import { durableJobFootnote, jobIcon, jobTitle, jobTone, leaseSummaryText, leaseSummaryTone, wakeTime } from '../work'

interface RunningWorkPanelProps {
  runningWorkSummary: RunningWorkSummary
  workLaneSummary: WorkLaneSummary[]
  actionableDurableJobs: Job[]
  projects: Project[]
  orchestrators: Settings['remoteOrchestrators']
  retryJob: (id: string, reason?: string, nextWakeAt?: number | null, resumePlan?: string | null) => Promise<void>
  resumeJob: (id: string, resumePlan?: string | null) => Promise<void>
  deadLetterJob: (id: string, reason?: string) => Promise<void>
  openDurableJob: (job: Job) => void
}

export function RunningWorkPanel(props: RunningWorkPanelProps) {
  const tx = useAuthoredText()
  const { runningWorkSummary, workLaneSummary, actionableDurableJobs, projects, orchestrators, retryJob, resumeJob, deadLetterJob, openDurableJob } = props
  const projectById = new Map(projects.map((project) => [project.id, project]))
  const remoteProjectLabel = (projectId?: string | null): string | null => {
    if (!projectId) return null
    const project = projectById.get(projectId)
    if (project?.executionTarget !== 'orchestrator') return null
    return orchestrators.find((host) => host.id === project.orchestratorId)?.name ?? 'Orchestrator'
  }

  return (
    <Panel title={tx("Running Work")} icon="terminal" action={`${runningWorkSummary.visibleTotal}/${runningWorkSummary.total} tracked`}>
      <div className="px-3 py-3">
        <div className="grid grid-cols-4 gap-px overflow-hidden rounded-lg border border-line bg-line">
          <RunStat label={tx("Queued")} value={runningWorkSummary.statusCounts.queued} tone={runningWorkSummary.statusCounts.queued ? 'low' : 'good'} />
          <RunStat label={tx("Running")} value={runningWorkSummary.statusCounts.running} tone={runningWorkSummary.statusCounts.running ? 'normal' : 'low'} />
          <RunStat label={tx("Stalled")} value={runningWorkSummary.statusCounts.stalled} tone={runningWorkSummary.statusCounts.stalled ? 'high' : 'good'} />
          <RunStat label={tx("Archived")} value={runningWorkSummary.statusCounts.dead_letter} tone={runningWorkSummary.statusCounts.dead_letter ? 'critical' : 'good'} />
        </div>

        <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-3">
          <RunFact icon="clock" label={tx("Next Wake")} value={runningWorkSummary.nextWake ? wakeTime(runningWorkSummary.nextWake.at) : 'None'} detail={runningWorkSummary.nextWake ? jobTitle(runningWorkSummary.nextWake.job) : 'No scheduled resume'} tone={runningWorkSummary.nextWake && runningWorkSummary.nextWake.at <= Date.now() ? 'high' : 'normal'} />
          <RunFact icon="shield" label={tx("Leases")} value={`${runningWorkSummary.leases.active} active`} detail={leaseSummaryText(runningWorkSummary)} tone={leaseSummaryTone(runningWorkSummary)} />
          <RunFact icon="clock" label={tx("Retries")} value={`${runningWorkSummary.attempts.total} attempts`} detail={`${runningWorkSummary.attempts.retryable} retryable / max ${runningWorkSummary.attempts.max}x`} tone={runningWorkSummary.attempts.retryable ? 'high' : runningWorkSummary.attempts.retried ? 'normal' : 'good'} />
        </div>

        <div className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-4">
          {workLaneSummary.map((lane) => (
            <div key={lane.id} className="rounded-md bg-card-2 px-2 py-1.5">
              <div className="truncate text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">{lane.label}</div>
              <div className={clsx('mt-0.5 text-[13px] font-semibold', lane.count ? toneText(lane.tone) : 'text-ink-4')}>{lane.count}</div>
            </div>
          ))}
        </div>

        <div className="mt-2 flex flex-wrap gap-1.5">
          {runningWorkSummary.kindCounts.length === 0 ? (
            <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-4">{tx("No durable jobs")}</span>
          ) : runningWorkSummary.kindCounts.map(({ kind, count }) => (
            <span key={kind} className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{cap(kind)} {count}</span>
          ))}
          {runningWorkSummary.attempts.retried > 0 && <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{runningWorkSummary.attempts.retried} {tx("retried")}</span>}
          {runningWorkSummary.statusCounts.failed > 0 && <span className="rounded-md bg-danger/10 px-1.5 py-0.5 text-[12px] text-danger">{runningWorkSummary.statusCounts.failed} {tx("failed")}</span>}
          {runningWorkSummary.unresolvedApprovals > 0 && <span className="rounded-md bg-warn/12 px-1.5 py-0.5 text-[12px] text-warn">{runningWorkSummary.unresolvedApprovals} {tx("approvals")}</span>}
        </div>

        {actionableDurableJobs.length > 0 && (
          <div className="mt-3 divide-y divide-line rounded-lg border border-line">
            {actionableDurableJobs.map((job) => (
              <div key={job.id} className="px-2.5 py-2.5">
                <div className="flex items-start gap-2.5">
                  <StatusIcon icon={jobIcon(job)} tone={jobTone(job)} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <div className={clsx('truncate text-[13px] font-semibold', jobTone(job) === 'critical' ? 'text-danger' : 'text-ink')}>{jobTitle(job)}</div>
                      {remoteProjectLabel(job.projectId) && (
                        <span title={tx("Project runs on {value0}", { value0: remoteProjectLabel(job.projectId) })} className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-accent/12 text-accent">
                          <Icon name="globe" size={11} />
                        </span>
                      )}
                      <span className={clsx('shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium', toneBadge(jobTone(job)))}>{job.status.replace(/_/g, ' ')}</span>
                    </div>
                    <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{job.blockedReason || job.resumePlan || job.owner}</div>
                    <div className="mt-1 font-mono text-[12px] text-ink-4">{durableJobFootnote(job)}</div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {(job.status === 'stalled' || job.status === 'failed' || job.status === 'dead_letter') && <Button variant="default" size="xs" icon="clock" onClick={() => void retryJob(job.id, 'Manual retry from Running Work summary.')}>{tx("Retry")}</Button>}
                      {job.status === 'stalled' && <Button variant="ghost" size="xs" icon="play" onClick={() => void resumeJob(job.id)}>{tx("Resume")}</Button>}
                      {job.status !== 'dead_letter' && <Button variant="danger" size="xs" icon="archive" onClick={() => void deadLetterJob(job.id, 'Archived from Running Work summary.')}>{tx("Archive")}</Button>}
                      {(job.currentThreadId || job.threadId || job.roomId || job.automationId) && <Button variant="ghost" size="xs" icon="arrowRight" onClick={() => openDurableJob(job)}>{tx("Inspect")}</Button>}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Panel>
  )
}
