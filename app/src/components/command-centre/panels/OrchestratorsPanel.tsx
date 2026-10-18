import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { relativeTime, shortPath } from '../../../lib/format'
import type { RemoteOrchestratorSnapshotResult, Settings } from '../../../lib/types'
import { Icon } from '../../../lib/icons'
import { Button } from '../../ui'
import { DisclosurePanel, EmptyState } from '../ui'
import type { CommandCentreSetView } from '../model'
import { jobTitle, wakeTime } from '../work'

interface OrchestratorsPanelProps {
  orchestrators: Settings['remoteOrchestrators']
  remoteSnapshots: Record<string, RemoteOrchestratorSnapshotResult>
  remoteBusy: string | null
  refreshRemoteHost: (hostId: string) => Promise<void>
  setView: CommandCentreSetView
  isOpen: boolean
  onToggle: () => void
}

export function OrchestratorsPanel(props: OrchestratorsPanelProps) {
  const tx = useAuthoredText()
  const { orchestrators, remoteSnapshots, remoteBusy, refreshRemoteHost, setView, isOpen, onToggle } = props

  return (
    <DisclosurePanel title={tx("Orchestrators")} icon="globe" badge={orchestrators.length ? `${orchestrators.length} paired` : 'none'} isOpen={isOpen} onToggle={onToggle}>
      <div className="space-y-2 px-3 py-3">
        {orchestrators.length === 0 ? (
          <EmptyState icon="globe" title={tx("No paired Orchestrators")} detail="Add one in Settings to control an always-on TensorFold from this app." />
        ) : orchestrators.map((host) => {
          const result = remoteSnapshots[host.id]
          const ok = result?.ok
          const active = (result?.activeThreadIds?.length ?? 0) + (result?.activeRoomIds?.length ?? 0) + (result?.activeMissionIds?.length ?? 0)
          const remoteWork = result?.work
          const remoteBlocked = remoteWork ? remoteWork.jobs.stalled + remoteWork.jobs.failed + remoteWork.jobs.deadLetter : 0
          const remoteLeaseRisk = remoteWork ? remoteWork.leases.expired + remoteWork.leases.expiringSoon + remoteWork.leases.missingRunning : 0
          const remoteProjects = (result?.projects ?? []).filter((project) => !project.archivedAt).slice(0, 3)
          const remoteJobs = (result?.jobs ?? [])
            .filter((job) => ['queued', 'running', 'waiting_approval', 'paused', 'stalled', 'failed', 'dead_letter'].includes(job.status))
            .slice(0, 3)
          return (
            <div key={host.id} className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
              <div className="flex items-start gap-2">
                <span className={clsx('mt-1 h-2 w-2 shrink-0 rounded-full', ok ? 'bg-positive' : host.lastError || result?.error ? 'bg-danger' : 'bg-ink-4')} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <div className="truncate text-[13px] font-semibold text-ink">{result?.name || host.name}</div>
                    <span className="ml-auto text-[12px] text-ink-4">{result?.generatedAt ? relativeTime(result.generatedAt) : host.lastConnectedAt ? tx("seen {value0}", { value0: relativeTime(host.lastConnectedAt) }) : tx("not checked")}</span>
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[12px] text-ink-4">{host.serverUrl}</div>
                  {result?.ok ? (
                    <>
                      <div className="mt-2 grid grid-cols-4 gap-1 text-center text-[12px] text-ink-3">
                        <span className="rounded-md bg-card px-1.5 py-1">{active} {tx("active")}</span>
                        <span className="rounded-md bg-card px-1.5 py-1">{result.totalProjects ?? 0} {tx("projects")}</span>
                        <span className="rounded-md bg-card px-1.5 py-1">{result.totalRooms ?? 0} {tx("rooms")}</span>
                        <span className="rounded-md bg-card px-1.5 py-1">{result.totalMissions ?? 0} {tx("missions")}</span>
                      </div>
                      {remoteWork && (
                        <div className="mt-1.5 grid grid-cols-4 gap-1 text-center text-[12px] text-ink-3">
                          <span className="rounded-md bg-card px-1.5 py-1">{remoteWork.jobs.visible}/{remoteWork.jobs.total} {tx("jobs")}</span>
                          <span className={clsx('rounded-md px-1.5 py-1', remoteBlocked ? 'bg-danger/10 text-danger' : 'bg-card')}>{remoteBlocked} {tx("blocked")}</span>
                          <span className={clsx('rounded-md px-1.5 py-1', remoteWork.approvals.pending ? 'bg-warn/12 text-warn' : 'bg-card')}>{remoteWork.approvals.pending} {tx("approvals")}</span>
                          <span className={clsx('rounded-md px-1.5 py-1', remoteLeaseRisk ? 'bg-warn/12 text-warn' : 'bg-card')}>{remoteLeaseRisk} {tx("lease risk")}</span>
                        </div>
                      )}
                      {remoteWork?.nextWakeAt && (
                        <div className="mt-1.5 truncate rounded-md bg-card px-2 py-1 text-[12px] text-ink-4">
                          {tx("Next wake ")}{wakeTime(remoteWork.nextWakeAt)}
                        </div>
                      )}
                      {remoteProjects.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {remoteProjects.map((project) => (
                            <div key={project.id} className="flex items-center gap-2 rounded-md bg-card px-2 py-1.5 text-[12px] text-ink-3">
                              <Icon name="globe" size={12} className="shrink-0 text-accent" />
                              <span className="min-w-0 flex-1 truncate text-ink-2">{project.name}</span>
                              <span className="max-w-[120px] truncate font-mono text-[11px] text-ink-4">{project.cwd ? shortPath(project.cwd, 2) : tx("no folder")}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {remoteJobs.length > 0 && (
                        <div className="mt-1.5 space-y-1">
                          {remoteJobs.map((job) => (
                            <div key={job.id} className="flex items-center gap-2 rounded-md bg-card px-2 py-1.5 text-[12px] text-ink-3">
                              <Icon name="globe" size={12} className="shrink-0 text-accent" />
                              <span className="min-w-0 flex-1 truncate">{jobTitle(job)}</span>
                              <span className={clsx('shrink-0 rounded px-1.5 py-0.5 text-[11px]', job.status === 'failed' || job.status === 'dead_letter' ? 'bg-danger/10 text-danger' : job.status === 'stalled' || job.status === 'waiting_approval' ? 'bg-warn/12 text-warn' : 'bg-ink-5 text-ink-3')}>{job.status.replace(/_/g, ' ')}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="mt-1.5 line-clamp-2 text-[13px] text-ink-3">{result?.error || host.lastError || tx("Refresh to fetch live state.")}</div>
                  )}
                </div>
                <Button variant="ghost" size="xs" icon="clock" aria-label={tx("Refresh {value0}", { value0: host.name })} onClick={() => void refreshRemoteHost(host.id)} disabled={remoteBusy === host.id}>
                  {remoteBusy === host.id ? tx("Checking") : tx("Refresh")}
                </Button>
              </div>
            </div>
          )
        })}
        <Button variant="default" size="xs" icon="settings" onClick={() => setView({ name: 'settings' })}>{tx("Settings")}</Button>
      </div>
    </DisclosurePanel>
  )
}
