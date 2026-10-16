import { formatDuration, relativeTime } from '../../lib/format'
import type { IconName } from '../../lib/icons'
import type { AssistantControlSnapshot, Automation, Job, Mission, MissionContextMode, MissionRepeatMode, MissionStatus, Room, Settings } from '../../lib/types'
import { ACTIONABLE_JOB_STATUSES, CONNECTOR_STALE_MS, LEASE_EXPIRING_SOON_MS, RUNNING_ROOM_STATES, VISIBLE_JOB_STATUSES, cap, type ConnectorHealthSignal, type LiveJob, type PowerGuardSignal, type RunningWorkSummary, type Tone, type WorkLaneSummary } from './model'
import { missionIsActive } from './attention'
import { scheduleTime } from './brief'

export function emptyJobStatusCounts(): Record<Job['status'], number> {
  return {
    queued: 0,
    running: 0,
    waiting_approval: 0,
    paused: 0,
    stalled: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
    dead_letter: 0,
  }
}

export function summarizeRunningWork(jobs: Job[], now = Date.now()): RunningWorkSummary {
  const statusCounts = emptyJobStatusCounts()
  const kindCounts = new Map<Job['kind'], number>()
  const leases = { active: 0, expired: 0, expiringSoon: 0, missingRunning: 0 }
  const attempts = { total: 0, retried: 0, retryable: 0, max: 0 }
  let visibleTotal = 0
  let unresolvedApprovals = 0
  let nextWake: RunningWorkSummary['nextWake'] = null

  for (const job of jobs) {
    statusCounts[job.status] += 1
    if (!VISIBLE_JOB_STATUSES.has(job.status)) continue

    visibleTotal += 1
    kindCounts.set(job.kind, (kindCounts.get(job.kind) ?? 0) + 1)
    unresolvedApprovals += (job.pendingApprovals ?? []).filter((approval) => !approval.resolvedAt).length
    attempts.total += job.attempt
    attempts.max = Math.max(attempts.max, job.attempt)
    if (job.attempt > 1) attempts.retried += 1
    if (ACTIONABLE_JOB_STATUSES.has(job.status)) attempts.retryable += 1

    if (job.nextWakeAt && (!nextWake || job.nextWakeAt < nextWake.at)) nextWake = { job, at: job.nextWakeAt }

    if (job.leaseExpiresAt) {
      if (job.leaseExpiresAt <= now) leases.expired += 1
      else {
        leases.active += 1
        if (job.leaseExpiresAt - now <= LEASE_EXPIRING_SOON_MS) leases.expiringSoon += 1
      }
    } else if (job.status === 'running') {
      leases.missingRunning += 1
    }
  }

  return {
    total: jobs.length,
    visibleTotal,
    statusCounts,
    kindCounts: [...kindCounts.entries()]
      .map(([kind, count]) => ({ kind, count }))
      .sort((a, b) => b.count - a.count || a.kind.localeCompare(b.kind))
      .slice(0, 6),
    nextWake,
    leases,
    attempts,
    unresolvedApprovals,
  }
}

export function summarizeWorkLanes(liveJobs: LiveJob[], durableJobs: Job[]): WorkLaneSummary[] {
  const liveCount = (prefix: string) => liveJobs.filter((job) => job.id.startsWith(prefix)).length
  const durableCount = (kind: Job['kind']) => durableJobs.filter((job) => job.kind === kind && VISIBLE_JOB_STATUSES.has(job.status)).length
  const unifiedCount = (prefix: string, kind: Job['kind']) => Math.max(liveCount(prefix), durableCount(kind))
  const lane = (id: string, label: string, count: number, tone: Tone = 'normal'): WorkLaneSummary => ({
    id,
    label,
    count,
    tone: count ? tone : 'low',
  })
  return [
    lane('threads', 'Threads', unifiedCount('thread:', 'thread')),
    lane('rooms', 'Rooms', unifiedCount('room:', 'room')),
    lane('tickets', 'Tickets', unifiedCount('ticket:', 'ticket')),
    lane('automations', 'Automations', unifiedCount('automation:', 'automation')),
    lane('missions', 'Missions', unifiedCount('mission:', 'mission')),
    lane('connectors', 'Connectors', durableCount('connector')),
    lane('pr-loops', 'PR loops', durableCount('pr_loop')),
  ]
}

export function countLongWorkSignals(jobs: Job[], missions: Mission[], rooms: Room[], roomStatus: Record<string, Room['state'] | undefined>, automations: Automation[]) {
  const activeJobStatuses = new Set<Job['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled'])
  const countedMissions = new Set<string>()
  const countedRooms = new Set<string>()
  const countedAutomations = new Set<string>()
  let count = 0

  for (const job of jobs) {
    if (!activeJobStatuses.has(job.status)) continue
    if (!['room', 'mission', 'automation', 'pr_loop', 'connector', 'ticket'].includes(job.kind)) continue
    count += 1
    if (job.missionId) countedMissions.add(job.missionId)
    if (job.roomId) countedRooms.add(job.roomId)
    if (job.automationId) countedAutomations.add(job.automationId)
  }

  for (const mission of missions) {
    if (missionIsActive(mission.status) && !countedMissions.has(mission.id)) count += 1
  }
  for (const room of rooms) {
    const state = roomStatus[room.id] ?? room.state
    if (RUNNING_ROOM_STATES.has(state) && !countedRooms.has(room.id)) count += 1
  }
  for (const automation of automations) {
    if (automation.runs[0]?.status === 'running' && !countedAutomations.has(automation.id)) count += 1
  }
  return count
}

export function powerGuardSignal(power: Settings['power'] | null | undefined, longWorkCount: number, nextWakeAt: number | null, leaseTone: Tone): PowerGuardSignal {
  if (!power) {
    return {
      label: 'Power unknown',
      detail: 'Settings have not loaded yet.',
      nextAction: 'Refresh settings before starting long work.',
      tone: 'low',
    }
  }
  if (power.lastError) {
    return {
      label: 'Awake guard error',
      detail: power.lastError,
      nextAction: 'Open Settings and repair the keep-awake guard.',
      tone: 'critical',
    }
  }
  if (!power.keepComputerAlive && longWorkCount > 0) {
    return {
      label: 'Sleep risk',
      detail: `${longWorkCount} long-running item${longWorkCount === 1 ? '' : 's'} may be interrupted if this Mac sleeps.`,
      nextAction: 'Turn on Keep computer alive before leaving work running.',
      tone: 'high',
    }
  }
  if (power.keepComputerAlive) {
    const started = power.lastStartedAt ? `Started ${relativeTime(power.lastStartedAt)}.` : 'Waiting for caffeinate start confirmation.'
    return {
      label: 'Awake guard on',
      detail: longWorkCount ? `${started} Protecting ${longWorkCount} long-running item${longWorkCount === 1 ? '' : 's'}.` : `${started} No long work is currently active.`,
      nextAction: nextWakeAt ? 'Leave the guard on until the next wake completes.' : 'Leave on for upcoming overnight or local runs, or disable when idle.',
      tone: leaseTone === 'critical' ? 'high' : longWorkCount ? 'good' : 'normal',
    }
  }
  return {
    label: 'Awake guard off',
    detail: 'No long-running work is active, so sleep prevention is not needed right now.',
    nextAction: 'Enable it before overnight rooms, missions, or local model jobs.',
    tone: 'low',
  }
}

export function jobTitle(job: Job): string {
  if (typeof job.metadata?.title === 'string') return job.metadata.title
  if (typeof job.metadata?.name === 'string') return job.metadata.name
  return `${cap(job.kind)} job`
}

export function jobIcon(job: Job): IconName {
  if (job.kind === 'room') return 'layers'
  if (job.kind === 'automation') return 'clock'
  if (job.kind === 'mission') return 'spark'
  if (job.kind === 'ticket') return 'file'
  if (job.kind === 'connector') return 'plug'
  if (job.kind === 'pr_loop') return 'branch'
  return 'terminal'
}

export function jobTone(job: Job): Tone {
  if (job.status === 'failed' || job.status === 'dead_letter') return 'critical'
  if (job.status === 'stalled' || job.status === 'waiting_approval') return 'high'
  if (job.status === 'completed') return 'good'
  if (job.status === 'queued' || job.status === 'paused' || job.status === 'cancelled') return 'low'
  return 'normal'
}

export function wakeTime(ts: number) {
  return ts <= Date.now() ? `Due ${relativeTime(ts)}` : scheduleTime(ts)
}

export function leaseTime(ts: number) {
  return ts <= Date.now() ? `expired ${relativeTime(ts)}` : `until ${scheduleTime(ts)}`
}

export function leaseSummaryText(summary: RunningWorkSummary) {
  const parts: string[] = []
  if (summary.leases.expired) parts.push(`${summary.leases.expired} expired`)
  if (summary.leases.expiringSoon) parts.push(`${summary.leases.expiringSoon} expiring`)
  if (summary.leases.missingRunning) parts.push(`${summary.leases.missingRunning} missing`)
  return parts.join(' / ') || 'No lease risk'
}

export function leaseSummaryTone(summary: RunningWorkSummary): Tone {
  if (summary.leases.expired || summary.leases.missingRunning) return 'critical'
  if (summary.leases.expiringSoon) return 'high'
  if (summary.leases.active) return 'normal'
  return 'good'
}

export function durableJobFootnote(job: Job): string {
  const parts = [`attempt ${job.attempt}`]
  const unresolved = (job.pendingApprovals ?? []).filter((approval) => !approval.resolvedAt).length
  if (unresolved) parts.push(`${unresolved} approval${unresolved === 1 ? '' : 's'}`)
  if (job.nextWakeAt) parts.push(`wake ${wakeTime(job.nextWakeAt)}`)
  if (job.leaseOwner && job.leaseExpiresAt) parts.push(`lease ${job.leaseOwner} ${leaseTime(job.leaseExpiresAt)}`)
  else if (job.leaseExpiresAt) parts.push(`lease ${leaseTime(job.leaseExpiresAt)}`)
  if (job.resumeAfterRestart !== 'ask') parts.push(`restart ${job.resumeAfterRestart.replace(/_/g, '-')}`)
  return parts.join(' / ')
}

export function missionRepeatLabel(mode: MissionRepeatMode) {
  const labels: Record<MissionRepeatMode, string> = {
    until_condition: 'Until condition',
    interval: 'Interval',
    self_paced: 'Self-paced',
    fixed_count: 'Fixed count',
    forever: 'Forever',
  }
  return labels[mode] ?? mode
}

export function missionContextLabel(mode: MissionContextMode) {
  const labels: Record<MissionContextMode, string> = {
    fresh: 'Fresh',
    build_on_previous: 'Build previous',
    fresh_with_notes: 'Fresh + notes',
  }
  return labels[mode] ?? mode
}

export function missionTone(status: MissionStatus): Tone {
  if (status === 'failed' || status === 'stalled') return 'critical'
  if (status === 'waiting_approval') return 'high'
  if (status === 'completed') return 'good'
  if (status === 'paused' || status === 'queued' || status === 'cancelled') return 'low'
  return 'normal'
}

export function connectorHealthSignal(connector: AssistantControlSnapshot['connectors'][number], settings: Settings | null, now: number): ConnectorHealthSignal {
  if (connector.status === 'error') {
    const detail = connector.error || connector.description || 'Connector reported an error.'
    return {
      label: 'Failed',
      detail,
      nextAction: connector.id === 'email' ? 'Retry local source sync.' : 'Open Sources and repair the connector configuration.',
      tone: 'critical',
      action: connector.id === 'email' ? 'sync-email' : 'open-sources',
    }
  }

  if (connector.pendingApprovals > 0) {
    return {
      label: 'Approval',
      detail: `${connector.pendingApprovals} pending action${connector.pendingApprovals === 1 ? '' : 's'} waiting before this connector can continue.`,
      nextAction: 'Review pending connector approvals.',
      tone: 'high',
      action: 'review-approvals',
    }
  }

  if (connector.status === 'pending') {
    return {
      label: 'Not connected',
      detail: connector.description,
      nextAction: 'Configure this connector in Sources.',
      tone: 'high',
      action: 'open-sources',
    }
  }

  if (connector.id === 'email') {
    const enabledSources = settings?.email.sources.filter((source) => source.enabled) ?? []
    const failedSources = enabledSources.filter((source) => source.lastError)
    if (failedSources.length) {
      return {
        label: 'Failed sync',
        detail: `${failedSources.length}/${enabledSources.length} email source${enabledSources.length === 1 ? '' : 's'} reported an error: ${failedSources[0].lastError}`,
        nextAction: 'Retry email sync or inspect the source path.',
        tone: 'critical',
        action: 'sync-email',
      }
    }
    const lastSyncAt = settings?.email.lastSyncAt ?? connector.lastEventAt
    if (!lastSyncAt) {
      return {
        label: 'Unsynced',
        detail: enabledSources.length ? 'Email has sources, but no sync has completed yet.' : 'No email source is connected.',
        nextAction: enabledSources.length ? 'Run the first email sync.' : 'Add an email source.',
        tone: enabledSources.length ? 'high' : 'low',
        action: enabledSources.length ? 'sync-email' : 'open-sources',
      }
    }
    if (now - lastSyncAt > CONNECTOR_STALE_MS) {
      return {
        label: 'Stale',
        detail: `Last email sync ${relativeTime(lastSyncAt)}.`,
        nextAction: 'Sync email to refresh inbox state.',
        tone: 'high',
        action: 'sync-email',
      }
    }
  }

  if (connector.lastEventAt && now - connector.lastEventAt > CONNECTOR_STALE_MS && connector.kind !== 'system') {
    return {
      label: 'Stale',
      detail: `Last connector activity ${relativeTime(connector.lastEventAt)}.`,
      nextAction: 'Open Sources and refresh or test this connector.',
      tone: 'high',
      action: 'open-sources',
    }
  }

  if (!connector.lastEventAt && connector.status === 'connected' && connector.kind !== 'system') {
    return {
      label: 'No events',
      detail: 'Connector is configured, but no source event has been recorded yet.',
      nextAction: connector.id === 'email' ? 'Sync email to pull source events.' : 'Open Sources and run a connector check.',
      tone: 'low',
      action: connector.id === 'email' ? 'sync-email' : 'open-sources',
    }
  }

  return {
    label: connector.status === 'mock' ? 'Contract' : 'Healthy',
    detail: connector.lastEventAt ? `Last activity ${relativeTime(connector.lastEventAt)}.` : 'No immediate connector action is needed.',
    nextAction: 'No action needed.',
    tone: 'good',
    action: 'none',
  }
}
