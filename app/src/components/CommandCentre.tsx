import { useCallback, useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { relativeTime, formatDuration, shortPath } from '../lib/format'
import { requestSettingsSection, useStore } from '../lib/store'
import { buildAwayBriefItems } from '../lib/away-brief'
import { buildNodeDashboard } from '../lib/node-dashboard'
import { summarizeTerminalActivity } from '../lib/terminal-sessions'
import { buildReadinessDoctor, listReadinessProviderProbes, summarizeDoctor, worstDoctorStatus, type DoctorStatus, type ProbeState } from '../lib/readiness-doctor'
import type { PersistedProviderStatus,
  AssistantControlSnapshot,
  Automation,
  CliDefinition,
  CliToolStatus,
  ConnectorAttentionItem,
  ConnectorEvent,
  ConnectorId,
  ConnectorPriority,
  ConnectorStatus,
  ContextPressureSnapshot,
  Job,
  ModelRouterSnapshot,
  Mission,
  MissionContextMode,
  MissionInspectionSnapshot,
  AgentRoleProfileId,
  FanOutBudget,
  MissionRepeatMode,
  MissionRuntimeSnapshot,
  MissionStatus,
  RemoteOrchestratorSnapshotResult,
  RunBudgetClass,
  Room,
  Ticket,
  ApprovalRule,
  Provider,
  Readiness,
  RemoteControlStatus,
  Settings,
  Thread,
  TokenUsageSnapshot,
} from '../lib/types'
import { Button } from './ui'
import { Icon, type IconName } from '../lib/icons'
import { BrandIcon, connectorBrand } from '../lib/brand'
import { CommandCentreLayout } from './command-centre/layout'
import {
  approvalRuleIcon,
  approvalRuleTitle,
  attentionFromConnectorEvent,
  attentionFromConnectorFeed,
  awayItemToRow,
  connectorName,
  groupApprovalRows,
  missionIsActive,
  missionIsTerminal,
  missionStatusLabel,
  projectName,
  roomStatusLabel,
  sortRows,
  sourceIcon,
  threadStatusLabel,
  ticketNeedsAttention,
} from './command-centre/attention'
import { briefOutputLines, briefSchedule, scheduleTime } from './command-centre/brief'
import { buildProviderVisibilityRows } from './command-centre/provider'
import { connectorHealthBadgeClass, toneBadge, toneDot, toneText } from './command-centre/styles'
import {
  countLongWorkSignals,
  connectorHealthSignal,
  durableJobFootnote,
  jobIcon,
  jobTitle,
  jobTone,
  leaseSummaryText,
  leaseSummaryTone,
  missionContextLabel,
  missionRepeatLabel,
  missionTone,
  powerGuardSignal,
  summarizeRunningWork,
  summarizeWorkLanes,
  wakeTime,
} from './command-centre/work'
import {
  ACTIONABLE_JOB_STATUSES,
  AWAY_BRIEF_WINDOW_MS,
  DISCLOSURE_AUTO_EXPAND_THRESHOLD,
  FAN_OUT_OPTIONS,
  ROLE_PROFILE_OPTIONS,
  RUNNING_ROOM_STATES,
  RUNNING_THREAD_STATES,
  TONE_RANK,
  cap,
  computeCommandStatusSummary,
  disclosureDefaultOpen,
  summaryDefaultExpanded,
  worstTone,
  type ApprovalRow,
  type AttentionRow,
  type AwayBriefRow,
  type LiveJob,
  type Tone,
} from './command-centre/model'

function doctorStatusTone(status: DoctorStatus): Tone {
  if (status === 'error') return 'critical'
  if (status === 'warning') return 'high'
  if (status === 'checking') return 'low'
  return 'good'
}

export function CommandCentre() {
  const threads = useStore((s) => s.threads)
  const projects = useStore((s) => s.projects)
  const rooms = useStore((s) => s.rooms)
  const roomStatus = useStore((s) => s.roomStatus)
  const tickets = useStore((s) => s.tickets)
  const durableJobs = useStore((s) => s.jobs)
  const automations = useStore((s) => s.automations)
  const status = useStore((s) => s.status)
  const settings = useStore((s) => s.settings)
  const approvals = useStore((s) => s.approvals)
  const terminals = useStore((s) => s.terminals)
  const setView = useStore((s) => s.setView)
  const openThread = useStore((s) => s.openThread)
  const openRoom = useStore((s) => s.openRoom)
  const approve = useStore((s) => s.approve)
  const revokeApprovalRule = useStore((s) => s.revokeApprovalRule)
  const updateTicket = useStore((s) => s.updateTicket)
  const workTicket = useStore((s) => s.workTicket)
  const retryJob = useStore((s) => s.retryJob)
  const resumeJob = useStore((s) => s.resumeJob)
  const deadLetterJob = useStore((s) => s.deadLetterJob)
  const runAutomation = useStore((s) => s.runAutomation)
  const cancel = useStore((s) => s.cancel)
  const cancelRoom = useStore((s) => s.cancelRoom)
  const syncRemoteProjects = useStore((s) => s.syncRemoteProjects)

  const [snapshot, setSnapshot] = useState<AssistantControlSnapshot | null>(null)
  const [providerStatusSnapshot, setProviderStatusSnapshot] = useState<PersistedProviderStatus | null>(null)
  const [missionSnapshot, setMissionSnapshot] = useState<MissionRuntimeSnapshot | null>(null)
  const [missionInspection, setMissionInspection] = useState<MissionInspectionSnapshot | null>(null)
  const [missionInspectingId, setMissionInspectingId] = useState<string | null>(null)
  const [contextSnapshot, setContextSnapshot] = useState<ContextPressureSnapshot | null>(null)
  const [modelRouterSnapshot, setModelRouterSnapshot] = useState<ModelRouterSnapshot | null>(null)
  const [usageSnapshot, setUsageSnapshot] = useState<TokenUsageSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [missionTitle, setMissionTitle] = useState('')
  const [missionObjective, setMissionObjective] = useState('')
  const [missionBudget, setMissionBudget] = useState<RunBudgetClass>('mission')
  const [missionRepeat, setMissionRepeat] = useState<MissionRepeatMode>('until_condition')
  const [missionStopCondition, setMissionStopCondition] = useState('')
  const [missionContext, setMissionContext] = useState<MissionContextMode>('build_on_previous')
  const [missionRoleProfile, setMissionRoleProfile] = useState<AgentRoleProfileId>('operator')
  const [missionFanOut, setMissionFanOut] = useState<FanOutBudget>('single')
  const [missionMaxRuns, setMissionMaxRuns] = useState('15')
  const [missionCostCap, setMissionCostCap] = useState('25')
  const [missionExpiresDays, setMissionExpiresDays] = useState('3')
  const [missionGateDestructive, setMissionGateDestructive] = useState(true)
  const [prTitle, setPrTitle] = useState('')
  const [prObjective, setPrObjective] = useState('')
  const [prProjectId, setPrProjectId] = useState('')
  const [prApprovals, setPrApprovals] = useState('1')
  const [prAutoStart, setPrAutoStart] = useState(true)
  const [prPublishRemote, setPrPublishRemote] = useState(false)
  const [prBusy, setPrBusy] = useState(false)
  const [readinessProviderProbes, setReadinessProviderProbes] = useState<Record<string, ProbeState<Readiness>>>({})
  const [readinessLocalClis, setReadinessLocalClis] = useState<ProbeState<Array<CliDefinition | CliToolStatus>>>({ status: 'checking' })
  const [readinessStudioStatus, setReadinessStudioStatus] = useState<RemoteControlStatus | null>(null)
  const [readinessRefreshNonce, setReadinessRefreshNonce] = useState(0)
  const [remoteSnapshots, setRemoteSnapshots] = useState<Record<string, RemoteOrchestratorSnapshotResult>>({})
  const [remoteBusy, setRemoteBusy] = useState<string | null>(null)
  const [revokingRuleId, setRevokingRuleId] = useState<string | null>(null)
  // Declutter disclosures (Task #62). null = follow the auto/default; boolean = explicit user choice.
  const [statusExpandedOverride, setStatusExpandedOverride] = useState<boolean | null>(null)
  const [awayOpenOverride, setAwayOpenOverride] = useState<boolean | null>(null)
  const [orchestratorsOpenOverride, setOrchestratorsOpenOverride] = useState<boolean | null>(null)
  const [missionFormsOpen, setMissionFormsOpen] = useState(false)
  const [showAllMissions, setShowAllMissions] = useState(false)
  // Relative ages and heartbeat freshness are live UI, not snapshot text. Keep a small
  // clock running while Command Centre is mounted so labels and stale thresholds advance.
  const [dashboardNow, setDashboardNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setDashboardNow(Date.now()), 15_000)
    return () => window.clearInterval(timer)
  }, [])
  const orchestrators = useMemo(() => (settings?.remoteOrchestrators ?? []).filter((host) => host.enabled), [settings?.remoteOrchestrators])
  const orchestratorIdsKey = useMemo(() => orchestrators.map((host) => host.id).join(','), [orchestrators])
  const projectById = useMemo(() => new Map(projects.map((project) => [project.id, project])), [projects])
  const readinessProviderEntries = useMemo(() => settings ? listReadinessProviderProbes(settings) : [], [settings?.providers])

  const refreshRemoteHost = useCallback(async (hostId: string) => {
    setRemoteBusy(hostId)
    try {
      const result = await api.getRemoteSnapshot(hostId)
      setRemoteSnapshots((current) => ({ ...current, [hostId]: result }))
      if (result.ok && result.projects?.length) {
        try {
          await syncRemoteProjects(hostId, result.projects)
        } catch (err: any) {
          setError(err?.message ?? 'Could not sync remote projects.')
        }
      }
    } finally {
      setRemoteBusy(null)
    }
  }, [syncRemoteProjects])

  const refreshReadiness = useCallback(() => {
    setReadinessRefreshNonce((value) => value + 1)
    for (const host of orchestrators) void refreshRemoteHost(host.id)
  }, [orchestrators, refreshRemoteHost])

  const openReadinessSettings = useCallback(() => {
    requestSettingsSection('readiness')
    setView({ name: 'settings', section: 'readiness' })
  }, [setView])

  const refresh = async () => {
    setLoading(true)
    setError('')
    setReadinessRefreshNonce((value) => value + 1)
    try {
      const [assistant, missions, context, routes, usage, providerStatus] = await Promise.all([
        api.getAssistantSnapshot(),
        api.getMissionSnapshot(),
        api.getContextPressure().catch(() => null),
        api.getModelRouterSnapshot().catch(() => null),
        api.getTokenUsage().catch(() => null),
        api.getProviderStatus().catch(() => null),
      ])
      setSnapshot(assistant)
      setMissionSnapshot(missions)
      setMissionInspection((current) => current && missions.missions.some((mission) => mission.id === current.mission.id) ? current : null)
      setContextSnapshot(context)
      setModelRouterSnapshot(routes)
      setUsageSnapshot(usage)
      setProviderStatusSnapshot(providerStatus)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const refreshMissions = async () => setMissionSnapshot(await api.getMissionSnapshot())
  const inspectMission = async (id: string) => {
    if (missionInspection?.mission.id === id) {
      setMissionInspection(null)
      return
    }
    setMissionInspectingId(id)
    try {
      setMissionInspection(await api.inspectMission(id))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setMissionInspectingId(null)
    }
  }
  const parseOptionalPositiveNumber = (label: string, raw: string): number | null => {
    if (!raw.trim()) return null
    const value = Number(raw)
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be a positive number.`)
    return value
  }
  const parseOptionalPositiveInteger = (label: string, raw: string): number | null => {
    const value = parseOptionalPositiveNumber(label, raw)
    return value == null ? null : Math.round(value)
  }
  const createMissionFromDraft = async () => {
    const objective = missionObjective.trim()
    if (!objective) return
    setError('')
    let maxRuns: number | null
    let costCapUsd: number | null
    let expiresDays: number | null
    try {
      maxRuns = parseOptionalPositiveInteger('Max runs', missionMaxRuns)
      costCapUsd = parseOptionalPositiveNumber('Cost cap', missionCostCap)
      expiresDays = parseOptionalPositiveInteger('Expires days', missionExpiresDays)
    } catch (err: any) {
      setError(err?.message ?? 'Mission limits are invalid.')
      return
    }
    await api.createMission({
      title: missionTitle.trim() || objective.slice(0, 64),
      objective,
      stopCondition: missionStopCondition.trim(),
      repeatMode: missionRepeat,
      contextMode: missionContext,
      roleProfileId: missionRoleProfile,
      fanOutBudget: missionFanOut,
      maxRuns,
      costCapUsd,
      expiresAt: expiresDays ? Date.now() + expiresDays * 24 * 60 * 60_000 : null,
      destructiveApprovalRequired: missionGateDestructive,
      runBudgetClass: missionBudget,
    })
    setMissionTitle('')
    setMissionObjective('')
    setMissionStopCondition('')
    setMissionRoleProfile('operator')
    setMissionFanOut('single')
    setMissionRepeat('until_condition')
    setMissionContext('build_on_previous')
    setMissionMaxRuns('15')
    setMissionCostCap('25')
    setMissionExpiresDays('3')
    setMissionGateDestructive(true)
    setMissionBudget('mission')
    await refreshMissions()
  }
  const createLocalPrLoop = async () => {
    const objective = prObjective.trim()
    if (!objective) return
    setError('')
    let requiredApprovals: number | null
    try {
      requiredApprovals = parseOptionalPositiveInteger('Review passes', prApprovals) ?? 1
    } catch (err: any) {
      setError(err?.message ?? 'Review pass count is invalid.')
      return
    }
    setPrBusy(true)
    try {
      const result = await api.createLocalPrLoop({
        title: prTitle.trim() || objective.slice(0, 64),
        objective,
        projectId: prProjectId || null,
        requiredApprovals,
        autoStart: prAutoStart,
        publishRemote: prPublishRemote,
      })
      setPrTitle('')
      setPrObjective('')
      setPrProjectId('')
      setPrApprovals('1')
      setPrAutoStart(true)
      setPrPublishRemote(false)
      await refreshMissions()
      openThread(result.implementationThread.id)
    } catch (err: any) {
      setError(err?.message ?? 'Failed to create local PR loop.')
    } finally {
      setPrBusy(false)
    }
  }
  const setMission = async (id: string, status: MissionStatus, message?: string) => {
    await api.setMissionStatus(id, status, message)
    await refreshMissions()
  }
  const beatMission = async (id: string) => {
    await api.recordMissionHeartbeat(id, 'Manual heartbeat from Command Centre.')
    await refreshMissions()
  }
  const syncEmailAndRefresh = async () => {
    await api.syncEmail()
    await refresh()
  }

  useEffect(() => { void refresh() }, [])
  useEffect(() => api.onUsageCleared?.(() => setUsageSnapshot(null)), [])
  useEffect(() => {
    for (const host of orchestrators) void refreshRemoteHost(host.id)
  }, [orchestratorIdsKey, refreshRemoteHost])
  useEffect(() => {
    let cancelled = false
    const initial = Object.fromEntries(readinessProviderEntries.map((entry) => [entry.key, { status: 'checking' as const }]))
    setReadinessProviderProbes(initial)
    for (const entry of readinessProviderEntries) {
      api.getModelReadiness({ providerId: entry.providerId, accountId: entry.accountId })
        .then((value) => {
          if (!cancelled) setReadinessProviderProbes((current) => ({ ...current, [entry.key]: { status: 'ok', value } }))
        })
        .catch((error) => {
          if (!cancelled) setReadinessProviderProbes((current) => ({ ...current, [entry.key]: { status: 'error', message: String(error?.message ?? error) } }))
        })
    }
    return () => { cancelled = true }
  }, [readinessProviderEntries, readinessRefreshNonce])
  useEffect(() => {
    let cancelled = false
    setReadinessLocalClis({ status: 'checking' })
    // Prefer the probed tool statuses (installed/version/effective command); fall back to the
    // plain registry so the doctor can still render 'not probed yet' entries.
    api.listCliTools()
      .then((value) => { if (!cancelled) setReadinessLocalClis({ status: 'ok', value }) })
      .catch(() => api.listClis()
        .then((value) => { if (!cancelled) setReadinessLocalClis({ status: 'ok', value }) })
        .catch((error) => { if (!cancelled) setReadinessLocalClis({ status: 'error', message: String(error?.message ?? error) }) }))
    return () => { cancelled = true }
  }, [readinessRefreshNonce])
  useEffect(() => {
    let cancelled = false
    api.getRemoteStatus()
      .then((value) => { if (!cancelled) setReadinessStudioStatus(value) })
      .catch(() => { if (!cancelled) setReadinessStudioStatus(null) })
    return () => { cancelled = true }
  }, [settings?.remoteControl.enabled, settings?.remoteControl.port, readinessRefreshNonce])
  // Live mission updates (executor heartbeats + status transitions push mission:update).
  useEffect(() => api.onMission(() => { void refreshMissions() }), [])
  // Additive event-driven path: subscribe to the orchestration broker so the mission/job
  // list reflects lifecycle transitions (mission created/started/checkpoint/finalized,
  // room started/turn/finalized, job status_changed) the instant they happen, instead of
  // only at the next poll. The periodic refresh + snapshot reconcile above stays as the
  // fallback, so a missed event never leaves the view stale. Re-pull the mission snapshot
  // on any event (cheap, and the snapshot is the Command Centre's source of truth for the
  // mission lane); jobs/rooms already mirror into the store via their own push channels.
  useEffect(() => api.onOrchestrationEvent(() => { void refreshMissions() }), [])

  const attentionRows = useMemo(() => {
    const rows: AttentionRow[] = []
    if (snapshot?.attention?.length) {
      for (const item of snapshot.attention) rows.push(attentionFromConnectorFeed(item))
    } else {
      for (const event of snapshot?.events ?? []) rows.push(attentionFromConnectorEvent(event))
    }
    for (const ticket of tickets) {
      if (!ticketNeedsAttention(ticket)) continue
      rows.push({
        id: `ticket:${ticket.id}`,
        icon: ticket.status === 'needs_clarification' ? 'warning' : 'check',
        source: projectName(projects, ticket.projectId),
        title: ticket.title,
        detail: ticket.body || ticket.tags.join(', ') || 'Ticket is waiting in the work queue.',
        tone: ticket.priority === 'urgent' ? 'critical' : ticket.priority === 'high' ? 'high' : ticket.status === 'review' ? 'good' : 'normal',
        age: relativeTime(ticket.updatedAt),
        action: ticket.status === 'review' ? 'Review' : ticket.status === 'needs_clarification' ? 'Clarify' : ticket.runState === 'running' ? 'Running' : 'Dispatch',
        createdAt: ticket.updatedAt,
        onOpen: () => {
          if (ticket.threadId) openThread(ticket.threadId)
          else if (ticket.roomId) openRoom(ticket.roomId)
          else void workTicket(ticket.id)
        },
      })
    }
    for (const automation of automations) {
      const lastRun = automation.runs[0]
      if (lastRun?.status !== 'error') continue
      rows.push({
        id: `automation:${automation.id}`,
        icon: 'warning',
        source: 'Automation',
        title: automation.name,
        detail: lastRun.error || automation.prompt,
        tone: 'critical',
        age: relativeTime(lastRun.finishedAt ?? lastRun.startedAt),
        action: 'Inspect',
        createdAt: lastRun.finishedAt ?? lastRun.startedAt,
        onOpen: () => setView({ name: 'automations' }),
      })
    }
    return rows.sort(sortRows).slice(0, 12)
  }, [snapshot, tickets, projects, automations, openThread, openRoom, workTicket, setView, dashboardNow])

  const selected = attentionRows.find((row) => row.id === selectedId) ?? attentionRows[0] ?? null

  useEffect(() => {
    if (!selectedId && attentionRows[0]) setSelectedId(attentionRows[0].id)
  }, [attentionRows, selectedId])

  const approvalRows = useMemo(() => {
    const rows: ApprovalRow[] = []
    const liveApprovalCallIds = new Set<string>()
    for (const [threadId, pending] of Object.entries(approvals)) {
      const thread = threads.find((t) => t.id === threadId)
      for (const approval of pending) {
        const item = thread?.timeline.find((i) => i.id === approval.itemId)
        const requestId = item?.approvalId ?? item?.id ?? approval.callId
        liveApprovalCallIds.add(requestId)
        rows.push({
          id: `tool:${requestId}`,
          icon: 'terminal',
          sourceGroup: 'Tool calls',
          title: item?.title || item?.tool || 'Tool approval',
          detail: item?.command || item?.path || item?.meta || thread?.title || 'Agent tool call is waiting for a decision.',
          meta: thread?.title ?? 'Thread',
          tone: 'critical',
          createdAt: item?.createdAt ?? thread?.updatedAt ?? Date.now(),
          primaryLabel: 'Allow',
          secondaryLabel: 'Deny',
          onPrimary: () => void approve(threadId, requestId, 'allow'),
          onSecondary: () => void approve(threadId, requestId, 'deny'),
        })
      }
    }
    for (const job of durableJobs) {
      if (job.missionId) continue
      for (const approval of job.pendingApprovals ?? []) {
        if (approval.resolvedAt) continue
        const threadId = approval.threadId ?? job.currentThreadId ?? job.threadId ?? ''
        const thread = threadId ? threads.find((t) => t.id === threadId) : null
        const item = approval.itemId ? thread?.timeline.find((entry) => entry.id === approval.itemId) : null
        const requestId = item?.approvalId ?? approval.itemId ?? approval.id ?? approval.callId
        if (liveApprovalCallIds.has(requestId)) continue
        rows.push({
          id: `job-approval:${job.id}:${approval.id}`,
          icon: 'terminal',
          sourceGroup: 'Restored jobs',
          title: item?.title || approval.policy.label || approval.tool || 'Restored tool approval',
          detail: item?.command || item?.path || approval.policy.explanation || job.blockedReason || 'Approval survived restart and is waiting for a decision.',
          meta: thread?.title ?? job.owner,
          tone: 'critical',
          createdAt: approval.requestedAt,
          primaryLabel: 'Allow',
          secondaryLabel: 'Deny',
          onPrimary: () => void approve(threadId, requestId, 'allow'),
          onSecondary: () => void approve(threadId, requestId, 'deny'),
        })
      }
    }
    for (const event of (snapshot?.events ?? []).filter((e) => e.requiresApproval && (e.approvalStatus ?? 'pending') === 'pending')) {
      rows.push({
        id: `connector:${event.id}`,
        icon: sourceIcon(event.connectorId),
        sourceGroup: 'Connectors',
        title: event.title,
        detail: event.summary,
        meta: `${connectorName(event.connectorId)} / ${event.actor}`,
        tone: 'critical',
        createdAt: event.createdAt,
        primaryLabel: 'Allow',
        secondaryLabel: 'Deny',
        onPrimary: () => void api.resolveConnectorApproval(event.id, 'approve').then(refresh),
        onSecondary: () => void api.resolveConnectorApproval(event.id, 'deny').then(refresh),
      })
    }
    for (const mission of missionSnapshot?.missions ?? []) {
      for (const approval of mission.pendingApprovals) {
        rows.push({
          id: `mission:${mission.id}:${approval.id}`,
          icon: 'shield',
          sourceGroup: 'Missions',
          title: approval.title,
          detail: approval.summary,
          meta: mission.title,
          tone: 'critical',
          createdAt: approval.requestedAt,
          primaryLabel: 'Allow',
          secondaryLabel: 'Deny',
          onPrimary: () => void api.resolveMissionApproval(mission.id, approval.id, 'allow').then(refreshMissions),
          onSecondary: () => void api.resolveMissionApproval(mission.id, approval.id, 'deny').then(refreshMissions),
        })
      }
    }
    for (const ticket of tickets.filter((t) => t.status === 'review' || t.status === 'needs_clarification')) {
      rows.push({
        id: `ticket-review:${ticket.id}`,
        icon: ticket.status === 'review' ? 'check' : 'warning',
        sourceGroup: 'Tickets',
        title: ticket.title,
        detail: ticket.body || 'Ticket is waiting for a human decision.',
        meta: projectName(projects, ticket.projectId),
        tone: ticket.status === 'needs_clarification' ? 'high' : 'good',
        createdAt: ticket.updatedAt,
        primaryLabel: ticket.status === 'review' ? 'Complete' : 'Mark ready',
        secondaryLabel: ticket.threadId ? 'Open' : undefined,
        onPrimary: () => void updateTicket(ticket.id, { status: ticket.status === 'review' ? 'completed' : 'ready' }),
        onSecondary: ticket.threadId ? () => openThread(ticket.threadId!) : undefined,
      })
    }
    return rows.sort(sortRows).slice(0, 8)
  }, [approvals, durableJobs, threads, snapshot, missionSnapshot, tickets, projects, approve, updateTicket, openThread])

  const approvalGroups = useMemo(() => groupApprovalRows(approvalRows), [approvalRows])

  const savedPolicies = useMemo(() => {
    return [...(settings?.approvalRules ?? [])]
      .filter((rule) => rule.duration === 'always')
      .sort((a, b) => (b.lastUsedAt ?? b.updatedAt) - (a.lastUsedAt ?? a.updatedAt))
      .slice(0, 12)
  }, [settings?.approvalRules])

  const revokePolicy = async (rule: ApprovalRule) => {
    if (revokingRuleId) return
    setRevokingRuleId(rule.id)
    setError('')
    try {
      await revokeApprovalRule(rule.id)
    } catch (err: any) {
      setError(err?.message ?? 'Could not revoke approval policy.')
    } finally {
      setRevokingRuleId(null)
    }
  }

  const runningWorkSummary = useMemo(() => summarizeRunningWork(durableJobs), [durableJobs])
  const longWorkCount = useMemo(() => countLongWorkSignals(durableJobs, missionSnapshot?.missions ?? [], rooms, roomStatus, automations), [durableJobs, missionSnapshot, rooms, roomStatus, automations])
  const powerGuard = useMemo(() => powerGuardSignal(settings?.power, longWorkCount, runningWorkSummary.nextWake?.at ?? null, leaseSummaryTone(runningWorkSummary)), [settings?.power, longWorkCount, runningWorkSummary])
  const actionableDurableJobs = useMemo(() => {
    return durableJobs
      .filter((job) => ACTIONABLE_JOB_STATUSES.has(job.status))
      .sort((a, b) => TONE_RANK[jobTone(a)] - TONE_RANK[jobTone(b)] || b.updatedAt - a.updatedAt)
      .slice(0, 4)
  }, [durableJobs])
  const openDurableJob = (job: Job) => {
    const threadId = job.currentThreadId ?? job.threadId
    if (threadId) openThread(threadId)
    else if (job.roomId) openRoom(job.roomId)
    else if (job.automationId) setView({ name: 'automations' })
  }

  const liveJobs = useMemo(() => {
    const jobs: LiveJob[] = []
    const seen = new Set<string>()
    const remember = (key: string) => { seen.add(key); return key }
    const remoteMetaForProject = (projectId?: string | null) => {
      if (!projectId) return null
      const project = projectById.get(projectId)
      if (project?.executionTarget !== 'orchestrator') return null
      const host = orchestrators.find((item) => item.id === project.orchestratorId)
      return host?.name || 'Orchestrator'
    }
    const remoteAttrs = (projectId?: string | null) => {
      const label = remoteMetaForProject(projectId)
      return label ? { remote: true, remoteLabel: label } : {}
    }
    for (const thread of threads) {
      const st = status[thread.id] ?? thread.status
      if (!RUNNING_THREAD_STATES.has(st)) continue
      jobs.push({
        id: remember(`thread:${thread.id}`),
        icon: thread.kind === 'chat' ? 'message' : 'terminal',
        title: thread.title,
        detail: thread.cwd ? shortPath(thread.cwd, 2) : 'Conversation',
        status: threadStatusLabel(st),
        tone: st === 'awaiting' ? 'high' : 'normal',
        startedAt: thread.updatedAt,
        onOpen: () => openThread(thread.id),
        onCancel: () => void cancel(thread.id),
        ...remoteAttrs(thread.projectId),
      })
    }
    for (const ticket of tickets.filter((t) => t.runState === 'running' || t.runState === 'queued')) {
      jobs.push({
        id: remember(`ticket:${ticket.id}`),
        icon: 'file',
        title: ticket.title,
        detail: `${projectName(projects, ticket.projectId)} / ${ticket.assignee ?? 'agent'}`,
        status: ticket.runState ?? ticket.status,
        tone: ticket.runState === 'queued' ? 'low' : 'normal',
        startedAt: ticket.startedAt ?? ticket.updatedAt,
        onOpen: () => ticket.threadId ? openThread(ticket.threadId) : void workTicket(ticket.id),
        ...remoteAttrs(ticket.projectId),
      })
    }
    for (const room of rooms) {
      const state = roomStatus[room.id] ?? room.state
      if (!RUNNING_ROOM_STATES.has(state)) continue
      jobs.push({
        id: remember(`room:${room.id}`),
        icon: 'layers',
        title: room.name,
        detail: room.problem || projectName(projects, room.projectId),
        status: roomStatusLabel(state),
        tone: state.includes('waiting') ? 'high' : 'normal',
        startedAt: room.updatedAt,
        onOpen: () => openRoom(room.id),
        onCancel: () => void cancelRoom(room.id),
        ...remoteAttrs(room.projectId),
      })
    }
    for (const automation of automations) {
      const run = automation.runs[0]
      if (run?.status !== 'running') continue
      jobs.push({
        id: remember(`automation:${automation.id}`),
        icon: 'clock',
        title: automation.name,
        detail: `${automation.mode} / ${automation.model ?? 'model'}`,
        status: 'running',
        tone: 'normal',
        startedAt: run.startedAt,
        onOpen: () => setView({ name: 'automations' }),
      })
    }
    for (const mission of missionSnapshot?.missions ?? []) {
      if (!missionIsActive(mission.status)) continue
      jobs.push({
        id: remember(`mission:${mission.id}`),
        icon: 'spark',
        title: mission.title,
        detail: mission.objective,
        status: missionStatusLabel(mission.status),
        tone: mission.status === 'waiting_approval' || mission.status === 'stalled' ? 'high' : mission.status === 'paused' ? 'low' : 'normal',
        startedAt: mission.startedAt ?? mission.createdAt,
        onCancel: mission.status === 'cancelled' ? undefined : () => void setMission(mission.id, 'cancelled'),
      })
    }
    for (const job of durableJobs) {
      if (!['queued', 'running', 'waiting_approval', 'paused', 'stalled', 'failed', 'dead_letter'].includes(job.status)) continue
      const key = job.ticketId ? `ticket:${job.ticketId}` : job.roomId ? `room:${job.roomId}` : job.missionId ? `mission:${job.missionId}` : job.automationId ? `automation:${job.automationId}` : job.threadId ? `thread:${job.threadId}` : `job:${job.id}`
      if (seen.has(key)) continue
      remember(key)
      const title = typeof job.metadata?.title === 'string' ? job.metadata.title : typeof job.metadata?.name === 'string' ? job.metadata.name : `${cap(job.kind)} job`
      const canResume = job.status === 'paused' || job.status === 'stalled'
      const canRetry = job.status === 'stalled' || job.status === 'failed' || job.status === 'dead_letter'
      const canDeadLetter = job.status === 'paused' || job.status === 'stalled' || job.status === 'waiting_approval' || job.status === 'failed'
      const unverifiable = job.executionVerification === 'unverifiable'
      jobs.push({
        id: `durable:${job.id}`,
        icon: jobIcon(job),
        title,
        detail: unverifiable ? job.executionVerificationReason || 'Remote execution cannot be verified.' : job.blockedReason || job.resumePlan || job.owner,
        footnote: unverifiable
          ? job.executionVerifiedAt ? `last verified ${relativeTime(job.executionVerifiedAt)}` : 'never verified'
          : durableJobFootnote(job),
        status: unverifiable ? `${job.status.replace(/_/g, ' ')} · unverifiable` : job.status.replace(/_/g, ' '),
        tone: job.status === 'failed' || job.status === 'dead_letter' ? 'critical' : unverifiable || job.status === 'stalled' || job.status === 'waiting_approval' ? 'high' : job.status === 'paused' ? 'low' : 'normal',
        startedAt: job.startedAt ?? job.updatedAt,
        onOpen: job.threadId ? () => openThread(job.threadId!) : job.roomId ? () => openRoom(job.roomId!) : job.automationId ? () => setView({ name: 'automations' }) : undefined,
        onResume: canResume ? () => void resumeJob(job.id) : undefined,
        onRetry: canRetry ? () => void retryJob(job.id, 'Manual retry from Command Centre.') : undefined,
        onDeadLetter: canDeadLetter ? () => void deadLetterJob(job.id, 'Archived from Command Centre.') : undefined,
        ...remoteAttrs(job.projectId),
      })
    }
    const remoteVisible = new Set(['queued', 'running', 'waiting_approval', 'paused', 'stalled', 'failed', 'dead_letter'])
    for (const host of orchestrators) {
      const result = remoteSnapshots[host.id]
      if (!result?.ok) continue
      const remoteProjects = new Map((result.projects ?? []).map((project) => [project.id, project]))
      for (const job of result.jobs ?? []) {
        if (!remoteVisible.has(job.status)) continue
        const project = job.projectId ? remoteProjects.get(job.projectId) : null
        const localKey = job.ticketId ? `ticket:${job.ticketId}` : job.roomId ? `room:${job.roomId}` : job.missionId ? `mission:${job.missionId}` : job.automationId ? `automation:${job.automationId}` : job.threadId ? `thread:${job.threadId}` : `job:${job.id}`
        const key = `remote:${host.id}:${localKey}`
        if (seen.has(localKey) || seen.has(key)) continue
        const title = typeof job.metadata?.title === 'string' ? job.metadata.title : typeof job.metadata?.name === 'string' ? job.metadata.name : `${cap(job.kind)} job`
        const unverifiable = job.executionVerification === 'unverifiable'
        jobs.push({
          id: remember(key),
          icon: jobIcon(job),
          title,
          detail: project ? `${project.name} / ${host.name}` : host.name,
          footnote: unverifiable ? job.executionVerificationReason || 'Remote execution cannot be verified.' : job.blockedReason || durableJobFootnote(job),
          status: unverifiable ? `${job.status.replace(/_/g, ' ')} · unverifiable` : job.status.replace(/_/g, ' '),
          tone: job.status === 'failed' || job.status === 'dead_letter' ? 'critical' : unverifiable || job.status === 'stalled' || job.status === 'waiting_approval' ? 'high' : job.status === 'paused' ? 'low' : 'normal',
          startedAt: job.startedAt ?? job.updatedAt,
          remote: true,
          remoteLabel: host.name,
          onOpen: () => void refreshRemoteHost(host.id),
        })
      }
    }
    return jobs.sort((a, b) => a.startedAt - b.startedAt).slice(0, 10)
  }, [threads, status, tickets, projectById, orchestrators, remoteSnapshots, refreshRemoteHost, rooms, roomStatus, automations, missionSnapshot, durableJobs, openThread, cancel, workTicket, openRoom, cancelRoom, setView, resumeJob, retryJob, deadLetterJob])
  const workLaneSummary = useMemo(() => summarizeWorkLanes(liveJobs, durableJobs), [liveJobs, durableJobs])

  const briefAutomations = useMemo(() => {
    const ranked = [...automations].sort((a, b) => (a.nextRunAt ?? Infinity) - (b.nextRunAt ?? Infinity))
    return ranked.filter((a) => a.scheduleKind === 'daily' || /brief|digest|standup/i.test(a.name)).slice(0, 3)
  }, [automations])
  const primaryBrief = briefAutomations[0] ?? null
  const briefLines = briefOutputLines(primaryBrief).slice(0, 5)
  const connectorBriefLines = snapshot?.briefing.lines.slice(0, 5) ?? []
  const dailyBriefLines = briefLines.length ? [...connectorBriefLines.slice(0, 2), ...briefLines].slice(0, 5) : connectorBriefLines
  const missionRows = missionSnapshot?.missions ?? []
  const awayBriefRows = useMemo(() => {
    const now = dashboardNow
    const since = now - AWAY_BRIEF_WINDOW_MS
    const rows: AwayBriefRow[] = []
    const add = (row: AwayBriefRow) => rows.push(row)

    for (const row of approvalRows) {
      add({
        id: `approval:${row.id}`,
        icon: row.icon,
        title: `Approval waiting: ${row.title}`,
        detail: row.detail,
        meta: `${row.sourceGroup} / ${relativeTime(row.createdAt)}`,
        tone: row.tone,
        createdAt: row.createdAt,
        actionLabel: 'Review',
        onOpen: () => document.getElementById('command-centre-approvals')?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      })
    }

    for (const job of durableJobs) {
      if (job.kind === 'pr_loop') continue
      const at = job.finishedAt ?? job.updatedAt
      const notable = ['failed', 'stalled', 'dead_letter', 'waiting_approval', 'completed'].includes(job.status)
      if (!notable || (at < since && job.status === 'completed')) continue
      add({
        id: `job:${job.id}`,
        icon: jobIcon(job),
        title: `${cap(job.status)}: ${jobTitle(job)}`,
        detail: job.blockedReason || job.resumePlan || durableJobFootnote(job),
        meta: `${cap(job.kind)} / ${relativeTime(at)}`,
        tone: jobTone(job),
        createdAt: at,
        actionLabel: 'Open',
        onOpen: () => openDurableJob(job),
      })
    }

    const approvalIds = new Set(approvalRows.map((row) => row.id))
    for (const item of buildAwayBriefItems({
      now,
      since,
      limit: 40,
      connectorEvents: snapshot?.events ?? [],
      rooms,
      roomStatus,
      missions: missionRows,
      automations,
      jobs: durableJobs,
    })) {
      if (item.kind === 'connector' && approvalIds.has(`connector:${item.subjectId}`)) continue
      add(awayItemToRow(item, {
        durableJobs,
        openDurableJob,
        openRoom,
        setView,
      }))
    }

    for (const ticket of tickets) {
      if (ticket.updatedAt < since && ticket.status !== 'review' && ticket.status !== 'needs_clarification') continue
      if (ticket.status === 'todo' && !ticket.runState) continue
      add({
        id: `ticket:${ticket.id}`,
        icon: ticket.status === 'needs_clarification' ? 'warning' : 'file',
        title: `${cap(ticket.status)}: ${ticket.title}`,
        detail: ticket.body || ticket.tags.join(', ') || 'Ticket changed state.',
        meta: `${projectName(projects, ticket.projectId)} / ${relativeTime(ticket.updatedAt)}`,
        tone: ticket.status === 'needs_clarification' ? 'high' : ticket.status === 'review' ? 'good' : ticket.priority === 'urgent' ? 'critical' : 'normal',
        createdAt: ticket.updatedAt,
        actionLabel: ticket.threadId ? 'Open' : 'Work',
        onOpen: () => {
          if (ticket.threadId) openThread(ticket.threadId)
          else void workTicket(ticket.id)
        },
      })
    }

    for (const thread of threads) {
      const threadState = status[thread.id] ?? thread.status
      if (thread.updatedAt < since && threadState !== 'error') continue
      if (threadState === 'idle') continue
      add({
        id: `thread:${thread.id}`,
        icon: thread.kind === 'chat' ? 'message' : 'terminal',
        title: `${threadStatusLabel(threadState)}: ${thread.title}`,
        detail: thread.cwd ? shortPath(thread.cwd, 2) : 'Conversation changed state.',
        meta: `${thread.kind === 'chat' ? 'Chat' : 'Task'} / ${relativeTime(thread.updatedAt)}`,
        tone: threadState === 'error' ? 'critical' : RUNNING_THREAD_STATES.has(threadState) ? 'normal' : 'low',
        createdAt: thread.updatedAt,
        actionLabel: 'Open',
        onOpen: () => openThread(thread.id),
      })
    }

    return rows.sort(sortRows).slice(0, 8)
  }, [approvalRows, durableJobs, snapshot, automations, tickets, projects, rooms, roomStatus, missionRows, threads, status, openThread, openDurableJob, openRoom, workTicket, setView, dashboardNow])

  const connectorRows = snapshot?.connectors ?? []
  const connectedCount = connectorRows.filter((c) => c.status === 'connected' || c.status === 'mock').length
  const pendingApprovals = approvalRows.length
  const unreadSignals = connectorRows.reduce((sum, c) => sum + c.unread, 0)
  const gapCount = snapshot?.gaps.length ?? 0
  const nextBrief = primaryBrief?.nextRunAt ?? null
  const activeMissions = missionRows.filter((mission) => missionIsActive(mission.status)).length
  const latestThread = useMemo(() => [...threads].sort((a, b) => b.updatedAt - a.updatedAt)[0] ?? null, [threads])
  const connectorHealthRows = useMemo(() => connectorRows.map((connector) => ({
    connector,
    health: connectorHealthSignal(connector, settings, dashboardNow),
  })), [connectorRows, settings, dashboardNow])
  const connectorActionCount = connectorHealthRows.filter((row) => row.health.action !== 'none').length
  const providerVisibilityRows = useMemo(() => buildProviderVisibilityRows({
    settings,
    threads,
    threadStatus: status,
    rooms,
    roomStatus,
    automations,
    missions: missionRows,
    contextSnapshot,
    modelRouterSnapshot,
    usageSnapshot,
  }), [settings, threads, status, rooms, roomStatus, automations, missionRows, contextSnapshot, modelRouterSnapshot, usageSnapshot])
  const activeProviderCount = providerVisibilityRows.reduce((sum, row) => sum + row.activeCount, 0)
  const readinessRemoteProbes = useMemo<Record<string, ProbeState<{ ok: boolean; error?: string }>>>(() => {
    if (!settings) return {}
    const entries = (settings.remoteOrchestrators ?? [])
      .filter((host) => host.enabled && host.serverUrl?.trim() && host.authToken?.trim())
      .map((host) => {
        if (remoteBusy === host.id) return [host.id, { status: 'checking' as const }] as const
        const result = remoteSnapshots[host.id]
        if (!result) return [host.id, { status: 'checking' as const }] as const
        if (!result.ok) {
          const message = result.error || host.lastError || 'Host check failed.'
          return [host.id, { status: 'error' as const, value: { ok: false, error: message }, message }] as const
        }
        return [host.id, { status: 'ok' as const, value: { ok: true } }] as const
      })
    return Object.fromEntries(entries)
  }, [settings?.remoteOrchestrators, remoteSnapshots, remoteBusy])
  const readinessStudioRemote = useMemo(() => {
    if (!settings) return undefined
    return readinessStudioStatus ?? {
      enabled: settings.remoteControl.enabled,
      listening: false,
      clients: 0,
      url: settings.remoteControl.enabled ? `http://${settings.remoteControl.allowLan ? settings.remoteControl.bindHost : '127.0.0.1'}:${settings.remoteControl.port}` : null,
      lastError: settings.remoteControl.lastError,
    }
  }, [settings?.remoteControl, readinessStudioStatus])
  // Node & inference dashboard (VST-014): one health view over every place work runs.
  const nodeDashboard = useMemo(() => buildNodeDashboard({
    now: dashboardNow,
    local: {
      activeJobs: durableJobs.filter((job) => job.status === 'running' || job.status === 'queued').length,
      pendingApprovals: Object.values(approvals).reduce((sum, list) => sum + (list?.length ?? 0), 0),
      terminals: summarizeTerminalActivity(Object.values(terminals).flat().filter((session) => session.hostLocation !== 'orchestrator')),
    },
    remoteHosts: (settings?.remoteOrchestrators ?? [])
      .filter((host) => host.enabled && host.serverUrl?.trim())
      .map((host) => {
        const result = remoteSnapshots[host.id]
        return {
          hostId: host.id,
          name: host.name || host.serverUrl,
          ok: result?.ok === true,
          error: result?.ok === false ? (result.error || host.lastError || 'Host check failed.') : null,
          generatedAt: result?.ok ? result.generatedAt ?? null : null,
          work: result?.ok ? result.work ?? null : null,
        }
      }),
    providers: settings?.providers ?? [],
    providerStatus: providerStatusSnapshot,
    cliTools: readinessLocalClis.status === 'ok' ? (readinessLocalClis.value ?? []).filter((cli): cli is CliToolStatus => 'installed' in cli) : [],
    tokenUsage: usageSnapshot,
  }), [dashboardNow, durableJobs, approvals, terminals, settings?.remoteOrchestrators, settings?.providers, remoteSnapshots, providerStatusSnapshot, readinessLocalClis, usageSnapshot])

  const readinessChecks = useMemo(() => settings ? buildReadinessDoctor({
    settings,
    providerReadiness: readinessProviderProbes,
    localClis: readinessLocalClis,
    remoteProbes: readinessRemoteProbes,
    studioRemote: readinessStudioRemote,
    connectors: snapshot ? { status: 'ok', value: snapshot.connectors } : { status: 'checking' },
    connectorEvents: snapshot?.events ?? [],
  }) : [], [settings, readinessProviderProbes, readinessLocalClis, readinessRemoteProbes, readinessStudioRemote, snapshot, dashboardNow])
  const readinessSummary = useMemo(() => summarizeDoctor(readinessChecks), [readinessChecks])
  const readinessTone = useMemo(() => doctorStatusTone(worstDoctorStatus(readinessChecks)), [readinessChecks])
  // Declutter: glanceable status + disclosure defaults (Task #62).
  const attentionMaxTone: Tone = useMemo(
    () => worstTone(...attentionRows.map((row) => row.tone)),
    [attentionRows],
  )
  const liveJobsTone: Tone =
    runningWorkSummary.statusCounts.failed || runningWorkSummary.statusCounts.dead_letter
      ? 'critical'
      : runningWorkSummary.statusCounts.stalled
        ? 'high'
        : liveJobs.length
          ? 'normal'
          : 'good'
  const connectorTone: Tone = connectorHealthRows.some((row) => row.health.tone === 'critical')
    ? 'critical'
    : connectorActionCount || gapCount
      ? 'high'
      : 'good'
  const statusSummary = useMemo(
    () => computeCommandStatusSummary({
      attentionCount: attentionRows.length,
      attentionMaxTone,
      approvalsCount: pendingApprovals,
      liveJobsTone,
      connectorTone,
      readinessTone,
      jobsStatus: {
        queued: runningWorkSummary.statusCounts.queued,
        stalled: runningWorkSummary.statusCounts.stalled,
        failed: runningWorkSummary.statusCounts.failed,
        deadLetter: runningWorkSummary.statusCounts.dead_letter,
      },
      missionCount: activeMissions,
    }),
    [attentionRows.length, attentionMaxTone, pendingApprovals, liveJobsTone, connectorTone, readinessTone, runningWorkSummary, activeMissions],
  )
  const awayMaxTone: Tone = useMemo(() => worstTone(...awayBriefRows.map((row) => row.tone)), [awayBriefRows])
  const statusExpanded = statusExpandedOverride ?? summaryDefaultExpanded(statusSummary.maxTone)
  const awayOpen = awayOpenOverride ?? disclosureDefaultOpen(awayBriefRows.length, DISCLOSURE_AUTO_EXPAND_THRESHOLD, awayMaxTone)
  const orchestratorsOpen = orchestratorsOpenOverride ?? orchestrators.length > 0

  return (
    <CommandCentreLayout
      snapshot={snapshot}
      attentionRows={attentionRows}
      pendingApprovals={pendingApprovals}
      liveJobs={liveJobs}
      activeMissions={activeMissions}
      connectedCount={connectedCount}
      connectorRows={connectorRows}
      refresh={refresh}
      loading={loading}
      latestThread={latestThread}
      setView={setView}
      primaryBrief={primaryBrief}
      runAutomation={runAutomation}
      error={error}
      unreadSignals={unreadSignals}
      runningWorkSummary={runningWorkSummary}
      connectorActionCount={connectorActionCount}
      gapCount={gapCount}
      connectorHealthRows={connectorHealthRows}
      nextBrief={nextBrief}
      missionRows={missionRows}
      selected={selected}
      setSelectedId={setSelectedId}
      awayBriefRows={awayBriefRows}
      dailyBriefLines={dailyBriefLines}
      briefAutomations={briefAutomations}
      orchestrators={orchestrators}
      remoteSnapshots={remoteSnapshots}
      remoteBusy={remoteBusy}
      refreshRemoteHost={refreshRemoteHost}
      approvalRows={approvalRows}
      approvalGroups={approvalGroups}
      savedPolicies={savedPolicies}
      revokePolicy={revokePolicy}
      revokingRuleId={revokingRuleId}
      workLaneSummary={workLaneSummary}
      actionableDurableJobs={actionableDurableJobs}
      retryJob={retryJob}
      resumeJob={resumeJob}
      deadLetterJob={deadLetterJob}
      openDurableJob={openDurableJob}
      powerGuard={powerGuard}
      settings={settings}
      longWorkCount={longWorkCount}
      prTitle={prTitle}
      setPrTitle={setPrTitle}
      prObjective={prObjective}
      setPrObjective={setPrObjective}
      prProjectId={prProjectId}
      setPrProjectId={setPrProjectId}
      projects={projects}
      prApprovals={prApprovals}
      setPrApprovals={setPrApprovals}
      prAutoStart={prAutoStart}
      setPrAutoStart={setPrAutoStart}
      prPublishRemote={prPublishRemote}
      setPrPublishRemote={setPrPublishRemote}
      createLocalPrLoop={createLocalPrLoop}
      prBusy={prBusy}
      missionTitle={missionTitle}
      setMissionTitle={setMissionTitle}
      missionObjective={missionObjective}
      setMissionObjective={setMissionObjective}
      missionStopCondition={missionStopCondition}
      setMissionStopCondition={setMissionStopCondition}
      missionRepeat={missionRepeat}
      setMissionRepeat={setMissionRepeat}
      missionContext={missionContext}
      setMissionContext={setMissionContext}
      missionRoleProfile={missionRoleProfile}
      setMissionRoleProfile={setMissionRoleProfile}
      missionFanOut={missionFanOut}
      setMissionFanOut={setMissionFanOut}
      missionMaxRuns={missionMaxRuns}
      setMissionMaxRuns={setMissionMaxRuns}
      missionCostCap={missionCostCap}
      setMissionCostCap={setMissionCostCap}
      missionExpiresDays={missionExpiresDays}
      setMissionExpiresDays={setMissionExpiresDays}
      missionBudget={missionBudget}
      setMissionBudget={setMissionBudget}
      missionGateDestructive={missionGateDestructive}
      setMissionGateDestructive={setMissionGateDestructive}
      createMissionFromDraft={createMissionFromDraft}
      missionInspection={missionInspection}
      missionInspectingId={missionInspectingId}
      inspectMission={inspectMission}
      setMission={setMission}
      beatMission={beatMission}
      providerVisibilityRows={providerVisibilityRows}
      activeProviderCount={activeProviderCount}
      modelRouterSnapshot={modelRouterSnapshot}
      nodeDashboard={nodeDashboard}
      readinessChecks={readinessChecks}
      readinessSummary={readinessSummary}
      readinessTone={readinessTone}
      refreshReadiness={refreshReadiness}
      openReadinessSettings={openReadinessSettings}
      syncEmailAndRefresh={syncEmailAndRefresh}
      statusSummary={statusSummary}
      statusExpanded={statusExpanded}
      toggleStatusExpanded={() => setStatusExpandedOverride(!statusExpanded)}
      awayOpen={awayOpen}
      toggleAwayOpen={() => setAwayOpenOverride(!awayOpen)}
      orchestratorsOpen={orchestratorsOpen}
      toggleOrchestratorsOpen={() => setOrchestratorsOpenOverride(!orchestratorsOpen)}
      missionFormsOpen={missionFormsOpen}
      toggleMissionForms={() => setMissionFormsOpen((value) => !value)}
      showAllMissions={showAllMissions}
      toggleShowAllMissions={() => setShowAllMissions((value) => !value)}
    />
  )
}
