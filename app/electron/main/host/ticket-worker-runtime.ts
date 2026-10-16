import { randomUUID } from 'node:crypto'
import path from 'node:path'
import { runAgent } from '../../agent/loop'
import { assertProviderEndpoint, auditPreview } from '../../agent/security'
import { normalizeTicketCapabilities, settleTicketExecution, startTicketExecution, ticketFileRefs, ticketStateAfterRun, ticketWorkPrompt } from '../../agent/tickets'
import { restrictTicketToolContext } from '../../agent/ticket-tool-policy'
import type { JobPatch } from '../../agent/jobs'
import type { AgentCue, AgentEvent, Job, Project, Provider, Room, Thread, Ticket, TicketWorkOptions, TimelineItem } from '../../agent/types'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { ensureRoomWorktree } from './room-worktree'
import { requireTicketDependenciesReady } from '../../../shared/ticket-dependencies'
import { inspectGitRepository } from './workspace'
import { summarizeTicketWorkerOutcome } from './ticket-helpers'
import { modelRouteAccountLabel, modelRouteAutoFailoverEnabled, modelRouteFailoverText, modelRouteResetPolicy, modelRouter, resolveProjectProvider, runtimeProviderWithFreshModels, ticketWorkOptionsWithProjectRoute } from './provider-runtime'
import { resolveWorkspaceCwd } from './workspace'
import { lexicalLocalWorkspaceKey, type WorkspaceMutationGuard } from './workspace-mutation-guard'

export interface TicketWorkerSource {
  roomId?: string | null
  seat?: string | null
  onSettled?: (summary: { codeChanged: boolean; testsPassed: boolean; failed: boolean; jobId?: string | null }) => void
}

export interface TicketWorkerInput {
  ticketId: string
  opts?: TicketWorkOptions
  source?: TicketWorkerSource
}

interface JobHandle {
  get(): Job | null
  set(job: Job | null): void
}

export interface TicketWorkerRuntime {
  startTicketWork(input: TicketWorkerInput): Promise<Thread>
  isTicketStarting(ticketId: string): boolean
}

export function assertActiveTicketWorkOwnership(
  ticket: Ticket,
  project: Project | null,
  room: Room | null,
  thread: Thread | null = null,
): void {
  if (ticket.archivedAt) throw new Error('Restore this archived ticket before starting work.')
  if (!project || project.id !== ticket.projectId || project.archivedAt) throw new Error('Restore or choose an active project before starting ticket work.')
  if (ticket.roomId && (!room || room.id !== ticket.roomId || room.projectId !== ticket.projectId || room.archivedAt)) {
    throw new Error('Restore or choose an active room in the same project before starting ticket work.')
  }
  if (ticket.threadId && (!thread || thread.id !== ticket.threadId || thread.projectId !== ticket.projectId || thread.archivedAt)) {
    throw new Error('Restore or choose an active ticket thread in the same project before starting work.')
  }
}

export async function awaitTicketStartupStep<T>(pending: Promise<T>, revalidate: () => void): Promise<T> {
  const value = await pending
  revalidate()
  return value
}

interface TicketWorkerRuntimeDeps {
  persistThreads(): void
  persistTickets(): void
  emitTicket(ticket: Ticket): void
  appendRoomSupervisorEvent(room: NonNullable<typeof store.rooms[number]>, text: string, extra?: any): any
  claimJob(ref: Parameters<typeof import('../../agent/jobs').enqueueJob>[1]): Job
  patchJob(job: Job | null, patch: JobPatch): Job | null
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal }): any
  drainThreadCues(threadId: string): AgentCue[]
  waitForThreadCue(threadId: string, timeoutMs: number, signal: AbortSignal): Promise<void>
  approvalResolver(controller: AbortController, jobHandle?: JobHandle): (callId: string) => Promise<'allow' | 'allow_always' | 'deny'>
  timelineErrorSince(thread: Thread, timelineStart: number): TimelineItem | null
  timelineError(errorItem: TimelineItem): Error
  isHighRiskTool(name: string): boolean
  emitToRenderer(event: AgentEvent): void
  addThreadNotice(thread: Thread, runId: string, text: string): void
  auditModelRoute(thread: Thread, audit: any, signal?: any): Promise<void>
  markRouteErrorRecovered(thread: Thread, err: unknown, runId: string, text: string): void
  injectThreadRouteHandoff(thread: Thread, reason: string): string
  aborters: Map<string, AbortController>
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'assertThreadStartAllowed' | 'reserveStart'>
  applyEventToThread(thread: Thread, event: AgentEvent): void
  activeJob(kind: Job['kind'], owner: string): Job | null
  finishJob(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  markUndrainedCues(thread: Thread, meta?: string): void
  effectiveContextLimit(provider: Provider, model: string | null): number
}

export function createTicketWorkerRuntime(deps: TicketWorkerRuntimeDeps): TicketWorkerRuntime {
  const {
    persistThreads,
    persistTickets,
    emitTicket,
    appendRoomSupervisorEvent,
    claimJob,
    patchJob,
    fullToolContext,
    drainThreadCues,
    waitForThreadCue,
    approvalResolver,
    timelineErrorSince,
    timelineError,
    isHighRiskTool,
    emitToRenderer,
    addThreadNotice,
    auditModelRoute,
    markRouteErrorRecovered,
    injectThreadRouteHandoff,
    aborters,
    workspaceMutationGuard,
    applyEventToThread,
    activeJob,
    finishJob,
    markUndrainedCues,
    effectiveContextLimit,
  } = deps
  const ticketStarts = new Set<string>()

async function startTicketWork(input: TicketWorkerInput): Promise<Thread> {
  requireSecurityRole('operator', 'work tickets')
  const requestedOpts = input.opts
  const phase = requestedOpts?.phase === 'review' ? 'review' : 'build'
  const ticket = store.tickets.find((t) => t.id === input.ticketId)
  if (!ticket) throw new Error('Ticket not found')
  if (ticketStarts.has(ticket.id)) throw new Error('Ticket work is already starting.')
  const project = ticket.projectId ? store.projects.find((p) => p.id === ticket.projectId) ?? null : null
  const room = ticket.roomId ? store.rooms.find((r) => r.id === ticket.roomId) ?? null : null
  const existingThread = ticket.threadId ? store.threads.find((thread) => thread.id === ticket.threadId) ?? null : null
  assertActiveTicketWorkOwnership(ticket, project, room, existingThread)
  requireTicketDependenciesReady(ticket, store.tickets)
  if (!project) throw new Error('Restore or choose an active project before starting ticket work.')
  if (existingThread && aborters.has(existingThread.id)) throw new Error('Ticket work is already running.')
  const configuredWorkspace = room?.workspace ?? project.workspace ?? null
  if (configuredWorkspace?.kind === 'ssh') {
    throw new Error('Local ticket workers do not support SSH workspaces; run this ticket on its remote Orchestrator.')
  }
  const initialPaths = [configuredWorkspace?.path, room?.workspaceCwd, room?.rootCwd, room?.scratchpadCwd, existingThread?.cwd, project.cwd]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
  const initialKeys = initialPaths.map((value) => lexicalLocalWorkspaceKey(value))
  if (existingThread) workspaceMutationGuard.assertThreadStartAllowed(existingThread.id, initialKeys.find(Boolean))
  const runLease = workspaceMutationGuard.reserveStart({ threadId: existingThread?.id, workspaceKeys: initialKeys })
  ticketStarts.add(ticket.id)
  let reservedThreadId: string | null = null
  let reservedController: AbortController | null = null
  let launched = false
  try {
  const opts = ticketWorkOptionsWithProjectRoute(project, requestedOpts, phase === 'review' ? 'ticket_review' : 'ticket_build')
  const roomWorkspace = project && room
    ? await awaitTicketStartupStep(ensureRoomWorktree(project, room), () => assertActiveTicketWorkOwnership(ticket, project, room, existingThread))
    : null
  if (room && project?.cwd && roomWorkspace && path.resolve(roomWorkspace) === path.resolve(project.cwd)) {
    throw new Error('Room ticket work must run in an isolated room worktree, not the base project folder.')
  }
  const resolvedRoute = resolveProjectProvider(project, opts?.providerAccountId, opts?.model, opts?.providerId)
  const baseProvider = resolvedRoute.baseProvider
  const routeBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
  const requestedProviderAccountId = resolvedRoute.providerAccountId
  const initialRoute = modelRouter.select({
    role: phase === 'review' ? 'ticket_review' : 'ticket_build',
    provider: routeBaseProvider,
    model: resolvedRoute.model,
    requestedAccountId: requestedProviderAccountId,
    allowFailover: modelRouteAutoFailoverEnabled(routeBaseProvider, requestedProviderAccountId),
    localOnly: project?.localOnly === true,
    contextLimit: effectiveContextLimit(routeBaseProvider, resolvedRoute.model),
    reasoningEffort: store.settings.reasoningEffort,
    budget: phase,
    source: 'ticket_worker_initial_route',
  })
  if (!initialRoute.selected) throw new Error(initialRoute.audit.reason)
  let providerAccountId = initialRoute.selectedAccountId
  let provider = providerAccountId === requestedProviderAccountId
    ? resolvedRoute.provider
    : await runtimeProviderWithFreshModels(routeBaseProvider, providerAccountId)
  let model = resolvedRoute.model
  assertProviderEndpoint(provider)
  const cwd = await resolveWorkspaceCwd(roomWorkspace || project?.cwd || store.settings.defaultCwd, 'ticket workspace')

  assertActiveTicketWorkOwnership(ticket, project, room, existingThread)

  const executionBranch = await awaitTicketStartupStep(
    inspectGitRepository(cwd, 'ticket workspace').then((repo) => repo.branch).catch(() => null),
    () => assertActiveTicketWorkOwnership(ticket, project, room, existingThread),
  )

  let thread = ticket.threadId ? store.threads.find((t) => t.id === ticket.threadId) : null
  assertActiveTicketWorkOwnership(ticket, project, room, thread)
  if (thread && aborters.has(thread.id)) throw new Error('Ticket work is already running.')
  const isNewThread = !thread
  if (!thread) {
    thread = {
      id: randomUUID(), title: ticket.title, kind: 'task', projectId: ticket.projectId ?? null,
      cwd,
      providerId: baseProvider.id, providerAccountId, model,
      createdAt: Date.now(), updatedAt: Date.now(), status: 'idle', apiMessages: [], timeline: [],
    }
  }
  runLease.addWorkspaceKeys?.([lexicalLocalWorkspaceKey(cwd)])
  runLease.addThreadId?.(thread.id)
  if (aborters.has(thread.id)) throw new Error('Ticket work is already running.')
  const controller = new AbortController()
  reservedThreadId = thread.id
  reservedController = controller
  aborters.set(thread.id, controller)
  if (isNewThread) {
    store.threads.unshift(thread); persistThreads()
  }
  ticket.threadId = thread.id
  ticket.status = 'in_progress'
  ticket.runState = 'running'
  ticket.startedAt = Date.now()
  ticket.assignee = input.source?.seat || ticket.assignee || model
  // Execution tracking (VST-018): branch / engine / run, stamped at start, settled below.
  ticket.execution = startTicketExecution({
    target: input.source?.roomId ? 'room' : (ticket.executionMode === 'frontier' || ticket.executionMode === 'cli') ? ticket.executionMode : 'solo',
    providerId: baseProvider.id,
    model,
    branch: executionBranch,
    threadId: thread.id,
    startedAt: ticket.startedAt,
  })
  ticket.updatedAt = Date.now()
  persistTickets(); emitTicket(ticket)
  if (room && input.source?.roomId) {
    appendRoomSupervisorEvent(room, `Started ${phase} worker for "${ticket.title}" in ${roomWorkspace ?? cwd}.`, {
      tool: 'start_ticket_work',
      args: { ticketId: ticket.id, phase, seat: input.source.seat ?? null, threadId: thread.id },
    })
  }
  let job: Job | null = claimJob({
    kind: 'ticket',
    owner: ticket.id,
    projectId: ticket.projectId,
    roomId: ticket.roomId ?? null,
    ticketId: ticket.id,
    threadId: thread.id,
    currentThreadId: thread.id,
    metadata: {
      title: ticket.title,
      phase,
      model,
      risk: ticket.risk,
      allowedCapabilities: ticket.allowedCapabilities,
      requiredEvidence: ticket.requiredEvidence,
    },
  })
  const jobHandle: JobHandle = {
    get: () => job,
    set: (next) => { job = next },
  }

  const allowedCapabilities = normalizeTicketCapabilities(ticket.allowedCapabilities, ['read'])
  const prompt = ticketWorkPrompt(ticket, phase)
  const userItem = { id: randomUUID(), kind: 'user' as const, text: prompt, createdAt: Date.now() }
  thread.timeline.push(userItem)
  thread.apiMessages.push({ role: 'user', content: prompt })
  thread.status = 'running'
  emitToRenderer({ runId: '', threadId: thread.id, type: 'item_add', item: userItem })

  const t = thread
  const emit = (e: AgentEvent) => { applyEventToThread(t, e); emitToRenderer(e) }
  t.cwd = cwd
  t.providerId = baseProvider.id
  t.providerAccountId = providerAccountId
  t.model = model
  if (initialRoute.audit.action !== 'failover') t.modelRoute = null
  emitToRenderer({ runId: '', threadId: t.id, type: 'thread_update', patch: { providerId: t.providerId, providerAccountId, model, modelRoute: t.modelRoute } })
  if (initialRoute.audit.action === 'failover') {
    const resetAt = initialRoute.candidates.find((candidate) => candidate.accountId === initialRoute.requestedAccountId)?.exhaustedUntil ?? null
    t.modelRoute = {
      preferredProviderAccountId: initialRoute.requestedAccountId,
      failoverProviderAccountId: initialRoute.selectedAccountId,
      resetPolicy: modelRouteResetPolicy(routeBaseProvider, initialRoute.requestedAccountId),
      failedOverAt: Date.now(),
      resetAt,
      reason: initialRoute.audit.reason,
    }
    job = patchJob(job, {
      metadata: {
        providerId: baseProvider.id,
        providerAccountId,
        model,
        modelRoute: { action: 'initial_failover', ...t.modelRoute },
      },
    })
    persistThreads()
    emitToRenderer({ runId: '', threadId: t.id, type: 'thread_update', patch: { modelRoute: t.modelRoute } })
    addThreadNotice(t, '', modelRouteFailoverText(routeBaseProvider, initialRoute.audit))
    await auditModelRoute(t, initialRoute.audit)
  }
  let activeRoute = { provider, providerAccountId, model }
  const runTicketWithRoute = async (route: { provider: Provider; providerAccountId: string | null; model: string }) => {
    const timelineStart = t.timeline.length
    const toolContext = restrictTicketToolContext(
      fullToolContext({ thread: t, provider: route.provider, model: route.model, signal: controller.signal }),
      allowedCapabilities,
    )
    await runAgent({
      thread: t,
      provider: route.provider,
      model: route.model,
      approvalMode: store.settings.approvalMode === 'auto' ? 'suggest' : store.settings.approvalMode,
      temperature: store.settings.temperature,
      reasoningEffort: store.settings.reasoningEffort,
      autoCompact: store.settings.autoCompact,
      contextLimit: effectiveContextLimit(route.provider, route.model),
      toolContext,
      drainCues: () => drainThreadCues(t.id),
      waitForCue: (timeoutMs, signal) => waitForThreadCue(t.id, timeoutMs, signal),
      maxSteps: 40,
      maxWallMs: 30 * 60_000,
      emit,
      signal: controller.signal,
      requestApproval: approvalResolver(controller, jobHandle),
      auditToolCall: async ({ tool, args, resultOk, output }) => {
        if (isHighRiskTool(tool)) await auditEvent('ticket.tool', { ticketId: ticket.id, threadId: t.id, tool, cwd: auditPreview(t.cwd), ok: resultOk, args: auditPreview(JSON.stringify(args)), output: auditPreview(output) })
      },
      persist: persistThreads,
    })
    const errorItem = timelineErrorSince(t, timelineStart)
    if (errorItem && !controller.signal.aborted) throw timelineError(errorItem)
  }
  runTicketWithRoute(activeRoute)
    .catch(async (err) => {
      const latestBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
      const routed = modelRouter.routeAfterError({
        role: phase === 'review' ? 'ticket_review' : 'ticket_build',
        provider: latestBaseProvider,
        accountId: activeRoute.providerAccountId,
        model: activeRoute.model,
        error: err,
        allowFailover: modelRouteAutoFailoverEnabled(latestBaseProvider, activeRoute.providerAccountId) || t.modelRoute?.resetPolicy === 'rotate_on_limit',
        localOnly: project?.localOnly === true,
        contextLimit: effectiveContextLimit(latestBaseProvider, activeRoute.model),
        reasoningEffort: store.settings.reasoningEffort,
        budget: phase,
        source: 'ticket_worker_recovery',
      })
      if (routed.signal) {
        for (const audit of routed.audit) await auditModelRoute(t, audit, routed.signal)
      }
      if (routed.signal && routed.route.selected && routed.route.selectedAccountId !== activeRoute.providerAccountId && !controller.signal.aborted) {
        markRouteErrorRecovered(t, err, '', `Recovered provider limit on ${modelRouteAccountLabel(latestBaseProvider, activeRoute.providerAccountId)} for ${activeRoute.model}; retrying ticket worker on ${modelRouteAccountLabel(latestBaseProvider, routed.route.selectedAccountId)}.`)
        const exhaustedAccountId = activeRoute.providerAccountId
        providerAccountId = routed.route.selectedAccountId
        provider = await runtimeProviderWithFreshModels(latestBaseProvider, providerAccountId)
        assertProviderEndpoint(provider)
        activeRoute = { provider, providerAccountId, model: activeRoute.model }
        t.providerId = latestBaseProvider.id
        t.providerAccountId = providerAccountId
        t.model = activeRoute.model
        t.modelRoute = {
          preferredProviderAccountId: exhaustedAccountId,
          failoverProviderAccountId: providerAccountId,
          resetPolicy: modelRouteResetPolicy(latestBaseProvider, exhaustedAccountId),
          failedOverAt: Date.now(),
          resetAt: routed.signal.resetAt,
          reason: routed.route.audit.reason,
        }
        job = patchJob(job, {
          metadata: {
            providerId: latestBaseProvider.id,
            providerAccountId,
            model: activeRoute.model,
            modelRoute: { action: 'ticket_failover', ...t.modelRoute },
          },
        })
        t.updatedAt = Date.now()
        persistThreads()
        emitToRenderer({ runId: '', threadId: t.id, type: 'thread_update', patch: { providerId: t.providerId, providerAccountId, model: activeRoute.model, modelRoute: t.modelRoute } })
        addThreadNotice(t, '', modelRouteFailoverText(latestBaseProvider, routed.route.audit, routed.signal))
        if (room && input.source?.roomId) {
          appendRoomSupervisorEvent(room, `Ticket worker "${ticket.title}" failed over from ${modelRouteAccountLabel(latestBaseProvider, exhaustedAccountId)} to ${modelRouteAccountLabel(latestBaseProvider, providerAccountId)} after ${routed.signal.kind.replace(/_/g, ' ')}.`, {
            tool: 'model_router_failover',
            args: { ticketId: ticket.id, phase, threadId: t.id, from: exhaustedAccountId, to: providerAccountId, model: activeRoute.model, resetAt: routed.signal.resetAt },
          })
        }
        injectThreadRouteHandoff(t, `Ticket worker "${ticket.title}" failed over after ${routed.signal.kind.replace(/_/g, ' ')}.`)
        try {
          await runTicketWithRoute(activeRoute)
        } catch (retryErr) {
          emitToRenderer({ runId: '', threadId: t.id, type: 'error', message: String((retryErr as any)?.message ?? retryErr) })
        }
        return
      }
      emitToRenderer({ runId: '', threadId: t.id, type: 'error', message: String((err as any)?.message ?? err) })
    })
    .finally(() => {
      markUndrainedCues(t, controller.signal.aborted ? 'canceled' : 'run ended')
      aborters.delete(t.id)
      runLease.release()
      const failed = t.timeline.some((i) => i.kind === 'error')
      const interrupted = controller.signal.aborted
      t.status = failed ? 'error' : 'idle'; t.updatedAt = Date.now()
      // When the run was aborted/paused (no completion), park the ticket without
      // claiming it as reviewed/completed: keep its in_progress status and clear
      // the active runState so it matches the paused job and can be re-dispatched.
      const settled = ticketStateAfterRun(ticket, { phase, failed, interrupted })
      ticket.runState = settled.runState
      ticket.status = settled.status
      ticket.updatedAt = Date.now()
      const summary = summarizeTicketWorkerOutcome(ticket, t, phase, failed, input.source?.seat ?? model)
      if (ticket.execution) {
        ticket.execution = settleTicketExecution(ticket.execution, {
          outcome: { phase, failed, interrupted },
          finishedAt: Date.now(),
          commands: summary.signals.commands,
          blockers: summary.signals.blockers,
        })
      }
      ticket.evidence = [
        ...(ticket.evidence ?? []),
        ...summary.evidence,
      ]
      const settledJobId = job?.id ?? activeJob('ticket', ticket.id)?.id ?? null
      if (room && input.source?.roomId) {
        appendRoomSupervisorEvent(room, summary.text, {
          tool: 'ticket_worker_result',
          args: {
            ticketId: ticket.id,
            phase,
            threadId: t.id,
            failed,
            status: ticket.status,
            files: summary.signals.files.slice(0, 8),
            diffs: summary.signals.diffs,
            commands: summary.signals.commands.map((item) => ({ command: item.command, state: item.state, output: item.output })),
            blockers: summary.signals.blockers,
            testsPassed: summary.testsPassed,
            codeChanged: summary.codeChanged,
          },
        })
      }
      finishJob(job, controller.signal.aborted ? 'paused' : failed ? 'failed' : 'completed', failed ? 'Ticket agent run ended with an error.' : undefined)
      if (room && input.source?.roomId) input.source.onSettled?.({ codeChanged: summary.codeChanged, testsPassed: summary.testsPassed, failed: failed || controller.signal.aborted, jobId: settledJobId })
      persistThreads(); persistTickets(); emitTicket(ticket)
    })
  launched = true
  ticketStarts.delete(ticket.id)
  return thread
  } catch (error) {
    ticketStarts.delete(ticket.id)
    if (reservedThreadId && reservedController && aborters.get(reservedThreadId) === reservedController) aborters.delete(reservedThreadId)
    runLease.release()
    throw error
  } finally {
    if (!launched) ticketStarts.delete(ticket.id)
  }
}
  return { startTicketWork, isTicketStarting: (ticketId) => ticketStarts.has(ticketId) }
}
