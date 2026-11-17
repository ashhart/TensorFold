import { resolveMessageQuotes } from '../../agent/conversation-quotation'
import { assertRecoveryAllowsNewRun } from '../../agent/run-outcome-review'
import { apiUserMessageRef } from '../../agent/message-edit'
import { assertThreadMemoryTransport, captureThreadMemoryPolicy, threadMemoryPolicyRevision } from '../../../shared/thread-memory-policy'
import { assertGoogleContextTransport } from '../../../shared/google-workspace-privacy'
import { saveThreadMemoryPolicy } from '../../store'
import { randomUUID } from 'node:crypto'
import { restartResumeApprovalMode, claimJobLease } from '../../agent/jobs'
import { vontraPath } from '../../store'
import { buildUserContent } from '../../agent/content'
import { normalizeAttachments } from '../../agent/attachment-policy'
import { appendThreadEvent } from '../../agent/thread-journal'
import { resolveProviderModel, surfaceModelOverride } from '../../agent/provider-accounts'
import { assertLocalEndpointAvailable } from '../../agent/local-endpoint'
import { routeSpendStatus } from '../../agent/model-router'
import { assertProviderEndpointResolved } from '../../agent/security'
import { ensureAgentEventTrace } from '../../agent/trace'
import { buildAgentRunContract, buildEventReceipt } from '../../agent/run-contract'
import type { AgentEvent, Job, Provider, StartRunPayload, Thread } from '../../agent/types'
import { threadTurnNumber } from './checkpoint-runtime'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { auditModelRetry, auditModelRoute, modelRouteAccountLabel, modelRouteAutoFailoverEnabled, modelRouteFailoverText, modelRouteResetPolicy, modelRouter, runtimeProviderWithFreshModels, selectedProviderAccountId } from './provider-runtime'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, sshWorkspaceKey } from './workspace-mutation-guard'
import { botRunApprovalMode } from '../../../shared/bot-policy'
import { botUsesGuardedExecution, isRestrictedBot } from '../../../shared/bot-tool-policy'
import { getBotVmHost } from './bot-vm-host'
import { assertBotLifecycleIdle, reserveBotVmRun } from './bot-run-boundary'
import { engineService } from './engine-runtime'
import { isEngineProvider } from '../../agent/engine/provider'

import type { ThreadMemoryPolicy } from '../../agent/types'
import type { ThreadRuntime, ThreadRuntimeDeps, ThreadRunAdmission, JobHandle } from './thread-runtime'
import { encodeJournalEvent, journalKey } from './thread-journal-crypto'
import { recordPreTurnCheckpoint, resolveThreadRunCwd, assertSshThreadTransportSupported, assertIncognitoThreadTransportSupported } from './thread-run-policy'
import { createThreadRouteRunner, type ThreadRouteContext, type ThreadRunReview } from './thread-run-execution'
import { settleThreadExecution } from './thread-run-settlement'
import { botRunReasoningEffort } from '../../../shared/bot-reasoning'

export interface ThreadRunSupport extends Pick<ThreadRuntime,
  'addThreadNotice' | 'applyEventToThread' | 'drainThreadCues' | 'waitForThreadCue' |
  'timelineErrorSince' | 'timelineError' | 'markRouteErrorRecovered' | 'markUndrainedCues' |
  'injectThreadRouteHandoff'> {
  availableModelsFromProviderError(error: unknown): string[]
  compactThreadJournalTail(threadId: string): Promise<void>
  reviewViewedSkills(input: { provider: Provider; model: string; thread: Thread; userText: string; viewed: string[]; signal: AbortSignal; memoryPolicy: ThreadMemoryPolicy; memoryRevision: number }): Promise<void>
  proposePostRunMemory(thread: Thread, policy: ThreadMemoryPolicy, revision: number, signal: AbortSignal): void
  maybeAutoTitle(thread: Thread, provider: Provider, model: string, text: string): void
}

export function createThreadRunStarter(deps: ThreadRuntimeDeps, support: ThreadRunSupport): ThreadRuntime['startThreadRun'] {
  const { aborters, jobLeaseOwner, sshManager, emitToRenderer, persistThreads, claimJob, patchJob, replaceJobResult, effectiveContextLimit } = deps
  const { addThreadNotice, applyEventToThread, maybeAutoTitle } = support

  async function startThreadRun(payload: StartRunPayload, existingJob?: Job, admission?: ThreadRunAdmission): Promise<{ runId: string }> {
    requireSecurityRole('operator', 'start agent runs')
    admission?.assertAllowed()
    if (Object.prototype.hasOwnProperty.call(payload, 'memoryPolicy')) throw new Error('Memory choices must be saved before starting a run.')
    const thread = store.threads.find((t) => t.id === payload.threadId)
    if (!thread) throw new Error('Thread not found')
    if (thread.archivedAt) throw new Error('Restore this archived thread before starting it.')
    const quotes = resolveMessageQuotes(thread, payload.quotes)
    if (quotes.length && (payload.sourceThreadKey || thread.engine === 'omp')) throw new Error('Quoted replies need a local TensorFold chat API conversation.')
    const runApprovalMode = botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, payload.approvalMode ?? restartResumeApprovalMode(existingJob?.resumeAfterRestart, store.settings.approvalMode))
    const botVm = thread.bot?.executionTarget === 'vm'
    const restrictedBot = isRestrictedBot(thread.bot)
    const localHelpersOnly = botUsesGuardedExecution(thread.bot) || thread.phoneLocalOnly === true || thread.googleLocalOnly === true
    const assertBotResearchOff = (provider: Provider | undefined) => {
      const engine = engineService()
      // Also cover custom API aliases of the built-in server, including startup races.
      if (engine?.hasExternalResearchEnabled() || isEngineProvider(provider) && !engine) throw new Error('Turn off Cortheon in AI settings before using an isolated or restricted bot; hidden research cannot enforce its execution boundary.')
    }
    const vmHost = botVm ? getBotVmHost() : null
    if (thread.bot) {
      assertBotLifecycleIdle(thread.bot.presetId)
      if (thread.engine && thread.engine !== 'native') throw new Error('Bot chats require TensorFold permission checks.')
      if (thread.providerId !== thread.bot.providerId || (thread.providerAccountId ?? null) !== thread.bot.providerAccountId || thread.model !== thread.bot.model) throw new Error('The model differs from the captured bot configuration; start a new bot chat.')
      const ticketBotThread = !botVm && store.tickets.some((ticket) => ticket.threadId === thread.id && ticket.projectId === thread.projectId && !ticket.archivedAt)
      if (thread.workspace || (!ticketBotThread && (thread.projectId || (botVm ? thread.cwd !== '/workspace' : !!thread.cwd)))) throw new Error('The bot execution location has changed; start a new bot chat.')
      if (botVm && !vmHost) throw new Error('The bot VM service is unavailable; the run will not use this Mac instead.')
    }
    // Guard against a second concurrent run for the same thread: it would overwrite the shared
    // aborters[threadId] entry, so the first run's finally cleanup would delete the live run's
    // controller (breaking cancel + cue queueing) and both runs would interleave apiMessages.
    // Checked here (before any thread mutation) so a rejected start leaves state untouched.
    if (aborters.has(thread.id)) throw new Error('Thread already running')
    assertRecoveryAllowsNewRun(thread)
    const project = thread.projectId ? store.projects.find((p) => p.id === thread.projectId) ?? null : null
    if (thread.projectId && (!project || project.archivedAt)) throw new Error('Restore or choose an active project before starting this thread.')
    const workspace = thread.workspace ?? project?.workspace ?? null
    const workspaceBoundary = payload.workspaceBoundary ?? thread.workspaceBoundary ?? project?.workspaceBoundary
    if (workspaceBoundary === 'home') await resolveThreadRunCwd(thread, workspace, workspaceBoundary)
    const attachments = normalizeAttachments(payload.attachments)
    const surfaceRoute = thread.providerId ? null : surfaceModelOverride(store.settings, thread.kind === 'chat' ? 'chat' : 'code')
    const baseProvider = store.settings.providers.find((p) => p.id === (thread.providerId ?? surfaceRoute?.providerId ?? store.settings.activeProviderId))
    assertGoogleContextTransport(thread, baseProvider)
    assertIncognitoThreadTransportSupported(thread, baseProvider)
    assertThreadMemoryTransport(thread, baseProvider)
    if (thread.incognito && existingJob && (existingJob.incognito !== true || existingJob.threadId !== thread.id || existingJob.currentThreadId !== thread.id)) throw new Error('An Incognito task cannot resume a saved or unrelated job; start a new private turn.')
    assertSshThreadTransportSupported(workspace, baseProvider)
    const initialWorkspaceKey = workspace?.kind === 'ssh'
      ? sshWorkspaceKey(workspace.hostId, workspace.path)
      : thread.kind === 'chat' || botVm ? null : lexicalLocalWorkspaceKey(workspace?.kind === 'local' ? workspace.path : thread.cwd)
    // Admission and publishing aborters below are synchronous: a comparison cannot start
    // between the local engine check and this run owning its host reservation.
    admission?.assertAllowed()
    if (isEngineProvider(baseProvider)) (deps.assertLocalEngineAdmission ?? (() => engineService()?.assertLocalWorkAdmission()))()
    deps.workspaceMutationGuard.assertThreadStartAllowed(thread.id, initialWorkspaceKey)
    const runLease = deps.workspaceMutationGuard.reserveStart({ threadId: thread.id, workspaceKeys: [initialWorkspaceKey] })
    let releaseBotVmRun = () => {}
    try { if (botVm) releaseBotVmRun = reserveBotVmRun(thread.bot!.presetId, thread.id) } catch (error) { runLease.release(); throw error }
    const controller = new AbortController()
    aborters.set(thread.id, controller)
    try {
    admission?.onController(controller)
    admission?.assertAllowed()
    const memoryPolicy = captureThreadMemoryPolicy(thread)
    if (botVm || restrictedBot) assertBotResearchOff(baseProvider)
    if (botVm) {
      await vmHost!.requireReady(thread.bot!.presetId)
      if (controller.signal.aborted) throw new Error('Run cancelled before VM startup completed.')
      assertBotResearchOff(baseProvider)
    }
    const canonicalWorkspaceKey = workspace?.kind === 'ssh'
      ? sshWorkspaceKey(workspace.hostId, workspace.path)
      : thread.kind === 'chat' || botVm ? null : await canonicalLocalWorkspaceKey(workspace?.kind === 'local' ? workspace.path : thread.cwd)
    runLease.addWorkspaceKeys?.([canonicalWorkspaceKey])
    if (workspace?.kind === 'ssh' && workspace.hostId) {
      try {
        await sshManager.ensureConnected(workspace.hostId)
      } catch (e) {
        const message = `SSH connection failed: ${e instanceof Error ? e.message : String(e)}`
        emitToRenderer({ runId: '', threadId: thread.id, type: 'error', message })
        throw new Error(message)
      }
    }

    // Per-surface defaults (VST-015): a thread that never had a provider stamped on it
    // launches on its SURFACE's default (chat → assistant model, task → coding model) before
    // falling back to the globally-active route. An explicitly-configured thread is untouched.
    if (!baseProvider) throw new Error('No provider configured. Open Settings to add one.')
    const storedProviderAccountId = Object.prototype.hasOwnProperty.call(thread, 'providerAccountId')
      ? (thread.providerAccountId ?? null)
      : surfaceRoute && surfaceRoute.providerId === baseProvider.id && surfaceRoute.providerAccountId
        ? surfaceRoute.providerAccountId
        : selectedProviderAccountId(baseProvider, undefined)
    let requestedProviderAccountId = storedProviderAccountId
    const routeMemory = thread.modelRoute ?? null
    const resetReturn = routeMemory?.resetPolicy === 'return_after_reset' &&
      routeMemory.resetAt != null &&
      routeMemory.resetAt <= Date.now() &&
      routeMemory.failoverProviderAccountId === storedProviderAccountId &&
      routeMemory.preferredProviderAccountId !== storedProviderAccountId
        ? { from: routeMemory.failoverProviderAccountId, to: routeMemory.preferredProviderAccountId, resetAt: routeMemory.resetAt }
        : null
    if (resetReturn) requestedProviderAccountId = resetReturn.to
    const requestedModel = thread.model ?? (surfaceRoute && surfaceRoute.providerId === baseProvider.id ? surfaceRoute.model : null) ?? store.settings.activeModel
    const threadRouteRole = thread.kind === 'chat' ? 'chat' : 'task'
    if (project?.localOnly && baseProvider.kind !== 'local') {
      const blockedModel = resolveProviderModel(baseProvider, requestedProviderAccountId, requestedModel) ?? requestedModel ?? store.settings.activeModel ?? baseProvider.models[0] ?? 'unknown'
      const blockedRoute = modelRouter.select({
        role: threadRouteRole,
        provider: baseProvider,
        model: blockedModel,
        requestedAccountId: requestedProviderAccountId,
        allowFailover: modelRouteAutoFailoverEnabled(baseProvider, requestedProviderAccountId),
        localOnly: true,
        contextLimit: effectiveContextLimit(baseProvider, blockedModel, requestedProviderAccountId),
        reasoningEffort: botRunReasoningEffort(thread, store.settings.reasoningEffort),
        budget: thread.kind,
        source: 'thread_local_only_guard',
      })
      await auditModelRoute(thread, blockedRoute.audit)
      throw new Error(blockedRoute.audit.reason)
    }
    const requestedProvider = await runtimeProviderWithFreshModels(baseProvider, requestedProviderAccountId)
    const routeBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
    const model = resolveProviderModel(requestedProvider, null, requestedModel)
    if (!model) throw new Error('No model selected.')
    if (thread.bot && model !== thread.bot.model) throw new Error('The saved bot model is unavailable; edit the bot and start a new chat.')
    const initialRoute = modelRouter.select({
      role: threadRouteRole,
      provider: routeBaseProvider,
      model,
      requestedAccountId: requestedProviderAccountId,
      allowFailover: !thread.bot && modelRouteAutoFailoverEnabled(routeBaseProvider, requestedProviderAccountId),
      localOnly: project?.localOnly === true,
      contextLimit: effectiveContextLimit(routeBaseProvider, model, requestedProviderAccountId),
      reasoningEffort: botRunReasoningEffort(thread, store.settings.reasoningEffort),
      budget: thread.kind,
      source: 'thread_initial_route',
    })
    if (!initialRoute.selected) {
      await auditModelRoute(thread, initialRoute.audit)
      throw new Error(initialRoute.audit.reason)
    }
    let providerAccountId = initialRoute.selectedAccountId
    let provider = providerAccountId === requestedProviderAccountId
      ? requestedProvider
      : await runtimeProviderWithFreshModels(routeBaseProvider, providerAccountId)
    await assertProviderEndpointResolved(provider)
    // Local endpoint availability (VST-015): if a local provider's server is not listening,
    // refuse now with a named error instead of failing mid-run with a generic fetch error.
    await assertLocalEndpointAvailable(provider)
    if (thread.providerAccountId !== providerAccountId || thread.model !== model) {
      const staleModel = requestedModel && requestedModel !== model ? requestedModel : null
      thread.providerId = baseProvider.id
      thread.providerAccountId = providerAccountId
      if (resetReturn && providerAccountId === resetReturn.to) thread.modelRoute = null
      thread.model = model
      thread.updatedAt = Date.now()
      persistThreads()
      emitToRenderer({ runId: '', threadId: thread.id, type: 'thread_update', patch: { providerId: thread.providerId, providerAccountId, model, modelRoute: thread.modelRoute } })
      if (staleModel) {
        const notice = { id: randomUUID(), kind: 'notice' as const, text: `Model "${staleModel}" is unavailable. Switched to "${model}".`, createdAt: Date.now() }
        thread.timeline.push(notice)
        emitToRenderer({ runId: '', threadId: thread.id, type: 'item_add', item: notice })
        await auditModelRetry(thread, {
          providerId: routeBaseProvider.id,
          providerAccountId,
          fromModel: staleModel,
          toModel: model,
          source: 'startup_model_resolution',
          reason: `Saved model "${staleModel}" was unavailable for the selected route.`,
          availableModels: provider.models,
        })
      }
    }
    if (resetReturn && providerAccountId === resetReturn.to) {
      addThreadNotice(thread, '', `Model router returned to ${modelRouteAccountLabel(routeBaseProvider, providerAccountId)} after the reset window.`)
      await auditEvent('model_router.return_after_reset', {
        threadId: thread.id,
        providerId: routeBaseProvider.id,
        model,
        previousAccountId: resetReturn.from,
        selectedAccountId: providerAccountId,
        resetAt: resetReturn.resetAt,
      })
    }
    if (initialRoute.audit.action === 'failover') {
      const resetAt = initialRoute.candidates.find((candidate) => candidate.accountId === initialRoute.requestedAccountId)?.exhaustedUntil ?? null
      thread.modelRoute = {
        preferredProviderAccountId: initialRoute.requestedAccountId,
        failoverProviderAccountId: initialRoute.selectedAccountId,
        resetPolicy: modelRouteResetPolicy(routeBaseProvider, initialRoute.requestedAccountId),
        failedOverAt: Date.now(),
        resetAt,
        reason: initialRoute.audit.reason,
      }
      thread.updatedAt = Date.now()
      persistThreads()
      emitToRenderer({ runId: '', threadId: thread.id, type: 'thread_update', patch: { modelRoute: thread.modelRoute } })
      addThreadNotice(thread, '', modelRouteFailoverText(routeBaseProvider, initialRoute.audit))
      await auditModelRoute(thread, initialRoute.audit)
    }
    if (thread.kind !== 'chat' && !botVm) thread.cwd = await resolveThreadRunCwd(thread, workspace, workspaceBoundary)

    if (controller.signal.aborted) throw new Error('Run cancelled before startup completed.')
    const acceptedMemoryThread = await saveThreadMemoryPolicy(() => store.threads, thread.id, memoryPolicy, threadMemoryPolicyRevision(thread))
    const memoryRevision = threadMemoryPolicyRevision(acceptedMemoryThread)
    // Publish the durable startup CAS pair before a possible Stop returns control to the UI.
    emitToRenderer({ runId: '', threadId: thread.id, type: 'thread_update', patch: { memoryPolicy: { ...captureThreadMemoryPolicy(acceptedMemoryThread) }, memoryPolicyRevision: memoryRevision } })
    if (controller.signal.aborted) throw new Error('Run cancelled after memory choices were accepted.')
    assertGoogleContextTransport(thread, provider)
    admission?.assertAllowed()
    if (quotes.length && provider.transport && provider.transport !== 'chat-completions') throw new Error('Quoted replies need a model using the TensorFold chat API.')
    resolveMessageQuotes(thread, quotes)
    const userContent = buildUserContent(payload.userText, attachments, quotes)
    const userItem = { id: randomUUID(), kind: 'user' as const, apiMessageRef: apiUserMessageRef(thread.apiMessages, userContent, thread.compactions?.length ?? 0), memorySource: payload.sourceThreadKey ? 'external' as const : 'user' as const, text: payload.userText, attachments, ...(quotes.length ? { quotedMessages: quotes } : {}), createdAt: Date.now() }
    thread.timeline.push(userItem)
    thread.apiMessages.push({ role: 'user', content: userContent })
    thread.status = 'running'
    thread.updatedAt = Date.now()
    emitToRenderer({ runId: '', threadId: thread.id, type: 'item_add', item: userItem })

    const runId = randomUUID()
    const runMaxWallMs = 30 * 60_000
    const runContract = buildAgentRunContract({
      thread,
      runId,
      objective: payload.userText,
      provider,
      providerAccountId,
      model,
      approvalMode: runApprovalMode,
      maxWallMs: runMaxWallMs,
    })
    let job: Job | null = existingJob
      ? replaceJobResult(claimJobLease(store.jobs, existingJob.id, jobLeaseOwner))
      : claimJob({
          kind: 'thread',
          incognito: thread.incognito === true,
          owner: thread.id,
          projectId: thread.projectId ?? null,
          threadId: thread.id,
          currentThreadId: thread.id,
          resumeAfterRestart: 'ask',
          metadata: { title: thread.title, kind: thread.kind, model },
        })
    job = patchJob(job, {
      currentRunId: runId,
      currentThreadId: thread.id,
      blockedReason: null,
      executionHostId: workspace?.kind === 'ssh' ? workspace.hostId : null,
      executionVerification: workspace?.kind === 'ssh' ? 'live' : null,
      executionVerificationReason: workspace?.kind === 'ssh' ? 'Remote workspace connected for this run.' : null,
      executionVerifiedAt: workspace?.kind === 'ssh' ? Date.now() : null,
      metadata: {
        currentRunId: runId,
        currentThreadId: thread.id,
        currentThreadTitle: thread.title,
        currentThreadKind: thread.kind,
        providerId: baseProvider.id,
        providerAccountId,
        model,
      },
    })
    const jobHandle: JobHandle = {
      get: () => job,
      set: (next) => { job = next },
    }

    const emit = (e: AgentEvent) => {
      const traced = ensureAgentEventTrace(e, 'agent.run')
      applyEventToThread(thread, traced)
      // Append to the crash-safe per-thread NDJSON journal: bounds data loss to the last unflushed
      // event (not the last whole-thread snapshot) and gives an ordered, replayable run record.
      // Encrypt the event with the app key so the on-disk journal matches threads.json's
      // at-rest encryption posture (the journal holds the full plaintext transcript otherwise).
      const journalEvent = thread.incognito ? null : encodeJournalEvent(traced, journalKey())
      if (journalEvent !== null) void appendThreadEvent(vontraPath('thread-journals'), thread.id, journalEvent)
      emitToRenderer(traced)
    }

    // Everything after the user turn belongs to this run's durable contract/receipt window.
    const runTimelineStart = thread.timeline.length
    emit({ runId, threadId: thread.id, type: 'thread_update', patch: { activeRunContract: runContract } })
    emit({
      runId,
      threadId: thread.id,
      type: 'item_add',
      item: {
        id: randomUUID(),
        kind: 'notice',
        title: 'Run contract',
        text: `Run contract saved: ${runContract.surface} on ${runContract.targetKind} with ${runContract.model || 'no model'}.`,
        state: 'ok',
        runContract,
        eventReceipt: buildEventReceipt({
          kind: 'run_contract_saved',
          runId,
          threadId: thread.id,
          state: 'ok',
          summary: 'Run contract saved before model execution.',
          evidenceRefs: [runContract.id],
        }),
        createdAt: Date.now(),
      },
    })

    // Spend telemetry (VST-015): a near-cap budget WARNS into the timeline before the first
    // model call. The hard stop needs no gate here — the initial modelRouter.select() above
    // already refuses over-budget routes (applySpendControls → no_route → the throw at the
    // route check), which the renderer surfaces as a visible run-refused error notice.
    const initialSpend = routeSpendStatus(routeBaseProvider, providerAccountId)
    if (initialSpend.warning) {
      emit({ runId, threadId: thread.id, type: 'item_add', item: { id: randomUUID(), kind: 'notice', title: 'Budget', text: initialSpend.warning, state: 'ok', createdAt: Date.now() } })
    }

    if (!localHelpersOnly && !thread.googleLocalOnly && (thread.title === 'New task' || thread.title === 'New chat')) maybeAutoTitle(thread, provider, model, payload.userText)

    const review: ThreadRunReview = { viewedSkillNames: new Set(), provider: null, model: '' }
    const routeContext: ThreadRouteContext = {
      deps,
      support,
      thread,
      controller,
      admission,
      botVm,
      restrictedBot,
      vmHost,
      workspace,
      memoryPolicy,
      memoryRevision,
      runId,
      runApprovalMode,
      runContract,
      runMaxWallMs,
      localHelpersOnly,
      payload,
      jobHandle,
      review,
      assertBotResearchOff,
      emit,
    }
    const runWithRoute = createThreadRouteRunner(routeContext)

    let activeRoute = { provider, providerAccountId, model }
    // Scope the run's success/failure to items THIS run appends — a leftover error item from a
    // previous run must not mark a later successful run as failed (which stalls the PR loop).

    // Skill self-improvement bookkeeping: which skills the agent consulted this run, and the
    // provider/model it ran on (so the post-run review reuses the same backend). Populated in
    // auditToolCall + runWithRoute; consumed in the .finally review pass.

    // Repo-authored skills rescan (VST-010): awaited so this run's context reflects the
    // trust state of the content on disk RIGHT NOW (a just-edited trusted skill drops out
    // until re-reviewed). Missing dirs and scan errors degrade to "no project skills".
    if (deps.refreshProjectSkills && thread.projectId) {
      const project = store.projects.find((p) => p.id === thread.projectId)
      if (project) await deps.refreshProjectSkills({ id: project.id, cwd: project.cwd, workspace }).catch(() => {})
    }

    // Finish the pre-turn file snapshot before the model can invoke a write tool.
    // Capture remains best-effort, but fire-and-forget here would race a fast local
    // model and could accidentally snapshot post-edit state. The returned SHA ties
    // the conversation record to a file ref only when capture really succeeded.
    const checkpointTurn = threadTurnNumber(thread.timeline)
    await recordPreTurnCheckpoint({
      threadId: thread.id,
      incognito: thread.incognito,
      turn: checkpointTurn,
      target: { cwd: thread.cwd, workspace, kind: thread.kind },
      conversation: {
        apiMessages: thread.apiMessages ?? [],
        timeline: thread.timeline ?? [],
        goal: thread.goal,
        goalStatus: (thread as { goalStatus?: string }).goalStatus,
        plan: (thread as { plan?: unknown[] }).plan,
        status: thread.status,
      },
    }, deps)

    settleThreadExecution(runWithRoute(activeRoute), {
      ...routeContext, initialRoute: activeRoute, baseProvider, project, threadRouteRole,
      runTimelineStart, runLease, releaseBotVmRun, runWithRoute,
    })

    return { runId }
    } catch (err) {
      if (aborters.get(thread.id) === controller) aborters.delete(thread.id)
      runLease.release()
      releaseBotVmRun()
      throw err
    }
  }

  return startThreadRun
}
