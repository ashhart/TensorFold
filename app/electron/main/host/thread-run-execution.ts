import { restrictTicketToolContext } from '../../agent/ticket-tool-policy'
import { selectedResponseLocale } from '../../agent/response-language'
import { RunRecoveryTracker } from '../../agent/run-recovery/tracker'
import { assertThreadMemoryTransport, canContributeThreadMemory } from '../../../shared/thread-memory-policy'
import { assertGoogleContextTransport } from '../../../shared/google-workspace-privacy'
import { randomUUID } from 'node:crypto'
import { runAgent } from '../../agent/loop'
import { runOmpThread } from '../../agent/omp-engine'
import { assertOmpEngineAllowed } from '../../agent/planning-guard'
import { executeTool, isReadOnlyTool } from '../../agent/tools'
import { createSubAgentFromLaunch, parentForThread, updateSubAgent } from '../../agent/subagents'
import { saveSubAgents } from '../../store'
import { selectChannelPrompt } from '../../agent/channel-prompt'
import { resolvePersona, personaSystemBlock } from '../../agent/persona'
import { createProjectMemoryCandidate } from '../../agent/project-memory'
import { projectMemoryAutoCandidateTurn, projectMemoryAutoMessages } from './project-memory-auto-policy'
import { persistProjectMemoryCandidates } from './persistence'
import { auditPreview } from '../../agent/security'
import type { AgentEvent, Provider, StartRunPayload, Thread, WorkspaceLocation } from '../../agent/types'
import { store } from '../state'
import { auditEvent } from './audit'
import { botInstructions, botRunApprovalMode } from '../../../shared/bot-policy'
import { scopeBotToolContext } from '../../agent/bot-tool-scope'
import { createBotVmToolContext, getBotVmHost } from './bot-vm-host'
import { assertBotLifecycleIdle } from './bot-run-boundary'
import { getUserQuestionHost } from './user-question-host'
import { isExternalProvider } from '../../../shared/provider-trust'
import { INCOGNITO_PERSISTENT_TOOLS } from '../../agent/incognito'
import { PluginRuntime } from '../../agent/plugin-runtime'
import type { UserQuestionInput, UserQuestionRun } from '../../agent/types'

import type { ApprovalMode, ThreadMemoryPolicy } from '../../agent/types'
import type { ThreadRuntimeDeps, ThreadRunAdmission, JobHandle } from './thread-runtime'
import type { ThreadRunSupport } from './thread-run-start'
import { assertIncognitoThreadTransportSupported, assertSshToolContextConnected } from './thread-run-policy'
import { botRunReasoningEffort } from '../../../shared/bot-reasoning'

export interface ThreadRoute { provider: Provider; providerAccountId: string | null; model: string }
export interface ThreadRunReview { viewedSkillNames: Set<string>; provider: Provider | null; model: string }
export interface ThreadRouteContext {
  deps: ThreadRuntimeDeps
  support: ThreadRunSupport
  thread: Thread
  controller: AbortController
  admission?: ThreadRunAdmission
  botVm: boolean
  restrictedBot: boolean
  vmHost: ReturnType<typeof getBotVmHost>
  workspace: WorkspaceLocation | null
  memoryPolicy: ThreadMemoryPolicy
  memoryRevision: number
  runId: string
  runApprovalMode: ApprovalMode
  runContract: NonNullable<Thread['activeRunContract']>
  runMaxWallMs: number
  localHelpersOnly: boolean
  payload: StartRunPayload
  jobHandle: JobHandle
  review: ThreadRunReview
  assertBotResearchOff(provider: Provider | undefined): void
  emit(event: AgentEvent): void
}

export function createThreadRouteRunner(input: ThreadRouteContext) {
  const { deps, support, thread, controller, admission, botVm, restrictedBot, vmHost, workspace, memoryPolicy, memoryRevision, runId, runApprovalMode, runContract, runMaxWallMs, localHelpersOnly, payload, jobHandle, review, assertBotResearchOff, emit } = input
  const { aborters, threadJobHeartbeatMs, emitToRenderer, persistThreads, fullToolContext, approvalResolver, isHighRiskTool, heartbeatJob, renderProjectMemoryBucket, effectiveContextLimit } = deps
  const { drainThreadCues, waitForThreadCue, timelineErrorSince, timelineError } = support
  const routeProviderForModel = (routeProvider: Provider, routeModel: string, runModel: string) =>
    runModel === routeModel ? routeProvider : { ...routeProvider, models: [runModel, ...routeProvider.models.filter((candidate) => candidate !== runModel)] }

  const runWithRoute = async (route: { provider: Provider; providerAccountId: string | null; model: string }, runModel = route.model) => {
    const providerAccountId = route.providerAccountId
    admission?.assertAllowed()
    const timelineStart = thread.timeline.length
    const activeProvider = routeProviderForModel(route.provider, route.model, runModel)
    assertGoogleContextTransport(thread, activeProvider)
    assertIncognitoThreadTransportSupported(thread, activeProvider)
    if (thread.bot) {
      assertBotLifecycleIdle(thread.bot.presetId)
      botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, runApprovalMode)
      if (botVm || restrictedBot) assertBotResearchOff(activeProvider)
    }
    review.provider = activeProvider
    review.model = runModel
    let lastJobBeat = Date.now()
    const beatThreadJob = () => {
      if (Date.now() - lastJobBeat < threadJobHeartbeatMs) return
      lastJobBeat = Date.now()
      jobHandle.set(heartbeatJob(jobHandle.get()) ?? jobHandle.get())
    }
    // Provider/route discovery above can take long enough for an SSH session to drop.
    // Capture the tool context once at the final execution seam, validate that exact
    // snapshot, and pass it through unchanged. Re-resolving later (or accepting an
    // undefined backend) could reinterpret the remote path as a local lookalike.
    if (thread.bot && (activeProvider.id !== thread.bot.providerId || route.providerAccountId !== thread.bot.providerAccountId || runModel !== thread.bot.model)) throw new Error('Bots keep their captured model and account; start a new bot chat to change them.')
    const baseToolContext = botVm ? { cwd: '/workspace', signal: controller.signal, currentThread: thread } : fullToolContext({ thread, provider: activeProvider, model: runModel, signal: controller.signal, memoryPolicy })
    let toolContext = scopeBotToolContext(thread.bot, botVm ? createBotVmToolContext(vmHost!, thread.bot!.presetId, baseToolContext) : baseToolContext)
    const botTicket = thread.bot ? store.tickets.find((ticket) => ticket.threadId === thread.id && !ticket.archivedAt) : undefined
    if (botTicket) toolContext = restrictTicketToolContext(toolContext, botTicket.allowedCapabilities) as typeof toolContext
    toolContext.memoryPolicy = memoryPolicy
    assertThreadMemoryTransport(thread, activeProvider, memoryPolicy)
    if (restrictedBot) toolContext.botToolScopeGuard = () => {
      assertBotLifecycleIdle(thread.bot!.presetId)
      botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, runApprovalMode)
      assertBotResearchOff(activeProvider)
    }
    if (thread.incognito) toolContext.disabledToolNames = [...new Set([...toolContext.disabledToolNames ?? [], ...INCOGNITO_PERSISTENT_TOOLS])]
    if (botVm) toolContext.botVm!.assertModelPrivacy = () => {
      assertBotLifecycleIdle(thread.bot!.presetId)
      botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, runApprovalMode)
      assertBotResearchOff(activeProvider)
    }
    assertSshToolContextConnected(workspace, toolContext)
    const questions = getUserQuestionHost()
    if (questions && thread.engine !== 'omp' && !isExternalProvider(activeProvider) && (!activeProvider.transport || activeProvider.transport === 'chat-completions')) {
      toolContext.askUser = (input: UserQuestionInput, run: UserQuestionRun) => questions.ask(input, { ...run, threadId: thread.id, runId })
    }
    // OMP engine threads (VST-050) run ON the installed omp coding agent via its
    // headless rpc mode: omp owns the agent loop, TensorFold mirrors the timeline, plan,
    // subagents, and Guide steering. The native loop below stays the default.
    if (thread.engine === 'omp') {
      assertOmpEngineAllowed()
      const delegated = new RunRecoveryTracker(runId, snapshot => {
        if (aborters.get(thread.id) !== controller) return
        thread.runRecovery = snapshot; persistThreads()
        emitToRenderer({ runId, threadId: thread.id, type: 'thread_update', patch: { runRecovery: snapshot } })
      })
      const activeTools = new Set<Promise<unknown>>()
      let acceptingTools = true, delegatedComplete = false
      const stopDelegated = () => delegated.stopRequested()
      delegated.start(); delegated.begin('delegated'); delegated.requestStarted()
      controller.signal.addEventListener('abort', stopDelegated, { once: true })
      try { await runOmpThread({
        thread,
        runId,
        emit: (event) => { emit(event); beatThreadJob() },
        signal: controller.signal,
        persist: persistThreads,
        userText: payload.userText,
        approvalMode: runApprovalMode,
        reasoningEffort: botRunReasoningEffort(thread, store.settings.reasoningEffort),
        drainCues: () => drainThreadCues(thread.id),
        onSession: (cursor) => { thread.ompResumeCursor = cursor },
        // Host-tools bridge: omp's model plans, TensorFold's approval gate and executor
        // own every mutation (write_file/edit_file/apply_patch/run_command).
        requestApproval: approvalResolver(controller, jobHandle),
        executeTool: async (name, args) => {
          if (!acceptingTools) throw new Error('The delegated run is stopping; no further tools can start.')
          const operation = randomUUID(), mutating = !isReadOnlyTool(name)
          delegated.begin(operation, mutating)
          const pending = executeTool(name, args, { ...toolContext, currentThread: thread, signal: controller.signal })
          activeTools.add(pending)
          try { const result = await pending; delegated.toolCompleted(mutating, result.ok); return result }
          catch (error) { if (mutating) delegated.toolUncertain(); throw error }
          finally { activeTools.delete(pending); delegated.end(operation) }
        },
        updateSubAgent: (record) => {
          const existing = store.subAgents.find((agent) => agent.id === record.id)
          if (existing) {
            const updated = updateSubAgent(store.subAgents, record.id, { status: record.status, name: record.name, task: record.task })
            if (updated.agent) store.subAgents = updated.agents
          } else {
            const created = createSubAgentFromLaunch(store.subAgents, {
              source: 'thread_tool',
              parent: parentForThread(thread),
              task: record.task,
              name: record.name,
              providerId: activeProvider.id,
              providerAccountId,
              model: runModel,
              runId: `omp-${record.id}`,
              cwd: thread.cwd,
            })
            // omp mints its own stable subagent ids; keep them as the TensorFold ids so
            // later snapshots update the same records.
            created.agent.id = record.id
            store.subAgents = created.agents
          }
          void saveSubAgents(store.subAgents)
        },
      }); delegated.requestComplete(); delegatedComplete = true
      } catch (error) { delegated.requestFailed('unknown'); throw error }
      finally {
        acceptingTools = false
        await Promise.allSettled([...activeTools])
        delegated.end('delegated'); delegated.settle(delegatedComplete)
        controller.signal.removeEventListener('abort', stopDelegated)
      }
    } else {
    await runAgent({
      responseLocale: selectedResponseLocale(store.settings.uiPreferences),
      onRecovery: snapshot => {
        if (aborters.get(thread.id) !== controller) return
        thread.runRecovery = snapshot
        persistThreads()
        emitToRenderer({ runId, threadId: thread.id, type: 'thread_update', patch: { runRecovery: snapshot } })
      },
      assertRequestAllowed: () => { admission?.assertAllowed(); assertGoogleContextTransport(thread, activeProvider) },
      thread,
      provider: activeProvider,
      model: runModel,
      runId,
      approvalMode: runApprovalMode,
      temperature: store.settings.temperature,
      reasoningEffort: botRunReasoningEffort(thread, store.settings.reasoningEffort),
      serviceTier: store.settings.serviceTier,
      autoCompact: store.settings.autoCompact,
      contextLimit: effectiveContextLimit(activeProvider, runModel),
      toolContext,
      observe: thread.incognito ? undefined : deps.observe,
      pluginRuntime: thread.incognito ? new PluginRuntime() : undefined,
      drainCues: () => drainThreadCues(thread.id),
      waitForCue: (timeoutMs, signal) => waitForThreadCue(thread.id, timeoutMs, signal),
      memoryPolicy,
      projectMemory: botVm || !memoryPolicy.useExisting ? '' : renderProjectMemoryBucket(store.projects.find((p) => p.id === thread.projectId), payload.userText),
      // Resolve the persona fresh each run (hot reload) for this thread's scope.
      personaBlock: thread.bot ? botInstructions(thread.bot) : personaSystemBlock(
        resolvePersona(store.settings.personas ?? [], { isAssistant: thread.kind === 'chat', workspaceId: thread.projectId ?? null }),
        { userProfile: store.settings.userProfile },
      ),
      // Per-channel ephemeral prompt: when this run acts on a connector conversation, inject
      // that source's channelPrompt for this run only (never into the persisted snapshot).
      channelPrompt: selectChannelPrompt(payload.sourceThreadKey, store.settings) ?? undefined,
      // File durable facts salvaged at compaction as pending project-memory candidates
      // (project-scoped; skipped for project-less chat threads).
      onMemoryHarvest: (candidates) => {
        if (thread.googleLocalOnly || !canContributeThreadMemory(thread, memoryPolicy, memoryRevision, controller.signal) || !store.threads.includes(thread) || !thread.projectId || !candidates.length) return
        const authored = projectMemoryAutoMessages(store.projectMemory, thread)
        for (const text of candidates) {
          const sourceTurn = projectMemoryAutoCandidateTurn(authored, text)
          if (sourceTurn === null) continue
          store.projectMemoryCandidates = [createProjectMemoryCandidate({ projectId: thread.projectId, text, sourceType: 'agent', reason: 'Salvaged from the conversation at compaction.', sourceThreadId: thread.id, sourceTurn }), ...store.projectMemoryCandidates]
        }
        persistProjectMemoryCandidates()
      },
      onProviderRateLimits: deps.recordProviderRateLimits,
      guardPolicy: runContract.guardPolicy,
      maxSteps: runContract.guardPolicy.maxSteps,
      maxWallMs: runMaxWallMs,
      emit: (event) => { emit(event); beatThreadJob() },
      signal: controller.signal,
      requestApproval: approvalResolver(controller, jobHandle),
      auditToolCall: async ({ tool, args, resultOk, output }) => {
        // Note skills the agent actually read this run — the post-run review only looks at these.
        if (tool === 'skill_view' && resultOk && args?.name) review.viewedSkillNames.add(String(args.name))
        if (isHighRiskTool(tool)) await auditEvent('agent.tool', { threadId: thread.id, runId, tool, ok: resultOk, ...(thread.incognito ? {} : { cwd: auditPreview(thread.cwd), args: auditPreview(JSON.stringify(args)), output: auditPreview(output) }) })
      },
      persist: persistThreads,
    })
    }
    const errorItem = timelineErrorSince(thread, timelineStart)
    if (errorItem && !controller.signal.aborted) throw timelineError(errorItem)
  }

  return runWithRoute
}
