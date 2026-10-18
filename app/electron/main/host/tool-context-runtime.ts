import { documentEventAllowed, documentPolicySignature, scopedDocumentLibrary, scopedDocumentSettings, validateDocumentSourceIds } from '../../document-library/scope'
import { selectedResponseLocale } from '../../agent/response-language'
import { documentLibrary } from './document-library-runtime'
import { assertSavedMemoryRead, assertMemoryContribution, captureThreadMemoryPolicy, assertThreadMemoryTransport, effectiveThreadMemoryPolicy, canContributeThreadMemory, threadMemoryPolicyRevision, roomWithoutSavedRecall } from '../../../shared/thread-memory-policy'
import { assertThreadChildSupported } from '../../../shared/thread-child-policy'
import { canRecallThread } from '../../agent/session-recall'
import { scopeProjectMemoryEntries } from '../../agent/project-memory'
import { randomUUID } from 'node:crypto'
import { runAgent } from '../../agent/loop'
import { assertProviderEndpoint } from '../../agent/security'
import { runControllerAction } from '../../agent/controller'
import { runComputerUse } from './computer-use-host'
import { phoneHost } from './phone-host'
import { engineService } from './engine-runtime'
import { isExternalProvider } from '../../../shared/provider-trust'
import { stripIncognitoThreads } from '../../agent/incognito'
import type { PhoneToolInput } from '../../agent/types'
import type { ComputerUseInput } from '../../agent/computer-use/actions'
import {
  appendSubAgentMessage,
  appendSubAgentEvent,
  cancelSubAgent,
  completeSubAgent,
  createSubAgentFromLaunch,
  failSubAgent,
  finalSubAgentText,
  parentForThread,
  readSubAgentMessages,
  startSubAgent,
  summarizeSubAgentTimeline,
} from '../../agent/subagents'
import { readOnlyChildRunError } from '../../agent/run-settlement'
import { resolveDelegationRoute } from '../../agent/delegation-route'
import { modelRouter, runtimeProvider } from './provider-runtime'
import { applySubAgentPreset, normalizeSubAgentPreset } from '../../agent/subagent-presets'
import { skillsForContext } from '../../agent/skill-access'
import { createArtifactRecord, type CreateArtifactInput } from '../../agent/artifacts'
import { threadTurnNumber } from './checkpoint-runtime'
import type { AgentEvent, Project, Provider, Skill, SubAgent, Thread, WorkspaceLocation } from '../../agent/types'
import type { ControllerActionInput } from '../../agent/controller-actions'
import { store } from '../state'
import { addSuggestion, type SuggestionInput } from '../../agent/suggestions'
import { persistArtifacts, persistSuggestions } from './persistence'
import type { SshManager } from './ssh'
import { blindSearchToken, searchSessionIndex, saveThreads, vontraPath } from '../../store'
import { sendToRenderer } from './renderer-events'
import { createManagedOutputStore } from './managed-outputs'
import { isWhatsAppEventInCurrentScope } from '../../../shared/whatsapp-event-scope'
import { createGoogleContextMarker, createGoogleWorkspaceToolRunner } from './google-workspace-tool-runtime'
import { googleWorkspaceService } from './google-workspace-runtime'

export interface ToolContextRuntime {
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal; subAgentId?: string; memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy; memoryOwner?: { id: string; revision: number } }): any
  runReadOnlySubAgent(
    agent: SubAgent,
    provider: Provider,
    model: string,
    input: {
      task: string
      parentLabel: string
      cwd: string
      projectId?: string | null
      providerAccountId?: string | null
      project?: Project | null
      workspace?: WorkspaceLocation | null
      signal: AbortSignal
      memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy
      incognito?: boolean
      memoryOwner?: { id: string; revision: number }
      maxSteps?: number
      maxWallMs?: number
    },
  ): Promise<{ agent: SubAgent; output: string }>
}

interface ToolContextRuntimeDeps {
  sshManager: SshManager
  persistSubAgents(): void
  persistThreads(): void
  applyEventToThread(thread: Thread, event: AgentEvent): void
  emitToRenderer(event: AgentEvent): void
  emitPluginEvent(event: AgentEvent): void
  renderProjectMemoryBucket(project: Project | undefined): string
  recordEmailDraft: any
  recordEmailEvent: any
  recordCalendarEvent: any
  recordMessageEvent: any
  runBrowserAutomation: any
  mcpToolContext(): any
  // Ends a thread's run; the computer-use indicator's Stop uses it.
  cancelThreadRun(threadId: string): void
  // Trusted repo-authored skills for a project (VST-010), served from the scan cache so
  // this stays synchronous. Absent → global skills only.
  projectSkills?(projectId: string | null | undefined, workspace?: WorkspaceLocation | null): Skill[]
  // Observer hooks (VST-012): subagent lifecycle observation, fail-open.
  observe?: (kind: 'subagent.start' | 'subagent.end', payload: Record<string, unknown>) => void
}

export function createToolContextRuntime(deps: ToolContextRuntimeDeps): ToolContextRuntime {
  const {
    sshManager,
    persistSubAgents,
    applyEventToThread,
    renderProjectMemoryBucket,
    recordEmailDraft,
    recordEmailEvent,
    recordCalendarEvent,
    recordMessageEvent,
    runBrowserAutomation,
    mcpToolContext,
  } = deps
  const managedOutputs = createManagedOutputStore(vontraPath('outputs'))
  const markGoogleContext = createGoogleContextMarker({ persist: () => saveThreads(store.threads), publish: (thread) => sendToRenderer('thread:update', thread) })

  async function runReadOnlySubAgent(
    agent: SubAgent,
    provider: Provider,
    model: string,
    input: {
      task: string
      parentLabel: string
      cwd: string
      projectId?: string | null
      providerAccountId?: string | null
      project?: Project | null
      workspace?: WorkspaceLocation | null
      signal: AbortSignal
      memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy
      incognito?: boolean
      memoryOwner?: { id: string; revision: number }
      maxSteps?: number
      maxWallMs?: number
    },
  ) {
    const storedAgent = store.subAgents.find((candidate) => candidate.id === agent.id)
    if (!storedAgent) throw new Error('The owning sub-agent is unavailable; refusing to start an unregistered child.')
    // The saved host record owns this assignment. A caller-supplied parent or memory
    // owner may add restrictions, but cannot replace the real owner with another chat.
    const ownedIds = [...new Set([storedAgent.threadId, storedAgent.parent.threadId].filter((id): id is string => !!id))]
    const ownerId = ownedIds[0] ?? input.memoryOwner?.id
    for (const id of new Set([...ownedIds, ...(input.memoryOwner?.id ? [input.memoryOwner.id] : [])])) {
      const actualOwner = store.threads.find((thread) => thread.id === id)
      if (!actualOwner || actualOwner.archivedAt) throw new Error('The owning chat is unavailable; refusing to create a permissive child.')
      assertThreadChildSupported(actualOwner)
    }
    if (input.memoryOwner && ownedIds.length && !ownedIds.includes(input.memoryOwner.id)) throw new Error('The child memory owner does not match its host-owned parent.')
    const owner = ownerId ? store.threads.find((thread) => thread.id === ownerId) : undefined
    if (ownerId && (!owner || owner.archivedAt)) throw new Error('The owning chat is unavailable; refusing to create a permissive child.')
    const inheritedPolicy = captureThreadMemoryPolicy(owner ?? {}, input.memoryPolicy)
    const inheritedOwner = input.memoryOwner ?? (owner ? { id: owner.id, revision: threadMemoryPolicyRevision(owner) } : undefined)
    const documentOwners = new Map([...new Set([...ownedIds, ...(input.memoryOwner?.id ? [input.memoryOwner.id] : [])])].map(id => [id, documentPolicySignature(store.threads.find(thread => thread.id === id)?.documentSourceIds)]))
    const assertRequestAllowed = () => {
      for (const id of new Set([...ownedIds, ...(input.memoryOwner?.id ? [input.memoryOwner.id] : [])])) {
        const current = store.threads.find((thread) => thread.id === id)
        if (!current || current.archivedAt) throw new Error('The owning chat is unavailable; refusing to continue this child.')
        assertThreadChildSupported(current)
        if (documentPolicySignature(current.documentSourceIds) !== documentOwners.get(id)) throw new Error('The owning chat document collections changed; this child cannot continue with stale sources.')
      }
    }
    assertRequestAllowed()
    const childThread: Thread = {
      id: `thread-${agent.id}`,
      documentSourceIds: owner?.documentSourceIds ? [...validateDocumentSourceIds(owner.documentSourceIds)!] : [],
      memoryPolicy: { ...inheritedPolicy },
      incognito: input.incognito || owner?.incognito,
      title: agent.name,
      kind: 'task',
      projectId: input.projectId ?? null,
      cwd: input.cwd,
      workspace: input.workspace ?? input.project?.workspace ?? null,
      providerId: provider.id,
      providerAccountId: input.providerAccountId ?? null,
      model,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'running',
      apiMessages: [{
        role: 'user',
        content: [
          `[Sub-agent assignment from ${input.parentLabel}]`,
          '',
          input.task,
          '',
          'Work as a read-only child agent. Use safe inspection tools when helpful. Do not edit files, run mutating commands, or claim changes.',
          'Use subagent_send for durable questions, handoffs, or heartbeats. Check subagent_inbox before your final answer for parent replies.',
          'If a reply is not yet available, continue other useful read-only work and state the unresolved question in your final handoff.',
          'Return concise findings, evidence, blockers, and the recommended next step.',
        ].join('\n'),
      }],
      timeline: [],
    }

    deps.observe?.('subagent.start', { agentId: agent.id, name: agent.name, model })
    try {
      assertProviderEndpoint(provider)
      await runAgent({
        responseLocale: selectedResponseLocale(store.settings.uiPreferences),
        assertRequestAllowed,
        thread: childThread,
        provider,
        model,
        approvalMode: 'readonly',
        temperature: store.settings.temperature,
        reasoningEffort: store.settings.reasoningEffort,
        emit: (event) => applyEventToThread(childThread, event),
        requestApproval: async () => 'deny',
        signal: input.signal,
        persist: () => {},
        autoCompact: true,
        maxSteps: input.maxSteps ?? 12,
        maxWallMs: input.maxWallMs ?? 180_000,
        memoryPolicy: inheritedPolicy,
        projectMemory: effectiveThreadMemoryPolicy(childThread).useExisting ? renderProjectMemoryBucket(input.project ?? store.projects.find((p) => p.id === input.projectId)) : '',
        toolContext: fullToolContext({ thread: childThread, provider, model, signal: input.signal, subAgentId: agent.id, memoryOwner: inheritedOwner }),
      })
    } catch (e: any) {
      const message = String(e?.message ?? e)
      const settled = input.signal.aborted
        ? cancelSubAgent(store.subAgents, agent.id, 'Sub-agent cancelled before completion.')
        : failSubAgent(store.subAgents, agent.id, message)
      store.subAgents = settled.agents
      persistSubAgents()
      deps.observe?.('subagent.end', { agentId: agent.id, ok: false })
      return { agent: settled.agent ?? agent, output: input.signal.aborted ? 'Sub-agent cancelled before completion.' : `Sub-agent failed: ${message}` }
    }

    let latest = store.subAgents.find((candidate) => candidate.id === agent.id) ?? agent
    for (const summary of summarizeSubAgentTimeline(childThread.timeline)) {
      const next = appendSubAgentEvent(store.subAgents, agent.id, summary.kind, summary.text)
      store.subAgents = next.agents
      latest = next.agent ?? latest
    }

    const runError = readOnlyChildRunError(childThread.timeline, input.signal.aborted)
    if (runError) {
      const settled = input.signal.aborted
        ? cancelSubAgent(store.subAgents, agent.id, runError)
        : failSubAgent(store.subAgents, agent.id, runError)
      store.subAgents = settled.agents
      persistSubAgents()
      deps.observe?.('subagent.end', { agentId: agent.id, ok: false })
      return { agent: settled.agent ?? latest, output: input.signal.aborted ? runError : `Sub-agent failed: ${runError}` }
    }

    const result = finalSubAgentText(childThread.timeline) || '(sub-agent completed with no final message)'
    const completed = completeSubAgent(store.subAgents, agent.id, result)
    store.subAgents = completed.agents
    persistSubAgents()
    deps.observe?.('subagent.end', { agentId: agent.id, ok: true })
    return { agent: completed.agent ?? latest, output: `Sub-agent result: ${result.slice(0, 8000)}` }
  }

  async function launchThreadSubAgent(thread: Thread, provider: Provider, model: string, input: { task: string; name?: string | null; model?: string | null; providerId?: string | null; accountId?: string | null; agent?: string | null; background?: boolean }, signal: AbortSignal, parentPolicy?: import('../../agent/types').ThreadMemoryPolicy, parentOwner?: { id: string; revision: number }, capturedOwner?: Pick<Thread, 'bot' | 'phoneLocalOnly'>) {
    const assertOwner = () => {
      assertThreadChildSupported(capturedOwner)
      assertThreadChildSupported(thread)
      const actual = store.threads.find((candidate) => candidate.id === thread.id)
      if (!actual || actual.archivedAt) throw new Error('The owning chat is unavailable; refusing to create a permissive child.')
      assertThreadChildSupported(actual)
      if (parentOwner) {
        const owner = store.threads.find((candidate) => candidate.id === parentOwner.id)
        if (!owner || owner.archivedAt) throw new Error('The owning chat is unavailable; refusing to create a permissive child.')
        assertThreadChildSupported(owner)
      }
    }
    assertOwner()
    if (signal.aborted || thread.archivedAt || parentOwner && threadMemoryPolicyRevision(thread) !== parentOwner.revision) throw new Error('This parent run can no longer launch a child.')
    if (thread.incognito) throw new Error('Incognito cannot launch a persistent sub-agent; use an ordinary saved task.')
    const preset = normalizeSubAgentPreset(input.agent)
    const task = applySubAgentPreset(preset, String(input.task || '').trim())
    if (!String(input.task || '').trim()) throw new Error('Sub-agent task is required.')
    const providerOverride = typeof input.providerId === 'string' && input.providerId.trim() ? input.providerId.trim() : null
    const accountOverride = typeof input.accountId === 'string' && input.accountId.trim() ? input.accountId.trim() : null
    const project = store.projects.find((p) => p.id === thread.projectId)
    let childProvider = provider
    let childAccountId = thread.providerAccountId ?? null
    let childModel = input.model && provider.models.includes(input.model) ? input.model : model
    let routeNotice: string | null = preset ? `${preset.label} preset active.` : null
    // Specialist presets (agent typing) may prefer a role-appropriate model when
    // the caller did not pin one. Advisory only, pinned to the parent provider.
    if (preset && !input.model && !providerOverride && !accountOverride) {
      try {
        const roleRoute = modelRouter.recommendRoleRoute({
          providers: store.settings.providers,
          role: preset.role,
          requestedProviderId: provider.id,
          localOnly: project?.localOnly === true,
        })
        const best = roleRoute.rankings.find((ranking) => ranking.providerId === provider.id && provider.models.includes(ranking.model))
        if (best) {
          childModel = best.model
          routeNotice = `${preset.label} preset active · routed to ${best.model}.`
        }
      } catch { /* presets never block the launch */ }
    }
    // Optional cross-model route: explicit providerId/accountId reroute the child; the
    // model is then resolved against the routed provider (absent fields inherit).
    // The localOnly project guard is enforced on the ROUTED provider, not just the parent's.
    if (providerOverride || accountOverride) {
      const routed = resolveDelegationRoute(store.settings.providers, {
        providerId: provider.id,
        providerAccountId: thread.providerAccountId,
        model: childModel,
      }, { providerId: providerOverride, accountId: accountOverride, model: input.model }, undefined, project?.localOnly === true)
      if (!routed.route) throw new Error(`Cannot launch sub-agent: ${routed.error}.`)
      const baseProvider = store.settings.providers.find((candidate) => candidate.id === routed.route.providerId)
      if (!baseProvider) throw new Error(`Cannot launch sub-agent: provider "${routed.route.providerId}" is not configured.`)
      childProvider = runtimeProvider(baseProvider, routed.route.providerAccountId)
      childAccountId = routed.route.providerAccountId ?? null
      childModel = routed.route.model
      routeNotice = routed.notice ?? null
    }
    assertOwner()
    const inheritedMemoryPolicy = captureThreadMemoryPolicy(thread, parentPolicy)
    assertThreadMemoryTransport(thread, childProvider, inheritedMemoryPolicy)
    const runId = `subagent-${randomUUID()}`
    const created = createSubAgentFromLaunch(store.subAgents, {
      source: 'thread_tool',
      parent: parentForThread(thread),
      task,
      name: input.name ?? preset?.label ?? undefined,
      providerId: childProvider.id,
      providerAccountId: childAccountId,
      model: childModel,
      runId,
      cwd: thread.cwd,
    })
    store.subAgents = created.agents
    persistSubAgents()
    const started = startSubAgent(store.subAgents, created.agent.id, runId)
    store.subAgents = started.agents
    persistSubAgents()

    const backgroundController = input.background ? new AbortController() : null
    const childSignal = backgroundController?.signal ?? signal
    const abortBackground = () => backgroundController?.abort()
    if (backgroundController) {
      if (signal.aborted) backgroundController.abort()
      else signal.addEventListener('abort', abortBackground, { once: true })
    }
    const work = runReadOnlySubAgent(created.agent, childProvider, childModel, {
      memoryPolicy: inheritedMemoryPolicy,
      memoryOwner: parentOwner ?? { id: thread.id, revision: threadMemoryPolicyRevision(thread) },
      incognito: thread.incognito,
      task,
      parentLabel: `parent thread "${thread.title || 'Parent thread'}"`,
      cwd: thread.cwd,
      projectId: thread.projectId ?? null,
      providerAccountId: childAccountId,
      signal: childSignal,
      project,
      workspace: thread.workspace ?? null,
    })
    if (input.background) {
      void work.then((launched) => {
        const parent = store.threads.find((candidate) => candidate.id === thread.id)
        const item = parent?.timeline.find((candidate) => candidate.subAgentId === launched.agent.id)
        if (parent && item) {
          const event: AgentEvent = {
            runId: launched.agent.runId ?? `background-${launched.agent.id}`,
            threadId: parent.id,
            type: 'item_update',
            id: item.id,
            patch: {
              subAgent: launched.agent,
              status: launched.agent.status,
              state: launched.agent.status === 'succeeded' ? 'ok' : 'error',
              output: launched.output,
              meta: `background sub-agent ${launched.agent.status}`,
            },
          }
          deps.applyEventToThread(parent, event)
          deps.emitToRenderer(event)
          deps.emitPluginEvent(event)
          deps.persistThreads()
        }
      }).catch(() => {}).finally(() => signal.removeEventListener('abort', abortBackground))
      const startedAgent = started.agent ?? created.agent
      return {
        agent: startedAgent,
        output: `${routeNotice ? `${routeNotice}\n` : ''}Background sub-agent launched. id=${startedAgent.id}. Continue working and call subagent_status later for its result.`,
      }
    }
    const launched = await work
    if (routeNotice) launched.output = `${routeNotice}\n${launched.output}`
    return launched
  }

  function fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal; subAgentId?: string; memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy; memoryOwner?: { id: string; revision: number } }) {
    const childOwner = run?.thread
    const operationalSubAgentId = run?.subAgentId
    const capturedChildOwner = Object.freeze({ bot: childOwner?.bot, phoneLocalOnly: childOwner?.phoneLocalOnly, googleLocalOnly: childOwner?.googleLocalOnly })
    const memoryPolicy = captureThreadMemoryPolicy(run?.thread ?? {}, run?.memoryPolicy)
    const memoryOwner = run?.memoryOwner ?? (run?.thread && store.threads.includes(run.thread) ? { id: run.thread.id, revision: threadMemoryPolicyRevision(run.thread) } : null)
    const recallThreads = () => store.threads.filter((thread) => canRecallThread(thread, childOwner?.id) && (memoryPolicy.useExisting || thread.id === childOwner?.id))
    const recallMissions = () => {
      const ids = new Set(recallThreads().map((thread) => thread.id))
      return store.missions.filter((mission) => !mission.threadId || ids.has(mission.threadId))
    }
    const subAgentSourceIds = (agent: SubAgent) => [...new Set([agent.threadId, agent.parent?.threadId].filter((id): id is string => !!id))]
    const subAgentVisible = (agent: SubAgent) => {
      const ids = subAgentSourceIds(agent)
      const ownWorker = !!operationalSubAgentId && agent.id === operationalSubAgentId
      if (!memoryPolicy.useExisting && !ownWorker && !ids.includes(childOwner?.id ?? '')) return false
      return ids.every((id) => {
        const source = store.threads.find((thread) => thread.id === id)
        return !!source && canRecallThread(source, childOwner?.id)
      })
    }
    const documentSourceIds = childOwner ? validateDocumentSourceIds(childOwner.documentSourceIds) : []
    const documentPolicy = documentPolicySignature(documentSourceIds)
    const sourceConfiguration = () => JSON.stringify([store.settings.documents.maxDocumentBytes, store.settings.documents.maxDocuments, store.settings.documents.sources.filter(source => documentSourceIds === undefined || documentSourceIds.includes(source.id)).map(source => [source.id, source.path, source.kind, source.enabled])])
    const capturedSourceConfiguration = sourceConfiguration()
    const documentOwnerStored = !!childOwner && store.threads.some(thread => thread.id === childOwner.id)
    const assertDocumentPolicyCurrent = () => {
      const currentOwner = documentOwnerStored ? store.threads.find(thread => thread.id === childOwner!.id) : childOwner
      if (sourceConfiguration() !== capturedSourceConfiguration || run?.signal?.aborted || documentOwnerStored && !currentOwner || documentPolicySignature(currentOwner?.documentSourceIds) !== documentPolicy) throw new Error('Document collections changed during this run; start a new turn with the selected sources.')
    }
    const recalledSourceIds = new Set<string>()
    const assertRecallSourcesAllowed = () => {
      assertDocumentPolicyCurrent()
      for (const id of recalledSourceIds) {
        const source = store.threads.find((thread) => thread.id === id)
        if (!source || !canRecallThread(source, childOwner?.id)) throw new Error('A recalled conversation is no longer available to this run; start a new conversation without that context.')
      }
    }
    const project = run?.thread?.projectId ? store.projects.find((p) => p.id === run.thread?.projectId) : null
    const cloudModelContext = !!run?.provider && run.provider.kind !== 'local'
    const capturedMemoryOwner = memoryOwner ? store.threads.find((thread) => thread.id === memoryOwner.id) : null
    const recallProjectMemory = () => {
      if (!project || !store.projects.includes(project) || childOwner?.projectId !== project.id || cloudModelContext || childOwner?.bot?.executionTarget === 'vm' || run?.signal?.aborted) return []
      if (memoryOwner && (!capturedMemoryOwner || !store.threads.includes(capturedMemoryOwner) || threadMemoryPolicyRevision(capturedMemoryOwner) !== memoryOwner.revision || !effectiveThreadMemoryPolicy(capturedMemoryOwner, memoryPolicy).useExisting)) return []
      if (!effectiveThreadMemoryPolicy(childOwner ?? {}, memoryPolicy).useExisting) return []
      return scopeProjectMemoryEntries(store.projectMemory, project.id)
    }
    const workspace = run?.thread?.workspace ?? project?.workspace ?? null
    const backend = sshManager.backendForWorkspace(workspace)
    if (workspace?.kind === 'ssh' && backend?.kind !== 'ssh') {
      throw new Error('The SSH workspace disconnected before tool execution; refusing to fall back to the local filesystem.')
    }
    return {
      backend,
      memoryPolicy,
      workspace: workspace ?? (run?.thread ? { kind: 'local' as const, path: run.thread.cwd } : null),
      disabledCapabilityIds: store.settings.capabilities?.disabledCapabilityIds ?? [],
      localScripts: store.settings.capabilities?.localScripts ?? [],
      packagedPlugins: store.settings.capabilities?.packagedPlugins ?? [],
      allowSandboxNetwork: store.settings.allowTaskNetwork ?? false,
      writeManagedOutput: (ref: string, content: string) => {
        if (run?.thread?.incognito) throw new Error('Incognito does not save automatic tool-output files; this output remains bounded in the current conversation.')
        return managedOutputs.write(ref, content)
      },
      readManagedOutput: managedOutputs.read,
      emailSettings: store.settings.email,
      slackSettings: store.settings.slack,
      calendarSettings: store.settings.calendar,
      documentSettings: scopedDocumentSettings(store.settings.documents, documentSourceIds),
      documentLibrary: run?.provider && !isExternalProvider(run.provider) && !run.thread?.bot && !run.thread?.orchestratorHostId ? scopedDocumentLibrary(documentLibrary(), documentSourceIds, assertDocumentPolicyCurrent) : undefined,
      browserSettings: store.settings.browser,
      threads: recallThreads(),
      recallThreads,
      recordRecallSources: (ids: string[]) => { for (const id of ids) recalledSourceIds.add(id) },
      assertRecallSourcesAllowed,
      listSubAgents: () => store.subAgents.filter(subAgentVisible),
      currentSubAgentId: run?.subAgentId,
      sendSubAgentMessage: (input: import('../../agent/tools').SubAgentMailboxSendInput) => {
        if (run?.thread?.incognito) throw new Error('Incognito cannot write a persistent sub-agent mailbox.')
        const target = store.subAgents.find((agent) => agent.id === input.agentId)
        if (!target) throw new Error(`Sub-agent "${input.agentId}" was not found.`)
        if (!subAgentVisible(target)) throw new Error('Private sub-agent history is unavailable to this conversation.')
        if (store.threads.some((thread) => thread.id === target.threadId && thread.incognito)) throw new Error('Private sub-agent history is unavailable.')
        if (run?.subAgentId) {
          if (target.id !== run.subAgentId) throw new Error('A child sub-agent may only write to its own mailbox.')
        } else if (!run?.thread || target.threadId !== run.thread.id) {
          throw new Error('The target sub-agent does not belong to this thread.')
        }
        const appended = appendSubAgentMessage(store.subAgents, target.id, {
          sender: run?.subAgentId ? 'child' : 'parent',
          kind: input.kind,
          text: input.text,
          replyToId: input.replyToId,
        })
        if (!appended.message) throw new Error('Mailbox messages cannot be empty.')
        store.subAgents = appended.agents
        persistSubAgents()
        return appended.message
      },
      readSubAgentMessages: (input: import('../../agent/tools').SubAgentMailboxReadInput) => {
        const target = store.subAgents.find((agent) => agent.id === input.agentId)
        if (!target) throw new Error(`Sub-agent "${input.agentId}" was not found.`)
        if (!subAgentVisible(target)) throw new Error('Private sub-agent history is unavailable to this conversation.')
        for (const id of subAgentSourceIds(target)) recalledSourceIds.add(id)
        assertRecallSourcesAllowed()
        if (store.threads.some((thread) => thread.id === target.threadId && thread.incognito)) throw new Error('Private sub-agent history is unavailable.')
        if (run?.subAgentId) {
          if (target.id !== run.subAgentId) throw new Error('A child sub-agent may only read its own mailbox.')
        } else if (!run?.thread || target.threadId !== run.thread.id) {
          throw new Error('The target sub-agent does not belong to this thread.')
        }
        const read = readSubAgentMessages(store.subAgents, target.id, {
          audience: run?.subAgentId ? 'child' : 'parent',
          unreadOnly: input.unreadOnly,
          markRead: input.markRead,
        })
        if (input.markRead !== false && read.messages.length) {
          store.subAgents = read.agents
          persistSubAgents()
        }
        return read.messages
      },
      searchSessions: (options: import('../../agent/session-recall').SessionSearchOptions) => {
        assertSavedMemoryRead(run?.thread ?? {}, memoryPolicy)
        return searchSessionIndex(vontraPath('state', 'session-search.sqlite'), recallThreads(), { ...options, currentThreadId: childOwner?.id }, blindSearchToken)
      },
      connectorEvents: store.connectorEvents.filter(event => documentEventAllowed(event, store.settings.documents, documentSourceIds)).filter((event) => event.connectorId !== 'whatsapp' || !event.threadKey.startsWith('whatsapp:') || isWhatsAppEventInCurrentScope(event, store.settings.whatsapp.sources)),
      missions: recallMissions(),
      recallMissions,
      rooms: memoryPolicy.useExisting ? store.rooms : store.rooms.map(roomWithoutSavedRecall),
      tickets: store.tickets,
      projectMemory: recallProjectMemory(),
      recallProjectMemory,
      sensitiveRetrievalAllowed: !cloudModelContext,
      sensitiveRetrievalReason: cloudModelContext
        ? `cloud model route via ${run?.provider?.name ?? 'cloud provider'}; use a local provider to include connector, chat, document, and memory retrieval sources`
        : project?.localOnly ? 'local-only project route' : 'local model route',
      // Scope rule (VST-010): global skills everywhere; a project's TRUSTED repo skills
      // only inside that project. skillsForContext drops any project-scoped stray.
      skills: skillsForContext([...store.skills, ...(deps.projectSkills?.(run?.thread?.projectId, workspace) ?? [])], run?.thread?.projectId),
      commands: store.commands,
      skillSuggestions: store.suggestions,
      // First-class artifact registration (VST-011): persisted with run provenance; the
      // run receipt counts save_artifact tool items, so registration IS receipt-visible.
      registerArtifact: (input: CreateArtifactInput) => {
        if (run?.thread?.incognito) throw new Error('Incognito does not retain artifact content or metadata in TensorFold. Save an explicitly requested output to the workspace instead.')
        const record = createArtifactRecord({
          ...input,
          provenance: {
            ...input.provenance,
            workspace: input.provenance.workspace ?? (workspace ?? (run?.thread ? { kind: 'local' as const, path: run.thread.cwd } : null)),
            turn: input.provenance.turn ?? (run?.thread ? threadTurnNumber(run.thread.timeline ?? []) : null),
          },
        })
        // Bounded store: artifacts are metadata + small inline bodies, never an archive.
        store.artifacts = [record, ...store.artifacts].slice(0, 500)
        persistArtifacts()
        return record
      },
      // The agent proposes a Skill/Command; it's filed (deduped) and pushed to the
      // renderer for review. Nothing is created until the user accepts.
      fileSuggestion: (input: SuggestionInput) => {
        if (run?.thread?.incognito) throw new Error('Incognito does not save learned skill or command suggestions.')
        assertMemoryContribution(run?.thread ?? {}, memoryPolicy)
        if (memoryOwner) {
          const owner = store.threads.find((thread) => thread.id === memoryOwner.id)
          if (!owner || !canContributeThreadMemory(owner, memoryPolicy, memoryOwner.revision, run?.signal)) throw new Error('The owning chat no longer permits memory proposals.')
        }
        if (run?.signal?.aborted || run?.thread?.archivedAt) throw new Error('This run can no longer propose memories.')
        const { added, list } = addSuggestion(store.suggestions, input, { skills: store.skills, commands: store.commands })
        if (added) {
          store.suggestions = list
          persistSuggestions()
          store.win?.webContents.send('suggestions:update', store.suggestions)
        }
      },
      recordEmailDraft,
      recordEmailEvent,
      recordCalendarEvent,
      recordMessageEvent,
      runBrowserAutomation,
      runGoogleWorkspaceTool: createGoogleWorkspaceToolRunner(run, {
        getThread: (id) => store.threads.find((thread) => thread.id === id),
        resolveProvider: (thread) => { const provider = store.settings.providers.find((candidate) => candidate.id === thread.providerId); return provider ? runtimeProvider(provider, thread.providerAccountId ?? null) : undefined },
        service: googleWorkspaceService,
        disabledCapabilities: () => store.settings.capabilities?.disabledCapabilityIds ?? [],
        disabledTools: () => [],
        assertRecallSourcesAllowed,
        hasExternalResearchEnabled: () => engineService()?.hasExternalResearchEnabled() === true,
        markContext: markGoogleContext,
      }),
      runControllerAction: (input: ControllerActionInput) => runControllerAction(store.settings.controller, input, 'TensorFold'),
      runComputerUse: (input: ComputerUseInput) => runComputerUse(input, { threadId: run?.thread?.id ?? null, signal: run?.signal }, deps.cancelThreadRun),
      phoneControlAvailable: !!run?.thread && !!run.provider && !isExternalProvider(run.provider) && run.thread.kind === 'chat' && !run.thread.orchestratorHostId && workspace?.kind !== 'ssh' && !run.subAgentId && phoneHost()?.service.isGranted(run.thread.id) === true,
      phoneLocalOnly: run?.thread?.phoneLocalOnly === true,
      phonePrivacyGuard: () => !run?.provider || isExternalProvider(run.provider) || engineService()?.hasExternalResearchEnabled()
        ? 'This phone conversation stays local; choose a model running on this Mac and turn Cortheon off before continuing.' : null,
      googlePrivacyGuard: () => !run?.provider || isExternalProvider(run.provider) || engineService()?.hasExternalResearchEnabled()
        ? 'This Google account conversation stays on this Mac; choose a local API model and turn external research off.' : null,
      runPhoneControl: async (input: PhoneToolInput) => {
        if (!run?.thread || !run.provider || isExternalProvider(run.provider) || run.thread.orchestratorHostId || workspace?.kind === 'ssh' || run.subAgentId) return { ok: false, title: 'Phone control', output: 'Phone control requires its explicitly allowed local chat; remote and external model routes cannot access the phone.' }
        const service = phoneHost()?.service
        if (!service) return { ok: false, title: 'Phone control', output: 'Phone control is unavailable in this process.' }
        return service.tool(input, run.thread.id, run.signal)
      },
      ...(run?.thread && run.provider && run.model && run.signal
        ? { launchSubAgent: (input: { task: string; name?: string | null; model?: string | null; providerId?: string | null; accountId?: string | null; background?: boolean }) => launchThreadSubAgent(childOwner!, run.provider!, run.model!, input, run.signal!, memoryPolicy, memoryOwner ?? undefined, capturedChildOwner) }
        : {}),
      ...mcpToolContext(),
    }
  }

  return { fullToolContext, runReadOnlySubAgent }
}
