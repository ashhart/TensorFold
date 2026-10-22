import { documentPolicySignature } from '../../../shared/document-collections'
import { selectedResponseLocale } from '../../agent/response-language'
import { captureThreadMemoryPolicy, assertThreadMemoryTransport, threadMemoryPolicyRevision } from '../../../shared/thread-memory-policy'
import { assertThreadChildSupported } from '../../../shared/thread-child-policy'
import { randomUUID } from 'node:crypto'
import { runBestOfN, type BestOfNResult, type VerifierVerdict } from '../../agent/best-of-n'
import { renderTranscript } from '../../agent/compaction'
import { runAgent } from '../../agent/loop'
import { addMissionCheckpoint, appendMissionJournal, completeMission, failMission, pauseMission, recordMissionHeartbeat, requestMissionApproval } from '../../agent/missions'
import { buildFrontendMissionEvidence, buildMissionRunPrompt, decideNextMissionStep, missionNeedsFrontendUiSmoke, missionRunCount, missionRunLimit, recordMissionCheckpoint, recordMissionStarted, shouldStartFreshMissionThread } from '../../agent/mission-supervisor'
import { streamChat } from '../../agent/openai'
import { resolveProviderModel } from '../../agent/provider-accounts'
import { ensureTensorFoldStudioStylePrompt } from '../../agent/prompt-style'
import { interruptedMissionSettlement } from '../../agent/run-settlement'
import { assertProviderEndpoint, auditPreview } from '../../agent/security'
import type { JobPatch } from '../../agent/jobs'
import type { AgentEvent, Job, Mission, MissionBudgetSnapshot, Project, Provider, Thread, TimelineItem } from '../../agent/types'
import { store } from '../state'
import { auditEvent } from './audit'
import { modelRouteAccountLabel, modelRouteAutoFailoverEnabled, modelRouteResetPolicy, modelRouter, rememberModelRouteHistory, runtimeProviderWithFreshModels, selectedProviderAccountId } from './provider-runtime'
import { resolveWorkspaceCwd } from './workspace'
import { cleanupMissionAttemptWorktrees, createMissionAttemptWorktrees, integrateMissionAttemptWinner, missionFanOutCount, type MissionAttemptBatch } from './mission-worktree-fanout'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, type WorkspaceMutationGuard } from './workspace-mutation-guard'

interface JobHandle {
  get(): Job | null
  set(job: Job | null): void
}

export interface MissionRuntime {
  runMission(mission: Mission): Promise<void>
}

interface MissionRuntimeDeps {
  resolveWorkspace?: typeof resolveWorkspaceCwd
  runningMissions: Set<string>
  missionAborters: Map<string, AbortController>
  missionApprovalResolvers: Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>
  claimJob(ref: Parameters<typeof import('../../agent/jobs').enqueueJob>[1]): Job
  patchJob(job: Job | null, patch: JobPatch): Job | null
  heartbeatJob(job: Job | null): Job | null
  finishJob(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  getMission(id: string): Mission
  replaceMission(next: Mission): Mission
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal; memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy; memoryOwner?: { id: string; revision: number } }): any
  renderProjectMemoryBucket(project: { id: string; name: string } | undefined): string
  applyEventToThread(thread: Thread, event: AgentEvent): void
  timelineErrorSince(thread: Thread, timelineStart: number): TimelineItem | null
  timelineError(errorItem: TimelineItem): Error
  isHighRiskTool(name: string): boolean
  pendingApprovalMeta: Map<string, { threadId: string; itemId: string; tool: string; policy: any }>
  persistJobApproval(jobHandle: JobHandle | undefined, callId: string, meta: { threadId: string; itemId: string; tool: string; policy: any }): void
  resolvePersistedJobApproval(callId: string, decision: 'allow' | 'allow_always' | 'deny', jobHandle?: JobHandle): Job | null
  markRouteErrorRecovered(thread: Thread, err: unknown, runId: string, text: string): void
  injectThreadRouteHandoff(thread: Thread, reason: string): string
  effectiveContextLimit(provider: Provider, model: string | null, accountId?: string | null): number
  workspaceMutationGuard?: Pick<WorkspaceMutationGuard, 'reserveStart'>
}

export function registerMissionApprovalResolver(
  callId: string,
  controller: AbortController,
  missionApprovalResolvers: Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>,
  pendingApprovalMeta: { delete(key: string): boolean },
  onResolve: (decision: 'allow' | 'allow_always' | 'deny') => void,
): (decision: 'allow' | 'allow_always' | 'deny') => void {
  let settled = false
  function done(d: 'allow' | 'allow_always' | 'deny') {
    if (settled) return
    settled = true
    controller.signal.removeEventListener('abort', onAbort)
    missionApprovalResolvers.delete(callId)
    pendingApprovalMeta.delete(callId)
    onResolve(d)
  }
  function onAbort() { done('deny') }
  missionApprovalResolvers.set(callId, done)
  // A queued fan-out request can start executing only after an earlier approval settles
  // — potentially via this very abort. addEventListener never fires on an
  // already-aborted signal, so without this check the promise would never settle and
  // the mission's runAgent would hang forever with the mission unrecoverable.
  if (controller.signal.aborted) { done('deny'); return done }
  controller.signal.addEventListener('abort', onAbort, { once: true })
  return done
}

type MissionWorkspaceState = {
  projects: Array<Pick<Project, 'id' | 'workspace' | 'archivedAt'>>
  threads: Array<Pick<Thread, 'id' | 'projectId' | 'workspace' | 'archivedAt' | 'bot' | 'phoneLocalOnly' | 'googleLocalOnly'>>
}

export interface MissionArchiveTarget {
  projectId?: string | null
  threadId?: string | null
}

type MissionArchiveState = {
  missions: Array<Pick<Mission, 'id' | 'projectId' | 'threadId'>>
  threads: Array<Pick<Thread, 'id' | 'projectId'>>
}

/** Pure ownership projection used before aborting archive-owned mission controllers. */
export function missionIdsForArchive(state: MissionArchiveState, target: MissionArchiveTarget): string[] {
  const projectId = target.projectId ?? null
  const threadId = target.threadId ?? null
  if (!projectId && !threadId) return []
  const threadProjects = new Map(state.threads.map((thread) => [thread.id, thread.projectId ?? null]))
  const matches = new Set<string>()
  for (const mission of state.missions) {
    if (threadId && mission.threadId === threadId) matches.add(mission.id)
    if (projectId && (
      mission.projectId === projectId ||
      (!!mission.threadId && threadProjects.get(mission.threadId) === projectId)
    )) matches.add(mission.id)
  }
  return [...matches]
}

/** Abort only currently-active controllers; runMission's settlement path pauses them durably. */
export function abortMissionsForArchive(
  state: MissionArchiveState,
  missionAborters: ReadonlyMap<string, AbortController>,
  target: MissionArchiveTarget,
): string[] {
  const aborted: string[] = []
  for (const id of missionIdsForArchive(state, target)) {
    const controller = missionAborters.get(id)
    if (!controller || controller.signal.aborted) continue
    controller.abort()
    aborted.push(id)
  }
  return aborted
}

/**
 * Missions currently execute their cwd, delegated providers, and lifecycle bookkeeping in this
 * main process. Although workspace tools can use an SSH backend, the rest of the mission runtime
 * cannot, so accepting an SSH-owned mission can run local processes against a same-named path.
 * Reject before claiming a job or changing any mission/run state.
 */
export function assertMissionWorkspaceSupported(
  mission: Pick<Mission, 'projectId' | 'threadId'>,
  state: MissionWorkspaceState = store,
): void {
  const thread = mission.threadId ? state.threads.find((candidate) => candidate.id === mission.threadId) ?? null : null
  if (mission.threadId && !thread) throw new Error('Mission source thread not found.')
  assertThreadChildSupported(thread)
  if (thread?.archivedAt) throw new Error('Restore the archived mission thread before running this mission.')
  if (mission.projectId != null && thread && mission.projectId !== (thread.projectId ?? null)) {
    throw new Error('Mission project does not match its source thread project.')
  }
  const projectId = mission.projectId ?? thread?.projectId ?? null
  const project = projectId ? state.projects.find((candidate) => candidate.id === projectId) ?? null : null
  if (projectId && !project) throw new Error('Mission project not found.')
  if (project?.archivedAt) throw new Error('Restore the archived project before running this mission.')
  if (thread?.workspace?.kind === 'ssh' || project?.workspace?.kind === 'ssh') {
    throw new Error('Local mission execution does not support SSH workspaces; run this mission on its remote Orchestrator.')
  }
}

export function createMissionRuntime(deps: MissionRuntimeDeps): MissionRuntime {
  const {
    runningMissions,
    missionAborters,
    missionApprovalResolvers,
    claimJob,
    patchJob,
    heartbeatJob,
    finishJob,
    getMission,
    replaceMission,
    fullToolContext,
    renderProjectMemoryBucket,
    applyEventToThread,
    timelineErrorSince,
    timelineError,
    isHighRiskTool,
    pendingApprovalMeta,
    persistJobApproval,
    resolvePersistedJobApproval,
    markRouteErrorRecovered,
    injectThreadRouteHandoff,
    effectiveContextLimit,
    workspaceMutationGuard,
  } = deps

function missionRunCaps(budget: MissionBudgetSnapshot): { maxSteps: number; maxWallMs: number } {
  const maxWallMs = budget.maxWallSeconds > 0 ? budget.maxWallSeconds * 1000 : 0 // 0 = no wall deadline (heartbeat-bounded)
  const maxSteps = budget.maxWallSeconds === 0 ? 100 : budget.maxWallSeconds >= 3600 ? 60 : 40
  return { maxSteps, maxWallMs }
}

function missionCheckpointNotes(mission: Mission): string[] {
  return mission.checkpoints
    .filter((checkpoint) => typeof checkpoint.data?.runIndex === 'number')
    .map((checkpoint) => checkpoint.summary)
    .filter(Boolean)
}

function missionPassSummary(thread: Thread, timelineStart = 0): string {
  const slice = thread.timeline.slice(timelineStart)
  const assistant = [...slice].reverse().find((item) => item.kind === 'assistant' && item.text?.trim())
  if (assistant?.text?.trim()) return assistant.text.trim().replace(/\s+/g, ' ').slice(0, 700)
  const error = [...slice].reverse().find((item) => item.kind === 'error' && item.text?.trim())
  if (error?.text?.trim()) return `Error: ${error.text.trim().replace(/\s+/g, ' ').slice(0, 650)}`
  return 'Mission pass completed without a final assistant summary.'
}

function missionErrorSince(thread: Thread, timelineStart: number): string | null {
  const item = thread.timeline.slice(timelineStart).find((candidate) => candidate.kind === 'error' && candidate.text?.trim())
  return item?.text?.trim() ?? null
}

function missionEvidenceItems(thread: Thread) {
  return thread.timeline.filter((item) => {
    if (item.kind === 'tool') return item.state === 'ok'
    if (item.kind === 'visualization') return item.state === 'ok' || !!item.visualization
    return false
  })
}

function missionEvidenceSummary(thread: Thread): string {
  const items = missionEvidenceItems(thread)
  if (!items.length) return 'No successful tool evidence recorded.'
  return items.slice(-12).map((item) => {
    const label = item.title || item.tool || item.kind
    const target = item.path || item.command || item.meta || ''
    const output = item.output ? item.output.replace(/\s+/g, ' ').slice(0, 260) : ''
    return `- ${label}${target ? ` (${target})` : ''}${output ? `: ${output}` : ''}`
  }).join('\n')
}

const MISSION_ATTEMPT_STRATEGIES = [
  'Use the most direct implementation that satisfies the stop condition.',
  'Prefer the smallest low-risk change and actively preserve existing behavior.',
  'Work test-first and probe adversarial edge cases before declaring success.',
  'Try an independent alternative design, then validate it against the stated constraints.',
]

function cloneMissionAttemptThread(source: Thread, cwd: string, attemptId: string, strategy: string): Thread {
  const clone: Thread = {
    ...source,
    documentSourceIds: source.documentSourceIds ? [...source.documentSourceIds] : [],
    id: randomUUID(),
    title: `${source.title} · ${attemptId}`,
    cwd,
    workspace: { kind: 'local', path: cwd },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: 'running',
    apiMessages: structuredClone(source.apiMessages),
    timeline: structuredClone(source.timeline),
  }
  let lastUser: Thread['apiMessages'][number] | undefined
  for (let index = clone.apiMessages.length - 1; index >= 0; index--) {
    if (clone.apiMessages[index].role === 'user') { lastUser = clone.apiMessages[index]; break }
  }
  if (!lastUser) clone.apiMessages.push({ role: 'user', content: `Attempt strategy: ${strategy}` })
  else if (typeof lastUser.content === 'string') lastUser.content = `${lastUser.content}\n\nATTEMPT STRATEGY: ${strategy}`
  else if (Array.isArray(lastUser.content)) lastUser.content.push({ type: 'text', text: `\n\nATTEMPT STRATEGY: ${strategy}` })
  return clone
}

function missionCommandTestSignal(thread: Thread, timelineStart: number): boolean | undefined {
  const verificationCommand = /(?:^|\s)(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|verify|typecheck|build)\b|(?:^|\s)(?:pytest|cargo\s+test|go\s+test|dotnet\s+test)\b/i
  const commands = thread.timeline.slice(timelineStart).filter((item) => item.kind === 'tool' && item.tool === 'run_command' && verificationCommand.test(item.command ?? ''))
  if (!commands.length) return undefined
  if (commands.some((item) => item.state !== 'ok')) return false
  return true
}

interface MissionFanOutPassResult {
  thread: Thread
  timelineStart: number
  verifier: Awaited<ReturnType<typeof verifyMissionStopCondition>>
  ranking: Array<{ id: string; score: number; ok: boolean; testsPassed: boolean | null; rationale: string | null }>
  integration: Awaited<ReturnType<typeof integrateMissionAttemptWinner>>
}

async function verifyMissionStopCondition(provider: Provider, model: string, mission: Mission, thread: Thread, signal: AbortSignal, assertRequestAllowed?: () => void) {
  assertRequestAllowed?.()
  const stopCondition = mission.stopCondition.trim()
  if (!stopCondition) return { done: true, note: 'No stop condition configured.', evidenceCount: missionEvidenceItems(thread).length }
  const evidence = missionEvidenceItems(thread)
  if (!evidence.length) return { done: false, note: 'No successful tool evidence was recorded for the stop condition.', evidenceCount: 0 }
  const frontendEvidence = missionNeedsFrontendUiSmoke(mission) ? buildFrontendMissionEvidence(thread) : null
  if (frontendEvidence && !frontendEvidence.complete) return { done: false, note: frontendEvidence.summary, evidenceCount: evidence.length }
  const sys = ensureTensorFoldStudioStylePrompt([
    'You are a strict mission stop-condition verifier.',
    'Decide whether the STOP CONDITION is actually satisfied from the transcript and evidence.',
    'Do not accept an assistant claim by itself. Require concrete tool evidence, citations, files, diffs, command output, or other recorded artifacts.',
    'Reply on one line starting with either "DONE: <evidence>" or "CONTINUE: <missing evidence>".',
  ].join('\n'))
  const user = [
    `MISSION: ${mission.title}`,
    `OBJECTIVE: ${mission.objective}`,
    `STOP CONDITION: ${stopCondition}`,
    '',
    'RECORDED EVIDENCE:',
    missionEvidenceSummary(thread),
    frontendEvidence ? `\nFRONTEND UI SMOKE CHECKLIST:\n${frontendEvidence.summary}` : '',
    '',
    'RECENT TRANSCRIPT:',
    renderTranscript(thread.apiMessages.slice(-24)),
  ].join('\n')
  let out = ''
  try {
    for await (const ch of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages: [{ role: 'system', content: sys }, { role: 'user', content: user }], temperature: 0.1, reasoningEffort: 'low', approvalMode: 'readonly', cwd: thread.cwd, signal, assertRequestAllowed })) {
      if (ch.type === 'content') out += ch.text
    }
  } catch {
    assertRequestAllowed?.()
    return { done: false, note: 'Stop-condition verification failed; continuing.', evidenceCount: evidence.length }
  }
  const text = out.trim()
  const done = /^\s*done\b/i.test(text)
  const note = text.replace(/^\s*(done|continue)\s*[:\-]?\s*/i, '').trim().slice(0, 300) || (done ? 'Stop condition verified.' : 'Stop condition not verified.')
  return { done, note, evidenceCount: evidence.length }
}

async function runMission(mission: Mission): Promise<void> {
  if (runningMissions.has(mission.id)) return
  assertMissionWorkspaceSupported(mission)
  // Resolve the host-owned record before accepting its source identity. A caller
  // cannot downgrade a saved guarded mission by supplying an ordinary thread ID.
  const ownedMission = getMission(mission.id)
  assertMissionWorkspaceSupported(ownedMission)
  mission = ownedMission
  const sourceThreadId = mission.threadId ?? null
  const sourceProjectId = mission.projectId ?? null
  const sourceDocumentPolicy = documentPolicySignature(store.threads.find(thread => thread.id === sourceThreadId)?.documentSourceIds)
  const assertRequestAllowed = () => {
    const live = getMission(mission.id)
    if ((live.threadId ?? null) !== sourceThreadId || (live.projectId ?? null) !== sourceProjectId) throw new Error('The mission source changed; stop and restart it with its current owner.')
    assertMissionWorkspaceSupported(live)
    if (sourceThreadId && documentPolicySignature(store.threads.find(thread => thread.id === sourceThreadId)?.documentSourceIds) !== sourceDocumentPolicy) throw new Error('The mission owner document collections changed; restart with its current sources.')
  }
  const requestGuards = new Map<string, () => void>()
  assertRequestAllowed()
  runningMissions.add(mission.id)
  const controller = new AbortController()
  missionAborters.set(mission.id, controller)
  let job: Job | null = null
  let stalledOut = false
  let watchdog: ReturnType<typeof setInterval> | null = null
  try {
    job = claimJob({ kind: 'mission', owner: mission.id, missionId: mission.id, projectId: mission.projectId, threadId: mission.threadId, currentThreadId: mission.threadId, metadata: { title: mission.title } })
    const baseProvider = store.settings.providers.find((p) => p.id === (mission.providerId ?? store.settings.activeProviderId))
    if (!baseProvider) throw new Error('No provider configured for this mission.')
    const memoryParent = mission.threadId ? store.threads.find((thread) => thread.id === mission.threadId) : undefined
    if (mission.threadId && !memoryParent) throw new Error('The source chat is unavailable; create a new mission with explicit settings.')
    const memoryPolicy = captureThreadMemoryPolicy(memoryParent ?? {})
    assertThreadMemoryTransport({ ...memoryParent, memoryPolicy }, baseProvider)
    const memoryOwner = memoryParent ? { id: memoryParent.id, revision: threadMemoryPolicyRevision(memoryParent) } : undefined
    const requestedProviderAccountId = selectedProviderAccountId(baseProvider, mission.providerAccountId === undefined ? undefined : mission.providerAccountId)
    const requestedProvider = await runtimeProviderWithFreshModels(baseProvider, requestedProviderAccountId)
    let model = resolveProviderModel(requestedProvider, null, mission.model, store.settings.activeModel)
    if (!model) throw new Error('No model configured for this mission.')
    const routeBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
    const initialRoute = modelRouter.select({
      role: 'mission',
      provider: routeBaseProvider,
      model,
      requestedAccountId: requestedProviderAccountId,
      allowFailover: modelRouteAutoFailoverEnabled(routeBaseProvider, requestedProviderAccountId),
      contextLimit: effectiveContextLimit(routeBaseProvider, model, requestedProviderAccountId),
      reasoningEffort: store.settings.reasoningEffort,
      budget: mission.runBudgetClass ?? mission.budget.executionClass,
      source: 'mission_initial_route',
    })
    if (!initialRoute.selected) throw new Error(initialRoute.audit.reason)
    let providerAccountId = initialRoute.selectedAccountId
    let provider = providerAccountId === requestedProviderAccountId
      ? requestedProvider
      : await runtimeProviderWithFreshModels(routeBaseProvider, providerAccountId)
    assertProviderEndpoint(provider)
    if (initialRoute.audit.action === 'failover') {
      const resetAt = initialRoute.candidates.find((candidate) => candidate.accountId === initialRoute.requestedAccountId)?.exhaustedUntil ?? null
      job = patchJob(job, {
        metadata: {
          providerId: baseProvider.id,
          providerAccountId,
          model,
          modelRoute: {
            action: 'mission_initial_failover',
            preferredProviderAccountId: initialRoute.requestedAccountId,
            failoverProviderAccountId: initialRoute.selectedAccountId,
            resetPolicy: modelRouteResetPolicy(routeBaseProvider, initialRoute.requestedAccountId),
            failedOverAt: Date.now(),
            resetAt,
            reason: initialRoute.audit.reason,
          },
        },
      })
      try {
        replaceMission(appendMissionJournal(getMission(mission.id), {
          kind: 'status',
          actor: 'system',
          message: `Model router failed over mission from ${modelRouteAccountLabel(routeBaseProvider, initialRoute.requestedAccountId)} to ${modelRouteAccountLabel(routeBaseProvider, initialRoute.selectedAccountId)} before the next pass.`,
          status: 'running',
          data: { providerId: baseProvider.id, from: initialRoute.requestedAccountId, to: initialRoute.selectedAccountId, model, resetAt },
        }))
      } catch { /* mission gone */ }
      rememberModelRouteHistory('mission', initialRoute.audit, null, { id: mission.id, label: mission.title })
      await auditEvent('model_router.mission_failover', {
        missionId: mission.id,
        providerId: routeBaseProvider.id,
        model,
        requestedAccountId: initialRoute.requestedAccountId,
        selectedAccountId: initialRoute.selectedAccountId,
        resetAt,
        reason: auditPreview(initialRoute.audit.reason),
      })
    }
    let activeRoute = { provider, providerAccountId, model }
    const cwd = await (deps.resolveWorkspace ?? resolveWorkspaceCwd)(mission.cwd ?? store.settings.defaultCwd, 'mission workspace')
    // Hardened headless posture: read-only + deny-all by default. A mission opts into
    // mutating tools only with metadata.allowMutations === true.
    const allowMutations = mission.metadata?.allowMutations === true
    const { maxSteps, maxWallMs } = missionRunCaps(mission.budget)
    const beatMs = (mission.budget.heartbeatIntervalSeconds || 30) * 1000
    const stallMs = beatMs + (mission.budget.stallGraceSeconds || 120) * 1000
    let lastBeat = Date.now()
    let lastActivity = Date.now()
    let runIndex = missionRunCount(mission)
    let thread: Thread | null = null
    let continuationPrompt: string | null = null
    let approvalTail: Promise<unknown> = Promise.resolve()
    const requestMissionToolApproval = (callId: string, current: Mission, targetThread: Thread) => {
      const request = () => new Promise<'allow' | 'allow_always' | 'deny'>((resolve) => {
        const jobHandle: JobHandle = {
          get: () => job,
          set: (next) => { job = next },
        }
        const settleApproval = registerMissionApprovalResolver(callId, controller, missionApprovalResolvers, pendingApprovalMeta, (decision) => {
          resolvePersistedJobApproval(callId, decision, jobHandle)
          job = patchJob(job, { status: controller.signal.aborted ? 'paused' : 'running', blockedReason: null })
          resolve(decision)
        })
        try {
          const meta = pendingApprovalMeta.get(callId)
          if (meta) persistJobApproval(jobHandle, callId, meta)
          replaceMission(requestMissionApproval(getMission(current.id), {
            callId,
            title: 'Mission tool approval',
            summary: `"${targetThread.title}" requested a tool that modifies files or runs commands.`,
          }))
          job = patchJob(job, { status: 'waiting_approval', blockedReason: 'Mission tool approval pending.' })
        } catch { settleApproval('deny') }
      })
      const ordered = approvalTail.then(request, request)
      approvalTail = ordered.then(() => undefined, () => undefined)
      return ordered
    }
    // Durable journal: record the run start on disk so a crashed mission is replayable
    // from its NDJSON journal. Fire-and-forget — never blocks or breaks the run.
    recordMissionStarted(mission, runIndex)
    // Watchdog: a mission without a wall deadline (unbounded_with_heartbeat) is otherwise
    // bounded only by maxSteps — abort if the agent goes silent past the stall grace.
    watchdog = setInterval(() => {
      heartbeatJob(job)
      try {
        const cur = getMission(mission.id)
        if (cur.status === 'waiting_approval') return
      } catch { /* mission gone */ }
      if (Date.now() - lastActivity > stallMs) { stalledOut = true; controller.abort() }
    }, Math.max(5_000, Math.min(beatMs, 30_000)))

    while (!controller.signal.aborted) {
      assertRequestAllowed()
      const current = getMission(mission.id)
      if (current.status !== 'running' && current.status !== 'waiting_approval') {
        finishJob(job, 'paused', `Mission is ${current.status}.`)
        return
      }
      const nextRunIndex = runIndex + 1
      const prompt = continuationPrompt ?? buildMissionRunPrompt(current, nextRunIndex, missionCheckpointNotes(current))
      if (!thread || shouldStartFreshMissionThread(current, nextRunIndex)) {
        thread = {
          documentSourceIds: memoryParent?.documentSourceIds ? [...memoryParent.documentSourceIds] : [],
          memoryPolicy: { ...memoryPolicy },
          id: randomUUID(), title: `${current.title} · pass ${nextRunIndex}`, kind: 'task', projectId: current.projectId ?? null,
          cwd, providerId: baseProvider.id, providerAccountId: activeRoute.providerAccountId, model: activeRoute.model, createdAt: Date.now(), updatedAt: Date.now(),
          status: 'running', apiMessages: [{ role: 'user', content: prompt }], timeline: [],
        }
      } else {
        thread.apiMessages.push({ role: 'user', content: prompt })
      }
      const timelineStart = thread.timeline.length
      job = patchJob(job, {
        currentRunId: `mission:${current.id}:${nextRunIndex}`,
        currentThreadId: thread.id,
        blockedReason: null,
        metadata: {
          title: current.title,
          runIndex: nextRunIndex,
          runLimit: missionRunLimit(current),
          providerId: baseProvider.id,
          providerAccountId: activeRoute.providerAccountId,
          model: activeRoute.model,
        },
      })

      const runMissionThreadWithRoute = async (
        targetThread: Thread,
        targetTimelineStart: number,
        route: { provider: Provider; providerAccountId: string | null; model: string },
      ) => {
        assertRequestAllowed()
        targetThread.providerId = baseProvider.id
        targetThread.providerAccountId = route.providerAccountId
        targetThread.model = route.model
        const toolContext = fullToolContext({ thread: targetThread, provider: route.provider, model: route.model, signal: controller.signal, memoryPolicy, memoryOwner })
        const recallGuard = toolContext.assertRecallSourcesAllowed
        const requestGuard = () => { assertRequestAllowed(); recallGuard?.() }
        requestGuards.set(targetThread.id, requestGuard)
        await runAgent({
        responseLocale: selectedResponseLocale(store.settings.uiPreferences),
          assertRequestAllowed: requestGuard,
          thread: targetThread, provider: route.provider, model: route.model,
          approvalMode: allowMutations ? 'suggest' : 'readonly',
          temperature: store.settings.temperature, reasoningEffort: store.settings.reasoningEffort,
          autoCompact: store.settings.autoCompact, contextLimit: effectiveContextLimit(route.provider, route.model),
          memoryPolicy,
          toolContext,
          projectMemory: memoryPolicy.useExisting ? renderProjectMemoryBucket(store.projects.find((p) => p.id === current.projectId)) : '',
          maxSteps, maxWallMs,
          emit: (e) => {
            applyEventToThread(targetThread, e)
            lastActivity = Date.now()
            if (Date.now() - lastBeat > beatMs) {
              lastBeat = Date.now()
              try { const cur = getMission(current.id); if (cur.status === 'running') replaceMission(recordMissionHeartbeat(cur, 'Agent active.')) } catch { /* mission gone */ }
              heartbeatJob(job)
            }
          },
          signal: controller.signal,
          // Mutating missions route approval through the durable mission flow (so it survives
          // and the Command Centre can resolve it), bridged to the agent loop by callId.
          requestApproval: allowMutations
            ? (callId: string) => requestMissionToolApproval(callId, current, targetThread)
            : async () => 'deny',
          auditToolCall: async ({ tool, args, resultOk, output }) => {
            if (isHighRiskTool(tool)) await auditEvent('mission.tool', { missionId: current.id, tool, cwd: auditPreview(targetThread.cwd), ok: resultOk, args: auditPreview(JSON.stringify(args)), output: auditPreview(output) })
          },
          persist: () => {},
        })
        const errorItem = timelineErrorSince(targetThread, targetTimelineStart)
        if (errorItem && !controller.signal.aborted) throw timelineError(errorItem)
      }

      const runMissionPassWithRoute = async (route: { provider: Provider; providerAccountId: string | null; model: string }) => {
        const attemptCount = missionFanOutCount(current.fanOutBudget)
        if (attemptCount <= 1) {
          await runMissionThreadWithRoute(thread!, timelineStart, route)
          return null
        }
        if (!workspaceMutationGuard) throw new Error('Multi-attempt missions require the workspace mutation guard.')

        const sourceThread = thread!
        const sourceWorkspace = sourceThread.workspace ?? null
        const lease = workspaceMutationGuard.reserveStart({ workspaceKeys: [lexicalLocalWorkspaceKey(cwd)] })
        let batch: MissionAttemptBatch | null = null
        let worktreesSettled = false
        let retainWorktrees = false
        try {
          lease.addWorkspaceKeys?.([await canonicalLocalWorkspaceKey(cwd)])
          batch = await createMissionAttemptWorktrees(cwd, current.id, nextRunIndex, attemptCount)
          lease.addWorkspaceKeys?.(batch.attempts.map((attempt) => lexicalLocalWorkspaceKey(attempt.cwd)))
          lease.addWorkspaceKeys?.(await Promise.all(batch.attempts.map((attempt) => canonicalLocalWorkspaceKey(attempt.cwd))))

          job = patchJob(job, {
            metadata: {
              fanOut: {
                budget: current.fanOutBudget,
                state: 'running',
                attempts: batch.attempts.map((attempt) => ({ id: attempt.id, branch: attempt.branch, cwd: attempt.cwd })),
              },
            },
          })
          const attemptRuns = new Map<string, { thread: Thread; timelineStart: number; verifier?: Awaited<ReturnType<typeof verifyMissionStopCondition>> }>()
          const descriptors = batch.attempts.map((attempt) => ({
            id: attempt.id,
            spec: MISSION_ATTEMPT_STRATEGIES[attempt.index] ?? MISSION_ATTEMPT_STRATEGIES[0],
            meta: { branch: attempt.branch, cwd: attempt.cwd },
          }))
          const result: BestOfNResult = await runBestOfN(descriptors, {
            maxConcurrent: attemptCount,
            signal: controller.signal,
            runAttempt: async (descriptor) => {
              const attempt = batch!.attempts.find((candidate) => candidate.id === descriptor.id)
              if (!attempt) throw new Error(`Mission attempt workspace ${descriptor.id} is missing.`)
              const attemptThread = cloneMissionAttemptThread(sourceThread, attempt.cwd, attempt.id, descriptor.spec)
              const attemptTimelineStart = attemptThread.timeline.length
              attemptRuns.set(attempt.id, { thread: attemptThread, timelineStart: attemptTimelineStart })
              await runMissionThreadWithRoute(attemptThread, attemptTimelineStart, route)
              return { ok: true, output: missionPassSummary(attemptThread, attemptTimelineStart) }
            },
            verify: async (descriptor, _outcome): Promise<VerifierVerdict> => {
              const settled = attemptRuns.get(descriptor.id)
              if (!settled) return { vote: 0, rationale: 'Attempt produced no thread state.' }
              const verifier = await verifyMissionStopCondition(route.provider, route.model, current, settled.thread, controller.signal, requestGuards.get(settled.thread.id) ?? assertRequestAllowed)
              settled.verifier = verifier
              return {
                testsPassed: missionCommandTestSignal(settled.thread, settled.timelineStart),
                vote: verifier.done ? 1 : Math.min(0.9, (verifier.evidenceCount ?? 0) / 20),
                rationale: verifier.note,
              }
            },
          })

          if (controller.signal.aborted) throw new Error('Mission fan-out was interrupted.')
          const winner = result.winner
          if (!winner?.outcome.ok) {
            const errors = result.ranked.map((candidate) => candidate.outcome.error).filter(Boolean).join('; ')
            throw new Error(`All mission attempts failed${errors ? `: ${errors}` : '.'}`)
          }
          const selected = attemptRuns.get(winner.descriptor.id)
          if (!selected?.verifier) throw new Error('Mission winner was not verified.')

          const ranking = result.ranked.map((candidate) => ({
            id: candidate.descriptor.id,
            score: candidate.score,
            ok: candidate.outcome.ok,
            testsPassed: candidate.verdict.testsPassed ?? null,
            rationale: candidate.verdict.rationale ?? candidate.outcome.error ?? null,
          }))
          if (winner.verdict.testsPassed === false) {
            retainWorktrees = true
            job = patchJob(job, { metadata: { fanOut: { budget: current.fanOutBudget, state: 'verification_failed', winnerId: winner.descriptor.id, ranking, retainedRoot: batch.root } } })
            throw new Error(`Every usable mission attempt failed a verification command. Worktrees were retained at ${batch.root}.`)
          }

          const integration = await integrateMissionAttemptWinner(batch, winner.descriptor.id, allowMutations)
          worktreesSettled = !integration.retained
          retainWorktrees = integration.retained
          if (!integration.ok) {
            job = patchJob(job, { metadata: { fanOut: { budget: current.fanOutBudget, state: 'integration_blocked', winnerId: winner.descriptor.id, ranking, retainedRoot: batch.root, message: integration.message } } })
            throw new Error(integration.message)
          }

          selected.thread.cwd = cwd
          selected.thread.workspace = sourceWorkspace
          thread = selected.thread
          job = patchJob(job, {
            currentThreadId: thread.id,
            metadata: { fanOut: { budget: current.fanOutBudget, state: 'selected', winnerId: winner.descriptor.id, ranking, integration } },
          })
          try {
            replaceMission(appendMissionJournal(getMission(current.id), {
              kind: 'status', actor: 'system', status: 'running',
              message: `Selected ${winner.descriptor.id} from ${attemptCount} isolated mission attempts. ${integration.message}`,
              data: { runIndex: nextRunIndex, fanOutBudget: current.fanOutBudget, winnerId: winner.descriptor.id, ranking, integration },
            }))
          } catch { /* mission gone */ }
          return { thread, timelineStart: selected.timelineStart, verifier: selected.verifier, ranking, integration } satisfies MissionFanOutPassResult
        } finally {
          if (batch && !worktreesSettled && !retainWorktrees) await cleanupMissionAttemptWorktrees(batch, true)
          lease.release()
        }
      }
      let fanOutPass: MissionFanOutPassResult | null = null
      try {
        fanOutPass = await runMissionPassWithRoute(activeRoute)
      } catch (err) {
        const latestBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
        const routed = modelRouter.routeAfterError({
          role: 'mission',
          provider: latestBaseProvider,
          accountId: activeRoute.providerAccountId,
          model: activeRoute.model,
          error: err,
          allowFailover: modelRouteAutoFailoverEnabled(latestBaseProvider, activeRoute.providerAccountId),
          contextLimit: effectiveContextLimit(latestBaseProvider, activeRoute.model, activeRoute.providerAccountId),
          reasoningEffort: store.settings.reasoningEffort,
          budget: mission.runBudgetClass ?? mission.budget.executionClass,
          source: 'mission_pass_recovery',
        })
        if (!routed.signal || !routed.route.selected || routed.route.selectedAccountId === activeRoute.providerAccountId || controller.signal.aborted) throw err
        markRouteErrorRecovered(thread!, err, '', `Recovered provider limit on ${modelRouteAccountLabel(latestBaseProvider, activeRoute.providerAccountId)} for ${activeRoute.model}; retrying mission pass on ${modelRouteAccountLabel(latestBaseProvider, routed.route.selectedAccountId)}.`)
        providerAccountId = routed.route.selectedAccountId
        provider = await runtimeProviderWithFreshModels(latestBaseProvider, providerAccountId)
        assertProviderEndpoint(provider)
        activeRoute = { provider, providerAccountId, model: activeRoute.model }
        job = patchJob(job, {
          metadata: {
            providerId: latestBaseProvider.id,
            providerAccountId,
            model: activeRoute.model,
            modelRoute: {
              action: 'mission_failover',
              preferredProviderAccountId: routed.route.requestedAccountId,
              failoverProviderAccountId: providerAccountId,
              resetPolicy: modelRouteResetPolicy(latestBaseProvider, routed.route.requestedAccountId),
              failedOverAt: Date.now(),
              resetAt: routed.signal.resetAt,
              reason: routed.route.audit.reason,
            },
          },
        })
        try {
          replaceMission(appendMissionJournal(getMission(current.id), {
            kind: 'status',
            actor: 'system',
            message: `Model router failed over mission pass ${nextRunIndex} from ${modelRouteAccountLabel(latestBaseProvider, routed.route.requestedAccountId)} to ${modelRouteAccountLabel(latestBaseProvider, providerAccountId)} after ${routed.signal.kind.replace(/_/g, ' ')}.`,
            status: 'running',
            data: { runIndex: nextRunIndex, providerId: latestBaseProvider.id, from: routed.route.requestedAccountId, to: providerAccountId, model: activeRoute.model, resetAt: routed.signal.resetAt },
        }))
      } catch { /* mission gone */ }
        rememberModelRouteHistory('mission', routed.route.audit, routed.signal, { id: current.id, label: current.title })
        await auditEvent('model_router.mission_failover', {
          missionId: current.id,
          providerId: latestBaseProvider.id,
          model: activeRoute.model,
          requestedAccountId: routed.route.requestedAccountId,
          selectedAccountId: providerAccountId,
          resetAt: routed.signal.resetAt,
          source: routed.signal.source,
          exhaustionKind: routed.signal.kind,
          reason: auditPreview(routed.route.audit.reason),
        })
        injectThreadRouteHandoff(thread!, `Mission pass ${nextRunIndex} failed over after ${routed.signal.kind.replace(/_/g, ' ')}.`)
        fanOutPass = await runMissionPassWithRoute(activeRoute)
      }
      if (controller.signal.aborted) break

      runIndex = nextRunIndex
      const passTimelineStart = fanOutPass?.timelineStart ?? timelineStart
      const verifier = fanOutPass?.verifier ?? await verifyMissionStopCondition(activeRoute.provider, activeRoute.model, current, thread, controller.signal, requestGuards.get(thread.id) ?? assertRequestAllowed)
      const frontendEvidence = missionNeedsFrontendUiSmoke(current) ? buildFrontendMissionEvidence(thread) : null
      let afterPass = addMissionCheckpoint(getMission(current.id), {
        label: `Mission pass ${runIndex}`,
        summary: missionPassSummary(thread, passTimelineStart),
        status: 'running',
        artifactRefs: frontendEvidence?.artifactRefs.length ? frontendEvidence.artifactRefs : undefined,
        data: {
          runIndex,
          threadId: thread.id,
          providerId: baseProvider.id,
          providerAccountId: activeRoute.providerAccountId,
          model: activeRoute.model,
          verifier,
          evidenceCount: verifier.evidenceCount ?? missionEvidenceItems(thread).length,
          timelineItems: thread.timeline.length - passTimelineStart,
          ...(fanOutPass ? { fanOut: { budget: current.fanOutBudget, ranking: fanOutPass.ranking, integration: fanOutPass.integration } } : {}),
          ...(frontendEvidence ? { frontendUiSmokeEvidence: frontendEvidence } : {}),
        },
      })
      // Durable journal: mirror the freshly-added checkpoint onto the on-disk journal.
      const newCheckpoint = afterPass.checkpoints.at(-1)
      if (newCheckpoint) {
        recordMissionCheckpoint(afterPass.id, {
          id: newCheckpoint.id,
          label: newCheckpoint.label,
          sequence: newCheckpoint.sequence,
          runIndex,
        })
      }
      afterPass = {
        ...afterPass,
        metadata: {
          ...(afterPass.metadata ?? {}),
          runCount: runIndex,
          lastVerifierNote: verifier.note,
          lastEvidenceCount: verifier.evidenceCount ?? missionEvidenceItems(thread).length,
        },
      }
      replaceMission(afterPass)

      const decision = decideNextMissionStep(afterPass, runIndex, verifier)
      // Surface an illegal-transition invariant on the existing error journal path.
      // Non-blocking: the lifecycle helpers already no-op terminal transitions, so
      // legal decisions are unaffected — this just records the typed reason.
      if (decision.invariantError) {
        try {
          replaceMission(appendMissionJournal(getMission(current.id), {
            kind: 'error',
            actor: 'system',
            message: decision.invariantError.message,
            data: decision.invariantError.toFields(),
          }))
        } catch { /* mission gone */ }
      }
      if (decision.action === 'complete') {
        try { replaceMission(completeMission(getMission(current.id), decision.reason)) } catch { /* mission gone */ }
        finishJob(job, 'completed')
        return
      }
      if (decision.action === 'pause') {
        try { replaceMission(pauseMission(getMission(current.id), decision.reason)) } catch { /* mission gone */ }
        finishJob(job, 'paused', decision.reason)
        return
      }
      continuationPrompt = decision.nextPrompt ?? null
      try { replaceMission(appendMissionJournal(getMission(current.id), { kind: 'status', actor: 'system', message: `Continuing mission: ${decision.reason}`, status: 'running', data: { runIndex } })) } catch { /* mission gone */ }
    }
    const interrupted = interruptedMissionSettlement(stalledOut, getMission(mission.id).status)
    if (interrupted.action === 'fail') {
      try { replaceMission(failMission(getMission(mission.id), interrupted.reason)) } catch { /* mission gone */ }
      finishJob(job, 'failed', interrupted.reason)
    } else if (interrupted.action === 'pause') {
      try { replaceMission(pauseMission(getMission(mission.id), interrupted.reason)) } catch { /* mission gone */ }
      finishJob(job, 'paused', interrupted.reason)
    }
  } catch (e: any) {
    const currentStatus = (() => { try { return getMission(mission.id).status } catch { return 'cancelled' as const } })()
    const interrupted = interruptedMissionSettlement(stalledOut, currentStatus)
    if (interrupted.action === 'fail') {
      try { replaceMission(failMission(getMission(mission.id), interrupted.reason)) } catch { /* mission gone */ }
      finishJob(job, 'failed', interrupted.reason)
    } else if (!controller.signal.aborted) {
      const message = String(e?.message ?? e)
      try { replaceMission(failMission(getMission(mission.id), message)) } catch { /* mission gone */ }
      finishJob(job, 'failed', message)
    } else if (interrupted.action === 'pause') {
      try { replaceMission(pauseMission(getMission(mission.id), interrupted.reason)) } catch { /* mission gone */ }
      finishJob(job, 'paused', interrupted.reason)
    }
  } finally {
    if (watchdog) clearInterval(watchdog)
    runningMissions.delete(mission.id)
    missionAborters.delete(mission.id)
  }
}
  return { runMission }
}
