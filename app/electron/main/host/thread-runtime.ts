import { randomUUID } from 'node:crypto'
import { restartResumeApprovalMode, claimJobLease } from '../../agent/jobs'
import type { JobPatch } from '../../agent/jobs'
import { runAgent } from '../../agent/loop'
import { runOmpThread } from '../../agent/omp-engine'
import { assertOmpEngineAllowed } from '../../agent/planning-guard'
import { executeTool } from '../../agent/tools'
import { createSubAgentFromLaunch, parentForThread, updateSubAgent } from '../../agent/subagents'
import { saveSubAgents, vontraPath } from '../../store'
import { selectChannelPrompt } from '../../agent/channel-prompt'
import { buildUserContent, contentText } from '../../agent/content'
import { normalizeAttachments } from '../../agent/attachment-policy'
import { appendThreadEvent, compactThreadJournal, readThreadJournal } from '../../agent/thread-journal'
import { quickComplete, streamChat } from '../../agent/openai'
import { createThreadTitleGenerator } from './thread-title'
import { harvestPostRunMemoryCandidates } from '../../agent/memory-harvest'
import { reviewSkillsAfterRun } from '../../agent/skill-self-improvement'
import { addSuggestion } from '../../agent/suggestions'
import { resolveProviderModel, surfaceModelOverride } from '../../agent/provider-accounts'
import { assertLocalEndpointAvailable } from '../../agent/local-endpoint'
import { routeSpendStatus } from '../../agent/model-router'
import { ensureTensorFoldStudioStylePrompt } from '../../agent/prompt-style'
import { resolvePersona, personaSystemBlock } from '../../agent/persona'
import { createProjectMemoryCandidate } from '../../agent/project-memory'
import { persistProjectMemoryCandidates, persistSuggestions } from './persistence'
import { assertProviderEndpointResolved, auditPreview } from '../../agent/security'
import { ensureAgentEventTrace, ensureTimelineItemTrace } from '../../agent/trace'
import { buildAgentRunContract, buildAgentRunReceipt, buildEventReceipt } from '../../agent/run-contract'
import type { AgentCue, AgentEvent, ApiMessage, CueRunPayload, CueRunResult, Job, Project, Provider, Room, Seat, StartRunPayload, Thread, TimelineItem, WorkspaceLocation } from '../../agent/types'
import { threadTurnNumber } from './checkpoint-runtime'
import { approvalPolicyScope, type ApprovalPolicyScope } from '../../agent/approval-policy'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import {
  auditModelRetry,
  auditModelRoute,
  modelRouteAccountLabel,
  modelRouteAutoFailoverEnabled,
  modelRouteFailoverText,
  modelRouteResetPolicy,
  modelRouter,
  rememberProviderModels,
  routeHandoffPreview,
  runtimeProviderWithFreshModels,
  selectedProviderAccountId,
} from './provider-runtime'
import { resolveWorkspaceCwd } from './workspace'
import { resolvePairedWorkspace } from '../../agent/local-workspace-policy'
import { encryptWithKey, decryptWithKey, resolveLocalSecretKeySync } from '../../local-crypto'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, sshWorkspaceKey, type WorkspaceMutationGuard } from './workspace-mutation-guard'
import { botInstructions, botRunApprovalMode } from '../../../shared/bot-policy'
import { createBotVmToolContext, getBotVmHost } from './bot-vm-host'
import { assertBotLifecycleIdle, reserveBotVmRun } from './bot-run-boundary'
import { engineService } from './engine-runtime'
import { isEngineProvider } from '../../agent/engine/provider'
import { getUserQuestionHost } from './user-question-host'
import { isExternalProvider } from '../../../shared/provider-trust'
import { INCOGNITO_PERSISTENT_TOOLS } from '../../agent/incognito'
import { PluginRuntime } from '../../agent/plugin-runtime'
import type { UserQuestionInput, UserQuestionRun } from '../../agent/types'

// Per-thread NDJSON journals hold the FULL transcript (messages, tool args, tool output,
// diffs) — the most sensitive data in the app. threads.json is encrypted at rest via
// writeProtectedJson, so writing the journal as plaintext NDJSON defeats the same
// at-rest posture (local-crypto.ts: casual disk reads / accidental settings-dir sharing).
// We encrypt each journal event with the same app key, keeping the {seq, ts} envelope in
// the clear so compaction (which only reads seq) keeps working without a key.
const JOURNAL_ENC_PREFIX = 'enc:v1:'

// Lazily resolve the same primary/fallback keys as store.ts. Failure disables the
// best-effort journal; it must never downgrade a full transcript to plaintext.
let cachedJournalKey: Buffer | null = null
function journalKey(): Buffer | null {
  if (cachedJournalKey) return cachedJournalKey
  try {
    cachedJournalKey = resolveLocalSecretKeySync(vontraPath('.secretkey')).key
    return cachedJournalKey
  } catch {
    return null
  }
}

// Wrap an AgentEvent for the journal: encrypt its JSON with the app key when available,
// emitting { e: 'enc:v1:<ciphertext>' }. If crypto is unavailable, return null so
// the caller skips this optional journal rather than leaking the event verbatim.
export function encodeJournalEvent(event: unknown, key: Buffer | null): unknown | null {
  if (!key) return null
  try {
    return { e: JOURNAL_ENC_PREFIX + encryptWithKey(JSON.stringify(event), key) }
  } catch {
    return null
  }
}

// Inverse of encodeJournalEvent: decrypt an encrypted journal envelope back to the
// original event, tolerating legacy/plaintext lines written before encryption existed.
export function decodeJournalEvent(stored: unknown, key: Buffer | null): unknown {
  if (stored && typeof stored === 'object' && typeof (stored as { e?: unknown }).e === 'string') {
    const payload = (stored as { e: string }).e
    if (payload.startsWith(JOURNAL_ENC_PREFIX)) {
      if (!key) return null // cannot decrypt without the key
      try {
        return JSON.parse(decryptWithKey(payload.slice(JOURNAL_ENC_PREFIX.length), key))
      } catch {
        return null // wrong key / tampered / corrupt
      }
    }
  }
  return stored // legacy plaintext event written before journal encryption
}

interface JobHandle {
  get(): Job | null
  set(job: Job | null): void
}

export interface ThreadRuntime {
  addThreadNotice(thread: Thread, runId: string, text: string): void
  applyEventToThread(thread: Thread, event: AgentEvent): void
  cancelThreadRun(threadId: string): void
  clearCues(): void
  drainThreadCues(threadId: string): AgentCue[]
  guideThreadRun(payload: CueRunPayload): Promise<CueRunResult>
  injectThreadRouteHandoff(thread: Thread, reason: string): string
  markRouteErrorRecovered(thread: Thread, err: unknown, runId: string, text: string): void
  markUndrainedCues(thread: Thread, meta?: string): void
  roomRouteHandoff(room: Room, seat: Seat, reason: string): string
  startThreadRun(payload: StartRunPayload, existingJob?: Job): Promise<{ runId: string }>
  timelineError(errorItem: TimelineItem): Error
  timelineErrorSince(thread: Thread, timelineStart: number): TimelineItem | null
  waitForThreadCue(threadId: string, timeoutMs: number, signal: AbortSignal): Promise<void>
}

interface ThreadRuntimeDeps {
  aborters: Map<string, AbortController>
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'assertThreadStartAllowed' | 'reserveStart'>
  pendingApprovalMeta: Map<string, { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }>
  jobLeaseOwner: string
  threadJobHeartbeatMs: number
  sshManager: {
    ensureConnected(hostId: string): Promise<unknown>
    status?(hostId: string): { phase: string; lastError?: string | null }
  }
  emitToRenderer(event: AgentEvent): void
  persistThreads(): void
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal }): any
  approvalResolver(controller: AbortController, jobHandle?: JobHandle): (callId: string) => Promise<'allow' | 'allow_always' | 'deny'>
  isHighRiskTool(name: string): boolean
  claimJob(ref: Parameters<typeof import('../../agent/jobs').enqueueJob>[1]): Job
  patchJob(job: Job | null, patch: JobPatch): Job | null
  replaceJobResult(result: { jobs: Job[]; job: Job | null }): Job
  heartbeatJob(job: Job | null): Job | null
  settleThreadRunJob(job: Job | null, thread: Thread, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  renderProjectMemoryBucket(project: { id: string; name: string } | undefined, query?: string): string
  effectiveContextLimit(provider: Provider, model: string | null): number
  // Best-effort per-turn worktree checkpoint (undo-a-turn / turn diff). Optional so
  // the chat/cues tests need not wire it; failures are swallowed by the service.
  captureCheckpoint?(threadId: string, turn: number, target: { cwd?: string | null; workspace?: WorkspaceLocation | null; kind?: Thread['kind'] }): Promise<{ ok: boolean; sha?: string }>
  // Record the conversation snapshot for this turn alongside the git checkpoint (VST-036),
  // so a later revert/fork restores files AND conversation together. Optional + best-effort.
  recordTurnCheckpoint?(threadId: string, turn: number, conversation: { apiMessages: unknown[]; timeline: unknown[]; goal?: string; goalStatus?: string; plan?: unknown[]; status?: string }, fileSha?: string | null): void
  // Persist a real codex account rate-limit snapshot (VST-037) for the picker + doctor.
  recordProviderRateLimits?(input: { providerId: string; accountId: string | null; snapshot: unknown }): void
  // Rescan the project's repo-authored skills before a run (VST-010) so trust decisions
  // apply to the content actually on disk — a stale cache must not run edited skills.
  refreshProjectSkills?(project: Pick<Project, 'id' | 'cwd' | 'workspace'> | null): Promise<void>
  // Observer hooks (VST-012): forwarded into the loop for api.request/response events.
  observe?: (kind: 'api.request' | 'api.response', payload: Record<string, unknown>) => void
}

export async function recordPreTurnCheckpoint(input: {
  threadId: string
  incognito?: boolean
  turn: number
  target: { cwd?: string | null; workspace?: WorkspaceLocation | null; kind?: Thread['kind'] }
  conversation: { apiMessages: unknown[]; timeline: unknown[]; goal?: string; goalStatus?: string; plan?: unknown[]; status?: string }
}, deps: Pick<ThreadRuntimeDeps, 'captureCheckpoint' | 'recordTurnCheckpoint'>): Promise<void> {
  if (input.incognito) return
  let fileSha: string | null = null
  if (deps.captureCheckpoint && input.target.kind !== 'chat') {
    try {
      const captured = await deps.captureCheckpoint(input.threadId, input.turn, input.target)
      if (captured.ok && captured.sha) fileSha = captured.sha
    } catch { /* file checkpoints are best-effort; still retain the conversation snapshot */ }
  }
  try { deps.recordTurnCheckpoint?.(input.threadId, input.turn, input.conversation, fileSha) } catch { /* best effort */ }
}

export async function resolveThreadRunCwd(thread: Pick<Thread, 'kind' | 'cwd' | 'workspaceBoundary'>, workspace: WorkspaceLocation | null, boundary = thread.workspaceBoundary): Promise<string> {
  if (thread.kind === 'chat') return thread.cwd
  if (boundary === 'home') return resolvePairedWorkspace(thread.cwd, workspace)
  if (workspace?.kind === 'ssh') {
    const remotePath = String(workspace.path || '').trim()
    if (!remotePath) throw new Error('Choose a remote workspace folder before running workspace tools.')
    return remotePath
  }
  const localPath = workspace?.kind === 'local' && workspace.path ? workspace.path : thread.cwd
  return resolveWorkspaceCwd(localPath)
}

const LOCAL_DELEGATED_TRANSPORTS = new Set<NonNullable<Provider['transport']>>([
  'codex-cli', 'claude-cli', 'opencode', 'kilo', 'pi-cli', 'cursor-cli', 'gemini-cli',
])

export function assertSshThreadTransportSupported(workspace: WorkspaceLocation | null, provider: Provider | null | undefined): void {
  if (workspace?.kind === 'ssh' && provider?.transport && LOCAL_DELEGATED_TRANSPORTS.has(provider.transport)) {
    throw new Error(`${provider.name || provider.transport} runs on this Mac and cannot safely use an SSH workspace. Choose an HTTP/model transport with SSH-backed tools, or run the delegated engine on the remote Orchestrator.`)
  }
}

export function assertSshToolContextConnected(workspace: WorkspaceLocation | null, toolContext: { backend?: unknown } | null | undefined): void {
  if (workspace?.kind === 'ssh' && !toolContext?.backend) {
    throw new Error('The SSH workspace disconnected during startup. Reconnect it before running the agent.')
  }
}

export function assertIncognitoThreadTransportSupported(thread: Pick<Thread, 'incognito' | 'engine'>, provider?: Pick<Provider, 'transport'>): void {
  if (thread.incognito && (thread.engine === 'omp' || provider?.transport && provider.transport !== 'chat-completions')) {
    throw new Error('Incognito requires a TensorFold API chat. This agent runtime keeps its own session history; choose an API model or start an ordinary saved task.')
  }
}

export function createThreadRuntime(deps: ThreadRuntimeDeps): ThreadRuntime {
  const {
    aborters,
    pendingApprovalMeta,
    jobLeaseOwner,
    threadJobHeartbeatMs,
    sshManager,
    emitToRenderer,
    persistThreads,
    fullToolContext,
    approvalResolver,
    isHighRiskTool,
    claimJob,
    patchJob,
    replaceJobResult,
    heartbeatJob,
    settleThreadRunJob,
    renderProjectMemoryBucket,
    effectiveContextLimit,
  } = deps
  const cueQueues = new Map<string, AgentCue[]>()
  const cueWaiters = new Map<string, Array<() => void>>()

  function enqueueThreadCue(thread: Thread, text: string): CueRunResult {
    const clean = text.trim()
    if (!clean || !aborters.has(thread.id)) return { queued: false, queueDepth: cueQueues.get(thread.id)?.length ?? 0 }
    const item = { id: randomUUID(), kind: 'cue' as const, title: 'Guide', text: clean, state: 'pending' as const, meta: 'queued', createdAt: Date.now() }
    ensureTimelineItemTrace(item, { threadId: thread.id, origin: 'renderer.cue' })
    const cue: AgentCue = { id: randomUUID(), itemId: item.id, text: clean, createdAt: item.createdAt }
    const queue = cueQueues.get(thread.id) ?? []
    queue.push(cue)
    cueQueues.set(thread.id, queue)
    thread.timeline.push(item)
    thread.updatedAt = Date.now()
    persistThreads()
    emitToRenderer({ runId: '', threadId: thread.id, type: 'item_add', item })
    const waiters = cueWaiters.get(thread.id) ?? []
    cueWaiters.delete(thread.id)
    for (const wake of waiters) wake()
    return { queued: true, queueDepth: queue.length }
  }

  function drainThreadCues(threadId: string): AgentCue[] {
    const queue = cueQueues.get(threadId) ?? []
    cueQueues.delete(threadId)
    return queue
  }

  function waitForThreadCue(threadId: string, timeoutMs: number, signal: AbortSignal): Promise<void> {
    if ((cueQueues.get(threadId)?.length ?? 0) > 0 || signal.aborted) return Promise.resolve()
    return new Promise((resolve) => {
      let done = false
      const finish = () => {
        if (done) return
        done = true
        clearTimeout(timer)
        signal.removeEventListener('abort', finish)
        const waiters = cueWaiters.get(threadId) ?? []
        const idx = waiters.indexOf(finish)
        if (idx >= 0) waiters.splice(idx, 1)
        if (waiters.length) cueWaiters.set(threadId, waiters)
        else cueWaiters.delete(threadId)
        resolve()
      }
      const timer = setTimeout(finish, timeoutMs)
      signal.addEventListener('abort', finish, { once: true })
      cueWaiters.set(threadId, [...(cueWaiters.get(threadId) ?? []), finish])
    })
  }

  function markUndrainedCues(thread: Thread, meta = 'not applied'): void {
    for (const cue of drainThreadCues(thread.id)) {
      const item = thread.timeline.find((candidate) => candidate.id === cue.itemId)
      if (item) Object.assign(item, { state: 'denied' as const, meta })
      emitToRenderer({ runId: '', threadId: thread.id, type: 'item_update', id: cue.itemId, patch: { state: 'denied', meta } })
    }
  }

  function addThreadNotice(thread: Thread, runId: string, text: string) {
    const notice = { id: randomUUID(), kind: 'notice' as const, text, createdAt: Date.now() }
    thread.timeline.push(notice)
    emitToRenderer({ runId, threadId: thread.id, type: 'item_add', item: notice })
  }

  function timelineErrorSince(thread: Thread, timelineStart: number): TimelineItem | null {
    return thread.timeline.slice(timelineStart).find((item) => item.kind === 'error' && item.text?.trim()) ?? null
  }

  function threadRouteHandoff(thread: Thread, reason: string): string {
    const recent = thread.timeline
      .slice(-10)
      .map((item) => {
        const label = item.tool ? `${item.kind}:${item.tool}` : item.kind
        const detail = item.text ?? item.title ?? item.output ?? item.meta ?? item.path ?? ''
        return `- ${label}: ${routeHandoffPreview(detail)}`
      })
      .filter((line) => !line.endsWith(': '))
    return [
      '<model_router_handoff>',
      `Reason: ${routeHandoffPreview(reason, 300)}`,
      `Thread: ${routeHandoffPreview(thread.title, 160)}`,
      `Workspace: ${routeHandoffPreview(thread.cwd, 180)}`,
      'Recent state:',
      ...(recent.length ? recent : ['- No recent timeline state was available.']),
      '</model_router_handoff>',
    ].join('\n')
  }

  function injectThreadRouteHandoff(thread: Thread, reason: string): string {
    const handoff = threadRouteHandoff(thread, reason)
    thread.apiMessages.push({ role: 'user', content: handoff })
    return handoff
  }

  function roomRouteHandoff(room: Room, seat: Seat, reason: string): string {
    const recent = room.events
      .slice(-10)
      .map((item) => `- tick ${item.tick} ${item.seat}/${item.kind}: ${routeHandoffPreview(item.text ?? item.tool ?? item.status ?? '')}`)
      .filter((line) => !line.endsWith(': '))
    return [
      'Model router handoff before retry.',
      `Reason: ${routeHandoffPreview(reason, 300)}`,
      `Seat: ${routeHandoffPreview(seat.name, 120)} (${routeHandoffPreview(seat.role, 180)})`,
      'Recent room state:',
      ...(recent.length ? recent : ['- No recent room state was available.']),
    ].join('\n')
  }

  function timelineError(errorItem: TimelineItem): Error {
    const err = new Error(errorItem.text ?? 'Agent run failed.')
    ;(err as Error & { timelineItemId?: string }).timelineItemId = errorItem.id
    return err
  }

  function markRouteErrorRecovered(thread: Thread, err: unknown, runId: string, text: string) {
    const itemId = (err as { timelineItemId?: string } | null)?.timelineItemId
    const item = itemId ? thread.timeline.find((candidate) => candidate.id === itemId && candidate.kind === 'error') : null
    if (!item) return
    item.kind = 'notice'
    item.text = text
    emitToRenderer({ runId, threadId: thread.id, type: 'item_update', id: item.id, patch: { kind: 'notice', text } })
  }

  function cancelThreadRun(threadId: string): void {
    requireSecurityRole('operator', 'cancel agent runs')
    aborters.get(threadId)?.abort()
  }

  async function guideThreadRun(payload: CueRunPayload): Promise<CueRunResult> {
    requireSecurityRole('operator', 'guide running threads')
    const thread = store.threads.find((t) => t.id === payload.threadId)
    if (!thread) throw new Error('Thread not found')
    if (thread.archivedAt) throw new Error('Restore this archived thread before starting it.')
    botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, store.settings.approvalMode)
    if (thread.bot) assertBotLifecycleIdle(thread.bot.presetId)
    const attachments = normalizeAttachments(payload.attachments)
    // Live steering is text-only. Preserve bounded document text and image notes, but never
    // forward an image URL through the cue channel.
    return enqueueThreadCue(thread, contentText(buildUserContent(String(payload.userText ?? ''), attachments)))
  }

  function availableModelsFromProviderError(error: unknown): string[] {
    const message = String((error as any)?.message ?? error ?? '')
    let providerMessage = message
    const jsonStart = message.indexOf('— {')
    if (jsonStart >= 0) {
      try {
        const parsed = JSON.parse(message.slice(jsonStart + 2).trim())
        providerMessage = String(parsed?.error?.message ?? parsed?.message ?? providerMessage)
      } catch { /* keep raw message */ }
    }
    const marker = 'Available models:'
    const index = providerMessage.indexOf(marker)
    if (index < 0) return []
    return Array.from(new Set(providerMessage
      .slice(index + marker.length)
      .replace(/[}"']+$/g, '')
      .split(',')
      .map((model) => model.trim())
      .filter(Boolean)))
  }

  // Keep only the most recent JOURNAL_TAIL_KEEP events of a thread's journal so it never grows
  // without bound. Best-effort: any fs/journal error is swallowed (the journal is a crash-recovery
  // convenience, not a correctness dependency).
  const JOURNAL_TAIL_KEEP = 500
  async function compactThreadJournalTail(threadId: string): Promise<void> {
    try {
      const baseDir = vontraPath('thread-journals')
      const entries = await readThreadJournal(baseDir, threadId)
      if (entries.length <= JOURNAL_TAIL_KEEP) return
      const keepFromSeq = entries[entries.length - JOURNAL_TAIL_KEEP].seq
      await compactThreadJournal(baseDir, threadId, keepFromSeq)
    } catch { /* journal compaction is best-effort */ }
  }

  // Aux-model skill review (opt-in). For each skill the agent consulted on a successful run,
  // ask the same backend whether the skill body could be sharper, and FILE any improvement as
  // a pending suggestion (deduped per target). Best-effort and fully detached from the run.
  async function reviewViewedSkills(args: { provider: Provider; model: string; thread: Thread; userText: string; viewed: string[]; signal: AbortSignal }): Promise<void> {
    if (args.thread.incognito) return
    try {
      const { provider, model } = args
      const digest = `User request: ${(args.userText || args.thread.title || '').slice(0, 600)}\nOutcome: the run completed successfully.`
      const complete = async (messages: { role: string; content: string }[]): Promise<string> => {
        let out = ''
        for await (const ch of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages: messages as ApiMessage[], temperature: 0.2, approvalMode: 'readonly', cwd: args.thread.cwd, signal: args.signal })) {
          if (ch.type === 'content') out += ch.text
        }
        return out
      }
      const proposals = await reviewSkillsAfterRun({ skills: store.skills, viewedNames: args.viewed, digest, complete, sourceThreadId: args.thread.id, max: 2 })
      if (!proposals.length) return
      const corpus = { skills: store.skills, commands: store.commands }
      let filed = 0
      for (const input of proposals) {
        const res = addSuggestion(store.suggestions, input, corpus)
        if (res.added) { store.suggestions = res.list; filed++ }
      }
      if (filed) {
        persistSuggestions()
        store.win?.webContents.send('suggestions:update', store.suggestions)
      }
    } catch { /* best effort — never disturb the settled run */ }
  }

  function proposePostRunMemory(thread: Thread): void {
    if (thread.incognito || !thread.projectId) return
    const existingTexts = [
      ...store.projectMemory.filter((entry) => entry.projectId === thread.projectId).map((entry) => entry.text),
      ...store.projectMemoryCandidates.filter((candidate) => candidate.projectId === thread.projectId && candidate.status === 'pending').map((candidate) => candidate.text),
    ]
    const candidates = harvestPostRunMemoryCandidates(thread.apiMessages, { existingTexts, cap: 3 })
    if (!candidates.length) return
    for (const text of candidates) {
      store.projectMemoryCandidates = [
        // Provenance (VST-009): the review UI links each proposal back to the exact
        // conversation turn that surfaced the pattern.
        createProjectMemoryCandidate({ projectId: thread.projectId, text, sourceType: 'agent', reason: 'Proposed after a completed run.', sourceThreadId: thread.id, sourceTurn: threadTurnNumber(thread.timeline) }),
        ...store.projectMemoryCandidates,
      ]
    }
    persistProjectMemoryCandidates()
  }

  async function startThreadRun(payload: StartRunPayload, existingJob?: Job): Promise<{ runId: string }> {
    requireSecurityRole('operator', 'start agent runs')
    const thread = store.threads.find((t) => t.id === payload.threadId)
    if (!thread) throw new Error('Thread not found')
    if (thread.archivedAt) throw new Error('Restore this archived thread before starting it.')
    const runApprovalMode = botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, payload.approvalMode ?? restartResumeApprovalMode(existingJob?.resumeAfterRestart, store.settings.approvalMode))
    const botVm = thread.bot?.executionTarget === 'vm'
    const vmHost = botVm ? getBotVmHost() : null
    if (thread.bot) {
      assertBotLifecycleIdle(thread.bot.presetId)
      if (thread.engine && thread.engine !== 'native') throw new Error('Bot chats require TensorFold permission checks.')
      if (thread.providerId !== thread.bot.providerId || (thread.providerAccountId ?? null) !== thread.bot.providerAccountId || thread.model !== thread.bot.model) throw new Error('The model differs from the captured bot configuration; start a new bot chat.')
      if (thread.workspace || thread.projectId || (botVm ? thread.cwd !== '/workspace' : !!thread.cwd)) throw new Error('The bot execution location has changed; start a new bot chat.')
      if (botVm && !vmHost) throw new Error('The bot VM service is unavailable; the run will not use this Mac instead.')
    }
    // Guard against a second concurrent run for the same thread: it would overwrite the shared
    // aborters[threadId] entry, so the first run's finally cleanup would delete the live run's
    // controller (breaking cancel + cue queueing) and both runs would interleave apiMessages.
    // Checked here (before any thread mutation) so a rejected start leaves state untouched.
    if (aborters.has(thread.id)) throw new Error('Thread already running')
    const project = thread.projectId ? store.projects.find((p) => p.id === thread.projectId) ?? null : null
    if (thread.projectId && (!project || project.archivedAt)) throw new Error('Restore or choose an active project before starting this thread.')
    const workspace = thread.workspace ?? project?.workspace ?? null
    const workspaceBoundary = payload.workspaceBoundary ?? thread.workspaceBoundary ?? project?.workspaceBoundary
    if (workspaceBoundary === 'home') await resolveThreadRunCwd(thread, workspace, workspaceBoundary)
    const attachments = normalizeAttachments(payload.attachments)
    const surfaceRoute = thread.providerId ? null : surfaceModelOverride(store.settings, thread.kind === 'chat' ? 'chat' : 'code')
    const baseProvider = store.settings.providers.find((p) => p.id === (thread.providerId ?? surfaceRoute?.providerId ?? store.settings.activeProviderId))
    assertIncognitoThreadTransportSupported(thread, baseProvider)
    if (thread.incognito && existingJob && (existingJob.incognito !== true || existingJob.threadId !== thread.id || existingJob.currentThreadId !== thread.id)) throw new Error('An Incognito task cannot resume a saved or unrelated job; start a new private turn.')
    assertSshThreadTransportSupported(workspace, baseProvider)
    const initialWorkspaceKey = workspace?.kind === 'ssh'
      ? sshWorkspaceKey(workspace.hostId, workspace.path)
      : thread.kind === 'chat' || botVm ? null : lexicalLocalWorkspaceKey(workspace?.kind === 'local' ? workspace.path : thread.cwd)
    deps.workspaceMutationGuard.assertThreadStartAllowed(thread.id, initialWorkspaceKey)
    const runLease = deps.workspaceMutationGuard.reserveStart({ threadId: thread.id, workspaceKeys: [initialWorkspaceKey] })
    let releaseBotVmRun = () => {}
    try { if (botVm) releaseBotVmRun = reserveBotVmRun(thread.bot!.presetId, thread.id) } catch (error) { runLease.release(); throw error }
    const controller = new AbortController()
    aborters.set(thread.id, controller)
    try {
    if (botVm) {
      await vmHost!.requireReady(thread.bot!.presetId)
      if (controller.signal.aborted) throw new Error('Run cancelled before VM startup completed.')
      if (isEngineProvider(baseProvider) && engineService()?.hasExternalResearchEnabled()) throw new Error('Turn off Cortheon in AI settings before using a bot VM; its model must keep VM data on this Mac.')
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
        contextLimit: effectiveContextLimit(baseProvider, blockedModel),
        reasoningEffort: store.settings.reasoningEffort,
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
      contextLimit: effectiveContextLimit(routeBaseProvider, model),
      reasoningEffort: store.settings.reasoningEffort,
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
    const userItem = { id: randomUUID(), kind: 'user' as const, text: payload.userText, attachments, createdAt: Date.now() }
    thread.timeline.push(userItem)
    thread.apiMessages.push({ role: 'user', content: buildUserContent(payload.userText, attachments) })
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

    if (thread.title === 'New task' || thread.title === 'New chat') maybeAutoTitle(thread, provider, model, payload.userText)

    const routeProviderForModel = (routeProvider: Provider, routeModel: string, runModel: string) =>
      runModel === routeModel ? routeProvider : { ...routeProvider, models: [runModel, ...routeProvider.models.filter((candidate) => candidate !== runModel)] }

    const runWithRoute = async (route: { provider: Provider; providerAccountId: string | null; model: string }, runModel = route.model) => {
      const timelineStart = thread.timeline.length
      const activeProvider = routeProviderForModel(route.provider, route.model, runModel)
      assertIncognitoThreadTransportSupported(thread, activeProvider)
      if (thread.bot) {
        assertBotLifecycleIdle(thread.bot.presetId)
        botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, runApprovalMode)
        if (botVm && isEngineProvider(activeProvider) && engineService()?.hasExternalResearchEnabled()) throw new Error('Turn off Cortheon in AI settings before using a bot VM; its model must keep VM data on this Mac.')
      }
      reviewProvider = activeProvider
      reviewModel = runModel
      let lastJobBeat = Date.now()
      const beatThreadJob = () => {
        if (Date.now() - lastJobBeat < threadJobHeartbeatMs) return
        lastJobBeat = Date.now()
        job = heartbeatJob(job) ?? job
      }
      // Provider/route discovery above can take long enough for an SSH session to drop.
      // Capture the tool context once at the final execution seam, validate that exact
      // snapshot, and pass it through unchanged. Re-resolving later (or accepting an
      // undefined backend) could reinterpret the remote path as a local lookalike.
      if (thread.bot && (activeProvider.id !== thread.bot.providerId || route.providerAccountId !== thread.bot.providerAccountId || runModel !== thread.bot.model)) throw new Error('Bots keep their captured model and account; start a new bot chat to change them.')
      const baseToolContext = botVm ? { cwd: '/workspace', signal: controller.signal, currentThread: thread } : fullToolContext({ thread, provider: activeProvider, model: runModel, signal: controller.signal })
      const toolContext = botVm ? createBotVmToolContext(vmHost!, thread.bot!.presetId, baseToolContext) : baseToolContext
      if (thread.incognito) toolContext.disabledToolNames = [...new Set([...toolContext.disabledToolNames ?? [], ...INCOGNITO_PERSISTENT_TOOLS])]
      if (botVm) toolContext.botVm!.assertModelPrivacy = () => {
        assertBotLifecycleIdle(thread.bot!.presetId)
        botRunApprovalMode(thread, store.settings.agentPresets, store.settings.providers, runApprovalMode)
        if (isEngineProvider(activeProvider) && engineService()?.hasExternalResearchEnabled()) throw new Error('Turn off Cortheon in AI settings before using a bot VM; its model must keep VM data on this Mac.')
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
        await runOmpThread({
          thread,
          runId,
          emit: (event) => { emit(event); beatThreadJob() },
          signal: controller.signal,
          persist: persistThreads,
          userText: payload.userText,
          approvalMode: runApprovalMode,
          reasoningEffort: store.settings.reasoningEffort,
          drainCues: () => drainThreadCues(thread.id),
          onSession: (cursor) => { thread.ompResumeCursor = cursor },
          // Host-tools bridge: omp's model plans, TensorFold's approval gate and executor
          // own every mutation (write_file/edit_file/apply_patch/run_command).
          requestApproval: approvalResolver(controller, jobHandle),
          executeTool: async (name, args) => executeTool(name, args, { ...toolContext, currentThread: thread, signal: controller.signal }),
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
        })
      } else {
      await runAgent({
        thread,
        provider: activeProvider,
        model: runModel,
        runId,
        approvalMode: runApprovalMode,
        temperature: store.settings.temperature,
        reasoningEffort: store.settings.reasoningEffort,
        serviceTier: store.settings.serviceTier,
        autoCompact: store.settings.autoCompact,
        contextLimit: effectiveContextLimit(activeProvider, runModel),
        toolContext,
        observe: thread.incognito ? undefined : deps.observe,
        pluginRuntime: thread.incognito ? new PluginRuntime() : undefined,
        drainCues: () => drainThreadCues(thread.id),
        waitForCue: (timeoutMs, signal) => waitForThreadCue(thread.id, timeoutMs, signal),
        projectMemory: botVm ? '' : renderProjectMemoryBucket(store.projects.find((p) => p.id === thread.projectId), payload.userText),
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
          if (thread.incognito || !thread.projectId || !candidates.length) return
          for (const text of candidates) {
            store.projectMemoryCandidates = [createProjectMemoryCandidate({ projectId: thread.projectId, text, sourceType: 'agent', reason: 'Salvaged from the conversation at compaction.', sourceThreadId: thread.id, sourceTurn: threadTurnNumber(thread.timeline) }), ...store.projectMemoryCandidates]
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
          if (tool === 'skill_view' && resultOk && args?.name) viewedSkillNames.add(String(args.name))
          if (isHighRiskTool(tool)) await auditEvent('agent.tool', { threadId: thread.id, runId, tool, ok: resultOk, ...(thread.incognito ? {} : { cwd: auditPreview(thread.cwd), args: auditPreview(JSON.stringify(args)), output: auditPreview(output) }) })
        },
        persist: persistThreads,
      })
      }
      const errorItem = timelineErrorSince(thread, timelineStart)
      if (errorItem && !controller.signal.aborted) throw timelineError(errorItem)
    }

    let activeRoute = { provider, providerAccountId, model }
    // Scope the run's success/failure to items THIS run appends — a leftover error item from a
    // previous run must not mark a later successful run as failed (which stalls the PR loop).

    // Skill self-improvement bookkeeping: which skills the agent consulted this run, and the
    // provider/model it ran on (so the post-run review reuses the same backend). Populated in
    // auditToolCall + runWithRoute; consumed in the .finally review pass.
    const viewedSkillNames = new Set<string>()
    let reviewProvider: Provider | null = null
    let reviewModel = ''

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

    runWithRoute(activeRoute)
      .catch(async (err) => {
        const liveModels = availableModelsFromProviderError(err)
        const fallbackModel = liveModels.find((candidate) => candidate !== activeRoute.model)
        if (fallbackModel && !thread.bot && !controller.signal.aborted) {
          const rejectedModel = activeRoute.model
          markRouteErrorRecovered(thread, err, runId, `Model "${rejectedModel}" was rejected by ${baseProvider.name}; retrying with "${fallbackModel}".`)
          rememberProviderModels(baseProvider.id, activeRoute.providerAccountId, liveModels)
          activeRoute = { ...activeRoute, model: fallbackModel }
          thread.model = fallbackModel
          thread.updatedAt = Date.now()
          persistThreads()
          emitToRenderer({ runId, threadId: thread.id, type: 'thread_update', patch: { model: fallbackModel } })
          const notice = { id: randomUUID(), kind: 'notice' as const, text: `Model "${rejectedModel}" was rejected by ${baseProvider.name}. Retrying with "${fallbackModel}".`, createdAt: Date.now() }
          thread.timeline.push(notice)
          emitToRenderer({ runId, threadId: thread.id, type: 'item_add', item: notice })
          await auditModelRetry(thread, {
            providerId: baseProvider.id,
            providerAccountId: activeRoute.providerAccountId,
            fromModel: rejectedModel,
            toModel: fallbackModel,
            source: 'provider_available_models',
            reason: `Provider rejected model "${rejectedModel}" and returned live alternatives.`,
            availableModels: liveModels,
          })
          try {
            await runWithRoute(activeRoute, fallbackModel)
          } catch (retryErr) {
            emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String((retryErr as any)?.message ?? retryErr) })
          }
          return
        }
        // The route-recovery setup below performs un-try-caught awaits (audit, provider refresh,
        // and assertProviderEndpointResolved — which throws when a failover endpoint resolves to a
        // private IP). Without this guard a throw here rejects the catch callback itself; the chain
        // has no further .catch, so it becomes an unhandled rejection and the user sees the run
        // silently die. Wrap it so any failure surfaces as the same user-facing error event the
        // non-recovered path emits, then fall through to the shared .finally.
        try {
          const latestBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
          const routed = modelRouter.routeAfterError({
            role: threadRouteRole,
            provider: latestBaseProvider,
            accountId: activeRoute.providerAccountId,
            model: activeRoute.model,
            error: err,
            allowFailover: !thread.bot && (modelRouteAutoFailoverEnabled(latestBaseProvider, activeRoute.providerAccountId) || thread.modelRoute?.resetPolicy === 'rotate_on_limit'),
            localOnly: project?.localOnly === true,
            contextLimit: effectiveContextLimit(latestBaseProvider, activeRoute.model),
            reasoningEffort: store.settings.reasoningEffort,
            budget: thread.kind,
            source: 'thread_recovery',
          })
          if (routed.signal) {
            for (const audit of routed.audit) await auditModelRoute(thread, audit, routed.signal)
          }
          if (routed.signal && routed.route.selected && routed.route.selectedAccountId !== activeRoute.providerAccountId && !controller.signal.aborted) {
            markRouteErrorRecovered(thread, err, runId, `Recovered provider limit on ${modelRouteAccountLabel(latestBaseProvider, activeRoute.providerAccountId)} for ${activeRoute.model}; retrying on ${modelRouteAccountLabel(latestBaseProvider, routed.route.selectedAccountId)}.`)
            const exhaustedAccountId = activeRoute.providerAccountId
            providerAccountId = routed.route.selectedAccountId
            provider = await runtimeProviderWithFreshModels(latestBaseProvider, providerAccountId)
            await assertProviderEndpointResolved(provider)
            activeRoute = { provider, providerAccountId, model: activeRoute.model }
            thread.providerId = latestBaseProvider.id
            thread.providerAccountId = providerAccountId
            thread.model = activeRoute.model
            thread.modelRoute = {
              preferredProviderAccountId: exhaustedAccountId,
              failoverProviderAccountId: providerAccountId,
              resetPolicy: modelRouteResetPolicy(latestBaseProvider, exhaustedAccountId),
              failedOverAt: Date.now(),
              resetAt: routed.signal.resetAt,
              reason: routed.route.audit.reason,
            }
            thread.updatedAt = Date.now()
            persistThreads()
            emitToRenderer({ runId, threadId: thread.id, type: 'thread_update', patch: { providerId: thread.providerId, providerAccountId, model: activeRoute.model, modelRoute: thread.modelRoute } })
            addThreadNotice(thread, runId, modelRouteFailoverText(latestBaseProvider, routed.route.audit, routed.signal))
            injectThreadRouteHandoff(thread, `Thread "${thread.title}" failed over after ${routed.signal.kind.replace(/_/g, ' ')}.`)
            try {
              await runWithRoute(activeRoute)
            } catch (retryErr) {
              emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String((retryErr as any)?.message ?? retryErr) })
            }
            return
          }
          emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String(err?.message ?? err) })
        } catch (recoveryErr) {
          // Recovery setup itself failed — surface the original error plus the recovery failure.
          emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String((recoveryErr as any)?.message ?? recoveryErr) })
        }
      })
      .finally(() => {
        markUndrainedCues(thread, controller.signal.aborted ? 'canceled' : 'run ended')
        const failed = !!timelineErrorSince(thread, runTimelineStart)
        const remoteTransport = workspace?.kind === 'ssh' && workspace.hostId && sshManager.status ? sshManager.status(workspace.hostId) : null
        const transportUnverifiable = failed && !!remoteTransport && (remoteTransport.phase === 'disconnected' || remoteTransport.phase === 'error')
        if (transportUnverifiable && workspace?.kind === 'ssh') {
          const reason = remoteTransport?.lastError
            ? `Remote transport lost: ${remoteTransport.lastError}`
            : 'Remote transport lost; the process state is unverifiable.'
          job = patchJob(job, {
            executionVerification: 'unverifiable',
            executionVerificationReason: reason,
            executionVerifiedAt: Date.now(),
          })
        }
        const receipt = buildAgentRunReceipt({ thread, contract: runContract, timelineStart: runTimelineStart, aborted: controller.signal.aborted })
        const turnReceipt = buildEventReceipt({
          kind: 'turn_complete',
          runId,
          threadId: thread.id,
          state: controller.signal.aborted ? 'cancelled' : failed ? 'error' : receipt.needsReview ? 'blocked' : 'ok',
          summary: receipt.summary,
          evidenceRefs: [receipt.id],
        })
        const jobReceipt = buildEventReceipt({
          kind: 'job_complete',
          runId,
          threadId: thread.id,
          state: controller.signal.aborted ? 'cancelled' : failed ? 'error' : receipt.needsReview ? 'blocked' : 'ok',
          summary: `Thread job finished with ${receipt.verdict}.`,
          evidenceRefs: [receipt.id],
        })
        const savedReceipt = buildEventReceipt({
          kind: 'run_receipt_saved',
          runId,
          threadId: thread.id,
          state: controller.signal.aborted ? 'cancelled' : failed ? 'error' : receipt.needsReview ? 'blocked' : 'ok',
          summary: 'Run receipt saved at run end.',
          evidenceRefs: [receipt.id],
        })
        receipt.eventReceipts.push(turnReceipt, jobReceipt, savedReceipt)
        emit({ runId, threadId: thread.id, type: 'thread_update', patch: { activeRunContract: null, lastRunReceipt: receipt } })
        emit({
          runId,
          threadId: thread.id,
          type: 'item_add',
          item: {
            id: randomUUID(),
            kind: 'notice',
            title: 'Run receipt',
            text: `Run receipt: ${receipt.verdict}. ${receipt.summary}`,
            state: receipt.needsReview ? 'awaiting' : 'ok',
            runReceipt: receipt,
            eventReceipt: savedReceipt,
            createdAt: Date.now(),
          },
        })
        thread.status = failed ? 'error' : 'idle'
        thread.updatedAt = Date.now()
        // Only retire our own controller — never clobber a controller a later run installed.
        if (aborters.get(thread.id) === controller) aborters.delete(thread.id)
        runLease.release()
        releaseBotVmRun()
        settleThreadRunJob(
          job,
          thread,
          controller.signal.aborted || transportUnverifiable ? 'paused' : failed ? 'failed' : 'completed',
          transportUnverifiable
            ? 'Remote transport disconnected; execution is unverifiable and must be reconciled after reconnecting.'
            : failed ? 'Thread agent run ended with an error.' : undefined,
        )
        persistThreads()
        // Bound the per-thread NDJSON journal: high-frequency item_delta chunks make it grow
        // without limit since compactThreadJournal was never called in production. At run-end the
        // events are already folded into the persisted thread snapshot, so keep only a recent tail
        // for crash replay. Fire-and-forget; failures must not affect run settlement.
        if (!thread.incognito) void compactThreadJournalTail(thread.id)

        // Skill self-improvement (opt-in): on a successful run that consulted a skill, let an
        // aux model review those skills and FILE improvement suggestions (consent-first; never
        // auto-applied). Fully fire-and-forget — it must never disturb the settled run.
        if (!thread.incognito && store.settings.skillSelfImprovement && !failed && !controller.signal.aborted && viewedSkillNames.size && reviewProvider) {
          void reviewViewedSkills({ provider: reviewProvider, model: reviewModel, thread, userText: payload.userText, viewed: [...viewedSkillNames], signal: controller.signal })
        }
        if (!failed && !controller.signal.aborted) {
          try { proposePostRunMemory(thread) } catch { /* best effort — never disturb the settled run */ }
        }
      })
      // Backstop: every awaited path inside .catch/.finally is itself guarded, but attach a
      // terminal handler so no recovery/journal failure can ever escape as an unhandled rejection.
      .catch(() => {
        if (aborters.get(thread.id) === controller) aborters.delete(thread.id)
        runLease.release()
        releaseBotVmRun()
      })

    return { runId }
    } catch (err) {
      if (aborters.get(thread.id) === controller) aborters.delete(thread.id)
      runLease.release()
      releaseBotVmRun()
      throw err
    }
  }

  function applyEventToThread(thread: Thread, e: AgentEvent) {
    if (e.type === 'item_add') {
      ensureTimelineItemTrace(e.item, { runId: e.runId, threadId: thread.id, parentTraceId: e.trace?.parentTraceId ?? null, origin: e.trace?.origin ?? 'agent.event' })
      if (!thread.timeline.find((i) => i.id === e.item.id)) thread.timeline.push({ ...e.item })
    } else if (e.type === 'item_delta') {
      const it = thread.timeline.find((i) => i.id === e.id)
      if (it) {
        if (!it.trace && e.trace) it.trace = e.trace
        it[e.field] = (it[e.field] ?? '') + e.chunk
      }
    } else if (e.type === 'item_update') {
      const it = thread.timeline.find((i) => i.id === e.id)
      if (it) {
        Object.assign(it, { ...(e.trace && !it.trace ? { trace: e.trace } : {}), ...e.patch })
        if (it.kind === 'tool' && !it.eventReceipt && ['ok', 'error', 'denied'].includes(String(it.state ?? ''))) {
          it.eventReceipt = buildEventReceipt({
            kind: it.diff?.length ? 'diff_finalized' : 'tool_result_saved',
            runId: e.runId,
            threadId: thread.id,
            itemId: it.id,
            state: it.state === 'ok' ? 'ok' : it.state === 'denied' ? 'blocked' : 'error',
            summary: it.diff?.length
              ? `Diff finalized for ${it.path ?? it.title ?? it.tool ?? 'tool call'}.`
              : `Tool result saved for ${it.title ?? it.tool ?? 'tool call'}.`,
            evidenceRefs: [it.id],
          })
        }
      }
    } else if (e.type === 'thread_update') {
      Object.assign(thread, e.patch)
    } else if (e.type === 'plan') {
      thread.plan = e.plan
      if (e.goalStatus) thread.goalStatus = e.goalStatus
    } else if (e.type === 'approval_request') {
      const item = thread.timeline.find((i) => i.id === e.itemId)
      if (item && !item.trace && e.trace) item.trace = e.trace
      if (item?.tool) pendingApprovalMeta.set(e.callId, { threadId: thread.id, itemId: item.id, tool: item.tool, policy: approvalPolicyScope('tool', item.tool, item.args ?? {}) })
    }
    thread.updatedAt = Date.now()
  }

  const maybeAutoTitle = createThreadTitleGenerator({
    completeTitle: quickComplete,
    reserveOperation: () => deps.workspaceMutationGuard.reserveStart({ workspaceKeys: ['title:background'] }),
    isCurrentThread: (thread) => store.threads.includes(thread),
    persistThreads, emitToRenderer,
  })

  function clearCues(): void {
    cueQueues.clear()
    cueWaiters.clear()
  }

  return {
    addThreadNotice,
    applyEventToThread,
    cancelThreadRun,
    clearCues,
    drainThreadCues,
    guideThreadRun,
    injectThreadRouteHandoff,
    markRouteErrorRecovered,
    markUndrainedCues,
    roomRouteHandoff,
    startThreadRun,
    timelineError,
    timelineErrorSince,
    waitForThreadCue,
  }
}
