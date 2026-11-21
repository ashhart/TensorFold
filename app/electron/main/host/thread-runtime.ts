import { createThreadRunStarter } from './thread-run-start'
import { canContributeThreadMemory } from '../../../shared/thread-memory-policy'
import { assertGoogleContextTransport } from '../../../shared/google-workspace-privacy'
import { randomUUID } from 'node:crypto'
import type { JobPatch } from '../../agent/jobs'
import { vontraPath } from '../../store'
import { buildUserContent, contentText } from '../../agent/content'
import { normalizeAttachments } from '../../agent/attachment-policy'
import { compactThreadJournal, readThreadJournal } from '../../agent/thread-journal'
import { quickComplete, streamChat } from '../../agent/openai'
import { createThreadTitleGenerator } from './thread-title'
import { harvestPostRunMemoryCandidates } from '../../agent/memory-harvest'
import { reviewSkillsAfterRun } from '../../agent/skill-self-improvement'
import { addSuggestion } from '../../agent/suggestions'
import { createProjectMemoryCandidate, scopeProjectMemoryEntries } from '../../agent/project-memory'
import { projectMemoryAutoCandidateTurn, projectMemoryAutoMessages } from './project-memory-auto-policy'
import { persistProjectMemoryCandidates, persistSuggestions } from './persistence'
import { ensureTimelineItemTrace } from '../../agent/trace'
import { buildEventReceipt } from '../../agent/run-contract'
import type { AgentCue, AgentEvent, ApiMessage, CueRunPayload, CueRunResult, Job, Project, Provider, Room, Seat, StartRunPayload, Thread, TimelineItem, WorkspaceLocation } from '../../agent/types'
import { approvalPolicyScope, type ApprovalPolicyScope } from '../../agent/approval-policy'
import { store } from '../state'
import { requireSecurityRole } from './audit'
import { routeHandoffPreview } from './provider-runtime'
import { type WorkspaceMutationGuard } from './workspace-mutation-guard'
import { botRunApprovalMode } from '../../../shared/bot-policy'
import { assertBotLifecycleIdle } from './bot-run-boundary'

export { encodeJournalEvent, decodeJournalEvent } from './thread-journal-crypto'

export interface JobHandle {
  get(): Job | null
  set(job: Job | null): void
}

/** Main-process admission only; renderer/model payloads never supply these hooks. */
export interface ThreadRunAdmission {
  assertAllowed(): void
  onController(controller: AbortController): void
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
  startThreadRun(payload: StartRunPayload, existingJob?: Job, admission?: ThreadRunAdmission): Promise<{ runId: string }>
  timelineError(errorItem: TimelineItem): Error
  timelineErrorSince(thread: Thread, timelineStart: number): TimelineItem | null
  waitForThreadCue(threadId: string, timeoutMs: number, signal: AbortSignal): Promise<void>
}

export interface ThreadRuntimeDeps {
  aborters: Map<string, AbortController>
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'assertThreadStartAllowed' | 'reserveStart'>
  assertLocalEngineAdmission?: () => void
  pendingApprovalMeta: Map<string, { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }>
  jobLeaseOwner: string
  threadJobHeartbeatMs: number
  sshManager: {
    ensureConnected(hostId: string): Promise<unknown>
    status?(hostId: string): { phase: string; lastError?: string | null }
  }
  emitToRenderer(event: AgentEvent): void
  persistThreads(): void
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal; memoryPolicy?: import('../../agent/types').ThreadMemoryPolicy }): any
  approvalResolver(controller: AbortController, jobHandle?: JobHandle): (callId: string) => Promise<'allow' | 'allow_always' | 'deny'>
  isHighRiskTool(name: string): boolean
  claimJob(ref: Parameters<typeof import('../../agent/jobs').enqueueJob>[1]): Job
  patchJob(job: Job | null, patch: JobPatch): Job | null
  replaceJobResult(result: { jobs: Job[]; job: Job | null }): Job
  heartbeatJob(job: Job | null): Job | null
  settleThreadRunJob(job: Job | null, thread: Thread, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  renderProjectMemoryBucket(project: { id: string; name: string } | undefined, query?: string): string
  effectiveContextLimit(provider: Provider, model: string | null, accountId?: string | null): number
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

export { recordPreTurnCheckpoint, resolveThreadRunCwd, assertSshThreadTransportSupported, assertSshToolContextConnected, assertIncognitoThreadTransportSupported } from './thread-run-policy'

export function createThreadRuntime(deps: ThreadRuntimeDeps): ThreadRuntime {
  const {
    aborters,
    pendingApprovalMeta,
    emitToRenderer,
    persistThreads,
  } = deps
  const cueQueues = new Map<string, AgentCue[]>()
  const cueWaiters = new Map<string, Array<() => void>>()

  function enqueueThreadCue(thread: Thread, text: string): CueRunResult {
    const clean = text.trim()
    if (!clean || !aborters.has(thread.id)) return { queued: false, queueDepth: cueQueues.get(thread.id)?.length ?? 0 }
    const item = { id: randomUUID(), kind: 'cue' as const, memorySource: 'user' as const, title: 'Guide', text: clean, state: 'pending' as const, meta: 'queued', createdAt: Date.now() }
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
    if ('quotes' in payload) throw new Error('Quotes stay in the saved draft until the current run finishes.')
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
    const jsonStart = message.indexOf(': {')
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
  async function reviewViewedSkills(args: { provider: Provider; model: string; thread: Thread; userText: string; viewed: string[]; signal: AbortSignal; memoryPolicy: import('../../agent/types').ThreadMemoryPolicy; memoryRevision: number }): Promise<void> {
    const allowed = () => !args.thread.googleLocalOnly && store.threads.includes(args.thread) && canContributeThreadMemory(args.thread, args.memoryPolicy, args.memoryRevision, args.signal)
    if (!allowed()) return
    try {
      const { provider, model } = args
      const digest = `User request: ${(args.userText || args.thread.title || '').slice(0, 600)}\nOutcome: the run completed successfully.`
      const complete = async (messages: { role: string; content: string }[]): Promise<string> => {
        if (!allowed()) throw new Error('Memory proposal review is no longer allowed.')
        let out = ''
        for await (const ch of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages: messages as ApiMessage[], temperature: 0.2, approvalMode: 'readonly', cwd: args.thread.cwd, signal: args.signal,
          assertRequestAllowed: () => { if (!allowed()) throw new Error('Memory proposal review is no longer allowed.'); assertGoogleContextTransport(args.thread, provider) } })) {
          if (ch.type === 'content') out += ch.text
        }
        return out
      }
      const proposals = await reviewSkillsAfterRun({ skills: store.skills, viewedNames: args.viewed, digest, complete, sourceThreadId: args.thread.id, max: 2 })
      if (!allowed() || !proposals.length) return
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

  function proposePostRunMemory(thread: Thread, memoryPolicy: import('../../agent/types').ThreadMemoryPolicy, memoryRevision: number, signal: AbortSignal): void {
    if (thread.googleLocalOnly || !store.threads.includes(thread) || !canContributeThreadMemory(thread, memoryPolicy, memoryRevision, signal) || !thread.projectId) return
    const activeEntries = scopeProjectMemoryEntries(store.projectMemory, thread.projectId)
    const existingTexts = [
      ...activeEntries.map((entry) => entry.text),
      ...store.projectMemoryCandidates.filter((candidate) => candidate.projectId === thread.projectId && candidate.status === 'pending' && (!candidate.targetEntryId || activeEntries.some((entry) => entry.id === candidate.targetEntryId))).map((candidate) => candidate.text),
    ]
    const authored = projectMemoryAutoMessages(store.projectMemory, thread)
    const candidates = harvestPostRunMemoryCandidates(authored, { existingTexts, cap: 3 })
    if (!candidates.length) return
    for (const text of candidates) {
      store.projectMemoryCandidates = [
        // Provenance (VST-009): the review UI links each proposal back to the exact
        // conversation turn that surfaced the pattern.
        createProjectMemoryCandidate({ projectId: thread.projectId, text, sourceType: 'agent', reason: 'Proposed after a completed run.', sourceThreadId: thread.id, sourceTurn: projectMemoryAutoCandidateTurn(authored, text) }),
        ...store.projectMemoryCandidates,
      ]
    }
    persistProjectMemoryCandidates()
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
    isCurrentThread: (thread) => !thread.googleLocalOnly && store.threads.includes(thread),
    persistThreads, emitToRenderer,
  })

  function clearCues(): void {
    cueQueues.clear()
    cueWaiters.clear()
  }

  const startThreadRun = createThreadRunStarter(deps, {
    addThreadNotice,
    applyEventToThread,
    drainThreadCues,
    waitForThreadCue,
    timelineErrorSince,
    timelineError,
    markRouteErrorRecovered,
    markUndrainedCues,
    injectThreadRouteHandoff,
    availableModelsFromProviderError,
    compactThreadJournalTail,
    reviewViewedSkills,
    proposePostRunMemory,
    maybeAutoTitle,
  })

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
