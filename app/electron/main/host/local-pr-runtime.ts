import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { addMissionCheckpoint, completeMission, createMission } from '../../agent/missions'
import { buildLocalPrBuilderResumePrompt, buildLocalPrLoopDraft, buildLocalPrLoopSummary, buildLocalPrReviewPrompt, extractLocalPrReviewOutcome, localPrApprovalEvidence, localPrRemotePublishGate, shouldAutoStartLocalPrReview, shouldResumeBuilderFromReview } from '../../agent/pr-loop'
import { publishLocalPr, nodeExecFilePrRunner } from '../../agent/pr-publish'
import { selectProviderAccount } from '../../agent/provider-accounts'
import { auditPreview } from '../../agent/security'
import type { JobPatch } from '../../agent/jobs'
import type { CreatePrLoopInput, CreatePrLoopResult, Job, Mission, Project, Provider, Thread } from '../../agent/types'
import type { ModelRouteDecision } from '../../agent/model-router'
import { safeSegment, vontraPath, writeTensorFoldStudioJson } from '../../store'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { auditScopedModelRoute, modelRouteAutoFailoverEnabled, modelRouteFailoverText, modelRouteMemoryForDecision, modelRouter, resolveProjectProvider, selectedProviderAccountId } from './provider-runtime'
import { inspectGitRepository, qq, resolveWorkspaceCwd, shStrict } from './workspace'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, type WorkspaceMutationGuard } from './workspace-mutation-guard'

type ThreadRunStarter = (payload: { threadId: string; userText: string }, existingJob?: Job) => Promise<{ runId: string }>

export interface LocalPrRuntime {
  createLocalPrLoop(input: CreatePrLoopInput): Promise<CreatePrLoopResult>
  settleThreadRunJob(job: Job | null, thread: Thread, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
}

interface LocalPrRuntimeDeps {
  persistThreads(): void
  persistMissions(): void
  emitMission(mission: Mission): void
  rememberJob(ref: Parameters<typeof import('../../agent/jobs').enqueueJob>[1]): Job
  patchJob(job: Job | null, patch: JobPatch): Job | null
  finishJob(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  replaceMission(next: Mission): Mission
  startThreadRun: ThreadRunStarter
  effectiveContextLimit(provider: Provider, model: string | null, accountId?: string | null): number
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'reserveStart'>
}

const TERMINAL_MISSION_STATES = new Set(['completed', 'failed', 'cancelled'])

/** Local PR loops create a local git worktree and launch local thread runtimes. */
export function assertLocalPrProjectSupported(project: Pick<Project, 'workspace' | 'archivedAt'> | null | undefined): void {
  if (project?.archivedAt) throw new Error('Restore the archived project before creating a local PR loop.')
  if (project?.workspace?.kind === 'ssh') {
    throw new Error('Local PR loops do not support SSH workspaces; create the PR loop on the remote Orchestrator.')
  }
}

export function createLocalPrRuntime(deps: LocalPrRuntimeDeps): LocalPrRuntime {
  const {
    persistThreads,
    persistMissions,
    emitMission,
    rememberJob,
    patchJob,
    finishJob,
    replaceMission,
    startThreadRun,
    effectiveContextLimit,
    workspaceMutationGuard,
  } = deps

async function createLocalPrLoop(input: CreatePrLoopInput): Promise<CreatePrLoopResult> {
  requireSecurityRole('operator', 'create local PR loops')
  const project = input.projectId ? store.projects.find((candidate) => candidate.id === input.projectId) : null
  if (input.projectId && !project) throw new Error('Project not found.')
  assertLocalPrProjectSupported(project)
  // Resolve all provider/model/local-only policy before mkdir/git worktree mutation. A bad route
  // must not leave an orphan branch or worktree behind.
  const resolved = resolvePrLoopProvider(project, input)
  const routeMemory = resolved.initialRoute.audit.action === 'failover'
    ? modelRouteMemoryForDecision(resolved.baseProvider, resolved.initialRoute)
    : null
  const routeNotice = routeMemory ? modelRouteFailoverText(resolved.baseProvider, resolved.initialRoute.audit) : null
  const selectedCwd = input.cwd || project?.cwd || store.settings.defaultCwd || ''
  if (!workspaceMutationGuard) throw new Error('Cannot create a local PR worktree because the workspace mutation guard is unavailable.')
  const worktreeLease = workspaceMutationGuard.reserveStart({ workspaceKeys: [lexicalLocalWorkspaceKey(selectedCwd)] })
  try {
  const workspace = await resolveWorkspaceCwd(selectedCwd, 'local PR workspace')
  worktreeLease.addWorkspaceKeys?.([await canonicalLocalWorkspaceKey(workspace)])
  const repo = await inspectGitRepository(workspace, 'local PR workspace')
  if (!repo.exists || !repo.topLevel) throw new Error('This folder is not initialized as a Git repo. Initialize Git before creating a local PR loop.')
  if (!repo.hasCommits) throw new Error('This Git repo has no commits yet. Make an initial commit before creating a local PR loop.')

  const localPrId = `lpr_${randomUUID().slice(0, 8)}`
  const title = String(input.title || input.objective || 'local-pr').trim()
  const baseRef = String(input.baseBranch || 'HEAD').trim()
  const baseCommit = (await shStrict(`git -C ${qq(repo.topLevel)} rev-parse --verify ${qq(baseRef)}`, repo.topLevel, 10_000)).trim()
  const baseLabel = input.baseBranch?.trim() || (repo.branch ? `${repo.branch} (${baseCommit.slice(0, 8)})` : baseCommit.slice(0, 8))
  const branch = `vontra/pr/${localPrId}`
  const root = vontraPath('pr-loops', `${safeSegment(title || 'local-pr')}-${localPrId}`)
  const worktree = path.join(root, 'worktree')
  worktreeLease.addWorkspaceKeys?.([lexicalLocalWorkspaceKey(repo.topLevel), lexicalLocalWorkspaceKey(worktree)])
  const existing = await fs.stat(worktree).catch(() => null)
  if (existing) throw new Error(`Local PR worktree already exists: ${worktree}`)
  await fs.mkdir(root, { recursive: true })
  await auditEvent('local_pr_loop.worktree_create', { localPrId, source: auditPreview(repo.topLevel), worktree: auditPreview(worktree), branch, base: auditPreview(baseLabel) })
  await shStrict(`git -C ${qq(repo.topLevel)} worktree add -b ${qq(branch)} ${qq(worktree)} ${qq(baseCommit)}`, repo.topLevel, 30_000)
  worktreeLease.addWorkspaceKeys?.([await canonicalLocalWorkspaceKey(worktree)])

  const draft = buildLocalPrLoopDraft(input, {
    localPrId,
    projectName: project?.name ?? null,
    cwd: workspace,
    worktree,
    branch,
    baseBranch: baseLabel,
  })
  const now = Date.now()
  const implementationThread: Thread = {
    documentSourceIds: [],
    id: randomUUID(),
    title: draft.implementationTitle,
    kind: 'task',
    projectId: project?.id ?? input.projectId ?? null,
    cwd: worktree,
    providerId: resolved.baseProvider.id,
    providerAccountId: resolved.providerAccountId,
    modelRoute: routeMemory ? { ...routeMemory } : null,
    model: resolved.model,
    createdAt: now,
    updatedAt: now,
    status: 'idle',
    apiMessages: [],
    timeline: [
      { id: randomUUID(), kind: 'notice', text: `Local PR ${localPrId} is ready on ${branch}.`, createdAt: now },
      ...(routeNotice ? [{ id: randomUUID(), kind: 'notice' as const, text: routeNotice, createdAt: now }] : []),
    ],
    goal: draft.implementationGoal,
    goalStatus: 'active',
    plan: draft.implementationPlan,
  }
  const reviewThread: Thread = {
    documentSourceIds: [],
    id: randomUUID(),
    title: draft.reviewTitle,
    kind: 'task',
    projectId: project?.id ?? input.projectId ?? null,
    cwd: worktree,
    providerId: resolved.baseProvider.id,
    providerAccountId: resolved.providerAccountId,
    modelRoute: routeMemory ? { ...routeMemory } : null,
    model: resolved.model,
    createdAt: now,
    updatedAt: now,
    status: 'idle',
    apiMessages: [],
    timeline: [
      { id: randomUUID(), kind: 'notice', text: `Waiting to review ${branch} after the implementation thread posts its local PR summary.`, createdAt: now },
      ...(routeNotice ? [{ id: randomUUID(), kind: 'notice' as const, text: routeNotice, createdAt: now }] : []),
    ],
    goal: draft.reviewGoal,
    goalStatus: 'active',
    plan: draft.reviewPlan,
  }
  store.threads = [implementationThread, reviewThread, ...store.threads]
  persistThreads()

  let mission = createMission({
    title: `Local PR loop: ${draft.title}`,
    objective: draft.objective,
    stopCondition: draft.stopCondition,
    repeatMode: 'until_condition',
    contextMode: 'build_on_previous',
    roleProfileId: 'builder',
    fanOutBudget: 'paired',
    maxRuns: null,
    costCapUsd: null,
    destructiveApprovalRequired: true,
    runBudgetClass: 'mission',
    threadId: implementationThread.id,
    projectId: project?.id ?? input.projectId ?? null,
    cwd: worktree,
    providerId: resolved.baseProvider.id,
    providerAccountId: resolved.providerAccountId,
    model: resolved.model,
    metadata: {
      workflow: 'local_pr_loop',
      localPrId,
      implementationThreadId: implementationThread.id,
      reviewThreadId: reviewThread.id,
      modelRoute: routeMemory ? { action: 'local_pr_initial_failover', ...routeMemory } : null,
      branch,
      baseBranch: baseLabel,
      baseCommit,
      worktree,
      requiredApprovals: draft.requiredApprovals,
      remotePublish: localPrRemotePublishGate(),
    },
  })
  mission = addMissionCheckpoint(mission, {
    label: 'Local PR worktree',
    summary: `Created ${branch} at ${worktree}.`,
    data: { localPrId, branch, baseCommit, worktree },
  })
  mission = addMissionCheckpoint(mission, {
    label: 'Review loop threads',
    summary: `Created implementation and review threads for ${branch}.`,
    data: { implementationThreadId: implementationThread.id, reviewThreadId: reviewThread.id },
  })
  store.missions.unshift(mission)
  persistMissions()
  emitMission(mission)
  let prLoopJob = rememberJob({
    kind: 'pr_loop',
    owner: localPrId,
    projectId: project?.id ?? input.projectId ?? null,
    threadId: implementationThread.id,
    currentThreadId: implementationThread.id,
    resumeAfterRestart: 'ask',
    metadata: {
      workflow: 'local_pr_loop',
      localPrId,
      missionId: mission.id,
      implementationThreadId: implementationThread.id,
      reviewThreadId: reviewThread.id,
      branch,
      baseBranch: baseLabel,
      worktree,
      title: draft.title,
      requiredApprovals: draft.requiredApprovals,
      publishRemote: input.publishRemote === true,
      remotePublish: localPrRemotePublishGate(),
      modelRoute: routeMemory ? { action: 'local_pr_initial_failover', ...routeMemory } : null,
    },
  })
  await writeTensorFoldStudioJson(['pr-loops', `${localPrId}.json`], {
    localPrId,
    branch,
    baseBranch: baseLabel,
    baseCommit,
    worktree,
    projectId: project?.id ?? input.projectId ?? null,
    missionId: mission.id,
    implementationThreadId: implementationThread.id,
    reviewThreadId: reviewThread.id,
    requiredApprovals: draft.requiredApprovals,
    providerId: resolved.baseProvider.id,
    providerAccountId: resolved.providerAccountId,
    model: resolved.model,
    modelRoute: routeMemory ? { action: 'local_pr_initial_failover', ...routeMemory } : null,
    remotePublish: localPrRemotePublishGate(),
    createdAt: now,
  }).catch(() => {})
  if (routeMemory) {
    await auditScopedModelRoute('local_pr', resolved.initialRoute.audit, null, {
      localPrId,
      implementationThreadId: implementationThread.id,
      reviewThreadId: reviewThread.id,
      worktree: auditPreview(worktree),
    })
  }
  await auditEvent('local_pr_loop.create', { localPrId, branch, projectId: project?.id ?? input.projectId ?? null, worktree: auditPreview(worktree), requiredApprovals: draft.requiredApprovals })
  if (input.autoStart !== false) {
    await startThreadRun({ threadId: implementationThread.id, userText: draft.implementationGoal }, prLoopJob)
  } else {
    prLoopJob = patchJob(prLoopJob, { status: 'paused', resumePlan: 'Start the local PR implementation thread when ready.' }) ?? prLoopJob
  }
  return { localPrId, branch, baseBranch: baseLabel, worktree, mission, implementationThread, reviewThread }
  } finally {
    worktreeLease.release()
  }
}

function resolvePrLoopProvider(project: Project | null | undefined, input: CreatePrLoopInput): { baseProvider: Provider; requestedProviderAccountId: string | null; providerAccountId: string | null; model: string; initialRoute: ModelRouteDecision } {
  const requestedProvider = input.providerId ? store.settings.providers.find((provider) => provider.id === input.providerId) : null
  const baseProvider = requestedProvider ?? resolveProjectProvider(project, input.providerAccountId, input.model).baseProvider
  if (!baseProvider) throw new Error('Configure a provider and model first.')
  if (project?.localOnly && baseProvider.kind !== 'local') throw new Error('This project is local-only. Choose a local provider for the local PR loop.')
  const requestedProviderAccountId = selectedProviderAccountId(baseProvider, input.providerAccountId ?? project?.defaultProviderAccountId ?? store.settings.activeProviderAccountId ?? baseProvider.defaultAccountId ?? null)
  const account = selectProviderAccount(baseProvider, requestedProviderAccountId)
  const choices = account?.models?.length ? account.models : baseProvider.models
  const model = input.model || (project?.defaultModel && choices.includes(project.defaultModel) ? project.defaultModel : (choices[0] ?? store.settings.activeModel ?? ''))
  if (!model) throw new Error(`Choose a model for ${baseProvider.name} before creating a local PR loop.`)
  const initialRoute = modelRouter.select({
    role: 'local_pr',
    provider: baseProvider,
    model,
    requestedAccountId: requestedProviderAccountId,
    allowFailover: modelRouteAutoFailoverEnabled(baseProvider, requestedProviderAccountId),
    localOnly: project?.localOnly === true,
    contextLimit: effectiveContextLimit(baseProvider, model, requestedProviderAccountId),
    reasoningEffort: store.settings.reasoningEffort,
    budget: 'local_pr',
    source: 'local_pr_initial_route',
  })
  if (!initialRoute.selected) throw new Error(initialRoute.audit.reason)
  return { baseProvider, requestedProviderAccountId, providerAccountId: initialRoute.selectedAccountId, model, initialRoute }
}


function stringJobMetadata(job: Job | null, key: string): string | null {
  const value = job?.metadata?.[key]
  return typeof value === 'string' && value ? value : null
}

// Opt-in flag (default false): only when a PR loop was created with publishRemote do we push +
// open a hosted draft PR after reviewer approval. Accepts boolean true or the string 'true'.
function booleanJobMetadata(job: Job | null, key: string): boolean {
  const value = job?.metadata?.[key]
  return value === true || value === 'true'
}

// The body for the GitHub draft PR: a short, honest note that TensorFold's local PR loop produced
// and reviewer-approved this change, plus the review summary evidence.
function buildRemotePrBody(evidence: { summary?: string }, review: { summary?: string; comments?: unknown[] }): string {
  const lines = ['Opened by TensorFold after the local PR loop reviewer approved the recorded diff/check evidence.', '']
  if (review.summary?.trim()) lines.push(review.summary.trim(), '')
  if (evidence.summary?.trim()) lines.push(evidence.summary.trim())
  return lines.join('\n').trim()
}

function numberJobMetadata(job: Job | null, key: string): number | null {
  const value = job?.metadata?.[key]
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function attachLocalPrSummaryToReview(job: Job, implementationThread: Thread, reviewThreadId: string) {
  const localPrId = stringJobMetadata(job, 'localPrId') ?? job.owner
  const branch = stringJobMetadata(job, 'branch') ?? 'local PR branch'
  const baseBranch = stringJobMetadata(job, 'baseBranch')
  const worktree = stringJobMetadata(job, 'worktree') ?? implementationThread.cwd
  const summary = buildLocalPrLoopSummary(implementationThread, {
    localPrId,
    branch,
    baseBranch,
    worktree,
    implementationThreadId: implementationThread.id,
    reviewThreadId,
  })
  const now = Date.now()
  const meta = `${summary.diffRefs.length} diff${summary.diffRefs.length === 1 ? '' : 's'} · ${summary.checkRefs.length} check${summary.checkRefs.length === 1 ? '' : 's'}`
  const existingSummary = implementationThread.timeline.some((item) => item.tool === 'local_pr_summary' && item.meta === meta)
  if (!existingSummary) {
    implementationThread.timeline.push({
      id: randomUUID(),
      kind: 'notice',
      tool: 'local_pr_summary',
      title: 'Local PR summary posted for review',
      text: summary.markdown,
      state: summary.readyForReview ? 'ok' : 'awaiting',
      meta,
      createdAt: now,
    })
    implementationThread.updatedAt = now
  }

  const reviewThread = store.threads.find((candidate) => candidate.id === reviewThreadId)
  if (reviewThread) {
    reviewThread.timeline.push({
      id: randomUUID(),
      kind: 'notice',
      tool: 'local_pr_summary',
      title: 'Implementation summary ready',
      text: summary.markdown,
      state: summary.readyForReview ? 'ok' : 'awaiting',
      meta,
      createdAt: now,
    })
    if (!reviewThread.goal?.includes('## Local PR Summary:')) {
      reviewThread.goal = `${reviewThread.goal ?? ''}\n\nImplementation summary:\n${summary.markdown}`.trim()
    }
    reviewThread.updatedAt = now
  }

  patchJob(job, {
    metadata: {
      localPrSummary: {
        readyForReview: summary.readyForReview,
        changedFiles: summary.changedFiles,
        diffRefs: summary.diffRefs.map((ref) => ({ itemId: ref.itemId, href: ref.href, path: ref.path })),
        checkRefs: summary.checkRefs.map((ref) => ({ itemId: ref.itemId, href: ref.href, command: ref.command, passed: ref.passed })),
        risks: summary.risks,
      },
    },
  })
  return summary
}

function prLoopReviewThreadAfterImplementation(job: Job | null, completedThreadId: string): string | null {
  if (!job || job.kind !== 'pr_loop') return null
  const implementationThreadId = stringJobMetadata(job, 'implementationThreadId')
  const reviewThreadId = stringJobMetadata(job, 'reviewThreadId')
  if (!implementationThreadId || !reviewThreadId || implementationThreadId !== completedThreadId) return null
  return reviewThreadId
}

function prLoopImplementationThreadAfterReview(job: Job | null, completedThreadId: string): string | null {
  if (!job || job.kind !== 'pr_loop') return null
  const implementationThreadId = stringJobMetadata(job, 'implementationThreadId')
  const reviewThreadId = stringJobMetadata(job, 'reviewThreadId')
  if (!implementationThreadId || !reviewThreadId || reviewThreadId !== completedThreadId) return null
  return implementationThreadId
}

function addLocalPrReviewGuideItem(builderThread: Thread, text: string): void {
  const now = Date.now()
  builderThread.timeline.push({
    id: randomUUID(),
    kind: 'cue',
    title: 'Reviewer comments',
    text,
    state: 'pending',
    meta: 'local PR review',
    createdAt: now,
  })
  builderThread.updatedAt = now
}

function recordLocalPrApprovalMissionEvidence(job: Job, reviewThread: Thread, evidence: ReturnType<typeof localPrApprovalEvidence>): void {
  const missionId = stringJobMetadata(job, 'missionId') ?? job.missionId
  if (!missionId) return
  const mission = store.missions.find((candidate) => candidate.id === missionId)
  if (!mission || TERMINAL_MISSION_STATES.has(mission.status)) return
  const next = addMissionCheckpoint(mission, {
    label: 'Reviewer approval',
    summary: `${evidence.summary} Local PR remains local-only; remote publish is not approved by this gate.`,
    status: mission.status,
    data: {
      workflow: 'local_pr_loop',
      localPrId: stringJobMetadata(job, 'localPrId') ?? job.owner,
      reviewThreadId: reviewThread.id,
      approvalEvidence: evidence,
      remotePublish: localPrRemotePublishGate(),
    },
  })
  replaceMission(next)
}

function completeLocalPrMission(job: Job, message: string): void {
  const missionId = stringJobMetadata(job, 'missionId') ?? job.missionId
  if (!missionId) return
  const mission = store.missions.find((candidate) => candidate.id === missionId)
  if (!mission || TERMINAL_MISSION_STATES.has(mission.status)) return
  replaceMission(completeMission(mission, message))
}

function settleThreadRunJob(job: Job | null, thread: Thread, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void {
  const reviewThreadId = outcome === 'completed' ? prLoopReviewThreadAfterImplementation(job, thread.id) : null
  if (job?.kind === 'pr_loop' && reviewThreadId) {
    const summary = attachLocalPrSummaryToReview(job, thread, reviewThreadId)
    const autoStartReview = shouldAutoStartLocalPrReview(summary)
    const reviewPrompt = buildLocalPrReviewPrompt(summary)
    const nextJob = patchJob(job, {
      status: autoStartReview ? 'queued' : 'paused',
      currentRunId: null,
      currentThreadId: reviewThreadId,
      nextWakeAt: null,
      lastUsefulProgressAt: Date.now(),
      blockedReason: autoStartReview ? null : 'Implementation thread finished, but diff/check evidence is incomplete.',
      resumePlan: autoStartReview ? 'Review thread auto-started from the local PR summary evidence.' : 'Start the local PR review thread after the builder posts missing diff/check evidence.',
      leaseOwner: null,
      leaseExpiresAt: null,
      metadata: {
        phase: autoStartReview ? 'review_auto_starting' : 'awaiting_review',
        lastImplementationThreadId: thread.id,
        reviewThreadId,
      },
    })
    if (autoStartReview && nextJob) {
      void startThreadRun({ threadId: reviewThreadId, userText: reviewPrompt }, nextJob).catch((err) => {
        const message = String((err as any)?.message ?? err)
        patchJob(nextJob, {
          status: 'paused',
          blockedReason: `Automatic review start failed: ${message}`,
          resumePlan: 'Start the local PR review thread manually from the implementation summary.',
          leaseOwner: null,
          leaseExpiresAt: null,
          currentRunId: null,
          metadata: { phase: 'awaiting_review', autoReviewError: message },
        })
      })
    }
    return
  }
  const implementationThreadId = outcome === 'completed' ? prLoopImplementationThreadAfterReview(job, thread.id) : null
  if (job?.kind === 'pr_loop' && implementationThreadId) {
    const reviewOutcome = extractLocalPrReviewOutcome(thread)
    const approvalEvidence = localPrApprovalEvidence(reviewOutcome, numberJobMetadata(job, 'requiredApprovals') ?? 1)
    const remotePublish = localPrRemotePublishGate()
    const builderThread = store.threads.find((candidate) => candidate.id === implementationThreadId)
    const metadata = {
      phase: approvalEvidence.approved ? 'approved' : shouldResumeBuilderFromReview(reviewOutcome) ? 'builder_followup' : reviewOutcome.approved ? 'awaiting_required_approvals' : 'review_needs_triage',
      lastReviewThreadId: thread.id,
      localPrReviewOutcome: {
        approved: reviewOutcome.approved,
        comments: reviewOutcome.comments,
        approvals: reviewOutcome.approvals,
        sourceItemIds: reviewOutcome.sourceItemIds,
        summary: reviewOutcome.summary,
      },
      localPrApprovalEvidence: approvalEvidence,
      localPrRemotePublish: remotePublish,
    }
    if (approvalEvidence.approved) {
      recordLocalPrApprovalMissionEvidence(job, thread, approvalEvidence)
      // Remote publish (opt-in via the loop's publishRemote metadata, default off): the reviewer
      // approval IS the approval. The local loop is DONE regardless, so settle it synchronously
      // and — when publish is requested — push + open a GitHub draft PR in the background,
      // recording the PR URL (or error) into job metadata when it resolves. A publish failure
      // never fails the already-approved local loop.
      const publishRequested = booleanJobMetadata(job, 'publishRemote')
      const publishGate = localPrRemotePublishGate({ requested: publishRequested, approved: true })
      const prBranch = stringJobMetadata(job, 'branch')
      const prWorktree = stringJobMetadata(job, 'worktree')
      const willPublish = publishGate.allowed && !!prBranch && !!prWorktree
      completeLocalPrMission(job, willPublish ? 'Local PR loop completed after reviewer approval; opening a remote draft PR.' : 'Local PR loop completed after reviewer approval evidence was recorded.')
      patchJob(job, { metadata: { ...metadata, localPrRemotePublish: publishGate } })
      finishJob(job, 'completed', willPublish
        ? 'Local PR reviewer approved; opening a hosted draft PR…'
        : publishRequested
          ? 'Local PR reviewer approved; remote publish skipped (missing branch/worktree).'
          : 'Local PR reviewer approved the recorded diff/check evidence; no remote publish was requested.')
      if (willPublish) {
        const prTitle = stringJobMetadata(job, 'title') || `TensorFold PR ${prBranch}`
        const recordPublish = (extra: Record<string, unknown>) =>
          patchJob(job, { metadata: { ...metadata, localPrRemotePublish: { ...publishGate, ...extra } } })
        void publishLocalPr(
          { branch: prBranch as string, baseBranch: null, title: prTitle, bodyMarkdown: buildRemotePrBody(approvalEvidence, reviewOutcome), worktree: prWorktree as string, draft: true },
          publishGate,
          nodeExecFilePrRunner(),
        )
          .then((result) => recordPublish({ publishOk: result.ok, prUrl: result.prUrl ?? null, publishError: result.error ?? null }))
          .catch((e: any) => recordPublish({ publishOk: false, prUrl: null, publishError: e?.message ?? String(e) }))
      }
      return
    }
    if (reviewOutcome.approved && approvalEvidence.missingApprovals > 0) {
      const passNumber = approvalEvidence.recordedApprovals + 1
      const reviewPrompt = [
        `Run independent review pass ${passNumber}/${approvalEvidence.requiredApprovals}.`,
        'Re-open the linked diff and check evidence with tools; do not rely on an earlier pass.',
        'Leave actionable comments or explicitly approve only after the evidence has been inspected in this run.',
      ].join(' ')
      const nextJob = patchJob(job, {
        status: 'queued',
        currentRunId: null,
        currentThreadId: thread.id,
        nextWakeAt: null,
        lastUsefulProgressAt: Date.now(),
        blockedReason: null,
        resumePlan: `Starting evidence-bound review pass ${passNumber}/${approvalEvidence.requiredApprovals}.`,
        leaseOwner: null,
        leaseExpiresAt: null,
        metadata,
      })
      if (nextJob) {
        void startThreadRun({ threadId: thread.id, userText: reviewPrompt }, nextJob).catch((err) => {
          const message = String((err as any)?.message ?? err)
          patchJob(nextJob, {
            status: 'paused',
            blockedReason: `Automatic review pass ${passNumber} failed: ${message}`,
            resumePlan: `Resume review pass ${passNumber}/${approvalEvidence.requiredApprovals} manually.`,
            leaseOwner: null,
            leaseExpiresAt: null,
            currentRunId: null,
            metadata: { phase: 'awaiting_required_review_passes', autoReviewError: message },
          })
        })
      }
      return
    }
    if (builderThread && shouldResumeBuilderFromReview(reviewOutcome)) {
      const guideText = buildLocalPrBuilderResumePrompt(reviewOutcome)
      addLocalPrReviewGuideItem(builderThread, guideText)
      const nextJob = patchJob(job, {
        status: 'queued',
        currentRunId: null,
        currentThreadId: implementationThreadId,
        nextWakeAt: null,
        lastUsefulProgressAt: Date.now(),
        blockedReason: null,
        resumePlan: 'Builder resumed from reviewer Guide comments.',
        leaseOwner: null,
        leaseExpiresAt: null,
        metadata,
      })
      if (nextJob) {
        void startThreadRun({ threadId: implementationThreadId, userText: guideText }, nextJob).catch((err) => {
          const message = String((err as any)?.message ?? err)
          patchJob(nextJob, {
            status: 'paused',
            blockedReason: `Builder resume from review failed: ${message}`,
            resumePlan: 'Resume the builder manually from the reviewer Guide comments.',
            leaseOwner: null,
            leaseExpiresAt: null,
            currentRunId: null,
            metadata: { phase: 'builder_followup', builderResumeError: message },
          })
        })
      }
      return
    }
    patchJob(job, {
      status: 'paused',
      currentRunId: null,
      currentThreadId: implementationThreadId,
      nextWakeAt: null,
      lastUsefulProgressAt: Date.now(),
      blockedReason: reviewOutcome.approved
        ? approvalEvidence.summary
        : 'Review completed without explicit approval or actionable comments.',
      resumePlan: reviewOutcome.approved
        ? 'Collect the remaining evidence-bound review pass, then resume the local PR loop.'
        : 'Guide the builder or reviewer manually with the missing decision.',
      leaseOwner: null,
      leaseExpiresAt: null,
      metadata,
    })
    return
  }
  finishJob(job, outcome, reason)
}

  return { createLocalPrLoop, settleThreadRunJob }
}
