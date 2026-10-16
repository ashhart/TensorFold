import type { TensorFoldStudioApi } from '../../lib/ipc'
import type { Project, Settings, Thread } from '../../lib/types'
import type { RunStatus } from '../../lib/store/types'

const PROVIDER_ID = 'tensorfold'
const BUSY = new Set(['thinking', 'streaming', 'tool', 'awaiting'])
const reservations = new Set<string>()

export type WorkerState = 'running' | 'waiting' | 'completed' | 'review' | 'failed' | 'stopped' | 'ready'
export type WorkerStage = 'checking' | 'worktree' | 'task' | 'configuring' | 'starting'

export interface WorkerDraft {
  commandId: string
  projectId: string
  sourceCwd: string
  title: string
  objective: string
  model: string
  branch: string
  path: string | null
  threadId: string | null
}

export function workerProjectCwd(project: Project): string {
  return project.workspace?.path?.trim() || project.cwd.trim()
}

export function canUseWorkerProject(project: Project): boolean {
  return !project.archivedAt && project.executionTarget !== 'orchestrator' && project.workspace?.kind !== 'ssh' && !!workerProjectCwd(project)
}

// Managed worktrees encode the unique worker branch in their folder name, so workers
// survive restarts without putting application metadata into the user's notes.
export function isLocalWorker(thread: Thread): boolean {
  return thread.kind === 'task' && !thread.archivedAt && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh'
    && /(?:^|\/)codex(?:-|\/)worker-[a-z0-9-]+$/.test((thread.workspace?.path || thread.cwd).replace(/\\/g, '/').replace(/\/+$/, ''))
}

export function workerState(thread: Thread, run?: RunStatus, approvals = 0): WorkerState {
  if (run === 'awaiting' || approvals > 0) return 'waiting'
  if (run && BUSY.has(run)) return 'running'
  if (run === 'error' || thread.status === 'error' || thread.lastRunReceipt?.verdict === 'failed') return 'failed'
  if (thread.lastRunReceipt?.verdict === 'cancelled') return 'stopped'
  if (thread.goalStatus === 'complete' || thread.lastRunReceipt?.verdict === 'verified') return 'completed'
  if (thread.lastRunReceipt || thread.timeline.some((item) => item.kind === 'assistant')) return 'review'
  // A persisted running flag without a live status is stale after a restart.
  return 'ready'
}

export function makeWorkerDraft(project: Project, title: string, objective: string, model: string, commandId: string): WorkerDraft {
  const finalObjective = objective.trim()
  const finalTitle = title.trim() || finalObjective.split('\n')[0].slice(0, 80)
  const slug = finalTitle.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 36) || 'task'
  const unique = commandId.replace(/[^a-z0-9]/gi, '').toLowerCase().slice(0, 16)
  if (!unique) throw new Error('Could not generate a unique worker identity.')
  return {
    commandId, projectId: project.id, sourceCwd: workerProjectCwd(project),
    title: finalTitle, objective: finalObjective, model,
    branch: `codex/worker-${slug}-${unique}`, path: null, threadId: null,
  }
}

export class WorkerLaunchError extends Error {
  constructor(message: string, readonly draft: WorkerDraft) {
    super(message)
    this.name = 'WorkerLaunchError'
  }
}

type WorkerApi = Pick<TensorFoldStudioApi,
  'engineSnapshot' | 'createWorktree' | 'getWorktrees' | 'getThreads' | 'createThread'
  | 'setThreadWorkspace' | 'configureThread' | 'renameThread' | 'patchThread' | 'startRun'>

export interface WorkerLaunchDependencies {
  api: WorkerApi
  getState(): { projects: Project[]; threads: Thread[]; status: Record<string, RunStatus>; settings: Settings | null }
  onThread(thread: Thread): void
  onStarting(threadId: string): void
  onStartFailed(threadId: string): void
  onStage?(stage: WorkerStage): void
}

function errorText(error: unknown): string {
  return String((error as Error)?.message ?? error).replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '')
}

function samePath(a: string, b: string): boolean {
  return a.replace(/[\\/]+$/, '') === b.replace(/[\\/]+$/, '')
}

/** A failed setup retains its worktree and exactly-once thread identity for retry. */
export async function launchLocalWorker(original: WorkerDraft, deps: WorkerLaunchDependencies): Promise<{ draft: WorkerDraft; threadId: string; runId: string }> {
  const draft = { ...original }
  const reservation = draft.path || draft.sourceCwd
  if (reservations.has(reservation)) throw new WorkerLaunchError('Another worker is being set up in this project. Wait for it to start, then try again.', draft)
  reservations.add(reservation)
  let starting = false
  const { api } = deps
  try {
    if (!draft.objective || !draft.title || !draft.model) throw new Error('Choose a project, load a model, and describe the work first.')
    const project = deps.getState().projects.find((p) => p.id === draft.projectId)
    if (!project || !canUseWorkerProject(project)) throw new Error('Choose an active local project with a folder.')
    if (!samePath(workerProjectCwd(project), draft.sourceCwd)) throw new Error('This project folder changed. Start a new worker using its current folder.')
    if (!deps.getState().settings?.providers.some((p) => p.id === PROVIDER_ID)) throw new Error('The built-in TensorFold provider is unavailable. Reopen TensorFold and try again.')
    const checkModel = async () => {
      const snapshot = await api.engineSnapshot()
      if (!snapshot.status.available || snapshot.status.state !== 'ready' || snapshot.status.model !== draft.model) {
        throw new Error(`Load ${draft.model} in Models before starting this worker.`)
      }
    }
    deps.onStage?.('checking')
    await checkModel()
    if (!draft.path) {
      deps.onStage?.('worktree')
      const worktree = await api.createWorktree({ cwd: draft.sourceCwd, workspace: { kind: 'local', path: draft.sourceCwd }, name: draft.branch, mode: 'worktree' })
      if (!worktree.ok) throw new Error(worktree.error || 'Could not create a worktree. Check that the project is a Git repository with an initial commit.')
      if (worktree.mode !== 'worktree' || !worktree.path || samePath(worktree.path, draft.sourceCwd)) throw new Error('TensorFold did not return an isolated worktree. The worker was not started.')
      draft.path = worktree.path
      draft.branch = worktree.branch
    }
    const worktrees = await api.getWorktrees({ cwd: draft.sourceCwd, workspace: { kind: 'local', path: draft.sourceCwd } })
    if (!worktrees.worktrees.some((w) => samePath(w.path, draft.path!) && w.branch === draft.branch && !w.bare)) {
      throw new Error('The worker worktree is missing or its branch changed. Open the project to inspect it before retrying.')
    }
    const assertNotBusy = (threads: Thread[]) => {
      const live = deps.getState()
      const busy = threads.find((t) => !t.archivedAt && !t.orchestratorHostId && t.workspace?.kind !== 'ssh'
        && samePath(t.workspace?.path || t.cwd, draft.path!) && (t.status === 'running' || BUSY.has(live.status[t.id])))
      if (busy) throw new Error(busy.id === draft.threadId ? 'This worker is already running. Open it to follow its progress.' : 'Another task is using this worktree. Stop that task before retrying.')
    }
    const authoritative = await api.getThreads()
    assertNotBusy(authoritative)
    deps.onStage?.('task')
    let thread: Thread
    if (draft.threadId) {
      const existing = authoritative.find((t) => t.id === draft.threadId)
      if (!existing || existing.archivedAt) throw new Error('This worker task was removed. Start a new worker instead.')
      thread = existing
    } else {
      thread = await api.createThread(draft.path, 'task', draft.projectId, draft.commandId)
      draft.threadId = thread.id
    }
    deps.onThread(thread)
    deps.onStage?.('configuring')
    thread = await api.setThreadWorkspace(thread.id, { kind: 'local', path: draft.path }, draft.path)
    deps.onThread(thread)
    await api.configureThread(thread.id, PROVIDER_ID, draft.model, null)
    await api.renameThread(thread.id, draft.title)
    thread = await api.patchThread(thread.id, { goal: draft.objective })
    // The host owns provider/workspace state; patchThread returns its configured record.
    deps.onThread(thread)
    if (thread.providerId !== PROVIDER_ID || thread.model !== draft.model || !samePath(thread.workspace?.path || thread.cwd, draft.path)) {
      throw new Error('The worker configuration changed during setup. Open the task to inspect it before retrying.')
    }
    await checkModel()
    assertNotBusy(await api.getThreads())
    deps.onStage?.('starting')
    deps.onStarting(thread.id)
    starting = true
    const { runId } = await api.startRun({ threadId: thread.id, userText: draft.objective, approvalMode: deps.getState().settings?.approvalMode })
    return { draft, threadId: thread.id, runId }
  } catch (error) {
    if (starting && draft.threadId) {
      // A lost IPC response does not prove the run failed to start. Keep a live
      // worker visible and prevent a blind retry if the host accepted the run.
      let accepted = false
      try { accepted = (await api.getThreads()).some((t) => t.id === draft.threadId && t.status === 'running') } catch { /* Preserve the original failure when the host is unavailable. */ }
      if (accepted) throw new WorkerLaunchError('The start response was interrupted, but this worker is running. Open it to follow its progress.', draft)
      deps.onStartFailed(draft.threadId)
    }
    throw new WorkerLaunchError(errorText(error), draft)
  } finally {
    reservations.delete(reservation)
  }
}
