import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useRef, useState } from 'react'
import clsx from 'clsx'
import { api } from '../../lib/ipc'
import { useEngine } from '../../lib/engine'
import { Icon, type IconName } from '../../lib/icons'
import { useStore } from '../../lib/store'
import { compactAge, threadDiffStats } from '../../lib/thread-row'
import { shortPath } from '../../lib/format'
import type { Thread } from '../../lib/types'
import { Button, Input } from '../ui'
import { canUseWorkerProject, isLocalWorker, launchLocalWorker, makeWorkerDraft, WorkerLaunchError, workerProjectCwd, workerState, type WorkerStage, type WorkerState } from './workers'

const STAGE_LABELS: Record<WorkerStage, string> = {
  get checking() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Checking model") }, get worktree() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Creating worktree") }, get task() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Creating task") }, get configuring() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Setting up worker") }, get starting() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Starting worker") },
}
const STATE_LABELS: Record<WorkerState, string> = {
  get running() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Running") }, get waiting() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Needs you") }, get completed() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Completed") }, get review() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Ready for review") }, get failed() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Failed") }, get stopped() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Stopped") }, get ready() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Not running") },
}
const STATE_ICONS: Record<WorkerState, IconName> = {
  running: 'bolt', waiting: 'shield', completed: 'check', review: 'fileText', failed: 'warning', stopped: 'stop', ready: 'clock',
}

export function WorkersPanel({ onChooseModel }: { onChooseModel?: () => void } = {}) {
  const tx = useAuthoredText()
  const projects = useStore((s) => s.projects)
  const threads = useStore((s) => s.threads)
  const status = useStore((s) => s.status)
  const approvals = useStore((s) => s.approvals)
  const settings = useStore((s) => s.settings)
  const openNewProject = useStore((s) => s.openNewProject)
  const { snapshot, error: engineError } = useEngine()
  const [projectId, setProjectId] = useState('')
  const [title, setTitle] = useState('')
  const [objective, setObjective] = useState('')
  const [stage, setStage] = useState<WorkerStage | null>(null)
  const [failure, setFailure] = useState<WorkerLaunchError | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [stopping, setStopping] = useState<string[]>([])
  const [rowError, setRowError] = useState<{ threadId: string; message: string } | null>(null)
  const [filter, setFilter] = useState<'all' | 'active' | 'finished'>('all')
  const [formOpen, setFormOpen] = useState(() => !useStore.getState().threads.some(isLocalWorker))
  const launchLock = useRef(false)
  const eligible = projects.filter(canUseWorkerProject)
  const retainedSetup = !!failure?.draft.path
  const chosenProjectId = retainedSetup ? failure!.draft.projectId : eligible.some((p) => p.id === projectId) ? projectId : eligible[0]?.id ?? ''
  const project = eligible.find((p) => p.id === chosenProjectId)
  const model = retainedSetup ? failure!.draft.model : snapshot?.status.model
  const modelReady = !!model && snapshot?.status.available && snapshot.status.state === 'ready' && snapshot.status.model === model
  const providerReady = !!settings?.providers.some((p) => p.id === 'tensorfold')
  const workers = threads.filter(isLocalWorker).sort((a, b) => b.updatedAt - a.updatedAt)
  const states = new Map(workers.map((t) => [t.id, workerState(t, status[t.id], approvals[t.id]?.length)]))
  const retainedBusy = !!failure?.draft.threadId && ['running', 'waiting'].includes(states.get(failure.draft.threadId) ?? '')
  const successState = success ? states.get(success) : null
  const activeCount = workers.filter((t) => ['running', 'waiting'].includes(states.get(t.id)!)).length
  const waitingCount = workers.filter((t) => states.get(t.id) === 'waiting').length
  const visibleWorkers = workers.filter((t) => filter === 'all' || (filter === 'active' ? ['running', 'waiting'].includes(states.get(t.id)!) : !['running', 'waiting'].includes(states.get(t.id)!)))

  const start = async () => {
    if (launchLock.current || !project || !modelReady || !model || !objective.trim() || !providerReady) return
    launchLock.current = true
    setStage('checking')
    setSuccess(null)
    try {
      const draft = retainedSetup ? failure!.draft : makeWorkerDraft(project, title, objective, model, crypto.randomUUID())
      const result = await launchLocalWorker(draft, {
        api,
        getState: useStore.getState,
        // These are host snapshots during setup, before the run starts. As in
        // store.launch, insert them directly so same-millisecond configuration
        // replies cannot lose their provider/model to the live-update tie rule.
        onThread: (thread) => useStore.setState((s) => ({ threads: [thread, ...s.threads.filter((t) => t.id !== thread.id)] })),
        onStage: setStage,
        onStarting: (id) => useStore.setState((s) => ({ status: { ...s.status, [id]: 'thinking' } })),
        onStartFailed: (id) => useStore.setState((s) => ({ status: { ...s.status, [id]: 'error' } })),
      })
      setFailure(null)
      setTitle('')
      setObjective('')
      setFilter('all')
      setSuccess(result.threadId)
      setFormOpen(false)
    } catch (error) {
      setFailure(error instanceof WorkerLaunchError ? error : new WorkerLaunchError(String((error as Error)?.message ?? error), makeWorkerDraft(project, title, objective, model, crypto.randomUUID())))
    } finally {
      launchLock.current = false
      setStage(null)
    }
  }

  const open = (id: string, review = false) => {
    const state = useStore.getState()
    state.openThread(id)
    if (review) state.openWorkPane('changes')
  }
  const stop = async (id: string) => {
    if (stopping.includes(id)) return
    setStopping((ids) => [...ids, id])
    setRowError(null)
    try { await useStore.getState().cancel(id) }
    catch (error) { setRowError({ threadId: id, message: String((error as Error)?.message ?? error) }) }
    finally { setStopping((ids) => ids.filter((candidate) => candidate !== id)) }
  }

  return (
    <section data-testid="workers-panel" aria-labelledby="workers-title" className="mt-8 min-w-0">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 id="workers-title" className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{tx("Workers")}</h2>
          <p className="mt-1 max-w-[62ch] text-[13px] leading-relaxed text-ink-2">{tx("Give each worker a task and its own copy of your project, powered by the loaded local model.")}</p>
        </div>
        <div className="flex items-center gap-2">
          {activeCount > 0 && <span className="rounded-full border border-line px-2.5 py-1 text-[13px] text-ink-2">{activeCount} {tx("active")}{waitingCount > 0 ? tx(" · Needs your attention: {count}", { count: waitingCount }) : ''}</span>}
          {workers.length > 0 && <Button size="sm" variant={formOpen ? 'ghost' : 'default'} icon={formOpen ? 'x' : 'plus'} onClick={() => setFormOpen((open) => !open)} disabled={!!stage}>{formOpen ? tx("Close setup") : tx("New worker")}</Button>}
        </div>
      </div>

      <form hidden={!formOpen && workers.length > 0} className="rounded-xl border border-line bg-card p-4" onSubmit={(e) => { e.preventDefault(); void start() }} aria-busy={!!stage}>
        {eligible.length === 0 ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-medium text-ink">{tx("Add a local project to start a worker")}</div>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Choose a Git repository with an initial commit so each worker can use an isolated worktree.")}</p>
            </div>
            <Button icon="folder" size="sm" onClick={openNewProject}>{tx("Add project")}</Button>
          </div>
        ) : (
          <>
            <div className="grid min-w-0 gap-3 sm:grid-cols-2">
              <label className="min-w-0 text-[13px] font-medium text-ink-2">
                {tx("Project")}<select data-testid="worker-project" value={chosenProjectId} disabled={!!stage || retainedSetup} onChange={(e) => { setProjectId(e.target.value); setFailure(null) }} className="no-drag mt-1.5 h-9 w-full min-w-0 rounded-lg border border-line bg-card px-2.5 text-[13px] font-normal text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/15 disabled:opacity-60">
                  {eligible.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
              </label>
              <label className="min-w-0 text-[13px] font-medium text-ink-2">
                {tx("Task name ")}<span className="ml-1 font-normal">{tx("optional")}</span>
                <Input data-testid="worker-title" value={title} maxLength={80} disabled={!!stage || retainedSetup} onChange={(e) => setTitle(e.target.value)} placeholder={tx("For example, improve the search")} className="mt-1.5 font-normal" />
              </label>
            </div>
            <label className="mt-3 block text-[13px] font-medium text-ink-2">
              {tx("What should this worker do?")}<textarea data-testid="worker-objective" value={objective} disabled={!!stage || retainedSetup} required maxLength={8_000} rows={3} onChange={(e) => setObjective(e.target.value)} placeholder={tx("Describe the change and how to check it works...")} aria-describedby="worker-worktree-hint" className="no-drag mt-1.5 block w-full resize-y rounded-lg border border-line bg-card px-3 py-2.5 text-[13px] font-normal leading-relaxed text-ink outline-none placeholder:text-ink-2 focus:border-accent focus:ring-2 focus:ring-accent/15 disabled:opacity-60" />
            </label>
            {!objective && !stage && !retainedSetup && <Button size="xs" variant="ghost" icon="fileText" className="mt-1" onClick={() => {
              setTitle('Update project documentation')
              setObjective('Review the project documentation and update it to reflect the current implementation. Check instructions and code examples against the repository, preserve the existing document format, and summarize the changes.')
            }}>{tx("Use document task")}</Button>}
            <p id="worker-worktree-hint" className="mt-2 flex items-start gap-1.5 text-[13px] leading-relaxed text-ink-2">
              <Icon name="branch" size={15} className="mt-0.5 shrink-0" />
              <span>{tx("A new branch starts from the latest commit; uncommitted changes stay in your project.")}</span>
            </p>
            {project && <div title={workerProjectCwd(project)} className="mt-1 truncate text-[13px] text-ink-2">{shortPath(workerProjectCwd(project), 3)}</div>}
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
              <div className="min-w-0 flex-1 text-[13px] leading-relaxed text-ink-2">
                <div className="flex items-start gap-1.5">
                  <Icon name={modelReady ? 'cpu' : 'clock'} size={15} className="mt-0.5 shrink-0" />
                  <span className="break-all">{modelReady ? model : snapshot?.status.state === 'starting' ? tx("Waiting for the model to load") : retainedSetup ? tx("Load {value0} to continue setup", { value0: model }) : tx("Choose and load a model to start a worker")}</span>
                </div>
                {!modelReady && snapshot?.status.state !== 'starting' && onChooseModel && <Button size="xs" icon="cpu" className="mt-2" onClick={onChooseModel}>{tx("Choose a model")}</Button>}
                {engineError && <div className="mt-1 text-danger">{engineError}</div>}
                {!providerReady && settings && <div className="mt-1 text-danger">{tx("The built-in provider is unavailable; reopen TensorFold to restore it.")}</div>}
              </div>
              <Button data-testid="worker-start" type="submit" variant="primary" size="sm" icon="play" disabled={!!stage || retainedBusy || !modelReady || !providerReady || !project || !objective.trim()}>{stage ? STAGE_LABELS[stage] : retainedBusy ? tx("Worker running") : retainedSetup ? tx("Continue setup") : tx("Start worker")}</Button>
            </div>
          </>
        )}
      </form>

      {failure && <div role="alert" className="mt-3 rounded-lg border border-danger/25 bg-danger/10 p-3 text-[13px] leading-relaxed">
        <div className="text-danger">{failure.message}</div>
        {failure.draft.path && <div className="mt-2 text-ink-2">{tx("Your worktree is preserved at ")}<span className="break-all">{shortPath(failure.draft.path, 3)}</span>; {retainedBusy ? tx("open the worker to follow its progress.") : tx("continue setup to reuse it.")}</div>}
        {failure.draft.path && <div className="mt-2 flex flex-wrap gap-2">
          {failure.draft.threadId && <Button size="xs" onClick={() => open(failure.draft.threadId!)}>{tx("Open task")}</Button>}
          <Button size="xs" variant="ghost" disabled={!!stage} onClick={() => setFailure(null)}>{tx("Use a new worktree")}</Button>
        </div>}
      </div>}
      {success && <div role="status" className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink-2">
        <span>{successState === 'completed' ? tx("Worker completed; review its changes.") : successState === 'review' ? tx("Worker finished; review its changes.") : successState === 'waiting' ? tx("Worker started and needs your approval.") : successState === 'failed' ? tx("Worker started but encountered an error; open it for details.") : successState === 'stopped' ? tx("Worker stopped; its worktree is preserved.") : tx("Worker started in its own worktree.")}</span>
        <Button size="xs" variant="ghost" icon="arrowRight" onClick={() => open(success)}>{tx("Open worker")}</Button>
      </div>}

      {workers.length > 0 && <div className="mt-5 flex flex-wrap items-center justify-between gap-2">
        <div className="text-[13px] font-medium text-ink-2">{tx("Your workers ")}<span className="font-normal">{workers.length}</span></div>
        <div role="group" aria-label={tx("Filter workers")} className="flex gap-1">
          {(['all', 'active', 'finished'] as const).map((value) => <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)} className={clsx('no-drag rounded-md px-2.5 py-1.5 text-[13px] transition-colors', filter === value ? 'bg-ink-5 font-medium text-ink' : 'text-ink-2 hover:bg-ink-5')}>{value === 'all' ? tx("All") : value === 'active' ? tx("Active") : tx("Finished")}</button>)}
        </div>
      </div>}
      <div className="mt-2 divide-y divide-line">
        {visibleWorkers.map((thread) => <WorkerRow key={thread.id} thread={thread} state={states.get(thread.id)!} projectName={projects.find((p) => p.id === thread.projectId)?.name} stopping={stopping.includes(thread.id)} error={rowError?.threadId === thread.id ? rowError.message : null} onOpen={() => open(thread.id)} onReview={() => open(thread.id, true)} onStop={() => void stop(thread.id)} />)}
      </div>
      {workers.length > 0 && visibleWorkers.length === 0 && <p className="py-5 text-[13px] text-ink-2">{filter === 'active' ? tx("No workers are running right now.") : tx("Your workers are still running; finished work will appear here.")}</p>}
      {workers.length === 0 && <div className="mt-4 flex items-start gap-2.5 px-1 text-[13px] leading-relaxed text-ink-2">
        <Icon name="layers" size={18} className="mt-0.5 shrink-0" />
        <p>{tx("Workers keep going while you use the workspace; open one to follow its progress or review its changes.")}</p>
      </div>}
    </section>
  )
}

function WorkerRow({ thread, state, projectName, stopping, error, onOpen, onReview, onStop }: {
  thread: Thread; state: WorkerState; projectName?: string; stopping: boolean; error: string | null;
  onOpen(): void; onReview(): void; onStop(): void;
}) {
  const tx = useAuthoredText()
  const diffs = threadDiffStats(thread)
  const busy = state === 'running' || state === 'waiting'
  const receipt = thread.lastRunReceipt
  const latest = [...thread.timeline].reverse().find((item) => item.kind === 'tool' || item.kind === 'assistant' || item.kind === 'error')
  const summary = busy ? state === 'waiting' ? tx("Open this worker to review its pending approval.") : latest?.kind === 'tool' ? latest.title || tx("Working on the task") : tx("Working on the task") : receipt?.summary || (state === 'failed' ? tx("Open this worker to inspect the error and continue.") : thread.goal || tx("Open this task to continue setup."))
  return (
    <article data-testid="worker-row" className="min-w-0 py-4">
      <div className="flex items-start gap-2.5">
        <span className={clsx('mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-ink-5', state === 'failed' ? 'text-danger' : 'text-ink-2')}><Icon name={STATE_ICONS[state]} size={16} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
            <button type="button" onClick={onOpen} className="min-w-0 max-w-full truncate text-left text-[13px] font-semibold text-ink hover:underline">{thread.title}</button>
            <span className={clsx('text-[13px]', state === 'failed' ? 'text-danger' : 'text-ink-2')}>{STATE_LABELS[state]}</span>
            <span className="text-[13px] text-ink-2">{compactAge(thread.updatedAt) === 'now' ? tx("Just now") : tx("{value0} ago", { value0: compactAge(thread.updatedAt) })}</span>
          </div>
          <p className="mt-1 line-clamp-2 break-words text-[13px] leading-relaxed text-ink-2">{summary}</p>
          <div className="mt-2 flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-ink-2">
            {projectName && <span>{projectName}</span>}
            <span title={thread.workspace?.path || thread.cwd} className="max-w-full truncate">{shortPath(thread.workspace?.path || thread.cwd, 1)}</span>
            {diffs && <span>{tx("Files changed: {count}", { count: diffs.files })} <span className="text-positive">+{diffs.additions}</span> <span className="text-danger">−{diffs.deletions}</span></span>}
          </div>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            <Button size="xs" variant="default" icon={state === 'waiting' ? 'shield' : 'message'} onClick={onOpen}>{state === 'waiting' ? tx("Review approval") : tx("Open")}</Button>
            <Button size="xs" variant="ghost" icon="fileText" onClick={onReview}>{tx("Review changes")}</Button>
            {busy && <Button size="xs" variant="ghost" icon="stop" disabled={stopping} onClick={onStop}>{stopping ? tx("Stopping") : tx("Stop")}</Button>}
          </div>
          {error && <div role="alert" className="mt-2 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</div>}
        </div>
      </div>
    </article>
  )
}
