import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { basename, shortPath } from '../lib/format'
import { Icon, type IconName } from '../lib/icons'
import { buildRoomAgentTree, type RoomAgentNodeStatus, type RoomAgentTreeNode } from '../lib/room-agent-tree'
import { buildThreadAgentTree, countThreadAgentNodes, type ThreadAgentNodeStatus, type ThreadAgentTreeNode } from '../lib/thread-agent-tree'
import { LoopPanel } from './LoopPanel'
import type { AssistantControlSnapshot, AppUpdateStatus, DiffLine, GitStatus, RoomState, Thread } from '../lib/types'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../lib/thread-workspace'

type WorktreeRow = { path: string; head: string | null; branch: string | null; detached: boolean; bare: boolean }

const BUSY = new Set(['thinking', 'streaming', 'tool', 'awaiting'])
const ACTIVE_ROOM_STATES = new Set<RoomState>(['queued', 'running', 'waiting_user', 'waiting_external'])
const ROOM_STATE_LABEL: Record<RoomState, string> = {
  idle: 'Idle',
  created: 'Created',
  ready: 'Ready',
  degraded: 'Degraded',
  queued: 'Queued',
  running: 'Running',
  waiting_user: 'Needs you',
  waiting_external: 'Waiting',
  complete: 'Complete',
  stopped: 'Stopped',
  failed: 'Failed',
  archived: 'Archived',
  error: 'Failed',
}

const UPDATE_STATE_LABEL: Record<AppUpdateStatus['state'], string> = {
  not_configured: 'No feed',
  not_checked: 'Not checked',
  checking: 'Checking',
  current: 'Current',
  available: 'Available',
  downloading: 'Downloading',
  ready: 'Ready to install',
  installing: 'Preparing restart',
  error: 'Error',
}

function activeWorkspace() {
  const { view, threads, projects, rooms, settings } = useStore.getState()
  if (view.name === 'thread') {
    const thread = threads.find((t) => t.id === view.id)
    const project = thread?.projectId ? projects.find((p) => p.id === thread.projectId) ?? null : null
    return { cwd: thread ? effectiveThreadCwd(thread, project) : settings?.defaultCwd || '', thread: thread ?? null, room: null, projectName: project?.name ?? null, workspace: thread ? effectiveThreadWorkspace(thread, project) : null }
  }
  if (view.name === 'project') {
    const project = projects.find((p) => p.id === view.id)
    return { cwd: project?.cwd || settings?.defaultCwd || '', thread: null, room: null, projectName: project?.name ?? null, workspace: project?.workspace ?? null }
  }
  if (view.name === 'room') {
    const room = rooms.find((r) => r.id === view.id)
    const project = room ? projects.find((p) => p.id === room.projectId) : null
    return { cwd: room?.workspaceCwd || project?.cwd || settings?.defaultCwd || '', thread: null, room: room ?? null, projectName: project?.name ?? null, workspace: room?.workspace ?? project?.workspace ?? null }
  }
  return { cwd: settings?.defaultCwd || '', thread: null, room: null, projectName: null, workspace: null }
}

function readSources(thread: Thread | null): string[] {
  if (!thread) return []
  const sources = new Set<string>()
  const outputs = new Set<string>()
  for (const item of thread.timeline) {
    if (item.kind !== 'tool' || item.state !== 'ok') continue
    if ((item.tool === 'edit_file' || item.tool === 'write_file') && item.path) outputs.add(item.path)
    if (item.tool === 'read_file' && item.path) sources.add(item.path)
  }
  return [...sources].filter((path) => !outputs.has(path)).slice(0, 5)
}

function workspaceLabel(cwd: string): string {
  if (!cwd) return 'No workspace'
  return basename(cwd) || shortPath(cwd, 3)
}

export function WorkspaceDock() {
  const view = useStore((s) => s.view)
  const threads = useStore((s) => s.threads)
  const projects = useStore((s) => s.projects)
  const rooms = useStore((s) => s.rooms)
  const roomStatus = useStore((s) => s.roomStatus)
  const tickets = useStore((s) => s.tickets)
  const automations = useStore((s) => s.automations)
  const updateStatus = useStore((s) => s.updateStatus)
  const status = useStore((s) => s.status)
  const settings = useStore((s) => s.settings)
  const checkForUpdates = useStore((s) => s.checkForUpdates)
  const openUpdateModal = useStore((s) => s.openUpdateModal)
  const open = useStore((s) => s.workspaceDockOpen)
  const setOpen = useStore((s) => s.setWorkspaceDockOpen)
  const setThreadWorkspace = useStore((s) => s.setThreadWorkspace)
  const [git, setGit] = useState<GitStatus | null>(null)
  const [assistantSnapshot, setAssistantSnapshot] = useState<AssistantControlSnapshot | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [wtOpen, setWtOpen] = useState(false)
  const [wt, setWt] = useState<{ branches: string[]; worktrees: WorktreeRow[]; current: string | null } | null>(null)
  const [wtBusy, setWtBusy] = useState(false)
  const [wtError, setWtError] = useState<string | null>(null)
  const [featureName, setFeatureName] = useState('')
  const [cpTurns, setCpTurns] = useState<number[]>([])
  const [cpBusy, setCpBusy] = useState(false)
  const [cpDiff, setCpDiff] = useState<string | null>(null)
  const [cpError, setCpError] = useState<string | null>(null)
  const [cpNotice, setCpNotice] = useState<string | null>(null)

  const { cwd, thread, room, projectName, workspace } = activeWorkspace()
  const remote = workspace?.kind === 'ssh'
  const provider = settings?.providers.find((p) => p.id === (thread?.providerId ?? settings.activeProviderId)) ?? settings?.providers[0]
  const model = thread?.model ?? settings?.activeModel ?? provider?.models[0] ?? null
  const sources = useMemo(() => readSources(thread), [thread])
  const plan = thread?.plan ?? []
  const planDone = plan.filter((p) => p.done).length
  const runningThreads = threads.filter((t) => BUSY.has(status[t.id] ?? ''))
  const runningRooms = rooms.filter((r) => ACTIVE_ROOM_STATES.has(roomStatus[r.id] ?? r.state))
  const runningAutomations = automations.filter((a) => a.runs[0]?.status === 'running')
  const runningCount = runningThreads.length + runningRooms.length + runningAutomations.length
  const currentStatus = thread ? status[thread.id] : null
  const roomState = room ? roomStatus[room.id] ?? room.state : null
  const roomTasks = room ? tickets.filter((t) => t.roomId === room.id) : []
  const openRoomTasks = roomTasks.filter((t) => t.status !== 'completed')
  const roomFiles = room ? [...new Set(room.events.filter((e) => e.path).map((e) => e.path!))] : []
  const roomQuestions = room ? room.events.filter((e) => e.kind === 'ask' && e.target && e.target !== 'all') : []
  const roomModels = room ? [...new Set(room.seats.map((s) => s.model))] : []
  const roomHasCloud = room ? room.seats.some((seat) => settings?.providers.find((p) => p.id === seat.providerId)?.kind !== 'local') : false
  const connectorCount = assistantSnapshot?.connectors.length ?? 0
  const activeConnectors = assistantSnapshot?.connectors.filter((c) => c.status !== 'pending').length ?? 0
  const gapCount = assistantSnapshot?.gaps.length ?? 0
  const updateAvailable = ['available', 'downloading', 'ready', 'installing'].includes(updateStatus?.state ?? '')
  const updateBusy = updateStatus?.state === 'checking'
  const threadAgentTree = useMemo(() => thread ? buildThreadAgentTree(thread) : null, [thread])
  const threadSubAgentCount = threadAgentTree ? countThreadAgentNodes(threadAgentTree) : 0
  const threadAgentStats = useMemo(() => thread ? threadAgentDiffStats(thread) : new Map<string, DiffStats>(), [thread])
  const roomAgentTree = useMemo(() => room ? buildRoomAgentTree(room) : [], [room])
  const roomSubAgentCount = useMemo(() => roomAgentTree.reduce((sum, node) => sum + countRoomAgentNodes(node), 0), [roomAgentTree])
  const subAgentCount = threadSubAgentCount || roomSubAgentCount

  useEffect(() => {
    let alive = true
    setGit(null)
    if (!cwd) return
    api.gitStatus(cwd, workspace).then((g) => { if (alive) setGit(g) }).catch(() => { if (alive) setGit(null) })
    return () => { alive = false }
  }, [cwd, thread?.updatedAt, view.name, refresh])

  useEffect(() => {
    let alive = true
    api.getAssistantSnapshot().then((snapshot) => { if (alive) setAssistantSnapshot(snapshot) }).catch(() => { if (alive) setAssistantSnapshot(null) })
    return () => { alive = false }
  }, [refresh, view.name])

  // Per-turn checkpoints exist only for task threads with a code workspace.
  useEffect(() => {
    let alive = true
    setCpTurns([]); setCpDiff(null); setCpError(null)
    if (!thread || thread.kind === 'chat' || !cwd) return
    api.checkpointList({ threadId: thread.id })
      .then((turns) => { if (alive) setCpTurns(turns) })
      .catch(() => { if (alive) setCpTurns([]) })
    return () => { alive = false }
  }, [thread?.id, thread?.kind, cwd, thread?.updatedAt, refresh, workspace])

  useEffect(() => { setCpNotice(null) }, [thread?.id])

  useEffect(() => {
    if (!wtOpen || !cwd) return
    let alive = true
    setWtBusy(true)
    api.getWorktrees({ cwd, workspace })
      .then((d) => { if (alive) setWt(d) })
      .catch(() => { if (alive) setWt(null) })
      .finally(() => { if (alive) setWtBusy(false) })
    return () => { alive = false }
  }, [wtOpen, cwd, refresh, workspace])

  const commit = async () => {
    if (!cwd || !git?.files.length || remote) return
    const msg = prompt('Commit message', thread?.goal || thread?.title || 'TensorFold changes')
    if (!msg) return
    try {
      await api.commit(cwd, msg, workspace)
      setRefresh((n) => n + 1)
    } catch (e: any) {
      alert(e?.message ?? 'Commit blocked by Git hygiene.')
    }
  }

  const switchBranch = async (branch: string) => {
    if (!cwd || wtBusy || branch === (wt?.current ?? git?.branch)) return
    setWtBusy(true); setWtError(null)
    try {
      const r = await api.checkoutBranch({ cwd, workspace, branch })
      if (!r.ok) { setWtError(r.error || 'Could not switch branch'); return }
      setRefresh((n) => n + 1)
    } catch (e: any) { setWtError(e?.message || 'Could not switch branch') }
    finally { setWtBusy(false) }
  }

  const switchWorktree = async (target: string) => {
    if (!thread || !target || target === cwd) return
    await setThreadWorkspace(thread.id, null, target)
    setWtOpen(false)
    setRefresh((n) => n + 1)
  }

  const createFeature = async (mode: 'worktree' | 'branch') => {
    const name = featureName.trim()
    if (!name || !cwd || wtBusy) return
    setWtBusy(true); setWtError(null)
    try {
      const r = await api.createWorktree({ cwd, workspace, name, mode })
      if (!r.ok) { setWtError(r.error || 'Could not create feature'); return }
      setFeatureName('')
      if (mode === 'worktree' && r.path && thread) {
        await setThreadWorkspace(thread.id, null, r.path)
        setWtOpen(false)
      }
      setRefresh((n) => n + 1)
    } catch (e: any) { setWtError(e?.message || 'Could not create feature') }
    finally { setWtBusy(false) }
  }

  const undoLastTurn = async () => {
    if (!thread || !cwd || cpBusy) return
    if (!confirm('Revert the file changes from the last agent turn? This restores the workspace to before that turn — the conversation is kept.')) return
    setCpBusy(true); setCpError(null); setCpNotice(null); setCpDiff(null)
    try {
      const r = await api.checkpointUndoLast({ threadId: thread.id })
      if (!r.ok) { setCpError(r.error || 'Nothing to undo.'); return }
      setCpNotice(r.warning ?? null)
      setRefresh((n) => n + 1)
    } catch (e: any) { setCpError(e?.message || 'Could not undo the last turn.') }
    finally { setCpBusy(false) }
  }

  const diffLastTurn = async () => {
    if (!thread || !cwd || cpBusy) return
    if (cpDiff !== null) { setCpDiff(null); return }
    setCpBusy(true); setCpError(null)
    try {
      const r = await api.checkpointDiffLast({ threadId: thread.id })
      if (!r.ok) { setCpError(r.error || 'No turn to diff.'); return }
      setCpDiff(r.diff?.trim() ? r.diff : '(no file changes in the last turn)')
    } catch (e: any) { setCpError(e?.message || 'Could not load the turn diff.') }
    finally { setCpBusy(false) }
  }

  if (!settings) return null

  if (!open) return null

  return (
    <aside className="no-drag fixed right-3 top-[66px] z-50 max-h-[calc(100vh-78px)] w-[282px] max-w-[calc(100vw-24px)] overflow-y-auto overflow-x-hidden rounded-xl border border-line bg-card shadow-pop scroll-thin sm:right-[52px] sm:max-w-[calc(100vw-64px)]">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2.5">
        <span className="text-[13px] font-semibold text-ink">Workspace</span>
        <span className="min-w-0 flex-1 truncate text-[13px] text-ink-3">{projectName || workspaceLabel(cwd)}</span>
        <button onClick={() => setOpen(false)} className="flex h-6 w-6 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label="Close workspace dock">
          <Icon name="x" size={13} />
        </button>
      </div>

      <Section title="Environment">
        <DockRow icon="file" label="Changes" value={git ? (git.files.length ? `${git.files.length} files` : 'clean') : 'checking'} accent={git && git.files.length > 0} />
        {room ? (
          <DockRow icon={roomHasCloud ? 'globe' : 'cpu'} label={roomHasCloud ? 'Mixed' : 'Local'} value={roomModels.length ? `${roomModels.length} model${roomModels.length === 1 ? '' : 's'}` : 'No seats'} />
        ) : (
          <DockRow icon={provider?.kind === 'local' ? 'cpu' : 'globe'} label={provider?.kind === 'local' ? 'Local' : 'Frontier'} value={model || 'No model'} />
        )}
        <button
          onClick={() => setWtOpen((v) => !v)}
          className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-ink-5"
        >
          <Icon name="branch" size={13} className="text-ink-3" />
          <span className="text-[13px] text-ink-3">Branch</span>
          <span className="ml-auto flex min-w-0 items-center gap-1">
            <span className="truncate text-[13px] font-medium text-ink">{git?.branch || wt?.current || 'no branch'}</span>
            <Icon name="chevronDown" size={12} className={clsx('text-ink-4 transition-transform', wtOpen && 'rotate-180')} />
          </span>
        </button>
        {wtOpen && (
          <div className="mb-1 mt-1 space-y-2 rounded-lg border border-line bg-card-2 p-2">
            {wtError && <div className="rounded-md bg-danger/10 px-2 py-1 text-[12px] leading-snug text-danger">{wtError}</div>}
            <div>
              <div className="mb-1 px-1 text-[12px] font-semibold uppercase tracking-wide text-ink-4">Branches</div>
              {wt?.branches.length ? (
                <div className="scroll-thin max-h-32 space-y-0.5 overflow-y-auto">
                  {wt.branches.map((b) => {
                    const active = b === (wt.current ?? git?.branch)
                    return (
                      <button
                        key={b}
                        onClick={() => switchBranch(b)}
                        disabled={wtBusy || active}
                        className={clsx('flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-[13px] transition-colors disabled:cursor-default', active ? 'text-ink' : 'text-ink-2 hover:bg-ink-5')}
                      >
                        <Icon name={active ? 'check' : 'branch'} size={11} className={active ? 'text-accent' : 'text-ink-4'} />
                        <span className="flex-1 truncate">{b}</span>
                      </button>
                    )
                  })}
                </div>
              ) : <div className="px-1.5 py-1 text-[12px] text-ink-4">{wtBusy ? 'Loading…' : 'No branches'}</div>}
            </div>
            {wt && wt.worktrees.length > 1 && (
              <div>
                <div className="mb-1 px-1 text-[12px] font-semibold uppercase tracking-wide text-ink-4">Worktrees</div>
                <div className="scroll-thin max-h-32 space-y-0.5 overflow-y-auto">
                  {wt.worktrees.map((w) => {
                    const here = w.path === cwd
                    return (
                      <button
                        key={w.path}
                        onClick={() => switchWorktree(w.path)}
                        disabled={!thread || here}
                        title={w.path}
                        className={clsx('flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-[13px] transition-colors disabled:cursor-default', here ? 'text-ink' : 'text-ink-2 hover:bg-ink-5')}
                      >
                        <Icon name={here ? 'check' : 'folder'} size={11} className={here ? 'text-accent' : 'text-ink-4'} />
                        <span className="flex-1 truncate">{w.branch || (w.detached ? 'detached' : basename(w.path))}</span>
                        {here && <span className="text-[11px] text-ink-4">here</span>}
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
            <div className="border-t border-line pt-2">
              <div className="mb-1 px-1 text-[12px] font-semibold uppercase tracking-wide text-ink-4">New feature</div>
              <input
                value={featureName}
                onChange={(e) => setFeatureName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && featureName.trim() && !remote) createFeature('worktree') }}
                placeholder="feature name"
                className="mb-1.5 w-full rounded-md border border-line bg-card px-2 py-1 text-[13px] text-ink placeholder:text-ink-4 focus:border-ink-3 focus:outline-none"
              />
              <div className="flex gap-1.5">
                <button
                  onClick={() => createFeature('worktree')}
                  disabled={!featureName.trim() || wtBusy || remote}
                  title={remote ? 'Worktrees are local-only for now — use Branch here.' : 'Create an isolated worktree (separate folder) and switch this chat to it.'}
                  className="flex flex-1 items-center justify-center gap-1 rounded-md bg-ink px-2 py-1.5 text-[13px] font-medium text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
                >
                  <Icon name="plus" size={11} /> Create worktree
                </button>
                <button
                  onClick={() => createFeature('branch')}
                  disabled={!featureName.trim() || wtBusy}
                  title="Create the branch here and switch to it in place."
                  className="flex flex-1 items-center justify-center gap-1 rounded-md border border-line px-2 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 disabled:opacity-40"
                >
                  <Icon name="branch" size={11} /> Branch here
                </button>
              </div>
            </div>
          </div>
        )}
        <button
          onClick={commit}
          disabled={!git?.files.length || remote}
          title={remote ? 'Commit runs on the local repo — use the agent for remote workspaces.' : undefined}
          className="mt-1 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] text-ink-2 transition-colors hover:bg-ink-5 disabled:opacity-45 disabled:hover:bg-transparent"
        >
          <Icon name="check" size={13} className="text-ink-3" />
          <span className="flex-1">Commit or push</span>
          {git?.files.length ? <span className="font-mono text-[12px] text-positive">+{git.additions}</span> : null}
          {git?.files.length ? <span className="font-mono text-[12px] text-danger">-{git.deletions}</span> : null}
        </button>
        {thread && (cpTurns.length > 0 || cpNotice || cpError) && (
          <div className="mt-1 rounded-lg border border-line/70 p-1.5">
            {cpTurns.length > 0 && <div className="flex items-center gap-1">
              <button
                onClick={undoLastTurn}
                disabled={cpBusy}
                title="Restore the workspace to before the last agent turn. Only files are reverted — the conversation is kept."
                className="flex flex-1 items-center gap-2 rounded-md px-1.5 py-1 text-left text-[13px] text-ink-2 transition-colors hover:bg-ink-5 disabled:opacity-45"
              >
                <Icon name="back" size={13} className="text-ink-3" />
                <span className="flex-1">Undo last agent turn</span>
                <span className="font-mono text-[12px] text-ink-3">turn {cpTurns[cpTurns.length - 1]}</span>
              </button>
              <button
                onClick={diffLastTurn}
                disabled={cpBusy}
                title="Show what the last agent turn changed."
                className={clsx('flex h-7 items-center rounded-md px-2 text-[12px] font-medium transition-colors disabled:opacity-45', cpDiff !== null ? 'bg-ink-5 text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink-2')}
              >
                Diff
              </button>
            </div>}
            {cpError && <p className="mt-1 px-1.5 text-[12px] text-danger">{cpError}</p>}
            {cpNotice && <p className="mt-1 px-1.5 text-[12px] text-ink-3">{cpNotice}</p>}
            {cpDiff !== null && (
              <pre className="mt-1.5 max-h-64 overflow-auto rounded-md bg-ink-5 p-2 font-mono text-[12px] leading-relaxed text-ink-2">
                {cpDiff.split('\n').map((line, i) => (
                  <div key={i} className={clsx('whitespace-pre', line.startsWith('+') && !line.startsWith('+++') && 'text-positive', line.startsWith('-') && !line.startsWith('---') && 'text-danger', line.startsWith('@@') && 'text-accent')}>
                    {line || ' '}
                  </div>
                ))}
              </pre>
            )}
          </div>
        )}
      </Section>

      {/* Loop Mode is rooms-only (rescope #56): multi-agent surfaces loop; a solo
          task just runs. */}
      {room && (
        <>
          <Divider />
          <Section title="Loop Mode">
            <LoopPanel cwd={cwd} source="room" surfaceId={room.id} projectId={room.projectId ?? null} />
          </Section>
        </>
      )}

      {(thread || room) && (
        <>
          <Divider />
          <Section title="Subagents" right={subAgentCount ? <Pill>{subAgentCount}</Pill> : undefined}>
            {threadAgentTree && threadSubAgentCount > 0 ? (
              <ThreadDockAgentTreeView tree={threadAgentTree} stats={threadAgentStats} />
            ) : room && roomAgentTree.length > 0 ? (
              <RoomDockAgentTreeView tree={roomAgentTree} />
            ) : (
              <div className="px-1 text-[13px] text-ink-4">No sub-agents yet.</div>
            )}
          </Section>
        </>
      )}

      <Divider />
      <Section title="App update" right={updateStatus ? <Pill>{UPDATE_STATE_LABEL[updateStatus.state]}</Pill> : undefined}>
        <DockRow
          icon={updateAvailable ? 'spark' : updateStatus?.state === 'error' ? 'warning' : 'check'}
          label={updateAvailable ? `v${updateStatus?.latestVersion ?? ''}` : 'TensorFold'}
          value={updateStatus?.state === 'not_configured' ? 'feed not set' : updateStatus?.currentVersion ? `v${updateStatus.currentVersion}` : 'checking'}
          accent={updateAvailable || updateStatus?.state === 'error'}
        />
        {updateStatus?.releaseName && updateAvailable && <div className="mt-1 px-2 text-[13px] font-medium text-ink-2">{updateStatus.releaseName}</div>}
        {updateAvailable && updateStatus?.integrity && <div className="mt-1 px-2 text-[12px] text-ink-4">Integrity: {updateStatus.integrity.replace(/_/g, ' ')}</div>}
        {updateStatus?.error && <div className="mt-1 line-clamp-2 px-2 text-[13px] leading-snug text-danger">{updateStatus.error}</div>}
        <button
          onClick={() => updateAvailable ? openUpdateModal() : checkForUpdates()}
          disabled={updateBusy}
          className="mt-1.5 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] text-ink-2 transition-colors hover:bg-ink-5 disabled:opacity-45 disabled:hover:bg-transparent"
        >
          <Icon name={updateAvailable ? 'spark' : 'clock'} size={13} className="text-ink-3" />
          <span className="flex-1">{updateAvailable ? 'View release notes' : updateBusy ? 'Checking...' : 'Check now'}</span>
        </button>
      </Section>

      {room && (
        <>
          <Divider />
          <Section title="Room" right={roomState ? <Pill>{ROOM_STATE_LABEL[roomState]}</Pill> : undefined}>
            <DockRow icon="cpu" label="Seats" value={String(room.seats.length)} />
            <DockRow icon="list" label="Tasks" value={`${openRoomTasks.length}/${roomTasks.length}`} accent={openRoomTasks.length > 0} />
            <DockRow icon="message" label="Questions" value={String(roomQuestions.length)} accent={roomQuestions.length > 0} />
            <DockRow icon="file" label="Files touched" value={String(roomFiles.length)} accent={roomFiles.length > 0} />
          </Section>
        </>
      )}

      <Divider />
      <Section title="Assistant" right={assistantSnapshot?.pendingApprovals ? <Pill>{assistantSnapshot.pendingApprovals}</Pill> : undefined}>
        <DockRow icon="plug" label="Connectors" value={connectorCount ? `${activeConnectors}/${connectorCount}` : 'checking'} accent={activeConnectors > 0} />
        <DockRow icon="warning" label="Gaps" value={String(gapCount)} accent={gapCount > 0} />
        <DockRow icon="check" label="Approvals" value={String(assistantSnapshot?.pendingApprovals ?? 0)} accent={(assistantSnapshot?.pendingApprovals ?? 0) > 0} />
      </Section>

      <Divider />
      <Section title="Running" right={runningCount ? <Pill>{runningCount}</Pill> : undefined}>
        {runningCount === 0 ? (
          <div className="px-1 text-[13px] text-ink-4">Nothing running.</div>
        ) : (
          <div className="space-y-1">
            {thread && currentStatus && BUSY.has(currentStatus) && <RunRow title={thread.title} detail={currentStatus} />}
            {runningThreads.filter((t) => t.id !== thread?.id).slice(0, 2).map((t) => <RunRow key={t.id} title={t.title} detail={status[t.id] ?? 'running'} />)}
            {runningRooms.slice(0, 2).map((room) => <RunRow key={room.id} title={room.name} detail={ROOM_STATE_LABEL[roomStatus[room.id] ?? room.state]} />)}
            {runningAutomations.slice(0, 2).map((automation) => <RunRow key={automation.id} title={automation.name} detail="automation" />)}
          </div>
        )}
      </Section>

      <Divider />
      <Section title="Plan" right={plan.length ? <Pill>{planDone}/{plan.length}</Pill> : undefined}>
        {thread?.goal && <div className="mb-2 rounded-lg bg-ink-5 px-2 py-1.5 text-[13px] font-medium leading-snug text-ink">{thread.goal}</div>}
        {plan.length === 0 ? (
          <div className="flex min-h-[74px] flex-col items-center justify-center rounded-lg border border-dashed border-line text-center">
            <Icon name="list" size={16} className="text-ink-4" />
            <div className="mt-1 text-[13px] text-ink-4">No plan yet</div>
          </div>
        ) : (
          <div className="space-y-1.5">
            {plan.slice(0, 5).map((step) => (
              <div key={step.id} className="flex items-start gap-2 text-[13px] leading-snug">
                <Icon name={step.done ? 'check' : 'dots'} size={12} className={clsx('mt-0.5 shrink-0', step.done ? 'text-positive' : 'text-ink-4')} />
                <span className={clsx(step.done ? 'text-ink-4 line-through' : 'text-ink-2')}>{step.text}</span>
              </div>
            ))}
          </div>
        )}
      </Section>

      <Divider />
      <Section title="Sources" right={sources.length ? <Pill>{sources.length}</Pill> : undefined}>
        {sources.length === 0 ? (
          <div className="px-1 text-[13px] text-ink-4">No sources yet</div>
        ) : (
          <div className="space-y-1">
            {sources.map((source) => (
              <div key={source} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-[13px] text-ink-2">
                <Icon name="file" size={13} className="shrink-0 text-ink-3" />
                <span className="truncate">{source}</span>
              </div>
            ))}
          </div>
        )}
      </Section>
    </aside>
  )
}

function Section({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="px-3 py-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{title}</span>
        {right}
      </div>
      {children}
    </div>
  )
}

function Divider() {
  return <div className="border-t border-line" />
}

function Pill({ children }: { children: React.ReactNode }) {
  return <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-3">{children}</span>
}

function DockRow({ icon, label, value, accent }: { icon: IconName; label: string; value: string; accent?: boolean | null }) {
  return (
    <div className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-[13px]">
      <Icon name={icon} size={13} className="shrink-0 text-ink-3" />
      <span className="text-ink-2">{label}</span>
      <span className={clsx('min-w-0 flex-1 truncate text-right', accent ? 'text-positive' : 'text-ink-3')}>{value}</span>
    </div>
  )
}

function RunRow({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-[13px]">
      <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent" />
      <span className="min-w-0 flex-1 truncate text-ink-2">{title}</span>
      <span className="shrink-0 text-[12px] capitalize text-ink-4">{detail}</span>
    </div>
  )
}

interface DiffStats {
  added: number
  removed: number
}

type AgentStatus = ThreadAgentNodeStatus | RoomAgentNodeStatus

const AGENT_STATUS_DOT: Record<AgentStatus, string> = {
  idle: 'bg-ink-4',
  queued: 'bg-ink-4',
  running: 'bg-accent',
  active: 'bg-accent',
  succeeded: 'bg-positive',
  done: 'bg-positive',
  failed: 'bg-danger',
  cancelled: 'bg-ink-4',
}

const AGENT_STATUS_TEXT: Record<AgentStatus, string> = {
  idle: 'text-ink-4',
  queued: 'text-ink-3',
  running: 'text-accent',
  active: 'text-accent',
  succeeded: 'text-positive',
  done: 'text-positive',
  failed: 'text-danger',
  cancelled: 'text-ink-4',
}

const AGENT_STATUS_LABEL: Record<AgentStatus, string> = {
  idle: 'idle',
  queued: 'queued',
  running: 'working',
  active: 'working',
  succeeded: 'done',
  done: 'done',
  failed: 'failed',
  cancelled: 'stopped',
}

function ThreadDockAgentTreeView({ tree, stats }: { tree: ThreadAgentTreeNode; stats: Map<string, DiffStats> }) {
  return (
    <div className="space-y-1">
      <ThreadDockAgentRow node={tree} stats={stats} />
    </div>
  )
}

function ThreadDockAgentRow({ node, stats }: { node: ThreadAgentTreeNode; stats: Map<string, DiffStats> }) {
  const nodeStats = stats.get(node.id) ?? (node.eventId ? stats.get(node.eventId) : undefined)
  return (
    <div className="min-w-0">
      <AgentRowShell
        name={node.name}
        status={node.status}
        detail={node.detail}
        childCount={node.children.length}
        stats={nodeStats}
        prominent={node.kind === 'thread'}
      />
      {node.children.length > 0 && (
        <div className="ml-3 mt-1 space-y-1 border-l border-line pl-2">
          {node.children.map((child) => <ThreadDockAgentRow key={child.id} node={child} stats={stats} />)}
        </div>
      )}
    </div>
  )
}

function RoomDockAgentTreeView({ tree }: { tree: RoomAgentTreeNode[] }) {
  const visible = tree.filter((node) => node.children.length > 0 || node.status !== 'idle')
  if (visible.length === 0) return <div className="px-1 text-[13px] text-ink-4">No sub-agents yet.</div>
  return (
    <div className="space-y-1">
      {visible.map((node) => <RoomDockAgentRow key={node.id} node={node} />)}
    </div>
  )
}

function RoomDockAgentRow({ node }: { node: RoomAgentTreeNode }) {
  return (
    <div className="min-w-0">
      <AgentRowShell
        name={node.name}
        status={node.status}
        detail={node.kind === 'seat' ? node.role || node.detail : node.detail}
        childCount={node.children.length}
        prominent={node.kind === 'seat'}
      />
      {node.children.length > 0 && (
        <div className="ml-3 mt-1 space-y-1 border-l border-line pl-2">
          {node.children.map((child) => <RoomDockAgentRow key={child.id} node={child} />)}
        </div>
      )}
    </div>
  )
}

function AgentRowShell({
  name,
  status,
  detail,
  childCount,
  stats,
  prominent,
}: {
  name: string
  status: AgentStatus
  detail: string
  childCount: number
  stats?: DiffStats
  prominent?: boolean
}) {
  return (
    <div className={clsx('flex min-w-0 items-start gap-2 rounded-lg px-2 py-1.5', prominent ? 'bg-card-2/70' : 'hover:bg-ink-5')}>
      <span className={clsx('mt-[7px] h-2 w-2 shrink-0 rounded-sm', AGENT_STATUS_DOT[status], (status === 'running' || status === 'active') && 'animate-pulse')} />
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-1.5">
          <span title={name} className={clsx('min-w-0 truncate text-[13px] font-medium', prominent ? 'text-ink' : 'text-ink-2')}>{name}</span>
          {childCount > 0 && <span className="shrink-0 rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{childCount}</span>}
        </div>
        <div title={detail} className="mt-0.5 truncate text-[12px] text-ink-4">{detail}</div>
      </div>
      <div className="flex shrink-0 flex-col items-end gap-0.5 pt-0.5">
        <span className={clsx('text-[12px] font-medium', AGENT_STATUS_TEXT[status])}>{AGENT_STATUS_LABEL[status]}</span>
        {stats && (stats.added > 0 || stats.removed > 0) && (
          <span className="font-mono text-[12px]">
            {stats.added > 0 && <span className="text-positive">+{stats.added}</span>}
            {stats.added > 0 && stats.removed > 0 && <span className="text-ink-4"> </span>}
            {stats.removed > 0 && <span className="text-danger">-{stats.removed}</span>}
          </span>
        )}
      </div>
    </div>
  )
}

function countRoomAgentNodes(node: RoomAgentTreeNode): number {
  return node.children.reduce((sum, child) => sum + 1 + countRoomAgentNodes(child), 0)
}

function threadAgentDiffStats(thread: Thread): Map<string, DiffStats> {
  const stats = new Map<string, DiffStats>()
  for (const item of thread.timeline) {
    if (item.kind !== 'tool' || !item.diff?.length) continue
    const aliases = [
      item.id,
      item.subAgentId,
      item.subAgent?.id,
      item.args?.agentId,
      item.args?.subAgentId,
      item.args?.subagentId,
      item.args?.id,
    ]
      .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
      .flatMap((value) => [value, value.startsWith('subagent:') ? value : `subagent:${value}`])
    if (aliases.length === 0) continue
    const diff = diffStats(item.diff)
    for (const alias of aliases) {
      const current = stats.get(alias) ?? { added: 0, removed: 0 }
      current.added += diff.added
      current.removed += diff.removed
      stats.set(alias, current)
    }
  }
  return stats
}

function diffStats(lines: DiffLine[]): DiffStats {
  return lines.reduce<DiffStats>((sum, line) => {
    if (line.type === 'add') sum.added += 1
    else if (line.type === 'del') sum.removed += 1
    return sum
  }, { added: 0, removed: 0 })
}
