import { useEffect, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import { IconButton } from './ui'
import { basename } from '../lib/format'
import { collectTags, hasUntaggedProjects, projectMatchesTag, UNTAGGED_FILTER } from '../lib/project-tags'
import { compactAge, groupThreads, threadDiffStats, threadPlace, threadRowState, type DiffStats } from '../lib/thread-row'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../lib/thread-workspace'
import type { Thread, Room, CliSession, TimelineItem } from '../lib/types'
import { terminalSessionKey } from '../lib/terminal-sessions'
import { NEEDS_ATTENTION } from './sidebar/constants'
import { NavRow, SectionTitle } from './sidebar/primitives'
import { ThreadRow } from './sidebar/rows'
import { ProjectsSection } from './sidebar/ProjectsSection'
import { MoreSection } from './sidebar/MoreSection'
import { EngineChip } from './sidebar/EngineChip'
import { UpdatePrompt } from './sidebar/UpdatePrompt'
import { useBranches } from './sidebar/useBranches'
import type { ThreadRowProps, RoomRowProps, TerminalRowProps } from './sidebar/types'

const COLLAPSED_KEY = 'vontra.collapsedProjects'
const MORE_KEY = 'vontra.sidebarMore'
const diffCache = new WeakMap<TimelineItem[], DiffStats | null>()

function readJson<T>(key: string, fallback: T): T {
  try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) as T : fallback } catch { return fallback }
}
function writeJson(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* ignore */ }
}
function diffStatsFor(t: Thread): DiffStats | null {
  if (!diffCache.has(t.timeline)) diffCache.set(t.timeline, threadDiffStats(t))
  return diffCache.get(t.timeline) ?? null
}

// One workspace: New task and search, pinned threads, projects with their threads, threads without
// a project by day, a compact More group, then the engine, Models and Settings.
export function Sidebar() {
  const threads = useStore((st) => st.threads)
  const projects = useStore((st) => st.projects)
  const rooms = useStore((st) => st.rooms)
  const roomStatus = useStore((st) => st.roomStatus)
  const tickets = useStore((st) => st.tickets)
  const automations = useStore((st) => st.automations)
  const view = useStore((st) => st.view)
  const status = useStore((st) => st.status)
  const approvals = useStore((st) => st.approvals)
  const jobs = useStore((st) => st.jobs)
  const settings = useStore((st) => st.settings)
  const terminals = useStore((st) => st.terminals)
  const activeTerminalId = useStore((st) => st.activeTerminalId)
  const pinnedThreadIds = useStore((st) => st.pinnedThreadIds)
  const tagFilter = useStore((st) => st.projectTagFilter)
  const assistantSource = useStore((st) => st.assistantSource)
  // Actions are stable references, so read them once instead of subscribing.
  const s = useStore.getState()
  const needsMe = useMemo(() => jobs.reduce((n, j) => n + (j.pendingApprovals?.length ?? 0), 0), [jobs])
  const pickSource = (id: string) => { s.setAssistantSource(id); s.setView({ name: 'assistant' }) }
  // Warm each project's terminal list so live sessions show before its Terminal tab is opened.
  const projectIdsKey = projects.map((p) => p.id).join(',')
  useEffect(() => { projects.forEach((p) => void s.loadTerminals(p.id)) }, [projectIdsKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const [query, setQuery] = useState('')
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(() => new Set(readJson<string[]>(COLLAPSED_KEY, [])))
  const [moreOpen, setMoreOpen] = useState(() => readJson<boolean>(MORE_KEY, true))
  const toggleCollapsed = (id: string) => setCollapsed((c) => { const next = new Set(c); if (next.has(id)) next.delete(id); else next.add(id); writeJson(COLLAPSED_KEY, [...next]); return next })
  const toggleMore = () => setMoreOpen((o) => { writeJson(MORE_KEY, !o); return !o })

  // Project tag filter: 'all' | 'untagged' | a tag. Falls back to 'all' when its tag is gone.
  const allTags = useMemo(() => collectTags(projects), [projects])
  const showUntagged = useMemo(() => hasUntaggedProjects(projects), [projects])
  const specificTag = tagFilter !== 'all' && tagFilter !== UNTAGGED_FILTER
  const effectiveFilter = specificTag && !allTags.includes(tagFilter) ? 'all' : tagFilter

  // Resizable width: drag the right edge; persisted across reloads.
  const MIN = 238, MAX = 360
  const [width, setWidth] = useState(() => {
    const saved = Number(localStorage.getItem('vontra.sidebarWidth'))
    return saved >= MIN && saved <= MAX ? saved : 272
  })
  const widthRef = useRef(width); widthRef.current = width
  const startResize = (e: React.PointerEvent) => {
    e.preventDefault()
    const startX = e.clientX, startW = widthRef.current
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev: PointerEvent) => { const w = Math.min(MAX, Math.max(MIN, startW + (ev.clientX - startX))); widthRef.current = w; setWidth(w) }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      try { localStorage.setItem('vontra.sidebarWidth', String(widthRef.current)) } catch { /* ignore */ }
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const projectsById = useMemo(() => new Map(projects.map((p) => [p.id, p])), [projects])
  const pinned = useMemo(() => new Set(pinnedThreadIds), [pinnedThreadIds])
  const folderOf = (t: Thread) => {
    const project = t.projectId ? projectsById.get(t.projectId) ?? null : null
    const cwd = effectiveThreadCwd(t, project)
    const local = !t.orchestratorHostId && effectiveThreadWorkspace(t, project)?.kind !== 'ssh'
    return { project, cwd, hasFolder: t.kind === 'task' && !!cwd, local }
  }
  const sortedThreads = useMemo(() => [...threads].sort((a, b) => b.updatedAt - a.updatedAt), [threads])
  const branches = useBranches(sortedThreads.map(folderOf).filter((f) => f.hasFolder && f.local).map((f) => f.cwd))

  const { pinnedThreads, projectGroups, looseGroups, chats } = useMemo(() => {
    const q = query.trim().toLowerCase()
    const match = (v: string | null | undefined) => !q || (v ?? '').toLowerCase().includes(q)
    const matchThread = (t: Thread) => match(t.title) || match(t.cwd) || match(t.projectId ? projectsById.get(t.projectId)?.name : 'chat')
    const visible = sortedThreads.filter(matchThread)
    const byProject = new Map<string, Thread[]>()
    const loose: Thread[] = []
    for (const t of visible) {
      if (pinned.has(t.id)) continue
      if (t.projectId && projectsById.has(t.projectId)) byProject.set(t.projectId, [...(byProject.get(t.projectId) ?? []), t])
      else loose.push(t)
    }
    const roomsByProject = new Map<string, Room[]>()
    for (const r of [...rooms].sort((a, b) => b.updatedAt - a.updatedAt)) {
      if (!match(r.name) && !match(r.problem)) continue
      roomsByProject.set(r.projectId, [...(roomsByProject.get(r.projectId) ?? []), r])
    }
    const projectGroups = [...projects]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .filter((p) => projectMatchesTag(p, effectiveFilter))
      .map((project) => ({
        project,
        threads: byProject.get(project.id) ?? [],
        rooms: roomsByProject.get(project.id) ?? [],
        ticketCount: tickets.filter((ticket) => ticket.projectId === project.id && match(ticket.title)).length,
      }))
      .filter((g) => !q || match(g.project.name) || match(g.project.cwd) || g.threads.length > 0 || g.rooms.length > 0)
    // Threads without a project carry no tag, so a specific tag hides them.
    const showLoose = effectiveFilter === 'all' || effectiveFilter === UNTAGGED_FILTER
    return {
      pinnedThreads: visible.filter((t) => pinned.has(t.id)),
      projectGroups,
      // Folder work by day; chats (no folder, no file tools) in their own section below it.
      looseGroups: showLoose ? groupThreads(loose.filter((t) => folderOf(t).hasFolder), new Set()) : [],
      chats: showLoose ? loose.filter((t) => !folderOf(t).hasFolder) : [],
    }
  }, [sortedThreads, projects, projectsById, rooms, tickets, query, effectiveFilter, pinned])

  const reviewTickets = useMemo(() => tickets.filter((t) => NEEDS_ATTENTION.has(t.status)).length, [tickets])
  const runningAutomations = useMemo(() => automations.filter((a) => a.runs[0]?.status === 'running').length, [automations])

  const cycleTheme = () => {
    if (!settings) return
    const order = ['system', 'light', 'dark'] as const
    s.saveSettings({ ...settings, theme: order[(order.indexOf(settings.theme) + 1) % order.length] })
  }
  const themeIcon = settings?.theme === 'system' ? 'monitor' : settings?.theme === 'dark' ? 'moon' : 'sun'
  const now = Date.now()

  // Prop builders keep the hoisted rows stable: they get plain props, not closures over Sidebar.
  const threadRowProps: ThreadRowProps = (t, opts) => {
    const key = `thread:${t.id}`
    const { project, cwd, hasFolder } = folderOf(t)
    const branch = hasFolder ? branches[cwd] ?? null : null
    const meta = opts?.inProject
      ? branch ?? (hasFolder ? basename(cwd) : 'Chat')
      : [threadPlace(t, project), branch].filter(Boolean).join(' · ')
    return {
      active: view.name === 'thread' && view.id === t.id,
      state: threadRowState({ thread: t, run: status[t.id], approvals: approvals[t.id]?.length ?? 0, jobs }),
      meta,
      diff: diffStatsFor(t),
      age: compactAge(t.updatedAt, now),
      pinned: pinned.has(t.id),
      editing: editingKey === key,
      onOpen: () => s.openThread(t.id),
      onStartEdit: () => setEditingKey(key),
      onCommit: (next) => { setEditingKey(null); if (next.trim() && next.trim() !== t.title) void s.renameThread(t.id, next) },
      onCancel: () => setEditingKey(null),
      onArchive: () => s.archiveItem('thread', t.id),
      onTogglePin: () => s.togglePinnedThread(t.id),
    }
  }
  const roomRowProps: RoomRowProps = (room: Room) => {
    const state = roomStatus[room.id] ?? room.state
    const key = `room:${room.id}`
    return {
      active: view.name === 'room' && view.id === room.id,
      state,
      running: ['queued', 'running', 'waiting_user', 'waiting_external'].includes(state),
      editing: editingKey === key,
      onOpen: () => s.openRoom(room.id),
      onStartEdit: () => setEditingKey(key),
      onCommit: (next: string) => { setEditingKey(null); if (next.trim() && next.trim() !== room.name) void s.renameRoom(room.id, next) },
      onCancel: () => setEditingKey(null),
      onArchive: () => s.archiveItem('room', room.id),
    }
  }
  const terminalRowProps: TerminalRowProps = (projectId: string, session: CliSession) => {
    const sessionKey = terminalSessionKey(session)
    const key = `terminal:${sessionKey}`
    return {
      session,
      active: view.name === 'project' && view.id === projectId && view.tab === 'terminals' && activeTerminalId[projectId] === sessionKey,
      editing: editingKey === key,
      onOpen: () => s.openTerminal(projectId, sessionKey),
      onStartEdit: () => setEditingKey(key),
      onCommit: (next: string) => { setEditingKey(null); if (next.trim() && next.trim() !== session.title) void s.renameTerminal(projectId, sessionKey, next) },
      onCancel: () => setEditingKey(null),
      onClose: () => void s.killTerminal(projectId, sessionKey),
    }
  }
  const orchestratorName = (id?: string | null) => settings?.remoteOrchestrators?.find((host) => host.id === id)?.name ?? 'Orchestrator'
  const engineReady = typeof api.onEngineEvent === 'function' && typeof api.engineSnapshot === 'function'

  return (
    <aside style={{ width }} className="relative flex h-full shrink-0 flex-col">
      <div className="drag flex h-[52px] shrink-0 items-center gap-3 px-4">
        <div className="h-3 w-[58px] shrink-0" aria-hidden="true" />
        <div className="ml-auto flex items-center gap-0.5">
          <button onClick={cycleTheme} className="no-drag flex h-7 w-7 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface hover:text-ink" aria-label="Theme" title="Theme"><Icon name={themeIcon} size={16} /></button>
          <button disabled className="no-drag flex h-7 w-7 cursor-not-allowed items-center justify-center rounded-lg text-ink-4 opacity-45" aria-label="Back"><Icon name="back" size={16} /></button>
          <button disabled className="no-drag flex h-7 w-7 cursor-not-allowed items-center justify-center rounded-lg text-ink-4 opacity-45" aria-label="Forward"><Icon name="arrowRight" size={16} /></button>
        </div>
      </div>

      <div className="space-y-0.5 px-2 pb-1">
        <NavRow icon="pencil" label="New task" hint="⌘N" active={view.name === 'home'} onClick={s.newTask} />
        <NavRow icon="message" label="New chat" hint="⇧⌘N" onClick={() => void s.newChat(null)} />
        <NavRow icon="spark" label="Bots" active={view.name === 'bots'} onClick={() => s.setView({ name: 'bots' })} />
        <label className="no-drag flex h-8 items-center gap-2 rounded-[10px] px-2.5 text-ink-3 transition-colors focus-within:bg-surface hover:bg-surface">
          <Icon name="search" size={16} className="shrink-0" />
          <input
            id="sidebar-search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') setQuery('') }}
            placeholder="Search"
            className="min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-3"
          />
        </label>
      </div>

      <nav className="scroll-thin min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {pinnedThreads.length > 0 && (
          <>
            <SectionTitle>Pinned</SectionTitle>
            {pinnedThreads.map((t) => <ThreadRow key={t.id} t={t} {...threadRowProps(t)} />)}
          </>
        )}

        <ProjectsSection
          view={view}
          editingKey={editingKey}
          setEditingKey={setEditingKey}
          allTags={allTags}
          showUntagged={showUntagged}
          effectiveFilter={effectiveFilter}
          setTag={s.setProjectTagFilter}
          projectGroups={projectGroups}
          collapsed={collapsed}
          forceOpen={!!query.trim()}
          toggleCollapsed={toggleCollapsed}
          terminals={terminals}
          settings={settings}
          threadRowProps={threadRowProps}
          roomRowProps={roomRowProps}
          terminalRowProps={terminalRowProps}
          onNewThread={s.newThread}
          openNewProject={s.openNewProject}
          openProject={s.openProject}
          setView={s.setView}
          renameProject={s.renameProject}
          updateProject={s.updateProject}
          archiveItem={s.archiveItem}
          createRoom={s.createRoom}
          startTerminal={s.startTerminal}
          openTerminal={s.openTerminal}
          orchestratorName={orchestratorName}
        />

        {looseGroups.map((group) => (
          <div key={group.key}>
            <SectionTitle>{group.label}</SectionTitle>
            {group.threads.map((t) => <ThreadRow key={t.id} t={t} {...threadRowProps(t)} />)}
          </div>
        ))}
        {chats.length > 0 && (
          <div>
            <SectionTitle>Chats</SectionTitle>
            {chats.map((t) => <ThreadRow key={t.id} t={t} {...threadRowProps(t)} />)}
          </div>
        )}
        {query.trim() && pinnedThreads.length === 0 && projectGroups.length === 0 && looseGroups.length === 0 && chats.length === 0 && (
          <div className="px-2.5 py-2 text-[13px] text-ink-3">Nothing matches "{query.trim()}".</div>
        )}

        <MoreSection
          view={view}
          open={moreOpen || ['assistant', 'projects', 'automations', 'plugins', 'commandCentre'].includes(view.name)}
          onToggle={toggleMore}
          settings={settings}
          assistantSource={assistantSource}
          needsMe={needsMe}
          reviewTicketCount={reviewTickets}
          runningAutomations={runningAutomations}
          pickSource={pickSource}
          setView={s.setView}
        />
      </nav>

      <div className="space-y-0.5 border-t border-line px-2 pb-2.5 pt-2">
        <UpdatePrompt />
        {engineReady && <EngineChip active={view.name === 'models'} onOpen={() => s.setView({ name: 'models' })} />}
        <div className="flex items-center gap-0.5">
          <div className="min-w-0 flex-1"><NavRow icon="cpu" label="Models" active={view.name === 'models'} onClick={() => s.setView({ name: 'models' })} /></div>
          <div className="min-w-0 flex-1"><NavRow icon="settings" label="Settings" active={false} onClick={() => s.setView({ name: 'settings' })} /></div>
          <IconButton name="sidebar" label="Collapse sidebar" onClick={() => s.toggleSidebar()} />
        </div>
      </div>

      <div onPointerDown={startResize} className={clsx('no-drag group absolute right-0 top-0 z-20 h-full w-1.5 translate-x-1/2 cursor-col-resize')} aria-label="Resize sidebar">
        <div className="absolute right-1/2 top-0 h-full w-px bg-transparent transition-colors group-hover:bg-line-strong" />
      </div>
    </aside>
  )
}
