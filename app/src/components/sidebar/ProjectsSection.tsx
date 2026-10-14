import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { Menu, MenuItem } from '../ui'
import type { View } from '../../lib/store'
import type { Project, Thread, Room, CliSession } from '../../lib/types'
import { TAG_PRESETS, UNTAGGED_FILTER, tagIconColor, toggleTag } from '../../lib/project-tags'
import { terminalSessionKey } from '../../lib/terminal-sessions'
import { ACTIVE_ROW, HOVER_ROW } from './constants'
import { InlineEditable, SectionTitle } from './primitives'
import { RoomRow, TerminalRow, ThreadRow, rowClass } from './rows'
import type { ThreadRowProps, RoomRowProps, TerminalRowProps } from './types'

export interface ProjectGroup {
  project: Project
  threads: Thread[]
  rooms: Room[]
  ticketCount: number
}

export interface ProjectsSectionProps {
  view: View
  editingKey: string | null
  setEditingKey: (key: string | null) => void
  allTags: string[]
  showUntagged: boolean
  effectiveFilter: string
  setTag: (tag: string) => void
  projectGroups: ProjectGroup[]
  collapsed: Set<string>
  forceOpen: boolean
  toggleCollapsed: (projectId: string) => void
  terminals: Record<string, CliSession[]>
  settings: { tagColors?: Record<string, string> | null } | null | undefined
  threadRowProps: ThreadRowProps
  roomRowProps: RoomRowProps
  terminalRowProps: TerminalRowProps
  onNewThread: (projectId: string | null) => void
  openNewProject: () => void
  openProject: (id: string) => void
  setView: (view: View) => void
  renameProject: (id: string, name: string) => void
  updateProject: (id: string, patch: Partial<Project>) => void
  archiveItem: (kind: 'project' | 'room' | 'thread', id: string) => void
  createRoom: (projectId: string, name: string) => void
  startTerminal: (projectId: string, cliId: string) => Promise<CliSession | null>
  openTerminal: (projectId: string, sessionId: string) => void
  orchestratorName: (id?: string | null) => string
}

function TagFilter({ allTags, showUntagged, effectiveFilter, setTag }: Pick<ProjectsSectionProps, 'allTags' | 'showUntagged' | 'effectiveFilter' | 'setTag'>) {
  const tab = (id: string, label: string) => (
    <button key={id} onClick={() => setTag(id)} className={clsx('rounded-md px-1.5 py-0.5 text-[13px] capitalize transition-colors', effectiveFilter === id ? 'bg-surface-hover text-ink' : 'text-ink-3 hover:text-ink')}>{label}</button>
  )
  return (
    <div className="flex flex-wrap gap-1 px-2 pb-1">
      {tab('all', 'All')}
      {allTags.map((t) => tab(t, t))}
      {showUntagged && tab(UNTAGGED_FILTER, 'Untagged')}
    </div>
  )
}

// Projects, each expandable to its Kanban, terminals, rooms and threads.
export function ProjectsSection(props: ProjectsSectionProps) {
  const { view, editingKey, setEditingKey, effectiveFilter, projectGroups, collapsed, forceOpen, toggleCollapsed, terminals, settings, threadRowProps, roomRowProps, terminalRowProps } = props
  return (
    <>
      <SectionTitle action={<button onClick={props.openNewProject} title="New project" className="no-drag flex h-6 w-6 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink" aria-label="New project"><Icon name="plus" size={16} /></button>}>Projects</SectionTitle>
      {props.allTags.length > 0 && <TagFilter {...props} />}
      {projectGroups.length === 0 && (
        <div className="px-2.5 py-1.5 text-[13px] text-ink-3">
          {effectiveFilter === UNTAGGED_FILTER ? 'No untagged projects.' : effectiveFilter !== 'all' ? `No "${effectiveFilter}" projects.` : 'No projects yet.'}
        </div>
      )}
      {projectGroups.map(({ project, threads, rooms, ticketCount }) => {
        const projectKey = `project:${project.id}`
        const editing = editingKey === projectKey
        const open = forceOpen || !collapsed.has(project.id)
        const overview = view.name === 'project' && view.id === project.id && (!view.tab || view.tab === 'overview')
        const boardActive = view.name === 'project' && view.id === project.id && view.tab === 'board'
        const count = threads.length + rooms.length
        return (
          <div key={project.id}>
            <div className="group relative">
              <div onClick={() => { if (!editing) props.openProject(project.id) }} className={clsx('no-drag relative flex h-8 w-full cursor-default items-center gap-2 rounded-[10px] pl-2.5 pr-16 text-[13px] transition-colors', overview ? ACTIVE_ROW : clsx('text-ink', HOVER_ROW))}>
                <button
                  onClick={(e) => { e.stopPropagation(); toggleCollapsed(project.id) }}
                  aria-label={open ? 'Collapse project' : 'Expand project'}
                  aria-expanded={open}
                  className="no-drag flex h-4 w-4 shrink-0 items-center justify-center text-ink-3 hover:text-ink"
                >
                  <Icon name="folder" size={16} className={clsx('group-hover:hidden', project.tags?.length ? tagIconColor(project.tags[0], settings?.tagColors) : '')} />
                  <Icon name={open ? 'chevronDown' : 'chevronRight'} size={16} className="hidden group-hover:block" />
                </button>
                <InlineEditable
                  value={project.name}
                  editing={editing}
                  onStartEdit={() => setEditingKey(projectKey)}
                  onCommit={(next) => { setEditingKey(null); if (next.trim() && next.trim() !== project.name) void props.renameProject(project.id, next) }}
                  onCancel={() => setEditingKey(null)}
                  className="min-w-0 flex-1 truncate text-left font-medium"
                />
                {project.executionTarget === 'orchestrator' && (
                  <span title={`Runs on ${props.orchestratorName(project.orchestratorId)}`} className="flex shrink-0 text-ink-3 group-hover:hidden"><Icon name="globe" size={14} /></span>
                )}
                {!editing && !open && count > 0 && <span className="shrink-0 text-[13px] tabular-nums text-ink-3 group-hover:hidden">{count}</span>}
              </div>
              {!editing && (
                <div className="no-drag absolute right-1.5 top-1/2 flex -translate-y-1/2 items-center">
                  <Menu align="end" width={176} trigger={(menuOpen) => (
                    <button className={clsx('flex h-6 w-6 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink', menuOpen ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')} aria-label="Add to project" title="Add a task, room, Kanban, terminal or tag"><Icon name="plus" size={16} /></button>
                  )}>
                    {(close) => (
                      <>
                        <MenuItem icon="file" onClick={() => { close(); void props.onNewThread(project.id) }}>Task</MenuItem>
                        <MenuItem icon="layers" onClick={() => { close(); void props.createRoom(project.id, 'New room') }}>Room</MenuItem>
                        <MenuItem icon="grid" onClick={() => { close(); props.setView({ name: 'project', id: project.id, tab: 'board' }) }}>Kanban</MenuItem>
                        <MenuItem icon="terminal" onClick={async () => { close(); const s = await props.startTerminal(project.id, 'shell'); if (s) props.openTerminal(project.id, terminalSessionKey(s)) }}>Terminal</MenuItem>
                        <div className="my-1 border-t border-line" />
                        <div className="px-2.5 pb-1 pt-0.5 text-[13px] text-ink-3">Tags</div>
                        {TAG_PRESETS.map((t) => {
                          const on = (project.tags ?? []).includes(t)
                          return (
                            <button key={t} onClick={() => { void props.updateProject(project.id, { tags: toggleTag(project.tags, t) }) }} className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] capitalize transition-colors hover:bg-surface">
                              <span className={clsx('size-1.5 rounded-full', on ? 'bg-accent' : 'bg-ink-4')} />
                              <span className={clsx('flex-1', on ? 'text-ink' : 'text-ink-3')}>{t}</span>
                              {on && <Icon name="check" size={14} className="text-accent" />}
                            </button>
                          )
                        })}
                        <MenuItem icon="plus" onClick={() => { close(); props.openProject(project.id) }}>Custom tag…</MenuItem>
                        {(project.tags ?? []).filter((t) => !(TAG_PRESETS as readonly string[]).includes(t)).map((t) => (
                          <button key={t} onClick={() => { void props.updateProject(project.id, { tags: toggleTag(project.tags, t) }) }} className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] capitalize text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger">
                            <Icon name="x" size={14} />Remove "{t}"
                          </button>
                        ))}
                      </>
                    )}
                  </Menu>
                  <button onClick={(e) => { e.stopPropagation(); props.archiveItem('project', project.id) }} className="flex h-6 w-6 items-center justify-center rounded-md text-ink-3 opacity-0 transition-opacity hover:bg-surface-hover hover:text-ink group-hover:opacity-100" aria-label="Archive project" title="Archive project"><Icon name="archive" size={16} /></button>
                </div>
              )}
            </div>
            {open && (
              <div className="pb-1">
                {ticketCount > 0 && (
                  <button onClick={() => props.setView({ name: 'project', id: project.id, tab: 'board' })} className={rowClass(boardActive)}>
                    <Icon name="grid" size={16} className="shrink-0 text-ink-3" />
                    <span className="min-w-0 flex-1 truncate">Kanban</span>
                    <span className="shrink-0 text-[13px] tabular-nums text-ink-3">{ticketCount}</span>
                  </button>
                )}
                {(terminals[project.id] ?? []).map((s) => <TerminalRow key={terminalSessionKey(s)} {...terminalRowProps(project.id, s)} />)}
                {rooms.map((room) => <RoomRow key={room.id} room={room} {...roomRowProps(room)} />)}
                {threads.map((t) => <ThreadRow key={t.id} t={t} indent {...threadRowProps(t, { inProject: true })} />)}
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}
