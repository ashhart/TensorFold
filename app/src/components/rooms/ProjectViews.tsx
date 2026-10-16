import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { relativeTime, shortPath } from '../../lib/format'
import { api } from '../../lib/ipc'
import { Icon, type IconName } from '../../lib/icons'
import { Button, Segmented, Toggle } from '../ui'
import { Board } from '../Board'
import { FileExplorer } from '../FileExplorer'
import { MemoryEntries } from '../MemoryEntries'
import { ProjectSkills } from '../ProjectSkills'
import { allowedProviders } from '../../lib/policy'
import { providerAccountChoices } from '../../lib/provider-accounts'
import { TAG_PRESETS, TAG_COLORS, tagTone, toggleTag, normalizeTag } from '../../lib/project-tags'
import type { GitRepositoryState, Project, ProjectExecutionTarget } from '../../lib/types'
import { Avatar, ROOM_STATE_LABEL } from './helpers'
import { TerminalsTab } from '../terminals/TerminalsTab'

// ---------------------------------------------------------------- Projects index
export function ProjectsIndex() {
  const projects = useStore((s) => s.projects)
  const rooms = useStore((s) => s.rooms)
  const archives = useStore((s) => s.archives)
  const archiveItem = useStore((s) => s.archiveItem)
  const restoreItem = useStore((s) => s.restoreItem)
  const openNewProject = useStore((s) => s.openNewProject)
  const openProject = useStore((s) => s.openProject)
  const [mode, setMode] = useState<'active' | 'archived'>('active')
  const archivedTotal = (archives?.projects.length ?? 0) + (archives?.threads.length ?? 0) + (archives?.rooms.length ?? 0) + (archives?.tickets.length ?? 0)

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[820px] px-7 py-9">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-[20px] font-semibold tracking-[-0.02em] text-ink">Projects</h1>
            <p className="mt-1 text-[13px] text-ink-3">A project holds shared memory and rooms where multiple agents work together.</p>
          </div>
          <Button variant="primary" icon="plus" onClick={openNewProject}>New project</Button>
        </div>

        <div className="mt-5 inline-flex rounded-lg border border-line bg-card-2 p-0.5">
          <button onClick={() => setMode('active')} className={clsx('rounded-[7px] px-3 py-1.5 text-[13px] font-medium transition-colors', mode === 'active' ? 'bg-card text-ink shadow-card' : 'text-ink-3 hover:text-ink')}>Active</button>
          <button onClick={() => setMode('archived')} className={clsx('rounded-[7px] px-3 py-1.5 text-[13px] font-medium transition-colors', mode === 'archived' ? 'bg-card text-ink shadow-card' : 'text-ink-3 hover:text-ink')}>Archive {archivedTotal > 0 && <span className="ml-1 text-ink-3">{archivedTotal}</span>}</button>
        </div>

        {mode === 'archived' ? (
          archivedTotal === 0 ? (
            <div className="mt-6 rounded-2xl border border-dashed border-line py-14 text-center text-[13px] text-ink-3">Nothing archived yet.</div>
          ) : (
            <div className="mt-5 space-y-4">
              <ArchiveGroup title="Projects" items={archives?.projects ?? []} icon="folder" onRestore={(id, hostId) => restoreItem('project', id, hostId)} />
              <ArchiveGroup title="Rooms" items={archives?.rooms ?? []} icon="layers" onRestore={(id, hostId) => restoreItem('room', id, hostId)} />
              <ArchiveGroup title="Chats and tasks" items={archives?.threads ?? []} icon="message" onRestore={(id, hostId) => restoreItem('thread', id, hostId)} />
              <ArchiveGroup title="Tickets" items={archives?.tickets ?? []} icon="archive" onRestore={(id, hostId) => restoreItem('ticket', id, hostId)} />
            </div>
          )
        ) : projects.length === 0 ? (
          <div className="mt-6 flex flex-col items-center justify-center rounded-2xl border border-dashed border-line py-14 text-center">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-ink-5 text-ink-3"><Icon name="folder" size={24} /></span>
            <div className="mt-3 text-[14px] font-medium text-ink">No projects yet</div>
            <div className="mt-1 max-w-[380px] text-[13px] text-ink-3">Create a project, then add rooms where two or more local/cloud models collaborate with shared memory.</div>
          </div>
        ) : (
          <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {projects.map((p) => {
              const count = rooms.filter((r) => r.projectId === p.id).length
              return (
                <div
                  key={p.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => openProject(p.id)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openProject(p.id) } }}
                  className="group cursor-default rounded-xl border border-line bg-card p-4 text-left shadow-card transition-colors hover:border-line-strong"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="folder" size={18} /></span>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[14px] font-medium text-ink">{p.name}</div>
                      <div className="text-[13px] text-ink-3">{count} room{count === 1 ? '' : 's'} · {relativeTime(p.updatedAt)}</div>
                    </div>
                    <button onClick={(e) => { e.stopPropagation(); archiveItem('project', p.id) }} className="flex h-7 items-center gap-1 rounded-lg border border-line px-2 text-[12px] font-medium text-ink-3 opacity-0 transition-opacity hover:bg-ink-5 hover:text-ink group-hover:opacity-100"><Icon name="archive" size={12} />Archive</button>
                  </div>
                  {p.memory && <p className="mt-2.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{p.memory}</p>}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none transition-colors focus:border-accent">
      {options.length === 0 && <option value="">No saved Orchestrators</option>}
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  )
}

function ArchiveGroup({ title, items, icon, onRestore }: { title: string; items: Array<{ id: string; name?: string; title?: string; updatedAt: number; archivedAt?: number | null; orchestratorHostId?: string | null; orchestratorId?: string | null }>; icon: 'folder' | 'layers' | 'message' | 'archive'; onRestore: (id: string, originHostId: string | null) => void }) {
  const hosts = useStore((state) => state.settings?.remoteOrchestrators ?? [])
  if (!items.length) return null
  return (
    <section>
      <h2 className="mb-2 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{title}</h2>
      <div className="space-y-2">
        {items.map((item) => {
          const originHostId = item.orchestratorHostId ?? item.orchestratorId ?? null
          const originLabel = originHostId ? (hosts.find((host) => host.id === originHostId)?.name ?? `Orchestrator ${originHostId.slice(0, 8)}`) : 'This Mac'
          return (
          <div key={`${originHostId ?? 'local'}:${item.id}`} className="flex items-center gap-3 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name={icon} size={15} /></span>
            <div className="min-w-0 flex-1">
              <div className="truncate text-[14px] font-medium text-ink">{item.name ?? item.title ?? 'Archived item'}</div>
              <div className="text-[13px] text-ink-3">Archived {relativeTime(item.archivedAt ?? item.updatedAt)} · {originLabel}</div>
            </div>
            <Button size="sm" variant="default" icon="back" onClick={() => onRestore(item.id, originHostId)}>Restore</Button>
          </div>
          )
        })}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- Project detail
// Compact tag editor for the project header: preset chips (Work/Personal) toggle on/off, custom
// tags show with a remove affordance, and "+ Tag" opens an INLINE input + colour palette (Electron
// has no window.prompt, which is why the old prompt-based button did nothing). The chosen colour is
// stored globally per tag name in settings.tagColors and drives both the chip and the tinted folder.
function ProjectTagEditor({ project }: { project: Project }) {
  const updateProject = useStore((s) => s.updateProject)
  const settings = useStore((s) => s.settings)
  const saveSettings = useStore((s) => s.saveSettings)
  const tagColors = settings?.tagColors
  const tags = project.tags ?? []
  const setTags = (next: string[]) => { if (next.length !== tags.length || next.some((t, i) => t !== tags[i])) void updateProject(project.id, { tags: next }) }
  const custom = tags.filter((t) => !(TAG_PRESETS as readonly string[]).includes(t))
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState('')
  const [color, setColor] = useState('neutral')

  const persistColor = (tag: string, colorName: string) => {
    const key = normalizeTag(tag)
    if (!key || !settings) return
    void saveSettings({ ...settings, tagColors: { ...(settings.tagColors ?? {}), [key]: colorName } }, { silent: true })
  }
  const commit = () => {
    const next = toggleTag(tags, draft)
    if (next.length !== tags.length) { persistColor(draft, color); setTags(next) }
    setAdding(false); setDraft('')
  }

  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5">
      <span className="mr-0.5 text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">Tags</span>
      {TAG_PRESETS.map((t) => {
        const active = tags.includes(t)
        return (
          <button
            key={t}
            onClick={() => setTags(toggleTag(tags, t))}
            className={clsx('rounded-md px-2 py-0.5 text-[12px] font-medium capitalize transition-colors', active ? tagTone(t, tagColors) : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}
          >
            {t}
          </button>
        )
      })}
      {custom.map((t) => (
        <button
          key={t}
          onClick={() => setTags(toggleTag(tags, t))}
          title={`Remove "${t}"`}
          className={clsx('group/tag flex items-center gap-1 rounded-md px-2 py-0.5 text-[12px] font-medium capitalize transition-colors hover:bg-danger/10 hover:text-danger', tagTone(t, tagColors))}
        >
          {t}
          <Icon name="x" size={10} className="opacity-60 group-hover/tag:opacity-100" />
        </button>
      ))}
      {adding ? (
        <span className="flex items-center gap-1.5 rounded-md border border-line bg-card px-1.5 py-1">
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') commit(); else if (e.key === 'Escape') { setAdding(false); setDraft('') } }}
            placeholder="tag name"
            className="h-5 w-24 bg-transparent text-[12px] text-ink outline-none placeholder:text-ink-4"
          />
          {TAG_COLORS.map((c) => (
            <button
              key={c.name}
              title={c.label}
              onClick={() => setColor(c.name)}
              className={clsx('h-4 w-4 rounded-full transition-transform', c.swatch, color === c.name ? 'ring-2 ring-ink/70 ring-offset-1 ring-offset-card' : 'opacity-60 hover:opacity-100')}
            />
          ))}
          <button onClick={commit} disabled={!draft.trim()} className="rounded bg-accent px-1.5 py-0.5 text-[12px] font-medium text-accent-ink transition-colors hover:bg-accent-strong disabled:opacity-40">Add</button>
        </span>
      ) : (
        <button
          onClick={() => { setAdding(true); setDraft(''); setColor('neutral') }}
          className="flex items-center gap-1 rounded-md px-2 py-0.5 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"
        >
          <Icon name="plus" size={11} />Tag
        </button>
      )}
    </div>
  )
}

export function ProjectView({ id, initialTab = 'overview' }: { id: string; initialTab?: 'overview' | 'rooms' | 'board' | 'chats' | 'files' | 'settings' | 'terminals' }) {
  const project = useStore((s) => s.projects.find((p) => p.id === id))
  const rooms = useStore((s) => s.rooms).filter((r) => r.projectId === id)
  const createRoom = useStore((s) => s.createRoom)
  const openRoom = useStore((s) => s.openRoom)
  const updateProject = useStore((s) => s.updateProject)
  const archiveItem = useStore((s) => s.archiveItem)
  const settings = useStore((s) => s.settings)

  const tickets = useStore((s) => s.tickets)
  const threads = useStore((s) => s.threads).filter((t) => t.projectId === id).sort((a, b) => b.updatedAt - a.updatedAt)
  const newThread = useStore((s) => s.newThread)
  const newChat = useStore((s) => s.newChat)
  const openThread = useStore((s) => s.openThread)
  const [tab, setTab] = useState<'overview' | 'rooms' | 'board' | 'chats' | 'files' | 'settings' | 'terminals'>(initialTab)

  useEffect(() => { setTab(initialTab) }, [initialTab, id])

  if (!project) return null
  const ticketCount = tickets.filter((t) => t.projectId === id).length

  const TabBtn = ({ k, label, icon, count }: { k: 'overview' | 'board' | 'chats' | 'rooms' | 'files' | 'settings' | 'terminals'; label: string; icon: IconName; count?: number }) => (
    <button onClick={() => setTab(k)} className={clsx('relative flex items-center gap-1.5 px-1 pb-2.5 text-[13px] font-medium transition-colors', tab === k ? 'text-ink' : 'text-ink-3 hover:text-ink-2')}>
      <Icon name={icon} size={13} />
      {label}{count != null && <span className="ml-1.5 text-ink-3">{count}</span>}
      {tab === k && <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-ink" />}
    </button>
  )
  const LayerCard = ({ icon, label, detail, action, onClick }: { icon: IconName; label: string; detail: string; action: string; onClick: () => void }) => (
    <button onClick={onClick} className="group rounded-xl border border-line bg-card p-4 text-left shadow-card transition-colors hover:border-line-strong">
      <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name={icon} size={18} /></span>
      <div className="mt-3 flex items-center gap-2">
        <span className="text-[14px] font-semibold text-ink">{label}</span>
        <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-3">{action}</span>
      </div>
      <div className="mt-1 text-[13px] leading-snug text-ink-3">{detail}</div>
    </button>
  )

  return (
    <div className="flex h-full flex-col">
      {/* The project name + back link live in the window title bar now. The Terminal tab drops the
          whole header (tags + tabs) for maximum screen real estate; other tabs keep tags + tabs. */}
      {tab !== 'terminals' && (
        <div className="border-b border-line px-7 pt-4">
          <ProjectTagEditor project={project} />
          <div className="mt-3 flex items-center gap-5">
            <TabBtn k="overview" icon="folder" label="Overview" />
            <TabBtn k="rooms" icon="layers" label="Rooms" count={rooms.length} />
            <TabBtn k="board" icon="grid" label="Kanban" count={ticketCount} />
            <TabBtn k="chats" icon="message" label="Solo" count={threads.length} />
            <TabBtn k="files" icon="folder" label="Files" />
            <TabBtn k="terminals" icon="terminal" label="Terminal" />
            <TabBtn k="settings" icon="settings" label="Settings" />
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1">
        {tab === 'overview' && (
          <div className="scroll-thin h-full overflow-y-auto px-7 py-6">
            <div className="mx-auto w-full max-w-[860px]">
              <div className="mb-4">
                <h2 className="text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Project layers</h2>
                <p className="mt-1 max-w-[620px] text-[13px] leading-relaxed text-ink-3">Start with a solo operation, a multi-agent room, or a Kanban board. They share the same project instructions and memory.</p>
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <LayerCard icon="terminal" label="Terminal" action="Terminal" detail="Open a shell or a CLI (Codex, Claude, …) scoped to this project." onClick={() => setTab('terminals')} />
                <LayerCard icon="message" label="Solo operation" action="Solo" detail="One model works in a project-scoped chat or task." onClick={() => newThread(id)} />
                <LayerCard icon="layers" label="Multi-agent room" action="Room" detail="Several agents deliberate, create tickets, and share room memory." onClick={() => createRoom(id, 'New room')} />
                <LayerCard icon="grid" label="Kanban board" action="Kanban" detail="Plan work as epics, bugs, and tasks before assigning agents." onClick={() => setTab('board')} />
              </div>
              <div className="mt-5 rounded-xl border border-line bg-card-2/50 px-4 py-3">
                <div className="flex flex-wrap gap-3 text-[13px] text-ink-3">
                  <span className="flex items-center gap-1.5"><Icon name="terminal" size={13} />Terminal</span>
                  <span className="flex items-center gap-1.5"><Icon name="message" size={13} />Solo operation</span>
                  <span className="flex items-center gap-1.5"><Icon name="layers" size={13} />Multi-agent room</span>
                  <span className="flex items-center gap-1.5"><Icon name="grid" size={13} />Kanban board</span>
                </div>
              </div>
            </div>
          </div>
        )}
        {tab === 'board' && <Board projectId={id} />}
        {tab === 'files' && <div className="h-full"><FileExplorer cwd={project.cwd || project.workspace?.path || settings?.defaultCwd || ''} workspace={project.workspace} /></div>}
        {tab === 'chats' && (
          <div className="scroll-thin h-full overflow-y-auto px-7 py-6">
            <div className="mx-auto w-full max-w-[820px]">
              <div className="flex items-center justify-between">
                <h2 className="text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Chats</h2>
                <div className="flex gap-2">
                  <Button variant="default" size="sm" icon="plus" onClick={() => newThread(id)}>New task</Button>
                  <Button variant="default" size="sm" icon="message" onClick={() => newChat(id)}>New chat</Button>
                </div>
              </div>
              {threads.length === 0 ? (
                <div className="mt-2 rounded-xl border border-dashed border-line py-8 text-center text-[13px] text-ink-3">No chats in this project yet. Start one for the frontend and another for the backend. They run independently.</div>
              ) : (
                <div className="mt-2 space-y-2">
                  {threads.map((t) => (
                    <div key={t.id} onClick={() => openThread(t.id)} className="group flex cursor-default items-center gap-3 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card transition-colors hover:border-line-strong">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name={t.kind === 'chat' ? 'message' : 'spark'} size={15} /></span>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[14px] font-medium text-ink">{t.title}</div>
                        <div className="text-[13px] text-ink-3">{t.kind === 'chat' ? 'Chat' : 'Task'} · {relativeTime(t.updatedAt)}</div>
                      </div>
                      <button onClick={(e) => { e.stopPropagation(); archiveItem('thread', t.id) }} className="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 opacity-0 transition-opacity hover:bg-ink-5 hover:text-ink group-hover:opacity-100" aria-label="Archive"><Icon name="archive" size={14} /></button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
        {tab === 'rooms' && (
          <div className="scroll-thin h-full overflow-y-auto px-7 py-6">
            <div className="mx-auto w-full max-w-[820px]">
              <div className="flex items-center justify-between">
                <h2 className="text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Rooms</h2>
                <Button variant="default" size="sm" icon="plus" onClick={() => createRoom(id, 'New room')}>New room</Button>
              </div>
              {rooms.length === 0 ? (
                <div className="mt-2 rounded-xl border border-dashed border-line py-8 text-center text-[13px] text-ink-3">No rooms yet. Create one and add agents.</div>
              ) : (
                <div className="mt-2 space-y-2">
                  {rooms.map((r) => (
                    <div key={r.id} onClick={() => openRoom(r.id)} className="group flex cursor-default items-center gap-3 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card transition-colors hover:border-line-strong">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="message" size={16} /></span>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[14px] font-medium text-ink">{r.name}</div>
                        <div className="text-[13px] text-ink-3">{r.seats.length} agent{r.seats.length === 1 ? '' : 's'} · {ROOM_STATE_LABEL[r.state]} · {relativeTime(r.updatedAt)}</div>
                      </div>
                      <div className="flex -space-x-1.5">{r.seats.slice(0, 4).map((s) => <Avatar key={s.id} name={s.name} size={22} />)}</div>
                      <button onClick={(e) => { e.stopPropagation(); archiveItem('room', r.id) }} className="ml-1 flex h-7 w-7 items-center justify-center rounded-md text-ink-3 opacity-0 transition-opacity hover:bg-ink-5 hover:text-ink group-hover:opacity-100" aria-label="Archive room"><Icon name="archive" size={14} /></button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
        {tab === 'terminals' && <TerminalsTab projectId={id} />}
        {tab === 'settings' && (
          <div className="scroll-thin h-full overflow-y-auto px-7 py-6">
            <div className="mx-auto w-full max-w-[820px] space-y-4">
              {/* Model policy */}
              <GitReadinessCard project={project} />

              <div className="rounded-xl border border-line bg-card p-4 shadow-card">
                <div className="mb-3 text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Run location</div>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink">
                      <Icon name={project.executionTarget === 'orchestrator' ? 'globe' : 'monitor'} size={14} className="text-ink-3" />
                      {project.executionTarget === 'orchestrator' ? 'Paired Orchestrator' : 'This Mac'}
                    </div>
                    <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
                      Choose whether new rooms and missions for this project should stay local or target an always-on TensorFold Orchestrator.
                    </div>
                  </div>
                  <Segmented<ProjectExecutionTarget>
                    value={project.executionTarget ?? 'local'}
                    onChange={(executionTarget) => updateProject(id, { executionTarget })}
                    options={[{ value: 'local', label: 'Local', icon: 'monitor' }, { value: 'orchestrator', label: 'Orchestrator', icon: 'globe' }]}
                  />
                </div>
                {project.executionTarget === 'orchestrator' && (() => {
                  const hosts = (settings?.remoteOrchestrators ?? []).filter((host) => host.enabled)
                  // The assigned host can disappear (removed/disabled in Settings) — flag the
                  // stale/empty state clearly so the project isn't silently mis-targeted (#5).
                  const hostMissing = !project.orchestratorId || !hosts.some((host) => host.id === project.orchestratorId)
                  return (
                  <div className="mt-3 rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-3">
                    <div className="mb-2 text-[12px] font-medium uppercase tracking-[0.08em] text-ink-3">Target host</div>
                    <Select
                      value={hostMissing ? '' : project.orchestratorId!}
                      onChange={(orchestratorId) => updateProject(id, { orchestratorId: orchestratorId || null })}
                      options={[...(hostMissing ? [{ value: '', label: hosts.length ? 'Pick a host…' : 'No Orchestrator configured' }] : []), ...hosts.map((host) => ({ value: host.id, label: `${host.name} · ${host.serverUrl}` }))]}
                    />
                    {hostMissing ? (
                      <div className="mt-2 text-warn">{hosts.length ? 'No host selected — pick one above, or switch the run location back to Local.' : 'This project targets an Orchestrator, but none is configured. Add one in Settings → Orchestrator, or switch to Local.'}</div>
                    ) : (
                      <div className="mt-2">This project is marked for Orchestrator execution. Remote rooms and missions use the selected host when remote routing is enabled.</div>
                    )}
                  </div>
                  )
                })()}
              </div>

              {settings && (() => {
                const allowed = allowedProviders(settings.providers, project)
                const dp = allowed.find((p) => p.id === project.defaultProviderId) || allowed[0]
                const accountChoices = dp ? providerAccountChoices(dp) : []
                const selectedAccount = accountChoices.find((account) => account.id === (project.defaultProviderAccountId ?? null)) ?? accountChoices.find((account) => account.id === (dp?.defaultAccountId ?? null)) ?? accountChoices[0]
                const models = selectedAccount?.models?.length ? selectedAccount.models : (dp?.models ?? [])
                return (
                  <div className="rounded-xl border border-line bg-card p-4 shadow-card">
                    <div className="mb-3 text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Model policy</div>
                    <label className="flex items-center gap-3">
                      <span className="flex-1">
                        <span className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><Icon name="cpu" size={14} className="text-ink-3" />Local models only</span>
                        <span className="mt-0.5 block text-[13px] leading-snug text-ink-3">Every chat and room agent in this project is restricted to on-device providers.</span>
                      </span>
                      <Toggle label="Local models only" checked={!!project.localOnly} onChange={(v) => {
                        const next = allowedProviders(settings.providers, { ...project, localOnly: v })
                        const pid = next.some((p) => p.id === project.defaultProviderId) ? project.defaultProviderId : next[0]?.id ?? null
                        const nextProvider = next.find((p) => p.id === pid)
                        const nextAccounts = nextProvider ? providerAccountChoices(nextProvider) : []
                        const nextAccount = nextAccounts.find((account) => account.id === (project.defaultProviderAccountId ?? null)) ?? nextAccounts.find((account) => account.id === (nextProvider?.defaultAccountId ?? null)) ?? nextAccounts[0]
                        const nextModels = nextAccount?.models?.length ? nextAccount.models : (nextProvider?.models ?? [])
                        updateProject(id, { localOnly: v, defaultProviderId: pid, defaultProviderAccountId: nextAccount?.id ?? null, defaultModel: nextModels[0] ?? null })
                      }} />
                    </label>
                    <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
                      <label className="block">
                        <div className="mb-1.5 text-[13px] text-ink-2">Default provider</div>
                        <select value={dp?.id ?? ''} onChange={(e) => {
                          const p = allowed.find((x) => x.id === e.target.value)
                          const accounts = p ? providerAccountChoices(p) : []
                          const account = accounts.find((choice) => choice.id === (p?.defaultAccountId ?? null)) ?? accounts[0]
                          const nextModels = account?.models?.length ? account.models : (p?.models ?? [])
                          updateProject(id, { defaultProviderId: e.target.value, defaultProviderAccountId: account?.id ?? null, defaultModel: nextModels[0] ?? null })
                        }} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card-2 px-2.5 text-[13px] text-ink outline-none focus:border-accent">
                          {allowed.map((p) => <option key={p.id} value={p.id}>{p.name}{p.kind === 'local' ? ' · local' : ''}</option>)}
                        </select>
                      </label>
                      <label className="block">
                        <div className="mb-1.5 text-[13px] text-ink-2">Default account</div>
                        <select value={selectedAccount?.id ?? ''} onChange={(e) => {
                          const account = accountChoices.find((choice) => (choice.id ?? '') === e.target.value) ?? accountChoices[0]
                          const nextModels = account?.models?.length ? account.models : (dp?.models ?? [])
                          updateProject(id, { defaultProviderAccountId: account?.id ?? null, defaultModel: nextModels[0] ?? null })
                        }} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card-2 px-2.5 text-[13px] text-ink outline-none focus:border-accent">
                          {accountChoices.map((account) => <option key={account.id ?? 'default'} value={account.id ?? ''}>{account.name}</option>)}
                        </select>
                      </label>
                      <label className="block">
                        <div className="mb-1.5 text-[13px] text-ink-2">Default model</div>
                        <select value={project.defaultModel ?? ''} onChange={(e) => updateProject(id, { defaultModel: e.target.value })} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card-2 px-2.5 text-[13px] text-ink outline-none focus:border-accent">
                          {models.length === 0 && <option value="">None</option>}
                          {models.map((m) => <option key={m} value={m}>{m}</option>)}
                        </select>
                      </label>
                    </div>
                  </div>
                )
              })()}

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <label className="block">
                  <div className="mb-1.5 text-[13px] font-medium text-ink-2">Shared memory</div>
                  <textarea value={project.memory} onChange={(e) => updateProject(id, { memory: e.target.value })} placeholder="Durable context every agent in this project sees…" rows={6} className="scroll-thin w-full resize-none rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
                </label>
                <label className="block">
                  <div className="mb-1.5 text-[13px] font-medium text-ink-2">Team charter</div>
                  <textarea value={project.team} onChange={(e) => updateProject(id, { team: e.target.value })} placeholder="How the team works together (roles, norms)…" rows={6} className="scroll-thin w-full resize-none rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
                </label>
              </div>
              <ProjectSkills projectId={id} />
              <MemoryEntries projectId={id} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function GitReadinessCard({ project }: { project: Project }) {
  const [state, setState] = useState<GitRepositoryState | null>(null)
  const [busy, setBusy] = useState(false)
  const [initBusy, setInitBusy] = useState(false)
  const [error, setError] = useState('')

  const refresh = async () => {
    if (!project.cwd || busy) return
    setBusy(true); setError('')
    try { setState(await api.inspectGitRepository(project.cwd)) }
    catch (e: any) {
      setState(null)
      setError(e?.message ?? 'Could not inspect Git repository.')
    } finally { setBusy(false) }
  }

  useEffect(() => {
    let live = true
    setState(null); setError('')
    if (!project.cwd) return
    setBusy(true)
    api.inspectGitRepository(project.cwd)
      .then((next) => { if (live) setState(next) })
      .catch((e) => { if (live) setError(e?.message ?? 'Could not inspect Git repository.') })
      .finally(() => { if (live) setBusy(false) })
    return () => { live = false }
  }, [project.cwd])

  const initialize = async () => {
    if (!project.cwd || initBusy) return
    setInitBusy(true); setError('')
    try { setState(await api.initGitRepository(project.cwd)) }
    catch (e: any) { setError(e?.message ?? 'Could not initialize Git.') }
    finally { setInitBusy(false) }
  }

  const ready = !!state?.exists && state.hasCommits
  return (
    <div className={clsx('rounded-xl border bg-card p-4 shadow-card', state?.exists === false ? 'border-warn/40' : 'border-line')}>
      <div className="flex items-start gap-3">
        <span className={clsx('flex h-8 w-8 shrink-0 items-center justify-center rounded-lg', ready ? 'bg-ink-5 text-ink-2' : 'bg-warn/15 text-warn')}>
          <Icon name={ready ? 'branch' : 'warning'} size={16} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-[13px] font-medium text-ink">Git and room worktrees</div>
              <div className="mt-0.5 text-[13px] text-ink-3">{project.cwd ? shortPath(project.cwd, 4) : 'No folder selected for this project.'}</div>
            </div>
            {project.cwd && <Button size="sm" variant="ghost" icon="branch" onClick={refresh} disabled={busy}>Refresh</Button>}
          </div>
          <div className="mt-3 rounded-lg border border-line bg-card-2/50 px-3 py-2 text-[13px] leading-snug text-ink-2">
            {!project.cwd
              ? 'Choose a project folder before running multi-agent rooms.'
              : busy
                ? 'Checking Git repository…'
                : ready
                  ? `Ready · ${state?.branch ? `branch ${state.branch}` : 'detached HEAD'} · new rooms get isolated worktrees.`
                  : state?.exists
                    ? 'Git exists, but it needs an initial commit before room worktrees can start.'
                    : 'This folder is not initialized as a Git repo. Initialize it before starting a room.'}
          </div>
          {state?.exists === false && (
            <div className="mt-3">
              <Button size="sm" variant="primary" icon="branch" onClick={initialize} disabled={initBusy}>{initBusy ? 'Initializing…' : 'Initialize Git'}</Button>
            </div>
          )}
          {error && <div className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-[13px] text-danger">{error}</div>}
        </div>
      </div>
    </div>
  )
}
