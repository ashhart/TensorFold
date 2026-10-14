import { useStore } from '../lib/store'
import { Icon } from '../lib/icons'
import { api } from '../lib/ipc'
import clsx from 'clsx'
import { ThreadTrustBadge } from './ThreadTrustBadge'
import { CortheonBadge } from './CortheonBadge'

// Top bar: compact (44px), borderless on the content card.
// Thread icon + title on the left; "Hand off" + panel toggles on the right.
// Folder/branch context lives in the composer's context strip, not up here.
export function TitleBar() {
  const view = useStore((s) => s.view)
  const threads = useStore((s) => s.threads)
  const projects = useStore((s) => s.projects)

  const thread = view.name === 'thread' ? threads.find((t) => t.id === view.id) : null
  const isTask = thread?.kind === 'task' && !!thread.cwd

  const sidebarCollapsed = useStore((s) => s.sidebarCollapsed)
  const toggleSidebar = useStore((s) => s.toggleSidebar)
  const toggleRightPanel = useStore((s) => s.toggleRightPanel)
  const workspaceDockOpen = useStore((s) => s.workspaceDockOpen)
  const toggleWorkspaceDock = useStore((s) => s.toggleWorkspaceDock)
  const setView = useStore((s) => s.setView)
  const openWorkPane = useStore((s) => s.openWorkPane)
  const showWorkspaceDockButton = view.name !== 'home' && view.name !== 'thread' && view.name !== 'settings'

  const title = thread
    ? thread.title
    : view.name === 'project'
      ? projects.find((p) => p.id === view.id)?.name ?? 'Project'
      : view.name === 'home'
        ? ''
        : view.name === 'commandCentre'
          ? 'Assistant'
          : view.name.charAt(0).toUpperCase() + view.name.slice(1)

  return (
    <header className={clsx('drag relative z-10 flex h-11 shrink-0 items-center gap-2 pr-3', sidebarCollapsed ? 'pl-[96px]' : 'pl-3')}>
      <button onClick={toggleSidebar} title="Toggle sidebar (⌘B)" className="no-drag flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink"><Icon name="sidebar" size={15} /></button>
      <div className="flex min-w-0 flex-1 items-center gap-2">
        {thread && <Icon name={thread.kind === 'chat' ? 'message' : 'file'} size={14} className="shrink-0 text-ink-3" />}
        {view.name === 'project' && (
          <button onClick={() => setView({ name: 'projects' })} title="Back to projects" className="no-drag flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink"><Icon name="back" size={13} /></button>
        )}
        {title && <span className="truncate text-[13px] font-medium text-ink">{title}</span>}
        {thread && <ThreadTrustBadge thread={thread} />}
        {thread?.bot && <button className="no-drag hidden max-w-[220px] shrink-0 truncate rounded-full border border-line px-2 py-0.5 text-[12px] text-ink-2 sm:block" onClick={() => setView({ name: 'bots' })} title="This chat keeps the bot instructions, model and permissions from when it was created.">{thread.bot.name}<span className="hidden lg:inline"> · {thread.bot.executionTarget === 'vm' ? 'Bot VM' : 'This Mac'} · {thread.bot.approvalScope === 'readonly' ? 'Read only' : 'Review actions'}</span></button>}
        {thread && <CortheonBadge thread={thread} />}
        {thread?.parentId && (
          <ThreadLineageChip parentId={thread.parentId} />
        )}
      </div>

      <div className="no-drag flex items-center gap-1">
        {/* Paired session (VST-035): this renderer is a browser driving a TensorFold host over
            /v1 — always visible so limited surfaces are never mistaken for broken ones. */}
        {api.platform === 'web-remote' && (
          <span className="mr-1 flex items-center gap-1.5 rounded-full border border-line bg-card-2 px-2.5 py-1 text-[12px] font-medium text-ink-2" title="This browser drives a TensorFold host over its paired session. Host-only surfaces (settings, connectors, local CLIs) are read-only or unavailable here.">
            <span className="h-1.5 w-1.5 rounded-full bg-positive" />
            Paired session
          </span>
        )}
        {/* No "Hand off" here: orchestrator mode already moves work
            across machines. Bundle export stays available via ⌘K → "Export this run…". */}
        <button onClick={() => setView({ name: 'commandCentre' })} className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink" aria-label="Assistant"><Icon name="bell" size={15} /></button>
        {thread && isTask && (
          <button
            onClick={() => openWorkPane('terminal')}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink"
            aria-label="Terminal"
          >
            <Icon name="terminal" size={15} />
          </button>
        )}
        {showWorkspaceDockButton && (
          <button
            onClick={toggleWorkspaceDock}
            className={clsx('flex h-7 w-7 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink', workspaceDockOpen && 'bg-surface-hover text-ink')}
            aria-label="Workspace dock"
            title="Workspace dock"
          >
            <Icon name="grid" size={15} />
          </button>
        )}
        <button onClick={toggleRightPanel} title="Toggle pane (⌥⌘B)" className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink" aria-label="Toggle pane"><Icon name="sidebar" size={15} className="rotate-180" /></button>
      </div>
    </header>
  )
}

// Lineage chip (VST-049): when a thread was forked or handed off from another,
// show a subtle "← from <parent>" breadcrumb next to the title. Clicking jumps
// to the parent. Keeps the new thread's origin visible without cluttering.
function ThreadLineageChip({ parentId }: { parentId: string }) {
  const parent = useStore((s) => s.threads.find((t) => t.id === parentId))
  const setView = useStore((s) => s.setView)
  if (!parent) return null
  return (
    <button
      onClick={() => setView({ name: 'thread', id: parentId })}
      title={`Continue viewing the source thread: ${parent.title}`}
      className="no-drag flex shrink-0 items-center gap-1 rounded-full border border-line bg-card-2 px-2 py-0.5 text-[12px] text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink-2"
    >
      <Icon name="back" size={10} />
      <span className="max-w-[140px] truncate">from {parent.title}</span>
    </button>
  )
}
