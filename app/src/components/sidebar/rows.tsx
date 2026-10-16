import clsx from 'clsx'
import { Icon, type IconName } from '../../lib/icons'
import type { Thread, Room, CliSession } from '../../lib/types'
import { ACTIVE_ROW, HOVER_ROW } from './constants'
import { InlineEditable, SidebarSpinner } from './primitives'
import type { ThreadRowView } from './types'

// Rows are hoisted to module scope so their identity stays stable across Sidebar renders;
// otherwise React remounts them on every status change and drops an in-progress rename.

function StatusMark({ state }: { state: ThreadRowView['state'] }) {
  if (state === 'running') return <SidebarSpinner size={12} />
  if (state === 'needs-you') return <span className="size-1.5 rounded-full bg-accent" />
  if (state === 'failed') return <span className="size-1.5 rounded-full bg-danger" />
  return null
}

function HoverAction({ icon, label, onClick }: { icon: IconName; label: string; onClick: () => void }) {
  return (
    <button onClick={(e) => { e.stopPropagation(); onClick() }} title={label} aria-label={label} className="flex h-6 w-6 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink">
      <Icon name={icon} size={16} />
    </button>
  )
}

// A thread: title and age, then where it runs and what it changed or that it needs you.
export function ThreadRow({ t, indent, active, state, meta, diff, age, pinned, editing, onOpen, onStartEdit, onCommit, onCancel, onArchive, onTogglePin }: ThreadRowView & { t: Thread; indent?: boolean }) {
  return (
    <div
      onClick={() => { if (!editing) onOpen() }}
      data-thread-row={t.id}
      className={clsx('no-drag group relative flex cursor-default gap-2 rounded-[10px] py-[5px] pr-2.5 transition-colors', indent ? 'pl-[26px]' : 'pl-2.5', active ? ACTIVE_ROW : HOVER_ROW)}
    >
      <span className="flex h-[18px] w-4 shrink-0 items-center justify-center text-ink-2"><StatusMark state={state} /></span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <InlineEditable
            value={t.title}
            editing={editing}
            onStartEdit={onStartEdit}
            onCommit={onCommit}
            onCancel={onCancel}
            className="min-w-0 flex-1 truncate text-[13px] leading-[18px] text-ink"
          />
          {!editing && t.incognito && <Icon name="incognito" size={14} className="shrink-0 text-ink-4" aria-label="Incognito, not saved" />}
          {!editing && <span className="shrink-0 text-[13px] leading-4 tabular-nums text-ink-3 group-hover:invisible">{age}</span>}
        </div>
        <div className="flex items-center gap-2 text-[13px] leading-4 text-ink-3">
          <span className="min-w-0 flex-1 truncate">{meta}</span>
          {state === 'needs-you'
            ? <span className="shrink-0 text-accent">Needs you</span>
            : diff && (
              <span className="shrink-0 tabular-nums" title={`${diff.files} file${diff.files === 1 ? '' : 's'} changed`}>
                <span className="text-positive">+{diff.additions}</span> <span className="text-danger">−{diff.deletions}</span>
              </span>
            )}
        </div>
      </div>
      {!editing && (
        <div className="absolute right-1.5 top-[3px] hidden items-center group-hover:flex">
          <HoverAction icon={pinned ? 'starFilled' : 'star'} label={pinned ? 'Unpin' : 'Pin'} onClick={onTogglePin} />
          <HoverAction icon="archive" label="Archive" onClick={onArchive} />
        </div>
      )}
    </div>
  )
}

// A plain 32 px row with a 16 px icon for a project's Kanban, rooms and terminals, indented one
// step so its icon sits under the project name.
export function rowClass(active: boolean) {
  return clsx('no-drag group relative flex h-8 w-full cursor-default items-center gap-2 rounded-[10px] pl-[26px] pr-2.5 text-left text-[13px] transition-colors', active ? ACTIVE_ROW : clsx('text-ink-2 hover:text-ink', HOVER_ROW))
}

// A live terminal session under its project. Renameable in place; click opens it in the Terminal tab.
export function TerminalRow({ session, active, editing, onOpen, onStartEdit, onCommit, onCancel, onClose }: {
  session: CliSession
  active: boolean
  editing: boolean
  onOpen: () => void
  onStartEdit: () => void
  onCommit: (next: string) => void
  onCancel: () => void
  onClose: () => void
}) {
  const running = session.status === 'busy' || session.status === 'attention'
  return (
    <div onClick={() => { if (!editing) onOpen() }} className={rowClass(active)}>
      <Icon name="terminal" size={16} className="shrink-0 text-ink-3" />
      <InlineEditable value={session.title} editing={editing} onStartEdit={onStartEdit} onCommit={onCommit} onCancel={onCancel} className="min-w-0 flex-1 truncate" />
      {!editing && running && <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-ink-3 group-hover:opacity-0" />}
      {!editing && <span className="absolute right-1.5 hidden group-hover:flex"><HoverAction icon="x" label="Close terminal" onClick={onClose} /></span>}
    </div>
  )
}

export function RoomRow({ room, active, running, state, editing, onOpen, onStartEdit, onCommit, onCancel, onArchive }: {
  room: Room
  active: boolean
  running: boolean
  state: Room['state']
  editing: boolean
  onOpen: () => void
  onStartEdit: () => void
  onCommit: (next: string) => void
  onCancel: () => void
  onArchive: () => void
}) {
  return (
    <div onClick={() => { if (!editing) onOpen() }} className={rowClass(active)}>
      <Icon name="layers" size={16} className="shrink-0 text-ink-3" />
      <InlineEditable value={room.name} editing={editing} onStartEdit={onStartEdit} onCommit={onCommit} onCancel={onCancel} className="min-w-0 flex-1 truncate" />
      {!editing && (running
        ? <span className="flex shrink-0 text-ink-3 group-hover:opacity-0"><SidebarSpinner size={12} /></span>
        : (state === 'failed' || state === 'error') && <span className="size-1.5 shrink-0 rounded-full bg-danger group-hover:opacity-0" />)}
      {!editing && <span className="absolute right-1.5 hidden group-hover:flex"><HoverAction icon="archive" label="Archive room" onClick={onArchive} /></span>}
    </div>
  )
}
