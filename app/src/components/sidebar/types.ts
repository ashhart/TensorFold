import type { Thread, Room, CliSession } from '../../lib/types'
import type { DiffStats, ThreadRowState } from '../../lib/thread-row'

// The closure-derived prop bundles the parent (Sidebar) computes for each hoisted,
// stable-identity row, so rows update instead of remounting on every Sidebar render.

export interface ThreadRowView {
  active: boolean
  state: ThreadRowState
  meta: string
  diff: DiffStats | null
  age: string
  pinned: boolean
  editing: boolean
  onOpen: () => void
  onStartEdit: () => void
  onCommit: (next: string) => void
  onCancel: () => void
  onArchive: () => void
  onTogglePin: () => void
}

export type ThreadRowProps = (t: Thread, opts?: { inProject?: boolean }) => ThreadRowView

export type RoomRowProps = (room: Room) => {
  active: boolean
  state: Room['state']
  running: boolean
  editing: boolean
  onOpen: () => void
  onStartEdit: () => void
  onCommit: (next: string) => void
  onCancel: () => void
  onArchive: () => void
}

export type TerminalRowProps = (projectId: string, session: CliSession) => {
  session: CliSession
  active: boolean
  editing: boolean
  onOpen: () => void
  onStartEdit: () => void
  onCommit: (next: string) => void
  onCancel: () => void
  onClose: () => void
}
