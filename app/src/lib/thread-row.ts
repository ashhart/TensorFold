import { basename } from './format'
import type { Job, Project, Thread } from './types'

// Pure helpers for the sidebar thread rows: run state, diff stats, place label, day groups, age.

export type ThreadRowState = 'running' | 'needs-you' | 'failed' | 'done'

const BUSY = new Set(['thinking', 'streaming', 'tool', 'awaiting'])

export interface ThreadRowStateInput {
  thread: Pick<Thread, 'id' | 'status'>
  run?: string
  approvals?: number
  jobs?: Pick<Job, 'pendingApprovals'>[]
}

// A persisted 'running' can be stale after a restart, so only the live run status counts as running.
export function threadRowState({ thread, run, approvals = 0, jobs = [] }: ThreadRowStateInput): ThreadRowState {
  const jobWaiting = jobs.some((job) => (job.pendingApprovals ?? []).some((a) => a.threadId === thread.id && !a.resolvedAt))
  if (approvals > 0 || run === 'awaiting' || jobWaiting) return 'needs-you'
  if (run && BUSY.has(run)) return 'running'
  if (run === 'error') return 'failed'
  if (run) return 'done'
  return thread.status === 'error' ? 'failed' : 'done'
}

export interface DiffStats { files: number; additions: number; deletions: number }

// What this thread's own edits added and removed, from the diffs on its tool items.
export function threadDiffStats(thread: Pick<Thread, 'timeline'>): DiffStats | null {
  const files = new Set<string>()
  let additions = 0
  let deletions = 0
  for (const item of thread.timeline) {
    if (item.kind !== 'tool' || !item.path || !item.diff?.length) continue
    if (item.state && item.state !== 'ok') continue
    files.add(item.path)
    for (const line of item.diff) {
      if (line.type === 'add') additions++
      else if (line.type === 'del') deletions++
    }
  }
  return files.size ? { files: files.size, additions, deletions } : null
}

// "Zora-Agent", "web-dashboard" or "Chat" when the thread has no folder.
export function threadPlace(thread: Pick<Thread, 'kind' | 'cwd' | 'projectId'>, project: Pick<Project, 'id' | 'name'> | null | undefined): string {
  if (project && thread.projectId === project.id) return project.name
  if (thread.kind === 'task' && thread.cwd) return basename(thread.cwd)
  return 'Chat'
}

export type ThreadGroupKey = 'pinned' | 'today' | 'yesterday' | 'earlier'
export interface ThreadGroup<T> { key: ThreadGroupKey; label: string; threads: T[] }

const GROUP_LABEL: Record<ThreadGroupKey, string> = { pinned: 'Pinned', today: 'Today', yesterday: 'Yesterday', earlier: 'Earlier' }

function startOfDay(now: number): number {
  const d = new Date(now)
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

// Pinned first, then by local day. Each group keeps newest first; empty groups are dropped.
export function groupThreads<T extends Pick<Thread, 'id' | 'updatedAt'>>(threads: T[], pinned: ReadonlySet<string>, now = Date.now()): ThreadGroup<T>[] {
  const today = startOfDay(now)
  const yesterday = startOfDay(today - 1)
  const buckets: Record<ThreadGroupKey, T[]> = { pinned: [], today: [], yesterday: [], earlier: [] }
  for (const t of [...threads].sort((a, b) => b.updatedAt - a.updatedAt)) {
    const key: ThreadGroupKey = pinned.has(t.id) ? 'pinned' : t.updatedAt >= today ? 'today' : t.updatedAt >= yesterday ? 'yesterday' : 'earlier'
    buckets[key].push(t)
  }
  return (Object.keys(buckets) as ThreadGroupKey[])
    .filter((key) => buckets[key].length > 0)
    .map((key) => ({ key, label: GROUP_LABEL[key], threads: buckets[key] }))
}

// Short age for a row: "now", "4m", "3h", "2d", "3w", then a date.
export function compactAge(ts: number, now = Date.now()): string {
  const s = Math.max(0, Math.floor((now - ts) / 1000))
  if (s < 60) return 'now'
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h`
  const d = Math.floor(h / 24)
  if (d < 7) return `${d}d`
  if (d < 35) return `${Math.floor(d / 7)}w`
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

// Threads that are waiting on the user, for the sidebar count and the dock badge.
export function countNeedsYou(states: Iterable<ThreadRowState>): number {
  let n = 0
  for (const s of states) if (s === 'needs-you') n++
  return n
}
