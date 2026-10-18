import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import { basename } from '../../lib/format'
import { api } from '../../lib/ipc'
import type { GitDiffFile, GitStatus, Thread } from '../../lib/types'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../../lib/thread-workspace'
import { STATUS_TINT } from './primitives'
import { ReviewChangesModal } from './ReviewChangesModal'

// ---- inline "N files changed" summary card (rendered in the transcript) ----
const FILES_CHANGED_COLLAPSED_KEY = 'vontra:filesChangedCollapsed'
export function FilesChangedCard({ thread, git, onChanged }: { thread: Thread; git: GitStatus | null; onChanged: () => void }) {
  const tx = useAuthoredText()
  const [busy, setBusy] = useState(false)
  // Collapse is a remembered global preference — the user can tuck the file list
  // away so the transcript isn't dominated by "N files changed".
  const [collapsed, setCollapsed] = useState(() => { try { return localStorage.getItem(FILES_CHANGED_COLLAPSED_KEY) === '1' } catch { return false } })
  const toggleCollapsed = () => setCollapsed((c) => { const next = !c; try { localStorage.setItem(FILES_CHANGED_COLLAPSED_KEY, next ? '1' : '0') } catch { /* ignore */ } return next })
  const [reviewOpen, setReviewOpen] = useState(false)
  const [reviewBusy, setReviewBusy] = useState(false)
  const [reviewError, setReviewError] = useState<string | null>(null)
  const [diffs, setDiffs] = useState<GitDiffFile[]>([])
  const [activePath, setActivePath] = useState<string | null>(null)
  const project = useStore((s) => thread.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const workspace = effectiveThreadWorkspace(thread, project)
  const cwd = effectiveThreadCwd(thread, project)
  if (!git || git.files.length === 0) return null
  // Reads (status/diff) are remote-aware; revert mutates the local repo, so it is
  // gated for remote workspaces — commit/revert remotely goes through the agent.
  const remote = workspace?.kind === 'ssh'
  const undoAll = async () => {
    if (busy || remote) return
    if (!confirm(tx("Revert changed files to their last committed state? Files: {count}.", { count: git.files.length }))) return
    setBusy(true)
    try {
      await api.revertFiles(cwd, git.files.map((f) => f.path), workspace)
      setReviewOpen(false)
      setDiffs([])
      onChanged()
    } finally { setBusy(false) }
  }
  const openReview = async () => {
    if (reviewBusy) return
    setReviewOpen(true)
    setReviewBusy(true)
    setReviewError(null)
    try {
      const next = await api.gitDiff(cwd, git.files.map((f) => f.path), workspace)
      setDiffs(next)
      setActivePath(next[0]?.path ?? null)
    } catch (err) {
      setDiffs([])
      setActivePath(null)
      setReviewError(err instanceof Error ? err.message : tx("Could not load the diff."))
    } finally {
      setReviewBusy(false)
    }
  }
  return (
    <>
      <div className="fade-in overflow-hidden rounded-xl border border-line bg-card-2">
        <div className={clsx('flex items-center gap-2 px-3.5 py-2.5', !collapsed && 'border-b border-line')}>
          <button onClick={toggleCollapsed} className="flex items-center gap-2 text-left" aria-expanded={!collapsed} title={collapsed ? tx("Show changed files") : tx("Hide changed files")}>
            <Icon name={collapsed ? 'chevronRight' : 'chevronDown'} size={12} className="text-ink-3" />
            <span className="text-[13px] font-medium text-ink">{tx("Files changed: {count}", { count: git.files.length })}</span>
            <span className="font-mono text-[13px] text-positive">+{git.additions}</span>
            <span className="font-mono text-[13px] text-danger">−{git.deletions}</span>
          </button>
          <div className="flex-1" />
          <button onClick={undoAll} disabled={busy || remote} title={remote ? tx("Revert runs on the local repo — use the agent for remote workspaces.") : undefined} className="flex items-center gap-1 text-[13px] text-ink-3 transition-colors hover:text-ink disabled:opacity-50">
            {busy ? tx("Reverting…") : tx("Undo")}<Icon name="back" size={12} />
          </button>
          <button onClick={openReview} disabled={reviewBusy} className="rounded-lg border border-line px-2.5 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-50">
            {reviewBusy ? tx("Loading…") : tx("Review")}
          </button>
        </div>
        {!collapsed && (
          <div>
            {git.files.map((f) => (
              <div key={f.path} className="flex items-center gap-2.5 px-3.5 py-2 text-[13px] last:pb-2.5">
                <FileGlyph path={f.path} size={16} />
                <span className="truncate text-ink-2">{basename(f.path)}</span>
                <span className={clsx('shrink-0 font-mono text-[12px] uppercase', STATUS_TINT[f.status])}>{f.status === '?' ? tx("new") : f.status}</span>
                <div className="flex-1" />
                <span className="shrink-0 font-mono text-[12px] text-positive">+{f.additions}</span>
                <span className="shrink-0 font-mono text-[12px] text-danger">−{f.deletions}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      {reviewOpen && <ReviewChangesModal files={diffs} activePath={activePath} loading={reviewBusy} error={reviewError} onPick={setActivePath} onClose={() => setReviewOpen(false)} threadId={thread.id} />}
    </>
  )
}
