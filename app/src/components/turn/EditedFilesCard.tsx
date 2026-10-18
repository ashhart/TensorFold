import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import { basename } from '../../lib/format'
import { editTotals, type TurnSummary } from '../../lib/turn-blocks'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../../lib/thread-workspace'
import type { GitDiffFile, Thread } from '../../lib/types'
import { ReviewChangesModal } from '../thread-panel/ReviewChangesModal'

type WorkPaneTab = 'changes' | 'terminal' | 'files' | 'preview' | 'details'
type OpenWorkPane = (tab: WorkPaneTab, opts?: { path?: string }) => void

// The work pane lands in parallel; until it exists, Review falls back to the review modal.
function workPaneOpener(): OpenWorkPane | null {
  const candidate = (useStore.getState() as { openWorkPane?: unknown }).openWorkPane
  return typeof candidate === 'function' ? candidate as OpenWorkPane : null
}

function dirname(path: string): string {
  const i = path.lastIndexOf('/')
  return i > 0 ? path.slice(0, i) : ''
}

// "Edited 3 files +43 −3" under a turn's answer, with Review and Undo.
export function EditedFilesCard({ thread, turn, canUndo, onChanged }: { thread: Thread; turn: TurnSummary; canUndo: boolean; onChanged: () => void }) {
  const tx = useAuthoredText()
  const project = useStore((s) => thread.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const [review, setReview] = useState<{ files: GitDiffFile[]; active: string | null; loading: boolean; error: string | null } | null>(null)
  const [undoing, setUndoing] = useState(false)
  const [note, setNote] = useState<{ tone: 'error' | 'warn'; text: string } | null>(null)
  const totals = editTotals(turn.edits)
  const guestFiles = thread.bot?.executionTarget === 'vm'
  if (!totals.files) return null
  const cwd = effectiveThreadCwd(thread, project)
  const workspace = effectiveThreadWorkspace(thread, project)

  const openReview = async (path?: string) => {
    if (guestFiles) return
    const pane = workPaneOpener()
    if (pane) { pane('changes', path ? { path } : undefined); return }
    setReview({ files: [], active: path ?? null, loading: true, error: null })
    try {
      const files = await api.gitDiff(cwd, turn.edits.map((edit) => edit.path), workspace)
      setReview({ files, active: path ?? files[0]?.path ?? null, loading: false, error: null })
    } catch (err) {
      setReview({ files: [], active: null, loading: false, error: err instanceof Error ? err.message : tx("Could not load the diff.") })
    }
  }

  // The turn checkpoint restores files and conversation together; the host asks to confirm.
  const undo = async () => {
    if (undoing || !canUndo) return
    setUndoing(true); setNote(null)
    try {
      const result = await api.checkpointRevertTurn({ threadId: thread.id, turn: turn.index })
      if (!result.ok) setNote({ tone: 'error', text: result.error || 'Could not undo this turn.' })
      else if (result.warning) setNote({ tone: 'warn', text: result.warning })
      onChanged()
    } catch (err) {
      setNote({ tone: 'error', text: err instanceof Error ? err.message.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '') : 'Could not undo this turn.' })
    } finally {
      setUndoing(false)
    }
  }

  return (
    <div className="fade-in overflow-hidden rounded-[12px] border border-line bg-card">
      <div className="flex items-center gap-2 px-3 py-2">
        <span className="text-[13px] font-medium leading-[18px] text-ink">{tx("Files changed: {count}", { count: totals.files })}</span>
        {(totals.additions > 0 || totals.deletions > 0) && (
          <span className="flex items-center gap-1.5 font-mono text-[13px] leading-4">
            <span className="text-positive">+{totals.additions}</span>
            <span className="text-danger">−{totals.deletions}</span>
          </span>
        )}
        <div className="flex-1" />
        {canUndo && !guestFiles && (
          <button type="button" onClick={() => void undo()} disabled={undoing} title={tx("Restore the files and the conversation to before this turn")} className="flex h-7 items-center gap-1.5 rounded-md px-2 text-[13px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-50">
            <Icon name="back" size={14} />{undoing ? tx("Undoing") : tx("Undo")}
          </button>
        )}
        {guestFiles ? <span className="text-[12px] text-ink-3">{tx("Saved in bot VM")}</span> : <button type="button" onClick={() => void openReview()} className="flex h-7 items-center rounded-md border border-line px-2.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">
          {tx("Review")}</button>}
      </div>
      <div className="border-t border-line py-1">
        {turn.edits.map((edit) => (
          <button key={edit.path} type="button" disabled={guestFiles} onClick={() => void openReview(edit.path)} title={guestFiles ? tx("{value0} inside the isolated bot VM", { value0: edit.path }) : tx("Review {value0}", { value0: edit.path })} className="flex w-full items-center gap-2.5 px-3 py-1.5 text-left transition-colors enabled:hover:bg-ink-5">
            <FileGlyph path={edit.path} size={16} />
            <span className="min-w-0 flex-1 truncate text-[13px] leading-[18px]">
              <span className="text-ink-2">{basename(edit.path)}</span>
              {dirname(edit.path) && <span className="ml-1.5 text-ink-4">{dirname(edit.path)}</span>}
            </span>
            {edit.counted && (
              <span className="flex shrink-0 items-center gap-1.5 font-mono text-[13px] leading-4">
                <span className="text-positive">+{edit.additions}</span>
                <span className="text-danger">−{edit.deletions}</span>
              </span>
            )}
          </button>
        ))}
      </div>
      {note && <div className={note.tone === 'error' ? 'border-t border-line px-3 py-2 text-[13px] text-danger' : 'border-t border-line px-3 py-2 text-[13px] text-warn'}>{note.text}</div>}
      {review && (
        <ReviewChangesModal
          files={review.files}
          activePath={review.active}
          loading={review.loading}
          error={review.error}
          onPick={(path) => setReview((current) => current ? { ...current, active: path } : current)}
          onClose={() => setReview(null)}
          threadId={thread.id}
        />
      )}
    </div>
  )
}
