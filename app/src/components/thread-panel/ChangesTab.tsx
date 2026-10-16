import { useEffect, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import { changeTotals, parseUnifiedDiff, relativeToCwd, type ChangeTurn, type ChangedFile } from '../../lib/thread-changes'
import type { DiffLine, Thread, WorkspaceLocation } from '../../lib/types'
import { DiffView } from './DiffView'

const BIG_DIFF_LINES = 600
const STATUS_LABEL: Record<string, string> = { A: 'Added', M: 'Modified', D: 'Deleted', R: 'Renamed', '?': 'New' }

function splitPath(path: string): { dir: string; name: string } {
  const at = path.lastIndexOf('/')
  return at < 0 ? { dir: '', name: path } : { dir: path.slice(0, at + 1), name: path.slice(at + 1) }
}

function matchFile(turns: ChangeTurn[], wanted: string): string | null {
  for (const turn of turns) for (const f of turn.files) {
    if (f.path === wanted || f.path.endsWith('/' + wanted) || wanted.endsWith('/' + f.path)) return f.path
  }
  return null
}

// This thread's changed files, newest turn first, each with its diff from the working tree.
export function ChangesTab({ thread, cwd, workspace, turns }: { thread: Thread; cwd: string; workspace: WorkspaceLocation | null; turns: ChangeTurn[] }) {
  const focus = useStore((s) => s.workPaneFocus)
  const [diffs, setDiffs] = useState<Map<string, DiffLine[]> | null>(null)
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())
  const [expandedBig, setExpandedBig] = useState<Set<string>>(() => new Set())
  const [active, setActive] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const totals = changeTotals(turns)
  const gitPaths = useMemo(() => turns.flatMap((t) => t.files.filter((f) => f.status).map((f) => f.path)), [turns])
  const gitKey = turns.map((t) => t.files.map((f) => `${f.path}:${f.additions}:${f.deletions}`).join(',')).join('|')

  useEffect(() => {
    if (!cwd || gitPaths.length === 0) { setDiffs(new Map()); return }
    let alive = true
    api.gitDiff(cwd, gitPaths, workspace)
      .then((files) => { if (alive) setDiffs(new Map(files.map((f) => [f.path, parseUnifiedDiff(f.diff)]))) })
      .catch(() => { if (alive) setDiffs(new Map()) })
    return () => { alive = false }
  }, [cwd, workspace, gitKey]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!focus || focus.threadId !== thread.id || focus.tab !== 'changes' || !focus.path) return
    const path = matchFile(turns, relativeToCwd(focus.path, cwd))
    if (!path) return
    setActive(path)
    setCollapsed((c) => { const next = new Set(c); next.delete(path); return next })
    const id = requestAnimationFrame(() => {
      const el = listRef.current?.querySelector(`[data-path="${CSS.escape(path)}"]`)
      el?.scrollIntoView({ block: 'start' })
    })
    return () => cancelAnimationFrame(id)
  }, [focus?.nonce]) // eslint-disable-line react-hooks/exhaustive-deps

  if (turns.length === 0) {
    return <div className="px-4 py-8 text-center text-[13px] text-ink-3">No changes yet.</div>
  }

  const toggle = (path: string) => setCollapsed((c) => { const next = new Set(c); if (next.has(path)) next.delete(path); else next.add(path); return next })

  const fileBody = (file: ChangedFile) => {
    const diff = diffs?.get(file.path) ?? file.editDiff
    if (!diff) return <div className="px-4 py-3 text-[13px] text-ink-3">{diffs ? 'No text diff for this file.' : 'Loading diff…'}</div>
    if (diff.length > BIG_DIFF_LINES && !expandedBig.has(file.path)) {
      return (
        <button onClick={() => setExpandedBig((s) => new Set(s).add(file.path))} className="w-full px-4 py-3 text-left text-[13px] text-ink-3 transition-colors hover:text-ink">
          Show {diff.length} lines
        </button>
      )
    }
    return <DiffView diff={diff} />
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-line px-4 text-[13px]">
        <span className="text-ink-2">{totals.files} file{totals.files === 1 ? '' : 's'}</span>
        <span className="tabular-nums text-positive">+{totals.additions}</span>
        <span className="tabular-nums text-danger">−{totals.deletions}</span>
        <div className="flex-1" />
        <button
          onClick={() => setCollapsed((c) => (c.size ? new Set() : new Set(turns.flatMap((t) => t.files.map((f) => f.path)))))}
          className="rounded-md px-1.5 py-0.5 text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink"
        >
          {collapsed.size ? 'Expand all' : 'Collapse all'}
        </button>
      </div>
      <div ref={listRef} className="scroll-thin min-h-0 flex-1 overflow-y-auto pb-6">
        {turns.map((turn) => (
          <section key={turn.key}>
            <div className="flex items-baseline gap-1.5 px-4 pb-1 pt-4 text-[13px] leading-4 text-ink-3">
              <span className="shrink-0 text-ink-2">{turn.turn === null ? 'Other changes in this folder' : `Turn ${turn.turn}`}</span>
              {turn.prompt && <span className="min-w-0 truncate">{turn.prompt}</span>}
            </div>
            {turn.files.map((file) => {
              const open = !collapsed.has(file.path)
              const { dir, name } = splitPath(file.path)
              return (
                <div key={file.path} data-path={file.path} className="scroll-mt-2 border-b border-line">
                  <button
                    onClick={() => { setActive(file.path); toggle(file.path) }}
                    title={file.path}
                    className={clsx('flex h-8 w-full items-center gap-2 px-4 text-left text-[13px] transition-colors hover:bg-surface-hover', active === file.path && 'bg-surface')}
                  >
                    <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} className="shrink-0 text-ink-4" />
                    <FileGlyph path={file.path} size={16} />
                    <span className="min-w-0 flex-1 truncate">
                      <span className="text-ink-3">{dir}</span><span className="text-ink">{name}</span>
                    </span>
                    {file.status && <span className="shrink-0 text-[13px] text-ink-3">{STATUS_LABEL[file.status] ?? file.status}</span>}
                    <span className="shrink-0 text-[13px] tabular-nums text-positive">+{file.additions}</span>
                    <span className="shrink-0 text-[13px] tabular-nums text-danger">−{file.deletions}</span>
                  </button>
                  {open && <div className="pb-2">{fileBody(file)}</div>}
                </div>
              )
            })}
          </section>
        ))}
      </div>
    </div>
  )
}
