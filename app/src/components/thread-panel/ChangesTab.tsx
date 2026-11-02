import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
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
const CONTROL_CLASS = 'rounded-md px-2 py-1 text-[12px] transition-colors hover:bg-surface-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent'

function splitPath(path: string): { dir: string; name: string } {
  const at = path.lastIndexOf('/')
  return at < 0 ? { dir: '', name: path } : { dir: path.slice(0, at), name: path.slice(at + 1) }
}

function matchFile(turns: ChangeTurn[], wanted: string): string | null {
  for (const turn of turns) for (const file of turn.files) {
    if (file.path === wanted || file.path.endsWith('/' + wanted) || wanted.endsWith('/' + file.path)) return file.path
  }
  return null
}

export function ChangesTab({ thread, cwd, workspace, turns }: { thread: Thread; cwd: string; workspace: WorkspaceLocation | null; turns: ChangeTurn[] }) {
  const tx = useAuthoredText()
  const focus = useStore((state) => state.workPaneFocus)
  const [diffs, setDiffs] = useState<Map<string, DiffLine[]> | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())
  const [lineLimits, setLineLimits] = useState<Map<string, number>>(() => new Map())
  const [active, setActive] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [wrap, setWrap] = useState(true)
  const listRef = useRef<HTMLDivElement>(null)
  const totals = changeTotals(turns)
  const gitPaths = useMemo(() => turns.flatMap((turn) => turn.files.filter((file) => file.status).map((file) => file.path)), [turns])
  const gitKey = turns.map((turn) => turn.files.map((file) => `${file.path}:${file.status}:${file.additions}:${file.deletions}`).join(',')).join('|')
  const visibleTurns = turns.map((turn) => ({ ...turn, files: turn.files.filter((file) => file.path.toLowerCase().includes(query.trim().toLowerCase())) })).filter((turn) => turn.files.length)
  const visibleFiles = visibleTurns.flatMap((turn) => turn.files)
  const allCollapsed = visibleFiles.length > 0 && visibleFiles.every((file) => collapsed.has(file.path))

  useEffect(() => {
    setCollapsed(new Set()); setLineLimits(new Map()); setActive(null); setQuery('')
  }, [thread.id])

  useEffect(() => {
    setLoadError(false)
    if (!cwd || gitPaths.length === 0) { setDiffs(new Map()); return }
    let alive = true
    setDiffs(null)
    api.gitDiff(cwd, gitPaths, workspace)
      .then((files) => { if (alive) setDiffs(new Map(files.map((file) => [file.path, parseUnifiedDiff(file.diff)]))) })
      .catch(() => { if (alive) { setLoadError(true); setDiffs(new Map()) } })
    return () => { alive = false }
  }, [thread.id, cwd, workspace, gitKey, refresh]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!focus || focus.threadId !== thread.id || focus.tab !== 'changes' || !focus.path) return
    const path = matchFile(turns, relativeToCwd(focus.path, cwd))
    if (!path) return
    setQuery(''); setActive(path)
    setCollapsed((current) => { const next = new Set(current); next.delete(path); return next })
    const id = requestAnimationFrame(() => {
      const element = listRef.current?.querySelector(`[data-path="${CSS.escape(path)}"]`)
      element?.scrollIntoView({ block: 'start' })
    })
    return () => cancelAnimationFrame(id)
  }, [focus?.nonce]) // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (path: string) => setCollapsed((current) => {
    const next = new Set(current)
    if (next.has(path)) next.delete(path); else next.add(path)
    return next
  })
  const toggleAll = () => setCollapsed((current) => {
    const next = new Set(current)
    for (const file of visibleFiles) { if (allCollapsed) next.delete(file.path); else next.add(file.path) }
    return next
  })

  const fileBody = (file: ChangedFile) => {
    const liveDiff = diffs?.get(file.path)
    const hasLiveDiff = !!liveDiff?.length
    const diff = hasLiveDiff ? liveDiff : file.editDiff
    if (!diff?.length) return <div className="px-3 py-4 text-[12px] leading-5 text-ink-3">{tx(diffs === null ? 'Loading current diff…' : loadError ? 'Current diff unavailable. Retry above to load it.' : 'No text diff is available for this file.')}</div>
    const shown = lineLimits.get(file.path) ?? (diff.length > BIG_DIFF_LINES ? 80 : diff.length)
    return (
      <>
        <div className="flex items-center gap-1.5 border-t border-line bg-card px-3 py-1.5 text-[11px] text-ink-3">
          <Icon name={hasLiveDiff ? 'branch' : 'clock'} size={12} />
          {tx(hasLiveDiff ? 'Current working tree' : 'Recorded edit from this conversation')}
        </div>
        <DiffView diff={diff.slice(0, shown)} wrap={wrap} label={tx('Diff for {path}', { path: file.path })} />
        {shown < diff.length && <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line bg-card px-3 py-2 text-[12px] text-ink-3">
          <span>{tx('Showing {shown} of {total} lines', { shown, total: diff.length })}</span>
          <button type="button" className={clsx(CONTROL_CLASS, 'text-accent')} onClick={() => setLineLimits((current) => new Map(current).set(file.path, shown + BIG_DIFF_LINES))}>{tx('Show next {count} lines', { count: Math.min(BIG_DIFF_LINES, diff.length - shown) })}</button>
        </div>}
      </>
    )
  }

  if (!turns.length) return (
    <div className="flex h-full flex-col items-center justify-center gap-2 px-6 pb-12 text-center">
      <span className="mb-1 rounded-xl border border-line bg-card p-3 text-ink-3"><Icon name="branch" size={22} /></span>
      <h3 className="text-[14px] font-medium text-ink">{tx('No changes yet')}</h3>
      <p className="max-w-64 text-[13px] leading-5 text-ink-3">{tx('File edits from this conversation will appear here, grouped by turn.')}</p>
    </div>
  )

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      <div className="shrink-0 space-y-2.5 border-b border-line bg-card px-3 py-3">
        <div className="flex flex-wrap items-center gap-2 text-[13px]">
          <span className="font-medium text-ink">{tx('Review changes')}</span>
          <span className="rounded bg-surface px-1.5 text-[12px] text-ink-3">{tx('Files: {count}', { count: totals.files })}</span>
          <span className="ml-auto font-mono text-[12px] tabular-nums text-positive">+{totals.additions}</span>
          <span className="font-mono text-[12px] tabular-nums text-danger">−{totals.deletions}</span>
        </div>
        <label className="flex items-center gap-2 rounded-lg border border-line bg-canvas px-2.5 focus-within:border-accent focus-within:ring-1 focus-within:ring-accent/20">
          <Icon name="search" size={14} className="shrink-0 text-ink-3" />
          <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tx('Filter changed files…')} aria-label={tx('Filter changed files')} className="h-8 min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-4" />
        </label>
        <div className="flex flex-wrap items-center gap-1">
          <button type="button" aria-pressed={wrap} onClick={() => setWrap(!wrap)} className={clsx(CONTROL_CLASS, wrap ? 'bg-accent-soft text-accent' : 'text-ink-3')}>{tx('Wrap lines')}</button>
          <button type="button" disabled={!visibleFiles.length} onClick={toggleAll} className={clsx(CONTROL_CLASS, 'text-ink-3 disabled:opacity-40')}>{tx(allCollapsed ? 'Expand all' : 'Collapse all')}</button>
          <button type="button" disabled={diffs === null} onClick={() => setRefresh((value) => value + 1)} className={clsx(CONTROL_CLASS, 'text-ink-3 disabled:opacity-40')}>{tx('Refresh')}</button>
          {query.trim() && <span role="status" className="ml-auto text-[11px] text-ink-3">{tx('{shown} of {total} files', { shown: visibleFiles.length, total: totals.files })}</span>}
        </div>
      </div>
      {diffs === null && <div role="status" className="shrink-0 border-b border-line px-3 py-2 text-[12px] text-ink-3">{tx('Loading current changes…')}</div>}
      {loadError && <div role="alert" className="flex shrink-0 items-start gap-2 border-b border-danger/20 bg-danger/8 px-3 py-2 text-[12px] leading-5 text-ink-2">
        <Icon name="warning" size={14} className="mt-0.5 shrink-0 text-danger" />
        <span className="min-w-0 flex-1">{tx('Could not load current changes. Recorded edits remain available.')}</span>
        <button type="button" onClick={() => setRefresh((value) => value + 1)} className={clsx(CONTROL_CLASS, 'shrink-0 text-accent')}>{tx('Retry')}</button>
      </div>}
      <div ref={listRef} className="scroll-thin min-h-0 min-w-0 flex-1 overflow-y-auto pb-6">
        {!visibleFiles.length && <div className="px-4 py-8 text-center text-[13px] text-ink-3">{tx('No changed files match your filter.')}<button type="button" onClick={() => setQuery('')} className={clsx(CONTROL_CLASS, 'mx-auto mt-2 block text-accent')}>{tx('Clear filter')}</button></div>}
        {visibleTurns.map((turn) => (
          <section key={turn.key}>
            <div className="border-b border-line bg-surface px-3 py-2.5 text-[12px] leading-5">
              <div className="font-medium text-ink-2">{turn.turn === null ? tx('Other changes in this folder') : tx('Turn {value0}', { value0: turn.turn })}</div>
              {turn.prompt && <p title={turn.prompt} className="mt-0.5 truncate text-ink-3">{turn.prompt}</p>}
            </div>
            {turn.files.map((file) => {
              const open = !collapsed.has(file.path)
              const { dir, name } = splitPath(file.path)
              const bodyId = `change-${encodeURIComponent(thread.id)}-${encodeURIComponent(file.path)}`
              return (
                <div key={file.path} data-path={file.path} className="scroll-mt-2 border-b border-line">
                  <button type="button" onClick={() => { setActive(file.path); toggle(file.path) }} title={file.path} aria-expanded={open} aria-controls={bodyId} className={clsx('flex w-full items-center gap-2 px-3 py-2.5 text-left text-[13px] transition-colors hover:bg-surface-hover focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent', active === file.path && 'bg-surface')}>
                    <Icon name={open ? 'chevronDown' : 'chevronRight'} size={13} className="shrink-0 text-ink-3" />
                    <FileGlyph path={file.path} size={16} />
                    <span className="min-w-0 flex-1"><span className="block truncate font-medium text-ink">{name}</span>{dir && <span className="block truncate text-[11px] text-ink-3">{dir}</span>}</span>
                    <span className="shrink-0 text-right">
                      <span className="flex justify-end gap-1.5 font-mono text-[11px] tabular-nums"><span className="text-positive">+{file.additions}</span><span className="text-danger">−{file.deletions}</span></span>
                      {file.status && <span className="block text-[11px] text-ink-3">{tx(STATUS_LABEL[file.status] ?? file.status)}</span>}
                    </span>
                  </button>
                  {open && <div id={bodyId} className="min-w-0 pb-2">{fileBody(file)}</div>}
                </div>
              )
            })}
          </section>
        ))}
      </div>
    </div>
  )
}
