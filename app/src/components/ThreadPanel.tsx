import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { Icon } from '../lib/icons'
import type { GitStatus, Thread } from '../lib/types'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../lib/thread-workspace'
import { changeTotals, threadChangeTurns } from '../lib/thread-changes'
import { WORK_PANE_TABS, resolveWorkPaneTab, workPaneOpen, workPaneTabsFor, type WorkPaneTab } from '../lib/work-pane'
import { ChangesTab } from './thread-panel/ChangesTab'
import { ThreadDetails } from './thread-panel/ThreadDetails'
import { Terminal } from './Terminal'
import { FileExplorer } from './FileExplorer'
import { Preview } from './Preview'

export { workingLine } from './thread-panel/primitives'
export { FilesChangedCard } from './thread-panel/FilesChangedCard'

const WIDTH_KEY = 'vontra.workPaneWidth'
const MIN_WIDTH = 340
const MAX_WIDTH = 760

function savedWidth(): number {
  try {
    const saved = Number(localStorage.getItem(WIDTH_KEY))
    return saved >= MIN_WIDTH && saved <= MAX_WIDTH ? saved : 440
  } catch { return 440 }
}

// The thread's right pane: somewhere to work (changes, terminal, files, preview) plus Details.
export function ThreadPanel({ thread, git, onChanged }: { thread: Thread; git: GitStatus | null; onChanged: () => void }) {
  const tx = useAuthoredText()
  const project = useStore((s) => thread.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const rightPanelHidden = useStore((s) => s.rightPanelHidden)
  const chatPaneOpenId = useStore((s) => s.chatPaneOpenId)
  const chosenTab = useStore((s) => s.workPaneTabs[thread.id])
  const setWorkPaneTab = useStore((s) => s.setWorkPaneTab)
  const closeWorkPane = useStore((s) => s.closeWorkPane)
  const workspace = effectiveThreadWorkspace(thread, project)
  const cwd = effectiveThreadCwd(thread, project)
  const hasFolder = thread.kind === 'task' && !!cwd
  const turns = useMemo(() => (hasFolder ? threadChangeTurns(thread, git, cwd) : []), [hasFolder, thread.timeline, git, cwd]) // eslint-disable-line react-hooks/exhaustive-deps
  const changeCount = changeTotals(turns).files
  const tab = resolveWorkPaneTab(chosenTab, { hasFolder, hasChanges: changeCount > 0 })

  const [width, setWidth] = useState(savedWidth)
  const narrow = useNarrowWindow()
  const widthRef = useRef(width)
  widthRef.current = width
  const startResize = (e: React.PointerEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = widthRef.current
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev: PointerEvent) => setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startW - (ev.clientX - startX))))
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      try { localStorage.setItem(WIDTH_KEY, String(widthRef.current)) } catch { /* ignore */ }
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  if (!workPaneOpen({ threadId: thread.id, hasFolder, rightPanelHidden, chatPaneOpenId })) return null
  const tabs = workPaneTabsFor(hasFolder)
  const pick = (next: WorkPaneTab) => setWorkPaneTab(thread.id, next)

  return (
    <aside
      style={{ width: narrow ? `min(${width}px, 88%)` : width }}
      className={clsx('flex h-full shrink-0 flex-col border-l border-line', narrow ? 'absolute inset-y-0 right-0 z-30 bg-canvas shadow-pop' : 'relative')}
      aria-label={tx("Work pane")}
    >
      <div onPointerDown={startResize} className="group absolute left-0 top-0 z-20 h-full w-1.5 -translate-x-1/2 cursor-col-resize" aria-label={tx("Resize work pane")}>
        <div className="absolute left-1/2 top-0 h-full w-px bg-transparent transition-colors group-hover:bg-line-strong" />
      </div>
      <div className="flex h-10 shrink-0 items-stretch gap-2 border-b border-line px-2">
        <div role="tablist" className="scroll-thin flex min-w-0 flex-1 items-stretch gap-3 overflow-x-auto">
        {WORK_PANE_TABS.filter((t) => tabs.includes(t.id)).map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => pick(t.id)}
            className={clsx('relative flex shrink-0 items-center gap-1.5 whitespace-nowrap text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-accent focus-visible:-outline-offset-2', tab === t.id ? 'text-ink' : 'text-ink-3 hover:text-ink-2')}
          >
            {t.label}
            {t.id === 'changes' && changeCount > 0 && <span className="text-[13px] tabular-nums text-ink-3">{changeCount}</span>}
            {tab === t.id && <span className="absolute inset-x-0 -bottom-px h-[1.5px] rounded-full bg-ink" />}
          </button>
        ))}
        </div>
        <button onClick={closeWorkPane} title={tx("Close pane (⌥⌘B)")} aria-label={tx("Close pane")} className="my-auto flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">
          <Icon name="x" size={16} />
        </button>
      </div>
      <div className="min-h-0 flex-1">
        {tab === 'changes' && <ChangesTab thread={thread} cwd={cwd} workspace={workspace} turns={turns} />}
        {tab === 'terminal' && <Terminal key={cwd} cwd={cwd} workspace={workspace} onClose={closeWorkPane} />}
        {tab === 'files' && <div className="flex h-full flex-col pt-3"><FileExplorer cwd={cwd} workspace={workspace} /></div>}
        {tab === 'preview' && <Preview onClose={closeWorkPane} />}
        {tab === 'details' && <div className="scroll-thin h-full overflow-y-auto"><ThreadDetails thread={thread} git={git} onChanged={onChanged} /></div>}
      </div>
    </aside>
  )
}

// Below this window width the conversation can't spare room for a docked pane, so the pane floats
// over it as a sheet instead of squeezing the transcript and composer.
const NARROW_QUERY = '(max-width: 1179px)'
function useNarrowWindow(): boolean {
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(NARROW_QUERY).matches)
  useEffect(() => {
    const media = window.matchMedia(NARROW_QUERY)
    // Going narrow folds the pane away once; it then opens on demand (title bar, Cmd+J, Review).
    const update = () => {
      setNarrow(media.matches)
      if (media.matches) useStore.setState({ rightPanelHidden: true })
    }
    if (media.matches) useStore.setState({ rightPanelHidden: true })
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  return narrow
}
