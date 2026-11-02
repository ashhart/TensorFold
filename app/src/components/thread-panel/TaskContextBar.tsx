import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { basename } from '../../lib/format'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'

interface Props {
  cwd: string
  projectName?: string
  remote: boolean
  branch: string | null
  gitAvailable: boolean
  changes: { files: number; additions: number; deletions: number }
  onWorkspace: () => void
  onReview: () => void
  onTerminal: () => void
}

export function TaskContextBar({ cwd, projectName, remote, branch, gitAvailable, changes, onWorkspace, onReview, onTerminal }: Props) {
  const tx = useAuthoredText()
  return <section aria-label={tx('Task workspace')} className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-card/50 px-4 py-2.5">
    <button type="button" onClick={onWorkspace} title={cwd}
      className="flex min-w-0 flex-1 basis-32 items-center gap-2 rounded-lg py-1 text-left focus-visible:outline-2 focus-visible:outline-accent">
      <Icon name="folder" size={16} className="shrink-0 text-ink-3" />
      <span className="min-w-0">
        <span className="block truncate text-[12px] font-medium text-ink">{projectName || basename(cwd) || tx('Choose a workspace')}</span>
        <span className="mt-0.5 block truncate text-[11px] text-ink-3">{remote ? tx('Remote workspace') : tx('Local workspace')} · {basename(cwd)}</span>
      </span>
      <Icon name="chevronDown" size={12} className="shrink-0 text-ink-3" />
    </button>
    <div className="flex min-w-0 max-w-48 items-center gap-1.5 text-[11px] text-ink-2" title={branch || undefined}>
      <Icon name="branch" size={14} className="shrink-0 text-ink-3" />
      <span className="truncate">{branch || (gitAvailable ? tx('No branch reported') : tx('Git status unavailable'))}</span>
    </div>
    <div className="flex items-center gap-1.5">
      <button type="button" onClick={onTerminal} aria-label={tx('Open workspace terminal')} title={tx('Terminal')}
        className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-2 hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent">
        <Icon name="terminal" size={15} />
      </button>
      <button type="button" onClick={onReview}
        className={clsx('flex h-8 items-center gap-2 rounded-lg border px-2.5 text-[12px] font-medium focus-visible:outline-2 focus-visible:outline-accent',
          changes.files ? 'border-accent/25 bg-accent/10 text-ink hover:bg-accent/15' : 'border-line text-ink-2 hover:bg-surface-hover')}>
        <Icon name="fileText" size={14} />
        <span>{tx('Review changes')}</span>
        {changes.files > 0 && <span className="rounded bg-surface-hover px-1.5 py-0.5 text-[10px] tabular-nums">{changes.files}</span>}
        {(changes.additions > 0 || changes.deletions > 0) && <span className="hidden gap-1.5 text-[11px] tabular-nums min-[900px]:flex">
          <span className="text-positive">+{changes.additions}</span><span className="text-danger">−{changes.deletions}</span>
        </span>}
      </button>
    </div>
  </section>
}
