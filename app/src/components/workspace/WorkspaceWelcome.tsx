import { useMemo } from 'react'
import { useStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Icon } from '../../lib/icons'
import { compactAge } from '../../lib/thread-row'
import { basename } from '../../lib/format'
import { BrandMark } from '../BrandMark'

export function WorkspaceWelcome() {
  const tx = useAuthoredText()
  return <header>
    <div className="flex items-center gap-2.5 text-[12px] font-medium text-ink-2">
      <BrandMark size={26} /><span>{tx('New task')}</span>
    </div>
    <h1 className="mt-4 text-[clamp(22px,2.5vw,28px)] font-medium leading-tight tracking-tight text-ink">{tx('What would you like to work on?')}</h1>
    <p className="mt-2 text-[13px] leading-relaxed text-ink-3">{tx('Choose a project and model, then describe the outcome.')}</p>
  </header>
}

export function RecentWorkspaceTasks() {
  const tx = useAuthoredText()
  const threads = useStore(state => state.threads)
  const projects = useStore(state => state.projects)
  const actions = useStore.getState()
  const recent = useMemo(() => threads.filter(thread => thread.kind === 'task' && !thread.incognito && !thread.archivedAt)
    .sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 3), [threads])
  if (!recent.length) return null
  return <section aria-label={tx('Recent tasks')} className="mt-8">
      <div className="mb-2 flex items-center justify-between gap-3">
        <h2 className="text-[12px] font-medium text-ink-2">{tx('Continue working')}</h2>
        <button type="button" onClick={() => actions.setView({ name: 'projects' })} className="rounded-md px-2 py-1 text-[12px] text-ink-3 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">{tx('All projects')}</button>
      </div>
      <ul className="divide-y divide-line/60">
        {recent.map(thread => {
          const project = projects.find(candidate => candidate.id === thread.projectId)
          return <li key={thread.id}><button type="button" onClick={() => actions.openThread(thread.id)}
            className="flex w-full items-center gap-3 rounded-lg px-2 py-3 text-left hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent">
            <Icon name="message" size={15} className="shrink-0 text-ink-3" />
            <span className="min-w-0 flex-1"><span className="block truncate text-[12px] text-ink">{thread.title || tx('Untitled task')}</span>
              <span className="mt-0.5 block truncate text-[11px] text-ink-3">{project?.name || basename(thread.cwd || '') || tx('Task')}</span></span>
            <span className="shrink-0 text-[11px] tabular-nums text-ink-3">{compactAge(thread.updatedAt, Date.now())}</span>
            <Icon name="arrowRight" size={13} className="shrink-0 text-ink-3" />
          </button></li>
        })}
      </ul>
  </section>
}
