import { Icon } from '../../lib/icons'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Menu } from '../ui'
import { ProjectList } from './ProjectList'
import type { ComposerWorkspace } from './useComposerWorkspace'

export function ComposerWorkspaceBar({ work, disabled }: { work: ComposerWorkspace; disabled: boolean }) {
  const tx = useAuthoredText()
  const scope = work.overflowWorkspace
  if (!scope) return null
  const host = scope.workspace?.kind === 'ssh'
    ? scope.hosts.find(item => item.id === scope.workspace?.hostId)?.label || tx('SSH workspace')
    : scope.orchestratorLabel || tx('Local workspace')
  return <section aria-label={tx('New task workspace')} className="mb-3 flex flex-wrap items-center gap-1.5 text-[12px]">
    <Menu width={300} placement="auto" trigger={open => <button type="button" disabled={disabled}
      aria-label={tx('Choose task project')} aria-expanded={open}
      className="flex h-8 max-w-64 items-center gap-2 rounded-lg border border-line bg-card px-2.5 text-ink hover:bg-surface disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-accent">
      <Icon name="layers" size={14} className="shrink-0 text-ink-3" />
      <span className="truncate">{scope.projectId ? scope.projectLabel : tx('Choose project')}</span><Icon name="chevronDown" size={12} className="shrink-0 text-ink-3" />
    </button>}>
      {close => <ProjectList value={scope.projectId} projects={scope.projects} onSelect={scope.onSelectProject} onAdd={scope.onAddProject} close={close} />}
    </Menu>
    <Menu width={300} placement="auto" trigger={open => <button type="button" disabled={disabled}
      aria-label={tx('Choose task folder')} aria-expanded={open} title={work.workCwd || tx('Choose folder')}
      className="flex h-8 max-w-64 items-center gap-1.5 rounded-lg px-2 text-ink-2 hover:bg-surface disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-accent">
      <Icon name="folder" size={14} className="shrink-0 text-ink-3" /><span className="truncate">{scope.folderLabel || tx('Choose folder')}</span>
      <Icon name="chevronDown" size={12} className="shrink-0 text-ink-3" />
    </button>}>
      {close => scope.renderFolder(close)}
    </Menu>
    {scope.branch && <span className="flex min-w-0 max-w-44 items-center gap-1.5 px-2 text-ink-3" title={scope.branch}>
      <Icon name="branch" size={13} className="shrink-0" /><span className="truncate">{scope.branch}</span>
    </span>}
    <span className="ml-auto flex min-w-0 max-w-44 items-center gap-1.5 text-[11px] text-ink-3" title={host}>
      <Icon name={scope.workspace?.kind === 'ssh' || scope.orchestratorLabel ? 'globe' : 'monitor'} size={13} className="shrink-0" /><span className="truncate">{host}</span>
    </span>
  </section>
}
