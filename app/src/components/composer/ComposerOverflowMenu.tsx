import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { MemorySettings, type MemorySettingsProps } from './MemorySettings'
import { useState } from 'react'
import clsx from 'clsx'
import { Menu } from '../ui'
import { Icon, type IconName } from '../../lib/icons'
import type { Project, SshConnectionState, SshHost, WorkspaceLocation } from '../../lib/types'
import { ProjectList } from './ProjectList'
import { WorkspaceList } from './WorkspaceList'

export interface OverflowWorkspace {
  projects: Project[]
  projectId: string | null
  projectLabel: string
  onSelectProject: (id: string | null) => void
  onAddProject: (kind: 'scratch' | 'folder') => Promise<Project | null>
  folderLabel: string | null
  renderFolder: (close: () => void) => React.ReactNode
  workspace: WorkspaceLocation | null
  hosts: SshHost[]
  statusMap: Record<string, SshConnectionState>
  orchestratorLabel: string | null
  onLocal: () => void
  onHost: (host: SshHost, path: string) => void
  onAddHost: () => void
  branch: string | null
}

type View = 'main' | 'project' | 'folder' | 'workspace' | 'memory'

const ROW = 'flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[13px] leading-[18px] transition-colors'

function Row({ icon, label, value, onClick, nav, active, disabled }: { icon: IconName; label: string; value?: string | null; onClick?: () => void; nav?: boolean; active?: boolean; disabled?: boolean }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} className={clsx(ROW, disabled ? 'cursor-default text-ink-3' : 'text-ink hover:bg-ink-5')}>
      <Icon name={icon} size={16} className="shrink-0 text-ink-3" />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {value && <span className="max-w-[150px] shrink-0 truncate text-ink-3">{value}</span>}
      {active && <Icon name="check" size={14} className="shrink-0 text-accent" />}
      {nav && <Icon name="chevronRight" size={14} className="shrink-0 text-ink-4" />}
    </button>
  )
}

function Panel({ workspace, voice, incognito, memory, onUseSkill, modes, close }: ComposerOverflowProps & { close: () => void }) {
  const tx = useAuthoredText()
  const [view, setView] = useState<View>('main')
  if (view === 'memory' && memory) return <div><button type="button" onClick={() => setView('main')} className="mb-1 flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[13px] font-medium text-ink-2 hover:bg-ink-5"><Icon name="back" size={14} />{tx("Memory")}</button><MemorySettings {...memory} /></div>
  if (view !== 'main' && workspace) {
    const title = view === 'project' ? tx("Project") : view === 'folder' ? tx("Folder") : tx("Runs on")
    return (
      <div>
        <button type="button" onClick={() => setView('main')} className="mb-1 flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">
          <Icon name="back" size={14} />{title}
        </button>
        {view === 'project' && <ProjectList value={workspace.projectId} projects={workspace.projects} onSelect={workspace.onSelectProject} onAdd={workspace.onAddProject} close={close} />}
        {view === 'folder' && workspace.renderFolder(close)}
        {view === 'workspace' && <WorkspaceList workspace={workspace.workspace} hosts={workspace.hosts} statusMap={workspace.statusMap} onLocal={workspace.onLocal} onHost={workspace.onHost} onAddHost={workspace.onAddHost} close={close} />}
      </div>
    )
  }
  const host = workspace?.workspace?.kind === 'ssh' ? workspace.hosts.find((h) => h.id === workspace.workspace?.hostId) ?? null : null
  return (
    <div>
      {workspace && (
        <>
          <Row icon="layers" label={tx("Project")} value={workspace.projectLabel} nav onClick={() => setView('project')} />
          <Row icon="folder" label={tx("Folder")} value={workspace.folderLabel ?? 'None'} nav onClick={() => setView('folder')} />
          {workspace.orchestratorLabel
            ? <Row icon="globe" label={tx("Runs on")} value={workspace.orchestratorLabel} disabled />
            : <Row icon={host ? 'globe' : 'monitor'} label={tx("Runs on")} value={host ? host.label : 'This Mac'} nav onClick={() => setView('workspace')} />}
          {workspace.branch && <Row icon="branch" label={tx("Branch")} value={workspace.branch} disabled />}
          <div className="my-1 border-t border-line" />
        </>
      )}
      <Row icon="mic" label={voice.dictating ? tx("Stop voice input") : tx("Voice input")} value={voice.available ? null : 'Off in Settings'} disabled={!voice.available} onClick={() => { voice.onToggle(); close() }} />
      {incognito && <Row icon="incognito" label={tx("Incognito")} value={incognito.on ? null : 'Not saved'} active={incognito.on} onClick={() => incognito.onToggle()} />}
      {incognito?.on && <p className="px-2.5 pb-2 text-[12px] leading-relaxed text-ink-3">{tx("Chat history stays in TensorFold memory while the app is open; providers and tools you use may retain their own records.")}</p>}
      {memory && <Row icon="layers" label={tx("Memory")} nav value={memory.problem || !memory.policy ? 'Review choices' : memory.unsupported ? 'Unavailable' : `${memory.policy.useExisting ? 'Recall on' : 'Recall off'} · ${!memory.incognito && memory.policy.contribute ? 'Proposals on' : 'Proposals off'}`} onClick={() => setView('memory')} />}
      <Row icon="spark" label={tx("Use a skill")} onClick={() => { close(); onUseSkill() }} />
      {modes && (
        <>
          <div className="my-1 border-t border-line" />
          <Row icon="refresh" label={tx("Start a mission")} onClick={() => { close(); modes.onMission() }} />
          <Row icon="layers" label={tx("Rooms")} onClick={() => { close(); modes.onRooms() }} />
          <Row icon="bell" label={tx("Missions and loops")} onClick={() => { close(); modes.onMonitor() }} />
        </>
      )}
    </div>
  )
}

export interface ComposerOverflowProps {
  memory?: MemorySettingsProps
  /** Task composers only: project, folder, host and branch. */
  workspace?: OverflowWorkspace
  voice: { available: boolean; dictating: boolean; onToggle: () => void }
  /** Home composer only: the next task is not saved. */
  incognito?: { on: boolean; onToggle: () => void }
  onUseSkill: () => void
  modes?: { onMission: () => void; onRooms: () => void; onMonitor: () => void }
}

// The single "…" menu: everything the composer used to show as chips and toggles.
export function ComposerOverflowMenu(props: ComposerOverflowProps) {
  const tx = useAuthoredText()
  return (
    <Menu width={300} align="end" placement="top" trigger={(open) => (
      <button type="button" aria-label={tx("More options")} title={tx("More options")} className={clsx('flex h-8 w-8 items-center justify-center rounded-full transition-colors', open ? 'bg-ink-5 text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}>
        <Icon name="dots" size={16} />
      </button>
    )}>
      {(close) => <Panel {...props} close={close} />}
    </Menu>
  )
}
