import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import { effectiveThreadWorkspace } from '../../lib/thread-workspace'
import type { Thread } from '../../lib/types'
import { PanelSection } from './primitives'

// Environment section: where this thread runs (Local / Orchestrator host /
// SSH), which model drives it, and the approval mode — with a gear into Settings.
export function EnvironmentSection({ thread }: { thread: Thread }) {
  const settings = useStore((s) => s.settings)
  const setView = useStore((s) => s.setView)
  const project = useStore((s) => thread.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const workspace = effectiveThreadWorkspace(thread, project)
  const host = thread.orchestratorHostId
    ? (settings?.remoteOrchestrators ?? []).find((h) => h.id === thread.orchestratorHostId)
    : null
  const sshName = workspace?.kind === 'ssh'
    ? (settings?.sshHosts ?? []).find((h) => h.id === workspace.hostId)?.label ?? 'SSH host'
    : null
  const target = sshName
    ? { label: `SSH · ${sshName}`, ok: true }
    : thread.orchestratorHostId
      ? { label: host?.name ?? 'Orchestrator', ok: !!host?.enabled }
      : { label: 'Local', ok: true }
  const provider = settings?.providers.find((p) => p.id === (thread.providerId ?? settings?.activeProviderId))
  const model = thread.model ?? settings?.activeModel ?? null
  return (
    <PanelSection
      title="Environment"
      right={<button onClick={() => setView({ name: 'settings', section: 'orchestrator' })} className="flex h-5 w-5 items-center justify-center rounded text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink" aria-label="Environment settings"><Icon name="settings" size={13} /></button>}
    >
      <div className="space-y-0.5">
        <div className="flex h-7 items-center gap-2 rounded-lg px-1.5 text-[13px] text-ink/85">
          <Icon name="globe" size={14} className="shrink-0 text-ink-3" />
          <span className="min-w-0 flex-1 truncate">{target.label}</span>
          <span className={clsx('size-[6px] shrink-0 rounded-full', target.ok ? 'bg-positive' : 'bg-danger')} />
        </div>
        {model && (
          <div className="flex h-7 items-center gap-2 rounded-lg px-1.5 text-[13px] text-ink/85">
            <Icon name="spark" size={14} className="shrink-0 text-ink-3" />
            <span className="min-w-0 flex-1 truncate">{provider ? `${provider.name} · ${model}` : model}</span>
          </div>
        )}
        {settings && (
          <div className="flex h-7 items-center gap-2 rounded-lg px-1.5 text-[13px] text-ink/85">
            <Icon name="shield" size={14} className="shrink-0 text-ink-3" />
            <span className="min-w-0 flex-1 truncate capitalize">{String(settings.approvalMode ?? 'ask').replace(/_/g, ' ')}</span>
          </div>
        )}
      </div>
    </PanelSection>
  )
}
