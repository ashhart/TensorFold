import { relativeTime } from '../../../lib/format'
import type { AssistantControlSnapshot, Automation, Thread } from '../../../lib/types'
import { Icon } from '../../../lib/icons'
import { Button } from '../../ui'
import type { AttentionRow, CommandCentreSetView } from '../model'

interface HeaderBarProps {
  snapshot: AssistantControlSnapshot | null
  attentionRows: AttentionRow[]
  pendingApprovals: number
  liveJobs: { length: number }
  activeMissions: number
  connectedCount: number
  connectorRows: { length: number }
  readinessLabel: string
  refresh: () => Promise<void>
  loading: boolean
  latestThread: Thread | null
  setView: CommandCentreSetView
  primaryBrief: Automation | null
  runAutomation: (id: string) => Promise<void>
  error: string
}

export function HeaderBar(props: HeaderBarProps) {
  const {
    snapshot,
    attentionRows,
    pendingApprovals,
    liveJobs,
    activeMissions,
    connectedCount,
    connectorRows,
    readinessLabel,
    refresh,
    loading,
    latestThread,
    setView,
    primaryBrief,
    runAutomation,
    error,
  } = props

  return (
    <div className="border-b border-line px-6 py-4 xl:pr-[328px]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="mb-1 flex items-center gap-2 text-[13px] text-ink-3">
            <Icon name="bell" size={14} />
            <span>Operations</span>
            <span className="text-ink-5">/</span>
            <span>{snapshot ? `Snapshot ${relativeTime(snapshot.generatedAt)}` : 'Loading snapshot'}</span>
          </div>
          <h1 className="text-[20px] font-semibold tracking-[-0.02em] text-ink">Assistant</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[13px] text-ink-3">
            <span>{attentionRows.length} attention items</span>
            <span className="text-ink-5">/</span>
            <span>{pendingApprovals} approvals</span>
            <span className="text-ink-5">/</span>
            <span>{liveJobs.length} live jobs</span>
            <span className="text-ink-5">/</span>
            <span>{activeMissions} missions</span>
            <span className="text-ink-5">/</span>
            <span>{connectedCount}/{connectorRows.length} connectors active</span>
            <span className="text-ink-5">/</span>
            <span>{readinessLabel}</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="default" size="sm" icon="clock" onClick={refresh} disabled={loading}>{loading ? 'Refreshing' : 'Refresh'}</Button>
          <Button variant="default" size="sm" icon="archive" onClick={() => latestThread && window.dispatchEvent(new CustomEvent('vontra:export', { detail: { threadId: latestThread.id } }))} disabled={!latestThread}>Export Latest</Button>
          <Button variant="default" size="sm" icon="play" onClick={() => window.dispatchEvent(new CustomEvent('vontra:replay'))}>Replay Bundle</Button>
          <Button variant="default" size="sm" icon="plug" aria-label="Open Sources" onClick={() => setView({ name: 'plugins' })}>Sources</Button>
          <Button variant="primary" size="sm" icon="spark" onClick={() => primaryBrief ? void runAutomation(primaryBrief.id) : setView({ name: 'automations' })}>Run Brief</Button>
        </div>
      </div>
      {error && <div className="mt-3 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger">{error}</div>}
    </div>
  )
}
