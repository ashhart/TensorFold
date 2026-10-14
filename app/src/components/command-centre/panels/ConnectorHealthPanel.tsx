import clsx from 'clsx'
import { formatDuration } from '../../../lib/format'
import type { AssistantControlSnapshot } from '../../../lib/types'
import { BrandIcon, connectorBrand } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button } from '../../ui'
import { EmptyState, Panel } from '../ui'
import type { CommandCentreSetView, ConnectorHealthSignal } from '../model'
import { connectorHealthBadgeClass, toneDot, toneText } from '../styles'

type Connector = AssistantControlSnapshot['connectors'][number]

interface ConnectorHealthPanelProps {
  connectorHealthRows: Array<{ connector: Connector; health: ConnectorHealthSignal }>
  connectorActionCount: number
  connectedCount: number
  connectorRows: { length: number }
  loading: boolean
  syncEmailAndRefresh: () => Promise<void>
  setView: CommandCentreSetView
}

export function ConnectorHealthPanel(props: ConnectorHealthPanelProps) {
  const { connectorHealthRows, connectorActionCount, connectedCount, connectorRows, loading, syncEmailAndRefresh, setView } = props

  return (
    <Panel title="Connector Health" icon="plug" action={connectorActionCount ? `${connectorActionCount} actions` : `${connectedCount}/${connectorRows.length || 0}`}>
      <div className="divide-y divide-line">
        {connectorHealthRows.length === 0 ? <EmptyState icon="plug" title="No connectors" detail={loading ? 'Loading connector snapshot.' : 'Connector contracts are unavailable.'} /> : connectorHealthRows.map(({ connector, health }) => (
          <div key={connector.id} className="px-3 py-2.5">
            <div className="flex items-start gap-2.5">
              <span className="relative shrink-0">
                <BrandIcon brand={connectorBrand(connector.id)} size={34} />
                <span className={clsx('absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-card', toneDot(health.tone))} />
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <div className="truncate text-[13px] font-semibold text-ink">{connector.name}</div>
                  <span className={clsx('ml-auto shrink-0 rounded-md px-1.5 py-0.5 text-[12px] font-medium', connectorHealthBadgeClass(health.tone))}>{health.label}</span>
                </div>
                <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{connector.description}</div>
                <div className="mt-2 grid grid-cols-3 gap-1 text-[12px] text-ink-4">
                  <span>{connector.unread} unread</span>
                  <span>{connector.pendingApprovals} approvals</span>
                  <span>{connector.tools.length} tools</span>
                </div>
                <div className="mt-2 rounded-md bg-card px-2 py-1.5 text-[13px] text-ink-3">
                  <div className="flex items-center gap-1.5">
                    <Icon name={health.tone === 'critical' ? 'warning' : health.tone === 'good' ? 'check' : 'clock'} size={12} className={toneText(health.tone)} />
                    <span className="min-w-0 flex-1 truncate">{health.detail}</span>
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate text-ink-4">Next: {health.nextAction}</span>
                    {health.action === 'sync-email' && <Button variant="default" size="xs" icon="mail" onClick={syncEmailAndRefresh} disabled={loading}>Sync email</Button>}
                    {health.action === 'open-sources' && <Button variant="ghost" size="xs" icon="plug" onClick={() => setView({ name: 'plugins' })}>Sources</Button>}
                    {health.action === 'review-approvals' && <Button variant="ghost" size="xs" icon="shield" onClick={() => document.getElementById('command-centre-approvals')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>Review</Button>}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

interface ControlGapsPanelProps {
  gapCount: number
  snapshot: AssistantControlSnapshot | null
}

export function ControlGapsPanel(props: ControlGapsPanelProps) {
  const { gapCount, snapshot } = props

  return (
    <Panel title="Control Gaps" icon="warning" action={gapCount ? `${gapCount} open` : 'clear'}>
      <div className="divide-y divide-line">
        {(snapshot?.gaps ?? []).length === 0 ? <EmptyState icon="check" title="No gaps detected" detail="Connector transport and policy gaps will be tracked here." /> : snapshot!.gaps.slice(0, 5).map((gap) => (
          <div key={gap.id} className="px-3 py-2.5">
            <div className={clsx('text-[13px] font-semibold', gap.severity === 'error' ? 'text-danger' : gap.severity === 'warning' ? 'text-warn' : 'text-ink')}>{gap.title}</div>
            <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{gap.detail}</div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

interface RunClassesPanelProps {
  snapshot: AssistantControlSnapshot | null
}

export function RunClassesPanel(props: RunClassesPanelProps) {
  const { snapshot } = props

  return (
    <Panel title="Run Classes" icon="cpu" action={`${snapshot?.budgets.length ?? 0} budgets`}>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line">
        {(snapshot?.budgets ?? []).slice(0, 6).map((budget) => (
          <div key={budget.executionClass} className="bg-card px-2.5 py-2">
            <div className="truncate text-[13px] font-semibold text-ink">{budget.label}</div>
            <div className="mt-0.5 text-[12px] text-ink-3">{formatDuration(budget.maxWallSeconds * 1000)}</div>
            <div className="mt-0.5 text-[12px] text-ink-4">{budget.requiresHeartbeat ? `${budget.heartbeatIntervalSeconds}s heartbeat` : 'no heartbeat'}</div>
          </div>
        ))}
      </div>
    </Panel>
  )
}
