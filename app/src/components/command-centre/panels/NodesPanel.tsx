import clsx from 'clsx'
import { formatEvidenceAge, type NodeDashboard } from '../../../lib/node-dashboard'
import { Panel } from '../ui'

interface NodesPanelProps {
  nodeDashboard: NodeDashboard
}

export function NodesPanel(props: NodesPanelProps) {
  const { nodeDashboard } = props

  return (
    <Panel title="Nodes" icon="monitor" action={`${nodeDashboard.nodes.filter((n) => n.health === 'online').length}/${nodeDashboard.nodes.length} online`}>
      <div className="space-y-2 px-3 py-3">
        {nodeDashboard.failures.length > 0 && (
          <div className="rounded-lg border border-danger/30 bg-danger/5 px-2.5 py-2 text-[13px] leading-snug text-danger">
            {nodeDashboard.failures.slice(0, 3).map((failure) => <div key={failure} className="truncate">{failure}</div>)}
          </div>
        )}
        <div className="divide-y divide-line rounded-lg border border-line">
          {nodeDashboard.nodes.map((node) => (
            <div key={node.id} className="flex items-center gap-2.5 px-2.5 py-2">
              <span className={clsx('h-2 w-2 shrink-0 rounded-full', node.health === 'online' ? 'bg-positive' : node.health === 'degraded' ? 'bg-warn' : node.health === 'offline' ? 'bg-danger' : 'bg-ink-4')} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="min-w-0 truncate text-[13px] font-medium text-ink">{node.name}</span>
                  <span className="shrink-0 rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-4">{node.kind === 'cli-subscription' ? 'CLI' : node.kind === 'cloud-provider' ? 'cloud' : node.kind}</span>
                  {node.latencyMs != null && <span className="shrink-0 text-[11px] tabular-nums text-ink-4">{node.latencyMs}ms</span>}
                  {node.evidenceAgeMs != null && node.kind !== 'local' && <span className="shrink-0 text-[11px] tabular-nums text-ink-4">{formatEvidenceAge(node.evidenceAgeMs)}</span>}
                </div>
                <div className="mt-0.5 truncate text-[12px] leading-snug text-ink-3">{node.detail}</div>
              </div>
              {node.pendingApprovals != null && node.pendingApprovals > 0 && (
                <span className="shrink-0 rounded-md bg-warn/15 px-1.5 py-0.5 text-[11px] font-medium text-warn">{node.pendingApprovals} approvals</span>
              )}
              {node.terminals != null && node.terminals.total > 0 && (
                <span className={clsx('shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium', node.terminals.attention ? 'bg-warn/15 text-warn' : 'bg-ink-5 text-ink-3')}>
                  {node.terminals.running}/{node.terminals.total} terminals
                </span>
              )}
            </div>
          ))}
        </div>
        {nodeDashboard.usage.totalTokens > 0 && (
          <div className="px-1 text-[12px] text-ink-4">
            {Math.round(nodeDashboard.usage.totalTokens / 1000)}k tokens across {Object.keys(nodeDashboard.usage.byAccount).length || 1} account{Object.keys(nodeDashboard.usage.byAccount).length === 1 ? '' : 's'}
          </div>
        )}
      </div>
    </Panel>
  )
}
