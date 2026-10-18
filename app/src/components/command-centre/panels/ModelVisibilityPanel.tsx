import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { relativeTime } from '../../../lib/format'
import type { ModelRouterSnapshot } from '../../../lib/types'
import { Icon } from '../../../lib/icons'
import { EmptyState, Panel, ProviderSignalBlock, StatusIcon } from '../ui'
import { scheduleTime } from '../brief'
import { cap, type ProviderVisibilityRow } from '../model'

interface ModelVisibilityPanelProps {
  providerVisibilityRows: ProviderVisibilityRow[]
  activeProviderCount: number
  modelRouterSnapshot: ModelRouterSnapshot | null
}

export function ModelVisibilityPanel(props: ModelVisibilityPanelProps) {
  const tx = useAuthoredText()
  const { providerVisibilityRows, activeProviderCount, modelRouterSnapshot } = props

  return (
    <Panel title={tx("Model Visibility")} icon="cpu" action={activeProviderCount ? `${activeProviderCount} active` : 'default'}>
      <div className="space-y-2 px-3 py-3">
        {providerVisibilityRows.length === 0 ? (
          <EmptyState icon="cpu" title={tx("No model route")} detail="Choose a provider and model in Settings to populate this view." />
        ) : providerVisibilityRows.map((row) => (
          <div key={row.id} className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
            <div className="flex items-start gap-2.5">
              <StatusIcon icon="cpu" tone={row.health.tone} pulse={row.activeCount > 0} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <div className="min-w-0 flex-1 truncate text-[13px] font-semibold text-ink">{row.providerName}</div>
                  <span className="shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[11px] font-medium text-ink-3">{row.label}</span>
                  <span className="shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[11px] font-medium text-ink-4">{row.providerKind}</span>
                </div>
                <div className="mt-0.5 truncate font-mono text-[12px] text-ink-3">{row.model}</div>
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px] text-ink-4">
                  <span className="rounded bg-ink-5 px-1.5 py-0.5">{row.account}</span>
                  <span className="min-w-0 truncate">{row.activity}</span>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-1.5">
                  <ProviderSignalBlock signal={row.health} />
                  <ProviderSignalBlock signal={row.route} />
                  <ProviderSignalBlock signal={row.context} />
                  <ProviderSignalBlock signal={row.latency} />
                  <ProviderSignalBlock signal={row.spend} />
                </div>
              </div>
            </div>
          </div>
        ))}
        {(modelRouterSnapshot?.history.length ?? 0) > 0 && (
          <div className="rounded-lg border border-line bg-card px-3 py-2.5">
            <div className="flex items-center gap-2">
              <Icon name="arrowRight" size={13} className="text-ink-3" />
              <div className="text-[13px] font-semibold text-ink">{tx("Recent failovers")}</div>
              <span className="ml-auto text-[12px] text-ink-4">{modelRouterSnapshot?.history.length ?? 0} {tx("recorded")}</span>
            </div>
            <div className="mt-2 space-y-1.5">
              {modelRouterSnapshot!.history.slice(0, 4).map((entry) => (
                <div key={entry.id} className="rounded-md bg-card-2 px-2 py-1.5">
                  <div className="flex items-center gap-1.5 text-[13px]">
                    <span className="shrink-0 rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-3">{cap(entry.scope.replace('_', ' '))}</span>
                    <span className="min-w-0 flex-1 truncate font-medium text-ink">{entry.subjectLabel || entry.providerName}</span>
                    <span className="shrink-0 text-[12px] text-ink-4">{relativeTime(entry.createdAt)}</span>
                  </div>
                  <div className="mt-1 flex items-center gap-1.5 text-[12px] text-ink-3">
                    <span className="min-w-0 truncate">{entry.fromAccountName}</span>
                    <Icon name="arrowRight" size={11} className="shrink-0 text-ink-4" />
                    <span className="min-w-0 truncate text-ink">{entry.toAccountName}</span>
                    <span className="shrink-0 rounded bg-ink-5 px-1 py-0.5 font-mono text-[11px] text-ink-4">{entry.model}</span>
                  </div>
                  <div className="mt-1 flex items-center gap-2 text-[12px] text-ink-4">
                    <span className="min-w-0 flex-1 truncate">{entry.reason}</span>
                    <span className="shrink-0">{entry.resetAt ? tx("reset {value0}", { value0: entry.resetAt > Date.now() ? scheduleTime(entry.resetAt) : relativeTime(entry.resetAt) }) : tx("no reset")}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Panel>
  )
}
