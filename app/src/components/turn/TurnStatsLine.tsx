import { useStore } from '../../lib/store'
import { useEngine } from '../../lib/engine'
import { statsModelLabel, turnStatsSegments, turnStatsTitle } from '../../lib/turn-stats-line'
import type { AssistantTurnStats, Provider } from '../../lib/types'
import { CortheonNote } from './CortheonNote'

// A narrow column wraps between figures, never inside one.
function Line({ segments, title }: { segments: string[]; title?: string }) {
  return (
    <div className="text-[13px] leading-4 text-ink-4 tabular-nums" title={title}>
      {segments.map((segment, index) => <span key={index}>{index > 0 && ' · '}<span className="whitespace-nowrap">{segment}</span></span>)}
    </div>
  )
}

// The built-in engine names its models from the engine's own catalog titles.
function EngineStatsLine({ stats, provider, locale }: { stats: AssistantTurnStats; provider: Provider; locale?: string }) {
  const { snapshot } = useEngine()
  return <>
    <Line segments={turnStatsSegments(stats, snapshot?.installed.find((model) => model.repo === stats.model)?.title ?? statsModelLabel(stats.model, provider, snapshot?.suggestions), locale)} title={turnStatsTitle(stats, locale)} />
    <CortheonNote stats={stats} />
  </>
}

// One quiet line of measured facts under a final answer.
export function TurnStatsLine({ stats }: { stats: AssistantTurnStats }) {
  const locale = useStore(state => state.settings?.uiPreferences?.locale)
  const provider = useStore((s) => s.settings?.providers.find((candidate) => candidate.id === stats.providerId) ?? null)
  if (provider?.source === 'tensorfold') return <EngineStatsLine stats={stats} provider={provider} locale={locale} />
  return <Line segments={turnStatsSegments(stats, statsModelLabel(stats.model, provider), locale)} title={turnStatsTitle(stats, locale)} />
}
