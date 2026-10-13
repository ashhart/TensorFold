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
function EngineStatsLine({ stats, provider }: { stats: AssistantTurnStats; provider: Provider }) {
  const { snapshot } = useEngine()
  return <>
    <Line segments={turnStatsSegments(stats, statsModelLabel(stats.model, provider, snapshot?.suggestions))} title={turnStatsTitle(stats)} />
    <CortheonNote stats={stats} />
  </>
}

// One quiet line of measured facts under a final answer.
export function TurnStatsLine({ stats }: { stats: AssistantTurnStats }) {
  const provider = useStore((s) => s.settings?.providers.find((candidate) => candidate.id === stats.providerId) ?? null)
  if (provider?.source === 'tensorfold') return <EngineStatsLine stats={stats} provider={provider} />
  return <Line segments={turnStatsSegments(stats, statsModelLabel(stats.model, provider))} title={turnStatsTitle(stats)} />
}
