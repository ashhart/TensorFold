import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { engineStateLabel, useEngine } from '../../lib/engine'
import type { EngineStatus } from '../../lib/types'

// Small engine indicators for the composer's model picker.

export function engineDotTone(status: EngineStatus | null | undefined): string {
  if (!status || !status.available) return 'bg-ink-4'
  if (status.state === 'ready') return 'bg-positive'
  if (status.state === 'starting') return 'bg-warn animate-pulse'
  if (status.state === 'failed') return 'bg-danger'
  return 'bg-ink-4'
}

export function EngineDot() {
  const { snapshot } = useEngine()
  return <span title={engineStateLabel(snapshot?.status)} className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', engineDotTone(snapshot?.status))} />
}

export function EngineStatusRow({ onOpenModels }: { onOpenModels: () => void }) {
  const tx = useAuthoredText()
  const { snapshot } = useEngine()
  const status = snapshot?.status
  return (
    <div data-testid="picker-engine-status" className="mb-1 flex items-center gap-2 rounded-lg bg-card-2 px-2.5 py-2 text-[13px] text-ink-2">
      <span className={clsx('h-2 w-2 shrink-0 rounded-full', engineDotTone(status))} />
      <span className="min-w-0 flex-1 truncate">{engineStateLabel(status)}</span>
      <button onClick={onOpenModels} className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink">
        <Icon name="cpu" size={11} />{tx("Models")}</button>
    </div>
  )
}
