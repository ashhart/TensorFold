import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import clsx from 'clsx'
import { api } from '../../lib/ipc'
import { useEngine } from '../../lib/engine'
import type { EngineStatus } from '../../lib/types'
import { repoDisplayName } from '../../lib/turn-stats-line'

type ChipState = 'loaded' | 'loading' | 'stopped' | 'failed' | 'unavailable'

export function engineChipState(status: EngineStatus | null | undefined, error?: string | null): ChipState {
  if (error != null || !status?.available) return 'unavailable'
  if (status.state === 'ready' && status.model?.trim()) return 'loaded'
  if (status.state === 'starting') return 'loading'
  if (status.state === 'failed') return 'failed'
  return 'stopped'
}

const STATE_LABEL: Record<ChipState, string> = {
  get loaded() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, 'Loaded') },
  get loading() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, 'Loading') },
  get stopped() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, 'No model loaded') },
  get failed() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, 'Stopped with an error') },
  get unavailable() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, 'Engine status unavailable') },
}

// Model navigation and Eject are sibling buttons so both remain reachable with the keyboard.
export function EngineChip({ active, onOpen }: { active: boolean; onOpen: () => void }) {
  const tx = useAuthoredText()
  const { snapshot, error } = useEngine()
  const [ejecting, setEjecting] = useState(false)
  const status = snapshot?.status
  const state = engineChipState(status, error)
  const label = !status && error == null ? tx('Checking engine') : STATE_LABEL[state]
  const model = state !== 'unavailable' && status?.model?.trim() ? snapshot?.installed.find((entry) => entry.repo === status.model)?.title?.trim() || repoDisplayName(status.model) : null
  const canEject = !!status?.model?.trim() && (state === 'loaded' || state === 'loading')
  const eject = async () => {
    if (ejecting || !canEject) return
    setEjecting(true)
    try { await api.engineUnload() } catch { /* the IPC layer shows the error */ } finally { setEjecting(false) }
  }

  return (
    <div data-testid="sidebar-engine-status" className={clsx('no-drag group flex h-8 w-full items-center rounded-[10px] text-[13px] transition-colors', active ? 'bg-surface-hover' : 'hover:bg-surface')}>
      <button type="button" onClick={onOpen} title={model ? `${model}: ${label}` : tx('Open Models')} className="flex h-full min-w-0 flex-1 items-center gap-2 rounded-[10px] px-2.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-accent">
        <span className="flex w-4 shrink-0 items-center justify-center">
          <span className={clsx('size-1.5 rounded-full', state === 'loaded' && 'bg-positive', state === 'loading' && 'animate-pulse bg-ink-3', (state === 'stopped' || state === 'unavailable' && error == null) && 'bg-ink-4', (state === 'failed' || error != null) && 'bg-danger')} />
        </span>
        <span className={clsx('min-w-0 flex-1 truncate', model ? 'text-ink' : 'text-ink-3')}>{model ?? label}</span>
        {model && <span className="shrink-0 text-[12px] text-ink-3">{label}</span>}
      </button>
      {canEject && <button type="button" onClick={() => { void eject() }} disabled={ejecting} className="mr-1 shrink-0 rounded-md px-1.5 py-1 text-[12px] text-ink-2 opacity-0 transition-opacity hover:bg-surface-hover hover:text-ink focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50 group-hover:opacity-100 group-focus-within:opacity-100">
        {ejecting ? tx('Ejecting') : tx('Eject')}
      </button>}
    </div>
  )
}
