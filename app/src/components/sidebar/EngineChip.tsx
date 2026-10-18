import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import clsx from 'clsx'
import { api } from '../../lib/ipc'
import { useEngine } from '../../lib/engine'
import type { EngineStatus } from '../../lib/types'
import { repoDisplayName } from '../../lib/turn-stats-line'

type ChipState = 'loaded' | 'loading' | 'stopped' | 'failed'

export function engineChipState(status: EngineStatus): ChipState {
  if (status.state === 'ready') return 'loaded'
  if (status.state === 'starting') return 'loading'
  if (status.state === 'failed') return 'failed'
  return 'stopped'
}

const STATE_LABEL: Record<ChipState, string> = { get loaded() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Loaded") }, get loading() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Loading") }, get stopped() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "No model loaded") }, get failed() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Stopped with an error") } }

// The built-in engine at the foot of the sidebar: which model is loaded, its state, and Eject.
export function EngineChip({ active, onOpen }: { active: boolean; onOpen: () => void }) {
  const tx = useAuthoredText()
  const { snapshot } = useEngine()
  const [ejecting, setEjecting] = useState(false)
  const status = snapshot?.status
  if (!status?.available) return null
  const state = engineChipState(status)
  const model = status.model ? snapshot?.installed.find((entry) => entry.repo === status.model)?.title ?? repoDisplayName(status.model) : null
  const canEject = !!status.model && (state === 'loaded' || state === 'loading')
  const eject = async () => {
    if (ejecting) return
    setEjecting(true)
    try { await api.engineUnload() } catch { /* the IPC layer shows the error */ } finally { setEjecting(false) }
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen() } }}
      title={status.model ? `${model}: ${STATE_LABEL[state].toLowerCase()}` : tx("Open Models")}
      className={clsx('no-drag group flex h-8 w-full cursor-default items-center gap-2 rounded-[10px] px-2.5 text-[13px] transition-colors', active ? 'bg-surface-hover' : 'hover:bg-surface')}
    >
      <span className="flex w-4 shrink-0 items-center justify-center">
        <span className={clsx('size-1.5 rounded-full', state === 'loaded' && 'bg-positive', state === 'loading' && 'animate-pulse bg-ink-3', state === 'stopped' && 'bg-ink-4', state === 'failed' && 'bg-danger')} />
      </span>
      <span className={clsx('min-w-0 flex-1 truncate', model ? 'text-ink' : 'text-ink-3')}>{model ?? STATE_LABEL.stopped}</span>
      {model && <span className={clsx('shrink-0 text-[13px] text-ink-3', canEject && 'group-hover:hidden')}>{STATE_LABEL[state]}</span>}
      {canEject && (
        <button
          onClick={(e) => { e.stopPropagation(); void eject() }}
          disabled={ejecting}
          className="hidden shrink-0 rounded-md px-1.5 text-[13px] text-ink-2 transition-colors hover:bg-surface-hover hover:text-ink disabled:opacity-50 group-hover:block"
        >
          {ejecting ? tx("Ejecting") : tx("Eject")}
        </button>
      )}
    </div>
  )
}
