import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { changeDesktopMode } from '../lib/desktop-mode'
import { ENGINE_PROVIDER_ID, loadEngineModel, useEngine } from '../lib/engine'
import { useLearning, learningBusy } from '../lib/learning'
import { useStore } from '../lib/store'
import { BrandMark } from './BrandMark'
import { Button } from './ui'
import { Models } from './Models'
import { EngineRuntimePanel } from './models/EngineRuntimePanel'
import { EngineApiPanel } from './models/EngineApiPanel'

export type MinimalTarget = 'overview' | 'models' | 'api'
const TABS = [{ id: 'overview', label: 'Server' }, { id: 'models', label: 'Models' }, { id: 'api', label: 'API connection' }] as const

export function MinimalWorkspace({ target, onTarget }: { target: MinimalTarget; onTarget: (target: MinimalTarget) => void }) {
  const { snapshot } = useEngine()
  const { snapshot: learning } = useLearning()
  const settings = useStore((s) => s.settings)
  const threads = useStore((s) => s.threads)
  const statuses = useStore((s) => s.status)
  const activeRuns = threads.filter((t) => t.providerId === ENGINE_PROVIDER_ID && !t.orchestratorHostId && t.workspace?.kind !== 'ssh' && ['thinking', 'streaming', 'tool', 'awaiting'].includes(statuses[t.id] ?? (t.status === 'running' ? 'thinking' : 'idle'))).length
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => setError(''), [target])
  const action = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await fn() } catch (e) { setError(String((e as Error)?.message ?? e)) } finally { setBusy(false) }
  }
  const selected = settings?.activeProviderId === ENGINE_PROVIDER_ID ? snapshot?.installed.find((m) => m.repo === settings.activeModel) : undefined
  const start = async () => {
    if (!selected) { onTarget('models'); return }
    if (activeRuns || learningBusy(learning)) throw new Error('Finish active local work before changing the server model.')
    await loadEngineModel(selected.repo)
  }
  return <div data-testid="minimal-workspace" className="flex h-full w-full min-w-0 flex-col bg-canvas text-ink">
    <header className="drag flex min-h-12 shrink-0 flex-wrap items-center justify-end gap-2 px-4 pb-2 pl-[96px] pt-2">
      <Button size="sm" variant="ghost" className="no-drag" disabled={busy} icon="settings" onClick={() => useStore.getState().setView({ name: 'settings', section: 'app' })}>Settings</Button>
      <Button size="sm" className="no-drag" disabled={busy} onClick={() => void action(() => changeDesktopMode('full'))}>Open Full</Button>
    </header>
    <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-[980px] px-4 pb-6 pt-4 sm:px-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3"><BrandMark size={36} tile /><div><h1 className="text-[20px] font-semibold tracking-tight">TensorFold Minimal</h1><p className="mt-1 text-[13px] text-ink-2">Your local model server for the tools you use.</p></div></div>
          <Button size="sm" disabled={busy || api.platform === 'browser' || api.platform === 'web-remote'} onClick={() => void action(async () => { const result = await api.desktopModeHide(); if (!result.hidden) throw new Error('The menu bar is unavailable; keep this window open to manage the server.') })}>Run in menu bar</Button>
        </div>
        <nav aria-label="Minimal workspace" className="mt-5 flex flex-wrap gap-2 border-b border-line pb-3">
          {TABS.map((tab) => <button type="button" key={tab.id} aria-current={target === tab.id ? 'page' : undefined} onClick={() => onTarget(tab.id)} className={clsx('rounded-lg px-3 py-2 text-[13px] focus-visible:outline-2 focus-visible:outline-accent', target === tab.id ? 'bg-ink-5 text-ink' : 'text-ink-2 hover:bg-ink-5')}>{tab.label}</button>)}
        </nav>
        {error && <p role="alert" className="mt-4 text-[13px] text-danger">{error}</p>}
        {target === 'overview' && <>
          <EngineRuntimePanel minimal activeRuns={activeRuns} onApi={() => onTarget('api')} onWorkers={() => {}} onUnload={() => void action(() => api.engineUnload())} />
          {!['ready', 'starting'].includes(snapshot?.status.state ?? '') && <div className="mt-4 flex flex-wrap items-center gap-3"><Button variant="primary" icon="play" disabled={busy || activeRuns > 0 || learningBusy(learning) || selected?.fitsMemory === false} onClick={() => void action(start)}>{selected ? 'Start server' : 'Choose a model'}</Button><p className="text-[13px] text-ink-2">{selected?.title ?? selected?.repo ?? 'Download a model or load one you already have.'}</p></div>}
          <section className="mt-6 grid gap-3 sm:grid-cols-3">
            <Pointer title="Choose your model" text="Models checks memory and format before a download; smaller models leave more room for your other apps." />
            <Pointer title="Connect your tools" text="Start a model, then copy its local API URL and generated key from API connection." />
            <Pointer title="Keep it running" text="Close this window to keep the server in the menu bar; Quit TensorFold stops it and frees model memory." />
          </section>
          <p className="mt-4 text-[12px] leading-relaxed text-ink-2">Downloads need internet access; supported model inference runs on this Mac, and API access stays on localhost with a session key.</p>
        </>}
        {target === 'models' && <div className="-mx-4 sm:-mx-6"><Models minimal /></div>}
        {target === 'api' && <EngineApiPanel minimal onModels={() => onTarget('models')} />}
      </div>
    </div>
  </div>
}

function Pointer({ title, text }: { title: string; text: string }) {
  return <div className="rounded-xl border border-line bg-card p-4"><h2 className="text-[14px] font-medium">{title}</h2><p className="mt-2 text-[13px] leading-relaxed text-ink-2">{text}</p></div>
}
