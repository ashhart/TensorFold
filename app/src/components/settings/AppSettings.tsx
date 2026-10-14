import { useState } from 'react'
import clsx from 'clsx'
import { api } from '../../lib/ipc'
import { changeDesktopMode } from '../../lib/desktop-mode'
import { useStore } from '../../lib/store'
import { restartOnboarding } from '../onboarding-model'
import { Button } from '../ui'
import { Section, SettingsPane } from './shared'

export function AppSettings({ blocked = false }: { blocked?: boolean }) {
  const mode = useStore((s) => s.settings?.desktopMode ?? 'full')
  const layout = useStore((s) => s.fullWorkspaceLayout)
  const chooseLayout = useStore((s) => s.setFullWorkspaceLayout)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const setMode = async (next: 'minimal' | 'full') => {
    setBusy(true); setError('')
    try { await changeDesktopMode(next) } catch (e) { setError(String((e as Error)?.message ?? e)) } finally { setBusy(false) }
  }
  return <SettingsPane>
    <Section title="How you use TensorFold" desc="Switch at any time; your models and conversations stay available.">
      <fieldset disabled={busy || blocked || api.platform === 'web-remote'}><legend className="sr-only">Desktop mode</legend><div className="grid gap-3 sm:grid-cols-2">
        {([{ id: 'minimal', name: 'Minimal', desc: 'A menu-bar server with model downloads and an authenticated local API.' }, { id: 'full', name: 'Full', desc: 'The complete Chat, Code and Assistant workspace with local model controls.' }] as const).map((option) => <label key={option.id} className="min-w-0 cursor-pointer"><input type="radio" name="desktop-mode" aria-label={option.name} checked={mode === option.id} onChange={() => void setMode(option.id)} className="peer sr-only" /><span className={clsx('block h-full rounded-xl border p-4 text-left transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-accent peer-disabled:opacity-50', mode === option.id ? 'border-accent bg-accent/5' : 'border-line bg-card hover:bg-ink-5')}><span className="text-[15px] font-medium text-ink">{option.name}</span><span className="mt-2 block text-[13px] leading-relaxed text-ink-2">{option.desc}</span></span></label>)}
      </div></fieldset>
      {blocked && <p className="mt-3 text-[13px] text-ink-2">Save or discard your settings draft before switching modes or reopening setup.</p>}
      {error && <p role="alert" className="mt-3 text-[13px] text-danger">{error}</p>}
    </Section>
    <Section title="Full workspace layout" desc="Choose how much navigation you see in Full. Models, permissions and running work stay the same; Minimal remains the tray/server manager.">
      <fieldset><legend className="sr-only">Full workspace layout</legend><div className="grid gap-3 sm:grid-cols-2">
        {([{ id: 'chat', name: 'Chat layout', desc: 'Conversations first, with Models, Bots, Settings and work activity in reach.' }, { id: 'workspace', name: 'Complete workspace', desc: 'The project sidebar, saved work panes and all developer views.' }] as const).map((option) => <label key={option.id} className="min-w-0 cursor-pointer"><input type="radio" name="full-workspace-layout" aria-label={option.name} checked={layout === option.id} onChange={() => chooseLayout(option.id)} className="peer sr-only" /><span className={clsx('block h-full rounded-xl border p-4 transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-accent', layout === option.id ? 'border-accent bg-accent/5' : 'border-line bg-card hover:bg-ink-5')}><span className="text-[15px] font-medium text-ink">{option.name}</span><span className="mt-2 block text-[13px] leading-relaxed text-ink-2">{option.desc}</span></span></label>)}
      </div></fieldset>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-2">Saved on this device separately from your model settings. Return with Complete workspace; your sidebar and pane positions are kept.</p>
    </Section>
    <Section title="Welcome guide" desc="Revisit model setup and the pointers for your chosen mode."><Button disabled={busy || blocked || api.platform === 'web-remote'} onClick={() => { useStore.getState().setView({ name: 'home' }); restartOnboarding() }}>Open welcome guide</Button></Section>
  </SettingsPane>
}
