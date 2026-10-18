import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useState } from 'react'
import clsx from 'clsx'
import { api } from '../../lib/ipc'
import { changeDesktopMode } from '../../lib/desktop-mode'
import { useStore } from '../../lib/store'
import { restartOnboarding } from '../onboarding-model'
import { Button } from '../ui'
import { Section, SettingsPane } from './shared'
import { PreferencesPanel } from '../preferences/PreferencesPanel'
import { useUiText } from '../../lib/preferences/hooks'

export function AppSettings({ blocked = false }: { blocked?: boolean }) {
  const t = useUiText()
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
    <PreferencesPanel blocked={blocked} />
    <Section title={t('app.modeTitle')} desc={t('app.modeDescription')}>
      <fieldset disabled={busy || blocked || api.platform === 'web-remote'}><legend className="sr-only">{t('app.desktopMode')}</legend><div className="grid gap-3 sm:grid-cols-2">
        {([{ id: 'minimal', name: t('app.minimal'), desc: t('app.minimalDescription') }, { id: 'full', name: t('app.full'), desc: t('app.fullDescription') }] as const).map((option) => <label key={option.id} className="min-w-0 cursor-pointer"><input type="radio" name="desktop-mode" aria-label={option.name} checked={mode === option.id} onChange={() => void setMode(option.id)} className="peer sr-only" /><span className={clsx('block h-full rounded-xl border p-4 text-left transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-accent peer-disabled:opacity-50', mode === option.id ? 'border-accent bg-accent/5' : 'border-line bg-card hover:bg-ink-5')}><span className="text-[15px] font-medium text-ink">{option.name}</span><span className="mt-2 block text-[13px] leading-relaxed text-ink-2">{option.desc}</span></span></label>)}
      </div></fieldset>
      {blocked && <p className="mt-3 text-[13px] text-ink-2">{t('app.blocked')}</p>}
      {error && <p role="alert" className="mt-3 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    </Section>
    <Section title={t('app.layoutTitle')} desc={t('app.layoutDescription')}>
      <fieldset><legend className="sr-only">{t('app.layoutTitle')}</legend><div className="grid gap-3 sm:grid-cols-2">
        {([{ id: 'chat', name: t('app.chatLayout'), desc: t('app.chatLayoutDescription') }, { id: 'workspace', name: t('app.completeLayout'), desc: t('app.completeLayoutDescription') }] as const).map((option) => <label key={option.id} className="min-w-0 cursor-pointer"><input type="radio" name="full-workspace-layout" aria-label={option.name} checked={layout === option.id} onChange={() => chooseLayout(option.id)} className="peer sr-only" /><span className={clsx('block h-full rounded-xl border p-4 transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-accent', layout === option.id ? 'border-accent bg-accent/5' : 'border-line bg-card hover:bg-ink-5')}><span className="text-[15px] font-medium text-ink">{option.name}</span><span className="mt-2 block text-[13px] leading-relaxed text-ink-2">{option.desc}</span></span></label>)}
      </div></fieldset>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{t('app.layoutSaved')}</p>
    </Section>
    <Section title={t('app.welcomeTitle')} desc={t('app.welcomeDescription')}><Button disabled={busy || blocked || api.platform === 'web-remote'} onClick={() => { useStore.getState().setView({ name: 'home' }); restartOnboarding() }}>{t('app.openWelcome')}</Button></Section>
  </SettingsPane>
}
