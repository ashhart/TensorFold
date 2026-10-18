import { languageDirection } from '../../../shared/languages'
import { authoredText, interfaceLanguages } from '../../lib/preferences/language-packs'
import { useEffect, useRef, useState } from 'react'
import type { UiPreferences } from '../../../shared/ui-preferences'
import { useStore } from '../../lib/store'
import { DEFAULT_UI_PREFERENCES, normalizeUiPreferences, UI_SCALES } from '../../lib/preferences/model'
import { translate } from '../../lib/preferences/catalogue'
import { Section } from '../settings/shared'

export function PreferencesPanel({ blocked = false }: { blocked?: boolean }) {
  const settings = useStore((state) => state.settings)
  const saved = normalizeUiPreferences(settings?.uiPreferences)
  const [draft, setDraft] = useState(saved), [busy, setBusy] = useState(false), [notice, setNotice] = useState<'saved' | 'failed' | null>(null)
  const pending = useRef(false), mounted = useRef(true)
  const tx = (source: string) => authoredText(draft.locale, source)
  const t = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) => translate(draft.locale, key, params)
  useEffect(() => { setDraft(normalizeUiPreferences(settings?.uiPreferences)); setNotice(null) }, [settings])
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  const edit = <K extends keyof UiPreferences>(key: K, value: UiPreferences[K]) => { setNotice(null); setDraft((previous) => normalizeUiPreferences({ ...previous, [key]: value })) }
  const save = async () => {
    const state = useStore.getState()
    if (pending.current || blocked || !state.settings) return
    if (state.settings !== settings) { setDraft(normalizeUiPreferences(state.settings.uiPreferences)); setNotice('failed'); return }
    pending.current = true; setBusy(true); setNotice(null)
    try {
      await state.saveSettings({ ...state.settings, uiPreferences: normalizeUiPreferences(draft) }, { silent: true })
      if (mounted.current) setNotice('saved')
    } catch { if (mounted.current) setNotice('failed') }
    finally { pending.current = false; if (mounted.current) setBusy(false) }
  }
  const control = 'mt-1 min-h-10 w-full rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-45'
  return <Section title={t('preferences.title')} desc={t('preferences.description')}>
    <p className="mb-4 text-[12px] leading-relaxed text-ink-3">{tx('Language packs are bundled for offline use. New replies follow your selection when the model supports that language; existing messages, code and original diagnostics stay unchanged.')}</p>
    <fieldset disabled={blocked || busy || !settings} className="grid gap-4 sm:grid-cols-2">
      <legend className="sr-only">{t('preferences.title')}</legend>
      <label className="text-[13px] text-ink-2">{t('preferences.language')}<select value={draft.locale} className={control} onChange={(event) => edit('locale', event.target.value as UiPreferences['locale'])}>{interfaceLanguages().map(language => <option key={language.id} value={language.id} lang={language.id} dir={language.direction}>{language.nativeName}</option>)}</select></label>
      <label className="text-[13px] text-ink-2">{t('preferences.direction')}<select value={draft.direction} className={control} onChange={(event) => edit('direction', event.target.value as UiPreferences['direction'])}><option value="auto">{t('preferences.auto')}</option><option value="ltr">{t('preferences.ltr')}</option><option value="rtl">{t('preferences.rtl')}</option></select></label>
      {(['chatScale', 'codeScale'] as const).map((kind) => <label key={kind} className="text-[13px] text-ink-2">{t(`preferences.${kind}`)}<select value={draft[kind]} className={control} onChange={(event) => edit(kind, Number(event.target.value) as UiPreferences[typeof kind])}>{UI_SCALES.map((scale) => <option key={scale} value={scale}>{t('preferences.scale', { percent: Math.round(scale * 100) })}</option>)}</select></label>)}
      <label className="text-[13px] text-ink-2">{t('preferences.density')}<select value={draft.density} className={control} onChange={(event) => edit('density', event.target.value as UiPreferences['density'])}><option value="comfortable">{t('preferences.comfortable')}</option><option value="compact">{t('preferences.compact')}</option></select></label>
      <label className="text-[13px] text-ink-2">{tx('Send a message with')}<select value={draft.sendKey ?? 'enter'} className={control} onChange={event => edit('sendKey', event.target.value as UiPreferences['sendKey'])}><option value="enter">{tx('Enter to send')}</option><option value="modified-enter">{tx('Command or Control + Enter to send')}</option></select><span className="mt-1 block text-[12px] text-ink-3">{tx('Shift + Enter always adds a new line; composition and text editing shortcuts are preserved.')}</span></label>
      <label className="flex items-start gap-2 text-[13px] text-ink-2"><input type="checkbox" className="mt-1 accent-accent" checked={draft.replyLanguage !== 'auto'} onChange={event => edit('replyLanguage', event.target.checked ? 'interface' : 'auto')} /><span>{tx('Use this language for new assistant replies')}<span className="mt-1 block text-[12px] text-ink-3">{tx('You can still ask for a different language in any message.')}</span></span></label>
      <div className="flex flex-wrap items-end gap-2"><button type="button" disabled={busy || blocked} onClick={() => { setDraft({ ...DEFAULT_UI_PREFERENCES }); setNotice(null) }} className={control}>{t('preferences.reset')}</button><button type="button" disabled={busy || blocked} onClick={() => void save()} className={control}>{busy ? t('preferences.saving') : t('common.save')}</button></div>
    </fieldset>
    <p className="mt-3 text-[12px] text-ink-3">{t('preferences.directionHelp')}</p>
    <section aria-label={t('preferences.preview')} dir={draft.direction === 'auto' ? languageDirection(draft.locale) : draft.direction} className="mt-4 space-y-3 rounded-xl border border-line bg-card-2 p-4">
      <p dir="auto" style={{ fontSize: `${14 * draft.chatScale}px`, lineHeight: draft.density === 'compact' ? 1.48 : 1.62 }} className="break-words text-ink-2">{t('preferences.previewText')}</p>
      <p dir="auto" className="break-words text-[13px] text-ink-3">{t('preferences.mixedSample')}</p>
      <pre dir="ltr" style={{ fontSize: `${13 * draft.codeScale}px` }} className="overflow-auto rounded-lg border border-line bg-card p-3 text-left font-mono text-ink-2">{t('preferences.sampleCode')}</pre>
    </section>
    <p className="mt-3 text-[12px] text-ink-3">{JSON.stringify(draft) === JSON.stringify(saved) ? t('preferences.savedWhere') : t('preferences.unsaved')}</p>
    {notice && <p role={notice === 'failed' ? 'alert' : 'status'} className={`mt-2 text-[13px] ${notice === 'failed' ? 'text-danger' : 'text-ink-2'}`}>{t(`preferences.${notice}`)}</p>}
  </Section>
}
