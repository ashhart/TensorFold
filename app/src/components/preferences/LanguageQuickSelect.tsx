import { interfaceLanguages } from '../../lib/preferences/language-packs'
import { useId, useRef, useState } from 'react'
import { normalizeLanguage } from '../../../shared/languages'
import { useStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
/** Language can be chosen before setup without downloading or loading a model. */
export function LanguageQuickSelect({ blocked = false }: { blocked?: boolean }) {
  const settings = useStore(state => state.settings), tx = useAuthoredText(), id = useId()
  const [busy, setBusy] = useState(false), [failed, setFailed] = useState(false), pending = useRef(false)
  const choose = async (input: string) => {
    const current = useStore.getState()
    if (pending.current || blocked || !settings || current.settings !== settings) return
    pending.current = true; setBusy(true); setFailed(false)
    try { await current.saveSettings({ ...settings, uiPreferences: { ...settings.uiPreferences, locale: normalizeLanguage(input), direction: settings.uiPreferences?.direction ?? 'auto', chatScale: settings.uiPreferences?.chatScale ?? 1, codeScale: settings.uiPreferences?.codeScale ?? 1, density: settings.uiPreferences?.density ?? 'comfortable' } }, { silent: true }) }
    catch { setFailed(true) }
    finally { pending.current = false; setBusy(false) }
  }
  return <div className="min-w-0 text-[12px] text-ink-3">
    <label htmlFor={id} className="flex min-w-0 flex-wrap items-center gap-2"><span>{tx('Language')}</span><select id={id} value={normalizeLanguage(settings?.uiPreferences?.locale)} disabled={busy || blocked || !settings} onChange={event => void choose(event.target.value)} className="min-h-10 max-w-full rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-45">{interfaceLanguages().map(language => <option key={language.id} value={language.id} lang={language.id} dir={language.direction}>{language.nativeName}</option>)}</select></label>
    {busy && <p role="status">{tx('Applying language…')}</p>}
    {failed && <p role="alert" className="mt-1 text-danger">{tx('The language could not be saved. Your previous selection is still active.')}</p>}
  </div>
}
