import { useEffect, useState } from 'react'
import type { QuickAskPreferences, QuickAskRoute } from '../../../shared/quick-ask'
import { AskDialog } from './AskDialog'
import type { UiTextParameters } from '../../lib/preferences/language-packs'

type Text = (source: string, params?: UiTextParameters) => string
export function shortcutForEvent(event: Pick<KeyboardEvent, 'code' | 'ctrlKey' | 'metaKey' | 'altKey' | 'shiftKey'>): string | null {
  const key = event.code === 'Space' ? 'Space' : /^(?:Key[A-Z]|Digit[0-9])$/.test(event.code) ? event.code.replace(/^(Key|Digit)/, '') : /^F(?:[1-9]|1[0-9]|2[0-4])$/.test(event.code) ? event.code : null
  if (!key || !(event.ctrlKey || event.metaKey || event.altKey)) return null
  return [...(event.metaKey ? ['Command'] : []), ...(event.ctrlKey ? ['Control'] : []), ...(event.altKey ? ['Alt'] : []), ...(event.shiftKey ? ['Shift'] : []), key].join('+')
}
export function AskPreferences({ value, routes, save, close, tx, error, record }: { value: QuickAskPreferences; routes: QuickAskRoute[]; save(value: QuickAskPreferences): Promise<boolean>; close(): void; tx: Text; error?: string; record(active: boolean): Promise<void> }) {
  const [draft, setDraft] = useState(() => structuredClone(value))
  const [selected, setSelected] = useState(value.presets[0].id), [recording, setRecording] = useState(false), [saving, setSaving] = useState(false)
  const preset = draft.presets.find(preset => preset.id === selected) ?? draft.presets[0]
  useEffect(() => { void record(recording).catch(() => {}); return () => { void record(false).catch(() => {}) } }, [recording, record])
  const patch = (patch: Partial<typeof preset>) => setDraft(current => ({ ...current, presets: current.presets.map(item => item.id === preset.id ? { ...item, ...patch } : item) }))
  return <AskDialog title={tx('Ask TensorFold settings')} onClose={close}>
    <div className="space-y-5">
      <label className="flex items-center gap-3 text-sm"><input type="checkbox" checked={draft.enabled} onChange={event => setDraft({ ...draft, enabled: event.target.checked })} />{tx('Open from any app with a global shortcut')}</label>
      <p className="text-xs leading-relaxed text-ink-3">{tx('When enabled, TensorFold stays available in the system tray after you close the main window.')}</p>
      <label className="block text-sm">{tx('Global shortcut')}<input aria-label={tx('Global shortcut')} className="ask-input mt-2 w-full font-mono" value={recording ? tx('Press your shortcut…') : draft.shortcut} onFocus={() => setRecording(true)} onBlur={() => setRecording(false)} onChange={() => {}} onKeyDown={event => {
        if (event.key === 'Tab' || event.key === 'Escape') { setRecording(false); return }
        event.preventDefault()
        const shortcut = shortcutForEvent(event.nativeEvent)
        if (shortcut) { setDraft({ ...draft, shortcut }); setRecording(false); event.currentTarget.blur() }
      }} /></label>
      <div className="border-t border-line pt-4">
        <div className="mb-3 flex items-center justify-between"><h3 className="text-sm font-medium">{tx('Your presets')}</h3><button className="ask-text-button" disabled={draft.presets.length >= 20} onClick={() => {
          const id = `preset-${Date.now()}`
          setDraft({ ...draft, presets: [...draft.presets, { id, name: tx('New preset'), command: `custom${draft.presets.length}`, prompt: 'Answer clearly and directly.', routeId: null }] }); setSelected(id)
        }}>{tx('Add preset')}</button></div>
        <label className="sr-only" htmlFor="ask-preset-edit">{tx('Preset to edit')}</label>
        <select id="ask-preset-edit" className="ask-input mb-4 w-full" value={preset.id} onChange={event => setSelected(event.target.value)}>{draft.presets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs text-ink-2">{tx('Name')}<input className="ask-input mt-1 w-full" value={preset.name} onChange={event => patch({ name: event.target.value })} maxLength={60} /></label>
          <label className="text-xs text-ink-2">{tx('Slash command')}<input className="ask-input mt-1 w-full" value={preset.command} onChange={event => patch({ command: event.target.value })} maxLength={40} /></label>
        </div>
        <label className="mt-3 block text-xs text-ink-2">{tx('Instructions')}<textarea className="ask-input mt-1 min-h-28 w-full resize-y" value={preset.prompt} onChange={event => patch({ prompt: event.target.value })} maxLength={8000} /></label>
        <label className="mt-3 block text-xs text-ink-2">{tx('Preset model')}<select className="ask-input mt-1 w-full" value={preset.routeId ?? ''} onChange={event => patch({ routeId: event.target.value || null })}>
          <option value="">{tx('Use the model selected for this ask')}</option>
          {preset.routeId && !routes.some(route => route.id === preset.routeId) && <option value={preset.routeId}>{tx('Saved model is unavailable')}</option>}
          {routes.map(route => <option key={route.id} value={route.id}>{route.label} · {route.location}</option>)}
        </select></label>
        {draft.presets.length > 1 && <button className="ask-text-button mt-3 text-danger" onClick={() => { const next = draft.presets.filter(item => item.id !== preset.id); setDraft({ ...draft, presets: next }); setSelected(next[0].id) }}>{tx('Delete preset')}</button>}
      </div>
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
      <div className="flex justify-end gap-2"><button className="ask-secondary" onClick={close}>{tx('Cancel')}</button><button className="ask-primary" disabled={saving} onClick={() => { setSaving(true); void save(draft).then(saved => { if (saved) close() }).finally(() => setSaving(false)) }}>{tx(saving ? 'Saving…' : 'Save settings')}</button></div>
    </div>
  </AskDialog>
}
