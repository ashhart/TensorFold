import type { QuickAskDraft, QuickAskSnapshot } from '../../../shared/quick-ask'
import { Icon } from '../../lib/icons'
import { askPresetLabel } from './preset-label'
import type { AskText } from './AskContext'

export function AskOptions({ snapshot, draft, blocked, edit, attachFiles, clipboard, screenshot, dictate, listening, settings, reset, tx }: {
  snapshot: QuickAskSnapshot; draft: QuickAskDraft; blocked: boolean
  edit(patch: Partial<QuickAskDraft>): void; attachFiles(): void; clipboard(): void; screenshot(): void; dictate(): void
  listening: boolean; settings(): void; reset(): void; tx: AskText
}) {
  return <div className="ask-options" aria-label={tx('More options')}>
    <div className="flex flex-wrap items-center gap-1">
      <button className="ask-option" disabled={blocked} onClick={attachFiles}><Icon name="plus" size={14} />{tx('Attach files')}</button>
      <button className="ask-option" disabled={blocked} onClick={clipboard}><Icon name="copy" size={14} />{tx('Attach clipboard text or image')}</button>
      <button className="ask-option" disabled={blocked} onClick={screenshot}><Icon name="monitor" size={14} />{tx('Attach a screenshot')}</button>
      <button className={`ask-option ${listening ? 'text-danger' : ''}`} disabled={blocked && !listening} onClick={dictate}><Icon name="mic" size={14} />{tx(listening ? 'Stop dictation' : 'Dictate question')}</button>
    </div>
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <select className="ask-input min-w-0 flex-1 py-1 text-xs" aria-label={tx('Ask preset')} value={draft.presetId} disabled={blocked || !!snapshot.session.route} onChange={event => edit({ presetId: event.target.value })}>
        {snapshot.preferences.presets.map(preset => <option key={preset.id} value={preset.id}>{askPresetLabel(preset, tx)}</option>)}
      </select>
      <label className="flex items-center gap-1.5 text-xs text-ink-3" title={tx('Allow public web research. Queries leave this computer. The default search covers news.')}>
        <input type="checkbox" checked={draft.web} disabled={blocked} onChange={event => edit({ web: event.target.checked })} />{tx('Web')}
      </label>
      <button className="ask-icon-button" aria-label={tx('New ask')} disabled={blocked} onClick={reset}><Icon name="pencil" size={14} /></button>
      <button className="ask-icon-button" aria-label={tx('Settings and presets')} disabled={blocked} onClick={settings}><Icon name="settings" size={14} /></button>
    </div>
    {draft.attachments.some(item => item.kind === 'image') && snapshot.routes.find(item => item.id === (snapshot.session.route?.id ?? draft.routeId))?.vision === 'unknown' && <p className="mt-2 text-xs text-warning">{tx('Image support has not been verified for this model. Check its answer against the attachment.')}</p>}
    {!!snapshot.routeErrors.length && <div className="mt-2 text-xs text-ink-3" role="status">{snapshot.routeErrors.map((error, index) => <p key={index}>{error}</p>)}</div>}
    {snapshot.shortcutError && <p role="status" className="mt-2 text-xs text-ink-3">{snapshot.shortcutError}</p>}
  </div>
}
