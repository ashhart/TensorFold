import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { Thread } from '../../lib/types'
import { emptySearchFilterDraft, type SearchFilterDraft } from './search-filter-draft'


export function SearchFilterControls({ value, onChange, threads, providers }: {
  value: SearchFilterDraft
  onChange(value: SearchFilterDraft): void
  threads: readonly Thread[]
  providers: readonly { id: string; name: string }[]
}) {
  const tx = useAuthoredText()
  const providerIds = [...new Set(threads.map(thread => thread.providerId).filter((id): id is string => !!id))]
  if (value.providerId && !providerIds.includes(value.providerId)) providerIds.push(value.providerId)
  const models = [...new Set(threads.filter(thread => !value.providerId || thread.providerId === value.providerId).map(thread => thread.model).filter((id): id is string => !!id))].sort()
  if (value.model && !models.includes(value.model)) models.push(value.model)
  const field = 'mt-1 block min-w-0 w-full rounded border border-line bg-card px-2 py-1.5 text-[12px] text-ink'
  return <fieldset className="min-w-0 rounded-lg border border-line px-3 pb-3">
    <legend className="px-1 text-[12px] font-medium text-ink-2">{tx('Filter conversations')}</legend>
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <label className="min-w-0 text-[11px] text-ink-3">{tx('Updated from')}<input aria-label={tx('Conversations updated from date')} type="date" min="1970-01-01" value={value.from} onChange={event => onChange({ ...value, from: event.target.value })} className={field} /></label>
      <label className="min-w-0 text-[11px] text-ink-3">{tx('Updated through')}<input aria-label={tx('Conversations updated through date')} type="date" min="1970-01-01" value={value.to} onChange={event => onChange({ ...value, to: event.target.value })} className={field} /></label>
      <label className="min-w-0 text-[11px] text-ink-3">{tx('Current provider')}<select aria-label={tx('Conversation current provider filter')} value={value.providerId} onChange={event => onChange({ ...value, providerId: event.target.value, model: '' })} className={field}><option value="">{tx('Any provider')}</option>{providerIds.map(id => <option key={id} value={id}>{providers.find(provider => provider.id === id)?.name ?? id}</option>)}</select></label>
      <label className="min-w-0 text-[11px] text-ink-3">{tx('Current model')}<select aria-label={tx('Conversation current model filter')} value={value.model} onChange={event => onChange({ ...value, model: event.target.value })} className={field}><option value="">{tx('Any model')}</option>{models.map(model => <option key={model} value={model}>{model}</option>)}</select></label>
    </div>
    <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[12px] text-ink-2"><label className="flex items-center gap-2"><input type="checkbox" checked={value.attachments} onChange={event => onChange({ ...value, attachments: event.target.checked })} />{tx('Has attached files')}</label><button type="button" onClick={() => onChange(emptySearchFilterDraft())} className="rounded px-2 py-1 text-accent focus-visible:outline-2 focus-visible:outline-accent">{tx('Clear filters')}</button></div>
    <p className="mt-2 text-[11px] text-ink-3">{tx('Dates use your local calendar and the conversation’s last update. Model and provider refer to its current selection, not past replies.')}</p>
  </fieldset>
}
