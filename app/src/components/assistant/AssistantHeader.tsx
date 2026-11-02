import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { AssistantControlSnapshot } from '../../lib/types'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import { sourceName, timeAgo } from './mappings'
import type { AssistantItem, SourceKey, SourceRow } from './types'

export function AssistantHeader({ snapshot, selectedSourceRow, selectedSource, openSignals, urgentCount, loading, refreshError, refreshedAt, paused, syncingSources, syncingEmail, onSyncAll, onRefresh, onSyncEmail, onBriefMe, onManageSources }: {
  snapshot: AssistantControlSnapshot | null
  selectedSourceRow: SourceRow | null
  selectedItem: AssistantItem | null
  selectedSource: SourceKey
  openSignals: number
  urgentCount: number
  loading: boolean
  refreshError?: string | null
  refreshedAt?: number | null
  paused?: boolean
  syncingSources: boolean
  syncingEmail: boolean
  onSyncAll: () => void
  onRefresh: () => void
  onSyncEmail: () => void
  onBriefMe: (text: string) => void
  onManageSources?: () => void
}) {
  const tx = useAuthoredText()
  const updated = refreshedAt ?? snapshot?.generatedAt
  const freshness = refreshError ? tx('Update delayed') : loading ? tx('Refreshing…') : updated ? tx('Checked {value0}', { value0: timeAgo(updated) }) : tx('Loading inbox…')
  return (
    <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-line bg-card px-4 py-4 sm:px-6">
      <div className="min-w-0">
        <div className="mb-1 flex flex-wrap items-center gap-2 text-[12px] text-ink-3">
          {selectedSourceRow ? <BrandIcon brand={connectorBrand(selectedSourceRow.id)} size={16} /> : <Icon name="bell" size={14} />}
          <span>{selectedSourceRow?.name ?? tx('All assistant sources')}</span>
          <span aria-hidden="true">·</span>
          <span data-testid="assistant-freshness" title={paused ? tx('Automatic checks resume when you return to this view.') : tx('Checks the local inbox every 15 seconds while visible. Source sync is separate.')}>{freshness}</span>
        </div>
        <h2 className="text-[22px] font-semibold tracking-[-0.025em] text-ink">{tx('Your inbox')}</h2>
        <p className="mt-1 text-[12px] leading-relaxed text-ink-3">{openSignals ? tx('Open signals: {count} · urgent: {urgent}', { count: openSignals, urgent: urgentCount }) : tx('Read conversations, prepare replies, and review actions here.')}</p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="ghost" size="sm" icon="refresh" onClick={onRefresh} disabled={loading} aria-label={tx('Refresh inbox')}>{loading ? tx('Refreshing') : tx('Refresh')}</Button>
        <details className="relative" onKeyDown={(event) => {
          if (event.key !== 'Escape') return
          event.currentTarget.open = false
          event.currentTarget.querySelector('summary')?.focus()
        }}>
          <summary className="flex h-8 cursor-pointer list-none items-center gap-1 rounded-lg px-2 text-[13px] text-ink-2 hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent">{tx('Source actions')}<Icon name="chevron-down" size={13} /></summary>
          <div className="absolute right-0 top-10 z-30 w-[260px] rounded-xl border border-line bg-card p-2 shadow-pop" onClick={(event) => { if ((event.target as HTMLElement).closest('button')) event.currentTarget.closest('details')?.removeAttribute('open') }}>
            <p className="px-2 py-1.5 text-[12px] leading-relaxed text-ink-3">{tx('Fetch new messages from your enabled sources.')}</p>
            <Button className="w-full justify-start" variant="ghost" size="sm" icon="refresh" disabled={syncingSources || syncingEmail} onClick={onSyncAll}>{syncingSources ? tx('Syncing sources…') : tx('Sync email and iMessage')}</Button>
            {(selectedSource === 'all' || selectedSource === 'email') && <Button className="w-full justify-start" variant="ghost" size="sm" icon="mail" disabled={syncingEmail || syncingSources} onClick={onSyncEmail}>{syncingEmail ? tx('Syncing email…') : tx('Sync email only')}</Button>}
            {onManageSources && <Button className="mt-1 w-full justify-start border-t border-line" variant="ghost" size="sm" icon="settings" onClick={onManageSources}>{tx('Manage sources')}</Button>}
          </div>
        </details>
        <Button variant="default" size="sm" icon="spark" disabled={!snapshot} onClick={() => onBriefMe(`Summarize ${selectedSource === 'all' ? 'everything that needs attention' : sourceName(selectedSource)} and tell me the next actions.`)}>{tx('Prepare briefing')}</Button>
      </div>
    </header>
  )
}
