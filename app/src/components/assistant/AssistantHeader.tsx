import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { AssistantControlSnapshot } from '../../lib/types'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import { sourceName, timeAgo } from './mappings'
import type { AssistantItem, SourceKey, SourceRow } from './types'

export function AssistantHeader({ snapshot, selectedSourceRow, selectedItem, selectedSource, openSignals, urgentCount, loading, syncingSources, syncingEmail, onSyncAll, onRefresh, onSyncEmail, onBriefMe }: {
  snapshot: AssistantControlSnapshot | null
  selectedSourceRow: SourceRow | null
  selectedItem: AssistantItem | null
  selectedSource: SourceKey
  openSignals: number
  urgentCount: number
  loading: boolean
  syncingSources: boolean
  syncingEmail: boolean
  onSyncAll: () => void
  onRefresh: () => void
  onSyncEmail: () => void
  onBriefMe: (text: string) => void
}) {
  const tx = useAuthoredText()
  return (
    <header className="flex flex-wrap items-start justify-between gap-4 px-8 pt-7">
      <div>
        <div className="mb-1 flex items-center gap-2 text-[13px] text-ink-3">
          {selectedSourceRow ? <BrandIcon brand={connectorBrand(selectedSourceRow.id)} size={18} className="rounded-[5px]" /> : <Icon name="bell" size={14} />}
          <span>{selectedSourceRow?.name ?? tx("All assistant sources")}</span>
          <span className="text-ink-5">/</span>
          <span>{snapshot ? tx("Updated {value0}", { value0: timeAgo(snapshot.generatedAt) }) : tx("Demo snapshot")}</span>
        </div>
        <h2 className="text-[24px] font-semibold tracking-[-0.025em] text-ink">{tx("Conversation")}</h2>
        <p className="mt-1 max-w-[660px] text-[13px] leading-relaxed text-ink-2">
          {selectedItem ? `${sourceName(selectedItem.source)} · ${selectedItem.from}` : openSignals ? tx("Open signals: {count} · urgent: {urgent}", { count: openSignals, urgent: urgentCount }) : tx("Pick a thread or start a task below.")}
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="default" size="sm" icon="refresh" onClick={onSyncAll} disabled={syncingSources}>{syncingSources ? tx("Syncing") : tx("Sync")}</Button>
        <Button variant="default" size="sm" icon="clock" onClick={onRefresh} disabled={loading}>{loading ? tx("Refreshing") : tx("Refresh")}</Button>
        {(selectedSource === 'all' || selectedSource === 'email') && (
          <Button variant="default" size="sm" icon="mail" onClick={onSyncEmail} disabled={syncingEmail}>{syncingEmail ? tx("Syncing") : tx("Sync email")}</Button>
        )}
        <Button variant="primary" size="sm" icon="spark" onClick={() => onBriefMe(`Summarize ${selectedSource === 'all' ? 'everything that needs attention' : sourceName(selectedSource)} and tell me the next actions.`)}>{tx("Brief me")}</Button>
      </div>
    </header>
  )
}
