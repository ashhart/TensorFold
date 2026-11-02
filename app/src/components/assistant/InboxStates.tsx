import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Icon } from '../../lib/icons'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Button } from '../ui'
import { sourceName } from './mappings'
import type { AssistantItem } from './types'

export function InboxLoadState({ loading, error, onRetry }: { loading: boolean; error: string | null; onRetry: () => void }) {
  const tx = useAuthoredText()
  return <section aria-live="polite" aria-busy={loading} className="flex min-h-0 flex-1 items-center justify-center px-6 py-12">
    <div className="w-full max-w-[380px] text-center">
      <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-surface text-ink-3"><Icon name={error ? 'refresh' : 'mail'} size={22} /></span>
      <h3 className="mt-4 text-[18px] font-semibold text-ink">{error ? tx('The inbox could not load') : tx('Loading your inbox')}</h3>
      <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{error ? tx('Try checking again. Your saved conversation drafts are kept separately.') : tx('Getting conversations and pending approvals from your sources.')}</p>
      {error ? <><p className="mt-3 break-words text-[12px] text-ink-3"><OriginalDiagnostic>{error}</OriginalDiagnostic></p><Button className="mt-5" variant="primary" icon="refresh" disabled={loading} onClick={onRetry}>{loading ? tx('Retrying…') : tx('Retry inbox')}</Button></> : <div aria-hidden="true" className="mt-6 space-y-2"><div className="h-3 rounded bg-surface" /><div className="mx-auto h-3 w-3/4 rounded bg-surface" /><div className="mx-auto h-3 w-1/2 rounded bg-surface" /></div>}
    </div>
  </section>
}

export function ConversationHeading({ item, messageCount, pending, onReview }: { item: AssistantItem; messageCount: number; pending: number; onReview: () => void }) {
  const tx = useAuthoredText()
  return <div className="flex shrink-0 flex-wrap items-start justify-between gap-3 border-b border-line bg-card px-4 py-4 sm:px-6">
    <div className="min-w-0 flex-1">
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-ink-3"><span>{sourceName(item.source)}</span><span aria-hidden="true">·</span><span>{item.from}</span><span aria-hidden="true">·</span><span>{tx('Messages: {count}', { count: messageCount })}</span></div>
      <h3 className="mt-1.5 break-words text-[17px] font-semibold leading-snug tracking-[-0.01em] text-ink">{item.title || item.from}</h3>
    </div>
    {pending > 0 && <Button className="hidden md:inline-flex" size="sm" icon="shield" onClick={onReview}>{tx('Review pending message')}</Button>}
  </div>
}

export function EmptyInbox({ onManageSources, onPrepareBriefing }: { onManageSources: () => void; onPrepareBriefing: () => void }) {
  const tx = useAuthoredText()
  return <section className="flex min-h-[190px] flex-1 items-center justify-center px-6 py-8 text-center">
    <div className="max-w-[380px]"><span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-soft text-accent"><Icon name="check" size={22} /></span><h3 className="mt-4 text-[18px] font-semibold text-ink">{tx('No conversations in this view')}</h3><p className="mt-2 text-[13px] leading-relaxed text-ink-3">{tx('Choose another source, sync your messages, or connect a source to bring work into your inbox.')}</p><div className="mt-5 flex flex-wrap justify-center gap-2"><Button onClick={onManageSources} icon="plug">{tx('Manage sources')}</Button><Button onClick={onPrepareBriefing} icon="spark" variant="ghost">{tx('Prepare a task')}</Button></div></div>
  </section>
}
