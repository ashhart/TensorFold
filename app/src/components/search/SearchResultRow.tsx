import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { ConversationSearchMatch } from '../../../shared/conversation-search'
import { searchDisplaySegments } from '../../lib/conversation-search-client'
import { Icon } from '../../lib/icons'

export function SearchResultRow({ match, selected, index, onOpen, disabled }: { match: ConversationSearchMatch; selected: boolean; index: number; onOpen(): void; disabled: boolean }) {
  const tx = useAuthoredText()
  return <button type="button" id={`conversation-search-result-${index}`} data-ui-row role="option" aria-selected={selected} disabled={disabled} onClick={onOpen}
    className={`w-full rounded-lg border p-3 text-left focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50 ${selected ? 'border-accent bg-accent/8' : 'border-line bg-card hover:bg-card-2'}`}>
    <span className="flex min-w-0 items-center gap-2"><Icon name={match.kind === 'chat' ? 'message' : 'terminal'} size={14} /><span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{match.title || tx("Untitled conversation")}</span><span className="shrink-0 text-[11px] text-ink-3">{new Date(match.updatedAt).toLocaleDateString(currentIntlLocale())}</span></span>
    <span className="mt-2 block whitespace-pre-wrap break-words text-[13px] leading-relaxed text-ink-2">{searchDisplaySegments(match.excerpt, match.highlights).map((part, i) => part.match ? <mark key={i} className="rounded-sm bg-accent/15 px-0.5 text-ink">{part.text}</mark> : <span key={i}>{part.text}</span>)}</span>
    <span className="mt-2 block text-[11px] text-ink-3">{match.source === 'history' ? tx("Saved model history · no visible message anchor") : match.source === 'title' ? tx("Conversation title") : tx("Open this message")}</span>
  </button>
}
