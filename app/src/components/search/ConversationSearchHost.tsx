import { SearchFilterControls } from './SearchFilterControls'
import { emptySearchFilterDraft, searchFiltersFromDraft } from './search-filter-draft'
import { conversationSearchFilterKey } from '../../../shared/conversation-search-filters'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useUiText } from '../../lib/preferences/hooks'
import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ConversationSearchMatch } from '../../../shared/conversation-search'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { createConversationSearchClient } from '../../lib/conversation-search-client'
import { useAccessibleDialog } from '../navigation/useAccessibleDialog'
import { Button } from '../ui'
import { SearchResultRow } from './SearchResultRow'
import { consumeConversationSearchRequest, OPEN_CONVERSATION_SEARCH, requestConversationSearch, revealConversationMatch } from './search-events'

export function ConversationSearchHost({ navigationBlocked = false }: { navigationBlocked?: boolean }) {
  const tx = useAuthoredText()
  const text = useUiText()
  const view = useStore((state) => state.view)
  const projects = useStore((state) => state.projects)
  const threads = useStore((state) => state.threads)
  const providers = useStore((state) => state.settings?.providers ?? [])
  const activeThreadId = view.name === 'thread' ? view.id : null
  const client = useMemo(() => createConversationSearchClient(api), [])
  const [filterDraft, setFilterDraft] = useState(emptySearchFilterDraft)
  const filterState = useMemo(() => { try { return { filters: searchFiltersFromDraft(filterDraft), error: false } } catch { return { filters: undefined, error: true } } }, [filterDraft])
  const filterKey = conversationSearchFilterKey(filterState.filters)
  const [state, setState] = useState(client.snapshot)
  const [open, setOpen] = useState(false), [query, setQuery] = useState('')
  const [scope, setScope] = useState<'all' | 'thread'>('all'), [projectId, setProjectId] = useState('')
  const [archived, setArchived] = useState(false), [selected, setSelected] = useState(0), [retry, setRetry] = useState(0)
  const panel = useRef<HTMLDivElement>(null), input = useRef<HTMLInputElement>(null)
  const close = () => { client.clear(); setOpen(false); setQuery('') }
  useAccessibleDialog({ open, onClose: close, dialogRef: panel, initialFocusRef: input })
  useEffect(() => client.subscribe(setState), [client])
  useEffect(() => {
    const show = () => {
      const request = consumeConversationSearchRequest(); if (!request) return
      if (api.platform === 'web-remote') { useStore.getState().setNotice({ kind: 'info', message: tx("Full-text conversation search is available in the local desktop workspace.") }); return }
      client.clear(); setQuery(request.query ?? ''); setScope(request.scope); setProjectId(''); setArchived(false); setFilterDraft(emptySearchFilterDraft()); setSelected(0); setOpen(true)
    }
    const key = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 'f' || api.platform === 'web-remote') return
      if (!open && document.querySelector('[aria-modal="true"]')) return
      if (!event.shiftKey && !activeThreadId) return
      event.preventDefault()
      requestConversationSearch({ scope: event.shiftKey ? 'all' : 'thread', threadId: activeThreadId ?? undefined })
    }
    window.addEventListener(OPEN_CONVERSATION_SEARCH, show); window.addEventListener('keydown', key); show()
    return () => { window.removeEventListener(OPEN_CONVERSATION_SEARCH, show); window.removeEventListener('keydown', key); client.clear() }
  }, [activeThreadId, client, open])
  useEffect(() => {
    client.clear(); setSelected(0)
    if (!open || filterState.error || scope === 'thread' && !activeThreadId) return
    const timer = setTimeout(() => void client.search({ scope, query, ...(scope === 'thread' ? { threadId: activeThreadId } : {}), ...(projectId ? { projectId } : {}), includeArchived: archived, filters: filterState.filters, limit: 50 }), 180)
    return () => { clearTimeout(timer); client.clear() }
  }, [open, query, scope, activeThreadId, projectId, archived, filterKey, filterState.error, retry, client])
  const ownsState = open && state.request?.query === query.trim().replace(/\s+/g, ' ')
    && state.request.scope === scope && (scope !== 'thread' || state.request.threadId === activeThreadId)
    && (state.request.projectId ?? '') === projectId && !!state.request.includeArchived === archived && !filterState.error && conversationSearchFilterKey(state.request.filters) === filterKey
  const filterThreads = (threads ?? []).filter(thread => !thread.orchestratorHostId
    && (scope === 'thread' ? thread.id === activeThreadId : !thread.incognito && !thread.phoneLocalOnly && !thread.googleLocalOnly && !thread.bot && (archived || !thread.archivedAt))
    && (!projectId || thread.projectId === projectId))
  const current = ownsState ? state : { result: null, error: '', busy: false }
  const matches = current.result?.matches ?? []
  const navigationOwner = useRef('')
  const ownership = JSON.stringify([open, scope, query.trim().replace(/\s+/g, ' '), activeThreadId, projectId, archived, filterKey, filterState.error, navigationBlocked])
  navigationOwner.current = ownership
  useEffect(() => {
    document.getElementById(`conversation-search-result-${selected}`)?.scrollIntoView({ block: 'nearest' })
  }, [selected])
  const openMatch = (match: ConversationSearchMatch) => {
    if (navigationOwner.current !== ownership || navigationBlocked || !ownsState || !client.snapshot().result?.matches.includes(match)) return
    if (scope === 'thread' && match.threadId !== activeThreadId) return
    close(); useStore.getState().openThread(match.threadId); revealConversationMatch(match)
    if (match.source === 'history') useStore.getState().setNotice({ kind: 'info', message: tx("This match is in saved model history without a visible message anchor. The conversation is open; no message was changed.") })
  }
  if (!open) return null
  return createPortal(<div className="fixed inset-0 z-[130] flex items-center justify-center bg-canvas/85 p-3 sm:p-6" onMouseDown={(event) => { if (event.target === event.currentTarget) close() }}>
    <div ref={panel} role="dialog" aria-modal="true" aria-labelledby="conversation-search-title" tabIndex={-1} className="flex max-h-full w-full max-w-[820px] flex-col rounded-xl border border-line bg-canvas shadow-pop outline-none">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-4 py-3"><h2 id="conversation-search-title" className="text-[15px] font-medium text-ink">{text('common.search')}</h2><Button size="sm" variant="ghost" onClick={close}>{tx("Close search")}</Button></header>
      <div className="space-y-3 border-b border-line p-4"><label className="block text-[12px] text-ink-2">{tx("Find words in your messages")}<input ref={input} aria-label={tx("Search message contents")} role="combobox" aria-autocomplete="list" aria-expanded={matches.length > 0} aria-controls="conversation-search-results" aria-activedescendant={matches[selected] ? `conversation-search-result-${selected}` : undefined} value={query} maxLength={512} onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => { if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setSelected((old) => matches.length ? (old + (event.key === 'ArrowDown' ? 1 : -1) + matches.length) % matches.length : 0) } else if (event.key === 'Enter' && matches[selected]) { event.preventDefault(); openMatch(matches[selected]) } }}
        placeholder={tx("Search a phrase, decision or question…")} className="mt-1 w-full rounded-lg border border-line bg-card px-3 py-2.5 text-[14px] text-ink focus-visible:outline-2 focus-visible:outline-accent" /></label>
        <div className="flex flex-wrap items-center gap-3 text-[12px] text-ink-2"><label>{tx("Search in ")}<select aria-label={tx("Conversation search scope")} value={scope} onChange={(event) => setScope(event.target.value as 'all' | 'thread')} className="ml-2 rounded border border-line bg-card px-2 py-1.5"><option value="all">{tx("All saved conversations")}</option><option value="thread" disabled={!activeThreadId}>{tx("Current conversation")}</option></select></label>
          {scope === 'all' && <><label>{tx("Project ")}<select aria-label={tx("Conversation search project")} value={projectId} onChange={(event) => setProjectId(event.target.value)} className="ml-2 max-w-[200px] rounded border border-line bg-card px-2 py-1.5"><option value="">{tx("All projects")}</option>{projects.map((project) => <option value={project.id} key={project.id}>{project.name}</option>)}</select></label><label className="flex items-center gap-2"><input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} />{tx("Include archived")}</label></>}
        </div><SearchFilterControls value={filterDraft} onChange={setFilterDraft} threads={filterThreads} providers={providers} />
        {filterState.error && <p role="alert" className="text-[12px] text-warn">{tx("Choose a valid date range; the start must be on or before the end.")}</p>}<p className="text-[11px] leading-relaxed text-ink-3">{tx("Search stays on this Mac and never asks a model. Private, phone, Google and bot histories are excluded from global results; search inside their active conversation instead.")}{api.platform === 'browser' ? tx(" This browser preview uses synthetic data.") : ''}</p>
      </div>
      <div className="scroll-thin min-h-0 overflow-y-auto p-4"><p role="status" aria-live="polite" className="mb-3 text-[12px] text-ink-3">{current.busy ? tx("Searching local messages…") : !query.trim() ? tx("Enter words to search your conversation content.") : current.result ? <>{tx("Matches: {count}", { count: matches.length })}{current.result.omitted ? tx(" · Additional matches in this scan: {count}", { count: current.result.omitted }) : null}</> : scope === 'thread' && !activeThreadId ? tx("Open a conversation to search inside it.") : ''}</p>
        {navigationBlocked && <p role="status" className="mb-3 rounded-lg border border-line bg-card p-3 text-[12px] text-warn">{tx("Save or discard your settings changes before opening a search result.")}</p>}
        {current.error && <div role="alert" className="mb-3 rounded-lg border border-line p-3 text-[13px] text-ink-2"><p>{<OriginalDiagnostic>{current.error}</OriginalDiagnostic>}</p><Button size="sm" onClick={() => setRetry((old) => old + 1)}>{tx("Retry search")}</Button></div>}
        {current.result?.truncated && <p role="status" className="mb-3 text-[12px] text-warn">{tx("This search reached a scan limit. More matches may exist; narrow the query or search one conversation.")}</p>}
        {current.result && !matches.length && <p className="rounded-lg border border-dashed border-line p-5 text-center text-[13px] text-ink-3">{tx("No matching messages found in this scope.")}</p>}
        <div id="conversation-search-results" role="listbox" aria-label={tx("Conversation search results")} className="space-y-2">{matches.map((match, index) => <SearchResultRow key={`${match.threadId}:${match.itemId ?? match.source}:${match.messageIndex ?? index}`} match={match} index={index} selected={index === selected} disabled={navigationBlocked} onOpen={() => openMatch(match)} />)}</div>
      </div>
    </div>
  </div>, document.body)
}
