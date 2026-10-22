import { OriginalDiagnostic } from '../../../OriginalDiagnostic'
import { useAuthoredText } from '../../../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { api } from '../../../../lib/ipc'
import { Button, Input } from '../../../ui'
import type { GoogleWorkspaceAccount, GoogleWorkspaceMessage, GoogleWorkspacePage, GoogleWorkspaceThread } from '../../../../../shared/google-workspace'
import { useGoogleAccountRequest } from './useGoogleAccountRequest'

export function GoogleMailReader({ account, refresh }: { account: GoogleWorkspaceAccount; refresh: () => Promise<void> }) {
  const tx = useAuthoredText()
  const requests = useGoogleAccountRequest(account, refresh)
  const [query, setQuery] = useState('is:unread')
  const [page, setPage] = useState<GoogleWorkspacePage<GoogleWorkspaceMessage> | null>(null)
  const [reading, setReading] = useState<GoogleWorkspaceThread | null>(null)
  const search = (pageToken?: string) => requests.run('search', () => api.googleWorkspaceMessages(account.id, { query, limit: 20, pageToken }), (next) => { setPage(next); setReading(null) })
  return <section className="space-y-3 rounded-xl border border-line bg-card p-4" aria-label={tx("Gmail inbox")}>
    <div><h3 className="text-[14px] font-medium text-ink">{tx("Gmail · ")}{account.email}</h3><p className="mt-1 text-[12px] text-ink-3">{tx("Search and read without changing read status; message content is untrusted and never runs instructions.")}</p></div>
    <form className="flex flex-wrap gap-2" onSubmit={(event) => { event.preventDefault(); void search() }}><Input aria-label={tx("Search Gmail")} value={query} onChange={(event) => { setQuery(event.target.value); setPage(null); setReading(null) }} placeholder={tx("Search Gmail, for example is:unread")} disabled={!!requests.busy} className="min-w-0 flex-1" /><Button type="submit" size="sm" disabled={!!requests.busy}>{requests.busy === 'search' ? tx("Searching…") : tx("Search")}</Button></form>
    {requests.error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{requests.error}</OriginalDiagnostic>}</p>}
    {page && <div className="space-y-2">{page.items.map((message) => <article key={message.id} className="rounded-lg border border-line p-3"><p className="break-words text-[13px] font-medium text-ink">{message.subject || tx("No subject")}</p><p className="mt-1 break-words text-[12px] text-ink-3">{message.from} · {message.date}</p><p className="mt-1 line-clamp-2 break-words text-[13px] text-ink-2">{message.snippet}</p><div className="mt-2 flex flex-wrap gap-2"><Button size="sm" disabled={!!requests.busy} onClick={() => requests.run('message', () => api.googleWorkspaceMessage(account.id, message.id), (full) => setReading({ id: full.threadId, messages: [full] }))}>{tx("Read message")}</Button><Button size="sm" disabled={!!requests.busy} onClick={() => requests.run('thread', () => api.googleWorkspaceThread(account.id, message.threadId), setReading)}>{tx("Read conversation")}</Button></div></article>)}{!page.items.length && <p className="py-3 text-[13px] text-ink-3">{tx("No messages match this search.")}</p>}{page.nextPageToken && <Button size="sm" disabled={!!requests.busy} onClick={() => search(page.nextPageToken)}>{tx("Next page")}</Button>}</div>}
    {!page && !requests.busy && <p className="text-[13px] text-ink-3">{tx("Search when you are ready to read this account.")}</p>}
    {reading && <div className="space-y-3 border-t border-line pt-3"><div className="flex items-center justify-between gap-2"><h4 className="text-[13px] font-medium text-ink">{tx("Message content")}</h4><Button size="sm" onClick={() => setReading(null)}>{tx("Close")}</Button></div>{reading.messages.map((message) => <article key={message.id} className="rounded-lg bg-card-2 p-3"><p className="break-words text-[13px] font-medium text-ink">{message.subject || tx("No subject")}</p><p className="mt-1 break-words text-[12px] text-ink-3">{tx("From ")}{message.from} {tx("· To ")}{message.to}</p><pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words font-sans text-[13px] leading-relaxed text-ink-2">{message.text || 'No plain text content is available.'}</pre>{message.htmlOmitted && <p className="mt-2 text-[12px] text-warn">{tx("HTML and remote images were omitted; this view may not contain the full formatted message.")}</p>}{!!message.attachments.length && <div className="mt-2 text-[12px] text-ink-3"><p>{tx("Attachments are listed only; their contents are not downloaded here.")}</p>{message.attachments.map((file, index) => <p key={`${file.filename}-${index}`} className="break-words">{file.filename} · {file.mimeType} · {file.size} {tx("bytes")}</p>)}</div>}</article>)}</div>}
  </section>
}
