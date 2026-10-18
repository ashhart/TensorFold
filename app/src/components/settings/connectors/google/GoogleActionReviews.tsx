import { currentIntlLocale } from '../../../../lib/preferences/current-locale'
import { useAuthoredText } from '../../../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { api } from '../../../../lib/ipc'
import type { GoogleWorkspaceActionReview, GoogleWorkspaceSettledAction } from '../../../../../shared/google-workspace'
import { Button } from '../../../ui'
import { calendarTimeLabel } from './google-calendar-time'

function ActionContent({ review }: { review: GoogleWorkspaceActionReview }) {
  const tx = useAuthoredText()
  const action = review.action
  if (action.kind === 'gmail.send') return <div className="space-y-2 text-[13px] text-ink-2"><p className="break-words">{tx("To: ")}{action.email.to.join(', ')}</p>{!!action.email.cc?.length && <p className="break-words">{tx("Cc: ")}{action.email.cc.join(', ')}</p>}{!!action.email.bcc?.length && <p className="break-words">{tx("Bcc: ")}{action.email.bcc.join(', ')}</p>}<p className="break-words font-medium text-ink">{action.email.subject || tx("No subject")}</p><pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words font-sans leading-relaxed">{action.email.text}</pre></div>
  const previous = review.previousEvent
  const event = action.kind === 'calendar.delete' ? undefined : action.event
  return <div className="space-y-3 text-[13px] text-ink-2">
    {previous && <div className="rounded-lg border border-line p-3"><p className="font-medium text-ink">{tx("Current Google event")}</p><p className="break-words">{previous.title}</p><p>{calendarTimeLabel(previous.start, previous.timeZone)} → {calendarTimeLabel(previous.end, previous.timeZone)}</p>{previous.description && <p className="whitespace-pre-wrap break-words">{previous.description}</p>}<p className="break-words">{tx("Guests: ")}{previous.attendees.join(', ') || tx("None")}</p></div>}
    {event ? <div className="space-y-1 rounded-lg bg-card-2 p-3"><p className="font-medium text-ink">{action.kind === 'calendar.create' ? tx("Create this event") : tx("After this change")}</p><p className="break-words">{event.title}</p><p>{calendarTimeLabel(event.start, event.timeZone)} → {calendarTimeLabel(event.end, event.timeZone)}</p>{/^\d{4}-\d{2}-\d{2}$/.test(event.end) && <p>{tx("All-day end date is excluded.")}</p>}<p>{tx("Event time zone: ")}{event.timeZone}</p>{event.description && <p className="max-h-60 overflow-auto whitespace-pre-wrap break-words">{event.description}</p>}<p className="break-words">{tx("Guests: ")}{event.attendees?.join(', ') || tx("None")}</p></div> : <p className="text-danger">{tx("Cancel this event: ")}{action.kind === 'calendar.delete' ? action.title : ''}</p>}
    <p className="break-words text-[12px] text-ink-3">{tx("Calendar: ")}{action.calendarId}</p><p className="text-[12px] text-ink-3">{(action.kind === 'calendar.delete' ? action.sendUpdates : action.event.sendUpdates) === 'all' ? tx("Google is asked to notify all guests.") : tx("No guest updates requested; Google may still send some notifications.")}</p>
  </div>
}
export function GoogleActionReviews({ reviews, recentActions, refresh }: { reviews: GoogleWorkspaceActionReview[]; recentActions: GoogleWorkspaceSettledAction[]; refresh: () => Promise<void> }) {
  const tx = useAuthoredText()
  const [busy, setBusy] = useState('')
  const [notice, setNotice] = useState('')
  const [now, setNow] = useState(Date.now())
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 15_000); return () => clearInterval(timer) }, [])
  const settle = async (review: GoogleWorkspaceActionReview, approve: boolean) => {
    if (busy || approve && review.expiresAt <= Date.now()) return
    setBusy(review.id); setNotice('')
    const receipt = { id: review.id, accountId: review.accountId, revision: review.revision, digest: review.digest }
    try {
      if (approve) {
        const result = await api.googleWorkspaceExecuteAction(receipt)
        setNotice(result.cancelled ? tx("Action cancelled; nothing was sent or changed.") : result.status === 'succeeded' ? tx("Google confirmed this action.") : result.status === 'uncertain' ? tx("Outcome uncertain; check Google before preparing anything again, as this action is never retried automatically.") : tx("Google did not complete this action; review the saved outcome before preparing another request."))
      } else { await api.googleWorkspaceCancelAction(receipt); setNotice(tx("Action cancelled; this review cannot be executed.")) }
      await refresh()
    } catch { setNotice(tx("This action or its status refresh failed; check Google and refresh the saved outcome before trying anything again.")) }
    finally { setBusy('') }
  }
  return <section className="space-y-3" aria-label={tx("Google action reviews")}>
    {reviews.length > 0 && <h3 className="text-[14px] font-medium text-ink">{tx("Review before sending or changing Google")}</h3>}
    {reviews.map((review) => <article key={review.id} className="space-y-3 rounded-xl border border-line-strong bg-card p-4"><p className="text-[13px] font-medium text-ink">{review.action.kind === 'gmail.send' ? tx("Send email") : review.action.kind === 'calendar.delete' ? tx("Cancel calendar event") : review.action.kind === 'calendar.create' ? tx("Create calendar event") : tx("Change calendar event")} · {review.accountEmail}</p><p className="text-[12px] text-ink-3">{tx("Check the full content and account; your approval applies only to this exact review, and the Mac asks for confirmation before execution.")}</p><ActionContent review={review} /><p className="text-[12px] text-ink-3">{review.expiresAt <= now ? tx("Review expired; refresh Google connections to clear it, then prepare a fresh request if still needed.") : tx("Review expires {value0}", { value0: new Date(review.expiresAt).toLocaleString(currentIntlLocale()) })}</p><div className="flex flex-wrap gap-2"><Button size="sm" variant="primary" disabled={!!busy || review.expiresAt <= now} onClick={() => settle(review, true)}>{busy === review.id ? tx("Working…") : tx("Review and approve on this Mac")}</Button><Button size="sm" disabled={!!busy || review.expiresAt <= now} onClick={() => settle(review, false)}>{tx("Cancel action")}</Button></div></article>)}
    {recentActions.length > 0 && <div className="space-y-2 rounded-xl border border-line bg-card p-4"><h3 className="text-[14px] font-medium text-ink">{tx("Recent Google action outcomes")}</h3><p className="text-[12px] text-ink-3">{tx("These receipts survive restart; no send or change is retried automatically.")}</p>{recentActions.map(({ review, result }) => <details key={result.id} className="rounded-lg border border-line p-3"><summary className={`cursor-pointer text-[13px] ${result.cancelled ? 'text-ink-3' : result.status === 'succeeded' ? 'text-positive' : 'text-warn'}`}>{result.cancelled ? tx("Cancelled by you") : result.status === 'succeeded' ? tx("Google confirmed") : result.status === 'uncertain' ? tx("Uncertain, check Google before repeating") : tx("Failed")} · {review.action.kind === 'gmail.send' ? review.action.email.subject || tx("Email") : review.action.kind === 'calendar.delete' ? review.action.title : review.action.event.title} · {review.accountEmail}</summary><div className="mt-3"><ActionContent review={review} /></div></details>)}</div>}
    {notice && <p role="status" className="text-[13px] text-ink-2">{notice}</p>}
  </section>
}
