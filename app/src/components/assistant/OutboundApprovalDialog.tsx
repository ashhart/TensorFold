import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef } from 'react'
import type { ConnectorEvent } from '../../lib/types'
import { Button } from '../ui'

export function canReviewOutbound(event: ConnectorEvent): boolean {
  return !event.outboundError && !!event.outbound && /^[a-f0-9]{64}$/.test(event.reviewDigest ?? '')
}

/** Shows the exact host-sealed payload; a summary never authorizes a message. */
export function OutboundApprovalDialog({ event, busy, error, onConfirm, onClose }: {
  event: ConnectorEvent
  busy: boolean
  error?: string
  onConfirm: (reviewDigest: string) => void
  onClose: () => void
}) {
  const tx = useAuthoredText()
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    ref.current?.focus()
    const key = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape' && !busy) { ev.preventDefault(); onClose() }
      if (ev.key !== 'Tab') return
      const nodes = [...(ref.current?.querySelectorAll<HTMLElement>('button:not(:disabled), [tabindex="0"]') ?? [])]
      if (!nodes.length) { ev.preventDefault(); return }
      const first = nodes[0], last = nodes[nodes.length - 1]
      if (ev.shiftKey && (document.activeElement === first || document.activeElement === ref.current)) { ev.preventDefault(); last.focus() }
      else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', key)
    return () => { document.removeEventListener('keydown', key); previous?.focus() }
  }, [busy, onClose])
  const payload = event.outbound
  return <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-4">
    <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="outbound-review-title" className="flex max-h-[85vh] w-full max-w-[620px] flex-col rounded-2xl border border-line-strong bg-card p-5 text-ink shadow-card outline-none">
      <h2 id="outbound-review-title" className="text-[16px] font-semibold">{tx("Review this message")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{tx("Approval applies only to the account, recipient and content shown here.")}</p>
      {payload && <><dl className="my-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[13px]"><dt className="text-ink-2">{tx("Service")}</dt><dd>{payload.connectorId}</dd><dt className="text-ink-2">{tx("Account")}</dt><dd className="break-all">{payload.accountId ?? tx("Configured account")}</dd><dt className="text-ink-2">{tx("To")}</dt><dd className="break-all">{payload.recipient}</dd>{payload.subject && <><dt className="text-ink-2">{tx("Subject")}</dt><dd className="break-words">{payload.subject}</dd></>}</dl><div tabIndex={0} aria-label={tx("Exact outgoing message")} className="min-h-[100px] overflow-y-auto whitespace-pre-wrap break-words rounded-xl border border-line bg-card-2 p-4 text-[14px] leading-relaxed">{payload.body}</div></>}
      {(error || event.outboundError || !canReviewOutbound(event)) && <p role="alert" className="mt-3 text-[13px] text-danger">{error || event.outboundError || tx("This draft has no valid delivery address or review snapshot; create a new draft before sending.")}</p>}
      <div className="mt-4 flex justify-end gap-2"><Button disabled={busy} onClick={onClose}>{tx("Cancel")}</Button><Button variant="primary" disabled={busy || !canReviewOutbound(event)} onClick={() => onConfirm(event.reviewDigest!)}>{busy ? tx("Sending…") : tx("Approve and send once")}</Button></div>
    </div>
  </div>
}
