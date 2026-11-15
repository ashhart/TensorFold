import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useStore } from '../../lib/store'
import type { Ticket } from '../../lib/types'
import { ticketDetails, ticketDetailsPatch, type TicketDetails } from '../../lib/ticket-details'
import { archiveOriginHostId } from '../../lib/store/shared/merge'
import { projectedProjectHostId } from '../../lib/store/shared/proxy-state'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useAccessibleDialog } from '../navigation/useAccessibleDialog'
import { Icon } from '../../lib/icons'
import { OriginalDiagnostic } from '../OriginalDiagnostic'

const control = 'rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink outline-none focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50'

function TicketDialog({ title, onClose, children }: { title: string; onClose(): void; children: ReactNode }) {
  const tx = useAuthoredText(), id = useId(), dialog = useRef<HTMLDivElement>(null)
  useAccessibleDialog({ open: true, onClose, dialogRef: dialog })
  return createPortal(<div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-4" onClick={(event) => event.stopPropagation()}>
    <div ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby={id} className="flex max-h-[85vh] w-full max-w-[680px] flex-col rounded-2xl border border-line-strong bg-card p-5 text-ink shadow-card outline-none">
      <header className="mb-4 flex items-center justify-between gap-3"><h2 id={id} className="text-[16px] font-semibold">{title}</h2><button type="button" className={control} onClick={onClose}>{tx('Close')}</button></header>
      <div className="scroll-thin min-h-0 overflow-y-auto">{children}</div>
    </div>
  </div>, document.body)
}

function TicketEditor({ ticket, onClose }: { ticket: Ticket; onClose(): void }) {
  const tx = useAuthoredText()
  const [original] = useState(() => ({ ...ticket }))
  const [draft, setDraft] = useState(() => ticketDetails(ticket))
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  const pending = useRef(false), alive = useRef(true)
  const active = ticket.runState === 'queued' || ticket.runState === 'running'
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const close = () => { if (!pending.current) onClose() }
  const update = <K extends keyof TicketDetails>(field: K, value: TicketDetails[K]) => setDraft((valueBefore) => ({ ...valueBefore, [field]: value }))
  const save = async () => {
    if (pending.current || active) return
    pending.current = true; setBusy(true); setError('')
    try {
      const current = useStore.getState().tickets.find((candidate) => candidate.id === ticket.id && archiveOriginHostId(candidate) === archiveOriginHostId(original))
      const patch = ticketDetailsPatch(original, current, draft)
      if (Object.keys(patch).length) await useStore.getState().updateTicket(ticket.id, patch)
      if (alive.current) onClose()
    } catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { pending.current = false; if (alive.current) setBusy(false) }
  }
  return <TicketDialog title={tx('Details')} onClose={close}>
    <form className="space-y-4" onSubmit={(event) => { event.preventDefault(); void save() }}>
      <label className="block text-[13px] text-ink-2">{tx('Title')}<input dir="auto" required maxLength={500} value={draft.title} readOnly={busy || active} onChange={(event) => update('title', event.target.value)} className={`mt-1 w-full ${control}`} /></label>
      <label className="block text-[13px] text-ink-2">{tx('Description')}<textarea dir="auto" rows={12} maxLength={64_000} value={draft.body} readOnly={busy || active} onChange={(event) => update('body', event.target.value)} className={`mt-1 w-full resize-y ${control}`} /></label>
      <div className="grid grid-cols-3 gap-3">
        {(['type', 'priority', 'effort'] as const).map((field) => <label key={field} className="block text-[13px] text-ink-2">{field === 'type' ? tx('Type') : field === 'priority' ? tx('Priority') : tx('Effort')}
          <select aria-label={field === 'type' ? tx('Type') : field === 'priority' ? tx('Priority') : tx('Effort')} disabled={busy || active} value={draft[field]} onChange={(event) => update(field, event.target.value as TicketDetails[typeof field])} className={`mt-1 w-full capitalize ${control}`}>
            {(field === 'type' ? ['task', 'bug', 'epic', 'subticket'] : field === 'priority' ? ['low', 'medium', 'high', 'urgent'] : ['small', 'medium', 'large']).map((value) => <option key={value} value={value}>{value}</option>)}
          </select></label>)}
      </div>
      {active && <p role="status" className="text-[13px] text-ink-3">{tx('Running')}</p>}
      {error && <p role="alert" className="text-[13px] text-danger"><OriginalDiagnostic>{error}</OriginalDiagnostic></p>}
      <div className="flex justify-end gap-2"><button type="button" disabled={busy} onClick={close} className={control}>{tx('Cancel')}</button><button type="submit" disabled={busy || active || !draft.title.trim()} className={control}>{tx('Save changes')}</button></div>
    </form>
  </TicketDialog>
}

export function TicketDetailsButton({ ticket }: { ticket: Ticket }) {
  const tx = useAuthoredText(), [open, setOpen] = useState(false)
  return <><button type="button" aria-label={`${tx('Details')}: ${ticket.title}`} title={tx('Details')} onClick={(event) => { event.stopPropagation(); setOpen(true) }} className="shrink-0 text-ink-3 hover:text-ink"><Icon name="file" size={13} /></button>
    {open && <TicketEditor key={`${archiveOriginHostId(ticket)}:${ticket.id}`} ticket={ticket} onClose={() => setOpen(false)} />}</>
}

function TicketArchive({ projectId, onClose }: { projectId: string; onClose(): void }) {
  const tx = useAuthoredText()
  const archives = useStore((state) => state.archives)
  const project = useStore((state) => state.projects.find((candidate) => candidate.id === projectId))
  const [loading, setLoading] = useState(true), [error, setError] = useState(''), [restoring, setRestoring] = useState<string | null>(null)
  const pending = useRef(false), alive = useRef(true)
  const hostId = project ? projectedProjectHostId(project) : null
  const tickets = (archives?.tickets ?? []).filter((ticket) => ticket.projectId === projectId && archiveOriginHostId(ticket) === hostId)
  const refresh = async () => {
    setLoading(true); setError('')
    try { await useStore.getState().refreshArchives() }
    catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { if (alive.current) setLoading(false) }
  }
  useEffect(() => { alive.current = true; void refresh(); return () => { alive.current = false } }, [])
  const restore = async (ticket: Ticket) => {
    if (pending.current) return
    pending.current = true; setRestoring(ticket.id); setError('')
    try { await useStore.getState().restoreItem('ticket', ticket.id, archiveOriginHostId(ticket)) }
    catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { pending.current = false; if (alive.current) setRestoring(null) }
  }
  return <TicketDialog title={tx('Archived')} onClose={() => { if (!pending.current) onClose() }}>
    <div className="space-y-3">
      <button type="button" disabled={loading || restoring !== null} onClick={() => void refresh()} className={control}>{tx('Refresh')}</button>
      {loading && <p role="status" className="text-[13px] text-ink-3">{tx('Loading…')}</p>}
      {error && <p role="alert" className="text-[13px] text-danger"><OriginalDiagnostic>{error}</OriginalDiagnostic></p>}
      {!loading && !tickets.length && <p className="text-[13px] text-ink-3">—</p>}
      {tickets.map((ticket) => <section key={`${archiveOriginHostId(ticket)}:${ticket.id}`} className="rounded-xl border border-line bg-card-2 p-3">
        <details><summary className="cursor-pointer break-words text-[14px] font-medium">{ticket.title}</summary><p dir="auto" className="mt-2 whitespace-pre-wrap break-words text-[13px] text-ink-2">{ticket.body}</p></details>
        <div className="mt-3 flex items-center justify-between gap-3"><span className="text-[12px] text-ink-3">{ticket.type} · {ticket.priority} · {ticket.effort}</span><button type="button" disabled={restoring !== null} onClick={() => void restore(ticket)} className={control}>{tx('Restore')}</button></div>
      </section>)}
    </div>
  </TicketDialog>
}

export function TicketArchiveButton({ projectId }: { projectId: string }) {
  const tx = useAuthoredText(), [open, setOpen] = useState(false)
  return <><button type="button" onClick={() => setOpen(true)} className={`inline-flex items-center gap-1.5 ${control}`}><Icon name="archive" size={14} />{tx('Archive')}</button>
    {open && <TicketArchive key={projectId} projectId={projectId} onClose={() => setOpen(false)} />}</>
}
