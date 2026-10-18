import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'

/** A staged follow-up in the steer queue above the composer. */
export type Steer = { id: string; text: string; status: 'queued' | 'guiding' | 'guided' }

export interface SteerQueueListProps {
  queued: Steer[]
  liveSteering: boolean
  /** An idle row is sent only by an explicit user click. */
  running?: boolean
  onGuide: (id: string, body: string) => void
  onRemove: (id: string) => void
  /** The owner applies a CAS replacement and maintains its saved-draft token separately. */
  onEdit?: (id: string, expectedText: string, text: string) => boolean | Promise<boolean>
  /** The owner suspends automatic queue draining while an edit is open. */
  onEditingChange?: (id: string | null) => void
}

type QueueRowProps = Omit<SteerQueueListProps, 'queued'> & { q: Steer }
export function QueuedFollowUpRow({ q, liveSteering, running = true, onGuide, onRemove, onEdit, onEditingChange }: QueueRowProps) {
  const tx = useAuthoredText()
  const [editing, setEditing] = useState(false), [text, setText] = useState(q.text), [saving, setSaving] = useState(false), [problem, setProblem] = useState(false)
  const expected = useRef(q.text), pending = useRef(false), mounted = useRef(true), current = useRef(q), isEditing = useRef(false)
  current.current = q
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; if (isEditing.current) onEditingChange?.(null) } }, [])
  const finish = () => { isEditing.current = false; setEditing(false); setProblem(false); onEditingChange?.(null) }
  const begin = () => {
    if (current.current.status !== 'queued' || pending.current || !onEdit) return
    expected.current = current.current.text; setText(current.current.text); setProblem(false)
    isEditing.current = true; onEditingChange?.(q.id); setEditing(true)
  }
  const save = async () => {
    if (pending.current || current.current.status !== 'queued' || !onEdit || !text.trim()) return
    pending.current = true; setSaving(true); setProblem(false)
    try {
      const accepted = await onEdit(q.id, expected.current, text)
      if (mounted.current) { if (accepted) finish(); else setProblem(true) }
    } catch { if (mounted.current) setProblem(true) }
    finally { pending.current = false; if (mounted.current) setSaving(false) }
  }
  return <div className={clsx('group flex items-start gap-2.5 rounded-2xl border px-3.5 py-2.5 fade-in transition-colors', q.status === 'guided' ? 'border-positive/25 bg-positive/5' : q.status === 'guiding' ? 'border-accent/30 bg-accent-soft' : 'border-line bg-card shadow-card')}>
    <span className="mt-0.5 flex shrink-0 items-center gap-1.5">{q.status === 'guided' ? <Icon name="check" size={13} className="text-positive" /> : q.status === 'guiding' ? <Icon name="refresh" size={13} className="animate-spin text-accent" /> : <Icon name="clock" size={13} className="text-ink-3" />}</span>
    <div className="min-w-0 flex-1">
      <div className="mb-0.5 text-[13px] font-medium text-ink-3">{q.status === 'guided' ? tx('Guided') : q.status === 'guiding' ? tx('Guiding') : tx('Queued')}</div>
      {editing ? <div className="space-y-2">
        <textarea aria-label={tx('Edit queued follow-up')} value={text} disabled={saving || q.status !== 'queued'} onChange={event => setText(event.target.value)} rows={3} className="max-h-52 min-h-20 w-full resize-y rounded-lg border border-line bg-card px-2.5 py-2 text-[13px] text-ink outline-none focus-visible:border-accent disabled:opacity-50" onKeyDown={event => { if (event.key === 'Escape' && !event.nativeEvent.isComposing && event.keyCode !== 229 && event.nativeEvent.keyCode !== 229 && !pending.current) { event.preventDefault(); finish() } }} />
        {q.status !== 'queued' && <p role="status" className="text-[12px] text-warn">{tx('This follow-up has already started; your unsaved edit cannot change the running request.')}</p>}
        {problem && <p role="alert" className="text-[12px] text-warn">{tx('This follow-up changed or could not be saved. Your edit remains here; cancel and reopen it to review the latest queued version.')}</p>}
        <div className="flex flex-wrap gap-2"><button type="button" disabled={saving || !text.trim() || q.status !== 'queued'} onClick={() => void save()} className="rounded-lg bg-accent px-3 py-1.5 text-[12px] font-medium text-accent-ink disabled:opacity-40">{saving ? tx('Saving…') : tx('Save changes')}</button><button type="button" disabled={saving} onClick={finish} className="rounded-lg border border-line px-3 py-1.5 text-[12px] text-ink-2 hover:bg-ink-5 disabled:opacity-40">{tx('Cancel')}</button></div>
        <p className="text-[11px] leading-relaxed text-ink-3">{tx('Saving keeps the same queue position and does not send this follow-up.')}</p>
      </div> : <div dir="auto" className={clsx('whitespace-pre-wrap break-words text-[13px] leading-snug', q.status === 'guided' ? 'text-ink-3 line-through' : 'text-ink-2')}>{q.text}</div>}
    </div>
    <div className="flex shrink-0 flex-wrap items-center gap-1">
      {q.status === 'queued' && !editing && <>
        {onEdit && <button type="button" onClick={begin} aria-label={tx('Edit queued follow-up')} title={tx('Edit queued follow-up')} className="flex h-6 w-6 items-center justify-center rounded-lg text-ink-3 hover:bg-ink-5 hover:text-ink"><Icon name="edit" size={13} /></button>}
        <button type="button" onClick={() => onGuide(q.id, q.text)} title={!running ? tx('Send this queued follow-up as the next turn') : liveSteering ? tx('Send into the running task now — steers the turn live') : tx("Send now — this engine can't take mid-turn input, so it applies when the current engine turn finishes")} className="flex items-center gap-1 rounded-lg border border-line bg-card px-2 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:border-line-strong hover:bg-card-2 hover:text-ink"><Icon name="spark" size={12} className="text-accent" />{running ? tx('Guide') : tx('Send follow-up')}</button>
      </>}
      {q.status === 'guiding' && <span className="flex items-center gap-1 px-2 py-1 text-[13px] font-medium text-ink-3"><Icon name="refresh" size={12} className="animate-spin" />{tx('Sending')}</span>}
      {q.status !== 'guiding' && <button type="button" disabled={saving} onClick={() => onRemove(q.id)} aria-label={tx('Remove from queue')} title={tx('Remove')} className="flex h-6 w-6 items-center justify-center rounded-lg text-ink-3 hover:bg-ink-5 hover:text-ink disabled:opacity-40"><Icon name="x" size={13} /></button>}
    </div>
  </div>
}

export function SteerQueueList({ queued, ...props }: SteerQueueListProps) {
  const tx = useAuthoredText()
  return <div className="mb-2 space-y-1.5">
    {props.onEdit && <p className="px-1 text-[11px] leading-relaxed text-ink-3">{tx('Queued follow-ups are session-only. Stop clears the queue; unfinished composer text stays saved separately.')}</p>}
    {queued.map(q => <QueuedFollowUpRow key={q.id} q={q} {...props} />)}
  </div>
}
