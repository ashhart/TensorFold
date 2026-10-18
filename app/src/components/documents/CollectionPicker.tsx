import { createPortal } from 'react-dom'
import { useAccessibleDialog } from '../navigation/useAccessibleDialog'
import { useEffect, useRef, useState } from 'react'
import type { DocumentLibraryApi, DocumentLibrarySource } from '../../../shared/document-library'
import { api } from '../../lib/ipc'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Button } from '../ui'

export function CollectionPicker({ threadId, sourceIds, disabled = false, onChange, client = api }: { threadId: string; sourceIds?: string[]; disabled?: boolean; onChange: (ids: string[]) => void | Promise<void>; client?: DocumentLibraryApi }) {
  const tx = useAuthoredText()
  const [open, setOpen] = useState(false), [sources, setSources] = useState<DocumentLibrarySource[]>([])
  const [selected, setSelected] = useState<string[]>([]), [error, setError] = useState(''), [pending, setPending] = useState(false), [loading, setLoading] = useState(false)
  const epoch = useRef(0), saving = useRef(false), loaded = useRef(false), panel = useRef<HTMLElement>(null), owner = useRef(threadId)
  owner.current = threadId
  const close = () => { epoch.current++; setOpen(false); setLoading(false); setError('') }
  useAccessibleDialog({ open, onClose: close, dialogRef: panel })
  useEffect(() => { if (disabled) close() }, [disabled])
  useEffect(() => { epoch.current++; setOpen(false); setPending(false); setError(''); return () => { epoch.current++ } }, [threadId])
  const show = async () => {
    const own = ++epoch.current; setOpen(true); setLoading(true); loaded.current = false; setError('')
    try {
      const snapshot = await client.documentLibraryList()
      if (own !== epoch.current) return
      loaded.current = true
      setSources(snapshot.sources)
      setSelected(sourceIds === undefined ? snapshot.sources.filter(source => source.enabled).map(source => source.id).slice(0, 64) : sourceIds.filter(id => snapshot.sources.some(source => source.id === id)))
    } catch (caught) { if (own === epoch.current) setError(String((caught as Error)?.message ?? caught)) }
    finally { if (own === epoch.current) setLoading(false) }
  }
  const save = async () => {
    if (saving.current || pending || loading || disabled || !loaded.current) return
    const own = epoch.current; saving.current = true; setPending(true); setError('')
    try { await onChange([...selected]); if (own === epoch.current) setOpen(false) }
    catch (caught) { if (own === epoch.current) setError(String((caught as Error)?.message ?? caught)) }
    finally { saving.current = false; if (owner.current === threadId) setPending(false) }
  }
  return <div className="relative min-w-0">
    <Button size="xs" variant="ghost" disabled={disabled || pending} aria-haspopup="dialog" aria-expanded={open} onClick={() => open ? close() : void show()}>{sourceIds === undefined ? tx('Knowledge: configured collections') : tx('Knowledge: {count} collections', { count: sourceIds.length })}</Button>
    {open && createPortal(<div className="fixed inset-0 z-[130] flex items-start justify-center bg-canvas/85 p-3 sm:p-6" onMouseDown={event => { if (event.target === event.currentTarget) close() }}><section ref={panel} role="dialog" aria-modal="true" tabIndex={-1} aria-label={tx('Choose knowledge for this chat')} className="scroll-thin max-h-full w-full max-w-lg space-y-3 overflow-y-auto rounded-xl border border-line bg-card p-4 shadow-pop outline-none">
      <h3 className="text-[13px] font-medium text-ink">{tx('Knowledge for this chat')}</h3>
      <p className="text-[12px] text-ink-3">{tx('Only selected collections can be searched by this chat. Local model and connector restrictions still apply.')}</p>
      {sourceIds === undefined && <p className="text-[12px] text-ink-3">{tx('This older chat currently uses enabled collections. Save an explicit selection to control its sources.')}</p>}
      {sourceIds?.some(id => !sources.some(source => source.id === id)) && !loading && <p className="text-[12px] text-ink-3">{tx('Some selected collections are no longer available. Save to remove them from this chat.')}</p>}
      <p className="text-[12px] text-ink-3">{tx('Choose up to 64 collections.')}</p>
      {error && <p role="alert" className="text-[12px] text-danger"><OriginalDiagnostic>{error}</OriginalDiagnostic></p>}
      {loading ? <p role="status">{tx('Loading collections…')}</p> : <div className="max-h-64 space-y-2 overflow-auto">{sources.map(source => <label key={source.id} className="flex items-start gap-2 text-[13px] text-ink-2"><input type="checkbox" disabled={pending || disabled || !source.enabled && !selected.includes(source.id) || !selected.includes(source.id) && selected.length >= 64} checked={selected.includes(source.id)} onChange={event => setSelected(current => event.target.checked ? [...current, source.id] : current.filter(id => id !== source.id))} /><span className="min-w-0 break-words">{source.label}{!source.enabled && <span className="text-ink-3">{tx(' · disabled')}</span>}</span></label>)}{!sources.length && <p className="text-[12px] text-ink-3">{tx('Add a document collection in Connections first.')}</p>}</div>}
      <p className="text-[12px] text-ink-3">{tx('No selection means this chat cannot retrieve document collections. Previously sent messages remain unchanged.')}</p>
      <div className="flex flex-wrap justify-end gap-2"><Button size="xs" variant="ghost" disabled={pending} onClick={close}>{tx('Cancel')}</Button><Button size="xs" disabled={loading || pending || disabled || !loaded.current} onClick={() => void save()}>{pending ? tx('Saving…') : tx('Save collections')}</Button></div>
    </section></div>, document.body)}
  </div>
}
