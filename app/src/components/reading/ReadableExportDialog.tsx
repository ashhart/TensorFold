import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ReadableExportApi, ReadableExportFormat, ReadableExportPreview, ReadableExportRequest } from '../../../shared/readable-export'
import { useAccessibleDialog } from '../navigation/useAccessibleDialog'

export interface ReadableExportChoice { id: string; label: string }
export interface ReadableExportDialogProps {
  threadId: string
  messages: ReadableExportChoice[]
  initialSelection?: string[]
  api: ReadableExportApi
  onClose(): void
}
const control = 'rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-45'

export function ReadableExportDialog({ threadId, messages, initialSelection = [], api, onClose }: ReadableExportDialogProps) {
  const tx = useAuthoredText()
  const [format, setFormat] = useState<ReadableExportFormat>('markdown')
  const [scope, setScope] = useState<'conversation' | 'selected'>(initialSelection.length ? 'selected' : 'conversation')
  const [selected, setSelected] = useState(initialSelection)
  const [preview, setPreview] = useState<ReadableExportPreview | null>(null)
  const [busy, setBusy] = useState<'preview' | 'save' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const generation = useRef(0), reviewed = useRef(-1), pending = useRef(false), alive = useRef(true)
  const dialog = useRef<HTMLDivElement>(null), initial = useRef<HTMLSelectElement>(null), id = useId()
  useAccessibleDialog({ open: true, onClose, dialogRef: dialog, initialFocusRef: initial })
  const invalidate = () => { generation.current++; setPreview(null); setError(null); setNotice(null) }
  useEffect(() => {
    invalidate(); setSelected(initialSelection); setScope(initialSelection.length ? 'selected' : 'conversation')
  }, [threadId])
  useEffect(() => { alive.current = true; return () => { alive.current = false; generation.current++ } }, [])
  const request: ReadableExportRequest = { threadId, format, scope, branch: 'active', ...(scope === 'selected' ? { itemIds: selected } : {}) }
  const build = async () => {
    if (pending.current || scope === 'selected' && !selected.length) return
    const version = generation.current
    pending.current = true; setBusy('preview'); setError(null); setNotice(null); setPreview(null)
    try {
      const result = await api.readableExportPreview(request)
      if (alive.current && version === generation.current) { reviewed.current = version; setPreview(result) }
    } catch (reason) { if (alive.current && version === generation.current) setError(reason instanceof Error ? reason.message : tx("The preview could not be built.")) }
    finally { pending.current = false; if (alive.current) setBusy(null) }
  }
  const save = async () => {
    if (pending.current || !preview || reviewed.current !== generation.current) return
    const version = generation.current
    pending.current = true; setBusy('save'); setError(null)
    try {
      const result = await api.readableExportSave({ approvalId: preview.approvalId })
      if (!alive.current || version !== generation.current) return
      setPreview(null); reviewed.current = -1
      if (result.saved) setNotice(tx("Export saved. {value0} bytes.", { value0: result.bytes.toLocaleString(currentIntlLocale()) }))
      else if (result.canceled) setNotice(tx("Save canceled. Build a new preview to save again."))
      else setError(result.error || tx("The export was not saved. Build a new preview and try again."))
    } catch (reason) {
      if (alive.current && version === generation.current) { setPreview(null); reviewed.current = -1; setError(reason instanceof Error ? reason.message : tx("The export was not saved.")) }
    } finally { pending.current = false; if (alive.current) setBusy(null) }
  }
  return createPortal(<div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
    <div ref={dialog} role="dialog" aria-modal={true} aria-labelledby={`${id}-title`} tabIndex={-1} className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-line bg-card shadow-pop" onClick={(event) => event.stopPropagation()}>
      <header className="flex items-center justify-between gap-3 border-b border-line px-5 py-4"><h2 id={`${id}-title`} className="text-[17px] font-semibold text-ink">{tx("Export readable conversation")}</h2><button type="button" onClick={onClose} className={control}>{tx("Close")}</button></header>
      <div className="scroll-thin min-h-0 space-y-4 overflow-y-auto p-5">
        <p className="text-[13px] leading-relaxed text-ink-2">{tx("Share the active conversation branch as a readable document. This is separate from a restorable workspace backup or a support diagnostics bundle.")}</p>
        <div className="flex flex-wrap gap-4">
          <label className="space-y-1 text-[12px] text-ink-2"><span className="block">{tx("Format")}</span><select ref={initial} value={format} disabled={!!busy} className={control} onChange={(event) => { invalidate(); setFormat(event.target.value as ReadableExportFormat) }}><option value="markdown">{tx("Markdown (.md)")}</option><option value="text">{tx("Plain text (.txt)")}</option><option value="html">{tx("Self-contained HTML (.html)")}</option></select></label>
          <label className="space-y-1 text-[12px] text-ink-2"><span className="block">{tx("Messages")}</span><select value={scope} disabled={!!busy} className={control} onChange={(event) => { invalidate(); setScope(event.target.value as 'conversation' | 'selected') }}><option value="conversation">{tx("Active conversation")}</option><option value="selected" disabled={!messages.length}>{tx("Selected messages")}</option></select></label>
        </div>
        <p className="text-[12px] text-ink-3">{tx("Branch: active conversation only. Open a different branch first to export its original questions and answers.")}</p>
        {format === 'html' && <p className="text-[12px] text-ink-3">{tx("HTML preserves message text, Markdown notation and citation labels without scripts, embedded files or remote resources.")}</p>}
        {scope === 'selected' && <fieldset className="rounded-lg border border-line p-3"><legend className="px-1 text-[12px] text-ink-2">{tx("Choose messages in conversation order")}</legend><div className="scroll-thin max-h-40 space-y-2 overflow-y-auto">{messages.map((message) => <label key={message.id} className="flex min-h-8 items-start gap-2 text-[13px] text-ink-2"><input type="checkbox" disabled={!!busy || selected.length >= 500 && !selected.includes(message.id)} checked={selected.includes(message.id)} onChange={(event) => { invalidate(); setSelected((previous) => event.target.checked ? [...previous, message.id] : previous.filter((value) => value !== message.id)) }} /><span className="break-words">{message.label}</span></label>)}</div><p className="mt-2 text-[12px] text-ink-3">{selected.length} {tx("selected, up to 500 messages.")}</p></fieldset>}
        <button type="button" className={control} disabled={!!busy || scope === 'selected' && !selected.length} onClick={() => void build()}>{busy === 'preview' ? tx("Building preview…") : tx("Preview export")}</button>
        {error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
        {notice && <p role="status" className="text-[13px] text-ink-2">{notice}</p>}
        {preview && <section aria-label={tx("Reviewed export preview")} className="space-y-3 rounded-xl border border-line p-4">
          <p className="text-[13px] font-medium text-ink">{preview.messageCount} {tx("messages, ")}{preview.bytes.toLocaleString(currentIntlLocale())} {tx("bytes, ")}{preview.format}, {preview.branchLabel.toLowerCase()}.</p>
          <p className="text-[12px] text-ink-3">{tx("Known secret and personal-detail patterns are redacted. Review the remaining text before sharing.")}</p>
          <ul className="space-y-1 text-[12px] text-ink-3">{preview.omissions.map((omission, index) => <li key={`${omission.kind}-${index}`}>{omission.label} {tx("Count: ")}{omission.count}.</li>)}</ul>
          <p className="text-[12px] text-ink-3">{preview.redactions.length ? preview.redactions.map((entry) => `${entry.kind}: ${entry.count}`).join(', ') : tx("No pattern-based redactions were needed.")}</p>
          <div className="scroll-thin max-h-72 space-y-4 overflow-auto rounded-lg bg-card-2 p-3">{preview.messages.map((message) => <article key={message.ordinal}><h3 className="mb-1 text-[12px] font-semibold text-ink-2">{message.role === 'user' ? tx("You") : tx("Assistant")}{tx(", message ")}{message.ordinal}</h3><pre className="whitespace-pre-wrap break-words font-sans text-[13px] leading-relaxed text-ink-2">{message.text}</pre></article>)}</div>
        </section>}
      </div>
      <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-4"><p role="status" className="text-[12px] text-ink-3">{busy === 'save' ? tx("Waiting for the save dialog…") : tx("Saving requires this preview and a local file choice.")}</p><button type="button" className={control} disabled={!preview || !!busy} onClick={() => void save()}>{busy === 'save' ? tx("Saving…") : tx("Save reviewed export")}</button></footer>
    </div>
  </div>, document.body)
}
