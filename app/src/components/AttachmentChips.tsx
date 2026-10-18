import { OriginalDiagnostic } from './OriginalDiagnostic'
import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Icon } from '../lib/icons'
import type { Attachment } from '../lib/types'
import { attachmentCanExtractText, MAX_DOC_CHARS } from '../lib/attachments'

interface AttachmentChipsProps {
  items?: Attachment[]
  onRemove?: (id: string) => void
  onRetry?: (id: string) => void
  onReselect?: (id: string) => void
  onExtractText?: (id: string) => void
  onReviewText?: (id: string) => void
  extractionAvailable?: boolean
  extractionUnavailableReason?: string
  className?: string
}

function statusLabel(attachment: Attachment): string {
  if (attachment.status === 'processing') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Processing…")
  if (attachment.status === 'error') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Needs attention")
  if (attachment.status === 'needs-reselection') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Select file again")
  if (attachment.status === 'review') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Review extracted text")
  if (attachment.extraction?.state === 'accepted') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Reviewed transcript ready")
  return attachment.kind === 'image' ? authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Image ready") : authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Text ready")
}

// Every selection keeps its own named row, including failures and restored references.
// Sent messages use the same display without mutation callbacks.
export function AttachmentChips({ items, onRemove, onRetry, onReselect, onExtractText, onReviewText, extractionAvailable = false, extractionUnavailableReason, className = '' }: AttachmentChipsProps) {
  const tx = useAuthoredText()
  if (!items?.length) return null
  const editable = !!onRemove
  return <div className={`flex flex-wrap items-start gap-2 ${className}`} aria-label={tx("Attached files")}>
    {items.map((attachment) => {
      const failed = attachment.status === 'error' || attachment.status === 'needs-reselection'
      const processing = attachment.status === 'processing'
      const needsReview = attachment.status === 'review'
      const preview = attachment.dataUrl ?? attachment.previewDataUrl
      const canExtract = attachmentCanExtractText(attachment) && attachment.extraction?.state !== 'accepted'
      const reviewWarnings = [...(attachment.warnings ?? []), ...(attachment.extraction?.warnings ?? [])]
      return <div key={attachment.id} data-attachment-id={attachment.id} className={`group flex min-w-0 max-w-full flex-col gap-1 rounded-lg border bg-card-2 p-2 text-[13px] text-ink-2 ${failed ? 'border-danger/40' : needsReview ? 'border-accent/40' : 'border-line'}`}>
        <div className="flex min-w-0 items-center gap-2">
          {preview ? <img src={preview} alt={attachment.name} className="h-8 w-8 shrink-0 rounded object-cover" />
            : <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded bg-ink-5 text-ink-3"><Icon name={failed ? 'warning' : attachment.mime === 'application/pdf' ? 'filePdf' : 'fileText'} size={15} /></span>}
          <div className="min-w-0 flex-1">
            <span className="block break-words font-medium text-ink" title={attachment.source?.relativePath ?? attachment.name}>{attachment.source?.relativePath ?? attachment.name}</span>
            <span role={editable ? 'status' : undefined} className={`block text-[12px] ${failed ? 'text-danger' : needsReview ? 'text-accent' : 'text-ink-3'}`}>{statusLabel(attachment)}</span>
          </div>
          {onRemove && <button type="button" onClick={() => onRemove(attachment.id)} className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-ink-3 transition-colors hover:bg-ink-5 hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent" aria-label={tx("Remove {value0}", { value0: attachment.name })}><Icon name="x" size={14} /></button>}
        </div>
        {attachment.error && <p className="max-w-[28rem] break-words text-[12px] leading-relaxed text-danger">{<OriginalDiagnostic>{attachment.error}</OriginalDiagnostic>}</p>}
        {editable && attachment.kind === 'image' && !needsReview && !processing && <p className="max-w-[28rem] text-[12px] text-ink-3">{tx("Use a vision-capable model for this image, or extract text for a text-only model.")}</p>}
        {reviewWarnings.length > 0 && <details className="max-w-[28rem] text-[12px] leading-relaxed text-ink-2"><summary className="cursor-pointer">{attachment.extraction ? tx("OCR review notes") : tx("File notes")}</summary>{[...new Set(reviewWarnings)].map((warning, index) => <p key={index} className="mt-1 break-words">{warning}</p>)}</details>}
        {editable && !processing && <div className="flex flex-wrap gap-1.5">
          {attachment.status === 'needs-reselection' && onReselect && <RowAction label={tx("Select {value0} again", { value0: attachment.name })} onClick={() => onReselect(attachment.id)}>{tx("Select again")}</RowAction>}
          {attachment.status === 'error' && onRetry && <RowAction label={tx("Retry {value0}", { value0: attachment.name })} onClick={() => onRetry(attachment.id)}>{tx("Retry")}</RowAction>}
          {needsReview && onReviewText && <RowAction label={tx("Review extracted text from {value0}", { value0: attachment.name })} onClick={() => onReviewText(attachment.id)}>{tx("Review text")}</RowAction>}
          {canExtract && onExtractText && !needsReview && attachment.status !== 'needs-reselection' && <RowAction label={tx("Extract text from {value0}", { value0: attachment.name })} disabled={!extractionAvailable} title={!extractionAvailable ? extractionUnavailableReason ?? tx("Text extraction requires the macOS app with its native helper.") : tx("Read text or handwriting on this Mac, then review it before sending.")} onClick={() => onExtractText(attachment.id)}>{tx("Extract text")}</RowAction>}
          {attachment.extraction?.state === 'accepted' && onReviewText && <RowAction label={tx("Edit transcript from {value0}", { value0: attachment.name })} onClick={() => onReviewText(attachment.id)}>{tx("Edit transcript")}</RowAction>}
        </div>}
        {editable && canExtract && !extractionAvailable && extractionUnavailableReason && <p className="max-w-[28rem] text-[12px] leading-relaxed text-ink-3">{extractionUnavailableReason}</p>}
      </div>
    })}
  </div>
}

function RowAction({ children, label, onClick, disabled, title }: { children: React.ReactNode; label: string; onClick: () => void; disabled?: boolean; title?: string }) {
  const tx = useAuthoredText()
  return <button type="button" aria-label={label} title={title} onClick={onClick} disabled={disabled} className="min-h-7 rounded-md border border-line px-2 text-[12px] text-ink-2 hover:bg-ink-5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50">{children}</button>
}

/** Explicit accept/cancel flow; editing here never changes model content until accepted. */
export function AttachmentTextReview({ attachment, onAccept, onCancel }: { attachment: Attachment; onAccept: (text: string) => void | Promise<void>; onCancel: () => void }) {
  const tx = useAuthoredText()
  const [text, setText] = useState(attachment.extraction?.text ?? '')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const dialog = useRef<HTMLDivElement>(null)
  const cancelRef = useRef(onCancel)
  cancelRef.current = onCancel
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    dialog.current?.querySelector<HTMLTextAreaElement>('textarea')?.focus()
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !saving) { event.preventDefault(); event.stopPropagation(); cancelRef.current() }
      if (event.key !== 'Tab') return
      const buttons = Array.from(dialog.current?.querySelectorAll<HTMLElement>('textarea,button:not([disabled])') ?? [])
      const first = buttons[0], last = buttons.at(-1)
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }
    document.addEventListener('keydown', key, true)
    return () => { document.removeEventListener('keydown', key, true); previous?.focus() }
  }, [saving])
  const accept = async () => {
    if (saving) return
    if (!text.trim()) { setError(tx("No readable text is available. Type a transcript or cancel and choose a clearer image.")); return }
    if (text.trim().length > MAX_DOC_CHARS - 200) { setError(tx("This transcript is too long. Shorten it before accepting.")); return }
    setSaving(true); setError('')
    try { await onAccept(text) }
    catch (cause) { setError(cause instanceof Error ? cause.message : tx("The transcript could not be accepted. Your current attachment is unchanged.")) }
    finally { setSaving(false) }
  }
  const extraction = attachment.extraction
  const preview = attachment.dataUrl ?? attachment.previewDataUrl
  return createPortal(<div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
    <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby="attachment-text-review-title" aria-describedby="attachment-text-review-notice" aria-busy={saving} className="no-drag flex max-h-[calc(100dvh-32px)] w-full max-w-[680px] flex-col overflow-hidden rounded-2xl border border-line bg-card text-ink shadow-pop">
      <div className="border-b border-line px-5 py-4"><h2 id="attachment-text-review-title" className="text-[16px] font-semibold">{tx("Review extracted text")}</h2><p className="mt-1 break-words text-[13px] text-ink-2">{attachment.name}</p></div>
      <div className="min-h-0 overflow-y-auto p-5">
        <p id="attachment-text-review-notice" className="text-[13px] leading-relaxed text-ink-2">{tx("Read locally with Apple Vision. No vision model or API key is needed. Check every line against the original, especially handwriting, names and numbers; recognition can omit or misread text.")}</p>
        {preview && <img src={preview} alt={tx("Original {value0}", { value0: attachment.name })} className="mt-3 max-h-48 max-w-full rounded-lg border border-line object-contain" />}
        <p className="mt-3 text-[12px] text-ink-2">{extraction?.processedPages ?? 0} {tx("of ")}{extraction?.pageCount ?? 0} {tx("pages processed")}{extraction?.confidence != null ? tx(" · Vision confidence score {value0} of 1.00, not a measure of accuracy", { value0: extraction.confidence.toFixed(2) }) : tx(" · No confidence score available")}</p>
        {(extraction?.warnings ?? []).map((warning, index) => <p key={index} className="mt-2 text-[12px] leading-relaxed text-ink-2">{warning}</p>)}
        {extraction?.truncated && <p role="status" className="mt-2 text-[12px] text-danger">{tx("This extraction is incomplete. Check the page and text limits before accepting.")}</p>}
        <label htmlFor="attachment-extracted-text" className="mb-2 mt-4 block text-[13px] font-medium">{tx("Editable transcript")}</label>
        <textarea id="attachment-extracted-text" aria-label={tx("Editable extracted text")} value={text} onChange={(event) => setText(event.target.value)} disabled={saving} spellCheck className="min-h-44 w-full resize-y select-text rounded-lg border border-line bg-canvas p-3 text-[14px] leading-relaxed text-ink outline-none focus:border-accent focus:ring-1 focus:ring-accent" />
        {error && <p role="alert" className="mt-2 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
        <p className="mt-2 text-[12px] leading-relaxed text-ink-3">{tx("Only the accepted transcript is attached to your draft, labelled as OCR text that may contain errors. No chat request is sent by this action.")}</p>
      </div>
      <div className="flex shrink-0 justify-end gap-2 border-t border-line px-5 py-3"><button type="button" onClick={onCancel} disabled={saving} className="h-9 rounded-lg border border-line px-3 text-[13px] text-ink-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50">{tx("Cancel")}</button><button type="button" onClick={() => void accept()} disabled={saving} className="h-9 rounded-lg bg-accent px-3 text-[13px] font-medium text-accent-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50">{saving ? tx("Accepting…") : tx("Accept transcript")}</button></div>
    </div>
  </div>, document.body)
}
