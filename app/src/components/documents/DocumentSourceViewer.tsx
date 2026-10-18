import { useEffect, useRef } from 'react'
import type { DocumentSourceResult } from '../../../shared/document-processing'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { documentLocationLabel } from './DocumentEvidence'

export function DocumentSourceViewer({ result, busy, onClose }: { result: DocumentSourceResult | null; busy: boolean; onClose: () => void }) {
  const tx = useAuthoredText()
  const panel = useRef<HTMLElement>(null), closeButton = useRef<HTMLButtonElement>(null)
  const open = !!result || busy
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const current = panel.current
    closeButton.current?.focus()
    return () => { if (previous?.isConnected && (document.activeElement === document.body || current?.contains(document.activeElement))) previous.focus() }
  }, [open])
  if (!result && !busy) return null
  const parserLabels = { 'plain-text': tx('Plain text parser'), workbook: tx('Workbook parser'), 'pdf-text': tx('PDF text parser'), 'office-text': tx('Office document parser') }
  return <section ref={panel} onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onClose() } }} aria-label={tx('Document source viewer')} aria-busy={busy} className="space-y-3 rounded-xl border border-accent/40 bg-card p-3.5">
    <div className="flex items-start justify-between gap-2"><h4 className="text-[13px] font-medium text-ink">{tx('Original and extracted text')}</h4><button ref={closeButton} type="button" className="rounded-lg px-2 py-1 text-[12px] text-ink-2 hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent" onClick={onClose}>{tx('Close source')}</button></div>
    {busy ? <p role="status" className="text-[13px] text-ink-3">{tx('Verifying the original revision and preparing a safe preview…')}</p> : result?.status === 'ready' ? <>
      <p className="break-all text-[13px] text-ink">{result.passage.sourceLabel} / {result.passage.relativePath}</p>
      <p className="text-[12px] text-ink-3">{parserLabels[result.parser]}</p>
      <p className="break-all text-[12px] text-ink-3">{tx('Current SHA-256 revision: {revision}', { revision: result.passage.revision })}</p>
      <p className="text-[12px] text-ink-3">{documentLocationLabel(result.passage.location)}</p>
      {result.locationNote && <p className="text-[12px] text-ink-3"><OriginalDiagnostic>{result.locationNote}</OriginalDiagnostic></p>}
      <div className="grid min-w-0 gap-3 lg:grid-cols-2"><section className="min-w-0"><h5 className="mb-2 text-[12px] font-medium text-ink">{tx('Original file text')}</h5>{result.original.kind === 'text' ? <><pre dir="auto" className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-line p-3 text-[12px] text-ink-2">{result.original.text}</pre>{result.original.truncated && <p className="mt-1 text-[12px] text-danger">{tx('Original preview is truncated at its display limit; the source file is unchanged.')}</p>}</> : <p className="rounded-lg border border-line p-3 text-[12px] text-ink-3"><OriginalDiagnostic>{result.original.reason}</OriginalDiagnostic></p>}</section><section className="min-w-0"><h5 className="mb-2 text-[12px] font-medium text-ink">{tx('Extracted document text')}</h5><pre dir="auto" className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-line p-3 text-[12px] text-ink-2">{result.extractedText}</pre>{result.extractedTruncated && <p className="mt-1 text-[12px] text-danger">{tx('Extracted preview is truncated at its display limit; indexed passages remain searchable.')}</p>}</section></div>
      <section><h5 className="text-[12px] font-medium text-ink">{tx('Cited passage')}</h5><blockquote dir="auto" className="mt-2 whitespace-pre-wrap break-words rounded-lg bg-accent/10 p-3 text-[13px] text-ink">{result.passage.text}</blockquote></section>
      <p className="text-[12px] text-ink-3">{tx('Original verified at {time}', { time: new Date(result.verifiedAt).toLocaleString(currentIntlLocale()) })}</p>
    </> : <p role="alert" className="text-[13px] text-danger"><OriginalDiagnostic>{result?.message ?? ''}</OriginalDiagnostic></p>}
  </section>
}
