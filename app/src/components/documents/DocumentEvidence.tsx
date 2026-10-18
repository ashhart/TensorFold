import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { DocumentEvidenceResult, DocumentLocation } from '../../../shared/document-library'
import { Button } from '../ui'

export function documentLocationLabel(location: DocumentLocation): string {
  if (location.kind === 'text') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "{value0}lines {value1}–{value2}, characters {value3}–{value4}", { value0: String(location.section ? `${location.section}, ` : ''), value1: String(location.lineStart), value2: String(location.lineEnd), value3: String(location.charStart + 1), value4: String(location.charEnd) })
  if (location.kind === 'table') return `${location.sheet ? `${location.sheet}, ` : ''}${location.range ?? `rows ${location.rowStart}–${location.rowEnd}, columns ${location.columnStart}–${location.columnEnd}`}`
  return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Extracted characters {value0}–{value1}; page mapping unavailable", { value0: String(location.charStart + 1), value1: String(location.charEnd) })
}

export function DocumentEvidence({ result, busy, onClose }: { result: DocumentEvidenceResult | null; busy: boolean; onClose: () => void }) {
  const tx = useAuthoredText()
  if (!result && !busy) return null
  return <section aria-label={tx("Document evidence")} aria-busy={busy} className="rounded-xl border border-accent/40 bg-card p-3.5">
    <div className="flex items-center justify-between gap-3">
      <h4 className="text-[13px] font-medium text-ink">{tx("Verified source evidence")}</h4>
      <Button variant="ghost" size="xs" onClick={onClose}>{tx("Close evidence")}</Button>
    </div>
    {busy ? <p role="status" className="mt-2 text-[13px] text-ink-3">{tx("Checking the original file and its revision…")}</p> : result?.status === 'ready' ? <>
      <p className="mt-2 break-all text-[13px] text-ink">{result.passage.sourceLabel} / {result.passage.relativePath}</p>
      <p className="mt-1 text-[12px] text-ink-3">{documentLocationLabel(result.passage.location)}</p>
      {result.passage.location.kind === 'extracted' && <p className="mt-1 text-[12px] text-ink-3">{result.passage.location.reason}</p>}
      <p className="mt-1 break-all text-[12px] text-ink-3">{tx("SHA-256 revision ")}{result.passage.revision}</p>
      <blockquote className="mt-3 whitespace-pre-wrap break-words rounded-lg bg-accent/10 p-3 text-[13px] text-ink"><mark className="bg-transparent text-inherit">{result.passage.text}</mark></blockquote>
      <p className="mt-2 text-[12px] text-ink-3">{tx("Original verified ")}{new Date(result.verifiedAt).toLocaleString(currentIntlLocale())}.</p>
    </> : <p role="alert" className="mt-2 text-[13px] text-danger">{result?.message}</p>}
  </section>
}
