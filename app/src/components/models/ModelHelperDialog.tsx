import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { Thread } from '../../lib/types'
import type { ModelHelperRole } from '../../../shared/model-fit'
import type { ModelHelperResult } from '../../../shared/model-helper'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { useAccessibleDialog } from '../navigation/useAccessibleDialog'
import { DocumentEvidenceButtons } from '../documents/DocumentEvidenceButtons'

export function ModelHelperDialog({ thread, onClose }: { thread: Thread; onClose(): void }) {
  const tx = useAuthoredText()
  const defaults = useStore(state => state.settings?.modelHelperDefaults)
  const [role, setRole] = useState<ModelHelperRole>('document_extraction'), [text, setText] = useState('')
  const [result, setResult] = useState<ModelHelperResult | null>(null), [error, setError] = useState(''), [busy, setBusy] = useState(false), [notice, setNotice] = useState('')
  const panel = useRef<HTMLDivElement>(null), initial = useRef<HTMLSelectElement>(null), request = useRef<string | null>(null), epoch = useRef(0)
  useAccessibleDialog({ open: true, onClose, dialogRef: panel, initialFocusRef: initial })
  useEffect(() => () => { epoch.current++; if (request.current) void api.modelHelperCancel(request.current).catch(() => {}) }, [])
  const run = async () => {
    if (busy || !text.trim() || !defaults?.[role]) return
    const own = ++epoch.current, id = crypto.randomUUID(); request.current = id
    setBusy(true); setError(''); setNotice(''); setResult(null)
    try { const output = await api.modelHelperRun({ requestId: id, threadId: thread.id, role, text }); if (epoch.current === own) setResult(output) }
    catch { if (epoch.current === own) setError(tx("The selected helper could not finish. Check its loaded model and capability evidence in model settings; no alternative provider was used.")) }
    finally { if (epoch.current === own) { setBusy(false); request.current = null } }
  }
  const control = 'rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50'
  return createPortal(<div className="fixed inset-0 z-[130] flex items-center justify-center bg-canvas/85 p-3"><div ref={panel} role="dialog" aria-modal="true" aria-labelledby="model-helper-title" tabIndex={-1} className="scroll-thin max-h-full w-full max-w-[700px] space-y-4 overflow-y-auto rounded-xl border border-line bg-canvas p-5 shadow-pop">
    <header className="flex items-center justify-between gap-3"><h2 id="model-helper-title" className="text-[16px] font-medium text-ink">{tx("Local helper jobs")}</h2><button type="button" className={control} onClick={onClose}>{tx("Close")}</button></header>
    <p className="text-[13px] leading-relaxed text-ink-2">{tx("Choose a small job and supply only the text you want it to use; helpers have no tools, never fall back to the cloud and never append to this conversation. Source answers also inspect matching enabled local document passages.")}</p>
    <label className="block text-[13px] text-ink-2">{tx("Job")}<select ref={initial} aria-label={tx("Helper job")} disabled={busy} value={role} className={`mt-1 w-full ${control}`} onChange={event => { epoch.current++; setRole(event.target.value as ModelHelperRole); setResult(null); setError(''); setNotice('') }}><option value="naming">{tx("Conversation title")}</option><option value="document_extraction">{tx("Extract document facts")}</option><option value="search">{tx("Answer from local sources")}</option></select></label>
    <p className="text-[12px] text-ink-3">{defaults?.[role] ? tx("Selected helper: {value0} · must be loaded and qualified", { value0: defaults[role]!.model }) : tx("No helper selected for this job; choose one in Settings → Models → Optional helper models.")}</p>
    <label className="block text-[13px] text-ink-2">{role === 'search' ? tx("Question or search words") : tx("Source text")}<textarea aria-label={tx("Helper source text")} dir="auto" rows={5} maxLength={20_000} disabled={busy} value={text} className={`mt-1 w-full resize-y ${control}`} onChange={event => { setText(event.target.value); setResult(null) }} /></label>
    <div className="flex flex-wrap gap-2"><button type="button" disabled={busy || !text.trim() || !defaults?.[role]} className={control} onClick={() => void run()}>{busy ? tx("Running selected helper…") : tx("Run local helper")}</button>{busy && <button type="button" className={control} onClick={() => request.current && void api.modelHelperCancel(request.current)}>{tx("Cancel helper")}</button>}</div>
    {error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}{notice && <p role="status" className="text-[12px] text-ink-2">{notice}</p>}
    {result && <section aria-label={tx("Helper result")} className="space-y-3 rounded-lg border border-line bg-card p-3"><p className="text-[12px] text-ink-3">{result.route.model} · {result.complete ? tx("Finished local response") : tx("Incomplete response")}</p><pre dir="auto" className="whitespace-pre-wrap break-words font-sans text-[14px] leading-relaxed text-ink">{result.text}</pre>{result.warning && <p className="text-[12px] text-warn">{result.warning}</p>}<div className="flex flex-wrap gap-2"><button type="button" className={control} onClick={() => void api.writeClipboard(result.text).then(value => setNotice(value?.ok === false ? tx("Copy failed; select the result text and copy it.") : tx("Helper result copied.")), () => setNotice(tx("Copy failed; select the result text and copy it.")))}>{tx("Copy result")}</button>{role === 'naming' && <button type="button" className={control} onClick={() => void useStore.getState().renameThread(thread.id, result.text.replace(/\s+/g, ' ').slice(0, 80)).then(() => setNotice(tx("Conversation renamed.")), () => setNotice(tx("The conversation could not be renamed.")))}>{tx("Use as conversation title")}</button>}</div>{result.sources.length > 0 && <DocumentEvidenceButtons references={result.sources} />}</section>}
  </div></div>, document.body)
}
