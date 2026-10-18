import type { DocumentSourceResult } from '../../../shared/document-processing'
import { DocumentSourceViewer } from './DocumentSourceViewer'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import type { DocumentEvidenceRequest } from '../../../shared/document-library'
import { api } from '../../lib/ipc'

export function DocumentEvidenceButtons({ references }: { references: DocumentEvidenceRequest[] }) {
  const tx = useAuthoredText()
  const [result, setResult] = useState<DocumentSourceResult | null>(null), [busy, setBusy] = useState(false)
  const epoch = useRef(0)
  useEffect(() => { epoch.current++; setResult(null); setBusy(false); return () => { epoch.current++ } }, [references])
  const open = async (request: DocumentEvidenceRequest) => {
    const own = ++epoch.current; setResult(null); setBusy(true)
    try { const value = await api.documentLibraryReadSource(request); if (epoch.current === own) setResult(value) }
    catch { if (epoch.current === own) setResult({ status: 'unavailable', message: tx("The source could not be verified. Sync the document library and retry.") }) }
    finally { if (epoch.current === own) setBusy(false) }
  }
  return <div className="mt-2 space-y-2"><div className="flex flex-wrap gap-2" aria-label={tx("Document source evidence")}>{references.slice(0, 20).map((request, index) => <button key={`${request.documentId}:${request.passageId}`} type="button" onClick={() => void open(request)} className="rounded-lg border border-line px-2.5 py-1 text-[12px] text-accent focus-visible:outline-2 focus-visible:outline-accent">{tx("Inspect source ")}{index + 1}</button>)}</div><DocumentSourceViewer result={result} busy={busy} onClose={() => { epoch.current++; setResult(null); setBusy(false) }} /></div>
}
