import type { DocumentSourceResult } from '../../../shared/document-processing'
import { DocumentProcessingPanel } from './DocumentProcessingPanel'
import { DocumentSourceViewer } from './DocumentSourceViewer'
import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { DocumentLibraryApi, DocumentLibraryPassage, DocumentLibrarySearchResult, DocumentLibrarySnapshot } from '../../../shared/document-library'
import { api } from '../../lib/ipc'
import { Button, Input } from '../ui'
import { documentLocationLabel } from './DocumentEvidence'

export function DocumentLibraryPanel({ refreshKey, busy = false, onRetry, client = api }: { refreshKey: string; busy?: boolean; onRetry: () => void | Promise<void>; client?: DocumentLibraryApi }) {
  const tx = useAuthoredText()
  const [snapshot, setSnapshot] = useState<DocumentLibrarySnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [sourceId, setSourceId] = useState('')
  const [results, setResults] = useState<DocumentLibrarySearchResult | null>(null)
  const [searching, setSearching] = useState(false)
  const [purging, setPurging] = useState(false)
  const [evidence, setEvidence] = useState<DocumentSourceResult | null>(null)
  const [evidenceBusy, setEvidenceBusy] = useState(false)
  const [visible, setVisible] = useState(100)
  const request = useRef(0)
  const evidenceRequest = useRef(0)
  const listRequest = useRef(0)
  const resetSearch = useCallback(() => { request.current++; evidenceRequest.current++; setSearching(false); setResults(null); setEvidence(null); setEvidenceBusy(false) }, [])
  const load = useCallback(async () => {
    const id = ++listRequest.current
    setLoading(true)
    setError('')
    try { const next = await client.documentLibraryList(); if (id === listRequest.current) setSnapshot(next) }
    catch (caught) { if (id === listRequest.current) setError(String((caught as Error)?.message ?? caught)) }
    finally { if (id === listRequest.current) setLoading(false) }
  }, [client])
  useEffect(() => { resetSearch(); setVisible(100); void load(); return () => { listRequest.current++; request.current++; evidenceRequest.current++ } }, [refreshKey, load, resetSearch])
  const search = async () => {
    const id = ++request.current
    evidenceRequest.current++
    setSearching(true); setError(''); setEvidence(null); setEvidenceBusy(false)
    try { const found = await client.documentLibrarySearch({ query, sourceId: sourceId || undefined, limit: 30 }); if (id === request.current) setResults(found) }
    catch (caught) { if (id === request.current) setError(String((caught as Error)?.message ?? caught)) }
    finally { if (id === request.current) setSearching(false) }
  }
  const open = async (passage: DocumentLibraryPassage) => {
    const id = ++evidenceRequest.current
    setEvidence(null); setEvidenceBusy(true)
    try { const result = await client.documentLibraryReadSource({ documentId: passage.documentId, revision: passage.revision, passageId: passage.passageId }); if (id === evidenceRequest.current) setEvidence(result) }
    catch (caught) { if (id === evidenceRequest.current) setEvidence({ status: 'unavailable', message: String((caught as Error)?.message ?? caught) }) }
    finally { if (id === evidenceRequest.current) setEvidenceBusy(false) }
  }
  const purge = async () => {
    setPurging(true); setError(''); resetSearch()
    try { setSnapshot(await client.documentLibraryPurge(sourceId || undefined)) }
    catch (caught) { setError(String((caught as Error)?.message ?? caught)) }
    finally { setPurging(false) }
  }
  const files = (snapshot?.files ?? []).filter(file => !sourceId || file.sourceId === sourceId)
  const selected = snapshot?.sources.find(source => source.id === sourceId)
  const ready = files.filter(file => file.status === 'ready').length
  const failed = files.filter(file => file.status === 'failed').length
  const unavailable = files.length - ready - failed
  return <section aria-label={tx("Managed document library")} aria-busy={busy || loading || purging} className="space-y-3 rounded-xl border border-line bg-card p-3.5 shadow-card">
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div><h3 className="text-[13px] font-medium text-ink">{tx("Local document library")}</h3><p className="mt-1 text-[12px] text-ink-3">{tx("Search complete indexed text within the configured limits, then verify the original before reading evidence.")}</p></div>
      <Button size="xs" variant="ghost" disabled={loading || busy || purging} onClick={() => void load()}>{tx("Refresh file states")}</Button>
    </div>
    {error && <p role="alert" className="break-words text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    {busy && <p role="status" className="text-[13px] text-ink-3">{tx("Processing enabled local collections…")}</p>}
    {loading && <p role="status" className="text-[13px] text-ink-3">{tx("Checking indexed files…")}</p>}
    <DocumentProcessingPanel client={client} refreshKey={refreshKey} disabled={purging} onChanged={load} />
    {snapshot && <>
      <label className="block text-[12px] text-ink-2">{tx("Collection")}<select value={sourceId} onChange={event => { setSourceId(event.target.value); setVisible(100); resetSearch() }} className="mt-1 block w-full min-w-0 rounded-lg border border-line bg-card px-2 py-2 text-[13px] text-ink"><option value="">{tx("All configured collections")}</option>{snapshot.sources.map(source => <option key={source.id} value={source.id}>{source.label}{source.enabled ? '' : tx(" · disabled")}</option>)}</select></label>
      <p role="status" className="text-[13px] text-ink-2">{ready} {tx("ready · ")}{failed} {tx("failed · ")}{unavailable} {tx("skipped or unsupported")}{selected && !selected.enabled ? tx(" · collection disabled") : ''}</p>
      {snapshot.sources.filter(source => (!sourceId || source.id === sourceId) && source.lastScan?.truncated).map(source => <p key={source.id} className="text-[13px] text-danger">{source.label}{tx(": scan limits were reached, so some paths were not examined.")}</p>)}
      <form onSubmit={event => { event.preventDefault(); void search() }} className="flex flex-wrap gap-2">
        <label className="min-w-0 flex-1 basis-48"><span className="sr-only">{tx("Search document contents")}</span><Input value={query} maxLength={256} onChange={event => { setQuery(event.target.value); resetSearch() }} placeholder={tx("Find a phrase anywhere in indexed text")} /></label>
        <Button type="submit" size="sm" disabled={!query.trim() || busy || purging || searching}>{tx("Search documents")}</Button>
        {searching && <Button size="sm" variant="ghost" onClick={resetSearch}>{tx("Cancel search")}</Button>}
      </form>
      {searching && <p role="status" className="text-[13px] text-ink-3">{tx("Searching locally and checking matching originals…")}</p>}
      {results && <div aria-label={tx("Document search results")} className="space-y-2">
        <p role="status" className="text-[13px] text-ink-2">{tx("Passages found: {count}", { count: results.passages.length })}{results.truncated ? tx("; result limit reached") : ''}.</p>
        {results.warnings?.map((warning, index) => <p key={index} className="break-words text-[12px] text-danger">{warning}</p>)}
        {results.passages.map(passage => <div key={`${passage.documentId}-${passage.passageId}`} className="rounded-lg border border-line p-3">
          <p className="break-all text-[13px] font-medium text-ink">{passage.sourceLabel} / {passage.relativePath}</p>
          <p className="mt-1 text-[12px] text-ink-3">{documentLocationLabel(passage.location)}</p>
          <p className="mt-2 whitespace-pre-wrap break-words text-[13px] text-ink-2">{passage.text}</p>
          <Button size="xs" variant="ghost" className="mt-2" onClick={() => void open(passage)} disabled={evidenceBusy}>{tx("Verify and open evidence")}</Button>
        </div>)}
      </div>}
      <DocumentSourceViewer result={evidence} busy={evidenceBusy} onClose={() => { evidenceRequest.current++; setEvidence(null); setEvidenceBusy(false) }} />
      {!files.length ? <p className="text-[13px] text-ink-3">{snapshot.sources.length ? tx("No indexed files in this view; sync enabled collections to build the library.") : tx("Add a local folder to create your first collection.")}</p> : <>
        <ul aria-label={tx("Collection files and processing outcomes")} className="max-h-80 space-y-2 overflow-auto">
          {files.slice(0, visible).map(file => <li key={`${file.sourceId}-${file.id}`} className="rounded-lg border border-line px-3 py-2">
            <div className="flex flex-wrap items-start justify-between gap-2"><span className="min-w-0 break-all text-[13px] text-ink">{file.relativePath}</span><span className={file.status === 'ready' ? 'text-[12px] text-positive' : 'text-[12px] text-ink-3'}>{file.verification && file.verification !== 'current' ? file.verification : file.status}</span></div>
            {!sourceId && <p className="mt-1 text-[12px] text-ink-3">{snapshot.sources.find(source => source.id === file.sourceId)?.label}</p>}
            {file.error && <p className="mt-1 break-words text-[12px] text-danger">{<OriginalDiagnostic>{file.error}</OriginalDiagnostic>}</p>}
            {file.locationNote && <p className="mt-1 text-[12px] text-ink-3">{file.locationNote}</p>}
            {file.revision && <p className="mt-1 text-[12px] text-ink-3">{tx("Revision ")}{file.revision.slice(0, 12)} · {file.indexedCharacters?.toLocaleString(currentIntlLocale()) ?? 0} {tx("indexed characters")}</p>}
            {file.retryable && <Button size="xs" variant="ghost" className="mt-1" disabled={busy || purging || !snapshot.sources.find(source => source.id === file.sourceId)?.enabled} onClick={() => void onRetry()}>{tx("Retry scan")}</Button>}
          </li>)}
        </ul>
        {files.length > visible && <Button size="xs" onClick={() => setVisible(count => count + 100)}>{tx("Show more files")}</Button>}
      </>}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3"><p className="text-[12px] text-ink-3">{tx("Index saved ")}{snapshot.indexedAt ? new Date(snapshot.indexedAt).toLocaleString(currentIntlLocale()) : tx("never")}{tx("; clearing removes cached text and leaves source files in place.")}</p><Button size="xs" variant="danger" disabled={busy || purging || !files.length && !snapshot.sources.length} onClick={() => void purge()}>{purging ? tx("Clearing index…") : selected ? tx("Clear collection index") : tx("Clear library index")}</Button></div>
    </>}
  </section>
}
