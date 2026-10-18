import { useCallback, useEffect, useRef, useState } from 'react'
import type { DocumentLibraryApi } from '../../../shared/document-library'
import type { DocumentProcessingRequest, DocumentProcessingSnapshot, DocumentJobState } from '../../../shared/document-processing'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Button } from '../ui'

const labels: Record<DocumentJobState, string> = { queued: 'Queued', processing: 'Processing', paused: 'Paused', ready: 'Ready', failed: 'Failed', unsupported: 'Unsupported', skipped: 'Skipped', cancelled: 'Cancelled' }
export function DocumentProcessingPanel({ client, refreshKey, disabled, onChanged }: { client: DocumentLibraryApi; refreshKey: string; disabled: boolean; onChanged: () => void | Promise<void> }) {
  const tx = useAuthoredText()
  const [queue, setQueue] = useState<DocumentProcessingSnapshot | null>(null), [error, setError] = useState(''), [pending, setPending] = useState(false)
  const epoch = useRef(0), changing = useRef(false), latestChanged = useRef(onChanged), previousRunning = useRef(false), requestEpoch = useRef(0)
  latestChanged.current = onChanged
  const refresh = useCallback(async () => {
    const own = epoch.current, request = ++requestEpoch.current
    try {
      const value = await client.documentLibraryProcessing()
      if (own !== epoch.current || request !== requestEpoch.current) return
      setQueue(value)
      if (previousRunning.current && !value.running) void latestChanged.current()
      previousRunning.current = value.running
    } catch (caught) { if (own === epoch.current) setError(String((caught as Error)?.message ?? caught)) }
  }, [client])
  useEffect(() => { epoch.current++; setQueue(null); setError(''); previousRunning.current = false; void refresh(); return () => { epoch.current++ } }, [refreshKey, refresh])
  useEffect(() => { if (!queue?.running && !queue?.jobs.some(job => job.state === 'queued' || job.state === 'processing')) return; const timer = setInterval(() => void refresh(), 800); return () => clearInterval(timer) }, [queue, refresh])
  const control = async (request: DocumentProcessingRequest) => {
    if (changing.current || disabled) return
    const own = epoch.current; requestEpoch.current++; changing.current = true; setPending(true); setError('')
    try { const value = await client.documentLibraryProcess(request); if (own === epoch.current) { setQueue(value); void latestChanged.current() } }
    catch (caught) { if (own === epoch.current) setError(String((caught as Error)?.message ?? caught)) }
    finally { changing.current = false; if (own === epoch.current) setPending(false) }
  }
  const active = queue?.jobs.some(job => ['queued', 'processing', 'paused'].includes(job.state)) ?? false
  const completed = queue?.jobs.filter(job => job.state === 'ready').length ?? 0
  return <section aria-label={tx('Document processing queue')} className="space-y-2 rounded-lg border border-line p-3">
    <div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-[13px] font-medium text-ink">{tx('Document processing')}</h4><div className="flex flex-wrap gap-2"><Button size="xs" disabled={disabled || pending || active} onClick={() => void control({ action: 'start' })}>{tx('Process enabled collections')}</Button>{active && <><Button size="xs" disabled={disabled || pending} onClick={() => void control({ action: queue?.paused ? 'resume' : 'pause' })}>{queue?.paused ? tx('Resume queue') : tx('Pause queue')}</Button><Button size="xs" variant="ghost" disabled={disabled || pending} onClick={() => void control({ action: 'cancel' })}>{tx('Cancel remaining')}</Button></>}</div></div>
    <p className="text-[12px] text-ink-3">{tx('One file is processed at a time. Progress is saved after each file; pausing discards an in-progress extraction and resumes from that file.')}</p>
    {error && <p role="alert" className="text-[12px] text-danger"><OriginalDiagnostic>{error}</OriginalDiagnostic></p>}
    {queue && <><p role="status" className="text-[12px] text-ink-2">{tx('Ready: {completed} of {total} files', { completed, total: queue.jobs.length })}</p>{queue.truncated && <p role="alert" className="text-[12px] text-danger">{tx('Inventory limits were reached; some paths were not examined. Narrow the configured collections.')}</p>}<ul className="max-h-64 space-y-1 overflow-auto">{queue.jobs.map(job => <li key={job.id} className="rounded-lg border border-line p-2"><div className="flex flex-wrap justify-between gap-2"><span className="min-w-0 break-all text-[12px] text-ink">{job.relativePath}</span><span className="text-[12px] text-ink-3">{tx(labels[job.state])}</span></div>{job.error && <p className="break-words text-[12px] text-danger"><OriginalDiagnostic>{job.error}</OriginalDiagnostic></p>}<div className="mt-1 flex flex-wrap gap-2">{['queued', 'processing'].includes(job.state) && <Button size="xs" variant="ghost" disabled={pending || disabled} onClick={() => void control({ action: 'pause', jobId: job.id })}>{tx('Pause file')}</Button>}{job.state === 'paused' && <Button size="xs" variant="ghost" disabled={pending || disabled} onClick={() => void control({ action: 'resume', jobId: job.id })}>{tx('Resume file')}</Button>}{['failed', 'cancelled', 'skipped'].includes(job.state) && <Button size="xs" variant="ghost" disabled={pending || disabled} onClick={() => void control({ action: 'retry', jobId: job.id })}>{tx('Retry file')}</Button>}{['queued', 'processing', 'paused'].includes(job.state) && <Button size="xs" variant="ghost" disabled={pending || disabled} onClick={() => void control({ action: 'cancel', jobId: job.id })}>{tx('Cancel file')}</Button>}</div></li>)}</ul></>}
  </section>
}
