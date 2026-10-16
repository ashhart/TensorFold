import clsx from 'clsx'
import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import type { Artifact, ArtifactPreview, Thread } from '../../lib/types'
import { PanelSection } from './primitives'

// VST-011 first-class artifacts: what this run PRODUCED, with provenance and a redaction
// status. Preview expands textual artifacts inline; Redact makes a safe masked copy inside
// the record (the original file is never edited).
export function ArtifactsSection({ thread }: { thread: Thread }) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [openPreview, setOpenPreview] = useState<{ id: string; preview: ArtifactPreview } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const reload = () => { api.listArtifacts(thread.id).then(setArtifacts).catch(() => setArtifacts([])) }
  useEffect(reload, [thread.id, thread.updatedAt])

  if (artifacts.length === 0) return null

  const togglePreview = async (artifact: Artifact) => {
    if (openPreview?.id === artifact.id) { setOpenPreview(null); return }
    try {
      setOpenPreview({ id: artifact.id, preview: await api.previewArtifact(artifact.id) })
    } catch (e) { setError(e instanceof Error ? e.message : 'Preview failed.') }
  }
  const redact = async (artifact: Artifact) => {
    setError(null)
    try {
      await api.redactArtifact(artifact.id)
      setOpenPreview(null)
      reload()
    } catch (e) { setError(e instanceof Error ? e.message : 'Redaction failed.') }
  }

  return (
    <>
      <div className="mx-4 border-t border-line" />
      <PanelSection title="Artifacts" right={<span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{artifacts.length}</span>}>
        <div className="space-y-0.5">
          {artifacts.map((artifact) => (
            <div key={artifact.id}>
              <div className="group flex h-7 items-center gap-2 rounded-lg px-1.5 text-[13px] transition-colors hover:bg-ink-5">
                <Icon name="file" size={13} className="shrink-0 text-ink-3" />
                <button onClick={() => void togglePreview(artifact)} className="min-w-0 flex-1 truncate text-left text-ink/85 hover:text-ink" title={`${artifact.kind}${artifact.provenance.turn != null ? ` · turn ${artifact.provenance.turn}` : ''}${artifact.provenance.sourceRef ? ` · from ${artifact.provenance.sourceRef}` : ''}`}>
                  {artifact.title}
                </button>
                <span className={clsx('shrink-0 text-[11px]', artifact.redaction === 'redacted' ? 'text-positive' : artifact.redaction === 'not_required' ? 'text-ink-4' : 'text-warn')}>
                  {artifact.redaction === 'redacted' ? 'redacted' : artifact.redaction === 'not_required' ? 'binary' : 'unredacted'}
                </span>
                {artifact.redaction === 'unredacted' && (
                  <button onClick={() => void redact(artifact)} title="Make a masked safe copy inside the record (the original file is untouched)." className="shrink-0 rounded-md px-1.5 py-0.5 text-[12px] font-medium text-ink-3 opacity-0 transition-all hover:bg-ink-5 hover:text-ink group-hover:opacity-100">
                    Redact
                  </button>
                )}
              </div>
              {openPreview?.id === artifact.id && (
                <div className="mb-1 ml-6 rounded-md border border-line bg-card-2 p-2">
                  <div className="mb-1 text-[11px] text-ink-4">{openPreview.preview.meta}</div>
                  {openPreview.preview.excerpt != null
                    ? <pre className="scroll-thin max-h-32 overflow-y-auto whitespace-pre-wrap break-words font-mono text-[12px] leading-snug text-ink-2">{openPreview.preview.excerpt}{openPreview.preview.truncated ? '\n…' : ''}</pre>
                    : <div className="text-[12px] text-ink-4">No inline preview for this kind.</div>}
                </div>
              )}
            </div>
          ))}
        </div>
        {error && <p className="mt-1 px-1.5 text-[12px] text-danger">{error}</p>}
      </PanelSection>
    </>
  )
}
