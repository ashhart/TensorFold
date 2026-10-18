import { OriginalDiagnostic } from './OriginalDiagnostic'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import type { RunExportBundle } from '../lib/types'

// build.md Phase 7 — the import/replay half of run export. Load a previously exported
// (or shared) run bundle back and re-run its original prompt as a fresh task, so an
// exported run becomes a reproducible eval. Self-contained overlay opened by a
// `vontra:replay` CustomEvent from the command palette.
export function ReplayPanel() {
  const tx = useAuthoredText()
  const [open, setOpen] = useState(false)
  const [bundle, setBundle] = useState<RunExportBundle | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [importing, setImporting] = useState(false)

  useEffect(() => {
    const onOpen = () => { setOpen(true); setBundle(null); setError(null) }
    window.addEventListener('vontra:replay', onOpen)
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('vontra:replay', onOpen); window.removeEventListener('keydown', onKey) }
  }, [])

  if (!open) return null
  const close = () => setOpen(false)
  const prompt = (bundle?.messages ?? []).find((m) => m?.role === 'user')?.content ?? ''
  const evidence = bundle?.evidence

  const choose = async () => {
    setImporting(true); setError(null)
    try {
      const res = await api.importBundle()
      if (res.error) setError(res.error)
      else if (res.bundle) setBundle(res.bundle)
    } catch (e: any) { setError(String(e?.message ?? e)) }
    finally { setImporting(false) }
  }

  const replay = async () => {
    if (!prompt.trim()) return
    const s = useStore.getState().settings
    if (!s?.activeProviderId || !s.activeModel) { setError(tx("Configure a provider and model in Settings first.")); return }
    const result = await useStore.getState().launch({ text: prompt, cwd: s.defaultCwd || '', providerId: s.activeProviderId, model: s.activeModel, approvalMode: s.approvalMode, providerAccountId: s.activeProviderAccountId, projectId: null })
    if (!result.ok) { setError(result.error || tx("The replay could not start.")); return }
    setOpen(false)
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[10vh] fade-in" onClick={close}>
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      <div className="relative flex max-h-[80vh] w-[min(620px,calc(100vw-40px))] flex-col overflow-hidden rounded-2xl border border-line bg-card" onClick={(e) => e.stopPropagation()} style={{ boxShadow: 'var(--shadow-pop)' }}>
        <div className="flex items-center gap-2.5 border-b border-line px-4 py-3">
          <Icon name="play" size={16} className="text-ink-2" />
          <div className="flex-1 text-[14px] font-medium text-ink">{tx("Replay a run")}</div>
          <button onClick={close} className="rounded-md border border-line px-1.5 py-0.5 text-[11px] text-ink-4 hover:text-ink">{tx("esc")}</button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto scroll-thin p-4">
          {!bundle ? (
            <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-line px-4 py-10 text-center">
              <Icon name="archive" size={24} className="text-ink-4" />
              <div className="text-[13px] text-ink-2">{tx("Open an exported run bundle")}</div>
              <div className="max-w-[360px] text-[13px] text-ink-4">{tx("A ")}<code className="text-ink-3">vontra.run-export</code> {tx("JSON file — full, share, or eval packet. Its original prompt becomes a fresh task you can run against any model.")}</div>
              <button onClick={choose} disabled={importing} className="mt-1 flex items-center gap-1.5 rounded-md bg-accent px-3.5 py-1.5 text-[13px] text-accent-ink hover:bg-accent-strong disabled:opacity-50">
                <Icon name="folder" size={13} />{importing ? tx("Opening…") : tx("Choose bundle…")}
              </button>
              {error && <div className="mt-1 flex items-center gap-1.5 text-[13px] text-danger"><Icon name="warning" size={12} />{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</div>}
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
                <span className="rounded-md border border-line bg-card-2 px-2 py-0.5 capitalize text-ink-2">{bundle.mode} {tx("packet")}</span>
                {bundle.redacted && <span className="rounded-md border border-line bg-card-2 px-2 py-0.5 text-ink-3">{tx("redacted")}</span>}
                {bundle.anonymized && <span className="rounded-md border border-line bg-card-2 px-2 py-0.5 text-ink-3">{tx("anonymized")}</span>}
                <span className="text-ink-4">{bundle.stats?.messages ?? bundle.messages?.length ?? 0} {tx("msgs · ")}{bundle.stats?.toolCalls ?? 0} {tx("tools")}</span>
                {evidence && <span className="text-ink-4">· {evidence.traceIds.length} {tx("trace ids")}</span>}
              </div>
              <div className="mt-2 text-[14px] font-medium text-ink">{bundle.thread?.title || tx("Imported run")}</div>
              {bundle.chronicle && (
                <div className="mt-2 rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] leading-snug text-ink-3">
                  {bundle.chronicle.summary}
                </div>
              )}
              {evidence && (
                <div className="mt-2 flex flex-wrap gap-1.5 text-[12px] text-ink-3">
                  <span className="rounded-md border border-line bg-card-2 px-2 py-0.5">{evidence.touchedFiles.length} {tx("files")}</span>
                  <span className="rounded-md border border-line bg-card-2 px-2 py-0.5">{evidence.commandOutputs.length} {tx("commands")}</span>
                  <span className="rounded-md border border-line bg-card-2 px-2 py-0.5">{evidence.citations.length} {tx("citations")}</span>
                  <span className="rounded-md border border-line bg-card-2 px-2 py-0.5">{evidence.assistantClaims?.length ?? 0} {tx("claims")}</span>
                  {evidence.gaps.length > 0 && <span className="rounded-md border border-warn/30 bg-warn/10 px-2 py-0.5 text-warn">{evidence.gaps.length} {tx("gaps")}</span>}
                </div>
              )}

              <div className="mt-3 text-[12px] font-medium uppercase tracking-wide text-ink-3">{tx("Original prompt")}</div>
              {prompt ? (
                <pre className="scroll-thin mt-1 max-h-[34vh] overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-canvas px-3 py-2 text-[13px] leading-relaxed text-ink-2">{prompt}</pre>
              ) : (
                <div className="mt-1 rounded-lg border border-dashed border-line px-3 py-3 text-[13px] text-ink-4">{tx("This bundle has no user prompt to replay.")}</div>
              )}
              {bundle.redacted && prompt && <div className="mt-1.5 flex items-center gap-1.5 text-[12px] text-ink-4"><Icon name="shield" size={11} />{tx("Secrets were redacted on export — the replay uses the redacted prompt text.")}</div>}
              {error && <div className="mt-2 flex items-center gap-1.5 text-[13px] text-danger"><Icon name="warning" size={12} />{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</div>}
            </>
          )}
        </div>

        <div className="flex items-center gap-2 border-t border-line px-4 py-2.5">
          {bundle && <button onClick={choose} className="rounded-md border border-line px-2.5 py-1.5 text-[13px] text-ink-3 hover:text-ink">{tx("Choose another…")}</button>}
          <div className="flex-1" />
          <button onClick={close} className="rounded-md px-3 py-1.5 text-[13px] text-ink-3 hover:text-ink">{tx("Close")}</button>
          <button onClick={replay} disabled={!prompt.trim()} className={clsx('flex items-center gap-1.5 rounded-md px-3.5 py-1.5 text-[13px]', prompt.trim() ? 'bg-accent text-accent-ink hover:bg-accent-strong' : 'bg-ink-5 text-ink-4')}>
            <Icon name="play" size={13} />{tx("Replay as new task")}</button>
        </div>
      </div>
    </div>
  )
}
