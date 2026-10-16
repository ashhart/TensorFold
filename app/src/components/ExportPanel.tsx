import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { Icon, type IconName } from '../lib/icons'
import type { ExportMode, ExportResult } from '../lib/types'

// build.md Phase 7 — structured run export & sharing. Self-contained overlay: opens
// on a `vontra:export` CustomEvent ({ detail: { threadId } }) dispatched from the
// command palette. Secret redaction is ON by default; share/eval packets also
// anonymize identifiers. The bundle is built in the main process (export.ts).
const MODES: { id: ExportMode; label: string; icon: IconName; blurb: string }[] = [
  { id: 'full', label: 'Full run', icon: 'archive', blurb: 'Everything — prompts, tool calls, diffs, outcomes. Secrets redacted; identifiers kept.' },
  { id: 'share', label: 'Share packet', icon: 'send', blurb: 'Anonymized for sharing — thread id, cwd, provider, paths and IPs stripped.' },
  { id: 'eval', label: 'Eval replay', icon: 'spark', blurb: 'Replay skeleton — prompts, tool calls and outcomes only; reasoning dropped, anonymized.' },
]

export function ExportPanel() {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [mode, setMode] = useState<ExportMode>('full')
  const [redact, setRedact] = useState(true)
  const [result, setResult] = useState<ExportResult | null>(null)
  const [building, setBuilding] = useState(false)
  const [saved, setSaved] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const reqRef = useRef(0)

  useEffect(() => {
    const onOpen = (e: Event) => {
      const id = (e as CustomEvent).detail?.threadId
      if (typeof id !== 'string') return
      setThreadId(id); setMode('full'); setRedact(true); setSaved(null); setCopied(false)
    }
    window.addEventListener('vontra:export', onOpen as EventListener)
    return () => window.removeEventListener('vontra:export', onOpen as EventListener)
  }, [])

  // Rebuild the bundle whenever the target or options change (request-id guards races).
  useEffect(() => {
    if (!threadId) { setResult(null); return }
    const req = ++reqRef.current
    setBuilding(true); setSaved(null); setCopied(false)
    api.buildExport(threadId, { mode, redact })
      .then((r) => { if (req === reqRef.current) setResult(r) })
      .catch(() => { if (req === reqRef.current) setResult(null) })
      .finally(() => { if (req === reqRef.current) setBuilding(false) })
  }, [threadId, mode, redact])

  if (!threadId) return null
  const close = () => setThreadId(null)
  const bundle = result?.bundle
  const totalRedactions = bundle?.redactionReport.reduce((n, r) => n + r.count, 0) ?? 0
  const evidence = bundle?.evidence
  const evidenceItems = evidence
    ? evidence.touchedFiles.length + evidence.diffs.length + evidence.commandOutputs.length + evidence.screenshots.length + evidence.citations.length + evidence.approvals.length + (evidence.assistantClaims?.length ?? 0)
    : 0

  const onCopy = async () => {
    if (!result) return
    try { await navigator.clipboard.writeText(result.json); setCopied(true); setTimeout(() => setCopied(false), 1600) } catch { /* clipboard blocked */ }
  }
  const onSave = async () => {
    if (!threadId) return
    const res = await api.saveExport(threadId, { mode, redact })
    if (res.saved) setSaved(res.path ?? 'saved')
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[9vh] fade-in" onClick={close}>
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      <div className="relative flex max-h-[82vh] w-[min(680px,calc(100vw-40px))] flex-col overflow-hidden rounded-2xl border border-line bg-card" onClick={(e) => e.stopPropagation()} style={{ boxShadow: 'var(--shadow-pop)' }}>
        {/* header */}
        <div className="flex items-center gap-2.5 border-b border-line px-4 py-3">
          <Icon name="archive" size={16} className="text-ink-2" />
          <div className="min-w-0 flex-1">
            <div className="text-[14px] font-medium text-ink">Export run</div>
            <div className="truncate text-[13px] text-ink-3">{bundle?.thread.title || 'Building bundle…'}</div>
          </div>
          <button onClick={close} className="rounded-md border border-line px-1.5 py-0.5 text-[11px] text-ink-4 hover:text-ink">esc</button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto scroll-thin p-4">
          {/* mode picker */}
          <div className="grid grid-cols-3 gap-2">
            {MODES.map((m) => (
              <button key={m.id} onClick={() => setMode(m.id)} className={clsx('flex flex-col gap-1 rounded-lg border px-2.5 py-2 text-left transition-colors', mode === m.id ? 'border-ink-3 bg-ink-5' : 'border-line hover:border-line-strong')}>
                <span className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><Icon name={m.icon} size={13} className="text-ink-3" />{m.label}</span>
              </button>
            ))}
          </div>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-3">{MODES.find((m) => m.id === mode)?.blurb}</p>

          {/* redaction control */}
          <label className="mt-3 flex items-start gap-2.5 rounded-lg border border-line bg-card-2 px-3 py-2.5">
            <input type="checkbox" checked={redact} onChange={(e) => setRedact(e.target.checked)} className="mt-0.5 accent-[var(--ink)]" />
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><Icon name="shield" size={13} className={redact ? 'text-positive' : 'text-warn'} />Redact secrets &amp; PII</span>
              <span className="mt-0.5 block text-[12px] leading-relaxed text-ink-3">{redact ? 'Scans every value for API keys, tokens, private keys, JWTs, emails and home paths before export.' : 'Off — the bundle may contain raw secrets. Only disable for a local, trusted destination.'}</span>
            </span>
            {redact && <span className="shrink-0 rounded-full bg-positive/10 px-2 py-0.5 text-[12px] font-medium text-positive">{totalRedactions} masked</span>}
          </label>
          {!redact && <div className="mt-1.5 flex items-center gap-1.5 text-[12px] text-warn"><Icon name="warning" size={12} />Unredacted export — handle the file as a secret.</div>}

          {/* stats + redaction breakdown */}
          {bundle && (
            <div className="mt-3 flex flex-wrap gap-1.5 text-[12px]">
              <Stat label="messages" value={bundle.stats.messages} />
              <Stat label="tool calls" value={bundle.stats.toolCalls} />
              <Stat label="diffs" value={bundle.stats.diffs} />
              <Stat label="evidence" value={evidenceItems} />
              {bundle.stats.denied > 0 && <Stat label="denied" value={bundle.stats.denied} tone="warn" />}
              {bundle.stats.errors > 0 && <Stat label="errors" value={bundle.stats.errors} tone="danger" />}
              {!!evidence?.gaps.length && <Stat label="gaps" value={evidence.gaps.length} tone="warn" />}
              {bundle.anonymized && <span className="rounded-md border border-line bg-card-2 px-2 py-0.5 text-ink-3">anonymized</span>}
            </div>
          )}
          {redact && bundle && bundle.redactionReport.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {bundle.redactionReport.map((r) => (
                <span key={r.kind} className="rounded-full border border-line px-2 py-0.5 text-[12px] text-ink-3"><span className="text-ink-2">{r.count}×</span> {r.kind.replace(/_/g, ' ')}</span>
              ))}
            </div>
          )}

          {bundle?.chronicle && (
            <div className="mt-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
              <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink">
                <Icon name="list" size={13} className="text-ink-3" />Run chronicle
              </div>
              <div className="mt-1 text-[13px] leading-snug text-ink-3">{bundle.chronicle.summary}</div>
            </div>
          )}

          {evidence && (
            <div className="mt-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
              <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink">
                <Icon name="archive" size={13} className="text-ink-3" />Evidence bundle
              </div>
              <div className="mt-2 grid grid-cols-2 gap-1.5 text-[12px] text-ink-3 sm:grid-cols-4">
                <Badge label="files" value={evidence.touchedFiles.length} />
                <Badge label="commands" value={evidence.commandOutputs.length} />
                <Badge label="citations" value={evidence.citations.length} />
                <Badge label="approvals" value={evidence.approvals.length} />
                <Badge label="claims" value={evidence.assistantClaims?.length ?? 0} />
              </div>
              {(evidence.touchedFiles.length > 0 || evidence.gaps.length > 0) && (
                <div className="mt-2 space-y-1.5">
                  {evidence.touchedFiles.slice(0, 3).map((file) => (
                    <div key={file.path} className="truncate font-mono text-[12px] text-ink-4">{file.path}</div>
                  ))}
                  {evidence.gaps.slice(0, 2).map((gap) => (
                    <div key={`${gap.itemId}:${gap.claim}`} className="line-clamp-2 rounded-md border border-warn/30 bg-warn/10 px-2 py-1.5 text-[12px] leading-snug text-warn">
                      {gap.reason}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* json preview */}
          <div className="mt-3 overflow-hidden rounded-lg border border-line bg-canvas">
            <div className="border-b border-line px-3 py-1.5 text-[12px] uppercase tracking-wide text-ink-4">Preview · {bundle?.schema ?? 'vontra.run-export/1'}</div>
            <pre className="scroll-thin max-h-[28vh] overflow-auto px-3 py-2 text-[12px] leading-relaxed text-ink-2">{building ? 'Building…' : (result ? result.json.slice(0, 6000) + (result.json.length > 6000 ? '\n… (truncated in preview; full bundle is saved/copied)' : '') : 'Nothing to export.')}</pre>
          </div>
        </div>

        {/* footer */}
        <div className="flex items-center gap-2 border-t border-line px-4 py-2.5">
          {saved && <span className="flex items-center gap-1.5 text-[13px] text-positive"><Icon name="check" size={13} />Saved</span>}
          <div className="flex-1" />
          <button onClick={onCopy} disabled={!result} className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-40">
            <Icon name="copy" size={13} />{copied ? 'Copied' : 'Copy JSON'}
          </button>
          <button onClick={onSave} disabled={!result} className="flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-[13px] text-accent-ink transition-colors hover:bg-accent-strong disabled:opacity-40">
            <Icon name="archive" size={13} />Save…
          </button>
        </div>
      </div>
    </div>
  )
}

function Badge({ label, value }: { label: string; value: number }) {
  return (
    <span className="rounded-md border border-line bg-card px-2 py-1">
      <span className="font-medium tabular-nums text-ink">{value}</span>{' '}
      <span>{label}</span>
    </span>
  )
}

function Stat({ label, value, tone }: { label: string; value: number; tone?: 'warn' | 'danger' }) {
  return (
    <span className="rounded-md border border-line bg-card-2 px-2 py-0.5">
      <span className={clsx('font-medium tabular-nums', tone === 'danger' ? 'text-danger' : tone === 'warn' ? 'text-warn' : 'text-ink')}>{value}</span>{' '}
      <span className="text-ink-4">{label}</span>
    </span>
  )
}
