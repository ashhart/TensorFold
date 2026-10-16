import { useEffect, useMemo, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { Icon, type IconName } from '../lib/icons'
import type { Incident, IncidentSeverity, IncidentStatus, IncidentEventKind, IncidentMitigation } from '../lib/types'

// build.md Phase 6 — production incident mode. A narrow, auditable lane that is its
// own entity (never a Room), so it can run a read-only first-pass scan, keep an
// append-only timeline, gate mitigations behind approval, and generate a postmortem
// without ever mutating the workspace by surprise. Self-contained overlay opened by
// a `vontra:incidents` CustomEvent from the command palette.
const SEVERITIES: IncidentSeverity[] = ['sev1', 'sev2', 'sev3', 'sev4']
const STATUSES: IncidentStatus[] = ['open', 'investigating', 'mitigating', 'resolved', 'closed']
const EVENT_KINDS: IncidentEventKind[] = ['symptom', 'log', 'metric', 'traffic', 'command', 'note']

const sevTint = (s: IncidentSeverity) => (s === 'sev1' || s === 'sev2' ? 'bg-danger/10 text-danger' : s === 'sev3' ? 'bg-warn/10 text-warn' : 'bg-ink-5 text-ink-3')
const statusTint = (s: IncidentStatus) => (s === 'resolved' ? 'text-positive' : s === 'closed' ? 'text-ink-4' : s === 'open' ? 'text-warn' : 'text-ink-2')
const eventTint = (k: IncidentEventKind): string => k === 'symptom' ? 'text-danger' : k === 'mitigation' ? 'text-ink' : k === 'scan' || k === 'command' ? 'text-ink-2' : 'text-ink-3'
const eventIcon = (k: IncidentEventKind): IconName => k === 'symptom' ? 'warning' : k === 'mitigation' ? 'shield' : k === 'scan' ? 'spark' : k === 'command' ? 'terminal' : 'clock'
const mitTint = (s: IncidentMitigation['status']) => s === 'approved' ? 'text-positive' : s === 'rejected' ? 'text-danger' : s === 'applied' ? 'text-ink' : 'text-warn'

function rel(ms: number): string {
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

export function IncidentCenter() {
  const [open, setOpen] = useState(false)
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [selId, setSelId] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    const onOpen = () => { setOpen(true); api.getIncidents().then(setIncidents).catch(() => {}) }
    window.addEventListener('vontra:incidents', onOpen)
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    // Live updates: main pushes incident:update when a scan finishes or a guarded mitigation mission
    // completes, so the open view reflects outcomes without a manual reopen.
    const off = api.onIncident((inc) => setIncidents((list) => list.some((i) => i.id === inc.id) ? list.map((i) => (i.id === inc.id ? inc : i)) : [inc, ...list]))
    return () => { window.removeEventListener('vontra:incidents', onOpen); window.removeEventListener('keydown', onKey); off() }
  }, [])

  const upsert = (inc: Incident) => setIncidents((list) => (list.some((i) => i.id === inc.id) ? list.map((i) => (i.id === inc.id ? inc : i)) : [inc, ...list]))
  const selected = incidents.find((i) => i.id === selId) ?? null

  if (!open) return null
  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[7vh] fade-in" onClick={() => setOpen(false)}>
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      <div className="relative flex h-[84vh] w-[min(960px,calc(100vw-40px))] overflow-hidden rounded-2xl border border-line bg-card" onClick={(e) => e.stopPropagation()} style={{ boxShadow: 'var(--shadow-pop)' }}>
        {/* list rail */}
        <div className="flex w-[260px] shrink-0 flex-col border-r border-line bg-canvas">
          <div className="flex items-center gap-2 border-b border-line px-3.5 py-3">
            <Icon name="warning" size={15} className="text-warn" />
            <div className="flex-1 text-[13px] font-medium text-ink">Incidents</div>
            <button onClick={() => { setCreating(true); setSelId(null) }} className="flex items-center gap-1 rounded-md border border-line px-1.5 py-1 text-[12px] text-ink-2 hover:border-line-strong hover:text-ink"><Icon name="plus" size={11} />New</button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto scroll-thin p-1.5">
            {incidents.length === 0 && <div className="px-2.5 py-6 text-center text-[13px] text-ink-3">No incidents.</div>}
            {incidents.map((i) => (
              <button key={i.id} onClick={() => { setSelId(i.id); setCreating(false) }} className={clsx('mb-1 flex w-full flex-col gap-1 rounded-lg border px-2.5 py-2 text-left transition-colors', selId === i.id ? 'border-ink-3 bg-ink-5' : 'border-transparent hover:bg-ink-6')}>
                <div className="flex items-center gap-1.5">
                  <span className={clsx('rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase', sevTint(i.severity))}>{i.severity}</span>
                  <span className={clsx('text-[11px] capitalize', statusTint(i.status))}>{i.status}</span>
                </div>
                <span className="truncate text-[13px] text-ink-2">{i.title}</span>
                <span className="text-[11px] text-ink-4">{rel(i.updatedAt)}</span>
              </button>
            ))}
          </div>
        </div>

        {/* detail */}
        <div className="min-w-0 flex-1 overflow-y-auto scroll-thin">
          {creating ? (
            <NewIncidentForm onCancel={() => setCreating(false)} onCreated={(inc) => { upsert(inc); setSelId(inc.id); setCreating(false) }} />
          ) : selected ? (
            <IncidentDetail key={selected.id} incident={selected} onChange={upsert} onDelete={(id) => { setIncidents((l) => l.filter((i) => i.id !== id)); setSelId(null) }} />
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-ink-3">
              <Icon name="shield" size={26} className="text-ink-4" />
              <div className="text-[13px]">Select an incident, or open a new one.</div>
              <div className="max-w-[340px] text-center text-[13px] text-ink-4">An incident runs a strictly read-only first-pass scan; mitigations are proposed and require approval before anything changes.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function NewIncidentForm({ onCancel, onCreated }: { onCancel: () => void; onCreated: (i: Incident) => void }) {
  const [title, setTitle] = useState('')
  const [severity, setSeverity] = useState<IncidentSeverity>('sev3')
  const [summary, setSummary] = useState('')
  const [cwd, setCwd] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const create = async () => {
    if (!title.trim() || busy) return
    setBusy(true); setErr(null)
    try { onCreated(await api.createIncident({ title: title.trim(), severity, summary: summary.trim() || undefined, cwd: cwd.trim() || undefined })) }
    catch (e: any) { setErr(String(e?.message ?? e)) }
    finally { setBusy(false) }
  }
  return (
    <div className="p-5">
      <div className="mb-3 text-[14px] font-medium text-ink">Open an incident</div>
      <div className="space-y-3">
        <Field label="Title">
          <input autoFocus value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Checkout 500s spike" className="w-full rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent" />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Severity">
            <select value={severity} onChange={(e) => setSeverity(e.target.value as IncidentSeverity)} className="w-full rounded-md border border-line bg-card-2 px-2 py-1.5 text-[13px] text-ink outline-none">
              {SEVERITIES.map((s) => <option key={s} value={s}>{s.toUpperCase()}</option>)}
            </select>
          </Field>
          <Field label="Workspace (optional)">
            <input value={cwd} onChange={(e) => setCwd(e.target.value)} placeholder="defaults to project cwd" className="w-full rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent" />
          </Field>
        </div>
        <Field label="What's happening?">
          <textarea value={summary} onChange={(e) => setSummary(e.target.value)} rows={3} placeholder="Symptoms, when it started, what changed…" className="scroll-thin w-full resize-none rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent" />
        </Field>
      </div>
      {err && <div className="mt-3 rounded-md border border-danger/30 bg-danger/5 px-2.5 py-1.5 text-[13px] text-danger">{err}</div>}
      <div className="mt-4 flex justify-end gap-2">
        <button onClick={onCancel} className="rounded-md px-3 py-1.5 text-[13px] text-ink-3 hover:text-ink">Cancel</button>
        <button onClick={create} disabled={!title.trim() || busy} className="rounded-md bg-accent px-3.5 py-1.5 text-[13px] text-accent-ink hover:bg-accent-strong disabled:opacity-40">{busy ? 'Opening…' : 'Open incident'}</button>
      </div>
    </div>
  )
}

function IncidentDetail({ incident, onChange, onDelete }: { incident: Incident; onChange: (i: Incident) => void; onDelete: (id: string) => void }) {
  const [scanning, setScanning] = useState(false)
  const [pmBusy, setPmBusy] = useState(false)
  const [evKind, setEvKind] = useState<IncidentEventKind>('symptom')
  const [evText, setEvText] = useState('')
  const [mitText, setMitText] = useState('')
  const [mitRationale, setMitRationale] = useState('')
  const [err, setErr] = useState<string | null>(null)

  const scanning2 = scanning || incident.scan.status === 'running'
  const sortedTimeline = useMemo(() => [...incident.timeline].sort((a, b) => b.at - a.at), [incident.timeline])

  // Every mutation routes through guard() so an IPC rejection (e.g. the incident was deleted in
  // another view, or a scan provider error) is shown to the user, never a silent unhandled rejection.
  const guard = async (p: Promise<Incident>) => { setErr(null); try { onChange(await p) } catch (e: any) { setErr(String(e?.message ?? e)) } }

  const runScan = async () => { if (scanning2) return; setScanning(true); try { await guard(api.scanIncident(incident.id)) } finally { setScanning(false) } }
  const genPm = async () => { if (pmBusy) return; setPmBusy(true); try { await guard(api.generatePostmortem(incident.id)) } finally { setPmBusy(false) } }
  const setSeverity = (severity: IncidentSeverity) => guard(api.updateIncident(incident.id, { severity }))
  const setStatus = (status: IncidentStatus) => guard(api.updateIncident(incident.id, { status }))
  const addEvent = () => { if (!evText.trim()) return; const text = evText.trim(); setEvText(''); guard(api.addIncidentEvent(incident.id, evKind, text)) }
  const propose = () => { if (!mitText.trim()) return; const t = mitText.trim(); const r = mitRationale.trim() || undefined; setMitText(''); setMitRationale(''); guard(api.proposeMitigation(incident.id, t, r)) }
  const decide = (mid: string, decision: IncidentMitigation['status']) => guard(api.decideMitigation(incident.id, mid, decision))
  const execute = (mid: string) => guard(api.executeMitigation(incident.id, mid))

  return (
    <div className="flex h-full flex-col">
      {/* header */}
      <div className="flex items-start gap-3 border-b border-line px-5 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[15px] font-medium text-ink">{incident.title}</div>
          {incident.summary && <div className="mt-0.5 line-clamp-2 text-[13px] text-ink-3">{incident.summary}</div>}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <select value={incident.severity} onChange={(e) => setSeverity(e.target.value as IncidentSeverity)} className={clsx('rounded-md border-0 px-2 py-0.5 text-[12px] font-semibold uppercase outline-none', sevTint(incident.severity))}>
              {SEVERITIES.map((s) => <option key={s} value={s}>{s.toUpperCase()}</option>)}
            </select>
            <select value={incident.status} onChange={(e) => setStatus(e.target.value as IncidentStatus)} className="rounded-md border border-line bg-card-2 px-2 py-0.5 text-[12px] capitalize text-ink-2 outline-none">
              {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <span className="text-[12px] text-ink-4">opened {rel(incident.createdAt)}</span>
          </div>
        </div>
        <button onClick={() => onDelete(incident.id)} className="shrink-0 rounded-md p-1.5 text-ink-4 hover:text-danger" title="Delete incident"><Icon name="trash" size={14} /></button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto scroll-thin p-5">
        {/* actions */}
        <div className="flex flex-wrap gap-2">
          <button onClick={runScan} disabled={scanning2} className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-50">
            <Icon name={scanning2 ? 'spark' : 'play'} size={13} className={scanning2 ? 'animate-pulse' : ''} />{scanning2 ? 'Scanning (read-only)…' : incident.scan.status === 'done' ? 'Re-scan' : 'Read-only scan'}
          </button>
          <button onClick={genPm} disabled={pmBusy} className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-50">
            <Icon name="fileText" size={13} className={pmBusy ? 'animate-pulse' : ''} />{pmBusy ? 'Writing…' : incident.postmortem ? 'Regenerate postmortem' : 'Generate postmortem'}
          </button>
        </div>
        <p className="mt-1.5 flex items-center gap-1.5 text-[12px] text-ink-4"><Icon name="shield" size={11} className="text-positive" />The scan runs with write &amp; command execution denied — it can read, but never change, the workspace.</p>
        {err && <div className="mt-2.5 flex items-start gap-1.5 rounded-md border border-danger/30 bg-danger/5 px-3 py-1.5 text-[13px] text-danger"><Icon name="warning" size={13} className="mt-0.5 shrink-0" /><span className="min-w-0 flex-1">{err}</span></div>}

        {/* scan findings */}
        {incident.scan.status !== 'idle' && (
          <Section title="Scan findings" icon="spark">
            {incident.scan.status === 'error'
              ? <div className="rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-[13px] text-danger">{incident.scan.error || 'Scan failed.'}</div>
              : incident.scan.findings
                ? <pre className="scroll-thin max-h-[34vh] overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-canvas px-3 py-2 text-[13px] leading-relaxed text-ink-2">{incident.scan.findings}</pre>
                : <div className="text-[13px] text-ink-3">Scanning…</div>}
            {incident.scan.ranAt && <div className="mt-1 text-[12px] text-ink-4">ran {rel(incident.scan.ranAt)}</div>}
          </Section>
        )}

        {/* mitigations */}
        <Section title="Mitigations" icon="shield">
          <div className="space-y-1.5">
            {incident.mitigations.length === 0 && <div className="text-[13px] text-ink-3">None proposed. Mitigations are recorded here and require explicit approval.</div>}
            {incident.mitigations.map((m) => (
              <div key={m.id} className="rounded-lg border border-line bg-card-2 px-3 py-2">
                <div className="flex items-start gap-2">
                  <span className="min-w-0 flex-1 text-[13px] text-ink">{m.text}</span>
                  <span className={clsx('shrink-0 text-[12px] font-medium capitalize', mitTint(m.status))}>{m.status}</span>
                </div>
                {m.rationale && <div className="mt-0.5 text-[12px] text-ink-3">{m.rationale}</div>}
                {m.status === 'proposed' && (
                  <div className="mt-2 flex gap-2">
                    <button onClick={() => decide(m.id, 'approved')} className="flex items-center gap-1 rounded-md border border-positive/40 px-2 py-0.5 text-[12px] text-positive hover:bg-positive/10"><Icon name="check" size={11} />Approve</button>
                    <button onClick={() => decide(m.id, 'rejected')} className="flex items-center gap-1 rounded-md border border-line px-2 py-0.5 text-[12px] text-ink-3 hover:text-danger"><Icon name="stop" size={11} />Reject</button>
                  </div>
                )}
                {m.status === 'approved' && !m.missionId && (
                  <div className="mt-2 flex gap-2">
                    <button onClick={() => execute(m.id)} className="flex items-center gap-1 rounded-md border border-line px-2 py-0.5 text-[12px] text-ink-2 hover:border-line-strong hover:text-ink"><Icon name="play" size={11} />Execute (guarded)</button>
                    <button onClick={() => decide(m.id, 'applied')} className="rounded-md border border-line px-2 py-0.5 text-[12px] text-ink-3 hover:text-ink">Mark applied</button>
                  </div>
                )}
                {m.missionId && m.status !== 'applied' && (
                  <div className="mt-2 flex items-center gap-1.5 text-[12px] text-ink-3"><Icon name="spark" size={11} className="animate-pulse" />Executing as a guarded mission — approve each mutating step in the Command Centre.</div>
                )}
              </div>
            ))}
          </div>
          <div className="mt-2 space-y-1.5 rounded-lg border border-dashed border-line p-2.5">
            <input value={mitText} onChange={(e) => setMitText(e.target.value)} placeholder="Propose a mitigation…" className="w-full rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent" />
            <div className="flex gap-2">
              <input value={mitRationale} onChange={(e) => setMitRationale(e.target.value)} placeholder="rationale (optional)" className="min-w-0 flex-1 rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink-2 outline-none focus:border-accent" />
              <button onClick={propose} disabled={!mitText.trim()} className="shrink-0 rounded-md bg-accent px-3 py-1.5 text-[13px] text-accent-ink hover:bg-accent-strong disabled:opacity-40">Propose</button>
            </div>
          </div>
        </Section>

        {/* timeline */}
        <Section title="Timeline" icon="clock">
          <div className="mb-2 flex gap-2">
            <select value={evKind} onChange={(e) => setEvKind(e.target.value as IncidentEventKind)} className="rounded-md border border-line bg-card-2 px-2 py-1.5 text-[13px] capitalize text-ink-2 outline-none">
              {EVENT_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <input value={evText} onChange={(e) => setEvText(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') addEvent() }} placeholder="Add to the timeline…" className="min-w-0 flex-1 rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent" />
            <button onClick={addEvent} disabled={!evText.trim()} className="shrink-0 rounded-md border border-line px-2.5 py-1.5 text-[13px] text-ink-2 hover:text-ink disabled:opacity-40">Add</button>
          </div>
          <div className="space-y-2">
            {sortedTimeline.map((ev) => (
              <div key={ev.id} className="flex gap-2.5">
                <Icon name={eventIcon(ev.kind)} size={13} className={clsx('mt-0.5 shrink-0', eventTint(ev.kind))} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className={clsx('text-[12px] font-medium uppercase tracking-wide', eventTint(ev.kind))}>{ev.kind}</span>
                    <span className="text-[11px] text-ink-4">{rel(ev.at)}{ev.source !== 'user' ? ` · ${ev.source}` : ''}</span>
                  </div>
                  <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-ink-2">{ev.text}</div>
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* postmortem */}
        {incident.postmortem && (
          <Section title="Postmortem" icon="fileText">
            <pre className="scroll-thin max-h-[40vh] overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-canvas px-3 py-2 text-[13px] leading-relaxed text-ink-2">{incident.postmortem}</pre>
            {incident.postmortemAt && <div className="mt-1 text-[12px] text-ink-4">generated {rel(incident.postmortemAt)}</div>}
          </Section>
        )}
      </div>
    </div>
  )
}

function Section({ title, icon, children }: { title: string; icon: IconName; children: ReactNode }) {
  return (
    <div className="mt-5">
      <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium uppercase tracking-wide text-ink-3"><Icon name={icon} size={12} />{title}</div>
      {children}
    </div>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[12px] font-medium text-ink-3">{label}</span>
      {children}
    </label>
  )
}
