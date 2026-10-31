import { useEffect, useMemo, useRef, useState } from 'react'
import { api, shortName, type BenchRunRow, type ConversationRow, type EngineInfo, type Model, type Settings } from './api'
import { ROUTES, type Route } from './App'
import { rank } from './fuzzy'
import { linkTo } from './route'

type Command = { id: string; label: string; hint?: string; icon: 'go' | 'play' | 'stop' | 'theme' | 'chat' | 'model' | 'run'; run: () => void | Promise<unknown>; entry?: boolean }
const ICON: Record<Command['icon'], string> = {
  go: 'M5 12h14M13 6l6 6-6 6', play: 'M6 4l14 8-14 8z', stop: 'M6 6h12v12H6z', theme: 'M12 3a9 9 0 1 0 0 18V3z', chat: 'M4 5h16v11H9l-5 4z',
  model: 'M4 7.5 12 3l8 4.5v9L12 21l-4 4.5zM4 7.5 12 12l8-4.5M12 12v9', run: 'M4 20h16M7 20v-6M12 20V8M17 20v-9',
}
const KINDS: Record<string, string> = { quick: 'quick score', profile: 'usage profile', prefill: 'prefill sweep', load: 'load sweep', quality: 'quality', tools: 'tool calls' }
const when = (t: number) => new Date(t * 1000).toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })

/** The command palette: screens, models to start or stop, the look, and once you type, every model, Bench run and
 *  conversation, each opening its own link. Opens with ⌘K or Ctrl+K; the matching forgives typos (`fuzzy.ts`). */
export default function Palette({ open, onClose, stale, engine, settings, onTheme, toast, refreshEngine, refreshSettings }:
  { open: boolean; onClose: () => void; stale: boolean; engine: EngineInfo | null; settings: Settings | null; onTheme: (t: Settings['theme']) => void; toast: (t: string, k?: 'ok' | 'bad') => void; refreshEngine: () => Promise<unknown>; refreshSettings: () => Promise<unknown> }) {
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(0)
  const [models, setModels] = useState<Model[]>([])
  const [runs, setRuns] = useState<BenchRunRow[]>([])
  const [convs, setConvs] = useState<ConversationRow[]>([])
  const input = useRef<HTMLInputElement>(null)
  // a modal: focus goes in when it opens and back where it was when it closes
  const before = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (!open) { before.current?.focus?.(); before.current = null; return }
    before.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    setQ(''); setSel(0)
    setTimeout(() => input.current?.focus(), 0)
    api.catalog().then((c) => setModels(c.models)).catch(() => {})
    api.benchRuns().then((r) => setRuns(r.runs.slice(0, 40))).catch(() => {})
    api.conversations().then((r) => setConvs(r.conversations.slice(0, 60))).catch(() => {})
  }, [open])

  const commands = useMemo<Command[]>(() => {
    const go = (r: Route, label: string, k: string): Command => ({ id: `go-${r}`, label, hint: k, icon: 'go', run: () => { location.hash = `#${r}` } })
    const out: Command[] = ROUTES.map((r, i) => go(r, `Go to ${r[0].toUpperCase()}${r.slice(1)}`, String(i + 1)))   // the sidebar's order and keys
    // no engine commands while TensorFold Web does not answer: they would act on figures that may be out of date
    for (const m of stale ? [] : models.filter((m) => m.installed && m.runs_here)) {
      const running = !!engine?.model && shortName(m.id).startsWith(engine.model)
      if (!running) out.push({ id: `start-${m.id}`, label: `Start ${m.name}`, hint: m.verdict?.label, icon: 'play', run: async () => {
        try { await api.start({ model: m.id, budget_fraction: m.verdict?.level === 'tight' ? m.verdict.budget_fraction : settings?.budget_fraction ?? null }); toast(`Starting ${m.name}`); await Promise.all([refreshEngine(), refreshSettings()]); location.hash = '#monitor' } catch (e) { toast(String(e), 'bad') }
      } })
    }
    if (!stale && engine?.owned && (engine.status === 'running' || engine.status === 'starting')) out.push({ id: 'stop', label: `Stop ${engine.model_name ?? engine.model ?? 'the AI'}`, icon: 'stop', run: async () => { try { await api.stop(); toast('Engine stopped'); await refreshEngine() } catch (e) { toast(String(e), 'bad') } } })
    out.push({ id: 'checks', label: "Check this machine (setup and diagnostic)", icon: 'go', run: () => { location.hash = '#settings'; setTimeout(() => document.getElementById('diag-title')?.scrollIntoView({ block: 'center' }), 300) } })
    out.push({ id: 'new-chat', label: 'New conversation', icon: 'chat', run: () => { location.hash = '#chat'; dispatchEvent(new CustomEvent('tfs:new-conversation')) } })
    for (const t of ['light', 'dark', 'system'] as const) out.push({ id: `theme-${t}`, label: t === 'system' ? 'Follow the system look' : `Use the ${t} look`, icon: 'theme', run: () => onTheme(t) })
    // entries, offered once something is typed: each opens its own link
    const open = (h: string) => () => { location.hash = h }
    for (const m of models) out.push({ id: `model-${m.id}`, label: `Model: ${m.name}`, hint: m.installed ? 'on disk' : m.runs_here ? m.verdict?.label : 'not here', icon: 'model', entry: true, run: open(linkTo('models', m.id)) })
    for (const r of runs) out.push({ id: `run-${r.id}`, label: `Run: ${KINDS[r.mode] ?? r.mode}, ${r.model ?? 'no model'}`, hint: when(r.created), icon: 'run', entry: true, run: open(linkTo('bench', r.id)) })
    for (const c of convs) out.push({ id: `conv-${c.id}`, label: `Conversation: ${c.title}`, hint: when(c.updated), icon: 'chat', entry: true, run: open(linkTo('chat', c.id)) })
    return out
  }, [models, runs, convs, stale, engine, settings, onTheme, toast, refreshEngine, refreshSettings])

  const shown = useMemo(() => (q.trim() ? rank(q, commands, (c) => c.label) : commands.filter((c) => !c.entry)).slice(0, 60), [commands, q])
  useEffect(() => { setSel(0) }, [q])
  const list = useRef<HTMLUListElement>(null)
  useEffect(() => { list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' }) }, [sel, shown])

  if (!open) return null
  const run = (c: Command) => { onClose(); c.run() }
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(shown.length - 1, s + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)) }
    else if (e.key === 'Enter' && shown[sel]) { e.preventDefault(); run(shown[sel]) }
    else if (e.key === 'Escape') onClose()
    else if (e.key === 'Tab') { e.preventDefault(); input.current?.focus() }   // the arrows move; Tab stays inside
  }
  return (
    <div className="veil" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Commands" onKeyDown={onKey}>
        <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} role="combobox" aria-expanded={shown.length > 0} aria-controls="palette-list" aria-activedescendant={shown[sel] ? `pal-${sel}` : undefined} placeholder="A screen, a model, a run, a conversation…" aria-label="Command" />
        {shown.length ? (
          <ul role="listbox" id="palette-list" ref={list} aria-label="Matches">
            {shown.map((c, i) => (
              <li key={c.id} id={`pal-${i}`} role="option" aria-selected={i === sel} onMouseEnter={() => setSel(i)} onClick={() => run(c)}>
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON[c.icon]} /></svg>{c.label}{c.hint && <span className="k">{c.hint}</span>}
              </li>
            ))}
          </ul>
        ) : <div className="empty">Nothing matches. Commands, models, Bench runs and conversations are all searched.</div>}
      </div>
    </div>
  )
}
