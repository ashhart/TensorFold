import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { Icon, type IconName } from '../lib/icons'
import type { ImportableSession } from '../lib/types'
import { threadPlace } from '../lib/thread-row'

// build.md Phase 5 — global ⌘K command palette. Self-contained: owns its hotkey +
// open state, reads everything from the store. Jump to any view/thread/room, create,
// switch model, toggle settings — keyboard-first.
interface Cmd { id: string; label: string; hint?: string; icon: IconName; group: string; run: () => void }

const score = (label: string, hint: string, q: string): number => {
  if (!q) return 0
  const l = label.toLowerCase(); const h = hint.toLowerCase()
  if (l.startsWith(q)) return 0
  if (l.includes(q)) return 1
  if (h.includes(q)) return 2
  return 3
}

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [idx, setIdx] = useState(0)

  const threads = useStore((s) => s.threads)
  const rooms = useStore((s) => s.rooms)
  const projects = useStore((s) => s.projects)
  const settings = useStore((s) => s.settings)
  const view = useStore((s) => s.view)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setOpen((o) => !o) }
    }
    // The sidebar's Search row opens the palette without a keystroke.
    const onOpenEvent = () => setOpen(true)
    window.addEventListener('keydown', onKey)
    window.addEventListener('vontra:open-palette', onOpenEvent)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('vontra:open-palette', onOpenEvent) }
  }, [])

  useEffect(() => { if (open) { setQuery(''); setIdx(0) } }, [open])

  // External harness sessions (~/.codex, ~/.claude) importable as threads (VST-033).
  // Fetched whenever the palette opens; failures just mean no import entries.
  const [importable, setImportable] = useState<ImportableSession[]>([])
  useEffect(() => {
    if (!open) return
    let cancelled = false
    api.listImportableSessions()
      .then((sessions) => { if (!cancelled) setImportable(sessions.slice(0, 8)) })
      .catch(() => { if (!cancelled) setImportable([]) })
    return () => { cancelled = true }
  }, [open])

  const commands = useMemo<Cmd[]>(() => {
    const s = useStore.getState()
    const list: Cmd[] = []
    // The palette is global search/create: every creation verb, always.
    list.push(
      { id: 'new-task', label: 'New task', hint: '⌘N', icon: 'pencil', group: 'Create', run: () => s.newTask() },
      { id: 'new-chat', label: 'New chat', icon: 'message', group: 'Create', run: () => void s.newChat(null) },
      { id: 'new-project', label: 'New project…', icon: 'plus', group: 'Create', run: () => s.openNewProject() },
    )
    list.push(
      ...(view.name === 'thread' ? [{ id: 'export-run', label: 'Export this run…', hint: 'redacted bundle', icon: 'archive', group: 'Create', run: () => window.dispatchEvent(new CustomEvent('vontra:export', { detail: { threadId: view.id } })) }] as Cmd[] : []),
      { id: 'replay-run', label: 'Import / replay a run…', hint: 'from a bundle', icon: 'play', group: 'Create', run: () => window.dispatchEvent(new CustomEvent('vontra:replay')) },
      { id: 'go-home', label: 'Home', icon: 'spark', group: 'Go to', run: () => s.setView({ name: 'home' }) },
      { id: 'go-projects', label: 'Projects', icon: 'folder', group: 'Go to', run: () => s.setView({ name: 'projects' }) },
      { id: 'go-cmd', label: 'Command Centre', icon: 'bell', group: 'Go to', run: () => s.setView({ name: 'commandCentre' }) },
      { id: 'go-assistant', label: 'Assistant', icon: 'mail', group: 'Go to', run: () => s.setView({ name: 'assistant' }) },
      { id: 'go-models', label: 'Models', icon: 'cpu', group: 'Go to', run: () => s.setView({ name: 'models' }) },
      { id: 'go-plugins', label: 'Sources & Connectors', icon: 'plug', group: 'Go to', run: () => s.setView({ name: 'plugins' }) },
      { id: 'go-autos', label: 'Automations', icon: 'clock', group: 'Go to', run: () => s.setView({ name: 'automations' }) },
      { id: 'go-settings', label: 'Settings', icon: 'settings', group: 'Go to', run: () => s.setView({ name: 'settings' }) },
      { id: 'incident-mode', label: 'Incident mode', hint: 'read-only triage', icon: 'warning', group: 'Go to', run: () => window.dispatchEvent(new CustomEvent('vontra:incidents')) },
      { id: 'toggle-approval', label: `Approval mode: ${settings?.approvalMode ?? '—'}`, hint: 'cycle', icon: 'shield', group: 'Toggle', run: () => s.cycleApprovalMode() },
      { id: 'toggle-sidebar', label: 'Toggle sidebar', icon: 'sidebar', group: 'Toggle', run: () => s.toggleSidebar() },
    )
    for (const t of threads) {
      const project = t.projectId ? projects.find((p) => p.id === t.projectId) ?? null : null
      list.push({ id: `t:${t.id}`, label: t.title || 'Untitled', hint: threadPlace(t, project), icon: t.kind === 'task' && t.cwd ? 'file' : 'message', group: 'Threads', run: () => s.openThread(t.id) })
    }
    {
      for (const r of rooms) list.push({ id: `r:${r.id}`, label: r.name, hint: 'Room', icon: 'layers', group: 'Rooms', run: () => s.openRoom(r.id) })
      for (const p of projects) list.push({ id: `p:${p.id}`, label: p.name, hint: 'Project', icon: 'folder', group: 'Projects', run: () => s.openProject(p.id) })
      for (const session of importable) {
        const sourceName = session.source === 'codex' ? 'Codex' : 'Claude Code'
        list.push({
          id: `imp:${session.source}:${session.sessionId}`,
          label: `Import: ${session.title}`,
          hint: `${sourceName} session · ${session.messageCount}${session.truncated ? '+' : ''} messages`,
          icon: 'archive',
          group: 'Import session',
          run: () => {
            void (async () => {
              try {
                const thread = await api.importExternalSession({ sessionId: session.sessionId, source: session.source, title: session.title, cwd: session.cwd, model: session.model })
                const state = useStore.getState()
                state.applyThreadUpdate(thread)
                state.openThread(thread.id)
              } catch (error) {
                useStore.getState().setNotice({ kind: 'error', message: error instanceof Error ? error.message : 'Session import failed.' })
              }
            })()
          },
        })
      }
    }
    for (const p of settings?.providers ?? []) for (const m of p.models) list.push({ id: `m:${p.id}:${m}`, label: m, hint: `${p.name} · set active`, icon: p.kind === 'local' ? 'cpu' : 'globe', group: 'Switch model', run: () => settings && s.saveSettings({ ...settings, activeProviderId: p.id, activeModel: m }) })
    for (const th of ['light', 'dark', 'system'] as const) list.push({ id: `theme:${th}`, label: `Theme: ${th}`, hint: 'appearance', icon: th === 'light' ? 'sun' : th === 'dark' ? 'moon' : 'monitor', group: 'Theme', run: () => settings && s.saveSettings({ ...settings, theme: th }) })
    return list
  }, [threads, rooms, projects, settings, view, importable])

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    return commands
      .filter((c) => !q || c.label.toLowerCase().includes(q) || (c.hint ?? '').toLowerCase().includes(q))
      .sort((a, b) => score(a.label, a.hint ?? '', q) - score(b.label, b.hint ?? '', q) || a.label.length - b.label.length)
      .slice(0, 40)
  }, [commands, query])

  useEffect(() => { setIdx((i) => (i >= results.length ? 0 : i)) }, [results.length])

  if (!open) return null
  const run = (c?: Cmd) => { if (!c) return; c.run(); setOpen(false) }
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setIdx((i) => (i + 1) % Math.max(1, results.length)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setIdx((i) => (i - 1 + results.length) % Math.max(1, results.length)) }
    else if (e.key === 'Enter') { e.preventDefault(); run(results[idx]) }
    else if (e.key === 'Escape') { e.preventDefault(); setOpen(false) }
  }

  let lastGroup = ''
  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[12vh] fade-in" onClick={() => setOpen(false)}>
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      <div className="relative w-[min(560px,calc(100vw-40px))] overflow-hidden rounded-2xl border border-line bg-card shadow-pop" onClick={(e) => e.stopPropagation()} style={{ boxShadow: 'var(--shadow-pop)' }}>
        <div className="flex items-center gap-2 border-b border-line px-3.5 py-2.5">
          <Icon name="search" size={15} className="text-ink-3" />
          <input autoFocus value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKeyDown} placeholder="Search actions, threads, rooms, models…" className="w-full bg-transparent text-[14px] text-ink outline-none placeholder:text-ink-3" />
          <kbd className="rounded border border-line px-1.5 py-0.5 text-[11px] text-ink-4">esc</kbd>
        </div>
        <div className="max-h-[52vh] overflow-y-auto scroll-thin p-1.5">
          {results.length === 0 && <div className="px-2.5 py-6 text-center text-[13px] text-ink-3">No matches.</div>}
          {results.map((c, i) => {
            const header = c.group !== lastGroup ? (lastGroup = c.group) : null
            return (
              <div key={c.id}>
                {header && <div className="px-2.5 pb-1 pt-2 text-[12px] font-medium uppercase tracking-wide text-ink-3">{header}</div>}
                <button
                  onMouseEnter={() => setIdx(i)}
                  onMouseDown={(e) => { e.preventDefault(); run(c) }}
                  className={clsx('flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors', i === idx ? 'bg-ink-5 text-ink' : 'text-ink-2')}
                >
                  <Icon name={c.icon} size={15} className="shrink-0 text-ink-3" />
                  <span className="min-w-0 flex-1 truncate">{c.label}</span>
                  {c.hint && <span className="shrink-0 text-[12px] text-ink-4">{c.hint}</span>}
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
