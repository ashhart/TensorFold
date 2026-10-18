import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../../lib/store'
import { resolveExecutionTarget } from '../../lib/execution-target'
import { Icon, type IconName } from '../../lib/icons'
import type { CliDefinition, CliSession } from '../../lib/types'
import { terminalSessionKey } from '../../lib/terminal-sessions'
import { LAYOUT_PRESETS, layoutGridStyle, loadLayoutPreset, paneGridArea, presetSlots, saveLayoutPreset, type LayoutPreset } from '../../lib/terminal-layout'
import { Menu } from '../ui'
import { TerminalPane } from './TerminalPane'

// A launcher the user can pick to spawn a session. Built-ins come from the runtime (filtered to the
// CLIs actually installed); a "Custom command…" entry covers anything else (gemini, pi, …).
interface Launcher {
  cliId: string
  label: string
  icon: IconName
  command?: string
}

function toLauncher(c: CliDefinition): Launcher {
  return { cliId: c.id, label: c.name, icon: (c.icon as IconName) || 'terminal', command: c.command || undefined }
}

const EMPTY_SESSIONS: CliSession[] = []

// The Terminal tab. Sessions are listed in the left sidebar (renameable, click to open); this view
// shows the active one, with a compact top row to start a new terminal or split the screen.
export function TerminalsTab({ projectId }: { projectId: string }) {
  const tx = useAuthoredText()
  const sessions = useStore((s) => s.terminals[projectId] ?? EMPTY_SESSIONS)
  const activeId = useStore((s) => s.activeTerminalId[projectId] ?? null)
  const project = useStore((s) => s.projects.find((candidate) => candidate.id === projectId) ?? null)
  const settings = useStore((s) => s.settings)
  // The same resolution the spawn path uses (project pin > localOnly > global mode), so the
  // remote hints here always match where the terminal will really run. Computed OUTSIDE the
  // zustand selector: resolveExecutionTarget returns a fresh object, and a selector that
  // returns a new reference every call makes useSyncExternalStore loop ("getSnapshot should
  // be cached" → Maximum update depth → blank tab).
  const decision = useMemo(() => resolveExecutionTarget(settings, project), [settings, project])
  const host = useMemo(
    () => (decision.hostId ? settings?.remoteOrchestrators?.find((candidate) => candidate.id === decision.hostId) ?? null : null),
    [decision.hostId, settings],
  )
  const startTerminal = useStore((s) => s.startTerminal)
  const killTerminal = useStore((s) => s.killTerminal)
  const loadTerminals = useStore((s) => s.loadTerminals)
  const listTerminalClis = useStore((s) => s.listTerminalClis)
  const [installed, setInstalled] = useState<Launcher[]>([])
  const [customOpen, setCustomOpen] = useState(false)
  const [customCmd, setCustomCmd] = useState('')
  const remoteTerminal = !!decision.hostId
  // Workspace pane layout (VST-039): a per-project preset (single/columns/rows/main+side/
  // four), persisted to localStorage and restored on open.
  const [layout, setLayout] = useState<LayoutPreset>('single')
  useEffect(() => { setLayout(loadLayoutPreset(projectId)) }, [projectId])
  const pickLayout = (preset: LayoutPreset) => { setLayout(preset); saveLayoutPreset(projectId, preset) }

  useEffect(() => { void loadTerminals(projectId) }, [projectId, loadTerminals])
  // Only offer CLIs really installed where this project's terminal will run.
  useEffect(() => {
    let live = true
    setInstalled([])
    listTerminalClis(projectId).then((clis) => { if (live) setInstalled(clis.map(toLauncher)) }).catch(() => { if (live) setInstalled([]) })
    return () => { live = false }
  }, [projectId, listTerminalClis])

  const launch = (l: Launcher) => { void startTerminal(projectId, l.cliId, l.command ? { command: l.command, label: l.label } : undefined) }
  // Electron has no window.prompt, so a custom command opens an inline input instead.
  const launchCustom = () => {
    if (remoteTerminal) return
    setCustomCmd('')
    setCustomOpen(true)
  }
  const submitCustom = () => {
    const cmd = customCmd.trim()
    if (!cmd) return
    launch({ cliId: `custom:${cmd}`, label: cmd, icon: 'terminal', command: cmd })
    setCustomOpen(false); setCustomCmd('')
  }
  const cancelCustom = () => { setCustomOpen(false); setCustomCmd('') }
  const close = (session: CliSession) => { void killTerminal(projectId, terminalSessionKey(session)) }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Small Split control: pops a CLI picker (like the project + menu) to add another terminal as
          a pane in this window. The empty state handles the very first terminal. */}
      {sessions.length > 0 && (
        <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
          <Menu width={200} trigger={() => (
            <span className="inline-flex items-center gap-1 rounded-md border border-line bg-card px-2 py-1 text-[13px] text-ink-2 transition-colors hover:bg-surface-hover" title={tx("Split — add another CLI to this terminal window")}>
              <Icon name="grid" size={13} /> {tx("Split")}</span>
          )}>
            {(closeMenu) => (
              <>
                {installed.map((l) => (
                  <button key={l.cliId} onClick={() => { closeMenu(); launch(l) }}
                    className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">
                    <Icon name={l.icon} size={14} className="text-ink-3" /> {l.label}
                  </button>
                ))}
                {!remoteTerminal && (
                  <>
                    <div className="my-1 border-t border-line" />
                    <button onClick={() => { closeMenu(); launchCustom() }}
                      className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink">
                      <Icon name="terminal" size={14} /> {tx("Custom command…")}</button>
                  </>
                )}
              </>
            )}
          </Menu>
          {/* Layout presets (VST-039): choose how the panes tile; persisted per project. */}
          {sessions.length > 1 && (
            <Menu width={180} trigger={() => {
              const current = LAYOUT_PRESETS.find((p) => p.value === layout) ?? LAYOUT_PRESETS[0]
              return (
                <span className="inline-flex items-center gap-1 rounded-md border border-line bg-card px-2 py-1 text-[13px] text-ink-2 transition-colors hover:bg-surface-hover" title={tx("Pane layout")}>
                  <Icon name={current.icon as IconName} size={13} /> {current.label}
                </span>
              )
            }}>
              {(closeMenu) => (
                <>
                  {LAYOUT_PRESETS.map((p) => (
                    <button key={p.value} onClick={() => { closeMenu(); pickLayout(p.value) }}
                      className={`flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors hover:bg-ink-5 ${p.value === layout ? 'text-ink' : 'text-ink-2'}`}>
                      <Icon name={p.icon as IconName} size={14} className={p.value === layout ? 'text-accent' : 'text-ink-3'} />
                      <span className="flex-1">{p.label}</span>
                      <span className="text-[12px] text-ink-4">{p.slots}</span>
                    </button>
                  ))}
                </>
              )}
            </Menu>
          )}
          {remoteTerminal && (
            <span className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-accent/25 bg-accent/10 px-2 py-1 text-[13px] text-accent" title={host?.serverUrl ?? undefined}>
              <Icon name="globe" size={12} />
              {host?.name ?? tx("Orchestrator")}
            </span>
          )}
        </div>
      )}
      {customOpen && !remoteTerminal && (
        <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
          <Icon name="terminal" size={13} className="shrink-0 text-ink-3" />
          <input
            autoFocus
            value={customCmd}
            onChange={(e) => setCustomCmd(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') submitCustom(); else if (e.key === 'Escape') cancelCustom() }}
            onBlur={() => { if (!customCmd.trim()) cancelCustom() }}
            placeholder={tx("command to run (e.g. pi, gemini)")}
            className="min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-4"
          />
          <button onMouseDown={(e) => e.preventDefault()} onClick={submitCustom} disabled={!customCmd.trim()}
            className="inline-flex items-center rounded-md border border-line bg-card px-2 py-0.5 text-[13px] text-ink-2 transition-colors hover:bg-surface-hover disabled:opacity-40">{tx("Run")}</button>
          <button onMouseDown={(e) => e.preventDefault()} onClick={cancelCustom}
            className="rounded-md px-1.5 py-0.5 text-[13px] text-ink-3 transition-colors hover:text-ink">{tx("Cancel")}</button>
        </div>
      )}
      <div className="min-h-0 flex-1 p-2">
        {sessions.length === 0
          ? <EmptyState installed={installed} onPick={launch} onCustom={launchCustom} allowCustom={!remoteTerminal} remoteName={remoteTerminal ? host?.name ?? 'Orchestrator' : null} />
          : <SplitGrid sessions={sessions} onClose={close} activeId={activeId} layout={layout} />}
      </div>
    </div>
  )
}

// First-run: no sessions yet. Offer the installed CLIs as one-click launchers so the user does not
// have to open the menu to get going.
function EmptyState({ installed, onPick, onCustom, allowCustom, remoteName }: { installed: Launcher[]; onPick: (l: Launcher) => void; onCustom: () => void; allowCustom: boolean; remoteName?: string | null }) {
  const tx = useAuthoredText()
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
      <div className="text-[13px] text-ink-4">
        {remoteName ? tx("No terminals on {value0} yet.", { value0: remoteName }) : tx("No terminals in this project yet.")}
      </div>
      <div className="flex max-w-[420px] flex-wrap items-center justify-center gap-1.5">
        {installed.map((l) => (
          <button key={l.cliId} onClick={() => onPick(l)}
            className="inline-flex items-center gap-1.5 rounded-md border border-line bg-card px-2.5 py-1 text-[13px] text-ink-2 transition-colors hover:bg-surface-hover">
            <Icon name={l.icon} size={13} /> {l.label}
          </button>
        ))}
        {allowCustom && (
          <button onClick={onCustom} className="inline-flex items-center gap-1 rounded-md border border-dashed border-line px-2 py-1 text-[13px] text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink-2">
            <Icon name="plus" size={13} /> {tx("Custom…")}</button>
        )}
      </div>
    </div>
  )
}

// Split mode: sessions tiled per the chosen layout preset (VST-039). Each pane is its own
// live xterm with a slim header. A single session ignores the preset (always full-bleed);
// with more sessions than the preset's slots, extras flow into the last cell's stack — no
// session is ever hidden.
function SplitGrid({ sessions, onClose, activeId, layout }: { sessions: CliSession[]; onClose: (session: CliSession) => void; activeId: string | null; layout: LayoutPreset }) {
  const tx = useAuthoredText()
  const effective = sessions.length <= 1 ? 'single' : layout
  const slots = presetSlots(effective)
  // Sessions beyond the last slot share that slot (stacked), so nothing disappears when there
  // are more terminals than tiles.
  const cells: CliSession[][] = Array.from({ length: Math.min(slots, sessions.length) }, () => [])
  sessions.forEach((s, i) => { cells[Math.min(i, cells.length - 1)].push(s) })

  return (
    <div className="grid h-full min-h-0 gap-2" style={layoutGridStyle(effective)}>
      {cells.map((cell, cellIndex) => (
        <div key={cellIndex} className="flex min-h-0 flex-col gap-2" style={paneGridArea(effective, cellIndex) ? { gridArea: paneGridArea(effective, cellIndex) } : undefined}>
          {cell.map((s) => (
            <div key={terminalSessionKey(s)} className={`flex min-h-0 flex-1 flex-col overflow-hidden rounded-md border ${terminalSessionKey(s) === activeId ? 'border-accent/50' : 'border-line'}`}>
              <div className="flex items-center justify-between border-b border-line bg-card px-2 py-1">
                <span className="flex min-w-0 items-center gap-1.5">
                  <span className="truncate text-[12px] text-ink-3">{s.title}</span>
                  {s.hostLocation === 'orchestrator' && <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[11px] font-medium text-accent" title={s.orchestratorHostId ? tx("Hosted by {value0}", { value0: s.orchestratorHostId }) : tx("Origin unavailable; controls are disabled")}>{s.orchestratorHostId ? tx("Remote") : tx("Remote · read-only")}</span>}
                  {s.managedBy === 'external' && <span className="rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-3">{tx("External")}</span>}
                </span>
                <button onClick={() => onClose(s)} className="flex h-4 w-4 items-center justify-center text-ink-4 transition-colors hover:text-ink" title={tx("Close terminal")} aria-label={tx("Close terminal")}>
                  <Icon name="x" size={11} />
                </button>
              </div>
              <div className="min-h-0 flex-1">
                <TerminalPane sessionKey={terminalSessionKey(s)} />
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
