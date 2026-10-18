import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { Icon, type IconName } from '../lib/icons'
import { Button, Menu, MenuItem } from './ui'
import { allowedProviders, projectDefaultModel } from '../lib/policy'
import { modelDisplayName, providerModelGroups } from '../lib/provider-accounts'
import type { KanbanPhaseModel, Project, ProjectKanbanSettings, Provider, Room, Ticket, TicketExecutionMode, TicketStatus, TicketWorkOptions } from '../lib/types'
import { resolveTicketDependencies } from '../../shared/ticket-dependencies'

const EXEC_MODES: { id: TicketExecutionMode; label: string; icon: IconName; desc: string }[] = [
  { id: 'solo', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Solo") }, icon: 'cpu', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "One agent on the configured model") } },
  { id: 'frontier', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Frontier") }, icon: 'globe', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "One agent on the best cloud model") } },
  { id: 'room', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Room") }, icon: 'layers', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "A multi-agent room (loops + review)") } },
  { id: 'remote', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Remote") }, icon: 'monitor', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "A thread on the project's remote Orchestrator") } },
  { id: 'cli', label: 'CLI', icon: 'terminal', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "One agent on an external CLI engine (Codex/Claude CLI)") } },
]

// Best frontier (cloud) model from the allowed providers — prefers a top-tier
// family, else the first cloud model offered.
const FRONTIER_RE = /(opus|gpt-5|gpt-4\.1|gpt-4o|o3|o4|gemini.*pro|claude.*(4|3\.[57])|sonnet)/i
function frontierModelChoice(providers: Provider[], project: Project | null): KanbanPhaseModel | null {
  const cloud = providers.filter((p) => p.kind !== 'local')
  for (const re of [FRONTIER_RE, /.*/]) {
    for (const p of cloud) {
      for (const g of providerModelGroups(p)) {
        const m = g.models.find((model) => re.test(model))
        if (m) return { providerId: p.id, providerAccountId: g.accountId, model: m }
      }
    }
  }
  return null
}

// External CLI engine for 'cli'-mode tickets (VST-018): the project's provider when it is
// CLI-backed, else the first connected CLI-transport provider. Null = none connected.
const CLI_TRANSPORTS = new Set(['codex-cli', 'claude-cli', 'cursor-cli', 'gemini-cli', 'pi-cli', 'opencode', 'kilo'])
function cliEngineChoice(providers: Provider[], project: Project | null): KanbanPhaseModel | null {
  const cliProviders = providers.filter((p) => CLI_TRANSPORTS.has(p.transport ?? ''))
  const preferred = project?.defaultProviderId ? cliProviders.find((p) => p.id === project.defaultProviderId) : null
  for (const p of [preferred, ...cliProviders].filter((x): x is Provider => !!x)) {
    for (const g of providerModelGroups(p)) {
      if (g.models.length) return { providerId: p.id, providerAccountId: g.accountId, model: g.models[0] }
    }
  }
  return null
}

function fmtElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

const COLUMNS: { key: TicketStatus; label: string }[] = [
  { key: 'todo', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Todo") } },
  { key: 'ready', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Ready") } },
  { key: 'in_progress', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "In Progress") } },
  { key: 'needs_clarification', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Needs Clarification") } },
  { key: 'review', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Review") } },
  { key: 'completed', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Completed") } },
]

const TYPE_STYLE: Record<string, string> = {
  epic: 'text-accent', subticket: 'text-ink-3', task: 'text-ink-3', bug: 'text-danger',
}
const PRIORITY_STYLE: Record<string, string> = {
  urgent: 'text-danger', high: 'text-danger', medium: 'text-warn', low: 'text-ink-3',
}
const RISK_STYLE: Record<string, string> = {
  high: 'border-danger/35 bg-danger/10 text-danger',
  medium: 'border-warn/35 bg-warn/10 text-warn',
  low: 'border-line bg-ink-5 text-ink-3',
}

function executionMeta(t: Ticket) {
  const required = t.requiredEvidence ?? []
  const evidence = t.evidence ?? []
  const caps = t.allowedCapabilities ?? []
  return {
    required,
    evidence,
    caps,
    risk: t.risk ?? 'low',
    evidenceLabel: required.length ? `${evidence.length}/${required.length}` : evidence.length ? `${evidence.length}` : '',
    capsLabel: caps.length ? caps.slice(0, 3).join(', ') : '',
    moreCaps: Math.max(0, caps.length - 3),
  }
}

function modelChoiceValue(providerId: string, accountId: string | null | undefined, model: string) {
  return JSON.stringify([providerId, accountId ?? null, model])
}

type KanbanPhase = 'draft' | 'ticketReview' | 'build' | 'review' | 'finalGate'
type RuntimeTicketPhase = NonNullable<TicketWorkOptions['phase']>

const PHASES: Array<{ id: KanbanPhase; label: string; desc: string }> = [
  { id: 'draft', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Ticket draft") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Generate cards") } },
  { id: 'ticketReview', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Ticket review") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Check scope") } },
  { id: 'build', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Build") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Implement card") } },
  { id: 'review', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Work review") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Review changes") } },
  { id: 'finalGate', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Final gate") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Ready to close") } },
]

function parseModelChoice(value: string): KanbanPhaseModel | null {
  try {
    const [providerId, providerAccountId, model] = JSON.parse(value)
    if (typeof providerId === 'string' && (typeof providerAccountId === 'string' || providerAccountId === null) && typeof model === 'string') {
      return { providerId, providerAccountId, model }
    }
  } catch {
    // Ignore malformed select values.
  }
  return null
}

function selectionValue(selection: KanbanPhaseModel | null | undefined) {
  return selection?.providerId && selection.model ? modelChoiceValue(selection.providerId, selection.providerAccountId ?? null, selection.model) : ''
}

function firstModelChoice(providers: Provider[], project: Project | null | undefined) {
  const def = projectDefaultModel(project ?? null, providers)
  const byDefault = def.providerId ? providers.find((provider) => provider.id === def.providerId) : null
  const provider = byDefault ?? providers.find((candidate) => providerModelGroups(candidate).some((group) => group.models.length)) ?? providers[0]
  const groups = provider ? providerModelGroups(provider) : []
  const defaultGroup = groups.find((group) => group.accountId === (def.providerAccountId ?? null)) ?? groups[0]
  const model = defaultGroup?.models.includes(def.model ?? '') ? def.model! : defaultGroup?.models[0] ?? provider?.models[0] ?? ''
  return provider && model ? modelChoiceValue(provider.id, defaultGroup?.accountId ?? null, model) : ''
}

function selectionLabel(selection: KanbanPhaseModel | null | undefined, providers: Provider[]) {
  if (!selection?.providerId || !selection.model) return 'No model'
  const provider = providers.find((p) => p.id === selection.providerId)
  const groups = provider ? providerModelGroups(provider) : []
  const group = groups.find((candidate) => candidate.accountId === (selection.providerAccountId ?? null))
  const account = group?.label ? ` · ${group.label}` : ''
  return `${provider ? modelDisplayName(provider, selection.model) : selection.model}${account}`
}

function modelForPhase(kanban: ProjectKanbanSettings | undefined, phase: KanbanPhase, fallback: KanbanPhaseModel): KanbanPhaseModel {
  if (kanban?.modelMode === 'phased') {
    const phaseFallback = phase === 'ticketReview' ? kanban.draft : phase === 'finalGate' ? kanban.review : undefined
    return kanban[phase] ?? phaseFallback ?? kanban.single ?? fallback
  }
  return kanban?.single ?? fallback
}

function workOptionsFor(selection: KanbanPhaseModel, phase?: TicketWorkOptions['phase']): TicketWorkOptions | undefined {
  if (!selection.providerId || !selection.model) return undefined
  return { providerId: selection.providerId, providerAccountId: selection.providerAccountId ?? null, model: selection.model, phase }
}

function KanbanModelSelect({
  label,
  desc,
  providers,
  value,
  onChange,
}: {
  label: string
  desc: string
  providers: Provider[]
  value: string
  onChange: (value: string) => void
}) {
  const tx = useAuthoredText()
  return (
    <label className="block rounded-lg border border-line bg-card-2 px-2.5 py-2">
      <span className="flex items-center justify-between gap-2">
        <span className="text-[13px] font-medium text-ink">{label}</span>
        <span className="text-[12px] text-ink-4">{desc}</span>
      </span>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="no-drag mt-2 h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent">
        {providers.length === 0 && <option value="">{tx("No configured models")}</option>}
        {value && !providers.some((provider) => providerModelGroups(provider).some((group) => group.models.some((model) => modelChoiceValue(provider.id, group.accountId, model) === value))) && (
          <option value={value}>{tx("Missing configured model")}</option>
        )}
        {providers.flatMap((provider) => providerModelGroups(provider).map((group) => (
          <optgroup key={group.id} label={group.detail ? `${group.label} · ${group.detail}` : group.label}>
            {group.models.map((model) => <option key={`${group.id}:${model}`} value={modelChoiceValue(provider.id, group.accountId, model)}>{modelDisplayName(provider, model)}</option>)}
          </optgroup>
        )))}
      </select>
    </label>
  )
}

function Card({
  t,
  rooms,
  allTickets,
  onDragStart,
  onRun,
  onSetMode,
}: {
  t: Ticket
  rooms: Room[]
  allTickets: Ticket[]
  onDragStart: (id: string) => void
  onRun: (id: string, phase: RuntimeTicketPhase, roomId?: string | null) => void
  onSetMode: (id: string, mode: TicketExecutionMode) => void
}) {
  const tx = useAuthoredText()
  const deleteTicket = useStore((s) => s.deleteTicket)
  const openThread = useStore((s) => s.openThread)
  const [now, setNow] = useState(() => Date.now())
  const running = t.runState === 'running'
  const activeRun = t.runState === 'queued' || running
  useEffect(() => {
    if (!running) return
    const i = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(i)
  }, [running])
  const elapsed = running && t.startedAt ? fmtElapsed(now - t.startedAt) : null
  const workPhase: RuntimeTicketPhase = t.status === 'review' ? 'review' : 'build'
  const dependency = resolveTicketDependencies(t, allTickets)
  const canWork = (t.status === 'ready' || t.status === 'todo' || t.status === 'review') && !activeRun && dependency.state === 'ready'
  const meta = executionMeta(t)
  const assignedRoom = t.roomId ? rooms.find((room) => room.id === t.roomId) ?? null : null

  return (
    <div
      draggable={!activeRun}
      onDragStart={(event) => { if (activeRun) event.preventDefault(); else onDragStart(t.id) }}
      onClick={() => { if (t.threadId) openThread(t.threadId) }}
      className={clsx('group mb-2 rounded-xl border border-line bg-card p-3 shadow-card transition-colors hover:border-line-strong', t.threadId ? 'cursor-pointer' : activeRun ? 'cursor-default' : 'cursor-grab active:cursor-grabbing')}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="text-[13px] font-medium leading-snug text-ink">{t.title}</div>
        <button onClick={(e) => { e.stopPropagation(); deleteTicket(t.id) }} className="shrink-0 text-ink-3 opacity-0 transition-opacity hover:text-danger group-hover:opacity-100" aria-label={tx("Archive ticket")}><Icon name="archive" size={13} /></button>
      </div>
      {t.body && <p className="mt-1 line-clamp-3 text-[13px] leading-snug text-ink-3">{t.body}</p>}
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[12px]">
        <span className={clsx('font-semibold capitalize', TYPE_STYLE[t.type])}>{t.type}</span>
        <span className={clsx('capitalize', PRIORITY_STYLE[t.priority])}>{t.priority}</span>
        {running && <span className="flex items-center gap-1 rounded bg-warn/15 px-1.5 py-0.5 font-mono text-warn"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warn" />{tx("Running")}{elapsed ? ` · ${elapsed}` : ''}</span>}
        {!running && t.status === 'ready' && <span className="rounded bg-ink-5 px-1.5 py-0.5 text-ink-3">{tx("Queued")}</span>}
        {!running && t.status === 'review' && <span className="flex items-center gap-1 rounded bg-positive/12 px-1.5 py-0.5 text-positive"><Icon name="check" size={11} />{tx("Review")}</span>}
        {dependency.state === 'waiting' && <span title={tx("Waiting on {value0}", { value0: dependency.waitingOn.join(', ') })} className="rounded bg-warn/12 px-1.5 py-0.5 text-warn">{tx("Waiting on ")}{dependency.waitingOn.length}</span>}
        {dependency.state === 'blocked' && <span title={dependency.blockers.join(' ')} className="rounded bg-danger/10 px-1.5 py-0.5 text-danger">{tx("Dependency blocked")}</span>}
        {assignedRoom && <span className="flex max-w-full items-center gap-1 rounded bg-accent-soft px-1.5 py-0.5 text-accent"><Icon name="layers" size={10} /><span className="truncate">{assignedRoom.name}</span></span>}
        <div className="flex-1" />
        {canWork && workPhase === 'build' && (() => {
          const mode = t.executionMode ?? 'solo'
          const cur = EXEC_MODES.find((m) => m.id === mode) ?? EXEC_MODES[0]
          return (
            <>
              <span onClick={(e) => e.stopPropagation()}>
              <Menu width={216} align="end" trigger={(open) => (
                <button title={tx("How to run this ticket")} className={clsx('flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-medium transition-colors', open ? 'border-line-strong text-ink' : 'border-line text-ink-2 hover:border-line-strong hover:text-ink')}>
                  <Icon name={cur.icon} size={11} />{cur.label}<Icon name="chevronDown" size={10} className="text-ink-3" />
                </button>
              )}>
                {(close) => (
                  <div>
                    <div className="px-2.5 pb-1 pt-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-3">{tx("Run as")}</div>
                    {EXEC_MODES.map((m) => (
                      <button key={m.id} onClick={(e) => { e.stopPropagation(); onSetMode(t.id, m.id); close() }} className={clsx('flex w-full items-start gap-2 rounded-lg px-2.5 py-1.5 text-left transition-colors hover:bg-ink-5', m.id === mode && 'bg-ink-5')}>
                        <Icon name={m.icon} size={13} className="mt-0.5 text-ink-3" />
                        <span className="flex-1">
                          <span className="flex items-center gap-1.5 text-[13px] font-medium text-ink">{m.label}{m.id === mode && <Icon name="check" size={12} className="text-accent" />}</span>
                          <span className="mt-0.5 block text-[12px] leading-snug text-ink-3">{m.desc}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </Menu>
              </span>
              {mode === 'room' ? (
                <span onClick={(e) => e.stopPropagation()}>
                  <Menu width={266} align="end" trigger={(open) => (
                    <button title={tx("Send this ticket to a project room")} className={clsx('flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-medium transition-colors', open ? 'border-line-strong text-ink' : 'border-line text-ink-2 hover:border-line-strong hover:text-ink')}>
                      <Icon name="layers" size={11} />{tx("Send")}<Icon name="chevronDown" size={10} className="text-ink-3" />
                    </button>
                  )}>
                    {(close) => (
                      <div>
                        <div className="px-2.5 pb-1 pt-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-3">{tx("Send ticket to")}</div>
                        {rooms.length > 0 && (
                          <div className="mb-1 max-h-[190px] overflow-y-auto pr-0.5 scroll-thin">
                            {rooms.map((room) => {
                              const roomTickets = allTickets.filter((candidate) => candidate.roomId === room.id)
                              const active = roomTickets.filter((candidate) => candidate.status !== 'completed').length
                              return (
                                <button
                                  key={room.id}
                                  onClick={() => { onRun(t.id, workPhase, room.id); close() }}
                                  className={clsx('flex w-full items-start gap-2 rounded-lg px-2.5 py-1.5 text-left transition-colors hover:bg-ink-5', room.id === t.roomId && 'bg-ink-5')}
                                >
                                  <Icon name="layers" size={13} className="mt-0.5 text-ink-3" />
                                  <span className="min-w-0 flex-1">
                                    <span className="flex min-w-0 items-center gap-1.5 text-[13px] font-medium text-ink">
                                      <span className="truncate">{room.name}</span>
                                      {room.id === t.roomId && <Icon name="check" size={12} className="shrink-0 text-accent" />}
                                    </span>
                                    <span className="mt-0.5 block text-[12px] capitalize leading-snug text-ink-3">{room.state.replace(/_/g, ' ')} · {active}/{roomTickets.length} {tx("open tickets")}</span>
                                  </span>
                                </button>
                              )
                            })}
                          </div>
                        )}
                        {rooms.length === 0 && <div className="mx-2.5 mb-1 rounded-lg border border-dashed border-line px-2 py-3 text-center text-[12px] text-ink-3">{tx("No rooms in this project yet.")}</div>}
                        <button
                          onClick={() => { onRun(t.id, workPhase, null); close() }}
                          className="flex w-full items-center gap-2 rounded-lg border-t border-line px-2.5 py-2 text-left text-[13px] font-medium text-ink transition-colors hover:bg-ink-5"
                        >
                          <Icon name="plus" size={13} className="text-ink-3" />
                          {tx("New room for this ticket")}</button>
                      </div>
                    )}
                  </Menu>
                </span>
              ) : (
                <button onClick={(e) => { e.stopPropagation(); onRun(t.id, workPhase) }} title={tx("Build this ticket")} className="flex items-center gap-1 rounded-md border border-line px-1.5 py-0.5 font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink">
                  <Icon name="play" size={11} />{tx("Build")}</button>
              )}
            </>
          )
        })()}
        {canWork && workPhase === 'review' && (
          <button onClick={(e) => { e.stopPropagation(); onRun(t.id, workPhase) }} title={tx("Run adversarial review")} className="flex items-center gap-1 rounded-md border border-line px-1.5 py-0.5 font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink">
            <Icon name="play" size={11} />{tx("Review")}</button>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
        <span className={clsx('rounded-md border px-1.5 py-0.5 font-medium capitalize', RISK_STYLE[meta.risk])}>{tx("Risk ")}{meta.risk}</span>
        {meta.capsLabel && <span className="rounded-md border border-line bg-card-2 px-1.5 py-0.5 text-ink-3">{tx("Caps ")}{meta.capsLabel}{meta.moreCaps ? ` +${meta.moreCaps}` : ''}</span>}
        {meta.evidenceLabel && <span className={clsx('rounded-md border px-1.5 py-0.5', meta.evidence.length >= meta.required.length ? 'border-positive/30 bg-positive/8 text-positive' : 'border-line bg-card-2 text-ink-3')}>{tx("Evidence ")}{meta.evidenceLabel}</span>}
      </div>
      {t.tags.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {t.tags.slice(0, 3).map((tag) => <span key={tag} className="rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-3">{tag}</span>)}
          {t.tags.length > 3 && <span className="text-[11px] text-ink-3">+{t.tags.length - 3}</span>}
        </div>
      )}
    </div>
  )
}

export function Board({ projectId }: { projectId: string }) {
  const tx = useAuthoredText()
  const tickets = useStore((s) => s.tickets).filter((t) => t.projectId === projectId)
  const rooms = useStore((s) => s.rooms).filter((room) => room.projectId === projectId)
  const project = useStore((s) => s.projects.find((p) => p.id === projectId)) ?? null
  const settings = useStore((s) => s.settings)
  const workTicket = useStore((s) => s.workTicket)
  const workTicketRemote = useStore((s) => s.workTicketRemote)
  const startRoomFromTicket = useStore((s) => s.startRoomFromTicket)
  const updateTicket = useStore((s) => s.updateTicket)
  const createTicket = useStore((s) => s.createTicket)
  const moveTicket = useStore((s) => s.moveTicket)
  const generateTickets = useStore((s) => s.generateTickets)
  const updateProject = useStore((s) => s.updateProject)

  const [idea, setIdea] = useState('')
  const [type, setType] = useState<Ticket['type']>('task')
  const [priority, setPriority] = useState<Ticket['priority']>('medium')
  const [effort, setEffort] = useState<Ticket['effort']>('medium')
  const [dragId, setDragId] = useState<string | null>(null)
  const [overCol, setOverCol] = useState<TicketStatus | null>(null)
  const [generating, setGenerating] = useState(false)
  const [kanbanOpen, setKanbanOpen] = useState(false)
  const modelProviders = useMemo(() => settings ? allowedProviders(settings.providers, project) : [], [settings, project, tx])
  const fallbackModel = useMemo(() => parseModelChoice(firstModelChoice(modelProviders, project)) ?? {}, [modelProviders, project, tx])
  const kanban: ProjectKanbanSettings = project?.kanban ?? { modelMode: 'single' as const }
  const modelMode = kanban.modelMode ?? 'single'
  const singleModel = kanban.single ?? fallbackModel
  const draftModel = modelForPhase(kanban, 'draft', fallbackModel)
  const reviewModel = modelForPhase(kanban, 'review', fallbackModel)
  const buildModel = modelForPhase(kanban, 'build', fallbackModel)

  const saveKanban = (patch: Partial<ProjectKanbanSettings>) => {
    if (!project) return
    void updateProject(projectId, { kanban: { ...kanban, ...patch } })
  }
  const setKanbanMode = (nextMode: ProjectKanbanSettings['modelMode']) => {
    if (nextMode === 'single') {
      saveKanban({ modelMode: 'single', single: singleModel })
      return
    }
    saveKanban({
      modelMode: 'phased',
      single: singleModel,
      draft: kanban.draft ?? singleModel,
      ticketReview: kanban.ticketReview ?? kanban.draft ?? singleModel,
      build: kanban.build ?? singleModel,
      review: kanban.review ?? singleModel,
      finalGate: kanban.finalGate ?? kanban.review ?? singleModel,
    })
  }
  const setKanbanModel = (key: 'single' | KanbanPhase, value: string) => {
    const selection = parseModelChoice(value)
    if (!selection) return
    saveKanban({ [key]: selection })
  }

  const byCol = useMemo(() => {
    const m: Record<string, Ticket[]> = {}
    for (const c of COLUMNS) m[c.key] = []
    for (const t of [...tickets].sort((a, b) => a.order - b.order)) (m[t.status] ?? m.todo).push(t)
    return m
  }, [tickets, tx])

  const addTicket = () => {
    const title = idea.trim()
    if (!title) return
    createTicket({ projectId, title: title.slice(0, 90), body: title.length > 90 ? title : '', type, priority, effort, status: 'todo' })
    setIdea('')
  }
  const generate = async () => {
    const text = idea.trim()
    if (!text || generating) return
    setGenerating(true)
    try { await generateTickets(projectId, text, workOptionsFor(draftModel)); setIdea('') } finally { setGenerating(false) }
  }
  // Build routes by the ticket's chosen execution mode (#55): room → multi-agent
  // room; frontier → solo on the best cloud model; solo → the configured build
  // model. Review is always a solo adversarial pass.
  const runTicket = (id: string, phase: RuntimeTicketPhase, roomId?: string | null) => {
    const ticket = tickets.find((t) => t.id === id)
    const mode = ticket?.executionMode ?? 'solo'
    if (phase === 'build' && mode === 'room') { void startRoomFromTicket(id, roomId === undefined ? (ticket?.roomId ?? null) : roomId); return }
    // Remote (VST-018): the build runs as a thread on the project's Orchestrator host;
    // the ticket tracks the remote thread's lifecycle. Review stays a local solo pass.
    if (phase === 'build' && mode === 'remote') { void workTicketRemote(id); return }
    if (phase === 'build' && mode === 'cli') {
      const cliProvider = cliEngineChoice(modelProviders, project)
      if (!cliProvider) {
        useStore.setState({ notice: { kind: 'error', message: tx("No external CLI engine is connected (Codex/Claude CLI). Add one in Settings → Models or pick another run mode.") } })
        return
      }
      void workTicket(id, workOptionsFor(cliProvider, phase))
      return
    }
    const model = phase === 'review'
      ? reviewModel
      : mode === 'frontier'
        ? (frontierModelChoice(modelProviders, project) ?? buildModel)
        : buildModel
    void workTicket(id, workOptionsFor(model, phase))
  }
  const setTicketMode = (id: string, mode: TicketExecutionMode) => { void updateTicket(id, { executionMode: mode }) }

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 px-5 pt-3">
        <div className="flex min-w-0 items-center gap-2">
          <Button variant={kanbanOpen ? 'default' : 'ghost'} size="sm" icon="settings" onClick={() => setKanbanOpen((open) => !open)}>
            {tx("Kanban settings")}</Button>
          <div className="hidden min-w-0 text-[13px] text-ink-3 md:block">
            {modelMode === 'single' ? tx("One model · {value0}", { value0: selectionLabel(singleModel, modelProviders) }) : tx("Phased models · Draft, ticket review, build, work review, final gate")}
          </div>
        </div>
        <Button variant="default" size="sm" icon="spark" onClick={generate} disabled={!idea.trim() || generating}>
          {generating ? tx("Generating…") : tx("Generate Tickets")}
        </Button>
      </div>
      {kanbanOpen && (
        <div className="mx-5 mt-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-[13px] font-medium text-ink">{tx("Kanban workflow models")}</div>
              <div className="mt-0.5 text-[13px] leading-snug text-ink-3">{tx("Pick one model for the whole board, or split draft, review, build, and gate phases.")}</div>
            </div>
            <div className="inline-flex rounded-lg border border-line bg-card-2 p-0.5">
              <button onClick={() => setKanbanMode('single')} className={clsx('rounded-[7px] px-2.5 py-1.5 text-[13px] font-medium transition-colors', modelMode === 'single' ? 'bg-card text-ink shadow-card' : 'text-ink-3 hover:text-ink')}>{tx("One model")}</button>
              <button onClick={() => setKanbanMode('phased')} className={clsx('rounded-[7px] px-2.5 py-1.5 text-[13px] font-medium transition-colors', modelMode === 'phased' ? 'bg-card text-ink shadow-card' : 'text-ink-3 hover:text-ink')}>{tx("By phase")}</button>
            </div>
          </div>
          {modelMode === 'single' ? (
            <div className="mt-3 max-w-[520px]">
              <KanbanModelSelect
                label={tx("Board model")}
                desc={tx("Used for every board phase.")}
                providers={modelProviders}
                value={selectionValue(singleModel)}
                onChange={(value) => setKanbanModel('single', value)}
              />
            </div>
          ) : (
            <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
              {PHASES.map((phase) => (
                <KanbanModelSelect
                  key={phase.id}
                  label={phase.label}
                  desc={phase.desc}
                  providers={modelProviders}
                  value={selectionValue(modelForPhase(kanban, phase.id, fallbackModel))}
                  onChange={(value) => setKanbanModel(phase.id, value)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      <div className="scroll-thin min-h-0 flex-1 overflow-x-auto px-5 py-4">
        <div className="flex h-full gap-3">
          {COLUMNS.map((col) => (
            <div
              key={col.key}
              onDragOver={(e) => { e.preventDefault(); setOverCol(col.key) }}
              onDragLeave={() => setOverCol((c) => (c === col.key ? null : c))}
              onDrop={() => {
                const dragged = dragId ? tickets.find((ticket) => ticket.id === dragId) : null
                if (dragged && dragged.runState !== 'queued' && dragged.runState !== 'running') moveTicket(dragged.id, col.key)
                setDragId(null); setOverCol(null)
              }}
              className={clsx('flex w-[270px] shrink-0 flex-col rounded-xl border p-2 transition-colors', overCol === col.key ? 'border-accent/50 bg-accent-soft/40' : 'border-line bg-card-2/40')}
            >
              <div className="flex items-center justify-between px-1.5 py-1.5">
                <span className="text-[13px] font-medium text-ink">{col.label}</span>
                <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] tabular-nums text-ink-3">{byCol[col.key].length}</span>
              </div>
              <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-0.5">
                {byCol[col.key].map((t) => <Card key={t.id} t={t} rooms={rooms} allTickets={tickets} onDragStart={setDragId} onRun={runTicket} onSetMode={setTicketMode} />)}
                {byCol[col.key].length === 0 && <div className="px-2 py-6 text-center text-[12px] text-ink-4">—</div>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* draft composer */}
      <div className="shrink-0 border-t border-line px-5 py-3">
        <div className="rounded-xl border border-line bg-card px-3 py-2 shadow-card focus-within:border-line-strong">
          <div className="flex items-center gap-2">
            <input
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') addTicket() }}
              placeholder={tx("Draft a ticket idea…  (Enter adds it; ✦ Generate Tickets decomposes into an epic)")}
              className="flex-1 bg-transparent text-[14px] text-ink outline-none placeholder:text-ink-3"
            />
            <button onClick={addTicket} disabled={!idea.trim()} className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-ink transition-colors hover:bg-accent-strong disabled:bg-ink-4 disabled:text-bg" aria-label={tx("Add ticket")}><Icon name="send" size={15} strokeWidth={2} /></button>
          </div>
          <div className="mt-1.5 flex items-center gap-3 text-[12px] text-ink-3">
            <Sel label={tx("Type")} value={type} onChange={(v) => setType(v as Ticket['type'])} options={['task', 'bug', 'epic']} />
            <Sel label={tx("Priority")} value={priority} onChange={(v) => setPriority(v as Ticket['priority'])} options={['low', 'medium', 'high', 'urgent']} />
            <Sel label={tx("Effort")} value={effort} onChange={(v) => setEffort(v as Ticket['effort'])} options={['small', 'medium', 'large']} />
          </div>
        </div>
      </div>
    </div>
  )
}

function Sel({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[] }) {
  const tx = useAuthoredText()
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-ink-4">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="no-drag cursor-pointer rounded-md border border-line bg-card-2 px-1.5 py-0.5 text-[12px] capitalize text-ink-2 outline-none">
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  )
}
