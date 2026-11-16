import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { basename, relativeTime } from '../../lib/format'
import { Icon, type IconName } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import { Button, Input } from '../ui'
import { allowedProviders, projectDefaultModel } from '../../lib/policy'
import { accountLabel, modelDisplayName, providerAccountChoices, providerModelGroups } from '../../lib/provider-accounts'
import { buildRoomAgentTree, type RoomAgentTreeNode } from '../../lib/room-agent-tree'
import type { AgentPreset, AgentRoleProfileId, FanOutBudget, Provider, Room, RoomSeatLane, Seat, Ticket } from '../../lib/types'
import { FAN_OUT_OPTIONS, FILE_TOOL_VERB, ROLE_PROFILES, Avatar, availableSeatName, isActiveRunState, modelChoiceValue, parseModelChoice, pickProvider, seatModelChoice } from './helpers'
import { ReasoningEffortSelect, effortForModel } from '../ReasoningEffortSelect'
import type { ReasoningEffortSetting } from '../../lib/types'

const AGENT_NODE_STATUS_LABEL: Record<RoomAgentTreeNode['status'], string> = {
  get idle() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Idle") },
  get active() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Active") },
  get done() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Done") },
  get failed() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Failed") },
}

function AgentStatusDot({ status }: { status: RoomAgentTreeNode['status'] }) {
  const tx = useAuthoredText()
  return (
    <span
      className={clsx(
        'h-1.5 w-1.5 shrink-0 rounded-full',
        status === 'failed' ? 'bg-danger' : status === 'done' ? 'bg-positive' : status === 'active' ? 'bg-accent' : 'bg-ink-4',
      )}
      title={AGENT_NODE_STATUS_LABEL[status]}
    />
  )
}

function RoomAgentDelegates({ node }: { node: RoomAgentTreeNode | undefined }) {
  const tx = useAuthoredText()
  if (!node?.children.length) return null
  return (
    <div className="mt-2 border-t border-line pt-1.5">
      <div className="mb-1 flex items-center gap-1.5 text-[12px] font-medium uppercase tracking-wide text-ink-3">
        <Icon name="branch" size={11} />
        <span>{tx("Delegates")}</span>
        <span className="ml-auto rounded bg-ink-5 px-1 text-[11px] text-ink-4">{node.children.length}</span>
      </div>
      <div className="space-y-1">
        {node.children.map((child) => <RoomAgentDelegateRow key={child.id} node={child} />)}
      </div>
    </div>
  )
}

function RoomAgentDelegateRow({ node }: { node: RoomAgentTreeNode }) {
  const tx = useAuthoredText()
  return (
    <div className="min-w-0">
      <div className="flex min-w-0 items-start gap-1.5 rounded-md bg-card-2 px-1.5 py-1">
        <AgentStatusDot status={node.status} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <span title={node.name} className="truncate text-[12px] font-medium text-ink-2">{node.name}</span>
            {node.children.length > 0 && <span className="shrink-0 rounded bg-ink-5 px-1 text-[11px] tabular-nums text-ink-4">{node.children.length}</span>}
            <span className="shrink-0 text-[11px] text-ink-4">{AGENT_NODE_STATUS_LABEL[node.status]}</span>
          </div>
          <div title={node.detail} className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-ink-3">{node.detail}</div>
        </div>
      </div>
      {node.children.length > 0 && (
        <div className="ml-2 mt-1 space-y-1 border-l border-line pl-2">
          {node.children.map((child) => <RoomAgentDelegateRow key={child.id} node={child} />)}
        </div>
      )}
    </div>
  )
}

function roomSeatFocus(room: Room, tickets: Ticket[], seat: Seat, node: RoomAgentTreeNode | undefined): { label: string; detail: string; icon: IconName; tone: 'active' | 'blocked' | 'done' | 'idle' } {
  const heartbeat = room.seatHeartbeats?.[seat.id]
  if (heartbeat?.missedSince) {
    return { get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Stalled") }, detail: `missed heartbeat ${relativeTime(heartbeat.missedSince)}`, icon: 'warning', tone: 'blocked' }
  }

  const pendingHandoff = [...room.events].reverse().find((event) => {
    if (!(event.kind === 'ask' || event.kind === 'assign') || event.target !== seat.name) return false
    return !room.events.some((candidate) =>
      candidate.createdAt > event.createdAt &&
      candidate.seat === seat.name &&
      !['yield', 'notice', 'error'].includes(candidate.kind),
    )
  })
  if (pendingHandoff) {
    return { get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Answering handoff") }, detail: `${pendingHandoff.seat} is waiting`, icon: 'message', tone: 'active' }
  }

  const claimed = tickets.find((t) => t.assignee === seat.name && t.projectId === room.projectId && t.roomId === room.id && (t.status === 'in_progress' || t.status === 'review' || isActiveRunState(t.runState)))
  if (claimed) {
    const running = isActiveRunState(claimed.runState)
    return { label: running ? 'Worker running' : claimed.status === 'review' ? 'Reviewing' : 'Working ticket', detail: claimed.title, icon: running ? 'play' : 'check', tone: 'active' }
  }

  if (node?.children.some((child) => child.status === 'active')) {
    return { get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Delegated") }, detail: `${node.children.filter((child) => child.status === 'active').length} active delegate${node.children.filter((child) => child.status === 'active').length === 1 ? '' : 's'}`, icon: 'branch', tone: 'active' }
  }

  const last = [...room.events].reverse().find((event) => event.seat === seat.name && event.kind !== 'yield')
  if (last) {
    const label = last.kind === 'tool' ? `Used ${last.tool ?? 'tool'}` : last.kind === 'claim' ? 'Claimed work' : last.kind === 'final' ? 'Concluded' : `${last.kind[0]?.toUpperCase() ?? ''}${last.kind.slice(1)}`
    return { label, detail: last.text.replace(/\s+/g, ' ').trim().slice(0, 90), icon: last.kind === 'tool' ? 'grid' : last.kind === 'final' ? 'check' : 'spark', tone: last.kind === 'final' ? 'done' : 'idle' }
  }

  return { get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Available") }, detail: seat.lane ? `${seat.lane} lane` : 'Ready for a handoff', icon: 'clock', tone: 'idle' }
}

function SeatFocusLine({ focus }: { focus: ReturnType<typeof roomSeatFocus> }) {
  const tx = useAuthoredText()
  return (
    <div className={clsx(
      'mt-1 flex min-w-0 items-center gap-1.5 rounded-md px-1.5 py-1 text-[12px]',
      focus.tone === 'blocked' ? 'bg-warn/10 text-warn' : focus.tone === 'active' ? 'bg-accent-soft text-accent' : focus.tone === 'done' ? 'bg-positive/12 text-positive' : 'bg-ink-5 text-ink-3',
    )}>
      <Icon name={focus.icon} size={11} className="shrink-0" />
      <span className="shrink-0 font-medium">{focus.label}</span>
      <span className="min-w-0 truncate opacity-80">{focus.detail}</span>
    </div>
  )
}

export function RoomAgentsSection({
  room,
  providers,
  tickets,
  running,
  adding,
  setAdding,
  setSeats,
}: {
  room: Room
  providers: Provider[]
  tickets: Ticket[]
  running: boolean
  adding: boolean
  setAdding: Dispatch<SetStateAction<boolean>>
  setSeats: (seats: Seat[]) => void
}) {
  const tx = useAuthoredText()
  const defaultEffort = useStore((state) => state.settings?.reasoningEffort)
  const modelGroups = providers.flatMap((provider) => providerModelGroups(provider).map((group) => ({ provider, group })))
  const agentTree = useMemo(() => buildRoomAgentTree(room), [room, tx])
  const agentTreeBySeat = useMemo(() => new Map(agentTree.map((node) => [node.seatName, node])), [agentTree, tx])
  const updateSeatModel = (seat: Seat, value: string) => {
    const picked = parseModelChoice(value)
    if (!picked) return
    const provider = providers.find((candidate) => candidate.id === picked.providerId)
    setSeats(room.seats.map((candidate) => candidate.id === seat.id ? {
      ...candidate, ...picked, reasoningEffort: effortForModel(candidate.reasoningEffort, provider, picked.model),
    } : candidate))
  }
  const removeSeat = (seatId: string) => setSeats(room.seats.filter((seat) => seat.id !== seatId))

  return (
    <div className="border-b border-line px-3 py-3">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <div>
          <div className="text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{tx("Room agents")}</div>
          <div className="mt-0.5 text-[12px] leading-snug text-ink-4">{tx("Pick the model each local or cloud agent uses.")}</div>
        </div>
        <button onClick={() => setAdding((v) => !v)} disabled={running} className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-45" aria-label={tx("Add agent")}><Icon name="plus" size={16} /></button>
      </div>
      <div className="space-y-1.5">
        {room.seats.map((seat) => {
          const claimed = tickets.find((t) => t.assignee === seat.name && t.projectId === room.projectId && t.roomId === room.id && (t.status === 'in_progress' || t.status === 'review'))
          const recentFiles = [...new Set([...room.events].reverse().filter((ev) => ev.seat === seat.name && ev.tool && ev.path && FILE_TOOL_VERB[ev.tool]).map((ev) => ev.path!))].slice(0, 2)
          const seatProvider = providers.find((p) => p.id === seat.providerId)
          const choiceValue = seatModelChoice(seat, providers)
          const treeNode = agentTreeBySeat.get(seat.name)
          const focus = roomSeatFocus(room, tickets, seat, treeNode)
          return (
            <div key={seat.id} role="group" aria-label={seat.name} className="group rounded-lg border border-line bg-card px-2.5 py-2">
              <div className="flex items-start gap-2">
                <Avatar name={seat.name} size={24} />
                <div className="min-w-0 flex-1">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="truncate text-[13px] font-medium text-ink">{seat.name}</span>
                    {seat.roleProfileId && <span className="shrink-0 rounded bg-ink-5 px-1.5 py-0.5 text-[11px] uppercase tracking-wide text-ink-3">{seat.roleProfileId}</span>}
                  </div>
                  <div className="mt-0.5 truncate text-[12px] text-ink-3">{seat.lane ? `${seat.lane} · ` : ''}{seat.role}</div>
                </div>
                <button onClick={() => removeSeat(seat.id)} disabled={running} className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-ink-3 opacity-0 transition-opacity hover:text-danger disabled:opacity-30 group-hover:opacity-100" aria-label={tx("Remove agent")}><Icon name="x" size={14} /></button>
              </div>
              <SeatFocusLine focus={focus} />
              <select
                aria-label={tx("Model")}
                value={choiceValue}
                disabled={running || modelGroups.length === 0}
                onChange={(e) => updateSeatModel(seat, e.target.value)}
                className="no-drag mt-2 h-8 w-full cursor-pointer rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-55"
              >
                {!choiceValue && <option value="">{tx("Choose model")}</option>}
                {modelGroups.map(({ provider, group }) => (
                  <optgroup key={group.id} label={group.detail ? `${group.label} · ${group.detail}` : group.label}>
                    {group.models.map((model) => (
                      <option key={`${group.id}:${model}`} value={modelChoiceValue(provider.id, group.accountId, model)}>
                        {modelDisplayName(provider, model)}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
              <div className="mt-2">
                <ReasoningEffortSelect provider={seatProvider} model={seat.model} value={seat.reasoningEffort}
                  defaultEffort={defaultEffort} disabled={running}
                  onChange={(reasoningEffort) => setSeats(room.seats.map((candidate) =>
                    candidate.id === seat.id ? { ...candidate, reasoningEffort } : candidate))} />
              </div>
              {seatProvider && <div className="mt-1 truncate text-[12px] text-ink-4">{seatProvider.name}</div>}
              {claimed && <div className="mt-1 flex items-center gap-1 truncate text-[12px] text-accent"><Icon name="play" size={9} />{claimed.title}</div>}
              {recentFiles.length > 0 && (
                <div className="mt-1 space-y-0.5 border-t border-line pt-1">
                  {recentFiles.map((path) => <div key={path} className="flex items-center gap-1.5 truncate text-[12px] text-ink-3"><FileGlyph path={path} size={11} /><span className="truncate">{basename(path)}</span></div>)}
                </div>
              )}
              <RoomAgentDelegates node={treeNode} />
            </div>
          )
        })}
        {adding && <AddSeat room={room} running={running} onAdd={(seat) => { setSeats([...room.seats, seat]); setAdding(false) }} onCancel={() => setAdding(false)} />}
        {room.seats.length === 0 && !adding && <div className="rounded-lg border border-dashed border-line px-2.5 py-5 text-center text-[13px] text-ink-3">{tx("No agents yet. Add one or use the preset below.")}</div>}
      </div>
    </div>
  )
}

function AddSeat({ room, running, onAdd, onCancel }: { room: Room; running: boolean; onAdd: (s: Seat) => void; onCancel: () => void }) {
  const tx = useAuthoredText()
  const settings = useStore((s) => s.settings)!
  const project = useStore((s) => s.projects.find((p) => p.id === room.projectId)) ?? null
  const agentPresets = (settings.agentPresets ?? []).filter((preset) => !preset.archivedAt && !preset.paused)
  const pickerProviders = allowedProviders(settings.providers, project)
  const def = projectDefaultModel(project, settings.providers)
  // Name seats after their role (planner, planner-2, …) so the chosen role is
  // visible in the name, and suggest a fitting model per role (#47): builders and
  // operators run local; planners, reviewers and researchers run on a frontier
  // (cloud) model.
  const roleSeatName = (base: string) => {
    const used = new Set(room.seats.map((s) => s.name.toLowerCase()))
    if (!used.has(base.toLowerCase())) return base
    for (let i = 2; i < 99; i++) if (!used.has(`${base}-${i}`)) return `${base}-${i}`
    return base
  }
  const suggestModelFor = (laneFor: RoomSeatLane) =>
    pickProvider(pickerProviders, laneFor === 'development' || laneFor === 'operations' ? 'local' : 'cloud', def)
  const [roleProfileId, setRoleProfileId] = useState<AgentRoleProfileId>('builder')
  const [name, setName] = useState(() => roleSeatName('builder'))
  const [nameTouched, setNameTouched] = useState(false)
  const [role, setRole] = useState('Engineer')
  const [lane, setLane] = useState<RoomSeatLane>('development')
  const [fanOutBudget, setFanOutBudget] = useState<FanOutBudget>('single')
  // Seed the form with the model that fits the default 'builder' role (local).
  const initialPick = suggestModelFor('development')
  const [providerId, setProviderId] = useState(initialPick.provider?.id || def.providerId || pickerProviders[0]?.id || '')
  const provider = pickerProviders.find((p) => p.id === providerId) || pickerProviders[0]
  const [providerAccountId, setProviderAccountId] = useState<string | null>(initialPick.provider ? initialPick.providerAccountId : (project?.defaultProviderAccountId ?? provider?.defaultAccountId ?? null))
  const accountChoices = provider ? providerAccountChoices(provider) : []
  const selectedAccount = accountChoices.find((account) => account.id === (providerAccountId ?? null)) ?? accountChoices[0]
  const modelChoices = selectedAccount?.models?.length ? selectedAccount.models : (provider?.models ?? [])
  const [model, setModel] = useState(initialPick.model || def.model || modelChoices[0] || '')
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffortSetting | undefined>()
  const selectedAccountLabel = provider ? accountLabel(provider, providerAccountId) : null
  const noAllowedProviders = pickerProviders.length === 0
  const currentModelChoice = provider && model ? modelChoiceValue(provider.id, providerAccountId, model) : ''

  useEffect(() => {
    setReasoningEffort((current) => effortForModel(current, provider, model))
  }, [provider, model])

  useEffect(() => {
    if (!provider) return
    const account = providerAccountChoices(provider).find((choice) => choice.id === (providerAccountId ?? null)) ?? providerAccountChoices(provider)[0]
    if ((providerAccountId ?? null) !== (account?.id ?? null)) setProviderAccountId(account?.id ?? null)
    const models = account?.models?.length ? account.models : provider.models
    if (models.length && !models.includes(model)) setModel(models[0])
  }, [provider?.id, providerAccountId, model, pickerProviders.length]) // eslint-disable-line

  const add = () => {
    if (running || !provider || !model) return
    const n = (name.trim() || role.toLowerCase()).replace(/\s+/g, '-')
    onAdd({ id: crypto.randomUUID(), name: n, role: role.trim() || 'Agent', lane, roleProfileId, fanOutBudget, providerId: provider.id, providerAccountId: providerAccountId ?? null, model, reasoningEffort })
  }
  const pickModel = (value: string) => {
    const picked = parseModelChoice(value)
    if (!picked) return
    setProviderId(picked.providerId)
    setProviderAccountId(picked.providerAccountId)
    setModel(picked.model)
  }
  const applyAgentPreset = (presetId: string) => {
    const preset = agentPresets.find((candidate) => candidate.id === presetId)
    if (!preset) return
    setName(availableSeatName(preset.name, room.seats))
    setNameTouched(true)
    setRole(preset.role)
    setLane(preset.lane)
    setProviderId(preset.providerId)
    setProviderAccountId(preset.providerAccountId ?? null)
    setModel(preset.model)
    setReasoningEffort(effortForModel(preset.reasoningEffort, pickerProviders.find((candidate) => candidate.id === preset.providerId), preset.model))
  }

  return (
    <fieldset disabled={running} className="mb-1.5 rounded-lg border border-line bg-card-2 p-2.5">
      {agentPresets.length > 0 && (
        <select
          defaultValue=""
          onChange={(e) => applyAgentPreset(e.target.value)}
          className="no-drag mb-2 h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent"
        >
          <option value="" disabled>{tx("Use saved agent...")}</option>
          {agentPresets.map((preset: AgentPreset) => <option key={preset.id} value={preset.id}>{preset.name} · {preset.role}</option>)}
        </select>
      )}
      <select
        value={roleProfileId}
        onChange={(e) => {
          const next = ROLE_PROFILES.find((profile) => profile.id === e.target.value) ?? ROLE_PROFILES[1]
          setRoleProfileId(next.id)
          setRole(next.role)
          setLane(next.lane)
          // Reflect the role in the name unless the user typed their own.
          if (!nameTouched) setName(roleSeatName(next.name))
          // Suggest a model that fits the role (local for builders, cloud for planners…).
          const pick = suggestModelFor(next.lane)
          if (pick.provider && pick.model) { setProviderId(pick.provider.id); setProviderAccountId(pick.providerAccountId); setModel(pick.model) }
        }}
        className="no-drag mb-2 h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent"
      >
        {ROLE_PROFILES.map((profile) => <option key={profile.id} value={profile.id}>{profile.label} {tx("room role")}</option>)}
      </select>
      <Input value={name} onChange={(e) => { setNameTouched(true); setName(e.target.value) }} placeholder={tx("Name (e.g. architect)")} className="mb-2 h-8" />
      <Input value={role} onChange={(e) => setRole(e.target.value)} placeholder={tx("Role (e.g. Architect)")} className="mb-2 h-8" />
      <div className="mb-2 grid grid-cols-2 gap-2">
        <select value={lane} onChange={(e) => setLane(e.target.value as RoomSeatLane)} className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent">
          <option value="planning">{tx("Planning")}</option>
          <option value="development">{tx("Development")}</option>
          <option value="review">{tx("Review")}</option>
          <option value="research">{tx("Research")}</option>
          <option value="operations">{tx("Operations")}</option>
        </select>
        <select value={fanOutBudget} onChange={(e) => setFanOutBudget(e.target.value as FanOutBudget)} className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent">
          {FAN_OUT_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
        </select>
      </div>
      {noAllowedProviders ? (
        <div className="mb-2 rounded-lg border border-warn/30 bg-warn/10 px-2.5 py-2 text-[13px] leading-snug text-warn">
          {tx("This project is local-only, but no local providers are configured.")}</div>
      ) : (
        <div className="mb-2">
          <select value={currentModelChoice} onChange={(e) => pickModel(e.target.value)} className="no-drag h-8 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent">
            {project?.localOnly && <option value="" disabled>{tx("Local models only")}</option>}
            {pickerProviders.flatMap((p) => providerModelGroups(p).map((group) => (
              <optgroup key={group.id} label={group.detail ? `${group.label} · ${group.detail}` : group.label}>
                {group.models.map((m) => (
                  <option key={`${group.id}:${m}`} value={modelChoiceValue(p.id, group.accountId, m)}>{modelDisplayName(p, m)}</option>
                ))}
              </optgroup>
            )))}
          </select>
          <div className="mt-1 truncate text-[12px] text-ink-4">{provider?.name}{selectedAccountLabel ? ` · ${selectedAccountLabel}` : ''}</div>
        </div>
      )}
      <div className="mb-2">
        <ReasoningEffortSelect provider={provider} model={model} value={reasoningEffort}
          defaultEffort={settings.reasoningEffort} disabled={running} onChange={setReasoningEffort} />
      </div>
      <div className="flex gap-2">
        <Button variant="primary" size="sm" className="flex-1 justify-center" onClick={add} disabled={noAllowedProviders || !model}>{tx("Add agent")}</Button>
        <Button variant="ghost" size="sm" onClick={onCancel}>{tx("Cancel")}</Button>
      </div>
    </fieldset>
  )
}
