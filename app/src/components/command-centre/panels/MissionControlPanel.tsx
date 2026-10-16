import type { Dispatch, SetStateAction } from 'react'
import clsx from 'clsx'
import { relativeTime } from '../../../lib/format'
import type {
  AgentRoleProfileId,
  AssistantControlSnapshot,
  FanOutBudget,
  Mission,
  MissionContextMode,
  MissionInspectionSnapshot,
  MissionRepeatMode,
  MissionStatus,
  Project,
  RunBudgetClass,
} from '../../../lib/types'
import { Icon } from '../../../lib/icons'
import { Button } from '../../ui'
import { EmptyState, Panel, StatusIcon } from '../ui'
import { missionIsTerminal, missionStatusLabel } from '../attention'
import {
  FAN_OUT_OPTIONS,
  MISSION_ROW_PREVIEW_LIMIT,
  ROLE_PROFILE_OPTIONS,
  limitMissionRows,
} from '../model'
import { missionContextLabel, missionRepeatLabel, missionTone } from '../work'

type StateSetter<T> = Dispatch<SetStateAction<T>>

interface MissionControlPanelProps {
  activeMissions: number
  missionRows: Mission[]
  snapshot: AssistantControlSnapshot | null
  projects: Project[]
  missionFormsOpen: boolean
  toggleMissionForms: () => void
  showAllMissions: boolean
  toggleShowAllMissions: () => void
  prTitle: string
  setPrTitle: StateSetter<string>
  prObjective: string
  setPrObjective: StateSetter<string>
  prProjectId: string
  setPrProjectId: StateSetter<string>
  prApprovals: string
  setPrApprovals: StateSetter<string>
  prAutoStart: boolean
  setPrAutoStart: StateSetter<boolean>
  prPublishRemote: boolean
  setPrPublishRemote: StateSetter<boolean>
  createLocalPrLoop: () => Promise<void>
  prBusy: boolean
  missionTitle: string
  setMissionTitle: StateSetter<string>
  missionObjective: string
  setMissionObjective: StateSetter<string>
  missionStopCondition: string
  setMissionStopCondition: StateSetter<string>
  missionRepeat: MissionRepeatMode
  setMissionRepeat: StateSetter<MissionRepeatMode>
  missionContext: MissionContextMode
  setMissionContext: StateSetter<MissionContextMode>
  missionRoleProfile: AgentRoleProfileId
  setMissionRoleProfile: StateSetter<AgentRoleProfileId>
  missionFanOut: FanOutBudget
  setMissionFanOut: StateSetter<FanOutBudget>
  missionMaxRuns: string
  setMissionMaxRuns: StateSetter<string>
  missionCostCap: string
  setMissionCostCap: StateSetter<string>
  missionExpiresDays: string
  setMissionExpiresDays: StateSetter<string>
  missionBudget: RunBudgetClass
  setMissionBudget: StateSetter<RunBudgetClass>
  missionGateDestructive: boolean
  setMissionGateDestructive: StateSetter<boolean>
  createMissionFromDraft: () => Promise<void>
  missionInspection: MissionInspectionSnapshot | null
  missionInspectingId: string | null
  inspectMission: (id: string) => Promise<void>
  setMission: (id: string, status: MissionStatus, message?: string) => Promise<void>
  beatMission: (id: string) => Promise<void>
}

export function MissionControlPanel(props: MissionControlPanelProps) {
  const {
    activeMissions,
    missionRows,
    snapshot,
    projects,
    missionFormsOpen,
    toggleMissionForms,
    showAllMissions,
    toggleShowAllMissions,
    prTitle,
    setPrTitle,
    prObjective,
    setPrObjective,
    prProjectId,
    setPrProjectId,
    prApprovals,
    setPrApprovals,
    prAutoStart,
    setPrAutoStart,
    prPublishRemote,
    setPrPublishRemote,
    createLocalPrLoop,
    prBusy,
    missionTitle,
    setMissionTitle,
    missionObjective,
    setMissionObjective,
    missionStopCondition,
    setMissionStopCondition,
    missionRepeat,
    setMissionRepeat,
    missionContext,
    setMissionContext,
    missionRoleProfile,
    setMissionRoleProfile,
    missionFanOut,
    setMissionFanOut,
    missionMaxRuns,
    setMissionMaxRuns,
    missionCostCap,
    setMissionCostCap,
    missionExpiresDays,
    setMissionExpiresDays,
    missionBudget,
    setMissionBudget,
    missionGateDestructive,
    setMissionGateDestructive,
    createMissionFromDraft,
    missionInspection,
    missionInspectingId,
    inspectMission,
    setMission,
    beatMission,
  } = props

  const missionPreview = limitMissionRows(missionRows, showAllMissions ? missionRows.length : MISSION_ROW_PREVIEW_LIMIT)

  return (
    <div id="command-centre-missions">
      <Panel title="Mission Control" icon="spark" action={activeMissions ? `${activeMissions} active` : 'ready'}>
        <div className="space-y-3 px-3 py-3">
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-card-2 px-2.5 py-2">
            <span className="min-w-0 flex-1 truncate text-[13px] text-ink-3">
              {missionRows.length ? `${activeMissions} active · ${missionRows.length} total` : 'No missions yet'}
            </span>
            <Button variant={missionFormsOpen ? 'default' : 'ghost'} size="xs" icon={missionFormsOpen ? 'chevronDown' : 'plus'} onClick={toggleMissionForms}>
              {missionFormsOpen ? 'Hide forms' : 'New mission'}
            </Button>
          </div>
          {missionFormsOpen && (
          <>
          <div className="rounded-lg border border-line bg-card-2 p-2.5">
            <div className="mb-2 flex items-center gap-2">
              <Icon name="branch" size={13} className="text-accent" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px] font-semibold text-ink">Local PR review loop</div>
                <div className="truncate text-[12px] text-ink-4">Build thread, review thread, feedback loop.</div>
              </div>
            </div>
            <input
              value={prTitle}
              onChange={(e) => setPrTitle(e.target.value)}
              placeholder="Local PR title"
              className="h-8 w-full rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent"
            />
            <textarea
              value={prObjective}
              onChange={(e) => setPrObjective(e.target.value)}
              placeholder="What should the implementation thread build?"
              rows={3}
              className="scroll-thin mt-2 w-full resize-none rounded-lg border border-line bg-card px-2.5 py-2 text-[13px] leading-snug text-ink outline-none placeholder:text-ink-3 focus:border-accent"
            />
            <div className="mt-2 grid grid-cols-[minmax(0,1fr)_96px] gap-2">
              <select
                value={prProjectId}
                onChange={(e) => setPrProjectId(e.target.value)}
                className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none"
              >
                <option value="">Default workspace</option>
                {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
              </select>
              <input
                value={prApprovals}
                onChange={(e) => setPrApprovals(e.target.value)}
                placeholder="Review passes"
                className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent"
              />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <button
                onClick={() => setPrAutoStart((value) => !value)}
                className={clsx('h-7 rounded-md border px-2 text-[13px] transition-colors', prAutoStart ? 'border-line bg-card text-ink' : 'border-line bg-card-2 text-ink-3')}
              >
                Auto-start builder
              </button>
              <button
                onClick={() => setPrPublishRemote((value) => !value)}
                title="After the reviewer approves, push the branch and open a GitHub draft PR (requires the gh CLI)."
                className={clsx('h-7 rounded-md border px-2 text-[13px] transition-colors', prPublishRemote ? 'border-accent bg-accent/15 text-accent' : 'border-line bg-card-2 text-ink-3')}
              >
                Publish to GitHub
              </button>
              <span className="min-w-0 flex-1" />
              <Button variant="primary" size="xs" icon="branch" onClick={createLocalPrLoop} disabled={!prObjective.trim() || prBusy}>
                {prBusy ? 'Creating' : 'Create loop'}
              </Button>
            </div>
          </div>
          <div className="rounded-lg border border-line bg-card-2 p-2.5">
            <input
              value={missionTitle}
              onChange={(e) => setMissionTitle(e.target.value)}
              placeholder="Mission title"
              className="h-8 w-full rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent"
            />
            <textarea
              value={missionObjective}
              onChange={(e) => setMissionObjective(e.target.value)}
              placeholder="Task it repeats"
              rows={3}
              className="scroll-thin mt-2 w-full resize-none rounded-lg border border-line bg-card px-2.5 py-2 text-[13px] leading-snug text-ink outline-none placeholder:text-ink-3 focus:border-accent"
            />
            <textarea
              value={missionStopCondition}
              onChange={(e) => setMissionStopCondition(e.target.value)}
              placeholder="Stop condition, e.g. no actionable unresolved errors remain."
              rows={2}
              className="scroll-thin mt-2 w-full resize-none rounded-lg border border-line bg-card px-2.5 py-2 text-[13px] leading-snug text-ink outline-none placeholder:text-ink-3 focus:border-accent"
            />
            <div className="mt-2 grid grid-cols-2 gap-2">
              <select
                value={missionRepeat}
                onChange={(e) => setMissionRepeat(e.target.value as MissionRepeatMode)}
                className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none"
              >
                <option value="until_condition">Until condition</option>
                <option value="interval">Interval</option>
                <option value="self_paced">Self-paced</option>
                <option value="fixed_count">Fixed count</option>
                <option value="forever">Forever</option>
              </select>
              <select
                value={missionContext}
                onChange={(e) => setMissionContext(e.target.value as MissionContextMode)}
                className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none"
              >
                <option value="fresh">Fresh each run</option>
                <option value="build_on_previous">Build on previous</option>
                <option value="fresh_with_notes">Fresh + notes</option>
              </select>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <select
                value={missionRoleProfile}
                onChange={(e) => setMissionRoleProfile(e.target.value as AgentRoleProfileId)}
                className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none"
              >
                {ROLE_PROFILE_OPTIONS.map((profile) => <option key={profile.id} value={profile.id}>{profile.label} profile</option>)}
              </select>
              <select
                value={missionFanOut}
                onChange={(e) => setMissionFanOut(e.target.value as FanOutBudget)}
                className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none"
              >
                {FAN_OUT_OPTIONS.map((budget) => <option key={budget.id} value={budget.id}>{budget.label} fan-out</option>)}
              </select>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2">
              <input value={missionMaxRuns} onChange={(e) => setMissionMaxRuns(e.target.value)} placeholder="Max runs" className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
              <input value={missionCostCap} onChange={(e) => setMissionCostCap(e.target.value)} placeholder="Cost cap $" className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
              <input value={missionExpiresDays} onChange={(e) => setMissionExpiresDays(e.target.value)} placeholder="Expires days" className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <select
                value={missionBudget}
                onChange={(e) => setMissionBudget(e.target.value as RunBudgetClass)}
                className="h-7 rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none"
              >
                {(snapshot?.budgets ?? []).map((budget) => <option key={budget.executionClass} value={budget.executionClass}>{budget.label}</option>)}
              </select>
              <button
                onClick={() => setMissionGateDestructive((value) => !value)}
                className={clsx('h-7 rounded-md border px-2 text-[13px] transition-colors', missionGateDestructive ? 'border-line bg-card text-ink' : 'border-line bg-card-2 text-ink-3')}
              >
                Gate destructive
              </button>
              <span className="min-w-0 flex-1" />
              <Button variant="primary" size="xs" icon="plus" onClick={createMissionFromDraft} disabled={!missionObjective.trim()}>Create</Button>
            </div>
          </div>
          </>
          )}

          {missionRows.length === 0 ? (
            <EmptyState icon="spark" title="No missions yet" detail="Create a resumable job with heartbeat and checkpoints." />
          ) : missionPreview.visible.map((mission) => (
            <div key={mission.id} className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
              {(() => {
                const inspected = missionInspection?.mission.id === mission.id ? missionInspection : null
                return (
              <div className="flex items-start gap-2.5">
                <StatusIcon icon="spark" tone={missionTone(mission.status)} pulse={mission.status === 'running'} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <div className="truncate text-[13px] font-semibold text-ink">{mission.title}</div>
                    <span className="shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{missionStatusLabel(mission.status)}</span>
                  </div>
                  <div className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{mission.objective}</div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{missionRepeatLabel(mission.repeatMode)}</span>
                    <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{missionContextLabel(mission.contextMode)}</span>
                    <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{mission.roleProfileId ?? 'operator'}</span>
                    <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{mission.fanOutBudget ?? 'single'} fan-out</span>
                    {mission.metadata?.workflow === 'local_pr_loop' && <span className="rounded-md bg-accent/15 px-1.5 py-0.5 text-[12px] text-accent">Local PR</span>}
                    {typeof mission.metadata?.branch === 'string' && <span className="max-w-[160px] truncate rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{mission.metadata.branch}</span>}
                    {mission.maxRuns != null && <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">Max {mission.maxRuns}x</span>}
                    {mission.costCapUsd != null && <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">Cap ${mission.costCapUsd}</span>}
                    {mission.destructiveApprovalRequired && <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">Gate destructive</span>}
                  </div>
                  {mission.stopCondition && <div className="mt-1 line-clamp-2 text-[13px] leading-snug text-ink-4">Stop: {mission.stopCondition}</div>}
                  <div className="mt-2 flex flex-wrap items-center gap-1.5">
                    <Button variant="default" size="xs" icon="fileText" onClick={() => void inspectMission(mission.id)}>
                      {missionInspectingId === mission.id ? 'Opening' : inspected ? 'Close' : 'Inspect'}
                    </Button>
                    {mission.status !== 'running' && mission.status !== 'waiting_approval' && !missionIsTerminal(mission.status) && <Button variant="default" size="xs" icon="play" onClick={() => setMission(mission.id, 'running')}>Start</Button>}
                    {mission.status === 'running' && <Button variant="ghost" size="xs" icon="clock" onClick={() => beatMission(mission.id)}>Beat</Button>}
                    {mission.status === 'running' && <Button variant="ghost" size="xs" icon="stop" onClick={() => setMission(mission.id, 'paused')}>Pause</Button>}
                    {!missionIsTerminal(mission.status) && <Button variant="ghost" size="xs" icon="x" onClick={() => setMission(mission.id, 'cancelled')}>Cancel</Button>}
                    {!missionIsTerminal(mission.status) && <Button variant="default" size="xs" icon="check" onClick={() => setMission(mission.id, 'completed')}>Complete</Button>}
                  </div>
                  {mission.checkpoints.length > 0 && (
                    <div className="mt-2 border-t border-line pt-2 text-[13px] text-ink-3">
                      Last checkpoint: {mission.checkpoints[mission.checkpoints.length - 1].summary}
                    </div>
                  )}
                  {inspected && (
                    <div className="mt-3 border-t border-line pt-3 text-[13px] text-ink-3">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="rounded-md bg-ink-5 px-1.5 py-0.5">Opened {relativeTime(inspected.generatedAt)}</span>
                        <span className="rounded-md bg-ink-5 px-1.5 py-0.5">{inspected.checkpointCount} checkpoints</span>
                        <span className="rounded-md bg-ink-5 px-1.5 py-0.5">{inspected.journalCount} journal events</span>
                        <span className="rounded-md bg-ink-5 px-1.5 py-0.5">{inspected.scratchMemoryCount} scratch notes</span>
                        <span className="rounded-md bg-ink-5 px-1.5 py-0.5">{inspected.rerunRequired ? 'Rerun needed' : 'No rerun needed'}</span>
                      </div>
                      {inspected.mission.checkpoints.length > 0 && (
                        <div className="mt-3">
                          <div className="mb-1 text-[12px] font-semibold uppercase tracking-[0.14em] text-ink-4">Checkpoints</div>
                          <div className="space-y-1.5">
                            {inspected.mission.checkpoints.slice(-3).map((checkpoint) => (
                              <div key={checkpoint.id} className="grid grid-cols-[auto,1fr] gap-x-2 gap-y-0.5">
                                <span className="text-ink-4">#{checkpoint.sequence}</span>
                                <span className="min-w-0 text-ink-2">{checkpoint.label}</span>
                                <span />
                                <span className="min-w-0 text-ink-4">{checkpoint.summary}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {inspected.mission.journal.length > 0 && (
                        <div className="mt-3">
                          <div className="mb-1 text-[12px] font-semibold uppercase tracking-[0.14em] text-ink-4">Recent journal</div>
                          <div className="space-y-1.5">
                            {inspected.mission.journal.slice(-4).map((event) => (
                              <div key={event.id} className="grid grid-cols-[88px,1fr] gap-x-2">
                                <span className="truncate text-ink-4">{event.actor} · {event.kind}</span>
                                <span className="min-w-0 text-ink-3">{event.message}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {inspected.mission.scratchMemory.length > 0 && (
                        <div className="mt-3">
                          <div className="mb-1 text-[12px] font-semibold uppercase tracking-[0.14em] text-ink-4">Scratch memory</div>
                          <div className="space-y-1.5">
                            {inspected.mission.scratchMemory.slice(-3).map((entry) => (
                              <div key={entry.id} className="grid grid-cols-[76px,1fr] gap-x-2">
                                <span className="truncate text-ink-4">{entry.kind}</span>
                                <span className="min-w-0 text-ink-3">{entry.text}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
                )
              })()}
            </div>
          ))}

          {(missionPreview.hasMore || showAllMissions) && (
            <button
              type="button"
              onClick={toggleShowAllMissions}
              className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-line bg-card-2 px-2.5 py-2 text-[13px] font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
            >
              {showAllMissions ? 'Show fewer missions' : `View all missions (${missionPreview.total})`}
            </button>
          )}
        </div>
      </Panel>
    </div>
  )
}
