import type { Dispatch, SetStateAction } from 'react'
import type { NodeDashboard } from '../../lib/node-dashboard'
import type { DoctorCheck, DoctorSummary } from '../../lib/readiness-doctor'
import type {
  AgentRoleProfileId,
  ApprovalRule,
  AssistantControlSnapshot,
  Automation,
  FanOutBudget,
  Job,
  Mission,
  MissionContextMode,
  MissionInspectionSnapshot,
  MissionRepeatMode,
  MissionStatus,
  ModelRouterSnapshot,
  Project,
  RemoteOrchestratorSnapshotResult,
  RunBudgetClass,
  Settings,
  Thread,
} from '../../lib/types'
import type {
  ApprovalGroup,
  ApprovalRow,
  AttentionRow,
  AwayBriefRow,
  CommandCentreSetView,
  CommandStatusSummary as CommandStatusSummaryModel,
  ConnectorHealthSignal,
  LiveJob,
  PowerGuardSignal,
  ProviderVisibilityRow,
  RunningWorkSummary,
  Tone,
  WorkLaneSummary,
} from './model'
import { ApprovalsPanel } from './panels/ApprovalsPanel'
import { AttentionPanel } from './panels/AttentionPanel'
import { ConnectorHealthPanel, ControlGapsPanel, RunClassesPanel } from './panels/ConnectorHealthPanel'
import { DailyBriefPanel } from './panels/DailyBriefPanel'
import { HeaderBar } from './panels/HeaderBar'
import { LiveJobsPanel, PowerGuardPanel } from './panels/MiscColumn2Panels'
import { MissionControlPanel } from './panels/MissionControlPanel'
import { ModelVisibilityPanel } from './panels/ModelVisibilityPanel'
import { NodesPanel } from './panels/NodesPanel'
import { OrchestratorsPanel } from './panels/OrchestratorsPanel'
import { ReadinessDoctorPanel } from './panels/ReadinessDoctorPanel'
import { RunningWorkPanel } from './panels/RunningWorkPanel'
import { StatusMetrics } from './panels/StatusMetrics'

type StateSetter<T> = Dispatch<SetStateAction<T>>

interface CommandCentreLayoutProps {
  snapshot: AssistantControlSnapshot | null
  attentionRows: AttentionRow[]
  pendingApprovals: number
  liveJobs: LiveJob[]
  activeMissions: number
  connectedCount: number
  connectorRows: AssistantControlSnapshot['connectors']
  refresh: () => Promise<void>
  loading: boolean
  latestThread: Thread | null
  setView: CommandCentreSetView
  primaryBrief: Automation | null
  runAutomation: (id: string) => Promise<void>
  error: string
  unreadSignals: number
  runningWorkSummary: RunningWorkSummary
  connectorActionCount: number
  gapCount: number
  connectorHealthRows: Array<{ connector: AssistantControlSnapshot['connectors'][number]; health: ConnectorHealthSignal }>
  nextBrief: number | null
  missionRows: Mission[]
  selected: AttentionRow | null
  setSelectedId: StateSetter<string>
  awayBriefRows: AwayBriefRow[]
  dailyBriefLines: string[]
  briefAutomations: Automation[]
  orchestrators: Settings['remoteOrchestrators']
  remoteSnapshots: Record<string, RemoteOrchestratorSnapshotResult>
  remoteBusy: string | null
  refreshRemoteHost: (hostId: string) => Promise<void>
  approvalRows: ApprovalRow[]
  approvalGroups: ApprovalGroup[]
  savedPolicies: ApprovalRule[]
  revokePolicy: (rule: ApprovalRule) => Promise<void>
  revokingRuleId: string | null
  workLaneSummary: WorkLaneSummary[]
  actionableDurableJobs: Job[]
  retryJob: (id: string, reason?: string, nextWakeAt?: number | null, resumePlan?: string | null) => Promise<void>
  resumeJob: (id: string, resumePlan?: string | null) => Promise<void>
  deadLetterJob: (id: string, reason?: string) => Promise<void>
  openDurableJob: (job: Job) => void
  powerGuard: PowerGuardSignal
  settings: Settings | null
  longWorkCount: number
  prTitle: string
  setPrTitle: StateSetter<string>
  prObjective: string
  setPrObjective: StateSetter<string>
  prProjectId: string
  setPrProjectId: StateSetter<string>
  projects: Project[]
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
  providerVisibilityRows: ProviderVisibilityRow[]
  activeProviderCount: number
  modelRouterSnapshot: ModelRouterSnapshot | null
  nodeDashboard: NodeDashboard
  readinessChecks: DoctorCheck[]
  readinessSummary: DoctorSummary
  readinessTone: Tone
  refreshReadiness: () => void
  openReadinessSettings: () => void
  syncEmailAndRefresh: () => Promise<void>
  statusSummary: CommandStatusSummaryModel
  statusExpanded: boolean
  toggleStatusExpanded: () => void
  awayOpen: boolean
  toggleAwayOpen: () => void
  orchestratorsOpen: boolean
  toggleOrchestratorsOpen: () => void
  missionFormsOpen: boolean
  toggleMissionForms: () => void
  showAllMissions: boolean
  toggleShowAllMissions: () => void
}

export function CommandCentreLayout(props: CommandCentreLayoutProps) {
  const {
    snapshot,
    attentionRows,
    pendingApprovals,
    liveJobs,
    activeMissions,
    connectedCount,
    connectorRows,
    refresh,
    loading,
    latestThread,
    setView,
    primaryBrief,
    runAutomation,
    error,
    unreadSignals,
    runningWorkSummary,
    connectorActionCount,
    gapCount,
    connectorHealthRows,
    nextBrief,
    missionRows,
    selected,
    setSelectedId,
    awayBriefRows,
    dailyBriefLines,
    briefAutomations,
    orchestrators,
    remoteSnapshots,
    remoteBusy,
    refreshRemoteHost,
    approvalRows,
    approvalGroups,
    savedPolicies,
    revokePolicy,
    revokingRuleId,
    workLaneSummary,
    actionableDurableJobs,
    retryJob,
    resumeJob,
    deadLetterJob,
    openDurableJob,
    powerGuard,
    settings,
    longWorkCount,
    prTitle,
    setPrTitle,
    prObjective,
    setPrObjective,
    prProjectId,
    setPrProjectId,
    projects,
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
    providerVisibilityRows,
    activeProviderCount,
    modelRouterSnapshot,
    nodeDashboard,
    readinessChecks,
    readinessSummary,
    readinessTone,
    refreshReadiness,
    openReadinessSettings,
    syncEmailAndRefresh,
    statusSummary,
    statusExpanded,
    toggleStatusExpanded,
    awayOpen,
    toggleAwayOpen,
    orchestratorsOpen,
    toggleOrchestratorsOpen,
    missionFormsOpen,
    toggleMissionForms,
    showAllMissions,
    toggleShowAllMissions,
  } = props

  const readinessLabel = readinessSummary.error
    ? `${readinessSummary.error} blocked`
    : readinessSummary.warning
      ? `${readinessSummary.warning} review`
      : readinessSummary.checking
        ? `${readinessSummary.checking} checking`
        : 'ready'

  return (
    <div className="flex h-full flex-col bg-canvas">
      <HeaderBar
        snapshot={snapshot}
        attentionRows={attentionRows}
        pendingApprovals={pendingApprovals}
        liveJobs={liveJobs}
        activeMissions={activeMissions}
        connectedCount={connectedCount}
        connectorRows={connectorRows}
        readinessLabel={readinessLabel}
        refresh={refresh}
        loading={loading}
        latestThread={latestThread}
        setView={setView}
        primaryBrief={primaryBrief}
        runAutomation={runAutomation}
        error={error}
      />

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 py-5 xl:pr-[304px]">
        <StatusMetrics
          statusSummary={statusSummary}
          statusExpanded={statusExpanded}
          toggleStatusExpanded={toggleStatusExpanded}
          attentionRows={attentionRows}
          unreadSignals={unreadSignals}
          pendingApprovals={pendingApprovals}
          liveJobs={liveJobs}
          runningWorkSummary={runningWorkSummary}
          connectedCount={connectedCount}
          connectorRows={connectorRows}
          connectorActionCount={connectorActionCount}
          gapCount={gapCount}
          connectorHealthRows={connectorHealthRows}
          nextBrief={nextBrief}
          primaryBrief={primaryBrief}
          missionRows={missionRows}
          activeMissions={activeMissions}
          readinessSummary={readinessSummary}
          readinessTone={readinessTone}
        />

        <div className="mt-4 grid grid-cols-1 gap-4 min-[1800px]:grid-cols-[minmax(330px,0.95fr)_minmax(420px,1.2fr)_minmax(330px,0.95fr)]">
          <div className="space-y-4">
            <AttentionPanel
              attentionRows={attentionRows}
              selected={selected}
              setSelectedId={setSelectedId}
              awayBriefRows={awayBriefRows}
              awayOpen={awayOpen}
              toggleAwayOpen={toggleAwayOpen}
            />

            <DailyBriefPanel
              primaryBrief={primaryBrief}
              runAutomation={runAutomation}
              setView={setView}
              dailyBriefLines={dailyBriefLines}
              briefAutomations={briefAutomations}
            />

            <OrchestratorsPanel
              orchestrators={orchestrators}
              remoteSnapshots={remoteSnapshots}
              remoteBusy={remoteBusy}
              refreshRemoteHost={refreshRemoteHost}
              setView={setView}
              isOpen={orchestratorsOpen}
              onToggle={toggleOrchestratorsOpen}
            />
          </div>

          <div className="space-y-4">
            <ApprovalsPanel
              pendingApprovals={pendingApprovals}
              approvalGroups={approvalGroups}
              approvalRows={approvalRows}
              savedPolicies={savedPolicies}
              revokePolicy={revokePolicy}
              revokingRuleId={revokingRuleId}
            />

            <RunningWorkPanel
              runningWorkSummary={runningWorkSummary}
              workLaneSummary={workLaneSummary}
              actionableDurableJobs={actionableDurableJobs}
              projects={projects}
              orchestrators={orchestrators}
              retryJob={retryJob}
              resumeJob={resumeJob}
              deadLetterJob={deadLetterJob}
              openDurableJob={openDurableJob}
            />

            <PowerGuardPanel
              powerGuard={powerGuard}
              settings={settings}
              longWorkCount={longWorkCount}
              runningWorkSummary={runningWorkSummary}
              setView={setView}
            />

            <LiveJobsPanel liveJobs={liveJobs} />

            <MissionControlPanel
              activeMissions={activeMissions}
              missionRows={missionRows}
              snapshot={snapshot}
              projects={projects}
              missionFormsOpen={missionFormsOpen}
              toggleMissionForms={toggleMissionForms}
              showAllMissions={showAllMissions}
              toggleShowAllMissions={toggleShowAllMissions}
              prTitle={prTitle}
              setPrTitle={setPrTitle}
              prObjective={prObjective}
              setPrObjective={setPrObjective}
              prProjectId={prProjectId}
              setPrProjectId={setPrProjectId}
              prApprovals={prApprovals}
              setPrApprovals={setPrApprovals}
              prAutoStart={prAutoStart}
              setPrAutoStart={setPrAutoStart}
              prPublishRemote={prPublishRemote}
              setPrPublishRemote={setPrPublishRemote}
              createLocalPrLoop={createLocalPrLoop}
              prBusy={prBusy}
              missionTitle={missionTitle}
              setMissionTitle={setMissionTitle}
              missionObjective={missionObjective}
              setMissionObjective={setMissionObjective}
              missionStopCondition={missionStopCondition}
              setMissionStopCondition={setMissionStopCondition}
              missionRepeat={missionRepeat}
              setMissionRepeat={setMissionRepeat}
              missionContext={missionContext}
              setMissionContext={setMissionContext}
              missionRoleProfile={missionRoleProfile}
              setMissionRoleProfile={setMissionRoleProfile}
              missionFanOut={missionFanOut}
              setMissionFanOut={setMissionFanOut}
              missionMaxRuns={missionMaxRuns}
              setMissionMaxRuns={setMissionMaxRuns}
              missionCostCap={missionCostCap}
              setMissionCostCap={setMissionCostCap}
              missionExpiresDays={missionExpiresDays}
              setMissionExpiresDays={setMissionExpiresDays}
              missionBudget={missionBudget}
              setMissionBudget={setMissionBudget}
              missionGateDestructive={missionGateDestructive}
              setMissionGateDestructive={setMissionGateDestructive}
              createMissionFromDraft={createMissionFromDraft}
              missionInspection={missionInspection}
              missionInspectingId={missionInspectingId}
              inspectMission={inspectMission}
              setMission={setMission}
              beatMission={beatMission}
            />
          </div>

          <div className="space-y-4">
            <NodesPanel nodeDashboard={nodeDashboard} />

            <ReadinessDoctorPanel
              readinessChecks={readinessChecks}
              readinessSummary={readinessSummary}
              readinessTone={readinessTone}
              refreshReadiness={refreshReadiness}
              openReadinessSettings={openReadinessSettings}
            />

            <ModelVisibilityPanel
              providerVisibilityRows={providerVisibilityRows}
              activeProviderCount={activeProviderCount}
              modelRouterSnapshot={modelRouterSnapshot}
            />

            <ConnectorHealthPanel
              connectorHealthRows={connectorHealthRows}
              connectorActionCount={connectorActionCount}
              connectedCount={connectedCount}
              connectorRows={connectorRows}
              loading={loading}
              syncEmailAndRefresh={syncEmailAndRefresh}
              setView={setView}
            />

            <ControlGapsPanel gapCount={gapCount} snapshot={snapshot} />

            <RunClassesPanel snapshot={snapshot} />
          </div>
        </div>
      </div>
    </div>
  )
}
