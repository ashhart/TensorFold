import { createApplicationThreadPersistence } from './main/composition/thread-persistence'
import { registerApplicationWorkspaceContent } from './main/composition/workspace-content'
import { createApplicationSecurityPurge } from './main/composition/security-purge'
import { createApplicationWorkspace } from './main/composition/workspace'
import { createApplicationConnections } from './main/composition/connections'
import { initializeApplicationDevices } from './main/composition/device-services'
import { registerApplicationWorkspaceServices } from './main/composition/workspace-services'
import { bootApplication, loadConfiguredPlugins, registerApplicationErrors } from './main/composition/startup'
import { createApplicationApprovalRules } from './main/composition/approval-rules'
import { createApplicationAttention } from './main/composition/attention'
import { createApplicationDesktop } from './main/composition/desktop-shell'
import { registerApplicationShutdown } from './main/composition/shutdown'
import { app, clipboard, Menu, MenuItem } from 'electron'
import { randomUUID } from 'node:crypto'
import { defaultSettings, loadSkillTrust, saveSkillTrust, saveSchedulerLease } from './store'
import { streamChat } from './agent/openai'
import type { JobPatch } from './agent/jobs'
import { emptyUpdateStatus } from './agent/update-checker'
import { confirmSensitiveAction } from './main/host/sensitive-action'
import { createApplyTheme, createEnsureWindowVisible, createHandleBootFailure, normalizeRemoteOrchestratorDestinations } from './main/host/boot-helpers'
import { createEmitHelpers } from './main/host/emit-helpers'
import { securityRoleForLaunch } from './agent/security'
import { type ApprovalPolicyScope } from './agent/approval-policy'
import type { StartRunPayload, Thread, Job } from './agent/types'
import { type EventSpineHost } from './agent/event-spine-host'
import { store } from './main/state'
import { handle } from './main/host/ipc'
import { persistArtifacts, persistJobs, persistMissions, persistRooms, persistSubAgents, persistThreads, persistTickets } from './main/host/persistence'
import { auditEvent, requireSecurityRole } from './main/host/audit'
import { applyPowerSettings } from './main/host/power'
import { checkForAppUpdate, openUpdateDownload, startUpdateChecks } from './main/host/updates'
import { runBrowserAutomation } from './main/host/browser-automation'
import { registerExportIpc } from './main/host/export-ipc'
import { WorkspaceMutationGuard } from './main/host/workspace-mutation-guard'
import type { SchedulerLease } from './agent/automation-scheduler'
import { registerCommandsIpc } from './main/host/commands-ipc'
import { killProjectTerminals, registerTerminalIpc, type TerminalRemoteApi } from './main/host/terminal-runtime'
import { createAutomationRuntime, registerAutomationIpc, type AutomationRuntime } from './main/host/automations'
import { createMissionRuntime, type MissionRuntime } from './main/host/mission-runtime'
import { createJobMissionCore } from './main/host/job-mission-core'
import { registerMiscIpc } from './main/host/misc-ipc'
import { registerThreadHandoffIpc } from './main/host/handoff-ipc'
import { registerPlanningGuardIpc } from './main/host/planning-guard-ipc'
import { createRemoteHostApi } from './main/host/remote-host-api'
import { registerCoreIpc } from './main/host/core-ipc'
import { buildCoreIpcConfig } from './main/host/core-ipc-config'
import { createTicketWorkerRuntime, type TicketWorkerInput, type TicketWorkerRuntime } from './main/host/ticket-worker-runtime'
import { createThreadRuntime, type ThreadRuntime, type ThreadRunAdmission } from './main/host/thread-runtime'
import { createLoopHost } from './main/host/loop-host'
import { createJobControlRuntime, buildJobControlConfig, type JobControlRuntime } from './main/host/job-control-runtime'
import { createBotPollingRuntime, type BotPollingRuntime } from './main/host/bot-polling-runtime'
import { createTelegramRemoteRuntime } from './main/host/telegram-remote-runtime'
import { createTelegramRemoteDispatch } from './main/host/telegram-remote-dispatch'
import { steerExistingRoom } from './main/host/room-steering-runtime'
import { renderProjectMemoryBucket } from './main/host/project-memory-renderer'
import { createRoomJobRuntime, type RoomJobRuntime } from './main/host/room-job-runtime'
import { createToolContextRuntime, type ToolContextRuntime } from './main/host/tool-context-runtime'
import { createProjectSkillsRuntime } from './main/host/project-skills-runtime'
import { createObserverHub } from './agent/observer-hooks'
import { hostPluginRuntime } from './agent/plugin-runtime'
import { sendToRenderer } from './main/host/renderer-events'
import { effectiveContextLimit } from './main/host/context-limit'
import { createApprovalRuntime, isHighRiskTool } from './main/host/approval-runtime'
import { registerIncidentIpc } from './main/host/incidents-ipc'
import { ensureApplicationEditMenu } from './main/host/native-text-editing'
import { type WorkspaceRuntimeIpc } from './main/host/workspace-runtime-ipc'
import { auditModelRoute } from './main/host/provider-runtime'
import { APP_NAME, HEADLESS_MODE } from './main/early-electron-paths'

process.env.VONTRA_SECURITY_ROLE = securityRoleForLaunch(process.env, {
  headless: HEADLESS_MODE,
})

const aborters = new Map<string, AbortController>()
const workspaceMutationGuard = new WorkspaceMutationGuard()
let automationRuntime: AutomationRuntime

const automationApi = {
  runAutomation: (...args: Parameters<AutomationRuntime['runAutomation']>) => automationRuntime.runAutomation(...args),
  startScheduler: () => automationRuntime.startScheduler(),
}

let remoteHostApi: ReturnType<typeof createRemoteHostApi>
let terminalRemoteApi: TerminalRemoteApi | null = null
let jobControlRuntime: JobControlRuntime

const jobControlApi = {
  createLocalPrLoop: (...args: Parameters<JobControlRuntime['createLocalPrLoop']>) => jobControlRuntime.createLocalPrLoop(...args),
  settleThreadRunJob: (...args: Parameters<JobControlRuntime['settleThreadRunJob']>) => jobControlRuntime.settleThreadRunJob(...args),
  setMissionStatusLocal: (...args: Parameters<JobControlRuntime['setMissionStatusLocal']>) => jobControlRuntime.setMissionStatusLocal(...args),
  recordMissionHeartbeatLocal: (...args: Parameters<JobControlRuntime['recordMissionHeartbeatLocal']>) => jobControlRuntime.recordMissionHeartbeatLocal(...args),
  resumeAutoJobsAfterRestart: () => jobControlRuntime.resumeAutoJobsAfterRestart(),
}

const {
  connectorRuntime, rememberConnectorEvents, recordEmailDraft, recordEmailEvent,
  recordCalendarEvent, recordMessageEvent, registerConnectorIpc, webhookServer, pairingRegistry,
  persistPairingSessions, remoteServer, mcpHost, mobileBridgeRuntime, sshManager,
} = createApplicationConnections({
  findApprovalRule, rememberApprovalRule, runAutomation: automationApi.runAutomation, botControlRuntime,
  aborters, workspaceMutationGuard, saveTrustedKey: (key) => coreIpc.saveTrustedKey(key), emitJob: (job) => emitJob(job),
  remoteConfig: {
    spineHost: () => spineHost,
    remoteHostApi: () => remoteHostApi,
    jobControlRuntime: () => jobControlRuntime,
    terminalRemoteApi: () => terminalRemoteApi,
    aborters: () => aborters,
    roomAborters: () => roomAborters,
    runningMissions: () => runningMissions,
    setMissionStatusLocal: jobControlApi.setMissionStatusLocal,
    recordMissionHeartbeatLocal: jobControlApi.recordMissionHeartbeatLocal,
  },
})
store.updateStatus = emptyUpdateStatus(app.getVersion())

const pendingApprovals = new Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>()
const pendingApprovalMeta = new Map<string, { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }>()

const applicationApprovalRules = createApplicationApprovalRules()
function findApprovalRule(...args: Parameters<typeof applicationApprovalRules.findApprovalRule>) {
  return applicationApprovalRules.findApprovalRule(...args)
}
function rememberApprovalRule(...args: Parameters<typeof applicationApprovalRules.rememberApprovalRule>) {
  return applicationApprovalRules.rememberApprovalRule(...args)
}

const roomAborters = new Map<string, AbortController>()

let observerHub: ReturnType<typeof createObserverHub>
let spineHost: EventSpineHost | null = null

const runAttention = createApplicationAttention()
const {
  emitRoom,
  emitTicket,
  emitAutomation,
  emitJob,
  emitMission,
  emitIncident,
  replaceMission,
  getMission,
  ensureRemoteControlToken,
  emitToRenderer,
} = createEmitHelpers({
  remoteServer: () => remoteServer,
  spineHost: () => spineHost,
  observerHub: () => observerHub,
  persistMissions,
  onAgentEvent: (e) => runAttention.observe(e),
})

const JOB_LEASE_OWNER = `${APP_NAME}:${process.pid}`
const THREAD_JOB_HEARTBEAT_MS = 30_000
const DEFAULT_ROOM_WORKER_BUDGET = { maxActive: 2, maxStarts: 4 }
const ROOM_WORKER_SLEEP_WAKE_MS = 2 * 60_000

let roomJobRuntime: RoomJobRuntime

const roomJobApi = {
  appendRoomSupervisorEvent: (...args: Parameters<RoomJobRuntime['appendRoomSupervisorEvent']>) => roomJobRuntime.appendRoomSupervisorEvent(...args),
  ticketsForRoom: (...args: Parameters<RoomJobRuntime['ticketsForRoom']>) => roomJobRuntime.ticketsForRoom(...args),
  patchRoomSupervisorJob: (...args: Parameters<RoomJobRuntime['patchRoomSupervisorJob']>) => roomJobRuntime.patchRoomSupervisorJob(...args),
  parkRoomIfWorkRemains: (...args: Parameters<RoomJobRuntime['parkRoomIfWorkRemains']>) => roomJobRuntime.parkRoomIfWorkRemains(...args),
  rememberJob: (...args: Parameters<RoomJobRuntime['rememberJob']>) => roomJobRuntime.rememberJob(...args),
  activeJob: (...args: Parameters<RoomJobRuntime['activeJob']>) => roomJobRuntime.activeJob(...args),
  roomWorkerBudget: (...args: Parameters<RoomJobRuntime['roomWorkerBudget']>) => roomJobRuntime.roomWorkerBudget(...args),
  ensureRoomSupervisorMission: (...args: Parameters<RoomJobRuntime['ensureRoomSupervisorMission']>) => roomJobRuntime.ensureRoomSupervisorMission(...args),
  finishRoomSupervisorMission: (...args: Parameters<RoomJobRuntime['finishRoomSupervisorMission']>) => roomJobRuntime.finishRoomSupervisorMission(...args),
  claimJob: (...args: Parameters<RoomJobRuntime['claimJob']>) => roomJobRuntime.claimJob(...args),
  heartbeatJob: (...args: Parameters<RoomJobRuntime['heartbeatJob']>) => roomJobRuntime.heartbeatJob(...args),
  finishJob: (...args: Parameters<RoomJobRuntime['finishJob']>) => roomJobRuntime.finishJob(...args),
  reconcileStuckJobs: () => roomJobRuntime.reconcileStuckJobs(),
}
roomJobRuntime = createRoomJobRuntime({
  defaultRoomWorkerBudget: DEFAULT_ROOM_WORKER_BUDGET,
  jobLeaseOwner: JOB_LEASE_OWNER,
  roomWorkerSleepWakeMs: ROOM_WORKER_SLEEP_WAKE_MS,
  emitRoom,
  emitJob,
  emitMission,
  persistJobs,
  persistMissions,
  patchJob,
  replaceMission,
})

const jobMissionCore = createJobMissionCore({ persistJobs, emitJob })
const runningMissions = jobMissionCore.runningMissions
const missionAborters = jobMissionCore.missionAborters
const missionApprovalResolvers = jobMissionCore.missionApprovalResolvers
const abortAndWaitMissionsForArchive = jobMissionCore.abortAndWaitMissionsForArchive
function patchJob(job: Job | null, patch: JobPatch): Job | null { return jobMissionCore.patchJob(job, patch) }
function replaceJobResult(result: { jobs: Job[]; job: Job | null }): Job { return jobMissionCore.replaceJobResult(result) }

let missionRuntime: MissionRuntime

const missionApi = {
  runMission: (...args: Parameters<MissionRuntime['runMission']>) => missionRuntime.runMission(...args),
}

missionRuntime = createMissionRuntime({
  runningMissions,
  missionAborters,
  missionApprovalResolvers,
  claimJob: roomJobApi.claimJob,
  patchJob,
  heartbeatJob: roomJobApi.heartbeatJob,
  finishJob: roomJobApi.finishJob,
  getMission,
  replaceMission,
  fullToolContext,
  renderProjectMemoryBucket,
  applyEventToThread,
  timelineErrorSince,
  timelineError,
  isHighRiskTool,
  pendingApprovalMeta,
  persistJobApproval,
  resolvePersistedJobApproval,
  markRouteErrorRecovered,
  injectThreadRouteHandoff,
  effectiveContextLimit,
  workspaceMutationGuard,
})

const purgeLocalMemory = createApplicationSecurityPurge({
  aborters, roomAborters, missionAborters, pendingApprovals, pendingApprovalMeta,
  missionApprovalResolvers, workspaceMutationGuard, automationRuntime: () => automationRuntime,
  clearThreadCues, checkpoints: () => checkpoints, pairingRegistry, persistPairingSessions,
  spineHost: () => spineHost, providerStatusRuntime: () => providerStatusRuntime, loopHost: () => loopHost,
  questionPersistence: () => questionPersistence,
})

function botControlRuntime() {
  return {
    abortRoom: (roomId: string) => roomAborters.get(roomId)?.abort(),
    hasActiveRoom: (roomId: string) => roomAborters.has(roomId),
    emitRoom,
    rememberConnectorEvents,
  }
}

function mcpToolContext() {
  return {
    mcpTools: mcpHost.toolDefinitions(),
    callMcpTool: (callName: string, args: Record<string, any>) => mcpHost.callTool(callName, args),
  }
}

let toolContextRuntime: ToolContextRuntime

function runReadOnlySubAgent(...args: Parameters<ToolContextRuntime['runReadOnlySubAgent']>) { return toolContextRuntime.runReadOnlySubAgent(...args) }
function fullToolContext(...args: Parameters<ToolContextRuntime['fullToolContext']>) { return toolContextRuntime.fullToolContext(...args) }

observerHub = createObserverHub()

const projectSkills = createProjectSkillsRuntime({ loadTrust: loadSkillTrust, saveTrust: saveSkillTrust })

toolContextRuntime = createToolContextRuntime({
  sshManager,
  persistSubAgents,
  persistThreads,
  applyEventToThread,
  emitToRenderer,
  emitPluginEvent: (event) => { void hostPluginRuntime.emitAgentEvent(event) },
  renderProjectMemoryBucket,
  recordEmailDraft,
  recordEmailEvent,
  recordCalendarEvent,
  recordMessageEvent,
  runBrowserAutomation,
  mcpToolContext,
  cancelThreadRun,
  projectSkills: (projectId, workspace) => projectSkills.cachedProjectSkills(projectId, workspace),
  observe: (kind, payload) => observerHub.emit(kind, { payload }),
})

const approvalRuntime = createApprovalRuntime({
  pendingApprovals,
  pendingApprovalMeta,
  emitJob,
  emitToRenderer,
  findApprovalRule,
  patchJob,
  persistJobs,
  persistThreads,

  onApprovalPending: ({ threadId, tool }) => runAttention.approvalPending({ threadId, tool }),
})

function persistJobApproval(...args: Parameters<typeof approvalRuntime.persistJobApproval>) { return approvalRuntime.persistJobApproval(...args) }
function resolvePersistedJobApproval(...args: Parameters<typeof approvalRuntime.resolvePersistedJobApproval>) { return approvalRuntime.resolvePersistedJobApproval(...args) }
function approvalResolver(...args: Parameters<typeof approvalRuntime.approvalResolver>) { return approvalRuntime.approvalResolver(...args) }

let ticketWorkerRuntime: TicketWorkerRuntime
function startTicketWork(input: TicketWorkerInput): Promise<Thread> {
  return ticketWorkerRuntime.startTicketWork(input)
}
ticketWorkerRuntime = createTicketWorkerRuntime({
  persistThreads,
  persistTickets,
  emitTicket,
  appendRoomSupervisorEvent: roomJobApi.appendRoomSupervisorEvent,
  claimJob: roomJobApi.claimJob,
  patchJob,
  fullToolContext,
  drainThreadCues,
  waitForThreadCue,
  approvalResolver,
  timelineErrorSince,
  timelineError,
  isHighRiskTool,
  emitToRenderer,
  addThreadNotice,
  auditModelRoute,
  markRouteErrorRecovered,
  injectThreadRouteHandoff,
  aborters,
  workspaceMutationGuard,
  applyEventToThread,
  activeJob: roomJobApi.activeJob,
  finishJob: roomJobApi.finishJob,
  markUndrainedCues,
  effectiveContextLimit,
})

let botPollingRuntime: BotPollingRuntime

const botPollingApi = {
  startBotControlPolling: () => botPollingRuntime.startBotControlPolling(),
  pollTelegramUpdates: (...args: Parameters<BotPollingRuntime['pollTelegramUpdates']>) => botPollingRuntime.pollTelegramUpdates(...args),
  stopBotControlPolling: () => botPollingRuntime.stopBotControlPolling(),
}
const telegramRemoteRuntime = createTelegramRemoteRuntime({
  dispatchTurn: createTelegramRemoteDispatch({
    settings: () => store.settings, threads: () => store.threads, rooms: () => store.rooms, aborters,
    requireOperator: () => requireSecurityRole('operator', 'run Telegram remote assistant requests'),
    startThreadRun: (payload, admission) => startThreadRun(payload, undefined, admission), streamChat,
    steerRoom: (roomId, text) => {
      steerExistingRoom({
        rooms: () => store.rooms, requireOperator: () => requireSecurityRole('operator', 'steer rooms'),
        createId: randomUUID, persistRooms, emitRoom, hasActiveRoom: (id) => roomAborters.has(id)
      }, roomId, text)
    },
  })
})
botPollingRuntime = createBotPollingRuntime({
  botControlRuntime,
  telegramRemote: telegramRemoteRuntime,

  resolveChannelApproval: (token, decision) => connectorRuntime.resolveChannelApproval(token, decision),
  resolveChannelApprovalByThread: (threadKey, decision) => connectorRuntime.resolveChannelApprovalByThread(threadKey, decision),
})

let bootSchedulerLease: SchedulerLease | null = null
automationRuntime = createAutomationRuntime({
  threadJobHeartbeatMs: THREAD_JOB_HEARTBEAT_MS,
  claimJob: roomJobApi.claimJob,
  patchJob,
  heartbeatJob: roomJobApi.heartbeatJob,
  finishJob: roomJobApi.finishJob,
  emitAutomation,
  effectiveContextLimit,
  fullToolContext,
  applyEventToThread,
  timelineErrorSince,
  timelineError,
  isHighRiskTool,
  rememberConnectorEvents,

  loadSchedulerLease: () => bootSchedulerLease,
  saveSchedulerLease: (lease) => { bootSchedulerLease = lease; void saveSchedulerLease(lease).catch(() => { }) },
})
registerAutomationIpc(automationRuntime)

const attachmentTextService = registerApplicationWorkspaceContent({ aborters, workspaceMutationGuard })

let applicationShutdown: ReturnType<typeof registerApplicationShutdown> | undefined
const { appWindowApi, desktopModeRuntime } = createApplicationDesktop({
  baseDir: __dirname, sshManager, aborters, workspaceMutationGuard,
  isDraftQuitSettled: () => applicationShutdown?.isDraftQuitSettled() ?? false,
  getCoreIpc: () => coreIpc,
})

let threadRuntime: ThreadRuntime

function startThreadRun(payload: StartRunPayload, existingJob?: Job, admission?: ThreadRunAdmission): Promise<{ runId: string }> {
  observerHub.emit('session.start', { threadId: payload.threadId, payload: { hasAttachments: !!payload.attachments?.length } })
  return threadRuntime.startThreadRun(payload, existingJob, admission)
}

function cancelThreadRun(...args: Parameters<ThreadRuntime['cancelThreadRun']>) { return threadRuntime.cancelThreadRun(...args) }
function guideThreadRun(...args: Parameters<ThreadRuntime['guideThreadRun']>) { return threadRuntime.guideThreadRun(...args) }
function drainThreadCues(...args: Parameters<ThreadRuntime['drainThreadCues']>) { return threadRuntime.drainThreadCues(...args) }
function waitForThreadCue(...args: Parameters<ThreadRuntime['waitForThreadCue']>) { return threadRuntime.waitForThreadCue(...args) }
function markUndrainedCues(...args: Parameters<ThreadRuntime['markUndrainedCues']>) { return threadRuntime.markUndrainedCues(...args) }
function addThreadNotice(...args: Parameters<ThreadRuntime['addThreadNotice']>) { return threadRuntime.addThreadNotice(...args) }
function timelineErrorSince(...args: Parameters<ThreadRuntime['timelineErrorSince']>) { return threadRuntime.timelineErrorSince(...args) }
function timelineError(...args: Parameters<ThreadRuntime['timelineError']>) { return threadRuntime.timelineError(...args) }
function markRouteErrorRecovered(...args: Parameters<ThreadRuntime['markRouteErrorRecovered']>) { return threadRuntime.markRouteErrorRecovered(...args) }
function injectThreadRouteHandoff(...args: Parameters<ThreadRuntime['injectThreadRouteHandoff']>) { return threadRuntime.injectThreadRouteHandoff(...args) }
function roomRouteHandoff(...args: Parameters<ThreadRuntime['roomRouteHandoff']>) { return threadRuntime.roomRouteHandoff(...args) }
function applyEventToThread(...args: Parameters<ThreadRuntime['applyEventToThread']>) { return threadRuntime.applyEventToThread(...args) }
function clearThreadCues() { return threadRuntime.clearCues() }

const { providerStatusRuntime, checkpoints, recordProviderRateLimits } = createApplicationThreadPersistence(sshManager)

threadRuntime = createThreadRuntime({
  refreshProjectSkills: (project) => projectSkills.refreshProjectSkills(project),
  observe: (kind, payload) => observerHub.emit(kind, { payload }),
  aborters,
  workspaceMutationGuard,
  pendingApprovalMeta,
  jobLeaseOwner: JOB_LEASE_OWNER,
  threadJobHeartbeatMs: THREAD_JOB_HEARTBEAT_MS,
  sshManager,
  emitToRenderer,
  persistThreads,
  fullToolContext,
  approvalResolver,
  isHighRiskTool,
  claimJob: roomJobApi.claimJob,
  patchJob,
  replaceJobResult,
  heartbeatJob: roomJobApi.heartbeatJob,
  settleThreadRunJob: jobControlApi.settleThreadRunJob,
  renderProjectMemoryBucket,
  effectiveContextLimit,
  captureCheckpoint: (threadId, turn, target) => checkpoints.capture(threadId, turn, target),
  recordTurnCheckpoint: (threadId, turn, conversation, fileSha) => checkpoints.recordTurn(threadId, turn, conversation as any, fileSha),
  recordProviderRateLimits: ({ providerId, accountId, snapshot }) => recordProviderRateLimits(providerId, accountId, snapshot),
})

registerThreadHandoffIpc({
  spine: () => spineHost,
  sendToRenderer,
  remoteServer,
  startThreadRun,
  persistThreads,
})
registerPlanningGuardIpc()

const loopHost = createLoopHost({ startThreadRun: (p) => startThreadRun(p), workspaceMutationGuard })
void loopHost.init()

let workspaceRuntimeIpc: WorkspaceRuntimeIpc

const workspaceApi = {
  ensureProjectInstructionFiles: (...args: Parameters<WorkspaceRuntimeIpc['ensureProjectInstructionFiles']>) => workspaceRuntimeIpc.ensureProjectInstructionFiles(...args),
  upgradeGeneratedClaudeGuidance: (...args: Parameters<WorkspaceRuntimeIpc['upgradeGeneratedClaudeGuidance']>) => workspaceRuntimeIpc.upgradeGeneratedClaudeGuidance(...args),
  cancelRoomRun: (...args: Parameters<WorkspaceRuntimeIpc['cancelRoomRun']>) => workspaceRuntimeIpc.cancelRoomRun(...args),
  startRoomRun: (...args: Parameters<WorkspaceRuntimeIpc['startRoomRun']>) => workspaceRuntimeIpc.startRoomRun(...args),
}

jobControlRuntime = createJobControlRuntime(buildJobControlConfig({
  observerHub,
  spineHost: () => spineHost,
  handle,
  persistThreads,
  persistMissions,
  persistJobs,
  emitMission,
  emitJob,
  rememberJob: roomJobApi.rememberJob,
  patchJob,
  finishJob: roomJobApi.finishJob,
  replaceMission,
  getMission,
  runMission: missionApi.runMission,
  runAutomation: automationApi.runAutomation,
  startRoomRun: workspaceApi.startRoomRun,
  activeJob: roomJobApi.activeJob,
  missionAborters,
  missionApprovalResolvers,
  replaceJobResult,
  startThreadRun,
  effectiveContextLimit,
  pendingApprovals,
  pendingApprovalMeta,
  rememberApprovalRule,
  resolvePersistedJobApproval,
  emitToRenderer,
  fullToolContext,
  mcpToolContext,
  isHighRiskTool,
  aborters,
  workspaceMutationGuard,
  sshManager,
  checkForAppUpdate,
  openUpdateDownload,
}))

jobControlRuntime.startRoomWakeSweeper()

const applyTheme = createApplyTheme(() => store.settings)
const ensureWindowVisible = createEnsureWindowVisible({ installSessionHardening: appWindowApi.installSessionHardening, createWindow: appWindowApi.createWindow })
const handleBootFailure = createHandleBootFailure({
  ensureWindowVisible,
  getSettingsRef: () => store,
  setSettings: (settings) => { store.settings = settings },
  defaultSettings,
})

const coreIpc = registerCoreIpc(buildCoreIpcConfig({
  clipboard,
  remoteServer,
  sshManager,
  mcpHost,
  aborters,
  roomAborters,
  abortAndWaitMissionsForArchive,
  loopHost,
  killProjectTerminals,
  workspaceMutationGuard,
  purgeLocalMemory,
  confirmSensitiveAction,
  normalizeRemoteOrchestratorDestinations,
  ensureRemoteControlToken,
  applyTheme,
  applyPowerSettings,
  applyDesktopMode: () => { desktopModeRuntime.applySettings(); ensureApplicationEditMenu(Menu, (options) => new MenuItem(options), store.settings?.uiPreferences?.locale) },
  hideDesktopWindow: () => desktopModeRuntime.hide(),
  startUpdateChecks,
  startBotControlPolling: botPollingApi.startBotControlPolling,
  webhookServer,
  reloadPlugins: loadConfiguredPlugins,
  pollTelegramUpdates: botPollingApi.pollTelegramUpdates,
  registerConnectorIpc,
  getMission,
  createLocalPrLoop: jobControlApi.createLocalPrLoop,
  emitToRenderer,
  rememberConnectorEvents,
  appName: APP_NAME,
  spineHost: () => spineHost,
}))

  ; ({ workspaceRuntimeIpc, remoteHostApi } = createApplicationWorkspace({
    aborters, roomAborters, loopHost, workspaceMutationGuard, emitRoom, emitTicket, emitJob, patchJob,
    startTicketWork, runReadOnlySubAgent, roomRouteHandoff, ticketWorkerRuntime, webhookServer,
    rememberConnectorEvents, DEFAULT_ROOM_WORKER_BUDGET, sshManager, checkpoints,
    spineHost: () => spineHost, remoteServer, abortAndWaitMissionsForArchive, roomJobApi, workspaceApi,
    startThreadRun, guideThreadRun, cancelThreadRun,
  }))

registerCommandsIpc()
mobileBridgeRuntime.registerIpc()
terminalRemoteApi = registerTerminalIpc({ remoteSink: (ev) => remoteServer.publish({ kind: 'terminal', data: ev }), workspaceMutationGuard }).remoteApi

registerMiscIpc({
  projectSkills,
  observerHub,
  providerStatus: providerStatusRuntime,
  cancelThreadRun,
  guideThreadRun,
  startThreadRun,
  loopHost,
  persistArtifacts,
})

registerExportIpc()

registerIncidentIpc({
  effectiveContextLimit,
  fullToolContext,
  applyEventToThread,
  replaceMission,
  getMission,
  runMission: missionApi.runMission,
  emitIncident,
})

const { phones, botVms, userQuestions, questionPersistence } = initializeApplicationDevices({
  aborters, workspaceMutationGuard, cancelThreadRun, emitToRenderer,
})

registerApplicationWorkspaceServices({
  aborters, roomAborters, missionAborters, runningMissions, pendingApprovals, pendingApprovalMeta,
  missionApprovalResolvers, mobileBridgeRuntime, workspaceMutationGuard,
})

bootApplication({
  pairingRegistry, persistPairingSessions, checkpoints, providerStatusRuntime, mcpHost, remoteServer, webhookServer,
  upgradeGeneratedClaudeGuidance: workspaceApi.upgradeGeneratedClaudeGuidance,
  reconcileStuckJobs: roomJobApi.reconcileStuckJobs,
  installSessionHardening: appWindowApi.installSessionHardening, createWindow: appWindowApi.createWindow,
  initializeDesktopMode: () => desktopModeRuntime.applySettings({ startup: true }),
  showDesktopWindow: () => desktopModeRuntime.show(), applyTheme,
  startScheduler: automationApi.startScheduler, startBotControlPolling: botPollingApi.startBotControlPolling,
  resumeAutoJobsAfterRestart: jobControlApi.resumeAutoJobsAfterRestart,
  getSpineHost: () => spineHost, setSpineHost: (host) => { spineHost = host },
  setBootSchedulerLease: (lease) => { bootSchedulerLease = lease },
}).then(async () => { await userQuestions?.recoverInterrupted(store.threads); await mobileBridgeRuntime.initialize(); if (!applicationShutdown?.isQuitting()) void connectorRuntime.resumeWhatsAppAccounts().catch((error) => { void auditEvent('whatsapp.resume_failed', { message: String(error instanceof Error ? error.message : 'Saved accounts could not resume').slice(0, 220) }) }) }).catch(handleBootFailure)

registerApplicationErrors()

applicationShutdown = registerApplicationShutdown({
  aborters, roomAborters, missionAborters, workspaceMutationGuard, automationRuntime,
  telegramRemoteRuntime, botPollingApi, desktopModeRuntime, attachmentTextService,
  phones, botVms, userQuestions, questionPersistence, mcpHost, webhookServer, mobileBridgeRuntime, remoteServer,
})
