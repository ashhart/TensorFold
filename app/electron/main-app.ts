import { assertClusterWorkspaceIdle, quiesceClusterWorkspace, loadClusterWorkspaceIdleState } from './main/host/cluster-runtime'
import { registerInputHistoryIpc } from './main/host/ipc/input-history-ipc'
import { registerMessageQuoteIpc } from './main/host/ipc/message-quote-ipc'
import { registerModelHelperRuntime, quiesceModelHelpers, resumeModelHelpers } from './main/host/model-helper-runtime'
import { registerModelQualificationRuntime, quiesceModelQualifications, resumeModelQualifications, purgeModelQualifications } from './main/host/model-qualification-runtime'
import { recoverWorkspaceBackupCheckpoints } from './main/host/workspace-backup/recovery'
import type { BackupHostState } from './main/host/workspace-backup/repository'
import { reviewedRunOutcome } from './agent/run-outcome-review'
import type { RunOutcomeReviewInput } from '../shared/run-recovery-api'
import { registerWorkspaceBackupRuntime } from './main/host/workspace-backup/runtime'
import { readProtectedJson, writeProtectedJson } from './store'
import { beginComposerDraftPurge, endComposerDraftPurge } from './store/composer-drafts'
import { normalizeAgentPresets } from '../shared/bot-policy'
import { registerReadableExportRuntime } from './main/host/readable-export-runtime'
import { registerDocumentLibraryRuntime, quiesceDocumentLibrary, resumeDocumentLibrary, purgeDocumentLibrary } from './main/host/document-library-runtime'
import { registerConversationSearchRuntime } from './main/host/conversation-search-runtime'
import { recoverInterruptedConnectorDeliveries } from './main/host/connector-outbound-delivery'
import { stopAllWhatsAppBridges } from './agent/whatsapp-bridge'
import { app, BrowserWindow, shell, session, clipboard, Notification, dialog, Tray, Menu, MenuItem } from 'electron'
import path from 'node:path'
import { promises as fs, statSync } from 'node:fs'
import os from 'node:os'
import { spawn } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { defaultSettings, ensureTensorFoldStudioHome, loadSettings, saveSettings, loadThreads, saveThreads, saveThreadQuestion, loadProjects, saveProjects, loadRooms, saveRooms, loadTickets, saveTickets, loadAutomations, saveAutomations, loadConnectorEvents, saveConnectorEvents, loadContactDirectory, loadCliSessions, loadMissions, saveMissions, loadJobs, saveJobs, loadSubAgents, saveSubAgents, loadSkills, saveSkills, loadCommands, saveCommands, loadSuggestions, loadMcpServers, saveMcpServers, loadBotControlSessions, saveBotControlSessions, loadProjectMemory, saveProjectMemory, loadProjectMemoryCandidates, saveProjectMemoryCandidates, loadTurnCheckpoints, saveTurnCheckpoints, loadPairingSessions, savePairingSessions, loadSkillTrust, saveSkillTrust, loadArtifacts, loadProviderStatus, saveProviderStatus, loadSchedulerLease, saveSchedulerLease, loadWebhooks, saveWebhooks, makeRemoteControlToken, purgeLegacyUserDataState, safeSegment, vontraPath, writeTensorFoldStudioJson, writeTensorFoldStudioProtectedJson, writeTensorFoldStudioStateJson } from './store'
import { runAgent, compactThread } from './agent/loop'
import { executeTool } from './agent/tools'
import { estimateTokens, kfmt, renderTranscript } from './agent/compaction'
import { runRoom } from './agent/rooms'
import { roomModeDefaults } from './agent/room-outcomes'
import { DEFAULT_ROOM_COLLABORATION_MODE, inferSeatLane } from './agent/room-collaboration'
import { resolveProviderModel, selectProviderAccount, surfaceModelOverride } from './agent/provider-accounts'
import { quickComplete, streamChat } from './agent/openai'
import { discoverLocalProvider } from './agent/local-providers'
import { collectTokenUsage } from './agent/token-usage'
import { buildContextPressureSnapshot } from './agent/context-pressure'
import { recordConnectorInteraction } from './agent/connector-spine'
import { stampSettingsSecretRotations } from './agent/secret-rotation'
import { cleanupMediaCache } from './agent/media-cache'
import {
  addMissionCheckpoint,
  appendMissionJournal,
  buildMissionInspectionSnapshot,
  buildMissionRuntimeSnapshot,
  cancelMission as cancelMissionState,
  completeMission,
  createMission,
  evaluateMissionHeartbeat,
  failMission,
  recordMissionHeartbeat,
  requestMissionApproval,
  resolveMissionApproval,
  startMission,
} from './agent/missions'
import { buildRoomSupervisorJobMetadata, cancelJob, claimJobLease, completeJob, deadLetterJob, detectStuckJobs, enqueueJob, failJob, heartbeatJobLease, markRemoteJobsUnverifiable, recordJobApproval, releaseJobLease, resolveJobApproval, resumeJob, restartPolicyAutoResumes, restartResumeApprovalMode, retryJob } from './agent/jobs'
import type { JobPatch, RoomSupervisorMetadataPatch } from './agent/jobs'
import { appendSubAgentEvent, completeSubAgent, createSubAgentFromLaunch, failSubAgent, finalSubAgentText, parentForRoomSeat, parentForThread, startSubAgent, summarizeSubAgentTimeline } from './agent/subagents'
import { normalizeTicketCapabilities, normalizeTicketEvidence, normalizeTicketExecution, normalizeStringList, roomFinalGateBlockers, roomLeftToWork, roomTicketConflictBlockers, summarizeRoomLeftToWork, ticketFileRefs, ticketNeedsReviewAfterBuilder } from './agent/tickets'
import { buildFrontendMissionEvidence, buildMissionRunPrompt, decideNextMissionStep, missionNeedsFrontendUiSmoke, missionRunCount, missionRunLimit, shouldStartFreshMissionThread } from './agent/mission-supervisor'
import { createMcpServer, McpHost, normalizeServerConfig } from './agent/mcp-client'
import { checkUpdateFeed, checkingUpdateStatus, emptyUpdateStatus } from './agent/update-checker'
import { createConnectorFetch } from './agent/secure-fetch'
import { sendWhatsAppCloudReply } from './main/host/whatsapp-reply'
import { confirmSensitiveAction } from './main/host/sensitive-action'
import { reconcileJobsAndMissionsAfterRestart, reconcileIncidentsAfterRestart } from './main/host/reconcile-after-restart'
import { createApplyTheme, createEnsureWindowVisible, createHandleBootFailure, migrateLegacyUserData, normalizeRemoteOrchestratorDestinations } from './main/host/boot-helpers'
import { bootAppState } from './main/host/boot-sequence'
import { createEmitHelpers } from './main/host/emit-helpers'
import { createRunAttention } from './main/host/run-attention'
import { createThreadRevertApply, createThreadForkFactory } from './main/host/thread-revert-fork'
import { buildRemoteControlConfig } from './main/host/remote-control-config'
import { buildWorkspaceRuntimeIpcConfig } from './main/host/workspace-runtime-config'
import { buildControllerStatus, captureControllerSnapshot, requestControllerAccess, runControllerAction } from './agent/controller'
import { auditPreview, assertProviderEndpoint, assertUpdateFeedUrl, buildSessionCsp, currentSecurityRoleFromEnv, securityRoleForLaunch, isSafeExternalUrl, isPrivateHost, mergeMcpServerSecrets, mergeSettingsSecrets, redactSettings } from './agent/security'
import { ensureTensorFoldStudioStylePrompt } from './agent/prompt-style'
import { createWebhookServer } from './agent/webhook-server'
import { createRemoteControlServer } from './agent/remote-control'
import { buildLocalPrBuilderResumePrompt, buildLocalPrLoopDraft, buildLocalPrLoopSummary, buildLocalPrReviewPrompt, extractLocalPrReviewOutcome, localPrApprovalEvidence, localPrRemotePublishGate, shouldAutoStartLocalPrReview, shouldResumeBuilderFromReview } from './agent/pr-loop'
import { approvalPolicyScope, type ApprovalPolicyScope } from './agent/approval-policy'
import type { ApprovalRuleContext } from './agent/approval-rules'
import type { ModelRouteDecision } from './agent/model-router'
import { approveProjectMemoryCandidate, createProjectMemoryCandidate, detectProjectMemoryWarnings, forgetProjectMemoryEntry, formatProjectMemoryEntryForPrompt, formatProjectMemoryWarnings, projectMemoryCandidateDiff, rejectProjectMemoryCandidate, rollbackProjectMemoryEntry, scopeProjectMemoryCandidates, scopeProjectMemoryEntries, type ProjectMemorySourceSnapshot } from './agent/project-memory'
import { ensureTimelineItemTrace } from './agent/trace'
import type { ControllerActionInput } from './agent/controller-actions'
import type { AgentCue, AgentEvent, ApiMessage, CueRunPayload, CueRunResult, Settings, StartRunPayload, Thread, TimelineItem, Project, Room, RoomEventItem, RoomStreamEvent, Ticket, TicketRemoteLifecycleCommand, TicketWorkOptions, Automation, AutomationRun, ConnectorEvent, Provider, ProviderDiscoveryKind, AppUpdateStatus, Mission, Job, JobPendingApproval, JobApprovalDecision, SubAgent, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, MissionStatus, MissionCheckpoint, MissionJournalEvent, MissionPendingApproval, MissionBudgetSnapshot, Skill, Command, McpServerConfig, BotControlPollResult, BotControlSession, Seat, GitRepositoryState, ControllerSnapshot, ControllerStatus, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookEndpoint, RemoteControlSnapshot, RemoteConnectionTestResult, RemoteDirectoryListResult, RemoteOrchestratorSnapshotResult, SecurityPurgeResult, SecurityRole, SshHost, Attachment } from './agent/types'
import type { RoomManualMemoryCommand } from './agent/types'
// Phase 6/7 (incident mode + run export) — kept as separate import lines so they
// merge additively alongside the parallel Codex lane's import edits.
import { loadIncidents, saveIncidents, vontraHomeDir } from './store'
import { createEventSpineHost, type EventSpineHost } from './agent/event-spine-host'

import { createPairingRegistry } from './agent/pairing-sessions'
import type { Incident, CreateIncidentInput, IncidentMitigation, IncidentEventKind } from './agent/types'
import { store } from './main/state'
import { handle } from './main/host/ipc'
import { persistArtifacts, persistAutomations, persistBotSessions, persistCommands, persistConnectorEvents, persistIncidents, persistJobs, persistMcpServers, persistMissions, persistProjectMemory, persistProjectMemoryCandidates, persistProjects, persistRooms, persistSkills, persistSubAgents, persistThreads, persistTickets, persistWebhooks, persistSettings, persistSettingsCritical, recordPersistenceFailure, preserveStateForUpdate } from './main/host/persistence'
import { auditActor, auditEvent, requireSecurityRole } from './main/host/audit'
import { applyPowerSettings, stopCaffeinate } from './main/host/power'
import { openExternalSafely } from './main/host/external'
import { checkForAppUpdate, configureUpdateInstallSafety, openUpdateDownload, registerUpdateInstallIpc, startUpdateChecks } from './main/host/updates'
import { assertUpdateQuiescent } from './main/host/update-quiescence'
import { archiveEntity, isActive } from './main/host/archive'
import { archiveProjectState, archiveRoomState, archiveThreadState, archiveTicketState, restoreProjectState, restoreRoomState, restoreThreadState, restoreTicketState, ticketArchiveClosure } from './main/host/archive-state'
import { SshManager } from './main/host/ssh'
import { upsertTrustedHostKey } from './agent/ssh-connection'
import { ensureCliPath } from './agent/cli-path'
import { localMemoryCounts, securityControlSnapshot } from './main/host/security-controls'
import { deliverOutboundWebhook, emitOutboundWebhooks, recordWebhookDelivery, webhookSnapshot as redactedWebhookSnapshot } from './main/host/webhooks'
import { registerWebhookIpc } from './main/host/webhook-ipc'
import { closeBrowserAutomationSessions, purgeBrowserAutomationSessions, runBrowserAutomation } from './main/host/browser-automation'
import { botChannelAllowed, ingestBotControl } from './main/host/bot-control'
import { createConnectorRuntime } from './main/host/connector-runtime'
import { registerExportIpc } from './main/host/export-ipc'
import { registerWorkspaceIpc } from './main/host/workspace-ipc'
import { WorkspaceMutationGuard } from './main/host/workspace-mutation-guard'
import { createCheckpointService } from './main/host/checkpoint-runtime'
import { appendRoomSteeringEvent, applyRoomManualMemoryCommand, assertRoomConfigUpdateAllowed, assertRoomSeatRoutesConfigured, assertTicketUserUpdateAllowed, pickRoomConfigPatch, pickRoomManualMemoryCommand, pickTicketCreateInput, pickTicketRemoteLifecycleCommand, pickTicketUserPatch, ticketRunEvidence } from './main/host/room-ticket-patch-policy'
import type { SchedulerLease } from './agent/automation-scheduler'
import { registerCommandsIpc } from './main/host/commands-ipc'
import { killAllTerminals, killProjectTerminals, liveTerminalCount, registerTerminalIpc, type TerminalRemoteApi } from './main/host/terminal-runtime'
import { createMobileBridgeRuntime } from './main/host/mobile-bridge-runtime'
import { createWatchCommandProjection } from './main/host/watch-command-projection'
import { createWatchSpeechRuntime } from './main/host/watch-speech-runtime'
import { getManagedKokoroService } from './main/host/ipc/provider-ipc'
import { closeAllCodexAppServerSessions } from './agent/codex-app-server'
import { createAutomationRuntime, registerAutomationIpc, type AutomationRuntime, type AutomationRunSource } from './main/host/automations'
import { createMissionRuntime, type MissionRuntime } from './main/host/mission-runtime'
import { createJobMissionCore } from './main/host/job-mission-core'
import { createProviderStatusRuntime } from './main/host/provider-status-runtime'
import { registerMiscIpc } from './main/host/misc-ipc'
import { registerThreadHandoffIpc } from './main/host/handoff-ipc'
import { registerPlanningGuardIpc } from './main/host/planning-guard-ipc'
import { createRemoteHostApi } from './main/host/remote-host-api'
import { enableOrchestrationJournal, quiesceOrchestrationJournal } from './agent/orchestration-journal'
import { wireOrchestrationEventsToWindow, type OrchestrationEvent } from './main/host/orchestration-events'
import { registerCoreIpc } from './main/host/core-ipc'
import { createDesktopModeRuntime } from './main/host/desktop-mode'
import { desktopTrayIcon } from './main/host/desktop-tray-icon'
import { buildCoreIpcConfig } from './main/host/core-ipc-config'
import { createLocalPrRuntime, type LocalPrRuntime } from './main/host/local-pr-runtime'
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
import { normalizeArtifactRecord } from './agent/artifacts'
import { createAppWindowRuntime } from './main/host/app-window'
import { rendererWindowAvailable, sendToRenderer } from './main/host/renderer-events'
import { headlessBootstrapMessages, prepareHeadlessRemoteControl } from './main/host/headless-bootstrap'
import { effectiveContextLimit, parseJsonLoose, roomContextLimit } from './main/host/context-limit'
import { createSecurityPurgeRuntime, buildSecurityPurgeConfig } from './main/host/security-purge-runtime'
import { createApprovalRuntime, isHighRiskTool, type JobHandle } from './main/host/approval-runtime'
import { createDurableApprovalRuleStore, type ApprovalRuleDetails } from './main/host/approval-rule-store'
import { durablePairingMutation } from './main/host/durable-pairing'
import { registerIncidentIpc } from './main/host/incidents-ipc'
import { createEngineRuntime, engineService } from './main/host/engine-runtime'
import { createLearningQuitGuard } from './main/host/learning-quit'
import { shutdownComputerUse, computerHelperPath } from './main/host/computer-use-host'
import { installNativeTextEditing, ensureApplicationEditMenu } from './main/host/native-text-editing'
import { registerComposerDraftIpc, requestRendererDraftFlush, observeComposerDraftRenderer } from './main/host/ipc/composer-draft-ipc'
import { createAttachmentTextService } from './main/host/attachment-text'
import { registerEngineIpc } from './main/host/ipc/engine-ipc'
import { registerModelBenchmarkIpc } from './main/host/ipc/model-benchmark-ipc'
import { initializePhoneHost, phoneHost } from './main/host/phone-host'
import { registerPhoneIpc } from './main/host/ipc/phone-ipc'
import { initializeBotVmHost, getBotVmHost } from './main/host/bot-vm-host'
import { registerBotVmIpc } from './main/host/ipc/bot-vm-ipc'
import { assertBotLifecycleIdle } from './main/host/bot-run-boundary'
import { captureBotConfig } from '../shared/bot-policy'
import { isRestrictedBot } from '../shared/bot-tool-policy'
import { initializeUserQuestionHost, getUserQuestionHost } from './main/host/user-question-host'
import { createUserQuestionPersistence } from './main/host/user-question-persistence'
import { registerUserQuestionIpc } from './main/host/ipc/user-question-ipc'
import { isExternalProvider } from '../shared/provider-trust'
import { assertTicketRelations, describeTicketForRoom, mkTicket, summarizeTicketWorkerOutcome, ticketEvidenceFromArgs } from './main/host/ticket-helpers'
import { createRoomSupervisorHelpers, type RoomSupervisorHelpers } from './main/host/room-supervisor-helpers'
import { ensureRoomWorktree } from './main/host/room-worktree'
import { createWorkspaceRuntimeIpc, type WorkspaceRuntimeIpc } from './main/host/workspace-runtime-ipc'
import { sanitizeProjectCreateInput, sanitizeProjectUpdatePatch, validateProjectProviderDefaults } from './main/host/project-patch-policy'
import { inspectGitRepository, qq, resolveWorkspaceCwd, sh, shStrict } from './main/host/workspace'
import {
  auditModelRetry,
  auditModelRoute,
  auditScopedModelRoute,
  listProviderModels,
  modelRouteAccountLabel,
  modelRouteAutoFailoverEnabled,
  modelRouteFailoverText,
  modelRouteMemoryForDecision,
  modelRouteResetPolicy,
  modelRouter,
  modelRouterCooldownSnapshot,
  rememberModelRouteHistory,
  rememberProviderModels,
  resolveProjectProvider,
  routeHandoffPreview,
  runtimeProvider,
  runtimeProviderWithFreshModels,
  selectedProviderAccountId,
  ticketWorkOptionsWithProjectRoute,
} from './main/host/provider-runtime'
import { APP_NAME, HEADLESS_MODE } from './main/early-electron-paths'

// The local desktop GUI is operated by the trusted machine owner, so it runs as
// 'owner' (may save settings + change policies). The headless/remote server keeps
// the safer 'operator' default. Set before any IPC handler runs, otherwise the
// owner-gated `settings:save` throws and settings silently never persist.
process.env.VONTRA_SECURITY_ROLE = securityRoleForLaunch(process.env, {
  headless: HEADLESS_MODE,
})

const connectorRuntime = createConnectorRuntime({
  findApprovalRule,
  rememberApprovalRule,
  remotePublish: (event) => remoteServer.publish({ kind: 'connector', data: event }),
  emitOutboundWebhook: (event) => emitOutboundWebhooks('connector', event),
  // The function is initialized later during main-process bootstrap; defer lookup until an
  // operator actually removes an account. Missing/destroyed window wiring fails closed.
  confirmSensitiveAction: (input) => confirmSensitiveAction(input),
  reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['connector:operation'] }),
})
const {
  connectorPolicyForEvent,
  rememberConnectorEvents,
  recordEmailDraft,
  recordEmailEvent,
  recordCalendarEvent,
  recordMessageEvent,
  registerConnectorIpc,
} = connectorRuntime

let automationRuntime: AutomationRuntime
// Lazy-accessor bundle: replaces the runAutomation + startScheduler forwarder shims.
const automationApi = {
  runAutomation: (...args: Parameters<AutomationRuntime['runAutomation']>) => automationRuntime.runAutomation(...args),
  startScheduler: () => automationRuntime.startScheduler(),
}

// Loopback webhook listener — disabled until a signed generic endpoint or fully configured
// Meta Cloud bot ingress is enabled.
const webhookServer = createWebhookServer({
  getEndpoints: () => store.webhooks,
  onConnectorEvent: (e) => rememberConnectorEvents([e]),
  runAutomationById: (id, source) => {
    const a = store.automations.find((x) => x.id === id)
    if (a) void automationApi.runAutomation(a, {
      trigger: 'webhook',
      connectorEventId: source?.eventId ?? null,
      connectorId: source?.connectorId ?? null,
      connectorAction: source?.action ?? null,
      threadKey: source?.threadKey ?? null,
    })
  },
  audit: (kind, data) => { void auditEvent(kind, data) },
  getWhatsAppSettings: () => store.settings.botControl.whatsapp,
  onWhatsAppBotInput: (input) => {
    const result = ingestBotControl(input, botControlRuntime())
    if (result.action === 'replay_ignored') return
    store.settings.botControl.whatsapp = { ...store.settings.botControl.whatsapp, lastEventAt: Date.now(), lastError: null }
    persistSettings('whatsappCloud.lastEventAt')
    void sendWhatsAppCloudReply(input.chatId, result.reply).catch(async (error) => {
      const message = String((error as any)?.message ?? error ?? 'WhatsApp Cloud reply failed.')
      store.settings.botControl.whatsapp = { ...store.settings.botControl.whatsapp, lastError: message.slice(0, 500) }
      persistSettings('whatsappCloud.lastError')
    })
  },
})


// The `*OnThisHost` REST-mutation layer (project/thread/room/ticket create/update/archive/
// restore + remote ticket execution + remote thread/room run entry points) lives in its own
// module (remote-host-api.ts). It is constructed lazily — see `remoteHostApi` below — because
// its collaborators (abort maps, loopHost, the thread/room run entry points, the remote server
// itself) are initialized later in the bootstrap. The remoteServer config object forwards
// requests to remoteHostApi.* at runtime, by which point it is assigned.
let remoteHostApi: ReturnType<typeof createRemoteHostApi>

// Set once registerTerminalIpc() runs (later in startup). The RC server's terminal deps reference
// it lazily — remote terminal requests only arrive at runtime, by which point it's assigned.
let terminalRemoteApi: TerminalRemoteApi | null = null
let jobControlRuntime: JobControlRuntime
// Lazy-accessor bundle: replaces 6 hoisted forwarder shims. Each arrow captures
// `jobControlRuntime` by reference and reads it at call time (after the assignment below).
const jobControlApi = {
  createLocalPrLoop: (...args: Parameters<JobControlRuntime['createLocalPrLoop']>) => jobControlRuntime.createLocalPrLoop(...args),
  settleThreadRunJob: (...args: Parameters<JobControlRuntime['settleThreadRunJob']>) => jobControlRuntime.settleThreadRunJob(...args),
  setMissionStatusLocal: (...args: Parameters<JobControlRuntime['setMissionStatusLocal']>) => jobControlRuntime.setMissionStatusLocal(...args),
  recordMissionHeartbeatLocal: (...args: Parameters<JobControlRuntime['recordMissionHeartbeatLocal']>) => jobControlRuntime.recordMissionHeartbeatLocal(...args),
  resumeAutoJobsAfterRestart: () => jobControlRuntime.resumeAutoJobsAfterRestart(),
  resolveDurableApprovalAfterRestart: (...args: Parameters<JobControlRuntime['resolveDurableApprovalAfterRestart']>) => jobControlRuntime.resolveDurableApprovalAfterRestart(...args),
}

// Per-device pairing + sessions (VST-035): a phone/laptop/browser pairs once and gets its
// own revocable session token, so the shared owner token isn't handed around. Persisted to
// the protected pairing-sessions.json (tokens are secrets) on every mutation and restored
// at app-ready, so pairings survive host restarts.
const pairingRegistry = createPairingRegistry()
async function persistPairingSessions(): Promise<void> {
  pairingRegistry.pruneExpiredLinks()
  try {
    await savePairingSessions(pairingRegistry.snapshot())
  } catch (error) {
    recordPersistenceFailure('pairingSessions', error)
    throw error
  }
}

const remoteServer = createRemoteControlServer(buildRemoteControlConfig({
  pairingRegistry,
  persistPairingSessions,
  // Forward-declared / later-assigned collaborators — resolved lazily at request time.
  spineHost: () => spineHost,
  remoteHostApi: () => remoteHostApi,
  jobControlRuntime: () => jobControlRuntime,
  terminalRemoteApi: () => terminalRemoteApi,
  // Mutable run maps — created once below this construction site; passed as accessors so the
  // config built here references the live collection identity by the time requests arrive.
  aborters: () => aborters,
  roomAborters: () => roomAborters,
  runningMissions: () => runningMissions,
  setMissionStatusLocal: jobControlApi.setMissionStatusLocal,
  recordMissionHeartbeatLocal: jobControlApi.recordMissionHeartbeatLocal,
}))
const mcpHost = new McpHost()

const mobileBridgeRuntime = createMobileBridgeRuntime({
  projectWatchSnapshot: createWatchCommandProjection({activeThreadIds: () => [...aborters.keys()]}),
  watchSpeech: createWatchSpeechRuntime(getManagedKokoroService, (id) => pairingRegistry.isWatchSpeechAllowed(id)),
  setWatchSpeech: (sessionId, enabled) => durablePairingMutation(pairingRegistry, () => pairingRegistry.setWatchSpeechAllowed(sessionId, enabled), () => savePairingSessions(pairingRegistry.snapshot()), { rollbackOnFailure: enabled }),
  profileRoot: () => vontraPath(),
  remoteServer,
  ownerToken: () => store.settings.remoteControl.authToken,
  reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['mobile:bridge'] }),
  authorizeSessionToken: (token) => pairingRegistry.authorizeSessionToken(token),
  enableLoopback: async () => {
    const current = store.settings
    const currentJson = JSON.stringify(current)
    const remoteControl = {
      ...current.remoteControl, enabled: true, bindHost: '127.0.0.1' as const, allowLan: false,
      authToken: current.remoteControl.authToken || makeRemoteControlToken(),
    }
    const next = stampSettingsSecretRotations({ ...current, remoteControl }, current)
    await saveSettings(next)
    if (store.settings !== current || JSON.stringify(store.settings) !== currentJson) {
      await saveSettings(store.settings)
      throw new Error('Settings changed while enabling the mobile connection. The current configuration was preserved; retry.')
    }
    store.settings = next
  },
  audit: (kind, data) => { void auditEvent(kind, data) },
})

async function loadConfiguredPlugins(settings: Settings): Promise<void> {
  hostPluginRuntime.clear()
  const configured = settings.capabilities?.packagedPlugins ?? []
  for (const plugin of configured) {
    // Plugins execute in the Electron main process. Loading therefore requires both
    // an explicit enable and the user's persisted trust decision.
    if (plugin.enabled !== true || plugin.trusted !== true || !plugin.manifestPath) continue
    const result = await hostPluginRuntime.loadPlugin(path.dirname(path.resolve(plugin.manifestPath)))
    if (!result.ok) console.error(`[plugin-runtime] Failed to load ${plugin.name || plugin.id}: ${result.error}`)
  }
}

// SSH remote-workspace manager: owns live sessions, bridges secret/trust prompts
// to the renderer, persists newly-trusted host keys, and supplies the agent loop
// with a remote WorkspaceBackend for connected hosts.
const sshManager = new SshManager({
  getHosts: () => store.settings.sshHosts ?? [],
  getTrustedKeys: () => store.settings.trustedSshHostKeys ?? [],
  interactivePrompts: () => !HEADLESS_MODE && rendererWindowAvailable(),
  saveTrustedKey: async key => { await coreIpc.saveTrustedKey(key) },
  sendEvent: (channel, payload) => { sendToRenderer(channel, payload) },
  onStateChange: (state) => {
    if (state.phase !== 'disconnected' && state.phase !== 'error') return
    const reason = state.lastError
      ? `Remote transport lost: ${state.lastError}`
      : 'Remote transport disconnected; process state cannot be verified.'
    const result = markRemoteJobsUnverifiable(store.jobs, state.hostId, reason)
    if (!result.changed) return
    store.jobs = result.jobs
    persistJobs()
    for (const job of result.affected) emitJob(job)
  },
})
store.updateStatus = emptyUpdateStatus(app.getVersion())

// Run bookkeeping
const aborters = new Map<string, AbortController>() // threadId -> controller
const workspaceMutationGuard = new WorkspaceMutationGuard()
const pendingApprovals = new Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>() // callId -> resolver
const pendingApprovalMeta = new Map<string, { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }>()





const approvalRuleStore = createDurableApprovalRuleStore({
  getRules: () => store.settings.approvalRules ?? [],
  setRules: (approvalRules) => { store.settings = { ...store.settings, approvalRules } },
  persist: (label) => persistSettingsCritical(label, store.settings),
  newId: () => randomUUID(),
  onSkipped: ({ scope, key, reason }) => auditEvent('approval.rule_skipped', { scope, key, reason }),
  onChanged: ({ action, scope, key, previous, next }) => auditEvent('approval.rule_changed', {
    action,
    scope,
    key,
    previous,
    next: {
      decision: next.decision,
      duration: next.duration,
      projectId: next.projectId,
      workspace: next.workspace,
      expiresAt: next.expiresAt,
      maxUses: next.maxUses,
      useCount: next.useCount,
    },
  }),
})

function findApprovalRule(scope: 'tool' | 'connector', key: string, context: ApprovalRuleContext = {}) {
  return approvalRuleStore.find(scope, key, context)
}

function rememberApprovalRule(
  scope: 'tool' | 'connector',
  key: string,
  decision: 'allow' | 'deny',
  duration: 'always' = 'always',
  details: ApprovalRuleDetails = {},
) {
  return approvalRuleStore.remember(scope, key, decision, duration, details)
}



const roomAborters = new Map<string, AbortController>()
// Observer hooks + ordered event spine are forward-declared here so the emit helpers factory
// below can close over them lazily; both are assigned later in the bootstrap (observerHub at
// its VST-012 construction, spineHost during app-ready spine restore).
let observerHub: ReturnType<typeof createObserverHub>
let spineHost: EventSpineHost | null = null

// Run notifications (click opens the thread) and the needs-you dock badge. Each notification stays
// referenced until clicked or closed so macOS can still deliver the click.
const liveNotifications = new Set<Notification>()
const runAttention = createRunAttention({
  isIncognito: (id) => {
    const thread = store.threads.find((thread) => thread.id === id)
    // Approval runtime calls attention directly, so admission must protect both
    // phone-derived titles and Incognito even when the event emitter is bypassed.
    return !thread || thread.incognito === true || thread.phoneLocalOnly === true || thread.googleLocalOnly === true
  },
  enabled: () => store.settings?.notifyRuns !== false,
  focused: () => { try { return !!store.win && !store.win.isDestroyed() && store.win.isFocused() } catch { return false } },
  // Headless hosts have no window to bring forward, so they never notify.
  supported: () => !!store.win && Notification.isSupported(),
  threadTitle: (id) => store.threads.find((t) => t.id === id)?.title,
  notify: (notice) => {
    const n = new Notification({ title: notice.title, body: notice.body })
    liveNotifications.add(n)
    n.on('click', () => {
      liveNotifications.delete(n)
      const win = store.win
      try { if (win && !win.isDestroyed()) { win.show(); win.focus() } } catch { /* best effort */ }
      sendToRenderer('app:open-thread', notice.threadId)
    })
    n.on('close', () => { liveNotifications.delete(n) })
    n.show()
  },
  setBadge: (text) => { app.dock?.setBadge(text) },
})

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
// Lazy-accessor bundle: replaces 14 hoisted forwarder shims. Each arrow captures
// `roomJobRuntime` by reference and reads it at call time (after the assignment below),
// exactly as the hoisted `function` shims did.
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

// ---- Mission executor: drive runAgent under a hardened headless posture ----
// Shared mutable substrate (running-missions set, abort map, approval-resolver bridge,
// job patch/replace helpers) lives in its own module so the mission/room-job/job-control/
// security-purge runtimes share a single identity for these collections.
const jobMissionCore = createJobMissionCore({ persistJobs, emitJob })
const runningMissions = jobMissionCore.runningMissions
const missionAborters = jobMissionCore.missionAborters
const missionApprovalResolvers = jobMissionCore.missionApprovalResolvers
const abortAndWaitMissionsForArchive = jobMissionCore.abortAndWaitMissionsForArchive
function patchJob(job: Job | null, patch: JobPatch): Job | null { return jobMissionCore.patchJob(job, patch) }
function replaceJobResult(result: { jobs: Job[]; job: Job | null }): Job { return jobMissionCore.replaceJobResult(result) }

let missionRuntime: MissionRuntime
// Lazy-accessor bundle: replaces the runMission forwarder shim.
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

const securityPurgeRuntime = createSecurityPurgeRuntime(buildSecurityPurgeConfig({
  quiesceDocumentLibrary: async () => { await Promise.all([quiesceDocumentLibrary(), quiesceModelQualifications(), quiesceModelHelpers(), quiesceClusterWorkspace()]) },
  resumeDocumentLibrary: () => { resumeDocumentLibrary(); resumeModelQualifications(); resumeModelHelpers() },
  purgeDocumentLibrary: async () => { await purgeDocumentLibrary(); await purgeModelQualifications() },
  quiescePhoneControl: async () => { await phoneHost()?.service.disconnect() },
  quiesceBotVms: async () => { await getBotVmHost()?.quiesce() },
  purgeBotVmData: async () => { await getBotVmHost()?.purgeOwnedData() },
  resumeBotVms: () => { getBotVmHost()?.resume() },
  quiesceUserQuestions: async () => { await getUserQuestionHost()?.quiesce(); await questionPersistence.drain() },
  resumeUserQuestions: () => { getUserQuestionHost()?.resume() },
  aborters,
  roomAborters,
  missionAborters,
  pendingApprovals,
  pendingApprovalMeta,
  missionApprovalResolvers,
  workspaceMutationGuard,
  automationRuntime: () => automationRuntime,
  clearThreadCues,
  checkpoints: () => checkpoints,
  pairingRegistry,
  persistPairingSessions,
  spineHost: () => spineHost,
  providerStatusRuntime: () => providerStatusRuntime,
  loopHost: () => loopHost,
  engineService,
}))
async function purgeLocalMemory(): Promise<SecurityPurgeResult> {
  requireSecurityRole('owner', 'purge local memory')
  assertClusterWorkspaceIdle()
  await quiesceClusterWorkspace()
  return securityPurgeRuntime.purgeLocalMemory()
}

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
// Hoisted forwarder shims: function declarations are hoisted so missionRuntime
// (constructed above) can reference them before toolContextRuntime is assigned.
function runReadOnlySubAgent(...args: Parameters<ToolContextRuntime['runReadOnlySubAgent']>) { return toolContextRuntime.runReadOnlySubAgent(...args) }
function fullToolContext(...args: Parameters<ToolContextRuntime['fullToolContext']>) { return toolContextRuntime.fullToolContext(...args) }
// Observer hooks (VST-012): read-only, fail-open observation of session/turn/api/tool/
// approval/subagent/remote events with sanitized bounded payloads.
observerHub = createObserverHub()

// Repo-authored project skills (VST-010): hash-pinned trust, scan cache for run contexts.
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

// Trust UI + review flow (VST-010), first-class artifacts (VST-011), observer health
// (VST-012), provider-rate-limit read, bare agent start/cue/cancel, loop controls, and
// window actions are registered together by registerMiscIpc() further below, once every
// dependency (loopHost, threadRuntime, providerStatusRuntime, …) is initialized.

const approvalRuntime = createApprovalRuntime({
  pendingApprovals,
  pendingApprovalMeta,
  emitJob,
  emitToRenderer,
  findApprovalRule,
  patchJob,
  persistJobs,
  persistThreads,
  // "Your turn": badge the dock and, in the background, notify; the click opens the thread.
  onApprovalPending: ({ threadId, tool }) => runAttention.approvalPending({ threadId, tool }),
})
// Lazy-accessor bundle: replaces the persistJobApproval / resolvePersistedJobApproval /
// approvalResolver forwarder shims.
// Hoisted forwarder shims: function declarations are hoisted so missionRuntime
// (constructed above) can reference them before approvalRuntime is assigned.
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
// Lazy-accessor bundle: replaces the startBotControlPolling / pollTelegramUpdates /
// stopBotControlPolling forwarder shims.
const botPollingApi = {
  startBotControlPolling: () => botPollingRuntime.startBotControlPolling(),
  pollTelegramUpdates: (...args: Parameters<BotPollingRuntime['pollTelegramUpdates']>) => botPollingRuntime.pollTelegramUpdates(...args),
  stopBotControlPolling: () => botPollingRuntime.stopBotControlPolling(),
}
const telegramRemoteRuntime = createTelegramRemoteRuntime({ dispatchTurn: createTelegramRemoteDispatch({
  settings: () => store.settings, threads: () => store.threads, rooms: () => store.rooms, aborters,
  requireOperator: () => requireSecurityRole('operator', 'run Telegram remote assistant requests'),
  startThreadRun: (payload, admission) => startThreadRun(payload, undefined, admission), streamChat,
  steerRoom: (roomId, text) => { steerExistingRoom({ rooms: () => store.rooms, requireOperator: () => requireSecurityRole('operator', 'steer rooms'),
    createId: randomUUID, persistRooms, emitRoom, hasActiveRoom: (id) => roomAborters.has(id) }, roomId, text) },
}) })
botPollingRuntime = createBotPollingRuntime({
  botControlRuntime,
  telegramRemote: telegramRemoteRuntime,
  // Inline-approval callbacks: a Telegram approve/deny button (token) or a "/approve" reply
  // resolves the parked in-channel approval before it's routed as a normal bot command.
  resolveChannelApproval: (token, decision) => connectorRuntime.resolveChannelApproval(token, decision),
  resolveChannelApprovalByThread: (threadKey, decision) => connectorRuntime.resolveChannelApprovalByThread(threadKey, decision),
})

// ---------------------------------------------------------------- Automations
// Automation scheduler lease (VST-038): a synchronous cache the runtime reads at
// startScheduler; the on-disk copy is loaded at app-ready (below) and updated on each
// acquire/renew so a restart re-adopts its own live lease instead of double-firing.
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
  // Scheduler lease persistence (VST-038): the loaded lease is a synchronous snapshot cached
  // at boot below; saves are fire-and-forget through the store's write chain.
  loadSchedulerLease: () => bootSchedulerLease,
  saveSchedulerLease: (lease) => { bootSchedulerLease = lease; void saveSchedulerLease(lease).catch(() => {}) },
})
registerAutomationIpc(automationRuntime)

registerComposerDraftIpc()
handle('runRecovery:review', async (_event, input: RunOutcomeReviewInput) => {
  requireSecurityRole('operator', 'review an uncertain run outcome')
  const lease = workspaceMutationGuard.reserveStart({ threadId: input?.threadId })
  try {
    const thread = store.threads.find(thread => thread.id === input?.threadId && !thread.orchestratorHostId)
    if (!thread) throw new Error('Choose a local conversation.')
    const next = reviewedRunOutcome(thread, input, aborters.has(thread.id))
    requireSecurityRole('operator', 'review an uncertain run outcome')
    await saveThreads(store.threads.map(item => item === thread ? next : item))
    requireSecurityRole('operator', 'review an uncertain run outcome')
    if (!store.threads.includes(thread) || thread.runRecovery?.runId !== input.runId || aborters.has(thread.id)) throw new Error('The run changed; review its current outcome.')
    thread.runRecoveryReviewedId = input.runId
    sendToRenderer('agent:event', { runId: input.runId, threadId: thread.id, type: 'thread_update', patch: { runRecoveryReviewedId: input.runId } })
    return { recovery: thread.runRecovery!, reviewedRunId: input.runId }
  } finally { lease.release() }
})
registerConversationSearchRuntime()
registerInputHistoryIpc({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['input:history'] }) })
registerMessageQuoteIpc({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['input:quotes'] }) })
registerReadableExportRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['export:readable'] }) })
registerDocumentLibraryRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['documents:library'] }) })
const attachmentTextService = createAttachmentTextService({ helperPath: computerHelperPath })
handle('attachments:textCapability', () => { requireSecurityRole('operator', 'check local text extraction'); return attachmentTextService.getCapability() })
handle('attachments:extractText', (_event, input) => { requireSecurityRole('operator', 'read text from a selected file'); return attachmentTextService.extract(input) })
const appWindowRuntime = createAppWindowRuntime({
  BrowserWindow,
  baseDir: __dirname,
  openExternalSafely,
  session,
  onWindowClosed: () => sshManager.cancelPendingPrompts(),
  onWindowCreated: (win) => { observeComposerDraftRenderer(win.webContents); desktopModeRuntime.attachWindow(win) },
  shouldShowWindow: () => desktopModeRuntime.shouldShowWindow(),
  installTextEditing: (win) => { ensureApplicationEditMenu(Menu, (options) => new MenuItem(options), store.settings?.uiPreferences?.locale); installNativeTextEditing(win, Menu, () => store.settings?.uiPreferences?.locale) },
  shouldGuardWindowClose: () => !draftQuitSettled,
  beforeWindowClose: (win) => requestRendererDraftFlush(win.webContents),
  onWindowCloseFailure: () => dialog.showErrorBox('Your draft needs attention', 'TensorFold could not confirm that unfinished messages were saved; keep the window open and retry saving the draft.'),
  // A blank window is otherwise silent: report the renderer failure to the terminal and a dialog.
  onRendererFailure: (detail) => {
    console.error(`[tensorfold] ${detail}`)
    dialog.showErrorBox('TensorFold could not start its interface', `${detail}\n\nThe interface did not load. Reinstalling the app or starting it with a fresh profile may help.`)
  },
})
// Lazy-accessor bundle: replaces the installSessionHardening + createWindow forwarder shims.
const appWindowApi = {
  installSessionHardening: () => appWindowRuntime.installSessionHardening(),
  createWindow: () => appWindowRuntime.createWindow(),
}

const desktopModeRuntime = createDesktopModeRuntime({
  headless: HEADLESS_MODE,
  getSettings: () => store.settings,
  getWindow: () => store.win,
  createWindow: () => appWindowRuntime.createWindow(),
  createTray: () => new Tray(desktopTrayIcon()),
  buildMenu: (items) => Menu.buildFromTemplate(items),
  showDock: () => { void app.dock?.show().catch(() => {}) },
  hideDock: () => app.dock?.hide(),
  changeMode: (mode) => coreIpc.setDesktopMode(mode),
  engineStatus: async () => engineService()?.status() ?? null,
  startModel: async (repo) => {
    requireSecurityRole('operator', 'start a local model')
    const reservation = workspaceMutationGuard.reserveStart({ workspaceKeys: ['engine:local'] })
    try {
      const engine = engineService()
      if (!engine) throw new Error('The built-in engine is unavailable.')
      await auditEvent('engine.load', { repo })
      return await engine.load(repo)
    } finally { reservation.release() }
  },
  stopModel: async () => {
    requireSecurityRole('operator', 'stop the local model')
    const reservation = workspaceMutationGuard.reserveStart({ workspaceKeys: ['engine:local'] })
    try { return await engineService()?.unload() } finally { reservation.release() }
  },
  activeLocalRuns: () => store.threads.filter((thread) => thread.providerId === 'tensorfold' && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && (thread.status === 'running' || aborters.has(thread.id))).length,
  learningBusy: () => engineService()?.isLearningBusy() ?? false,
  navigate: (target) => { sendToRenderer('desktop:navigate', { target }) },
  quit: () => app.quit(),
  reportError: (message) => dialog.showErrorBox('TensorFold', message),
})
app.on('second-instance', () => { if (app.isReady() && store.settings) desktopModeRuntime.show() })

// Ordered event spine (VST-034). Created at boot; null only during the pre-ready window.
// (spineHost is forward-declared above the emit helpers factory; assigned during app-ready.)

let threadRuntime: ThreadRuntime
// Semi-shim: emits an observer event before delegating — keep as a named function.
function startThreadRun(payload: StartRunPayload, existingJob?: Job, admission?: ThreadRunAdmission): Promise<{ runId: string }> {
  observerHub.emit('session.start', { threadId: payload.threadId, payload: { hasAttachments: !!payload.attachments?.length } })
  return threadRuntime.startThreadRun(payload, existingJob, admission)
}
// Hoisted forwarder shims: function declarations are hoisted so runtimes
// constructed above can reference them before threadRuntime is assigned.
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
// Per-turn workspace checkpoints (undo-a-turn / turn diff). Picks the SSH backend
// for remote workspaces, else the local host git runner. Ledgers persist to the
// protected turn-checkpoints.json (VST-036): incognito threads are never written,
// deleted threads are pruned at save/load time via the persistable filter, and the
// uuid id source keeps restored snapshot ids collision-free across restarts.
// Provider rate-limit status (VST-037): real codex account windows persisted per
// provider/account so the model picker + readiness doctor show live pace/ETA. In-memory
// map restored at boot; each recorded snapshot merges and fire-and-forget persists.
const providerStatusRuntime = createProviderStatusRuntime({
  saveProviderStatus,
  sendToRenderer,
})
function recordProviderRateLimits(providerId: string, accountId: string | null, rawSnapshot: unknown): void {
  return providerStatusRuntime.record(providerId, accountId, rawSnapshot)
}

const checkpoints = createCheckpointService({
  resolveBackend: (ws) => sshManager.backendForWorkspace(ws),
  resolveCwd: (cwd) => resolveWorkspaceCwd(cwd),
  conversationDeps: { clock: { now: () => Date.now() }, newId: () => randomUUID() },
  persistence: { load: loadTurnCheckpoints, save: saveTurnCheckpoints },
  isPersistableThread: (threadId) => {
    const thread = store.threads.find((candidate) => candidate.id === threadId)
    return !!thread && !thread.incognito
  },
  // Fork worktrees live under the app state dir, NOT inside the user's repo — a fork
  // must not pollute the source project's git status or tree views.
  forkWorktreePath: ({ threadId, turn, branch }) =>
    path.join(os.homedir(), '.vontra', 'worktrees', `${threadId.slice(0, 8)}-turn${turn}-${branch.replace(/[^a-zA-Z0-9_-]/g, '')}`),
})
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

// Cross-engine handoff (VST-036): flatten the source conversation into a seed message and
// continue it on a DIFFERENT engine in a fresh thread. The transcript never round-trips
// through the renderer — the host rebuilds it from the live thread (same rule as session
// import), and the new thread carries NO resume cursors: a different engine cannot resume
// the source harness session, the seed transcript is its context.
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
// Lazy-accessor bundle: replaces the ensureProjectInstructionFiles /
// upgradeGeneratedClaudeGuidance / cancelRoomRun / startRoomRun forwarder shims.
// Declared before its first use in createJobControlRuntime; the closures read
// workspaceRuntimeIpc lazily at call time (after it is assigned below).
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

// Live-session wake sweeper: parked rooms (nextWakeAt set by parkRoomIfWorkRemains)
// are woken on a 30s cadence; unref'd so it never keeps the process alive.
jobControlRuntime.startRoomWakeSweeper()



// Boot lifecycle helpers (extracted into boot-helpers.ts). applyTheme is passed into
// registerCoreIpc below and handleBootFailure is the whenReady catch handler, so both must
// be constructed before those calls; installSessionHardening/createWindow are wired above.
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

registerWorkspaceIpc({
  resolveBackend: (ws) => sshManager.backendForWorkspace(ws),
  checkpoints,
  confirmSensitiveAction,
  isThreadRunning: (threadId) => aborters.has(threadId),
  isRoomRunning: (roomId) => roomAborters.has(roomId),
  workspaceMutationGuard,
  // Apply a turn-revert's restored conversation to the live thread (VST-036) / fork a thread
  // at a turn (VST-036): both owned here but built in thread-revert-fork.ts. The remoteServer +
  // spineHost deps are late-bound accessors (assigned later in the bootstrap).
  applyThreadRevert: createThreadRevertApply({ persistThreads, remoteServer: () => remoteServer, spineHost: () => spineHost }),
  createForkThread: createThreadForkFactory({ persistThreads, remoteServer: () => remoteServer, spineHost: () => spineHost }),
})

// ---- Projects & multi-agent Rooms ----
workspaceRuntimeIpc = createWorkspaceRuntimeIpc(buildWorkspaceRuntimeIpcConfig({
  fs,
  path,
  randomUUID,
  store,
  handle,
  spineHost: () => spineHost,
  isActive,
  archiveEntity,
  killProjectTerminals,
  abortMissionsForArchive: abortAndWaitMissionsForArchive,
  loopHost,
  workspaceMutationGuard,
  requireSecurityRole,
  auditEvent,
  auditPreview,
  persistProjects,
  persistThreads,
  persistRooms,
  persistTickets,
  persistSubAgents,
  persistSkills,
  persistProjectMemory,
  persistProjectMemoryCandidates,
  aborters,
  roomAborters,
  DEFAULT_ROOM_COLLABORATION_MODE,
  DEFAULT_ROOM_WORKER_BUDGET,
  ensureRoomWorktree,
  emitRoom,
  emitTicket,
  emitJob,
  remoteServer: () => remoteServer,
  activeJob: roomJobApi.activeJob,
  patchRoomSupervisorJob: roomJobApi.patchRoomSupervisorJob,
  finishRoomSupervisorMission: roomJobApi.finishRoomSupervisorMission,
  finishJob: roomJobApi.finishJob,
  parkRoomIfWorkRemains: roomJobApi.parkRoomIfWorkRemains,
  ensureRoomSupervisorMission: roomJobApi.ensureRoomSupervisorMission,
  roomWorkerBudget: roomJobApi.roomWorkerBudget,
  claimJob: roomJobApi.claimJob,
  patchJob,
  buildRoomSupervisorJobMetadata,
  ticketsForRoom: roomJobApi.ticketsForRoom,
  runtimeProvider,
  inferSeatLane,
  appendRoomSupervisorEvent: roomJobApi.appendRoomSupervisorEvent,
  ticketNeedsReviewAfterBuilder,
  roomTicketConflictBlockers,
  summarizeRoomLeftToWork,
  describeTicketForRoom,
  mkTicket,
  assertTicketRelations,
  normalizeTicketCapabilities,
  normalizeStringList,
  ticketEvidenceFromArgs,
  ticketFileRefs,
  startTicketWork,
  createSubAgentFromLaunch,
  parentForRoomSeat,
  startSubAgent,
  runReadOnlySubAgent,
  executeTool,
  modelRouter,
  modelRouteAutoFailoverEnabled,
  effectiveContextLimit,
  roomContextLimit,
  auditScopedModelRoute,
  runtimeProviderWithFreshModels,
  assertProviderEndpoint,
  modelRouteFailoverText,
  roomRouteHandoff,
  modelRouteResetPolicy,
  roomFinalGateBlockers,
  runRoom,
  normalizeTicketExecution,
  ticketWorkOptionsWithProjectRoute,
  ticketWorkerRuntime,
  resolveProjectProvider,
  ensureTensorFoldStudioStylePrompt,
  streamChat,
  parseJsonLoose,
  createProjectMemoryCandidate,
  approveProjectMemoryCandidate,
  rejectProjectMemoryCandidate,
  projectMemoryCandidateDiff,
  rollbackProjectMemoryEntry,
  forgetProjectMemoryEntry,
  scopeProjectMemoryEntries,
  scopeProjectMemoryCandidates,
  registerWebhookIpc,
  webhookServer,
  rememberConnectorEvents,
  confirmSensitiveAction,
}))

// Construct the *OnThisHost REST layer now that every collaborator it closes over is
// initialized (abort maps, emit helpers, abortAndWaitMissionsForArchive, loopHost,
// ticketWorkerRuntime, the thread/room run entry points, remoteServer, …). The late-bound
// deps are passed as accessors so lookups stay lazy, mirroring the original closures.
remoteHostApi = createRemoteHostApi({
  sendToRenderer,
  persistProjects,
  persistThreads,
  persistRooms,
  persistTickets,
  defaultRoomCollaborationMode: DEFAULT_ROOM_COLLABORATION_MODE,
  defaultRoomWorkerBudget: DEFAULT_ROOM_WORKER_BUDGET,
  ensureProjectInstructionFiles: workspaceApi.ensureProjectInstructionFiles,
  remoteServer: () => remoteServer,
  aborters: () => aborters,
  roomAborters: () => roomAborters,
  emitTicket,
  emitRoom,
  abortAndWaitMissionsForArchive: () => abortAndWaitMissionsForArchive,
  loopHost: () => loopHost,
  ticketWorkerRuntime: () => ticketWorkerRuntime,
  startThreadRun,
  guideThreadRun,
  cancelThreadRun,
  startRoomRun: workspaceApi.startRoomRun,
  cancelRoomRun: workspaceApi.cancelRoomRun,
})

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

// ── Phase 6: production incident mode ────────────────────────────────────────
registerIncidentIpc({
  effectiveContextLimit,
  fullToolContext,
  applyEventToThread,
  replaceMission,
  getMission,
  runMission: missionApi.runMission,
  emitIncident,
})

// Built-in TensorFold engine: Models page, pulls, and the engine behind the built-in provider.
createEngineRuntime()
const assertCanBenchmark = () => {
  requireSecurityRole('operator', 'benchmark a local model')
  if (store.threads.some((thread) => {
    const providerId = thread.providerId ?? surfaceModelOverride(store.settings, thread.kind === 'chat' ? 'chat' : 'code')?.providerId ?? store.settings.activeProviderId
    return providerId === 'tensorfold' && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && (thread.status === 'running' || aborters.has(thread.id))
  })) throw new Error('Finish or stop active local conversations before benchmarking a model.')
}
registerModelBenchmarkIpc(engineService, { assertCanBenchmark, reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['engine:local'] }) })
registerEngineIpc(engineService, {
  assertCanCompare: () => {
    requireSecurityRole('operator', 'compare two local models')
    if (store.threads.some((thread) => {
      const providerId = thread.providerId ?? surfaceModelOverride(store.settings, thread.kind === 'chat' ? 'chat' : 'code')?.providerId ?? store.settings.activeProviderId
      return providerId === 'tensorfold' && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && (thread.status === 'running' || aborters.has(thread.id))
    })) throw new Error('Finish or stop active local conversations before comparing models.')
  },
  reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['engine:local'] }),
  assertCanEnableCortheon: () => {
    if (store.threads.some((thread) => aborters.has(thread.id) && (thread.phoneLocalOnly || thread.googleLocalOnly || thread.bot?.executionTarget === 'vm' || isRestrictedBot(thread.bot)))) {
      throw new Error('Stop active phone, isolated or restricted bot conversations before enabling Cortheon; hidden research cannot bypass their execution boundaries.')
    }
  },
})
const phones = initializePhoneHost({
  appHome: vontraHomeDir(), supported: !HEADLESS_MODE,
  authorizeThread: (id) => {
    const thread = store.threads.find((candidate) => candidate.id === id)
    if (HEADLESS_MODE || !thread || thread.kind !== 'chat' || thread.orchestratorHostId || thread.workspace?.kind === 'ssh') throw new Error('Phone control requires a chat in this local desktop app.')
    const provider = store.settings.providers.find((candidate) => candidate.id === (thread.providerId ?? store.settings.activeProviderId))
    if (!provider || isExternalProvider(runtimeProvider(provider, thread.providerAccountId))) throw new Error('Choose the built-in TensorFold engine or another model running on this Mac before granting phone access.')
    if (engineService()?.hasExternalResearchEnabled()) throw new Error('Turn Cortheon off in AI settings before granting phone access; phone chats keep research and model processing local.')
  },
  markChatLocal: async (id) => {
    const thread = store.threads.find((candidate) => candidate.id === id)
    if (!thread) throw new Error('This chat is no longer available.')
    if (thread.phoneLocalOnly) return
    thread.phoneLocalOnly = true
    try { await saveThreads(store.threads) } catch (error) { delete thread.phoneLocalOnly; throw error }
    sendToRenderer('thread:update', thread)
  },
  cancelThread: (id) => { cancelThreadRun(id) },
  emit: (snapshot) => { sendToRenderer('phone:event', snapshot) },
})
registerPhoneIpc(phones, { reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['phone:local'] }) })
const botVms = initializeBotVmHost({
  appHome: vontraHomeDir(), supported: !HEADLESS_MODE,
  reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['bot:vm'] }),
})
registerBotVmIpc(botVms, {
  authorizeBot: (id) => {
    if (HEADLESS_MODE || !store.settings?.agentPresets.some((bot) => bot.id === id)) throw new Error('Choose a saved bot in this local desktop app.')
  },
  authorizeBotStart: (id) => {
    const preset = store.settings?.agentPresets.find((bot) => bot.id === id)
    if (HEADLESS_MODE || !preset || preset.executionTarget !== 'vm') throw new Error('Choose a saved VM bot in this local desktop app.')
    assertBotLifecycleIdle(id)
    captureBotConfig(preset, store.settings.providers)
  },
  cancelBotRuns: (id) => {
    for (const thread of store.threads) if (thread.bot?.presetId === id) aborters.get(thread.id)?.abort()
  },
  reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['bot:vm'] }),
})
const isQuestionRunActive = (threadId: string, runId: string) => {
  const controller = aborters.get(threadId)
  return !!controller && !controller.signal.aborted && store.threads.find((thread) => thread.id === threadId)?.activeRunContract?.runId === runId
}
const questionPersistence = createUserQuestionPersistence({
  getThread: (id) => store.threads.find((thread) => thread.id === id) ?? null,
  save: (run, patch) => saveThreadQuestion(() => store.threads, run, patch),
  emit: (run, patch) => {
    // A slow encrypted write must not raise a fresh attention notice after Stop.
    if (patch.question?.state !== 'interrupted' && !isQuestionRunActive(run.threadId, run.runId)) return
    if (patch.question?.state === 'pending' && !getUserQuestionHost()?.isWaiting(patch.question.id)) return
    emitToRenderer({ ...run, type: 'item_update', id: run.itemId, patch })
  },
})
const userQuestions = HEADLESS_MODE ? null : initializeUserQuestionHost({
  getThread: (id) => store.threads.find((thread) => thread.id === id) ?? null,
  isRunActive: isQuestionRunActive,
  commitQuestion: questionPersistence.commit,
})
if (userQuestions) registerUserQuestionIpc(userQuestions, { reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['user:question'] }) })
function assertWorkspaceIdle(): void {
  assertClusterWorkspaceIdle()
  assertUpdateQuiescent({
    ready: app.isReady() && !!store.settings,
    threadControllers: aborters.size,
    roomControllers: roomAborters.size,
    missionControllers: missionAborters.size,
    runningMissions: runningMissions.size,
    pendingApprovals: pendingApprovals.size + pendingApprovalMeta.size + missionApprovalResolvers.size,
    runningThreads: store.threads.filter((thread) => thread.status === 'running').length,
    activeJobs: store.jobs.filter((job) => ['queued', 'running', 'waiting_approval'].includes(job.status)).length,
    runningAutomations: store.automations.filter((automation) => automation.runs?.some((run) => run.status === 'running')).length,
    liveTerminals: liveTerminalCount(),
    mobileOperations: mobileBridgeRuntime.activeOperations(),
    engineRequests: engineService()?.supervisor.activeRequestCount ?? 0,
    modelLoading: engineService()?.supervisor.status().state === 'starting',
    modelDownloads: engineService()?.pullList().filter((pull) => ['queued', 'running', 'verifying'].includes(pull.state)).length ?? 0,
  })
}
async function persistWorkspaceSnapshot(next: BackupHostState): Promise<void> {
  next.settings.agentPresets = normalizeAgentPresets(next.settings.agentPresets ?? [], true)
  const results = await Promise.allSettled([saveSettings(next.settings), saveThreads(next.threads), saveProjects(next.projects)])
  if (results.some(result => result.status === 'rejected')) throw new Error('Workspace storage could not complete; restore will roll back the reviewed items.')
}
registerModelHelperRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['models:helper'] }), assertIdle: assertWorkspaceIdle })
registerModelQualificationRuntime({ reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['models:qualification'] }), assertIdle: assertWorkspaceIdle })
registerWorkspaceBackupRuntime({
  handle,
  authorize: () => requireSecurityRole('owner', 'back up or restore this workspace'),
  profileRoot: vontraHomeDir,
  getState: () => ({ settings: store.settings, threads: store.threads, projects: store.projects }),
  setState: next => { store.settings = next.settings; store.threads = next.threads; store.projects = next.projects },
  persist: persistWorkspaceSnapshot,
  readProtectedJson, writeProtectedJson,
  flushDrafts: event => requestRendererDraftFlush(event.sender),
  pauseDraftWrites: beginComposerDraftPurge,
  resumeDraftWrites: endComposerDraftPurge,
  acquireLease: () => workspaceMutationGuard.acquireGlobal('Workspace backup or restore'),
  assertIdle: assertWorkspaceIdle,
  chooseOpen: async () => { const result = await dialog.showOpenDialog({ title: 'Choose an encrypted workspace backup', properties: ['openFile'], filters: [{ name: 'TensorFold backup', extensions: ['tfbackup'] }] }); return result.canceled ? null : result.filePaths[0] ?? null },
  chooseSave: async () => { const result = await dialog.showSaveDialog({ title: 'Save encrypted workspace backup', defaultPath: 'tensorfold-workspace.tfbackup', filters: [{ name: 'TensorFold backup', extensions: ['tfbackup'] }] }); return result.canceled ? null : result.filePath ?? null },
  publishRestored: () => { sendToRenderer('composer-drafts:reset', null); setImmediate(() => store.win?.webContents.reload()) },
})
configureUpdateInstallSafety({
  acquireLease: () => workspaceMutationGuard.acquireGlobal('App update installation'),
  assertIdle: assertWorkspaceIdle,
  preserve: preserveStateForUpdate,
})
registerUpdateInstallIpc()


// ---------------------------------------------------------------- lifecycle

app.whenReady().then(() => bootAppState({
  app,
  BrowserWindow,
  fs,
  path,
  HEADLESS_MODE,
  ensureCliPath,
  migrateLegacyUserData,
  ensureTensorFoldStudioHome,
  recoverWorkspaceBackup: async () => { await recoverWorkspaceBackupCheckpoints({ profileRoot: vontraHomeDir, readProtectedJson, writeProtectedJson, persist: persistWorkspaceSnapshot }); await loadClusterWorkspaceIdleState() },
  enableOrchestrationJournal,
  wireOrchestrationEventsToWindow,
  sendToRenderer,
  createEventSpineHost,
  vontraHomeDir,
  store,
  loadSettings,
  loadConfiguredPlugins,
  saveSettings,
  loadThreads,
  loadProjects,
  loadRooms,
  loadTickets,
  loadAutomations,
  loadConnectorEvents,
  recoverConnectorDeliveries: recoverInterruptedConnectorDeliveries,
  loadContactDirectory,
  loadCliSessions,
  loadMissions,
  loadJobs,
  loadSubAgents,
  loadSkills,
  loadCommands,
  loadSuggestions,
  loadBotControlSessions,
  loadProjectMemory,
  loadProjectMemoryCandidates,
  loadWebhooks,
  loadIncidents,
  loadMcpServers,
  loadArtifacts,
  loadPairingSessions,
  loadProviderStatus,
  loadSchedulerLease,
  recordPersistenceFailure,
  auditEvent,
  writeTensorFoldStudioStateJson,
  purgeLegacyUserDataState,
  vontraPath,
  normalizeServerConfig,
  normalizeArtifactRecord,
  pairingRegistry,
  persistPairingSessions,
  durablePairingMutation,
  checkpoints,
  providerStatusRuntime,
  mcpHost,
  remoteServer,
  webhookServer,
  prepareHeadlessRemoteControl,
  makeRemoteControlToken,
  headlessBootstrapMessages,
  cleanupMediaCache,
  reconcileJobsAndMissionsAfterRestart,
  reconcileIncidentsAfterRestart,
  persistJobs,
  persistMissions,
  persistIncidents,
  upgradeGeneratedClaudeGuidance: workspaceApi.upgradeGeneratedClaudeGuidance,
  reconcileStuckJobs: roomJobApi.reconcileStuckJobs,
  installSessionHardening: appWindowApi.installSessionHardening,
  createWindow: appWindowApi.createWindow,
  initializeDesktopMode: () => desktopModeRuntime.applySettings({ startup: true }),
  showDesktopWindow: () => desktopModeRuntime.show(),
  applyTheme,
  applyPowerSettings,
  startScheduler: automationApi.startScheduler,
  startUpdateChecks,
  startBotControlPolling: botPollingApi.startBotControlPolling,
  resumeAutoJobsAfterRestart: jobControlApi.resumeAutoJobsAfterRestart,
  getSpineHost: () => spineHost,
  setSpineHost: (host) => { spineHost = host },
  setBootSchedulerLease: (lease) => { bootSchedulerLease = lease },
})).then(async () => { await userQuestions?.recoverInterrupted(store.threads); await mobileBridgeRuntime.initialize(); if (!quitAdmissionPause) void connectorRuntime.resumeWhatsAppAccounts().catch((error) => { void auditEvent('whatsapp.resume_failed', { message: String(error instanceof Error ? error.message : 'Saved accounts could not resume').slice(0, 220) }) }) }).catch(handleBootFailure)

// Last-resort safety net: a rejected promise or thrown error anywhere in the main
// process should be logged (and surfaced), not silently swallowed into a windowless
// app. These never re-throw, so they can't themselves crash the process.
process.on('unhandledRejection', (reason) => {
  console.error('[main] unhandledRejection', reason)
})
process.on('uncaughtException', (err) => {
  console.error('[main] uncaughtException', err)
})

let quitAdmissionPause: { release(): void } | null = null
function resumeAfterFailedQuit() {
  draftQuitSettled = false
  if (store.win && !store.win.isDestroyed()) store.win.setEnabled(true)
  // A later device fence can fail after earlier services have already stopped.
  // Reopen their admission without reconnecting phones or restarting bot VMs.
  try {
    resumeDocumentLibrary(); resumeModelQualifications(); resumeModelHelpers(); documentQuitSettled = false
    userQuestions?.resume(); questionQuitSettled = false
    phones.service.resume(); phoneQuitSettled = false
    botVms.resume(); botVmQuitSettled = false
    quitAdmissionPause?.release(); quitAdmissionPause = null
  } catch { /* retain the gate until the still-stopping service settles and Quit is retried */ }
}
const deferLearningQuit = createLearningQuitGuard({
  service: engineService,
  quit: () => app.quit(),
  failed: (error) => {
    resumeAfterFailedQuit()
    console.error('[engine] Could not finish learning before quit', error)
    dialog.showErrorBox('Living Weights is still saving', 'TensorFold could not confirm that your private model is safely saved, so it remains open; inspect the learning graph before quitting again.')
  },
})
let phoneQuitSettled = false
const deferPhoneQuit = createLearningQuitGuard({
  service: () => ({ isLearningBusy: () => !phoneQuitSettled, prepareToQuit: async () => { await phones.shutdown(); phoneQuitSettled = true } }),
  quit: () => { setImmediate(() => app.quit()) },
  failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Phone control is stopping', 'TensorFold is waiting for its phone connection to close safely; try Quit again after it stops.') },
})
let botVmQuitSettled = false
const deferBotVmQuit = createLearningQuitGuard({
  service: () => ({ isLearningBusy: () => !botVmQuitSettled, prepareToQuit: async () => { await botVms.shutdown(); botVmQuitSettled = true } }),
  quit: () => { setImmediate(() => app.quit()) },
  failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Bot VMs are stopping', 'TensorFold could not confirm that its isolated bot work has stopped; your VM workspaces are preserved, so inspect the bot status before quitting again.') },
})
let questionQuitSettled = false
const deferQuestionQuit = createLearningQuitGuard({
  service: () => userQuestions ? { isLearningBusy: () => !questionQuitSettled, prepareToQuit: async () => { await userQuestions.shutdown(); await questionPersistence.drain(); questionQuitSettled = true } } : null,
  quit: () => { setImmediate(() => app.quit()) },
  failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Questions are stopping', 'TensorFold could not confirm that its pending questions were safely interrupted; try Quit again after the current save finishes.') },
})
let documentQuitSettled = false
const deferDocumentQuit = createLearningQuitGuard({
  service: () => ({ isLearningBusy: () => !documentQuitSettled, prepareToQuit: async () => { await Promise.all([quiesceDocumentLibrary(), quiesceModelQualifications(), quiesceModelHelpers(), quiesceClusterWorkspace()]); documentQuitSettled = true } }),
  quit: () => { setImmediate(() => app.quit()) },
  failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Document indexing is stopping', 'TensorFold is waiting for document indexing to stop safely; try Quit again after the current scan finishes.') },
})
let draftQuitSettled = false
const deferDraftQuit = createLearningQuitGuard({
  service: () => ({ isLearningBusy: () => !draftQuitSettled, prepareToQuit: async () => { await requestRendererDraftFlush(store.win?.webContents ?? null); draftQuitSettled = true } }),
  quit: () => { setImmediate(() => app.quit()) },
  failed: () => { resumeAfterFailedQuit(); dialog.showErrorBox('Your draft needs attention', 'TensorFold could not save an unfinished message, so it remains open; retry saving the draft before quitting.') },
})
app.on('before-quit', (event) => {
  if (store.win && !store.win.isDestroyed()) store.win.setEnabled(false)
  quitAdmissionPause ??= workspaceMutationGuard.pauseAdmission('TensorFold is quitting')
  telegramRemoteRuntime.cancelPending()
  // Stop model work before asynchronous save/device fences can release a parked
  // question; Quit must never interpret an interrupted question as permission.
  for (const controller of aborters.values()) controller.abort()
  for (const controller of roomAborters.values()) controller.abort()
  for (const controller of missionAborters.values()) controller.abort()
  automationRuntime?.abortAll()
  if (deferLearningQuit(event)) return
  if (deferQuestionQuit(event)) return
  if (deferPhoneQuit(event)) return
  if (deferBotVmQuit(event)) return
  // The renderer cannot accept new keyboard input while device/model fences run; save last.
  if (deferDocumentQuit(event)) return
  if (deferDraftQuit(event)) return
  desktopModeRuntime.dispose()
  botPollingApi.stopBotControlPolling()
  stopAllWhatsAppBridges()
  closeBrowserAutomationSessions()
  stopCaffeinate()
  // Abort in-flight thread/room/mission/automation runs: CLI one-shot transports spawn
  // detached and are only killed via their run's abort signal, so an unaborted run would
  // outlive the app as a parentless process against the user's workspace.
  try { for (const controller of aborters.values()) controller.abort() } catch { /* best effort */ }
  try { for (const controller of roomAborters.values()) controller.abort() } catch { /* best effort */ }
  try { for (const controller of missionAborters.values()) controller.abort() } catch { /* best effort */ }
  try { automationRuntime?.abortAll() } catch { /* best effort */ }
  // The computer-use helper never outlives the app.
  try { shutdownComputerUse() } catch { /* best effort */ }
  attachmentTextService.shutdown()
  // FIX 1b: reap all live PTYs on app quit so shells don't outlive the process
  try { killAllTerminals() } catch { /* best effort */ }
  // Close every pooled codex app-server process so it doesn't outlive the app either.
  try { closeAllCodexAppServerSessions() } catch { /* best effort */ }
  // The built-in engine holds a model in memory; it gets SIGTERM now and SIGKILL at process exit.
  try { engineService()?.shutdown(false) } catch { /* best effort */ }
  void mcpHost.shutdown()
  void webhookServer.stop()
  void mobileBridgeRuntime.stop()
  void remoteServer.stop()
})
app.on('window-all-closed', () => { if (!HEADLESS_MODE && process.platform !== 'darwin' && !desktopModeRuntime.canBackground()) app.quit() })
