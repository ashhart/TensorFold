import path from 'node:path'
import { promises as fs } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { executeTool } from '../../agent/tools'
import { runRoom } from '../../agent/rooms'
import { DEFAULT_ROOM_COLLABORATION_MODE, inferSeatLane } from '../../agent/room-collaboration'
import { streamChat } from '../../agent/openai'
import { buildRoomSupervisorJobMetadata } from '../../agent/jobs'
import { createSubAgentFromLaunch, parentForRoomSeat, startSubAgent } from '../../agent/subagents'
import { normalizeTicketCapabilities, normalizeTicketExecution, normalizeStringList, roomFinalGateBlockers, roomTicketConflictBlockers, summarizeRoomLeftToWork, ticketFileRefs, ticketNeedsReviewAfterBuilder } from '../../agent/tickets'
import { confirmSensitiveAction } from '../host/sensitive-action'
import { createThreadRevertApply, createThreadForkFactory } from '../host/thread-revert-fork'
import { buildWorkspaceRuntimeIpcConfig } from '../host/workspace-runtime-config'
import { auditPreview, assertProviderEndpoint } from '../../agent/security'
import { ensureTensorFoldStudioStylePrompt } from '../../agent/prompt-style'
import { createRemoteControlServer } from '../../agent/remote-control'
import { approveProjectMemoryCandidate, createProjectMemoryCandidate, forgetProjectMemoryEntry, projectMemoryCandidateDiff, rejectProjectMemoryCandidate, rollbackProjectMemoryEntry, scopeProjectMemoryCandidates, scopeProjectMemoryEntries } from '../../agent/project-memory'
import { type EventSpineHost } from '../../agent/event-spine-host'
import { store } from '../state'
import { handle } from '../host/ipc'
import { persistProjectMemory, persistProjectMemoryCandidates, persistProjects, persistRooms, persistSkills, persistSubAgents, persistThreads, persistTickets } from '../host/persistence'
import { auditEvent, requireSecurityRole } from '../host/audit'
import { archiveEntity, isActive } from '../host/archive'
import { SshManager } from '../host/ssh'
import { registerWebhookIpc } from '../host/webhook-ipc'
import { registerWorkspaceIpc } from '../host/workspace-ipc'
import { createCheckpointService } from '../host/checkpoint-runtime'
import { killProjectTerminals } from '../host/terminal-runtime'
import { createRemoteHostApi } from '../host/remote-host-api'
import { type ThreadRuntime } from '../host/thread-runtime'
import { type RoomJobRuntime } from '../host/room-job-runtime'
import { sendToRenderer } from '../host/renderer-events'
import { effectiveContextLimit, parseJsonLoose, roomContextLimit } from '../host/context-limit'
import { assertTicketRelations, describeTicketForRoom, mkTicket, ticketEvidenceFromArgs } from '../host/ticket-helpers'
import { ensureRoomWorktree } from '../host/room-worktree'
import { createWorkspaceRuntimeIpc, type WorkspaceRuntimeIpc } from '../host/workspace-runtime-ipc'
import { auditScopedModelRoute, modelRouteAutoFailoverEnabled, modelRouteFailoverText, modelRouteResetPolicy, modelRouter, resolveProjectProvider, runtimeProvider, runtimeProviderWithFreshModels, ticketWorkOptionsWithProjectRoute } from '../host/provider-runtime'

type WorkspaceConfig = Parameters<typeof buildWorkspaceRuntimeIpcConfig>[0]
type RoomWorkspaceActions = Pick<RoomJobRuntime, 'activeJob' | 'patchRoomSupervisorJob' | 'finishRoomSupervisorMission' | 'finishJob' | 'parkRoomIfWorkRemains' | 'ensureRoomSupervisorMission' | 'roomWorkerBudget' | 'claimJob' | 'ticketsForRoom' | 'appendRoomSupervisorEvent'>
interface ApplicationWorkspaceDeps extends Pick<WorkspaceConfig, 'aborters' | 'roomAborters' | 'loopHost' | 'workspaceMutationGuard' | 'emitRoom' | 'emitTicket' | 'emitJob' | 'patchJob' | 'startTicketWork' | 'runReadOnlySubAgent' | 'roomRouteHandoff' | 'ticketWorkerRuntime' | 'webhookServer' | 'rememberConnectorEvents' | 'DEFAULT_ROOM_WORKER_BUDGET'> {
  sshManager: SshManager
  checkpoints: ReturnType<typeof createCheckpointService>
  spineHost(): EventSpineHost | null
  remoteServer: ReturnType<typeof createRemoteControlServer>
  abortAndWaitMissionsForArchive: WorkspaceConfig['abortMissionsForArchive']
  roomJobApi: RoomWorkspaceActions
  workspaceApi: Pick<WorkspaceRuntimeIpc, 'ensureProjectInstructionFiles' | 'startRoomRun' | 'cancelRoomRun'>
  startThreadRun: ThreadRuntime['startThreadRun']
  guideThreadRun: ThreadRuntime['guideThreadRun']
  cancelThreadRun: ThreadRuntime['cancelThreadRun']
}

export function createApplicationWorkspace(deps: ApplicationWorkspaceDeps) {
  const { aborters, roomAborters, loopHost, workspaceMutationGuard, emitRoom, emitTicket, emitJob, patchJob, startTicketWork, runReadOnlySubAgent, roomRouteHandoff, ticketWorkerRuntime, webhookServer, rememberConnectorEvents, DEFAULT_ROOM_WORKER_BUDGET, sshManager, checkpoints, spineHost, remoteServer, abortAndWaitMissionsForArchive, roomJobApi, workspaceApi, startThreadRun, guideThreadRun, cancelThreadRun } = deps
  registerWorkspaceIpc({
    resolveBackend: (ws) => sshManager.backendForWorkspace(ws),
    checkpoints,
    confirmSensitiveAction,
    isThreadRunning: (threadId) => aborters.has(threadId),
    isRoomRunning: (roomId) => roomAborters.has(roomId),
    workspaceMutationGuard,

    applyThreadRevert: createThreadRevertApply({ persistThreads, remoteServer: () => remoteServer, spineHost }),
    createForkThread: createThreadForkFactory({ persistThreads, remoteServer: () => remoteServer, spineHost }),
  })

  const workspaceRuntimeIpc = createWorkspaceRuntimeIpc(buildWorkspaceRuntimeIpcConfig({
    fs,
    path,
    randomUUID,
    store,
    handle,
    spineHost,
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

  const remoteHostApi = createRemoteHostApi({
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

  return { workspaceRuntimeIpc, remoteHostApi }
}
