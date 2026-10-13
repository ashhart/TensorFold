import type { RemoteWorkspaceScope } from '../../agent/local-workspace-policy'
import path from 'node:path'
import { app } from 'electron'
import { type RemoteControlServerDeps } from '../../agent/remote-control'
import { type EventSpineHost } from '../../agent/event-spine-host'
import { type PairingRegistry } from '../../agent/pairing-sessions'
import { HEADLESS_MODE } from '../early-electron-paths'
import { archiveSnapshot } from './archive'
import { redactRemoteControlEventForPaired, pairedRemoteArchiveSnapshot, pairedRemoteControlSnapshot, remoteControlDoctorSnapshot, remoteControlSnapshot } from './remote-snapshot'
import { durablePairingMutation } from './durable-pairing'
import { auditEvent } from './audit'
import { persistSettings } from './persistence'
import { store } from '../state'
import type { JobControlRuntime } from './job-control-runtime'
import type { TerminalRemoteApi } from './terminal-runtime'
import { engineService } from './engine-runtime'

export interface RemoteControlConfigDeps {
  // Pairing (constructed at module scope, before this factory): direct references are fine.
  pairingRegistry: PairingRegistry
  persistPairingSessions(): Promise<void>
  // Late-bound collaborators — forward-declared `let` bindings assigned later in the bootstrap.
  // Accessors keep the lookups lazy so the factory can be built early while the resolved values
  // are read at runtime, exactly mirroring the original inline closure semantics.
  spineHost(): EventSpineHost | null
  remoteHostApi(): {
    createThreadOnThisHost(input: { cwd?: string; kind?: 'task' | 'chat'; projectId?: string | null }, scope?: RemoteWorkspaceScope): Promise<import('../../agent/types').Thread>
    startRemoteThreadOnThisHost(payload: import('../../agent/types').StartRunPayload, scope?: RemoteWorkspaceScope): Promise<{ runId: string }>
    guideRemoteThreadOnThisHost(payload: import('../../agent/types').CueRunPayload, scope?: RemoteWorkspaceScope): Promise<import('../../agent/types').CueRunResult>
    cancelRemoteThreadOnThisHost(threadId: string): void | Promise<void>
    archiveProjectOnThisHost(id: string): Promise<import('../../agent/types').Project | null>
    restoreProjectOnThisHost(id: string): Promise<import('../../agent/types').Project | null>
    archiveThreadOnThisHost(id: string): Promise<import('../../agent/types').Thread | null>
    restoreThreadOnThisHost(id: string): Promise<import('../../agent/types').Thread | null>
    updateProjectOnThisHost(id: string, patch: Partial<import('../../agent/types').Project>): Promise<import('../../agent/types').Project>
    createProjectOnThisHost(input: Partial<import('../../agent/types').Project> & { name?: string; cwd?: string }, scope?: RemoteWorkspaceScope): Promise<import('../../agent/types').Project>
    createRoomOnThisHost(input: { projectId: string; name?: string }): Promise<import('../../agent/types').Room>
    updateRoomOnThisHost(id: string, patch: Partial<import('../../agent/types').Room>): Promise<import('../../agent/types').Room>
    renameRoomOnThisHost(id: string, name: string): Promise<import('../../agent/types').Room>
    steerRoomOnThisHost(id: string, text: string, scope?: RemoteWorkspaceScope): Promise<import('../../agent/types').RoomEventItem>
    mutateRoomMemoryOnThisHost(id: string, command: import('../../agent/types').RoomManualMemoryCommand): Promise<import('../../agent/types').Room>
    archiveRoomOnThisHost(id: string): Promise<import('../../agent/types').Room | null>
    restoreRoomOnThisHost(id: string): Promise<import('../../agent/types').Room | null>
    startRemoteRoomOnThisHost(roomId: string, problem: string, attachments?: import('../../agent/types').Attachment[], scope?: RemoteWorkspaceScope): Promise<{ ok: boolean; room?: import('../../agent/types').Room }>
    cancelRemoteRoomOnThisHost(roomId: string): void | Promise<void>
    createTicketOnThisHost(input: Partial<import('../../agent/types').Ticket> & { projectId: string }): Promise<import('../../agent/types').Ticket>
    startRemoteTicketExecutionOnThisHost(id: string, input: { hostId: string; userText: string }, scope?: RemoteWorkspaceScope): Promise<{ ticket: import('../../agent/types').Ticket; thread: import('../../agent/types').Thread; runId: string }>
    updateTicketOnThisHost(id: string, patch: Partial<import('../../agent/types').Ticket>): Promise<import('../../agent/types').Ticket>
    transitionRemoteTicketOnThisHost(id: string, command: import('../../agent/types').TicketRemoteLifecycleCommand): Promise<import('../../agent/types').Ticket>
    archiveTicketOnThisHost(id: string): Promise<import('../../agent/types').Ticket | null>
    restoreTicketOnThisHost(id: string): Promise<import('../../agent/types').Ticket | null>
    listHostDirectories(dir: string, options?: { confineToHome?: boolean }): Promise<import('../../agent/types').RemoteDirectoryListResult>
  }
  jobControlRuntime(): Pick<JobControlRuntime, 'resolveToolApproval'>
  terminalRemoteApi(): TerminalRemoteApi | null
  // Mutable abort/run maps — declared later than this factory in the bootstrap, but created ONCE
  // and passed by reference so identity is preserved across every reader.
  aborters(): Map<string, AbortController>
  roomAborters(): Map<string, AbortController>
  runningMissions(): Set<string>
  // Mission status/heartbeat forwarders — hoisted in main-app; accessor keeps the lookup lazy.
  setMissionStatusLocal(id: string, status: import('../../agent/types').MissionStatus, message?: string): Promise<import('../../agent/types').Mission>
  recordMissionHeartbeatLocal(id: string, message?: string): Promise<import('../../agent/types').Mission>
}

/**
 * Builds the `RemoteControlServerDeps` config object consumed by `createRemoteControlServer`.
 *
 * This is pure relocation of the ~110-line inline blob from main-app.ts. Every forward-declared
 * symbol (`remoteHostApi`, `jobControlRuntime`, `terminalRemoteApi`, `spineHost`, and the later
 * `aborters`/`roomAborters`/`runningMissions` collections) is reached through an accessor so the
 * config can be assembled at module scope while the resolved values are read at request time —
 * mirroring the original closures verbatim.
 */
export function buildRemoteControlConfig(deps: RemoteControlConfigDeps): RemoteControlServerDeps {
  const {
    pairingRegistry,
    persistPairingSessions,
    spineHost,
    remoteHostApi,
    jobControlRuntime,
    terminalRemoteApi,
    aborters,
    roomAborters,
    runningMissions,
    setMissionStatusLocal,
    recordMissionHeartbeatLocal,
  } = deps

  return {
    getSettings: () => store.settings.remoteControl,
    getDeviceEngine: async () => {
      const snapshot = await engineService()?.snapshot()
      return { status: snapshot?.status ?? null, performance: snapshot?.performance ?? null }
    },
    pairing: {
      issuePairingLink: async (label) => {
        return durablePairingMutation(pairingRegistry, () => pairingRegistry.issuePairingLink(label), persistPairingSessions)
      },
      authorizeSessionToken: (token) => pairingRegistry.authorizeSessionToken(token),
      redeemPairingLink: async (credential, clientInfo) => {
        return durablePairingMutation(
          pairingRegistry,
          () => pairingRegistry.redeemPairingLink(credential, clientInfo),
          persistPairingSessions,
          { changed: (result) => result.ok },
        )
      },
      listSessions: () => pairingRegistry.listSessions(),
      revokeSession: async (id) => {
        // Revocation is a security boundary: never acknowledge success until the protected
        // snapshot is durable. On write failure the token remains revoked in-memory and the
        // caller receives an error instead of a false-success response.
        return durablePairingMutation(
          pairingRegistry,
          () => pairingRegistry.revokeSession(id),
          persistPairingSessions,
          { changed: (revoked) => revoked, rollbackOnFailure: false },
        )
      },
    },
    // The built renderer bundle, so a paired browser loads the SAME app this window runs
    // (VST-035). Dev (vite server) has no dist next to dist-electron → serving degrades to
    // an explanatory 404 rather than guessing at the vite URL.
    staticRoot: () => path.join(__dirname, '../dist'),
    // Durable spine replay for remote clients (VST-034); degrades to an empty page in the
    // pre-ready window before the spine restore completes.
    spineReadSince: (since, options) => spineHost()?.readSince(since, options) ?? { events: [], hasMore: false, snapshotRequired: false },
    spineReceipt: (commandId) => spineHost()?.receipt(commandId) ?? null,
    getSnapshot: () => remoteControlSnapshot({
      appVersion: app.getVersion(),
      headless: HEADLESS_MODE,
      activeThreadIds: Array.from(aborters().keys()),
      activeRoomIds: Array.from(roomAborters().keys()),
      activeMissionIds: Array.from(runningMissions()),
    }),
    getArchives: () => archiveSnapshot(),
    getPairedSnapshot: () => pairedRemoteControlSnapshot({
      appVersion: app.getVersion(),
      headless: HEADLESS_MODE,
      activeThreadIds: Array.from(aborters().keys()),
      activeRoomIds: Array.from(roomAborters().keys()),
      activeMissionIds: Array.from(runningMissions()),
    }),
    getPairedArchives: () => pairedRemoteArchiveSnapshot(archiveSnapshot()),
    redactPairedEvent: redactRemoteControlEventForPaired,
    getDoctor: () => remoteControlDoctorSnapshot({
      appVersion: app.getVersion(),
      headless: HEADLESS_MODE,
      activeThreadIds: Array.from(aborters().keys()),
      activeRoomIds: Array.from(roomAborters().keys()),
      activeMissionIds: Array.from(runningMissions()),
    }),
    createThread: (input, scope) => remoteHostApi().createThreadOnThisHost(input, scope),
    startThread: (payload, scope) => remoteHostApi().startRemoteThreadOnThisHost(payload, scope),
    guideThread: (payload, scope) => remoteHostApi().guideRemoteThreadOnThisHost(payload, scope),
    cancelThread: (threadId) => remoteHostApi().cancelRemoteThreadOnThisHost(threadId),
    approve: (callId, decision) => jobControlRuntime().resolveToolApproval(callId, decision),
    updateProject: (id, patch) => remoteHostApi().updateProjectOnThisHost(id, patch),
    deleteProject: (id) => remoteHostApi().archiveProjectOnThisHost(id),
    restoreProject: (id) => remoteHostApi().restoreProjectOnThisHost(id),
    deleteThread: (id) => remoteHostApi().archiveThreadOnThisHost(id),
    restoreThread: (id) => remoteHostApi().restoreThreadOnThisHost(id),
    createRoom: (input) => remoteHostApi().createRoomOnThisHost(input),
    updateRoom: (id, patch) => remoteHostApi().updateRoomOnThisHost(id, patch),
    renameRoom: (id, name) => remoteHostApi().renameRoomOnThisHost(id, name),
    steerRoom: (id, text, scope) => remoteHostApi().steerRoomOnThisHost(id, text, scope),
    mutateRoomMemory: (id, command) => remoteHostApi().mutateRoomMemoryOnThisHost(id, command),
    deleteRoom: (id) => remoteHostApi().archiveRoomOnThisHost(id),
    restoreRoom: (id) => remoteHostApi().restoreRoomOnThisHost(id),
    startRoom: (roomId, problem, attachments, scope) => remoteHostApi().startRemoteRoomOnThisHost(roomId, problem, attachments, scope),
    cancelRoom: (roomId) => remoteHostApi().cancelRemoteRoomOnThisHost(roomId),
    createTicket: (input) => remoteHostApi().createTicketOnThisHost(input),
    startTicketExecution: (id, input, scope) => remoteHostApi().startRemoteTicketExecutionOnThisHost(id, input, scope),
    updateTicket: (id, patch) => remoteHostApi().updateTicketOnThisHost(id, patch),
    transitionTicket: (id, command) => remoteHostApi().transitionRemoteTicketOnThisHost(id, command),
    deleteTicket: (id) => remoteHostApi().archiveTicketOnThisHost(id),
    restoreTicket: (id) => remoteHostApi().restoreTicketOnThisHost(id),
    setMissionStatus: (id, status, message) => setMissionStatusLocal(id, status, message),
    recordMissionHeartbeat: (id, message) => recordMissionHeartbeatLocal(id, message),
    listDirectories: (dir, options) => remoteHostApi().listHostDirectories(dir, options),
    createProject: (input, scope) => remoteHostApi().createProjectOnThisHost(input, scope),
    listClis: () => terminalRemoteApi()?.listClis() ?? [],
    startTerminal: (input) => { const api = terminalRemoteApi(); if (!api) throw new Error('Terminals are not ready yet.'); return api.start(input) },
    writeTerminal: (id, data) => { const api = terminalRemoteApi(); if (!api) throw new Error('Terminals are not ready yet.'); api.write(id, data) },
    resizeTerminal: (id, cols, rows) => { const api = terminalRemoteApi(); if (!api) throw new Error('Terminals are not ready yet.'); api.resize(id, cols, rows) },
    renameTerminal: (id, title) => { const api = terminalRemoteApi(); if (!api) throw new Error('Terminals are not ready yet.'); api.rename(id, title) },
    killTerminal: (id) => { const api = terminalRemoteApi(); if (!api) throw new Error('Terminals are not ready yet.'); api.kill(id) },
    listTerminals: (projectId) => terminalRemoteApi()?.list(projectId) ?? [],
    onClientSeen: (at) => {
      store.settings = { ...store.settings, remoteControl: { ...store.settings.remoteControl, lastClientAt: at, lastError: null } }
      persistSettings('remoteControl.lastClientAt')
    },
    onStarted: (at, error) => {
      store.settings = { ...store.settings, remoteControl: { ...store.settings.remoteControl, lastStartedAt: at, lastError: error } }
      persistSettings('remoteControl.lastStartedAt')
    },
    onError: (message) => {
      store.settings = { ...store.settings, remoteControl: { ...store.settings.remoteControl, lastError: message } }
      persistSettings('remoteControl.lastError')
    },
    audit: (kind, data) => { void auditEvent(kind, data) },
  }
}
