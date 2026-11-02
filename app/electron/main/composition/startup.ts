import { loadClusterWorkspaceIdleState } from '../host/cluster-runtime'
import { recoverWorkspaceBackupCheckpoints } from '../host/workspace-backup/recovery'
import { readProtectedJson, writeProtectedJson } from '../../store'
import { recoverInterruptedConnectorDeliveries } from '../host/connector-outbound-delivery'
import { app, BrowserWindow } from 'electron'
import path from 'node:path'
import { promises as fs } from 'node:fs'
import { ensureTensorFoldStudioHome, loadSettings, saveSettings, loadThreads, loadProjects, loadRooms, loadTickets, loadAutomations, loadConnectorEvents, loadContactDirectory, loadCliSessions, loadMissions, loadJobs, loadSubAgents, loadSkills, loadCommands, loadSuggestions, loadMcpServers, loadBotControlSessions, loadProjectMemory, loadProjectMemoryCandidates, loadPairingSessions, loadArtifacts, loadProviderStatus, loadSchedulerLease, loadWebhooks, makeRemoteControlToken, purgeLegacyUserDataState, vontraPath, writeTensorFoldStudioStateJson } from '../../store'
import { cleanupMediaCache } from '../../agent/media-cache'
import { normalizeServerConfig } from '../../agent/mcp-client'
import { reconcileJobsAndMissionsAfterRestart, reconcileIncidentsAfterRestart } from '../host/reconcile-after-restart'
import { migrateLegacyUserData } from '../host/boot-helpers'
import { bootAppState } from '../host/boot-sequence'
import type { Settings } from '../../agent/types'
import { loadIncidents, vontraHomeDir } from '../../store'
import { createEventSpineHost } from '../../agent/event-spine-host'
import { store } from '../state'
import { persistIncidents, persistJobs, persistMissions, recordPersistenceFailure } from '../host/persistence'
import { auditEvent } from '../host/audit'
import { applyPowerSettings } from '../host/power'
import { startUpdateChecks } from '../host/updates'
import { ensureCliPath } from '../../agent/cli-path'
import { enableOrchestrationJournal } from '../../agent/orchestration-journal'
import { wireOrchestrationEventsToWindow } from '../host/orchestration-events'
import { hostPluginRuntime } from '../../agent/plugin-runtime'
import { normalizeArtifactRecord } from '../../agent/artifacts'
import { sendToRenderer } from '../host/renderer-events'
import { headlessBootstrapMessages, prepareHeadlessRemoteControl } from '../host/headless-bootstrap'
import { durablePairingMutation } from '../host/durable-pairing'
import { HEADLESS_MODE } from '../early-electron-paths'
import type { BootStateDeps } from '../host/boot-sequence'
import { persistWorkspaceSnapshot } from './workspace-services'

export function registerApplicationErrors(): void {
  process.on('unhandledRejection', (reason) => {
    console.error('[main] unhandledRejection', reason)
  })
  process.on('uncaughtException', (error) => {
    console.error('[main] uncaughtException', error)
  })
}

export async function loadConfiguredPlugins(settings: Settings): Promise<void> {
  hostPluginRuntime.clear()
  const configured = settings.capabilities?.packagedPlugins ?? []
  for (const plugin of configured) {

    if (plugin.enabled !== true || plugin.trusted !== true || !plugin.manifestPath) continue
    const result = await hostPluginRuntime.loadPlugin(path.dirname(path.resolve(plugin.manifestPath)))
    if (!result.ok) console.error(`[plugin-runtime] Failed to load ${plugin.name || plugin.id}: ${result.error}`)
  }
}

type ApplicationStartupDeps = Pick<BootStateDeps,
  'pairingRegistry'
  | 'persistPairingSessions'
  | 'checkpoints'
  | 'providerStatusRuntime'
  | 'mcpHost'
  | 'remoteServer'
  | 'webhookServer'
  | 'upgradeGeneratedClaudeGuidance'
  | 'reconcileStuckJobs'
  | 'installSessionHardening'
  | 'createWindow'
  | 'initializeDesktopMode'
  | 'showDesktopWindow'
  | 'applyTheme'
  | 'startScheduler'
  | 'startBotControlPolling'
  | 'resumeAutoJobsAfterRestart'
  | 'getSpineHost'
  | 'setSpineHost'
  | 'setBootSchedulerLease'>

export function bootApplication(deps: ApplicationStartupDeps): Promise<void> {
  const { pairingRegistry, persistPairingSessions, checkpoints, providerStatusRuntime, mcpHost, remoteServer, webhookServer, upgradeGeneratedClaudeGuidance, reconcileStuckJobs, installSessionHardening, createWindow, initializeDesktopMode, showDesktopWindow, applyTheme, startScheduler, startBotControlPolling, resumeAutoJobsAfterRestart, getSpineHost, setSpineHost, setBootSchedulerLease } = deps
  return app.whenReady().then(() => bootAppState({
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
    upgradeGeneratedClaudeGuidance,
    reconcileStuckJobs,
    installSessionHardening,
    createWindow,
    initializeDesktopMode,
    showDesktopWindow,
    applyTheme,
    applyPowerSettings,
    startScheduler,
    startUpdateChecks,
    startBotControlPolling,
    resumeAutoJobsAfterRestart,
    getSpineHost,
    setSpineHost,
    setBootSchedulerLease,
  }))
}
