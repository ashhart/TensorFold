import type { CoreIpcDeps } from './core-ipc'
import type { EventSpineHost } from '../../agent/event-spine-host'

export interface CoreIpcConfigDeps {
  // Forward-declared / late-bound collaborators — passed as accessors where the value is
  // assigned after this factory runs, direct references otherwise. Built here exactly as the
  // original inline `registerCoreIpc({...})` literal composed them.
  clipboard: CoreIpcDeps['clipboard']
  remoteServer: CoreIpcDeps['remoteServer']
  sshManager: CoreIpcDeps['sshManager']
  mcpHost: CoreIpcDeps['mcpHost']
  aborters: CoreIpcDeps['aborters']
  roomAborters: CoreIpcDeps['roomAborters']
  abortAndWaitMissionsForArchive: CoreIpcDeps['abortMissionsForArchive']
  loopHost: { stopLoopsForArchive(kind: Parameters<CoreIpcDeps['stopLoopsForArchive']>[0], id: string): Promise<unknown> }
  killProjectTerminals: CoreIpcDeps['killProjectTerminals']
  workspaceMutationGuard: CoreIpcDeps['workspaceMutationGuard']
  purgeLocalMemory: CoreIpcDeps['purgeLocalMemory']
  confirmSensitiveAction: CoreIpcDeps['confirmSensitiveAction']
  normalizeRemoteOrchestratorDestinations: CoreIpcDeps['normalizeRemoteOrchestratorDestinations']
  ensureRemoteControlToken: CoreIpcDeps['ensureRemoteControlToken']
  applyTheme: CoreIpcDeps['applyTheme']
  applyPowerSettings: CoreIpcDeps['applyPowerSettings']
  applyDesktopMode?: CoreIpcDeps['applyDesktopMode']
  hideDesktopWindow?: CoreIpcDeps['hideDesktopWindow']
  startUpdateChecks: CoreIpcDeps['startUpdateChecks']
  startBotControlPolling: CoreIpcDeps['startBotControlPolling']
  webhookServer: { syncFromEndpoints(): Promise<void> }
  reloadPlugins: CoreIpcDeps['reloadPlugins']
  pollTelegramUpdates: CoreIpcDeps['pollTelegramUpdates']
  registerConnectorIpc: CoreIpcDeps['registerConnectorIpc']
  getMission: CoreIpcDeps['getMission']
  createLocalPrLoop: CoreIpcDeps['createLocalPrLoop']
  emitToRenderer: CoreIpcDeps['emitToRenderer']
  rememberConnectorEvents: CoreIpcDeps['rememberConnectorEvents']
  appName: CoreIpcDeps['appName']
  // Late-bound spine accessor (assigned during app-ready spine restore).
  spineHost(): EventSpineHost | null
}

/**
 * Builds the `CoreIpcDeps` config consumed by `registerCoreIpc`.
 *
 * Pure relocation of the ~31-line inline blob from main-app.ts. The two computed entries
 * (stopLoopsForArchive wrapper, spine accessor) are assembled here from the same
 * collaborators the inline closures used; everything else is forwarded verbatim.
 */
export function buildCoreIpcConfig(deps: CoreIpcConfigDeps): CoreIpcDeps {
  const {
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
    applyDesktopMode,
    hideDesktopWindow,
    startUpdateChecks,
    startBotControlPolling,
    webhookServer,
    reloadPlugins,
    pollTelegramUpdates,
    registerConnectorIpc,
    getMission,
    createLocalPrLoop,
    emitToRenderer,
    rememberConnectorEvents,
    appName,
    spineHost,
  } = deps

  return {
    clipboard,
    remoteServer,
    sshManager,
    mcpHost,
    aborters,
    roomAborters,
    abortMissionsForArchive: abortAndWaitMissionsForArchive,
    stopLoopsForArchive: (kind: Parameters<CoreIpcDeps['stopLoopsForArchive']>[0], id: string) => loopHost.stopLoopsForArchive(kind, id),
    killProjectTerminals,
    workspaceMutationGuard,
    purgeLocalMemory,
    confirmSensitiveAction,
    normalizeRemoteOrchestratorDestinations,
    ensureRemoteControlToken,
    applyTheme,
    applyPowerSettings,
    applyDesktopMode,
    hideDesktopWindow,
    startUpdateChecks,
    startBotControlPolling,
    syncWebhookServer: () => webhookServer.syncFromEndpoints(),
    reloadPlugins,
    pollTelegramUpdates,
    registerConnectorIpc,
    getMission,
    createLocalPrLoop,
    emitToRenderer,
    rememberConnectorEvents,
    appName,
    // Accessor, not the instance: core-ipc registers at module scope, the spine restores
    // asynchronously at app-ready. Handles degrade gracefully in the pre-ready window.
    spine: () => spineHost(),
  }
}
