import type { Clipboard } from 'electron'
import { shell } from 'electron'
import { collectTokenUsage } from '../../agent/token-usage'
import { buildContextPressureSnapshot } from '../../agent/context-pressure'
import { auditPreview, isSecretMask, mergeSettingsSecrets, redactSettings } from '../../agent/security'
import { createSerialTaskQueue } from '../../agent/serial-task-queue'
import { stampSettingsSecretRotations } from '../../agent/secret-rotation'
import { assertDedicatedWhatsAppSourceRemoval } from '../../agent/settings-save'
import type { EventSpineHost } from '../../agent/event-spine-host'
import type { ArchiveKind, CreatePrLoopInput, CreatePrLoopResult, Mission, Settings } from '../../agent/types'
import type { McpHost } from '../../agent/mcp-client'
import { makeRemoteControlToken, saveSettings, vontraPath } from '../../store'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { openExternalSafely } from './external'
import { handle } from './ipc'
import { modelRouterCooldownSnapshot } from './provider-runtime'
import { securityControlSnapshot } from './security-controls'
import { requireNativeConfirmation, type ConfirmSensitiveAction } from './sensitive-confirmation'
import type { WorkspaceMutationGuard } from './workspace-mutation-guard'
import { registerRemoteControlIpc, syncRemoteEventStreams } from './ipc/remote-control-ipc'
import { registerSshIpc, type SshManagerLike } from './ipc/ssh-ipc'
import { registerWorkspaceIpc } from './ipc/workspace-ipc'
import { registerSessionIpc } from './ipc/session-ipc'
import { registerSpineIpc } from './ipc/spine-ipc'
import { registerMcpIpc } from './ipc/mcp-ipc'
import { registerProviderIpc } from './ipc/provider-ipc'
import { registerControllerIpc } from './ipc/controller-ipc'
import { registerMissionsIpc } from './ipc/missions-ipc'
import { registerBotIpc } from './ipc/bot-ipc'
import { sendToRenderer } from './renderer-events'
import { normalizeDesktopMode } from './desktop-mode'
import { normalizeAgentPresets } from '../../../shared/bot-policy'
import { reserveBotLifecycleTransition } from './bot-run-boundary'
import { getBotVmHost } from './bot-vm-host'

export { requireNativeConfirmation } from './sensitive-confirmation'
// Re-export the renderer thread patch sanitizer so existing import sites keep resolving here.
export { sanitizeRendererThreadPatch } from './ipc/workspace-ipc'

interface RemoteServerLike {
  status(): { url: string | null }
  syncFromSettings(): Promise<void>
  publish(event: Pick<import('../../agent/types').RemoteControlEvent, 'kind' | 'data'>): void
  // Per-device pairing (VST-035).
  issuePairingLink(label: string): Promise<{ credential: string; expiresAt: number }>
  listDeviceSessions(): Array<{ id: string; label: string; createdAt: number; lastSeenAt: number; revoked: boolean; clientInfo?: string | null }>
  revokeDeviceSession(id: string): Promise<boolean>
}

export interface CoreIpcDeps {
  clipboard: Clipboard
  remoteServer: RemoteServerLike
  // Ordered event spine accessor (VST-034); null until the async boot restore completes.
  spine: () => EventSpineHost | null
  sshManager: SshManagerLike
  mcpHost: McpHost
  aborters: Map<string, AbortController>
  roomAborters: Map<string, AbortController>
  abortMissionsForArchive(target: { projectId?: string | null; threadId?: string | null }): void
  stopLoopsForArchive(kind: ArchiveKind, id: string): Promise<unknown>
  killProjectTerminals(projectId: string): void
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  purgeLocalMemory(): Promise<unknown>
  confirmSensitiveAction: ConfirmSensitiveAction
  normalizeRemoteOrchestratorDestinations(next: Settings, current: Settings): Promise<Settings>
  ensureRemoteControlToken(): void
  applyTheme(): void
  applyPowerSettings(): void
  applyDesktopMode?(): void
  hideDesktopWindow?(): { hidden: boolean }
  startUpdateChecks(): void
  startBotControlPolling(): void
  syncWebhookServer(): Promise<void>
  reloadPlugins(settings: Settings): Promise<void>
  pollTelegramUpdates(): Promise<unknown>
  registerConnectorIpc(): void
  getMission(id: string): Mission
  createLocalPrLoop(input: CreatePrLoopInput): Promise<CreatePrLoopResult>
  emitToRenderer(event: any): void
  rememberConnectorEvents(events: any[]): void
  appName: string
}

export function registerCoreIpc(deps: CoreIpcDeps) {
  const {
    clipboard,
    remoteServer,
    sshManager,
    mcpHost,
    aborters,
    roomAborters,
    abortMissionsForArchive,
    stopLoopsForArchive,
    killProjectTerminals,
    workspaceMutationGuard,
    purgeLocalMemory,
    confirmSensitiveAction,
    normalizeRemoteOrchestratorDestinations,
    ensureRemoteControlToken,
    applyTheme,
    applyPowerSettings,
    startUpdateChecks,
    startBotControlPolling,
    syncWebhookServer,
    reloadPlugins,
    pollTelegramUpdates,
    registerConnectorIpc,
    getMission,
    createLocalPrLoop,
    emitToRenderer,
    rememberConnectorEvents,
    appName,
    spine,
  } = deps

  const settingsChanges = createSerialTaskQueue()
  async function commitSettingsChange(
    prepare: (current: Settings) => Promise<Settings>,
    afterCommit: (next: Settings) => Promise<void>,
  ): Promise<Settings> {
    const reservation = workspaceMutationGuard.reserveStart({ workspaceKeys: ['settings:configuration'] })
    try {
      return await settingsChanges.run('settings', async () => {
        const current = store.settings
        const currentJson = JSON.stringify(current)
        const stillCurrent = () => store.settings === current && JSON.stringify(store.settings) === currentJson
        const next = await prepare(current)
        if (!stillCurrent()) throw new Error('Settings changed while preparing this save. Retry to use the current configuration.')
        await saveSettings(next)
        if (!stillCurrent()) {
          // A background trust/rule/connector update owns the newer live snapshot.
          // Restore it durably before rejecting this stale full-settings transaction.
          await saveSettings(store.settings)
          throw new Error('Settings changed while saving. The current configuration was preserved; retry your edit.')
        }
        store.settings = next
        try {
          await afterCommit(next)
        } catch (error) {
          console.warn('[settings] Saved configuration could not be applied:', auditPreview(error))
          throw new Error('Settings were saved, but a runtime service could not apply them. Check the connection status and retry.')
        }
        return redactSettings(store.settings)
      })
    } finally {
      reservation.release()
    }
  }

  // --- domain IPC modules ---
  registerRemoteControlIpc({ remoteServer, confirmSensitiveAction, ensureRemoteControlToken })
  registerSshIpc({ sshManager })
  registerWorkspaceIpc({
    remoteServer,
    spine,
    aborters,
    roomAborters,
    abortMissionsForArchive,
    stopLoopsForArchive,
    killProjectTerminals,
    workspaceMutationGuard,
    emitToRenderer,
  })
  registerSessionIpc({ spine, remoteServer })
  registerSpineIpc({ spine })
  registerMcpIpc({ mcpHost, confirmSensitiveAction })
  registerProviderIpc()
  registerControllerIpc({ appName, rememberConnectorEvents })
  registerMissionsIpc({ getMission, createLocalPrLoop })
  registerBotIpc({ pollTelegramUpdates, recipes: {
    currentSettings: () => store.settings,
    commitImport: (prepare) => commitSettingsChange(prepare, async () => {}),
  } })

// ---------------------------------------------------------------- IPC

handle('security:getControls', async () => securityControlSnapshot())
handle('security:purgeLocalMemory', async () => {
  requireSecurityRole('owner', 'purge all local memory/state')
  await requireNativeConfirmation(confirmSensitiveAction, {
    title: 'Erase local TensorFold memory?',
    message: 'This permanently removes local conversations, work records, paired sessions, protected runtime state, and private Living Weights models and learned facts.',
    detail: 'Original downloaded models and configured provider credentials remain. This confirmation is shown by the main process so renderer content cannot silently trigger the wipe.',
    confirmLabel: 'Erase local memory',
  })
  return purgeLocalMemory()
})

// Open the macOS Full Disk Access pane so the user can grant TensorFold access to
// protected databases (iMessage chat.db). The URL scheme is Apple's official deep
// link into System Settings; it only opens the pane, it can't auto-grant.
handle('system:openFullDiskAccess', async () => {
  // macOS-only; no-op elsewhere so the renderer can call it unconditionally.
  if (process.platform !== 'darwin') return
  await shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles')
})

handle('system:openExternal', async (_e, url: string) => {
  await openExternalSafely(url)
})

handle('settings:get', async () => {
  syncRemoteEventStreams()
  return redactSettings(store.settings)
})
async function setDesktopMode(mode: unknown, onboardingComplete?: unknown): Promise<Settings> {
  requireSecurityRole('owner', 'change desktop mode')
  if (mode !== 'minimal' && mode !== 'full') throw new Error('Choose Minimal or Full mode.')
  if (onboardingComplete !== undefined && typeof onboardingComplete !== 'boolean') throw new Error('Onboarding completion must be true or false.')
  return commitSettingsChange(async (current) => ({
    ...current, desktopMode: mode,
    ...(onboardingComplete === undefined ? {} : { desktopOnboardingComplete: onboardingComplete }),
  }), async () => {
    await auditEvent('desktop.mode', { mode, onboardingComplete: onboardingComplete ?? null })
    deps.applyDesktopMode?.()
    sendToRenderer('desktop:modeChanged', redactSettings(store.settings))
  })
}

handle('desktop:modeSet', async (_e, input: { mode?: unknown; onboardingComplete?: unknown }) => setDesktopMode(input?.mode, input?.onboardingComplete))
handle('desktop:hide', async () => {
  requireSecurityRole('operator', 'hide the desktop window')
  return deps.hideDesktopWindow?.() ?? { hidden: false }
})

handle('settings:save', async (_e, s: Settings) => {
  requireSecurityRole('owner', 'save settings')
  const releaseBotTransitions: Array<() => void> = []
  try { return await commitSettingsChange(async (current) => {
    assertDedicatedWhatsAppSourceRemoval(s, current)
    const merged = mergeSettingsSecrets(s, current)
    merged.agentPresets = normalizeAgentPresets(merged.agentPresets ?? [], true)
    // Renderer snapshots never own approval rules. Merge masks against the live
    // configuration only once this transaction reaches the head of the queue.
    let next = await normalizeRemoteOrchestratorDestinations({ ...merged, desktopMode: normalizeDesktopMode(merged.desktopMode), desktopOnboardingComplete: merged.desktopOnboardingComplete === true, approvalRules: current.approvalRules }, current)
    for (const preset of current.agentPresets) {
      const nextPreset = next.agentPresets.find((candidate) => candidate.id === preset.id)
      if (nextPreset && !nextPreset.paused && !nextPreset.archivedAt) continue
      if ((preset.paused || preset.archivedAt) && nextPreset && preset.paused === nextPreset.paused && (preset.archivedAt ?? null) === (nextPreset.archivedAt ?? null)) continue
      releaseBotTransitions.push(reserveBotLifecycleTransition(preset.id))
      for (const thread of store.threads) if (thread.bot?.presetId === preset.id) aborters.get(thread.id)?.abort()
      if (preset.executionTarget === 'vm' || store.threads.some((thread) => thread.bot?.presetId === preset.id && thread.bot.executionTarget === 'vm')) await getBotVmHost()?.stop(preset.id)
    }
    if (next.remoteControl.enabled && (!next.remoteControl.authToken || isSecretMask(next.remoteControl.authToken))) {
      next = { ...next, remoteControl: { ...next.remoteControl, authToken: makeRemoteControlToken() } }
    }
    return stampSettingsSecretRotations(next, current)
  }, async (next) => {
    for (const thread of store.threads) {
      if (!thread.bot) continue
      const preset = next.agentPresets.find((candidate) => candidate.id === thread.bot!.presetId)
      if (!preset || preset.paused || preset.archivedAt) aborters.get(thread.id)?.abort()
    }
    await auditEvent('settings.save', { providers: next.providers.length, defaultCwd: next.defaultCwd ? auditPreview(next.defaultCwd) : null })
    deps.applyDesktopMode?.()
    sendToRenderer('desktop:modeChanged', redactSettings(store.settings))
    await reloadPlugins(next)
    applyTheme()
    applyPowerSettings()
    await remoteServer.syncFromSettings()
    syncRemoteEventStreams()
    startUpdateChecks()
    startBotControlPolling()
    await syncWebhookServer()
  }) } finally { for (const release of releaseBotTransitions) release() }
})

handle('approval:revokeRule', async (_e, id: string) => {
  requireSecurityRole('owner', 'revoke approval policies')
  const ruleId = String(id || '').trim()
  let existing: NonNullable<Settings['approvalRules']>[number] | undefined
  return commitSettingsChange(async (current) => {
    existing = (current.approvalRules ?? []).find((rule) => rule.id === ruleId)
    if (!existing) return current
    return { ...current, approvalRules: (current.approvalRules ?? []).filter((rule) => rule.id !== ruleId) }
  }, async () => {
    if (!existing) return
    await auditEvent('approval.rule_revoked', {
      scope: existing.scope, key: existing.key, legacyTool: existing.legacyTool ?? null,
      label: existing.label ? auditPreview(existing.label) : null,
      explanation: existing.explanation ? auditPreview(existing.explanation) : null,
    })
  })
})

handle('clipboard:writeText', async (_e, text: string) => {
  clipboard.writeText(String(text ?? ''))
  return { ok: true }
})

handle('usage:tokens', async () => {
  return collectTokenUsage(vontraPath('usage'))
})

handle('usage:context-pressure', async () => {
  return buildContextPressureSnapshot(store.settings, store.threads)
})

handle('modelRouter:snapshot', async () => {
  return modelRouterCooldownSnapshot()
})

registerConnectorIpc()

  return { setDesktopMode }
}
