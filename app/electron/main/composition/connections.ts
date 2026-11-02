import { saveSettings, savePairingSessions, makeRemoteControlToken, vontraPath } from '../../store'
import { stampSettingsSecretRotations } from '../../agent/secret-rotation'
import { markRemoteJobsUnverifiable } from '../../agent/jobs'
import { McpHost } from '../../agent/mcp-client'
import { sendWhatsAppCloudReply } from '../host/whatsapp-reply'
import { confirmSensitiveAction } from '../host/sensitive-action'
import { createEmitHelpers } from '../host/emit-helpers'
import { buildRemoteControlConfig } from '../host/remote-control-config'
import { createWebhookServer } from '../../agent/webhook-server'
import { createRemoteControlServer } from '../../agent/remote-control'
import { createPairingRegistry } from '../../agent/pairing-sessions'
import { store } from '../state'
import { persistJobs, persistSettings, recordPersistenceFailure } from '../host/persistence'
import { auditEvent } from '../host/audit'
import { SshManager } from '../host/ssh'
import { emitOutboundWebhooks } from '../host/webhooks'
import { ingestBotControl } from '../host/bot-control'
import { createConnectorRuntime } from '../host/connector-runtime'
import { WorkspaceMutationGuard } from '../host/workspace-mutation-guard'
import { createMobileBridgeRuntime } from '../host/mobile-bridge-runtime'
import { createWatchCommandProjection } from '../host/watch-command-projection'
import { createWatchSpeechRuntime } from '../host/watch-speech-runtime'
import { getManagedKokoroService } from '../host/ipc/provider-ipc'
import { type AutomationRuntime } from '../host/automations'
import { rendererWindowAvailable, sendToRenderer } from '../host/renderer-events'
import { durablePairingMutation } from '../host/durable-pairing'
import { HEADLESS_MODE } from '../early-electron-paths'

interface ApplicationConnectionsDeps {
  findApprovalRule: Parameters<typeof createConnectorRuntime>[0]['findApprovalRule']
  rememberApprovalRule: Parameters<typeof createConnectorRuntime>[0]['rememberApprovalRule']
  runAutomation: AutomationRuntime['runAutomation']
  botControlRuntime(): Parameters<typeof ingestBotControl>[1]
  aborters: Map<string, AbortController>
  workspaceMutationGuard: WorkspaceMutationGuard
  saveTrustedKey: ConstructorParameters<typeof SshManager>[0]['saveTrustedKey']
  emitJob: ReturnType<typeof createEmitHelpers>['emitJob']
  remoteConfig: Omit<Parameters<typeof buildRemoteControlConfig>[0], 'pairingRegistry' | 'persistPairingSessions'>
}

export function createApplicationConnections(deps: ApplicationConnectionsDeps) {
  const { findApprovalRule, rememberApprovalRule, runAutomation, botControlRuntime,
    aborters, workspaceMutationGuard, saveTrustedKey, emitJob, remoteConfig } = deps
  const connectorRuntime = createConnectorRuntime({
    findApprovalRule,
    rememberApprovalRule,
    remotePublish: (event) => remoteServer.publish({ kind: 'connector', data: event }),
    emitOutboundWebhook: (event) => emitOutboundWebhooks('connector', event),

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

  const webhookServer = createWebhookServer({
    getEndpoints: () => store.webhooks,
    onConnectorEvent: (e) => rememberConnectorEvents([e]),
    runAutomationById: (id, source) => {
      const a = store.automations.find((x) => x.id === id)
      if (a) void runAutomation(a, {
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
    ...remoteConfig, pairingRegistry, persistPairingSessions,
  }))
  const mcpHost = new McpHost()

  const mobileBridgeRuntime = createMobileBridgeRuntime({
    projectWatchSnapshot: createWatchCommandProjection({ activeThreadIds: () => [...aborters.keys()] }),
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

  const sshManager = new SshManager({
    getHosts: () => store.settings.sshHosts ?? [],
    getTrustedKeys: () => store.settings.trustedSshHostKeys ?? [],
    interactivePrompts: () => !HEADLESS_MODE && rendererWindowAvailable(),
    saveTrustedKey: async key => { await saveTrustedKey(key) },
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

  return {
    connectorRuntime, connectorPolicyForEvent, rememberConnectorEvents,
    recordEmailDraft, recordEmailEvent, recordCalendarEvent, recordMessageEvent, registerConnectorIpc,
    webhookServer, pairingRegistry, persistPairingSessions, remoteServer, mcpHost, mobileBridgeRuntime, sshManager,
  }
}
