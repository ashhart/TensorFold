import { forgetWhatsAppBridgeRuntime, getWhatsAppBridgeLifecycleEpoch, getWhatsAppBridgeRuntime, makeWhatsAppBridgeSource, normalizeWhatsAppConversationJid, validateWhatsAppSourcePatch, sendWhatsAppMessage, startWhatsAppBridge, stopWhatsAppBridge, whatsappEventFromMessage, whatsappPairingErrorPatch } from '../../../agent/whatsapp-bridge'
import { createProtectedWhatsAppAuthState, hasProtectedRegisteredWhatsAppAuthState, shredWhatsAppAuthState } from '../whatsapp-auth-state'
import { saveSettings, vontraPath } from '../../../store'
import { persistSettings } from '../persistence'
import { auditEvent, requireSecurityRole } from '../audit'
import { auditPreview } from '../../../agent/security'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { removeWhatsAppSourceWithNativeConfirmation } from '../connector-runtime'
import { publicConnectorSettings } from '../connector-event-spine'
import type { ConnectorIpcDeps } from '../connector-ipc'
import { createSerialTaskQueue } from '../../../agent/serial-task-queue'

const accountEdits = createSerialTaskQueue()

function publishWhatsAppStatus(id: string, patch: Partial<import('../../../agent/types').WhatsAppBridgeSource>): void {
  if (!store.settings.whatsapp.sources.some((source) => source.id === id)) return
  store.settings = { ...store.settings, whatsapp: { ...store.settings.whatsapp, sources: store.settings.whatsapp.sources.map((source) => source.id === id ? { ...source, ...patch, updatedAt: Date.now() } : source) } }
  persistSettings('whatsapp.status')
  store.win?.webContents.send('whatsapp:status', { sourceId: id, ...patch })
}

function startAccount(id: string, deps: Pick<ConnectorIpcDeps, 'channels'>, requireRegistered: boolean): Promise<unknown> {
  const source = store.settings.whatsapp.sources.find((candidate) => candidate.id === id)
  if (!source?.enabled) throw new Error('Enable this WhatsApp account before connecting.')
  return startWhatsAppBridge(source, vontraPath('.'),
    (qr) => store.win?.webContents.send('whatsapp:qr', { sourceId: id, qr }),
    (event) => deps.channels.rememberWhatsAppEvents([event]),
    (patch) => publishWhatsAppStatus(id, patch), createProtectedWhatsAppAuthState,
    {
      requireRegistered,
      getSource: () => store.settings.whatsapp.sources.find((candidate) => candidate.id === id),
      onRuntime: (snapshot) => store.win?.webContents.send('whatsapp:runtime', snapshot),
    },
  ).catch((error: unknown) => {
    // Do not let late start errors resurrect a removed/disabled account.
    if (!store.settings.whatsapp.sources.some((candidate) => candidate.id === id && candidate.enabled)) return
    // startWhatsAppBridge already publishes only while it still owns this generation.
    // A cancelled older start must not overwrite a replacement session's current status.
    if (!getWhatsAppBridgeRuntime(id).some((snapshot) => snapshot.phase === 'error')) return
    void auditEvent('whatsapp.connection_failed', { sourceId: id, error: auditPreview(whatsappPairingErrorPatch(error).lastError ?? '') })
  })
}

/** Invoke once after settings/auth initialization; never creates auth or pairs an unlinked phone. */
export async function resumeSavedWhatsAppAccounts(deps: Pick<ConnectorIpcDeps, 'channels'>): Promise<void> {
  const epoch = getWhatsAppBridgeLifecycleEpoch()
  const active = () => getWhatsAppBridgeLifecycleEpoch() === epoch
  for (const source of [...store.settings.whatsapp.sources]) {
    if (!active()) return
    if (!source.enabled || !source.paired) continue
    try {
      const registered = await hasProtectedRegisteredWhatsAppAuthState(vontraPath('.'), source.id)
      // Quit/purge can stop every bridge while protected-auth I/O is in progress.
      // A later start must not capture the newer generation and reopen that fence.
      if (!active()) return
      if (!registered) {
        publishWhatsAppStatus(source.id, { paired: false, lastError: 'Saved protected credentials are unavailable; link the phone again.' })
        continue
      }
      // Re-read after protected disk I/O: remove/disable wins over startup.
      const current = store.settings.whatsapp.sources.find((candidate) => candidate.id === source.id)
      if (current?.enabled && current.paired && active()) await startAccount(source.id, deps, true)
      if (!active()) return
    } catch (error) {
      if (!active()) return
      publishWhatsAppStatus(source.id, { lastError: whatsappPairingErrorPatch(error).lastError })
    }
  }
}

export function registerWhatsAppIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, callbacks } = deps
  const { rememberConnectorEvents } = spine

  handle('whatsapp:runtime', async () => {
    requireSecurityRole('owner', 'view WhatsApp pairing state')
    return getWhatsAppBridgeRuntime().filter((snapshot) => store.settings.whatsapp.sources.some((source) => source.id === snapshot.sourceId))
  })
  handle('whatsapp:updateSource', async (_e, input: { id: string; patch: unknown }) => {
    requireSecurityRole('owner', 'edit WhatsApp account scope')
    if (!input || typeof input !== 'object' || typeof input.id !== 'string') throw new Error('Choose a valid WhatsApp account before editing its settings.')
    const { id, patch } = input
    const validated = validateWhatsAppSourcePatch(patch)
    return accountEdits.run('whatsapp', async () => {
      const current = store.settings
      if (!current.whatsapp.sources.some((source) => source.id === id)) throw new Error('WhatsApp account not found.')
      const next = { ...current, whatsapp: { ...current.whatsapp, sources: current.whatsapp.sources.map((source) => source.id === id ? { ...source, ...validated, updatedAt: Date.now() } : source) } }
      // No in-memory scope broadening until durable storage accepts this exact edit.
      await saveSettings(next)
      if (store.settings !== current) {
        await saveSettings(store.settings)
        throw new Error('WhatsApp account settings changed while saving; retry your edit.')
      }
      store.settings = next
      if (validated.enabled === false) stopWhatsAppBridge(id)
      return publicConnectorSettings().whatsapp
    })
  })

  handle('whatsapp:addSource', async (_e, label: string) => {
    requireSecurityRole('owner', 'add WhatsApp bridge accounts')
    const source = makeWhatsAppBridgeSource(label)
    store.settings = { ...store.settings, whatsapp: { ...store.settings.whatsapp, sources: [...store.settings.whatsapp.sources, source] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().whatsapp
  })

  handle('whatsapp:removeSource', async (_e, id: string) => {
    requireSecurityRole('owner', 'remove WhatsApp bridge accounts')
    const source = store.settings.whatsapp.sources.find((candidate) => candidate.id === id)
    if (!source) throw new Error('WhatsApp account not found.')
    await removeWhatsAppSourceWithNativeConfirmation(source, callbacks.confirmSensitiveAction, {
      stop: stopWhatsAppBridge,
      // Removal is a credential wipe, not just a settings-row delete. The helper overwrites
      // regular files and unlinks (never follows) symlinks; a failure keeps the source visible.
      shred: (sourceId) => shredWhatsAppAuthState(vontraPath('.'), sourceId),
      remove: async (sourceId) => {
        store.settings = { ...store.settings, whatsapp: { ...store.settings.whatsapp, sources: store.settings.whatsapp.sources.filter((candidate) => candidate.id !== sourceId) } }
        await saveSettings(store.settings)
        forgetWhatsAppBridgeRuntime(sourceId)
      },
    })
    return publicConnectorSettings().whatsapp
  })

  handle('whatsapp:startPairing', async (_e, id: string) => {
    requireSecurityRole('owner', 'pair WhatsApp accounts')
    const source = store.settings.whatsapp.sources.find((candidate) => candidate.id === id)
    if (!source) throw new Error('WhatsApp account not found.')
    stopWhatsAppBridge(id)
    publishWhatsAppStatus(id, { enabled: true, lastError: null })
    void startAccount(id, deps, false)
    return { started: true }
  })

  handle('whatsapp:disconnect', async (_e, id: string) => {
    requireSecurityRole('owner', 'disconnect WhatsApp accounts')
    if (!store.settings.whatsapp.sources.some((source) => source.id === id)) throw new Error('WhatsApp account not found.')
    // Disable auto-resume too; retained credentials allow an explicit Reconnect later.
    stopWhatsAppBridge(id)
    publishWhatsAppStatus(id, { enabled: false, lastError: null })
    await saveSettings(store.settings)
    return { ok: true }
  })

  handle('whatsapp:send', async (_e, { sourceId, to, text }: { sourceId: string; to: string; text: string }) => {
    // Direct owner-authored sends use the same connected transport; agent-composed
    // sends must reach the channel adapter only after immutable approval review.
    requireSecurityRole('owner', 'send WhatsApp messages')
    const result = await sendWhatsAppMessage(sourceId, to, text)
    if (result.ok) rememberConnectorEvents([whatsappEventFromMessage({ sourceId, jid: normalizeWhatsAppConversationJid(to)!, fromMe: true, text, sender: 'You', ts: Date.now(), messageId: result.sentId, authenticatedAccountJid: result.authenticatedAccountJid })])
    return result
  })
}
