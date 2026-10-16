import { makeWhatsAppBridgeSource, sendWhatsAppMessage, startWhatsAppBridge, stopWhatsAppBridge, whatsappEventFromMessage, whatsappPairingErrorPatch } from '../../../agent/whatsapp-bridge'
import { createProtectedWhatsAppAuthState, shredWhatsAppAuthState } from '../whatsapp-auth-state'
import { saveSettings, vontraPath } from '../../../store'
import { persistSettings } from '../persistence'
import { auditEvent, requireSecurityRole } from '../audit'
import { auditPreview } from '../../../agent/security'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { removeWhatsAppSourceWithNativeConfirmation } from '../connector-runtime'
import { publicConnectorSettings } from '../connector-event-spine'
import type { ConnectorIpcDeps } from '../connector-ipc'

export function registerWhatsAppIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, channels, callbacks } = deps
  const { rememberConnectorEvents } = spine
  const { rememberWhatsAppEvents } = channels

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
      },
    })
    return publicConnectorSettings().whatsapp
  })

  handle('whatsapp:startPairing', async (_e, id: string) => {
    requireSecurityRole('owner', 'pair WhatsApp accounts')
    const source = store.settings.whatsapp.sources.find((s) => s.id === id)
    if (!source) throw new Error('WhatsApp account not found.')
    // Start the bridge; QR codes + inbound events are pushed to the renderer.
    void startWhatsAppBridge(
      source,
      vontraPath('.'),
      (qr) => { store.win?.webContents.send('whatsapp:qr', { sourceId: id, qr }) },
      (event) => { rememberWhatsAppEvents([event]) },
      async (patch) => {
        store.settings = {
          ...store.settings,
          whatsapp: {
            ...store.settings.whatsapp,
            sources: store.settings.whatsapp.sources.map((s) => (s.id === id ? { ...s, ...patch, updatedAt: Date.now() } : s)),
          },
        }
        persistSettings('whatsapp.status')
        store.win?.webContents.send('whatsapp:status', { sourceId: id, ...patch })
      },
      createProtectedWhatsAppAuthState,
    ).catch((e: any) => {
      // The bridge runs fire-and-forget, so a start failure (commonly baileys failing to load
      // in the bundled main) must be PUSHED to the renderer — persisting it only to settings
      // leaves the live draft showing nothing, which is the "Start pairing does nothing" bug.
      const patch = whatsappPairingErrorPatch(e)
      store.settings = {
        ...store.settings,
        whatsapp: { ...store.settings.whatsapp, sources: store.settings.whatsapp.sources.map((s) => (s.id === id ? { ...s, ...patch, updatedAt: Date.now() } : s)) },
      }
      void saveSettings(store.settings)
      store.win?.webContents.send('whatsapp:status', { sourceId: id, ...patch })
      void auditEvent('whatsapp.pairing_failed', { sourceId: id, error: auditPreview(patch.lastError ?? '') })
    })
    return { started: true }
  })

  handle('whatsapp:disconnect', async (_e, id: string) => {
    requireSecurityRole('owner', 'disconnect WhatsApp accounts')
    stopWhatsAppBridge(id)
    return { ok: true }
  })

  handle('whatsapp:send', async (_e, { sourceId, to, text }: { sourceId: string; to: string; text: string }) => {
    // Outbound WhatsApp sends are approval-gated: only the agent's approved send
    // path should call this. Direct renderer sends are denied for safety.
    requireSecurityRole('owner', 'send WhatsApp messages')
    const result = await sendWhatsAppMessage(sourceId, to, text)
    rememberConnectorEvents([whatsappEventFromMessage({ sourceId, jid: to, fromMe: true, text, sender: 'You', ts: Date.now() })])
    return result
  })
}
