import { blueBubblesIdentity } from '../../../agent/bluebubbles-connector'
import { imessageAccessState, imessageEventDedupKey, readIMessageMessage, scanIMessageMessages, sendIMessage, syncIMessageConnector } from '../../../agent/imessage-connector'
import { enrichContacts } from '../connector-enrich'
import { saveSettings } from '../../../store'
import { persistContacts } from '../persistence'
import { auditEvent, requireSecurityRole } from '../audit'
import { auditPreview } from '../../../agent/security'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import type { ConnectorIpcDeps } from '../connector-ipc'

export function registerIMessageIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('imessage:access', async () => {
    requireSecurityRole('operator', 'check iMessage access')
    return imessageAccessState(store.settings.imessage)
  })

  handle('imessage:sync', async () => {
    requireSecurityRole('operator', 'sync iMessage')
    return syncQueue.run('imessage', async () => {
      const selected = { ...store.settings.imessage }
      if (!selected.enabled) throw new Error('Enable iMessage before syncing.')
      const identity = blueBubblesIdentity(selected)
      try {

        const known = new Set(store.connectorEvents.filter((e) => e.connectorId === 'imessage').map(imessageEventDedupKey))

        const result = await syncIMessageConnector(selected, known, selected.cursor)
        if (blueBubblesIdentity(store.settings.imessage) !== identity || !store.settings.imessage.enabled) throw new Error('iMessage connection changed during sync; sync again.')
        store.settings = { ...store.settings, imessage: { ...store.settings.imessage, lastSyncAt: result.syncedAt, lastError: null, bridgeCursorIdentity: identity, cursor: result.cursor ?? store.settings.imessage.cursor } }
        await saveSettings(store.settings)

        const observations = result.events
          .filter((e) => e.direction === 'inbound')
          .map((e) => ({ platform: 'phone' as const, handle: String(e.payload?.handle ?? ''), name: e.actor, source: 'imessage' }))
        const enriched = enrichContacts(store.contacts, result.events, observations)
        store.contacts = enriched.directory
        persistContacts()
        rememberConnectorEvents(enriched.events)
        return { ...result, events: enriched.events }
      } catch (e: any) {
        const error = String(e?.message ?? e)
        store.settings = { ...store.settings, imessage: { ...store.settings.imessage, lastError: error } }
        await saveSettings(store.settings)
        return { events: [], messages: 0, syncedAt: Date.now(), error }
      }
    })
  })

  handle('imessage:list', async () => {
    requireSecurityRole('operator', 'list iMessage messages')
    return scanIMessageMessages(store.settings.imessage)
  })

  handle('imessage:read', async (_e, id: string) => {
    requireSecurityRole('operator', 'read iMessage messages')
    return readIMessageMessage(id, store.settings.imessage)
  })

  handle('imessage:send', async (_e, { to, text }: { to: string; text: string }) => {
    // Outbound iMessage sends are approval-gated (owner) — the agent's approved send path calls this.
    requireSecurityRole('owner', 'send iMessages')
    const selected = { ...store.settings.imessage }
    const result = await sendIMessage(to, text, selected, () => {
      if (!store.settings.imessage.enabled || blueBubblesIdentity(store.settings.imessage) !== blueBubblesIdentity(selected)) throw new Error('The iMessage connection changed before dispatch.')
    })
    await auditEvent('imessage.send', { to: auditPreview(to), chars: String(text ?? '').length })
    return result
  })
}
