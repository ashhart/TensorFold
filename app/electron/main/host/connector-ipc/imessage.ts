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
    return imessageAccessState()
  })

  handle('imessage:sync', async () => {
    requireSecurityRole('operator', 'sync iMessage')
    return syncQueue.run('imessage', async () => {
      try {
        // Per-message dedup keys (payload.messageId), NOT threadKey — threadKey is now
        // conversation-scoped, so keying `known` off it would drop every follow-up message.
        const known = new Set(store.connectorEvents.filter((e) => e.connectorId === 'imessage').map(imessageEventDedupKey))
        // INCREMENTAL: the persisted cursor skips messages not newer than last sync before
        // the per-message dedup, so chat.db isn't re-walked into events each poll.
        const result = await syncIMessageConnector(store.settings.imessage, known, store.settings.imessage.cursor)
        store.settings = { ...store.settings, imessage: { ...store.settings.imessage, lastSyncAt: result.syncedAt, lastError: null, cursor: result.cursor ?? store.settings.imessage.cursor } }
        await saveSettings(store.settings)
        // CONTACT NAMES: iMessage shows raw phone numbers; learn handle -> contact name from
        // the address-book-resolved chatName the connector already attached, and resolve raw
        // numbers in the feed to that name. Only learns when the message is from a contact.
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
    return readIMessageMessage(id)
  })

  handle('imessage:send', async (_e, { to, text }: { to: string; text: string }) => {
    // Outbound iMessage sends are approval-gated (owner) — the agent's approved send path calls this.
    requireSecurityRole('owner', 'send iMessages')
    const result = await sendIMessage(to, text)
    await auditEvent('imessage.send', { to: auditPreview(to), chars: String(text ?? '').length })
    return result
  })
}
