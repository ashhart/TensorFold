import { extractDiscordMediaUrls, fetchDiscordMessages, listDiscordChannels, makeDiscordSource, postDiscordMessage, verifyDiscordToken } from '../../../agent/discord-connector'
import { attachMediaPaths, enrichContacts } from '../connector-enrich'
import { recordConnectorInteraction } from '../../../agent/connector-spine'
import { saveSettings } from '../../../store'
import { persistContacts } from '../persistence'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { dedupeDiscordEvents } from '../connector-runtime'
import { publicConnectorSettings } from '../connector-event-spine'
import type { ConnectorIpcDeps } from '../connector-ipc'

export function registerDiscordIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('discord:addSource', async (_e, label: string) => {
    requireSecurityRole('owner', 'add Discord sources')
    const source = makeDiscordSource(label)
    store.settings = { ...store.settings, discord: { ...store.settings.discord, sources: [...store.settings.discord.sources, source] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().discord
  })
  handle('discord:removeSource', async (_e, id: string) => {
    requireSecurityRole('owner', 'remove Discord sources')
    store.settings = { ...store.settings, discord: { ...store.settings.discord, sources: store.settings.discord.sources.filter((s) => s.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().discord
  })
  handle('discord:verify', async (_e, { id }: { id: string }) => {
    requireSecurityRole('owner', 'verify a Discord token')
    const source = store.settings.discord.sources.find((s) => s.id === id)
    if (!source?.token) throw new Error('Discord source not found or has no token.')
    return verifyDiscordToken(source.token)
  })
  handle('discord:listChannels', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'list Discord channels')
    const source = store.settings.discord.sources.find((s) => s.id === id)
    if (!source?.token) throw new Error('Discord source not found or has no token.')
    return listDiscordChannels(source.token)
  })
  handle('discord:sync', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'sync Discord')
    return syncQueue.run(`discord:${id}`, async () => {
      const source = store.settings.discord.sources.find((s) => s.id === id)
      if (!source) throw new Error('Discord source not found.')
      try {
        const messages = await fetchDiscordMessages(source)
        // Dedup against already-ingested messages by stable Discord message id so
        // repeated polls don't re-create the same ~50 messages with fresh UUIDs.
        // INCREMENTAL: the source cursor skips messages not newer than last sync.
        const { events, cursor } = dedupeDiscordEvents(messages, store.connectorEvents, source.cursor)
        // CONTACT NAMES: learn senderId -> senderName from this batch and resolve raw ids
        // in the event actor/title to the known name. Persist the directory when it grows.
        const observations = messages.map((m) => ({ platform: 'discord' as const, handle: m.senderId, name: m.senderName, source: 'discord' }))
        const enriched = enrichContacts(store.contacts, events, observations)
        store.contacts = enriched.directory
        persistContacts()
        // MEDIA CACHE: download each new event's Discord attachments/image-embeds to local
        // files so the assistant/vision tool can read them by path after the CDN link expires.
        const byMessageId = new Map(messages.map((m) => [m.id, m]))
        await attachMediaPaths(enriched.events, (event) => {
          const mid = typeof event.payload?.messageId === 'string' ? event.payload.messageId : null
          const message = mid ? byMessageId.get(mid) : undefined
          return message ? extractDiscordMediaUrls(message.raw) : []
        })
        rememberConnectorEvents(enriched.events)
        const syncedAt = Date.now()
        store.settings = { ...store.settings, discord: { ...store.settings.discord, sources: store.settings.discord.sources.map((s) => s.id === id ? { ...s, lastSyncAt: syncedAt, lastError: null, cursor: cursor ?? s.cursor } : s), lastSyncAt: syncedAt } }
        await saveSettings(store.settings)
        return { messages: messages.length, events: events.length }
      } catch (e: any) {
        store.settings = { ...store.settings, discord: { ...store.settings.discord, sources: store.settings.discord.sources.map((s) => s.id === id ? { ...s, lastError: String(e?.message ?? e) } : s) } }
        await saveSettings(store.settings)
        throw e
      }
    })
  })
  handle('discord:send', async (_e, { sourceId, channelId, text, replyTo }: { sourceId: string; channelId: string; text: string; replyTo?: string }) => {
    requireSecurityRole('owner', 'send Discord messages')
    const source = store.settings.discord.sources.find((s) => s.id === sourceId)
    if (!source) throw new Error('Discord source not found.')
    const result = await postDiscordMessage(source, channelId, text, replyTo)
    rememberConnectorEvents([recordConnectorInteraction({
      connectorId: 'discord',
      pluginName: 'Discord',
      action: 'message_sent',
      direction: 'outbound',
      target: channelId,
      output: { title: 'Sent a Discord message', summary: `${text.slice(0, 120)} -> ${channelId}`, threadKey: `discord:${source.id}:${channelId}` },
    })])
    return result
  })
}
