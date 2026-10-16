import { buildMattermostSync, extractMattermostMediaUrls, fetchMattermostMessages, listMattermostChannels, makeMattermostSource, postMattermostMessage, verifyMattermostConnection } from '../../../agent/mattermost-connector'
import { attachAuthedMediaPaths, enrichContacts } from '../connector-enrich'
import { recordConnectorInteraction } from '../../../agent/connector-spine'
import { isSecretMask, assertConnectorBaseUrl } from '../../../agent/security'
import { stampSecretRotations } from '../../../agent/secret-rotation'
import { saveSettings } from '../../../store'
import { persistContacts } from '../persistence'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { publicConnectorSettings } from '../connector-event-spine'
import type { ConnectorIpcDeps } from '../connector-ipc'
import type { MattermostSource } from '../../../agent/types'

export function registerMattermostIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('mattermost:addSource', async (_e, label: string) => {
    requireSecurityRole('owner', 'add Mattermost sources')
    const source = makeMattermostSource(label)
    store.settings = { ...store.settings, mattermost: { ...store.settings.mattermost, sources: [...store.settings.mattermost.sources, source] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().mattermost
  })
  handle('mattermost:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<MattermostSource> }) => {
    requireSecurityRole('owner', 'update Mattermost sources')
    store.settings = {
      ...store.settings,
      mattermost: {
        ...store.settings.mattermost,
        sources: store.settings.mattermost.sources.map((s) => {
          if (s.id !== id) return s
          const safePatch = { ...patch }
          if (safePatch.token && isSecretMask(safePatch.token)) safePatch.token = s.token
          const next = { ...s, ...safePatch, id: s.id, updatedAt: Date.now() }
          // Save-time mirror of the live-fetch guard (resolveConnectorRequest): the merged
          // URL must pass the same public-host rules, honoring the private-network opt-in.
          assertConnectorBaseUrl(next.baseUrl, 'Mattermost', { allowPrivateNetwork: next.allowPrivateNetwork })
          return stampSecretRotations(next, s, { token: [next.token, s.token] })
        }),
      },
    }
    await saveSettings(store.settings)
    return publicConnectorSettings().mattermost
  })
  handle('mattermost:removeSource', async (_e, id: string) => {
    requireSecurityRole('owner', 'remove Mattermost sources')
    store.settings = { ...store.settings, mattermost: { ...store.settings.mattermost, sources: store.settings.mattermost.sources.filter((s) => s.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().mattermost
  })
  handle('mattermost:verify', async (_e, { id }: { id: string }) => {
    requireSecurityRole('owner', 'verify a Mattermost token')
    const source = store.settings.mattermost.sources.find((s) => s.id === id)
    if (!source?.token || !source.baseUrl) throw new Error('Mattermost source not found or missing server URL/token.')
    return verifyMattermostConnection(source.baseUrl, source.token, source.allowPrivateNetwork)
  })
  handle('mattermost:listChannels', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'list Mattermost channels')
    const source = store.settings.mattermost.sources.find((s) => s.id === id)
    if (!source?.token || !source.baseUrl) throw new Error('Mattermost source not found or missing server URL/token.')
    return listMattermostChannels(source.baseUrl, source.token, source.allowPrivateNetwork)
  })
  handle('mattermost:sync', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'sync Mattermost')
    return syncQueue.run(`mattermost:${id}`, async () => {
      const source = store.settings.mattermost.sources.find((s) => s.id === id)
      if (!source) throw new Error('Mattermost source not found.')
      try {
      const messages = await fetchMattermostMessages(source)
      // INCREMENTAL: buildMattermostSync filters by the source cursor + builds events; stamp the
      // source post id into payload.messageId so rememberConnectorEvents' insert-time dedup
      // recognises an already-ingested post on a later poll (mirrors dedupeDiscordEvents).
      // mattermostEventFromMessage already stamps payload.messageId = the source post id (so
      // insert-time dedup recognises an already-ingested post). Do NOT re-derive it here by
      // matching threadKey+text — two distinct posts with identical text in a channel would
      // resolve to the SAME (first) post and the second would be deduped away.
      const { events, cursor } = buildMattermostSync(source, messages)
      // CONTACT NAMES: learn senderId -> senderName so a later "reply to <name>" resolves an id.
      // Mattermost isn't a first-class ContactPlatform, so observations ride the 'other' bucket.
      const observations = messages.map((m) => ({ platform: 'other' as const, handle: m.senderId, name: m.senderName, source: 'mattermost' }))
      const enriched = enrichContacts(store.contacts, events, observations)
      store.contacts = enriched.directory
      persistContacts()
      // MEDIA CACHE: Mattermost files download from /api/v4/files/{id} WITH the bot token, so use
      // the auth-aware path with an Authorization: Bearer header per event's source.
      const byMessageId = new Map(messages.map((m) => [m.id, m]))
      const authHeader = { Authorization: `Bearer ${source.token.trim()}` }
      await attachAuthedMediaPaths(enriched.events, (event) => {
        const mid = typeof event.payload?.messageId === 'string' ? event.payload.messageId : null
        const message = mid ? byMessageId.get(mid) : undefined
        return message ? extractMattermostMediaUrls(message.raw, source.baseUrl).map((url) => ({
          url,
          headers: authHeader,
          endpoint: { baseUrl: source.baseUrl, label: 'Mattermost media', options: { allowPrivateNetwork: source.allowPrivateNetwork } },
        })) : []
      })
      rememberConnectorEvents(enriched.events)
        const syncedAt = Date.now()
        store.settings = { ...store.settings, mattermost: { ...store.settings.mattermost, sources: store.settings.mattermost.sources.map((s) => s.id === id ? { ...s, lastSyncAt: syncedAt, lastError: null, cursor: cursor ?? s.cursor } : s), lastSyncAt: syncedAt } }
        await saveSettings(store.settings)
        return { messages: messages.length, events: events.length }
      } catch (e: any) {
        store.settings = { ...store.settings, mattermost: { ...store.settings.mattermost, sources: store.settings.mattermost.sources.map((s) => s.id === id ? { ...s, lastError: String(e?.message ?? e) } : s) } }
        await saveSettings(store.settings)
        throw e
      }
    })
  })
  handle('mattermost:send', async (_e, { sourceId, channelId, text, replyTo }: { sourceId: string; channelId: string; text: string; replyTo?: string }) => {
    requireSecurityRole('owner', 'send Mattermost messages')
    const source = store.settings.mattermost.sources.find((s) => s.id === sourceId)
    if (!source) throw new Error('Mattermost source not found.')
    const result = await postMattermostMessage(source, channelId, text, replyTo)
    rememberConnectorEvents([recordConnectorInteraction({
      connectorId: 'mattermost',
      pluginName: 'Mattermost',
      action: 'message_sent',
      direction: 'outbound',
      target: channelId,
      output: { title: 'Sent a Mattermost message', summary: `${text.slice(0, 120)} -> ${channelId}`, threadKey: `mattermost:${source.id}:${channelId}` },
    })])
    return result
  })
}
