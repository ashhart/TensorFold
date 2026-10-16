import { listGoogleChatChannels, makeGoogleChatSource, postGoogleChatMessage, syncGoogleChatSource, verifyGoogleChatConnection } from '../../../agent/googlechat-connector'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'
import { recordConnectorInteraction } from '../../../agent/connector-spine'
import { isSecretMask } from '../../../agent/security'
import { stampSecretRotations } from '../../../agent/secret-rotation'
import { saveSettings } from '../../../store'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { publicConnectorSettings } from '../connector-event-spine'
import type { ConnectorIpcDeps } from '../connector-ipc'
import type { GoogleChatSource } from '../../../agent/types'

export function registerGoogleChatIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('googlechat:addSource', async (_e, label: string) => {
    requireSecurityRole('owner', 'add Google Chat sources')
    const source = makeGoogleChatSource(label)
    store.settings = { ...store.settings, googlechat: { ...store.settings.googlechat, sources: [...store.settings.googlechat.sources, source] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().googlechat
  })
  handle('googlechat:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<GoogleChatSource> }) => {
    requireSecurityRole('owner', 'update Google Chat sources')
    store.settings = {
      ...store.settings,
      googlechat: {
        ...store.settings.googlechat,
        sources: store.settings.googlechat.sources.map((s) => {
          if (s.id !== id) return s
          const safePatch = { ...patch }
          if (safePatch.webhookUrl && isSecretMask(safePatch.webhookUrl)) safePatch.webhookUrl = s.webhookUrl
          if (safePatch.accessToken && isSecretMask(safePatch.accessToken)) safePatch.accessToken = s.accessToken
          const next = { ...s, ...safePatch, id: s.id, updatedAt: Date.now() }
          return stampSecretRotations(next, s, { webhookUrl: [next.webhookUrl, s.webhookUrl], accessToken: [next.accessToken, s.accessToken] })
        }),
      },
    }
    await saveSettings(store.settings)
    return publicConnectorSettings().googlechat
  })
  handle('googlechat:removeSource', async (_e, id: string) => {
    requireSecurityRole('owner', 'remove Google Chat sources')
    store.settings = { ...store.settings, googlechat: { ...store.settings.googlechat, sources: store.settings.googlechat.sources.filter((s) => s.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().googlechat
  })
  handle('googlechat:verify', async (_e, { id }: { id: string }) => {
    requireSecurityRole('owner', 'verify a Google Chat source')
    const source = store.settings.googlechat.sources.find((s) => s.id === id)
    if (!source) throw new Error('Google Chat source not found.')
    return verifyGoogleChatConnection(source)
  })
  handle('googlechat:listChannels', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'list Google Chat spaces')
    const source = store.settings.googlechat.sources.find((s) => s.id === id)
    if (!source) throw new Error('Google Chat source not found.')
    return listGoogleChatChannels(source)
  })
  handle('googlechat:sync', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'sync Google Chat')
    return syncQueue.run(`googlechat:${id}`, async () => {
      const source = store.settings.googlechat.sources.find((s) => s.id === id)
      if (!source) throw new Error('Google Chat source not found.')
      try {
      // Self-contained: syncGoogleChatSource fetches + cursor-filters + returns events and an
      // updated source. A webhook-only source returns [] with an explanatory lastError (it can
      // only send). MEDIA: Google Chat attachment downloadUris are public https — use the
      // unauth path. Stamp the message resource name into payload.messageId for insert dedup.
      const result = await syncGoogleChatSource(source)
        const updated = result.sources[0] ?? source
        rememberConnectorEvents(result.events)
        store.settings = { ...store.settings, googlechat: { ...store.settings.googlechat, sources: mergeConnectorSyncSources(store.settings.googlechat.sources, [{ ...updated, lastSyncAt: result.syncedAt }]), lastSyncAt: result.syncedAt } }
        await saveSettings(store.settings)
        return { messages: result.messages, events: result.events.length }
      } catch (e: any) {
        store.settings = { ...store.settings, googlechat: { ...store.settings.googlechat, sources: store.settings.googlechat.sources.map((s) => s.id === id ? { ...s, lastError: String(e?.message ?? e) } : s) } }
        await saveSettings(store.settings)
        throw e
      }
    })
  })
  handle('googlechat:send', async (_e, { sourceId, space, text, threadKey, threadName }: { sourceId: string; space: string; text: string; threadKey?: string; threadName?: string }) => {
    requireSecurityRole('owner', 'send Google Chat messages')
    const source = store.settings.googlechat.sources.find((s) => s.id === sourceId)
    if (!source) throw new Error('Google Chat source not found.')
    const result = await postGoogleChatMessage(source, space, text, threadName ? { threadName } : threadKey ? { threadKey } : undefined)
    rememberConnectorEvents([recordConnectorInteraction({
      connectorId: 'googlechat',
      pluginName: 'Google Chat',
      action: 'message_sent',
      direction: 'outbound',
      target: space,
      output: { title: 'Sent a Google Chat message', summary: `${text.slice(0, 120)} -> ${space}`, threadKey: `googlechat:${source.id}:${result.threadName ?? space}` },
    })])
    return result
  })
}
