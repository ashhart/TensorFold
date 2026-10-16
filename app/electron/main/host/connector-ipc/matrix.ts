import { listMatrixChannels, makeMatrixSource, postMatrixMessage, syncMatrixSource, verifyMatrixConnection } from '../../../agent/matrix-connector'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'
import { recordConnectorInteraction } from '../../../agent/connector-spine'
import { isSecretMask, assertConnectorBaseUrl } from '../../../agent/security'
import { stampSecretRotations } from '../../../agent/secret-rotation'
import { saveSettings } from '../../../store'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { publicConnectorSettings } from '../connector-event-spine'
import type { ConnectorIpcDeps } from '../connector-ipc'
import type { MatrixSource } from '../../../agent/types'

export function registerMatrixIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('matrix:addSource', async (_e, label: string) => {
    requireSecurityRole('owner', 'add Matrix sources')
    const source = makeMatrixSource(label)
    store.settings = { ...store.settings, matrix: { ...store.settings.matrix, sources: [...store.settings.matrix.sources, source] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().matrix
  })
  handle('matrix:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<MatrixSource> }) => {
    requireSecurityRole('owner', 'update Matrix sources')
    store.settings = {
      ...store.settings,
      matrix: {
        ...store.settings.matrix,
        sources: store.settings.matrix.sources.map((s) => {
          if (s.id !== id) return s
          const safePatch = { ...patch }
          if (safePatch.token && isSecretMask(safePatch.token)) safePatch.token = s.token
          const next = { ...s, ...safePatch, id: s.id, updatedAt: Date.now() }
          // Save-time mirror of the live-fetch guard (resolveConnectorRequest): the merged
          // URL must pass the same public-host rules, honoring the private-network opt-in.
          assertConnectorBaseUrl(next.baseUrl, 'Matrix', { allowPrivateNetwork: next.allowPrivateNetwork })
          return stampSecretRotations(next, s, { token: [next.token, s.token] })
        }),
      },
    }
    await saveSettings(store.settings)
    return publicConnectorSettings().matrix
  })
  handle('matrix:removeSource', async (_e, id: string) => {
    requireSecurityRole('owner', 'remove Matrix sources')
    store.settings = { ...store.settings, matrix: { ...store.settings.matrix, sources: store.settings.matrix.sources.filter((s) => s.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().matrix
  })
  handle('matrix:verify', async (_e, { id }: { id: string }) => {
    requireSecurityRole('owner', 'verify a Matrix token')
    const source = store.settings.matrix.sources.find((s) => s.id === id)
    if (!source?.token || !source.baseUrl) throw new Error('Matrix source not found or missing homeserver URL/token.')
    return verifyMatrixConnection(source.baseUrl, source.token, source.allowPrivateNetwork)
  })
  handle('matrix:listChannels', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'list Matrix rooms')
    const source = store.settings.matrix.sources.find((s) => s.id === id)
    if (!source?.token || !source.baseUrl) throw new Error('Matrix source not found or missing homeserver URL/token.')
    return listMatrixChannels(source)
  })
  handle('matrix:sync', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'sync Matrix')
    return syncQueue.run(`matrix:${id}`, async () => {
      const source = store.settings.matrix.sources.find((s) => s.id === id)
      if (!source) throw new Error('Matrix source not found.')
      try {
      // Self-contained: syncMatrixSource fetches + cursor-filters + returns events and an
      // updated source (with the advanced cursor). The incremental cursor on the source is the
      // dedup mechanism here (matrixEventFromMessage stores text/room/thread, not the event id).
      const result = await syncMatrixSource(source)
      // MEDIA: Matrix media events carry mxc:// URIs that convert to an authenticated homeserver
      // download URL — but syncMatrixSource returns only the built events (text/room/thread), not
      // the raw timeline events, so there is no raw payload to extract attachment URIs from here.
      // Media ingest for Matrix is deferred (the inbound text still lands). Persist like slack:sync.
      rememberConnectorEvents(result.events)
        const updated = result.sources[0] ?? source
        store.settings = { ...store.settings, matrix: { ...store.settings.matrix, sources: mergeConnectorSyncSources(store.settings.matrix.sources, [{ ...updated, lastSyncAt: result.syncedAt, lastError: null }]), lastSyncAt: result.syncedAt } }
        await saveSettings(store.settings)
        return { messages: result.messages, events: result.events.length }
      } catch (e: any) {
        store.settings = { ...store.settings, matrix: { ...store.settings.matrix, sources: store.settings.matrix.sources.map((s) => s.id === id ? { ...s, lastError: String(e?.message ?? e) } : s) } }
        await saveSettings(store.settings)
        throw e
      }
    })
  })
  handle('matrix:send', async (_e, { sourceId, roomId, text, replyTo, threadRootId }: { sourceId: string; roomId: string; text: string; replyTo?: string; threadRootId?: string }) => {
    requireSecurityRole('owner', 'send Matrix messages')
    const source = store.settings.matrix.sources.find((s) => s.id === sourceId)
    if (!source) throw new Error('Matrix source not found.')
    const result = await postMatrixMessage(source, roomId, text, { replyTo: replyTo ?? null, threadRootId: threadRootId ?? null })
    rememberConnectorEvents([recordConnectorInteraction({
      connectorId: 'matrix',
      pluginName: 'Matrix',
      action: 'message_sent',
      direction: 'outbound',
      target: roomId,
      output: { title: 'Sent a Matrix message', summary: `${text.slice(0, 120)} -> ${roomId}`, threadKey: `matrix:${source.id}:${threadRootId ?? roomId}` },
    })])
    return result
  })
}
