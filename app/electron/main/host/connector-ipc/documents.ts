import { documentLibrary } from '../document-library-runtime'
import { makeDocumentSource, readDocumentItem, scanDocumentItems, summarizeDocuments, syncDocumentConnector } from '../../../agent/document-connector'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'
import { saveSettings, vontraHomeDir } from '../../../store'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { publicConnectorSettings } from '../connector-event-spine'
import type { DocumentListOptions, DocumentSourceConfig } from '../../../agent/types'
import type { ConnectorIpcDeps } from './types'

export function registerDocumentIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('documents:addSource', async (_e, target: string) => {
    requireSecurityRole('operator', 'add document sources')
    const source = makeDocumentSource(target)
    store.settings = { ...store.settings, documents: { ...store.settings.documents, sources: [source, ...store.settings.documents.sources] } }
    await saveSettings(store.settings)
    await documentLibrary().reconcile()
    return publicConnectorSettings().documents
  })

  handle('documents:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<DocumentSourceConfig> }) => {
    requireSecurityRole('operator', 'update document sources')
    store.settings = {
      ...store.settings,
      documents: {
        ...store.settings.documents,
        sources: store.settings.documents.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source),
      },
    }
    await saveSettings(store.settings)
    await documentLibrary().reconcile()
    return publicConnectorSettings().documents
  })

  handle('documents:removeSource', async (_e, id: string) => {
    requireSecurityRole('operator', 'remove document sources')
    store.settings = { ...store.settings, documents: { ...store.settings.documents, sources: store.settings.documents.sources.filter((source) => source.id !== id) } }
    await saveSettings(store.settings)
    await documentLibrary().reconcile()
    return publicConnectorSettings().documents
  })

  handle('documents:sync', async () => {
    requireSecurityRole('operator', 'sync documents')
    const profile = vontraHomeDir()
    const assertCurrent = () => { requireSecurityRole('operator', 'sync documents'); if (vontraHomeDir() !== profile) throw new Error('Document profile changed while processing; try again in the selected profile.') }
    return syncQueue.run('documents', async () => {
      assertCurrent()
      await documentLibrary().sync()
      assertCurrent()
      const result = await syncDocumentConnector(store.settings.documents, store.connectorEvents, await documentLibrary().connectorItems())
      assertCurrent()
      store.settings = { ...store.settings, documents: { ...store.settings.documents, sources: mergeConnectorSyncSources(store.settings.documents.sources, result.sources), lastSyncAt: result.syncedAt } }
      await saveSettings(store.settings)
      assertCurrent()
      rememberConnectorEvents(result.events)
      return { ...result, sources: publicConnectorSettings().documents.sources }
    })
  })

  handle('documents:list', async (_e, opts: DocumentListOptions = {}) => {
    requireSecurityRole('operator', 'list documents')
    return (await scanDocumentItems(store.settings.documents, opts)).documents
  })

  handle('documents:read', async (_e, id: string) => {
    requireSecurityRole('operator', 'read documents')
    return readDocumentItem(store.settings.documents, id)
  })

  handle('documents:summary', async (_e, opts: DocumentListOptions = {}) => {
    requireSecurityRole('operator', 'summarize documents')
    return summarizeDocuments(store.settings.documents, opts)
  })
}
