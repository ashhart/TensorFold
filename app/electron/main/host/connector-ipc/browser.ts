import { makeBrowserCurrentTabSource, makeBrowserHistorySource, readBrowserItem, scanBrowserItems, summarizeBrowser, syncBrowserConnector } from '../../../agent/browser-connector'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'
import { saveSettings } from '../../../store'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { publicConnectorSettings } from '../connector-event-spine'
import type { BrowserListOptions, BrowserSnapshotInput, BrowserSourceConfig } from '../../../agent/types'
import type { ConnectorIpcDeps } from './types'

export function registerBrowserIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('browser:addCurrentTabSource', async (_e, input: BrowserSnapshotInput) => {
    requireSecurityRole('operator', 'add browser current-tab source')
    const source = makeBrowserCurrentTabSource(input)
    store.settings = { ...store.settings, browser: { ...store.settings.browser, sources: [source, ...store.settings.browser.sources] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().browser
  })

  handle('browser:addHistorySource', async (_e, target: string) => {
    requireSecurityRole('operator', 'add browser history source')
    const source = makeBrowserHistorySource(target)
    store.settings = { ...store.settings, browser: { ...store.settings.browser, sources: [source, ...store.settings.browser.sources] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().browser
  })

  handle('browser:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<BrowserSourceConfig> }) => {
    requireSecurityRole('operator', 'update browser sources')
    store.settings = {
      ...store.settings,
      browser: {
        ...store.settings.browser,
        sources: store.settings.browser.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, kind: source.kind, updatedAt: Date.now() } : source),
      },
    }
    await saveSettings(store.settings)
    return publicConnectorSettings().browser
  })

  handle('browser:removeSource', async (_e, id: string) => {
    requireSecurityRole('operator', 'remove browser sources')
    store.settings = { ...store.settings, browser: { ...store.settings.browser, sources: store.settings.browser.sources.filter((source) => source.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().browser
  })

  handle('browser:sync', async () => {
    requireSecurityRole('operator', 'sync browser sources')
    return syncQueue.run('browser', async () => {
      const result = await syncBrowserConnector(store.settings.browser, store.connectorEvents)
      store.settings = { ...store.settings, browser: { ...store.settings.browser, sources: mergeConnectorSyncSources(store.settings.browser.sources, result.sources), lastSyncAt: result.syncedAt } }
      await saveSettings(store.settings)
      rememberConnectorEvents(result.events)
      return { ...result, sources: publicConnectorSettings().browser.sources }
    })
  })

  handle('browser:list', async (_e, opts: BrowserListOptions = {}) => {
    requireSecurityRole('operator', 'list browser sources')
    return (await scanBrowserItems(store.settings.browser, opts)).items
  })

  handle('browser:read', async (_e, id: string) => {
    requireSecurityRole('operator', 'read browser sources')
    return readBrowserItem(store.settings.browser, id)
  })

  handle('browser:summary', async (_e, opts: BrowserListOptions = {}) => {
    requireSecurityRole('operator', 'summarize browser sources')
    return summarizeBrowser(store.settings.browser, opts)
  })
}
