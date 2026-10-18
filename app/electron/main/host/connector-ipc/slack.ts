import { listSlackChannels, postSlackMessage, verifySlackToken } from '../../../agent/slack-live'
import { makeSlackSource, readSlackMessage, scanSlackMessages, summarizeSlack, syncSlackConnector } from '../../../agent/slack-connector'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'
import { recordConnectorInteraction } from '../../../agent/connector-spine'
import { isSecretMask } from '../../../agent/security'
import { stampSecretRotations } from '../../../agent/secret-rotation'
import { saveSettings } from '../../../store'
import { persistContacts } from '../persistence'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { attachAuthedMediaPaths, enrichContacts, slackContactObservations } from '../connector-enrich'
import { publicConnectorSettings } from '../connector-event-spine'
import type { ConnectorIpcDeps } from '../connector-ipc'
import type { ConnectorEvent, SlackListOptions, SlackSourceConfig } from '../../../agent/types'

export function registerSlackIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { rememberConnectorEvents } = spine

  handle('slack:addSource', async (_e, dir: string) => {
    requireSecurityRole('operator', 'add Slack sources')
    const source = makeSlackSource(dir)
    store.settings = { ...store.settings, slack: { ...store.settings.slack, sources: [source, ...store.settings.slack.sources] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().slack
  })

  handle('slack:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<SlackSourceConfig> }) => {
    requireSecurityRole('operator', 'update Slack sources')
    store.settings = {
      ...store.settings,
      slack: {
        ...store.settings.slack,
        sources: store.settings.slack.sources.map((source) => {
          if (source.id !== id) return source
          const safePatch = { ...patch }
          if (safePatch.token && isSecretMask(safePatch.token)) safePatch.token = source.token
          const next = { ...source, ...safePatch, id: source.id, updatedAt: Date.now() }
          return stampSecretRotations(next, source, { token: [next.token, source.token] })
        }),
      },
    }
    await saveSettings(store.settings)
    return publicConnectorSettings().slack
  })

  handle('slack:removeSource', async (_e, id: string) => {
    requireSecurityRole('operator', 'remove Slack sources')
    store.settings = { ...store.settings, slack: { ...store.settings.slack, sources: store.settings.slack.sources.filter((source) => source.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().slack
  })

  handle('slack:sync', async (_e, input?: { id?: string }) => {
    requireSecurityRole('operator', 'sync Slack')
    if (input !== undefined && (!input || typeof input !== 'object' || Array.isArray(input))) throw new Error('Slack sync input must identify a source or be omitted.')
    return syncQueue.run('slack', async () => {
      const id = input?.id
      if (id !== undefined && (typeof id !== 'string' || !store.settings.slack.sources.some((source) => source.id === id))) throw new Error('Slack source not found.')
      const settings = id ? { ...store.settings.slack, sources: store.settings.slack.sources.filter((source) => source.id === id) } : store.settings.slack
      const result = await syncSlackConnector(settings, store.connectorEvents)
      store.settings = { ...store.settings, slack: { ...store.settings.slack, sources: mergeConnectorSyncSources(store.settings.slack.sources, result.sources), lastSyncAt: result.syncedAt } }
      await saveSettings(store.settings)
      // CONTACT NAMES: learn Slack user id -> display-name from this batch so "reply to <name>"
      // can resolve a Slack id later. The actor is the channel name, so enrichContacts leaves
      // events untouched; we only persist the grown directory.
      const enriched = enrichContacts(store.contacts, result.events, slackContactObservations(result.events))
      store.contacts = enriched.directory
      persistContacts()
      // MEDIA CACHE: a Slack message's file attachments expose url_private, which needs the
      // workspace bot token as `Authorization: Bearer <token>`. Download each inbound event's
      // url_private refs (carried on payload.slackFiles by the connector) with that token via the
      // auth-aware cache path, and stamp the local paths onto mediaPaths. The token is looked up
      // per event from its source. Best-effort; a bad/expired file never blocks ingest.
      await attachAuthedMediaPaths(enriched.events, (event) => slackMediaItemsFor(event))
      rememberConnectorEvents(enriched.events)
      return { ...result, sources: publicConnectorSettings().slack.sources.filter((source) => !id || source.id === id), events: enriched.events }
    })
  })

  handle('slack:list', async (_e, opts: SlackListOptions = {}) => {
    requireSecurityRole('operator', 'list Slack messages')
    return (await scanSlackMessages(store.settings.slack, opts)).messages
  })

  handle('slack:read', async (_e, id: string) => {
    requireSecurityRole('operator', 'read Slack messages')
    return readSlackMessage(store.settings.slack, id)
  })

  handle('slack:summary', async (_e, opts: SlackListOptions = {}) => {
    requireSecurityRole('operator', 'summarize Slack')
    return summarizeSlack(store.settings.slack, opts)
  })

  // Live Slack (api_token): verify the token works, list channels, and send/reply.
  handle('slack:verify', async (_e, { id }: { id: string }) => {
    requireSecurityRole('owner', 'verify a Slack token')
    const source = store.settings.slack.sources.find((s) => s.id === id && s.kind === 'api_token')
    if (!source?.token) throw new Error('Slack source not found or has no token.')
    const result = await verifySlackToken(source.token)
    const current = store.settings.slack.sources.find((candidate) => candidate.id === id && candidate.kind === 'api_token')
    if (!current || current.token !== source.token) throw new Error('Slack credentials changed during verification; verify the current connection again.')
    return result
  })
  handle('slack:listChannels', async (_e, { id }: { id: string }) => {
    requireSecurityRole('operator', 'list Slack channels')
    const source = store.settings.slack.sources.find((s) => s.id === id && s.kind === 'api_token')
    if (!source?.token) throw new Error('Slack source not found or has no token.')
    return listSlackChannels(source.token)
  })
  handle('slack:send', async (_e, { sourceId, channel, text, threadTs }: { sourceId: string; channel: string; text: string; threadTs?: string }) => {
    // Approval-gated outbound: only the agent's approved reply path calls this.
    requireSecurityRole('owner', 'send Slack messages')
    const source = store.settings.slack.sources.find((s) => s.id === sourceId && s.kind === 'api_token')
    if (!source) throw new Error('Live Slack source not found.')
    const result = await postSlackMessage(source, channel, text, threadTs)
    rememberConnectorEvents([recordConnectorInteraction({
      connectorId: 'slack',
      pluginName: 'Slack',
      action: 'message_sent',
      direction: 'outbound',
      target: channel,
      output: { title: 'Sent a Slack message', summary: `${text.slice(0, 120)} -> ${channel}`, threadKey: threadTs ? `slack:${source.id}:${channel}:${threadTs}` : `slack:${source.id}:${channel}` },
    })])
    return result
  })
}

// Map a synced Slack event to the auth'd download items for its file attachments. Reads the
// url_private refs the connector stamped on payload.slackFiles and pairs each with the bot
// token of the event's Slack source as an `Authorization: Bearer` header. Returns [] when the
// event has no files or its source has no api_token (export sources can't download url_private).
function slackMediaItemsFor(event: ConnectorEvent): { url: string; headers?: Record<string, string> }[] {
  const refs = event.payload?.slackFiles
  if (!Array.isArray(refs) || !refs.length) return []
  const sourceId = event.sources?.[0]?.sourceId ?? null
  const source = sourceId ? store.settings.slack.sources.find((s) => s.id === sourceId && s.kind === 'api_token') : null
  const token = source?.token?.trim()
  if (!token) return []
  const headers = { Authorization: `Bearer ${token}` }
  const items: { url: string; headers?: Record<string, string> }[] = []
  for (const ref of refs as { urlPrivate?: unknown }[]) {
    const url = typeof ref?.urlPrivate === 'string' ? ref.urlPrivate : null
    if (url) items.push({ url, headers })
  }
  return items
}
