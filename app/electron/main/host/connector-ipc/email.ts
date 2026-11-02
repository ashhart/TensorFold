import { connectorSourceBinding } from '../../../agent/connector-source-binding'
import { registerMicrosoftMailIpc } from './microsoft-mail'
import { microsoftMailService } from '../microsoft-mail-runtime'
import { emailSourceRef, emailThreadResolvedEvent, readEmailMessage, readEmailThread, scanEmailMessages, summarizeEmail, syncEmailConnector, validateAndMakeEmailSource } from '../../../agent/email-connector'
import { attachMediaBytesPaths, emailContactObservations, enrichContacts } from '../connector-enrich'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'
import { isSecretMask } from '../../../agent/security'
import { stampSecretRotations } from '../../../agent/secret-rotation'
import { saveSettings } from '../../../store'
import { persistContacts } from '../persistence'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { publicConnectorSettings } from '../connector-event-spine'
import type { EmailDraftInput, EmailListOptions, EmailResolveInput, EmailSourceConfig } from '../../../agent/types'
import type { ConnectorEvent } from '../../../agent/types'
import type { ConnectorIpcDeps } from '../connector-ipc'

export function registerEmailIpc(deps: ConnectorIpcDeps): void {
  registerMicrosoftMailIpc(deps)
  const handle = deps.handle ?? defaultHandle
  const { spine, emailTransport, syncQueue } = deps
  const { rememberConnectorEvents } = spine
  const { recordEmailDraft, recordEmailEvent } = emailTransport

  handle('email:addSource', async (_e, dir: string) => {
    requireSecurityRole('operator', 'add email sources')
    let source
    try {
      source = await validateAndMakeEmailSource(dir)
    } catch (e: any) {
      throw new Error(`Cannot add email folder: ${String(e?.message ?? e)}`)
    }
    store.settings = { ...store.settings, email: { ...store.settings.email, sources: [source, ...store.settings.email.sources] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().email
  })

  handle('email:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<EmailSourceConfig> }) => {
    requireSecurityRole('operator', 'update email sources')
    store.settings = {
      ...store.settings,
      email: {
        ...store.settings.email,
        sources: store.settings.email.sources.map((source) => {
          if (source.id !== id) return source
          const safePatch = source.kind === 'microsoft_graph' ? {
            ...(typeof patch.label === 'string' && patch.label.trim() ? { label: patch.label.trim().slice(0, 200) } : {}),
            ...(typeof patch.enabled === 'boolean' ? { enabled: patch.enabled } : {}),
          } : { ...patch }
          if (safePatch.password && isSecretMask(safePatch.password)) safePatch.password = source.password
          const next = { ...source, ...safePatch, id: source.id, updatedAt: Date.now() }
          return stampSecretRotations(next, source, { password: [next.password, source.password] })
        }),
      },
    }
    await saveSettings(store.settings)
    return publicConnectorSettings().email
  })

  handle('email:removeSource', async (_e, id: string) => {
    requireSecurityRole('operator', 'remove email sources')
    if (store.settings.email.sources.some(source => source.id === id && source.kind === 'microsoft_graph')) {
      requireSecurityRole('owner', 'disconnect Microsoft mail')
      await microsoftMailService().remove(id)
    }
    store.settings = { ...store.settings, email: { ...store.settings.email, sources: store.settings.email.sources.filter((source) => source.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().email
  })

  handle('email:sync', async () => {
    requireSecurityRole('operator', 'sync email')
    return syncQueue.run('email', async () => {
      const result = await syncEmailConnector(store.settings.email, store.connectorEvents)
      result.events = result.events.filter(event => event.sources?.every(ref => {
        const current = store.settings.email.sources.find(source => source.id === ref.sourceId && source.enabled)
        return current && (!ref.sourceRevision || ref.sourceRevision === connectorSourceBinding(current))
      }))
      store.settings = { ...store.settings, email: { ...store.settings.email, sources: mergeConnectorSyncSources(store.settings.email.sources, result.sources), lastSyncAt: result.syncedAt } }
      await saveSettings(store.settings)

      const enriched = enrichContacts(store.contacts, result.events, emailContactObservations(result.events))
      store.contacts = enriched.directory
      persistContacts()

      const attachmentsByEvent = result.attachmentsByEvent
      if (attachmentsByEvent) {
        await attachMediaBytesPaths(enriched.events, (event) => attachmentsByEvent[event.id] ?? [])
      }
      rememberConnectorEvents(enriched.events)
      const { attachmentsByEvent: _attachmentsByEvent, ...publicResult } = result
      return { ...publicResult, sources: publicConnectorSettings().email.sources, events: enriched.events }
    })
  })

  handle('email:list', async (_e, opts: EmailListOptions = {}) => {
    requireSecurityRole('operator', 'list email')
    return (await scanEmailMessages(store.settings.email, opts)).messages
  })

  handle('email:read', async (_e, id: string) => {
    requireSecurityRole('operator', 'read email')
    return readEmailMessage(store.settings.email, id)
  })

  handle('email:thread', async (_e, id: string) => {
    requireSecurityRole('operator', 'read email threads')
    return readEmailThread(store.settings.email, id)
  })

  handle('email:summary', async (_e, opts: EmailListOptions = {}) => {
    requireSecurityRole('operator', 'summarize email')
    return summarizeEmail(store.settings.email, opts)
  })

  handle('email:compose', async (_e, input: EmailDraftInput) => {
    requireSecurityRole('operator', 'compose email drafts')
    let sources: ConnectorEvent['sources'] = []
    const normalized: EmailDraftInput = { ...input }
    if (input.threadKey || input.inReplyTo) {
      const thread = await readEmailThread(store.settings.email, input.threadKey || input.inReplyTo || '')
      sources = thread.messages.map(emailSourceRef)
      normalized.threadKey = `email:${thread.root.id}`
      normalized.inReplyTo = thread.root.id
    }
    return recordEmailDraft(normalized, sources)
  })

  handle('email:resolve', async (_e, input: EmailResolveInput) => {
    requireSecurityRole('operator', 'resolve email threads')
    const thread = await readEmailThread(store.settings.email, input.id)
    return recordEmailEvent(emailThreadResolvedEvent(input, thread))
  })
}
