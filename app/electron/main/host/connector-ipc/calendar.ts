import { calendarMutationEvent, makeCalendarSource, readCalendarEvent, scanCalendarEvents, summarizeCalendarDay } from '../../../agent/calendar-connector'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'
import { saveSettings } from '../../../store'
import { requireSecurityRole } from '../audit'
import { handle as defaultHandle } from '../ipc'
import { store } from '../../state'
import { publicConnectorSettings } from '../connector-event-spine'
import type { CalendarListOptions, CalendarMutationInput, CalendarSourceConfig } from '../../../agent/types'
import type { ConnectorIpcDeps } from './types'

export function registerCalendarIpc(deps: ConnectorIpcDeps): void {
  const handle = deps.handle ?? defaultHandle
  const { spine, syncQueue } = deps
  const { recordCalendarEvent } = spine

  handle('calendar:addSource', async (_e, target: string) => {
    requireSecurityRole('operator', 'add calendar sources')
    const source = makeCalendarSource(target)
    store.settings = { ...store.settings, calendar: { ...store.settings.calendar, sources: [source, ...store.settings.calendar.sources] } }
    await saveSettings(store.settings)
    return publicConnectorSettings().calendar
  })

  handle('calendar:updateSource', async (_e, { id, patch }: { id: string; patch: Partial<CalendarSourceConfig> }) => {
    requireSecurityRole('operator', 'update calendar sources')
    store.settings = {
      ...store.settings,
      calendar: {
        ...store.settings.calendar,
        sources: store.settings.calendar.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source),
      },
    }
    await saveSettings(store.settings)
    return publicConnectorSettings().calendar
  })

  handle('calendar:removeSource', async (_e, id: string) => {
    requireSecurityRole('operator', 'remove calendar sources')
    store.settings = { ...store.settings, calendar: { ...store.settings.calendar, sources: store.settings.calendar.sources.filter((source) => source.id !== id) } }
    await saveSettings(store.settings)
    return publicConnectorSettings().calendar
  })

  handle('calendar:sync', async () => {
    requireSecurityRole('operator', 'sync calendar')
    return syncQueue.run('calendar', async () => {
      const syncedAt = Date.now()
      const result = await scanCalendarEvents(store.settings.calendar, { limit: store.settings.calendar.maxEvents })
      store.settings = { ...store.settings, calendar: { ...store.settings.calendar, sources: mergeConnectorSyncSources(store.settings.calendar.sources, result.sources), lastSyncAt: syncedAt } }
      await saveSettings(store.settings)
      return { events: result.events, sources: publicConnectorSettings().calendar.sources, syncedAt }
    })
  })

  handle('calendar:list', async (_e, opts: CalendarListOptions = {}) => {
    requireSecurityRole('operator', 'list calendar events')
    return (await scanCalendarEvents(store.settings.calendar, opts)).events
  })

  handle('calendar:read', async (_e, id: string) => {
    requireSecurityRole('operator', 'read calendar events')
    return readCalendarEvent(store.settings.calendar, id)
  })

  handle('calendar:summary', async (_e, opts: CalendarListOptions = {}) => {
    requireSecurityRole('operator', 'summarize calendar')
    return summarizeCalendarDay(store.settings.calendar, opts)
  })

  handle('calendar:create', async (_e, input: CalendarMutationInput) => {
    requireSecurityRole('operator', 'request calendar event creation')
    const enabledSources = store.settings.calendar.sources.filter((source) => source.enabled)
    const source = input.sourceId
      ? enabledSources.find((item) => item.id === input.sourceId)
      : enabledSources[0]
    if (!source) throw new Error('No enabled calendar source is available for the create request.')
    const event = calendarMutationEvent({ ...input, mode: 'create', sourceId: source.id })
    return recordCalendarEvent(event)
  })

  handle('calendar:update', async (_e, input: CalendarMutationInput & { id: string }) => {
    requireSecurityRole('operator', 'request calendar event update')
    const existing = await readCalendarEvent(store.settings.calendar, input.id)
    const event = calendarMutationEvent({ ...input, mode: 'update', sourceId: existing.sourceId }, Date.now(), existing)
    return recordCalendarEvent(event)
  })
}
