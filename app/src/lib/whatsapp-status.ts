import type { Settings } from './types'

export interface WhatsAppStatusPush {
  sourceId: string
  paired?: boolean
  lastError?: string | null
}

/** Apply host-authored bridge status to a local Settings draft without persisting it. */
export function applyWhatsAppStatusPush(settings: Settings, status: WhatsAppStatusPush, now = Date.now()): Settings {
  return {
    ...settings,
    whatsapp: {
      ...settings.whatsapp,
      sources: settings.whatsapp.sources.map((source) => source.id === status.sourceId
        ? {
            ...source,
            paired: status.paired ?? source.paired,
            lastError: status.lastError ?? null,
            lastEventAt: now,
          }
        : source),
    },
  }
}
