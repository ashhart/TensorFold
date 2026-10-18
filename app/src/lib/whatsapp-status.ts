import type { Settings, WhatsAppBridgeRuntimeSnapshot } from './types'

/** A setup remount's async snapshot must never replace a newer lifecycle push. */
export function applyWhatsAppRuntimeSnapshot(current: Record<string, WhatsAppBridgeRuntimeSnapshot>, snapshot: WhatsAppBridgeRuntimeSnapshot): Record<string, WhatsAppBridgeRuntimeSnapshot> {
  const previous = current[snapshot.sourceId]
  return previous && previous.updatedAt >= snapshot.updatedAt ? current : { ...current, [snapshot.sourceId]: snapshot }
}

export interface WhatsAppStatusPush {
  sourceId: string
  paired?: boolean
  phoneNumber?: string | null
  credentialRotatedAt?: number | null
  enabled?: boolean
  lastQrAt?: number | null
  lastEventAt?: number | null
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
            ...('phoneNumber' in status ? { phoneNumber: status.phoneNumber ?? null } : {}),
            ...('credentialRotatedAt' in status ? { credentialRotatedAt: status.credentialRotatedAt ?? null } : {}),
            ...('enabled' in status ? { enabled: status.enabled } : {}),
            ...('lastQrAt' in status ? { lastQrAt: status.lastQrAt ?? null } : {}),
            lastError: status.lastError !== undefined ? status.lastError : status.paired === true ? null : source.lastError,
            lastEventAt: status.lastEventAt !== undefined ? status.lastEventAt : status.paired === true ? now : source.lastEventAt,
          }
        : source),
    },
  }
}
