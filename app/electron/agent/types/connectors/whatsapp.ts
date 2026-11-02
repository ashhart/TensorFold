

export interface WhatsAppBridgeSource {
  id: string
  label: string
  enabled: boolean
  // Exact conversation JIDs selected by the owner; empty means ingest no contacts.
  allowedChatJids?: string[]
  // Only the authenticated account's own chat, never arbitrary fromMe messages.
  allowSelfChat?: boolean
  phoneNumber: string | null
  paired: boolean
  lastQrAt: number | null
  lastEventAt: number | null
  lastError: string | null
  // Main-process-authored timestamp of the last actual paired-credential update.
  credentialRotatedAt?: number | null
  createdAt: number
  updatedAt: number
}

export type WhatsAppBridgePhase = 'disconnected' | 'connecting' | 'pairing' | 'connected' | 'reconnecting' | 'error'

// Ephemeral host state; QR material is never persisted in settings.
export interface WhatsAppBridgeRuntimeSnapshot {
  sourceId: string
  phase: WhatsAppBridgePhase
  qr: string | null
  qrExpiresAt: number | null
  phoneNumber: string | null
  reconnectAttempt: number
  nextReconnectAt: number | null
  lastError: string | null
  updatedAt: number
}

export interface WhatsAppBridgeConnectorSettings {
  sources: WhatsAppBridgeSource[]
  lastSyncAt: number | null
}
