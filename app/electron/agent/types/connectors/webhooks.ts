import type { SecretRotationMetadata } from '../base'

// Inbound/outbound webhooks (build.md Phase 2), inbound-first with a loopback signed listener.
export type WebhookDirection = 'inbound' | 'outbound'

export type WebhookSignatureMode = 'hmac-sha256'

export type WebhookCapturePolicy = 'metadata' | 'redacted_payload' | 'ttl_payload'

export type WebhookTriggerPolicy = 'notify' | 'suggest' | 'approval' | 'run_automation'

export type WebhookEventKind = 'agent' | 'room' | 'ticket' | 'automation' | 'mission' | 'connector' | 'test'

export interface WebhookDeliveryRecord {
  id: string
  direction: WebhookDirection
  eventKind: WebhookEventKind
  urlOrPath: string
  status: 'queued' | 'delivered' | 'failed' | 'deduped'
  attempts: number
  httpStatus: number | null
  error: string | null
  idempotencyKey: string
  payloadPreview: string
  createdAt: number
  finishedAt: number | null
}

export interface WebhookEndpoint {
  id: string
  name: string
  direction: WebhookDirection
  enabled: boolean
  scope: { projectId?: string | null; workspaceId?: string | null }
  url?: string // outbound only
  path?: string // inbound route, e.g. '/hooks/github'
  secretRef: string // HMAC secret (safeStorage-protected at rest, masked to the renderer)
  signatureMode: WebhookSignatureMode
  replayWindowSeconds: number
  capturePolicy: WebhookCapturePolicy
  triggerPolicy: WebhookTriggerPolicy
  runAutomationId?: string | null // which automation 'run_automation' runs (never built from payload)
  eventKinds?: WebhookEventKind[] // outbound filter; empty means all event kinds
  approvalRequired?: boolean
  approvedAt?: number | null
  deliveries?: WebhookDeliveryRecord[]
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}

export interface WebhookServerStatus {
  listening: boolean
  port: number | null
  routes: number
  lastDeliveryAt: number | null
}

export interface WebhookSnapshot {
  endpoints: WebhookEndpoint[]
  listener: WebhookServerStatus
}
