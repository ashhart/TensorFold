export type PhonePlatform = 'android' | 'ios'
export type PhoneTransport = 'usb' | 'wifi' | 'emulator' | 'mirroring'
export interface PhoneDevice {
  id: string
  name: string
  platform: PhonePlatform
  transport: PhoneTransport
  state: 'ready' | 'unauthorized' | 'offline' | 'setup-required'
  detail: string | null
}
export interface PhoneBounds { x: number; y: number; width: number; height: number }
export interface PhoneNode {
  id: string
  text: string
  description: string
  role: string
  bounds: PhoneBounds
  editable: boolean
  secure: boolean
  enabled?: boolean
  focused?: boolean
  interactive?: boolean
}
export interface PhoneObservation {
  id: string
  sessionId: string
  deviceId: string
  capturedAt: number
  width: number
  height: number
  image: string | null
  nodes: PhoneNode[]
  app: string | null
  warnings: string[]
}
export interface PhoneSession {
  id: string
  deviceId: string
  name: string
  platform: PhonePlatform
  transport: PhoneTransport
  threadId: string | null
  actions: number
  startedAt: number
  grantExpiresAt?: number | null
}
export interface PhoneSnapshot {
  supported: boolean
  adb: { available: boolean; path: string | null; version: string | null }
  devices: PhoneDevice[]
  session: PhoneSession | null
  observation: PhoneObservation | null
  busy: boolean
  error: string | null
  mirroring: { available: boolean; detail: string | null }
}
export interface PhoneConnectInput { deviceId: string }
export interface PhoneDeveloperConnectInput { endpoint: string; transport: 'usb' | 'wifi' }
export interface PhonePairInput { endpoint: string; code: string }
export type PhoneAction =
  | { action: 'tap'; x: number; y: number }
  | { action: 'swipe'; x: number; y: number; toX: number; toY: number; durationMs?: number }
  | { action: 'type'; text: string }
  | { action: 'home' | 'back' | 'enter' }
  | { action: 'launch'; app: string }
export type PhoneActionInput = PhoneAction & { sessionId: string; observationId: string }
export type PhoneToolInput = { action: 'observe'; sessionId?: string } | (PhoneActionInput & { note?: string })
