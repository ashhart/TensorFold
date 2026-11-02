import type { SecretRotationMetadata } from '../base'

export type VoiceProvider = 'browser' | 'openai' | 'custom' | 'elevenlabs' | 'local' | 'kokoro'

export interface VoiceSettings {
  enabled: boolean
  ttsEnabled: boolean
  sttEnabled: boolean
  ttsProvider: VoiceProvider
  sttProvider: VoiceProvider
  voiceName: string
  rate: number
  pitch: number
  volume: number
  autoSummarizeBeforeRead: boolean
  summaryMaxChars: number
  autoReadRoomUpdates: boolean

  apiBaseUrl: string
  apiKey: string
  ttsModel: string
  sttModel: string

  elevenLabsApiKey: string
  elevenLabsVoiceId: string
  elevenLabsModelId: string
  elevenLabsStability: number
  elevenLabsStyle: number

  localTtsBaseUrl: string
  localTtsVoice: string
  localTtsEmotion: string
  // Managed offline Kokoro. Optional for backward-compatible saved settings.
  kokoroVoice?: string
  secretRotations?: SecretRotationMetadata
}

export interface VoiceSynthesisResult {
  ok: boolean
  audioBase64?: string
  mimeType?: string
  error?: string
  // 'fallback' tells the renderer to use the local Web Speech voice instead (engine down).
  fallback?: boolean
}

export interface VoiceTranscriptionInput {
  audioBase64: string
  mimeType?: string
}

export interface VoiceTranscriptionResult {
  ok: boolean
  text?: string
  error?: string
}

export interface LocalTtsDiscovery {
  found: boolean
  baseUrl: string
  voices: string[]
  message: string
}
