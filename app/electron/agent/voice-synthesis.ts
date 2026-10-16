import type { VoiceSettings } from './types'

// Pure request builders for voice TTS providers (OpenAI-compatible, ElevenLabs, and a local
// expressive engine). Kept side-effect-free so request shape, key handling, and emotion
// mapping are unit-testable without a network. The main host does every provider fetch.

export const ELEVENLABS_DEFAULT_MODEL = 'eleven_multilingual_v2'
export const ELEVENLABS_DEFAULT_VOICE = 'EXAVITQu4vr4xnSDxMaL' // ElevenLabs "Sarah" preset
export const OPENAI_VOICE_DEFAULT_BASE_URL = 'https://api.openai.com/v1'
export const OPENAI_TTS_DEFAULT_MODEL = 'gpt-4o-mini-tts'
export const OPENAI_STT_DEFAULT_MODEL = 'whisper-1'
export const LOCAL_TTS_DEFAULT_BASE_URL = 'http://127.0.0.1:8080'
export const LOCAL_TTS_DEFAULT_VOICE = 'af_heart' // Kokoro-class expressive default

export interface VoiceHttpRequest {
  url: string
  headers: Record<string, string>
  body: string
}

function clamp01(n: number, fallback: number): number {
  if (!Number.isFinite(n)) return fallback
  return Math.min(1, Math.max(0, n))
}

function trimTrailingSlash(url: string): string {
  return url.replace(/\/+$/, '')
}

export function voiceApiBaseUrl(cfg: VoiceSettings): string {
  return trimTrailingSlash((cfg.apiBaseUrl || '').trim() || OPENAI_VOICE_DEFAULT_BASE_URL)
}

// OpenAI-compatible TTS request. The main process supplies the unmasked key from its
// protected settings store; renderer settings are never accepted as request input.
export function buildOpenAiVoiceRequest(text: string, cfg: VoiceSettings, apiKey: string): VoiceHttpRequest {
  const speed = Number.isFinite(cfg.rate) ? Math.min(3, Math.max(0.5, cfg.rate)) : 1
  return {
    url: `${voiceApiBaseUrl(cfg)}/audio/speech`,
    headers: {
      'content-type': 'application/json',
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
    },
    body: JSON.stringify({
      model: (cfg.ttsModel || '').trim() || OPENAI_TTS_DEFAULT_MODEL,
      voice: (cfg.voiceName || '').trim() || 'alloy',
      input: text,
      speed,
    }),
  }
}

// Build the ElevenLabs text-to-speech request. Emotion/character ride on voice_settings:
// stability (lower = more expressive/variable) and style (higher = more stylised delivery).
// The xi-api-key header carries the secret; callers must never log `headers`.
export function buildElevenLabsRequest(text: string, cfg: VoiceSettings, apiKey: string): VoiceHttpRequest {
  const voiceId = (cfg.elevenLabsVoiceId || '').trim() || ELEVENLABS_DEFAULT_VOICE
  const modelId = (cfg.elevenLabsModelId || '').trim() || ELEVENLABS_DEFAULT_MODEL
  const stability = clamp01(cfg.elevenLabsStability, 0.4)
  const style = clamp01(cfg.elevenLabsStyle, 0.3)
  return {
    url: `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voiceId)}`,
    headers: {
      'content-type': 'application/json',
      accept: 'audio/mpeg',
      'xi-api-key': apiKey,
    },
    body: JSON.stringify({
      text,
      model_id: modelId,
      voice_settings: {
        stability,
        similarity_boost: 0.75,
        style,
        use_speaker_boost: true,
      },
    }),
  }
}

// Build the local expressive-engine request. Targets an OpenAI-/Kokoro-compatible
// `/v1/audio/speech` on a loopback/LAN endpoint the user runs. voice + emotion are passed
// through so an engine that supports expressive presets (e.g. Kokoro emotion tags) can use
// them; engines that ignore unknown fields still synthesise plainly.
export function buildLocalTtsRequest(text: string, cfg: VoiceSettings): VoiceHttpRequest {
  const base = trimTrailingSlash((cfg.localTtsBaseUrl || '').trim() || LOCAL_TTS_DEFAULT_BASE_URL)
  const voice = (cfg.localTtsVoice || '').trim() || LOCAL_TTS_DEFAULT_VOICE
  const emotion = (cfg.localTtsEmotion || '').trim()
  const speed = Number.isFinite(cfg.rate) ? Math.min(3, Math.max(0.5, cfg.rate)) : 1
  return {
    url: `${base}/v1/audio/speech`,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'tts-1',
      input: text,
      voice,
      ...(emotion ? { emotion } : {}),
      speed,
      response_format: 'mp3',
    }),
  }
}

// The local engine's models/voices endpoint (OpenAI-compatible). Used by discovery to both
// prove reachability and surface installable expressive voices.
export function localTtsModelsUrl(baseUrl: string): string {
  return `${trimTrailingSlash(baseUrl || LOCAL_TTS_DEFAULT_BASE_URL)}/v1/models`
}

// Redact any secret-looking token from a string before it reaches a log sink. ElevenLabs
// keys are long opaque tokens; we also strip an explicitly-known key if provided.
export function redactVoiceSecret(text: string, knownKey = ''): string {
  let out = text
  const key = (knownKey || '').trim()
  if (key.length >= 4) out = out.split(key).join('[redacted]')
  // Generic xi-api-key / Bearer style leakage.
  out = out.replace(/(xi-api-key"?\s*[:=]\s*"?)[^"\s,}]+/gi, '$1[redacted]')
  out = out.replace(/(Bearer\s+)[A-Za-z0-9._-]{8,}/g, '$1[redacted]')
  return out
}
