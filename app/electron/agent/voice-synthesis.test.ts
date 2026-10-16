import assert from 'node:assert/strict'
import {
  ELEVENLABS_DEFAULT_MODEL,
  ELEVENLABS_DEFAULT_VOICE,
  LOCAL_TTS_DEFAULT_VOICE,
  OPENAI_TTS_DEFAULT_MODEL,
  buildElevenLabsRequest,
  buildLocalTtsRequest,
  buildOpenAiVoiceRequest,
  localTtsModelsUrl,
  redactVoiceSecret,
} from './voice-synthesis'
import type { VoiceSettings } from './types'

const BASE: VoiceSettings = {
  enabled: true,
  ttsEnabled: true,
  sttEnabled: true,
  ttsProvider: 'elevenlabs',
  sttProvider: 'browser',
  voiceName: '',
  rate: 1.6,
  pitch: 1,
  volume: 1,
  autoSummarizeBeforeRead: false,
  summaryMaxChars: 900,
  autoReadRoomUpdates: false,
  apiBaseUrl: '',
  apiKey: '',
  ttsModel: '',
  sttModel: '',
  elevenLabsApiKey: '',
  elevenLabsVoiceId: '',
  elevenLabsModelId: '',
  elevenLabsStability: 0.4,
  elevenLabsStyle: 0.3,
  localTtsBaseUrl: '',
  localTtsVoice: '',
  localTtsEmotion: '',
}

function main() {
  // ElevenLabs: voiceId goes into the path, key into the header, defaults when blank.
  {
    const req = buildElevenLabsRequest('Hello world', BASE, 'sk-secret-key-123456')
    assert.equal(req.url, `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_DEFAULT_VOICE}`)
    assert.equal(req.headers['xi-api-key'], 'sk-secret-key-123456')
    const body = JSON.parse(req.body)
    assert.equal(body.text, 'Hello world')
    assert.equal(body.model_id, ELEVENLABS_DEFAULT_MODEL)
    assert.equal(body.voice_settings.stability, 0.4)
    assert.equal(body.voice_settings.style, 0.3)
    console.log('ok: ElevenLabs default request shape')
  }

  // OpenAI-compatible TTS: stored key goes into the host request, never its settings mask.
  {
    const cfg: VoiceSettings = { ...BASE, ttsProvider: 'openai', apiBaseUrl: '', voiceName: '', rate: 9 }
    const req = buildOpenAiVoiceRequest('Read securely', cfg, 'stored-secret')
    assert.equal(req.url, 'https://api.openai.com/v1/audio/speech')
    assert.equal(req.headers.Authorization, 'Bearer stored-secret')
    const body = JSON.parse(req.body)
    assert.equal(body.model, OPENAI_TTS_DEFAULT_MODEL)
    assert.equal(body.voice, 'alloy')
    assert.equal(body.speed, 3)
    console.log('ok: OpenAI-compatible TTS host request shape')
  }

  // ElevenLabs: explicit voice/model + emotion knobs clamped to 0..1.
  {
    const cfg: VoiceSettings = { ...BASE, elevenLabsVoiceId: 'voiceX', elevenLabsModelId: 'eleven_turbo_v2', elevenLabsStability: 2, elevenLabsStyle: -1 }
    const req = buildElevenLabsRequest('Hi', cfg, 'k')
    assert.equal(req.url, 'https://api.elevenlabs.io/v1/text-to-speech/voiceX')
    const body = JSON.parse(req.body)
    assert.equal(body.model_id, 'eleven_turbo_v2')
    assert.equal(body.voice_settings.stability, 1) // clamped from 2
    assert.equal(body.voice_settings.style, 0) // clamped from -1
    console.log('ok: ElevenLabs overrides + clamp')
  }

  // Local TTS: default base/voice, speed clamped from rate, no emotion field when blank.
  {
    const req = buildLocalTtsRequest('Read this', BASE)
    assert.equal(req.url, 'http://127.0.0.1:8080/v1/audio/speech')
    const body = JSON.parse(req.body)
    assert.equal(body.voice, LOCAL_TTS_DEFAULT_VOICE)
    assert.equal(body.input, 'Read this')
    assert.equal(body.speed, 1.6)
    assert.equal('emotion' in body, false)
    console.log('ok: local TTS default request')
  }

  // Local TTS: custom base URL (trailing slash trimmed) + emotion + voice passthrough.
  {
    const cfg: VoiceSettings = { ...BASE, localTtsBaseUrl: 'http://127.0.0.1:9999/', localTtsVoice: 'af_bella', localTtsEmotion: 'cheerful', rate: 9 }
    const req = buildLocalTtsRequest('Yo', cfg)
    assert.equal(req.url, 'http://127.0.0.1:9999/v1/audio/speech')
    const body = JSON.parse(req.body)
    assert.equal(body.voice, 'af_bella')
    assert.equal(body.emotion, 'cheerful')
    assert.equal(body.speed, 3) // clamped from 9
    console.log('ok: local TTS overrides + emotion')
  }

  // Models URL builder trims trailing slash.
  {
    assert.equal(localTtsModelsUrl('http://127.0.0.1:8080/'), 'http://127.0.0.1:8080/v1/models')
    console.log('ok: localTtsModelsUrl')
  }

  // Redaction: removes the known key and generic header/Bearer leakage.
  {
    const key = 'sk-supersecretvalue'
    const line = `POST failed for xi-api-key: ${key} and Authorization: Bearer ${key}`
    const safe = redactVoiceSecret(line, key)
    assert.equal(safe.includes(key), false, 'known key removed')
    assert.equal(safe.includes('[redacted]'), true)
    const generic = redactVoiceSecret('{"xi-api-key":"abc123def456"}')
    assert.equal(generic.includes('abc123def456'), false, 'generic header value removed')
    console.log('ok: redactVoiceSecret')
  }

  console.log('all voice-synthesis tests passed')
}

main()
