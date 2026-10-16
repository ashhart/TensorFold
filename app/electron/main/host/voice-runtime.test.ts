import assert from 'node:assert/strict'
import type { VoiceSettings } from '../../agent/types'
import { store } from '../state'
import { assertVoiceApiBaseUrl, synthesizeElevenLabs, synthesizeLocalTts, synthesizeOpenAiCompatible, transcribeOpenAiCompatible, type VoiceRuntimeDeps } from './voice-runtime'

const VOICE: VoiceSettings = {
  enabled: true,
  ttsEnabled: true,
  sttEnabled: true,
  ttsProvider: 'openai',
  sttProvider: 'openai',
  voiceName: 'alloy',
  rate: 1.2,
  pitch: 1,
  volume: 1,
  autoSummarizeBeforeRead: false,
  summaryMaxChars: 900,
  autoReadRoomUpdates: false,
  apiBaseUrl: 'https://voice.example/v1',
  apiKey: 'stored-main-key',
  ttsModel: 'tts-model',
  sttModel: 'stt-model',
  elevenLabsApiKey: '',
  elevenLabsVoiceId: '',
  elevenLabsModelId: '',
  elevenLabsStability: 0.4,
  elevenLabsStyle: 0.3,
  localTtsBaseUrl: '',
  localTtsVoice: '',
  localTtsEmotion: '',
}

store.settings = { voice: { ...VOICE } } as any

const noAudit: NonNullable<VoiceRuntimeDeps['audit']> = async () => {}

// TTS gets provider/base/model/key solely from the protected main store. Its IPC-facing
// function accepts text only, so a masked renderer Settings object cannot become a header.
{
  const requests: Array<{ url: string; init?: RequestInit }> = []
  const result = await synthesizeOpenAiCompatible('Host-owned credentials', {
    validateVoiceApiBaseUrl: async (raw) => { assert.equal(raw, VOICE.apiBaseUrl); return raw },
    fetchImpl: (async (url: RequestInfo | URL, init?: RequestInit) => {
      requests.push({ url: String(url), init })
      return new Response(new Uint8Array([1, 2, 3]), { status: 200, headers: { 'content-type': 'audio/mpeg' } })
    }) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(result.ok, true)
  assert.equal(result.audioBase64, 'AQID')
  const request = requests[0]!
  assert.equal(request.url, 'https://voice.example/v1/audio/speech')
  assert.equal((request.init?.headers as Record<string, string>).Authorization, 'Bearer stored-main-key')
  assert.equal(request.init?.redirect, 'error', 'redirects cannot pivot a vetted endpoint into an SSRF target')
  assert.deepEqual(JSON.parse(String(request.init?.body)), {
    model: 'tts-model', voice: 'alloy', input: 'Host-owned credentials', speed: 1.2,
  })
}

// Every TTS backend rejects wrong MIME and both declared and chunked over-limit bodies.
{
  const wrongMime = await synthesizeOpenAiCompatible('not audio', {
    validateVoiceApiBaseUrl: async (raw) => raw,
    fetchImpl: (async () => new Response('<html>nope</html>', { headers: { 'content-type': 'text/html' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(wrongMime.ok, false)
  assert.match(wrongMime.error ?? '', /non-audio/i)

  const chunk = 13 * 1024 * 1024
  const chunkedTooLarge = await synthesizeOpenAiCompatible('bounded stream', {
    validateVoiceApiBaseUrl: async (raw) => raw,
    fetchImpl: (async () => new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(chunk))
        controller.enqueue(new Uint8Array(chunk))
        controller.close()
      },
    }), { headers: { 'content-type': 'audio/mpeg' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(chunkedTooLarge.ok, false)
  assert.match(chunkedTooLarge.error ?? '', /too large/i)

  store.settings.voice = { ...VOICE, ttsProvider: 'elevenlabs', elevenLabsApiKey: 'eleven-secret', elevenLabsVoiceId: 'voice-1' }
  const elevenWrongMime = await synthesizeElevenLabs('wrong mime', {
    fetchImpl: (async () => new Response('{}', { headers: { 'content-type': 'application/json' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(elevenWrongMime.ok, false)
  assert.match(elevenWrongMime.error ?? '', /non-audio/i)
  const elevenTooLarge = await synthesizeElevenLabs('too large', {
    fetchImpl: (async () => new Response(new Uint8Array([1]), { headers: { 'content-type': 'audio/mpeg', 'content-length': '999999999' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(elevenTooLarge.ok, false)
  assert.match(elevenTooLarge.error ?? '', /too large/i)

  store.settings.voice = { ...VOICE, ttsProvider: 'local', localTtsBaseUrl: 'http://127.0.0.1:8080' }
  const localWrongMime = await synthesizeLocalTts('wrong mime', {
    fetchImpl: (async () => new Response('proxy error', { headers: { 'content-type': 'text/plain' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(localWrongMime.ok, false)
  assert.equal(localWrongMime.fallback, true)
  assert.match(localWrongMime.error ?? '', /non-audio/i)
  const localTooLarge = await synthesizeLocalTts('too large', {
    fetchImpl: (async () => new Response(new Uint8Array([1]), { headers: { 'content-type': 'audio/wav', 'content-length': '999999999' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(localTooLarge.ok, false)
  assert.match(localTooLarge.error ?? '', /too large/i)
  store.settings.voice = { ...VOICE }
}

// Transcription responses are bounded JSON; HTML and oversized JSON never reach res.json().
{
  const input = { audioBase64: Buffer.from('audio').toString('base64'), mimeType: 'audio/webm' }
  const wrongMime = await transcribeOpenAiCompatible(input, {
    validateVoiceApiBaseUrl: async (raw) => raw,
    fetchImpl: (async () => new Response('<html>bad gateway</html>', { headers: { 'content-type': 'text/html' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(wrongMime.ok, false)
  assert.match(wrongMime.error ?? '', /non-json/i)
  const tooLarge = await transcribeOpenAiCompatible(input, {
    validateVoiceApiBaseUrl: async (raw) => raw,
    fetchImpl: (async () => new Response('{"text":"x"}', { headers: { 'content-type': 'application/json', 'content-length': '999999999' } })) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(tooLarge.ok, false)
  assert.match(tooLarge.error ?? '', /too large/i)
}

// STT likewise receives only bounded recorded audio over IPC and constructs multipart/auth
// in main using the stored key/model.
{
  const requests: Array<{ url: string; init?: RequestInit }> = []
  const result = await transcribeOpenAiCompatible({
    audioBase64: Buffer.from('recorded-audio').toString('base64'),
    mimeType: 'audio/webm;codecs=opus',
  }, {
    validateVoiceApiBaseUrl: async (raw) => raw,
    fetchImpl: (async (url: RequestInfo | URL, init?: RequestInit) => {
      requests.push({ url: String(url), init })
      return Response.json({ text: 'transcribed securely' })
    }) as typeof fetch,
    audit: noAudit,
  })
  assert.deepEqual(result, { ok: true, text: 'transcribed securely' })
  const request = requests[0]!
  assert.equal(request.url, 'https://voice.example/v1/audio/transcriptions')
  assert.equal((request.init?.headers as Record<string, string>).Authorization, 'Bearer stored-main-key')
  assert.equal(request.init?.redirect, 'error')
  const form = request.init?.body as FormData
  assert.equal(form.get('model'), 'stt-model')
  const file = form.get('file') as File
  assert.equal(file.name, 'dictation.webm')
  assert.equal(file.type, 'audio/webm')
}

// Mask sentinel values are renderer-only and must never be emitted as credentials even if
// an invariant violation put one into the main store.
{
  store.settings.voice = { ...VOICE, apiKey: '••••••••' }
  let fetched = false
  const result = await synthesizeOpenAiCompatible('no mask header', {
    validateVoiceApiBaseUrl: async (raw) => raw,
    fetchImpl: (async () => { fetched = true; return new Response() }) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(result.ok, false)
  assert.match(result.error ?? '', /Add a Voice API key/)
  assert.equal(fetched, false)
  store.settings.voice = { ...VOICE }
}

// Endpoint policy rejects plaintext, embedded credentials, modified URLs, and private or
// metadata destinations before fetch.
await assert.rejects(() => assertVoiceApiBaseUrl('http://voice.example/v1'), /HTTPS/)
await assert.rejects(() => assertVoiceApiBaseUrl('https://user:pass@voice.example/v1'), /must not include credentials/)
await assert.rejects(() => assertVoiceApiBaseUrl('https://voice.example/v1?target=other'), /query string or fragment/)
await assert.rejects(() => assertVoiceApiBaseUrl('https://127.0.0.1/v1'), /local\/private\/loopback/)
await assert.rejects(() => assertVoiceApiBaseUrl('https://169.254.169.254/latest'), /local\/private\/link-local\/metadata/)

// Errors are scrubbed with the same stored key before returning to the renderer or audit.
{
  const result = await synthesizeOpenAiCompatible('redact failure', {
    validateVoiceApiBaseUrl: async (raw) => raw,
    fetchImpl: (async () => { throw new Error(`request failed with ${VOICE.apiKey}`) }) as typeof fetch,
    audit: noAudit,
  })
  assert.equal(result.ok, false)
  assert.equal(result.error?.includes(VOICE.apiKey), false)
  assert.match(result.error ?? '', /\[redacted\]/)
}

console.log('voice runtime credential, endpoint, TTS, and STT tests passed')
