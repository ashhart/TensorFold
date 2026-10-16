import { buildElevenLabsRequest, buildLocalTtsRequest, buildOpenAiVoiceRequest, localTtsModelsUrl, redactVoiceSecret, voiceApiBaseUrl, LOCAL_TTS_DEFAULT_BASE_URL, OPENAI_STT_DEFAULT_MODEL } from '../../agent/voice-synthesis'
import { assertConnectorBaseUrlResolved, isSecretMask } from '../../agent/security'
import { createConnectorFetch, createLocalProviderFetch } from '../../agent/secure-fetch'
import { discardResponseBody } from '../../agent/pinned-fetch'
import type { LocalTtsDiscovery, VoiceSettings, VoiceSynthesisResult, VoiceTranscriptionInput, VoiceTranscriptionResult } from '../../agent/types'
import { store } from '../state'
import { auditEvent } from './audit'

const SYNTH_TIMEOUT_MS = 20_000
const DISCOVER_TIMEOUT_MS = 2_500
const MAX_TTS_CHARS = 5_000
const MAX_TTS_AUDIO_BYTES = 25 * 1024 * 1024
const MAX_STT_AUDIO_BYTES = 20 * 1024 * 1024
const MAX_STT_BASE64_CHARS = Math.ceil(MAX_STT_AUDIO_BYTES / 3) * 4 + 4
const MAX_TRANSCRIPT_CHARS = 100_000
const MAX_STT_RESPONSE_BYTES = 1024 * 1024

export interface VoiceRuntimeDeps {
  fetchImpl?: typeof fetch
  validateVoiceApiBaseUrl?: (raw: string) => Promise<string>
  audit?: (kind: string, data: Record<string, unknown>) => Promise<void>
}

// Resolve the unmasked voice config from the in-memory store (loadSettings reveals secrets at
// boot), so the renderer never hands us any cleartext provider key.
function voiceConfig(): VoiceSettings {
  return store.settings.voice
}

function errorText(e: unknown, key = ''): string {
  const raw = String((e as { cause?: { message?: string; code?: string }; message?: string })?.cause?.message
    || (e as { cause?: { code?: string } })?.cause?.code
    || (e as { message?: string })?.message
    || e
    || 'request failed')
  return redactVoiceSecret(raw, key).slice(0, 240)
}

function uniqueSorted(values: Array<string | null | undefined>): string[] {
  return Array.from(new Set(values.map((v) => String(v || '').trim()).filter(Boolean))).sort()
}

function protectedVoiceApiKey(cfg: VoiceSettings): string {
  const key = String(cfg.apiKey || '').trim()
  return !key || isSecretMask(key) ? '' : key
}

function runtimeFetch(deps?: VoiceRuntimeDeps): typeof fetch {
  return deps?.fetchImpl ?? fetch
}

function cloudVoiceFetch(baseUrl: string, deps: VoiceRuntimeDeps): typeof fetch {
  // A custom validator is an explicit unit-test seam; production always uses the real
  // endpoint policy and a DNS-pinned transport immediately before the request.
  return deps.validateVoiceApiBaseUrl
    ? runtimeFetch(deps)
    : createConnectorFetch(baseUrl, 'Voice API', {}, runtimeFetch(deps))
}

async function runtimeAudit(deps: VoiceRuntimeDeps | undefined, kind: string, data: Record<string, unknown>): Promise<void> {
  await (deps?.audit ?? auditEvent)(kind, data)
}

function normalizedMime(response: Response): string {
  return String(response.headers.get('content-type') || '').split(';', 1)[0].trim().toLowerCase()
}

function requireAudioResponse(response: Response, label: string): string {
  const mime = normalizedMime(response)
  if (!mime.startsWith('audio/')) throw new Error(`${label} returned a non-audio response (${mime || 'missing content-type'}).`)
  return mime
}

function requireJsonResponse(response: Response, label: string): void {
  const mime = normalizedMime(response)
  if (mime !== 'application/json' && !mime.endsWith('+json')) {
    throw new Error(`${label} returned a non-JSON response (${mime || 'missing content-type'}).`)
  }
}

export async function readBoundedVoiceBody(response: Response, maxBytes: number, label: string): Promise<Buffer> {
  const declared = Number(response.headers.get('content-length') ?? 0)
  if (Number.isFinite(declared) && declared > maxBytes) {
    await discardResponseBody(response)
    throw new Error(`${label} response was too large.`)
  }
  if (!response.body) return Buffer.alloc(0)
  const reader = response.body.getReader()
  const chunks: Buffer[] = []
  let bytes = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = Buffer.from(value)
      bytes += chunk.length
      if (bytes > maxBytes) {
        await reader.cancel().catch(() => {})
        throw new Error(`${label} response was too large.`)
      }
      chunks.push(chunk)
    }
  } finally {
    reader.releaseLock()
  }
  return Buffer.concat(chunks, bytes)
}

async function boundedAudio(response: Response, label: string): Promise<{ buffer: Buffer; mimeType: string }> {
  const mimeType = requireAudioResponse(response, label)
  const buffer = await readBoundedVoiceBody(response, MAX_TTS_AUDIO_BYTES, `${label} audio`)
  return { buffer, mimeType }
}

async function boundedJson(response: Response, label: string): Promise<unknown> {
  requireJsonResponse(response, label)
  const buffer = await readBoundedVoiceBody(response, MAX_STT_RESPONSE_BYTES, label)
  try { return JSON.parse(buffer.toString('utf8')) } catch { return null }
}

// Voice API calls carry a protected Bearer key, so custom endpoints must be public HTTPS,
// credential-free URLs. The resolved connector guard rejects loopback/private/link-local/
// metadata destinations and DNS answers before any audio or secret leaves the process.
export async function assertVoiceApiBaseUrl(raw: string): Promise<string> {
  let url: URL
  try { url = new URL(String(raw || '').trim()) } catch { throw new Error('Voice API base URL is invalid.') }
  if (url.protocol !== 'https:') throw new Error('Voice API base URL must use HTTPS.')
  if (url.username || url.password) throw new Error('Voice API base URL must not include credentials.')
  if (url.search || url.hash) throw new Error('Voice API base URL must not include a query string or fragment.')
  await assertConnectorBaseUrlResolved(url.href, 'Voice API')
  return url.href.replace(/\/+$/, '')
}

export async function synthesizeOpenAiCompatible(text: string, deps: VoiceRuntimeDeps = {}): Promise<VoiceSynthesisResult> {
  const cfg = voiceConfig()
  if (cfg.ttsProvider !== 'openai' && cfg.ttsProvider !== 'custom') return { ok: false, error: 'OpenAI-compatible speech is not selected.' }
  const apiKey = protectedVoiceApiKey(cfg)
  if (cfg.ttsProvider === 'openai' && !apiKey) return { ok: false, error: 'Add a Voice API key in Voice settings to enable API speech.' }
  const body = String(text || '').slice(0, MAX_TTS_CHARS).trim()
  if (!body) return { ok: false, error: 'Nothing to read.' }

  const validate = deps.validateVoiceApiBaseUrl ?? assertVoiceApiBaseUrl
  try {
    const baseUrl = await validate(voiceApiBaseUrl(cfg))
    const req = buildOpenAiVoiceRequest(body, { ...cfg, apiBaseUrl: baseUrl }, apiKey)
    const res = await cloudVoiceFetch(baseUrl, deps)(req.url, {
      method: 'POST',
      headers: req.headers,
      body: req.body,
      redirect: 'error',
      signal: AbortSignal.timeout(SYNTH_TIMEOUT_MS),
    })
    if (!res.ok) {
      await discardResponseBody(res)
      await runtimeAudit(deps, 'voice.synthesize', { provider: cfg.ttsProvider, ok: false, status: res.status })
      return { ok: false, error: `Voice API returned HTTP ${res.status}.` }
    }
    const { buffer, mimeType } = await boundedAudio(res, 'Voice API')
    await runtimeAudit(deps, 'voice.synthesize', { provider: cfg.ttsProvider, ok: true, bytes: buffer.length })
    return { ok: true, audioBase64: buffer.toString('base64'), mimeType }
  } catch (e: unknown) {
    const error = errorText(e, apiKey)
    await runtimeAudit(deps, 'voice.synthesize', { provider: cfg.ttsProvider, ok: false, error })
    return { ok: false, error }
  }
}

export async function synthesizeConfiguredVoice(text: string): Promise<VoiceSynthesisResult> {
  const provider = voiceConfig().ttsProvider
  if (provider === 'elevenlabs') return synthesizeElevenLabs(text)
  if (provider === 'openai' || provider === 'custom') return synthesizeOpenAiCompatible(text)
  return { ok: false, error: 'The selected text-to-speech provider does not use cloud synthesis.' }
}

const AUDIO_TYPES: Record<string, string> = {
  'audio/webm': 'webm',
  'audio/mp4': 'm4a',
  'audio/mpeg': 'mp3',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
  'audio/ogg': 'ogg',
}

export async function transcribeOpenAiCompatible(input: VoiceTranscriptionInput, deps: VoiceRuntimeDeps = {}): Promise<VoiceTranscriptionResult> {
  const cfg = voiceConfig()
  if (cfg.sttProvider !== 'openai' && cfg.sttProvider !== 'custom') return { ok: false, error: 'API speech-to-text is not selected.' }
  const apiKey = protectedVoiceApiKey(cfg)
  if (cfg.sttProvider === 'openai' && !apiKey) return { ok: false, error: 'Add a Voice API key in Voice settings to enable API transcription.' }
  const encoded = typeof input?.audioBase64 === 'string' ? input.audioBase64.trim() : ''
  if (!encoded) return { ok: false, error: 'No recorded audio was provided.' }
  if (encoded.length > MAX_STT_BASE64_CHARS || encoded.length % 4 === 1 || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) return { ok: false, error: 'Recorded audio is invalid or too large.' }
  const audio = Buffer.from(encoded, 'base64')
  if (!audio.length || audio.length > MAX_STT_AUDIO_BYTES) return { ok: false, error: 'Recorded audio is invalid or too large.' }
  const mimeType = String(input.mimeType || 'audio/webm').split(';', 1)[0].trim().toLowerCase()
  const extension = AUDIO_TYPES[mimeType]
  if (!extension) return { ok: false, error: 'Recorded audio format is not supported.' }

  const validate = deps.validateVoiceApiBaseUrl ?? assertVoiceApiBaseUrl
  try {
    const baseUrl = await validate(voiceApiBaseUrl(cfg))
    const form = new FormData()
    form.append('file', new Blob([new Uint8Array(audio)], { type: mimeType }), `dictation.${extension}`)
    form.append('model', (cfg.sttModel || '').trim() || OPENAI_STT_DEFAULT_MODEL)
    const res = await cloudVoiceFetch(baseUrl, deps)(`${baseUrl}/audio/transcriptions`, {
      method: 'POST',
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
      body: form,
      redirect: 'error',
      signal: AbortSignal.timeout(SYNTH_TIMEOUT_MS),
    })
    if (!res.ok) {
      await discardResponseBody(res)
      await runtimeAudit(deps, 'voice.transcribe', { provider: cfg.sttProvider, ok: false, status: res.status, bytes: audio.length })
      return { ok: false, error: `Voice API returned HTTP ${res.status}.` }
    }
    const json = await boundedJson(res, 'Voice API transcription')
    const transcript = String((json as { text?: unknown } | null)?.text ?? '').trim().slice(0, MAX_TRANSCRIPT_CHARS)
    if (!transcript) return { ok: false, error: 'Voice API returned an empty transcription.' }
    await runtimeAudit(deps, 'voice.transcribe', { provider: cfg.sttProvider, ok: true, bytes: audio.length, chars: transcript.length })
    return { ok: true, text: transcript }
  } catch (e: unknown) {
    const error = errorText(e, apiKey)
    await runtimeAudit(deps, 'voice.transcribe', { provider: cfg.sttProvider, ok: false, error, bytes: audio.length })
    return { ok: false, error }
  }
}

// ElevenLabs cloud TTS. The protected key lives in store.settings; an empty key disables the
// provider with a clear hint. Audio returns base64 (mp3) for the renderer to play. We never
// pass the key or response body into a log/audit sink in cleartext.
export async function synthesizeElevenLabs(text: string, deps: VoiceRuntimeDeps = {}): Promise<VoiceSynthesisResult> {
  const cfg = voiceConfig()
  const apiKey = (cfg.elevenLabsApiKey || '').trim()
  if (!apiKey) return { ok: false, error: 'Add an ElevenLabs API key in Voice settings to enable ElevenLabs speech.' }
  const body = String(text || '').slice(0, MAX_TTS_CHARS).trim()
  if (!body) return { ok: false, error: 'Nothing to read.' }

  const req = buildElevenLabsRequest(body, cfg, apiKey)
  try {
    const fetchImpl = deps.fetchImpl
      ? runtimeFetch(deps)
      : createConnectorFetch(new URL(req.url).origin, 'ElevenLabs API')
    const res = await fetchImpl(req.url, {
      method: 'POST',
      headers: req.headers,
      body: req.body,
      redirect: 'error',
      signal: AbortSignal.timeout(SYNTH_TIMEOUT_MS),
    })
    if (!res.ok) {
      let detail = ''
      try { detail = redactVoiceSecret((await readBoundedVoiceBody(res, 16 * 1024, 'ElevenLabs error')).toString('utf8'), apiKey).slice(0, 160) } catch { /* ignore */ }
      const hint = res.status === 401 ? ' — the API key was rejected.' : ''
      await runtimeAudit(deps, 'voice.synthesize', { provider: 'elevenlabs', ok: false, status: res.status })
      return { ok: false, error: `ElevenLabs HTTP ${res.status}${hint}${detail ? ` (${detail})` : ''}` }
    }
    const { buffer, mimeType } = await boundedAudio(res, 'ElevenLabs')
    await runtimeAudit(deps, 'voice.synthesize', { provider: 'elevenlabs', ok: true, bytes: buffer.length })
    return { ok: true, audioBase64: buffer.toString('base64'), mimeType }
  } catch (e: unknown) {
    await runtimeAudit(deps, 'voice.synthesize', { provider: 'elevenlabs', ok: false, error: errorText(e, apiKey) })
    return { ok: false, error: errorText(e, apiKey) }
  }
}

// Local expressive engine (Kokoro/Piper-class) over a loopback/LAN HTTP endpoint the user
// runs. On any unreachability we signal `fallback` so the renderer drops back to the Web
// Speech voice rather than failing the read. The SSRF guard vets the configured base URL.
export async function synthesizeLocalTts(text: string, deps: VoiceRuntimeDeps = {}): Promise<VoiceSynthesisResult> {
  const cfg = voiceConfig()
  const baseUrl = (cfg.localTtsBaseUrl || '').trim() || LOCAL_TTS_DEFAULT_BASE_URL
  const body = String(text || '').slice(0, MAX_TTS_CHARS).trim()
  if (!body) return { ok: false, error: 'Nothing to read.' }
  const req = buildLocalTtsRequest(body, cfg)
  try {
    const res = await createLocalProviderFetch(baseUrl, runtimeFetch(deps))(req.url, {
      method: 'POST',
      headers: req.headers,
      body: req.body,
      signal: AbortSignal.timeout(SYNTH_TIMEOUT_MS),
    })
    if (!res.ok) {
      await discardResponseBody(res)
      await runtimeAudit(deps, 'voice.synthesize', { provider: 'local', ok: false, status: res.status })
      // A reachable-but-erroring engine is still a real failure; fall back so audio isn't lost.
      return { ok: false, fallback: true, error: `Local TTS engine HTTP ${res.status}. Falling back to the system voice.` }
    }
    const { buffer, mimeType } = await boundedAudio(res, 'Local TTS engine')
    await runtimeAudit(deps, 'voice.synthesize', { provider: 'local', ok: true, bytes: buffer.length })
    return { ok: true, audioBase64: buffer.toString('base64'), mimeType }
  } catch (e: unknown) {
    await runtimeAudit(deps, 'voice.synthesize', { provider: 'local', ok: false, error: errorText(e) })
    return { ok: false, fallback: true, error: `${errorText(e)} — falling back to the system voice.` }
  }
}

// Probe the local engine's OpenAI-compatible /v1/models to confirm it's running and surface
// installable expressive voices for the picker. Reuses the local-provider SSRF guard.
export async function discoverLocalTts(rawBaseUrl?: string): Promise<LocalTtsDiscovery> {
  const baseUrl = (rawBaseUrl || '').trim() || (store.settings.voice.localTtsBaseUrl || '').trim() || LOCAL_TTS_DEFAULT_BASE_URL
  try {
    const res = await createLocalProviderFetch(baseUrl)(localTtsModelsUrl(baseUrl), { signal: AbortSignal.timeout(DISCOVER_TIMEOUT_MS) })
    if (!res.ok) {
      await discardResponseBody(res)
      return { found: false, baseUrl, voices: [], message: `Engine reachable but returned HTTP ${res.status}.` }
    }
    let json: any = null
    try { json = await boundedJson(res, 'Local TTS discovery') } catch { json = null }
    const data: any[] = json?.data ?? json?.models ?? json?.voices ?? (Array.isArray(json) ? json : [])
    const voices = uniqueSorted(data.map((m) => (typeof m === 'string' ? m : m?.id ?? m?.name ?? m?.voice)))
    return { found: true, baseUrl, voices, message: voices.length ? `Found a local TTS engine with ${voices.length} voice${voices.length === 1 ? '' : 's'}.` : 'Found a local TTS engine.' }
  } catch (e: unknown) {
    return { found: false, baseUrl, voices: [], message: `${errorText(e)} Is your local TTS engine running on ${baseUrl}?` }
  }
}
