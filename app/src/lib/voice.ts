import { api } from './ipc'
import type { VoiceSettings } from './types'

type DictationCallbacks = {
  onText: (text: string) => void
  onError?: (message: string) => void
  onEnd?: () => void
}

// Every networked provider is synthesized in the main process and returned as base64.
// The renderer never receives or submits provider credentials; browser uses Web Speech.
const MAIN_SYNTH_PROVIDERS = new Set(['openai', 'custom', 'elevenlabs', 'local'])

let currentAudio: HTMLAudioElement | null = null
let currentAudioUrl = ''

const FALLBACK_VOICE: VoiceSettings = {
  enabled: true,
  ttsEnabled: true,
  sttEnabled: true,
  ttsProvider: 'browser',
  sttProvider: 'browser',
  voiceName: '',
  rate: 1.6,
  pitch: 1,
  volume: 1,
  autoSummarizeBeforeRead: true,
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

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(n) ? n : min))
}

export function speechSupported(settings?: VoiceSettings): boolean {
  const cfg = settings ?? FALLBACK_VOICE
  const webSpeech = typeof window !== 'undefined' && 'speechSynthesis' in window
  // The local engine can always fall back to Web Speech, so it's "supported" if either the
  // engine path (fetch+Audio) or Web Speech is available.
  if (cfg.ttsProvider === 'local') return webSpeech || (typeof window !== 'undefined' && typeof Audio !== 'undefined')
  if (cfg.ttsProvider !== 'browser') return typeof window !== 'undefined' && typeof Audio !== 'undefined'
  return webSpeech
}

export function dictationSupported(settings?: VoiceSettings): boolean {
  const cfg = settings ?? FALLBACK_VOICE
  if (typeof window === 'undefined') return false
  if (cfg.sttProvider !== 'browser') return !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined'
  return !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)
}

export function availableVoices(): SpeechSynthesisVoice[] {
  if (!speechSupported()) return []
  return window.speechSynthesis.getVoices()
}

export function stopSpeaking(): void {
  if (currentAudio) {
    currentAudio.pause()
    currentAudio = null
  }
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl)
    currentAudioUrl = ''
  }
  if (speechSupported()) window.speechSynthesis.cancel()
}

export function stripSpeechText(input: string): string {
  return input
    .replace(/```[\s\S]*?```/g, ' code block omitted. ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*]\([^)]*\)/g, '')
    .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^[>\-*+]\s+/gm, '')
    .replace(/\|/g, ' ')
    .replace(/[_*~#]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

export function summarizeForSpeech(input: string, maxChars = 900): string {
  const text = stripSpeechText(input)
  if (text.length <= maxChars) return text
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
  const important = sentences.filter((sentence) => /\b(done|built|fixed|changed|blocked|failed|passed|next|todo|risk|error|warning|summary|goal|room|task)\b/i.test(sentence))
  const picked = [...sentences.slice(0, 2), ...important.slice(0, 5)]
  const out: string[] = []
  for (const sentence of picked) {
    if (out.includes(sentence)) continue
    if ((out.join(' ').length + sentence.length) > maxChars) break
    out.push(sentence)
  }
  const summary = out.join(' ').trim() || text.slice(0, maxChars)
  return `Summary. ${summary}`
}

export function speechText(input: string, settings?: VoiceSettings, forceSummary = false): string {
  const cfg = settings ?? FALLBACK_VOICE
  return forceSummary || cfg.autoSummarizeBeforeRead ? summarizeForSpeech(input, cfg.summaryMaxChars) : stripSpeechText(input)
}

function base64ToBlob(base64: string, mimeType: string): Blob {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Blob([bytes], { type: mimeType || 'audio/mpeg' })
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('Could not read recorded audio.'))
    reader.onload = () => {
      const value = String(reader.result ?? '')
      const comma = value.indexOf(',')
      if (comma < 0) reject(new Error('Could not encode recorded audio.'))
      else resolve(value.slice(comma + 1))
    }
    reader.readAsDataURL(blob)
  })
}

function playAudioBlob(blob: Blob, cfg: VoiceSettings, opts: { onEnd?: () => void; onError?: (message: string) => void }): void {
  currentAudioUrl = URL.createObjectURL(blob)
  currentAudio = new Audio(currentAudioUrl)
  currentAudio.volume = clamp(cfg.volume, 0, 1)
  // Cloud/local engines already render at the requested pace; keep playbackRate at 1 so we
  // don't double-apply speed in the renderer.
  currentAudio.onended = () => { stopSpeaking(); opts.onEnd?.() }
  currentAudio.onerror = () => { stopSpeaking(); opts.onError?.('Speech playback failed.') }
  void currentAudio.play().catch((e: any) => { stopSpeaking(); opts.onError?.(e?.message ?? 'Speech playback failed.') })
}

// Web Speech (browser) readback. Also serves as the graceful fallback when the local engine
// is unreachable. Returns false if Web Speech isn't available in this runtime.
function speakWebSpeech(body: string, cfg: VoiceSettings, opts: { onEnd?: () => void; onError?: (message: string) => void }): boolean {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) { opts.onError?.('Text-to-speech is not available in this runtime.'); return false }
  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(body)
  const voice = cfg.voiceName ? availableVoices().find((candidate) => candidate.name === cfg.voiceName) : null
  if (voice) utterance.voice = voice
  utterance.rate = clamp(cfg.rate, 0.5, 3)
  utterance.pitch = clamp(cfg.pitch, 0.5, 2)
  utterance.volume = clamp(cfg.volume, 0, 1)
  utterance.onend = () => opts.onEnd?.()
  utterance.onerror = (event) => opts.onError?.(event.error || 'Speech playback failed.')
  window.speechSynthesis.speak(utterance)
  return true
}

export function speakText(input: string, settings?: VoiceSettings, opts: { summarize?: boolean; onEnd?: () => void; onError?: (message: string) => void } = {}): boolean {
  const cfg = settings ?? FALLBACK_VOICE
  if (!cfg.enabled || !cfg.ttsEnabled) { opts.onError?.('Voice output is disabled.'); return false }
  if (!speechSupported(cfg)) { opts.onError?.('Text-to-speech is not available in this runtime.'); return false }
  const body = speechText(input, cfg, !!opts.summarize)
  if (!body) return false
  // Provider synthesis runs in the main process (keys never leave it; endpoint guards run
  // beside fetch). Audio comes back base64; local errors fall back to Web Speech.
  if (MAIN_SYNTH_PROVIDERS.has(cfg.ttsProvider)) {
    stopSpeaking()
    const run = async () => {
      try {
        const result = cfg.ttsProvider === 'local' ? await api.synthesizeLocalVoice(body) : await api.synthesizeVoice(body)
        if (result.ok && result.audioBase64) {
          playAudioBlob(base64ToBlob(result.audioBase64, result.mimeType || 'audio/mpeg'), cfg, opts)
          return
        }
        if (result.fallback && speakWebSpeech(body, cfg, opts)) return
        opts.onError?.(result.error ?? 'Speech synthesis failed.')
      } catch (e: any) {
        // A hard IPC failure on the local engine should still fall back to Web Speech.
        if (cfg.ttsProvider === 'local' && speakWebSpeech(body, cfg, opts)) return
        opts.onError?.(e?.message ?? 'Speech synthesis failed.')
      }
    }
    void run()
    return true
  }
  return speakWebSpeech(body, cfg, opts)
}

export function startDictation(settings: VoiceSettings | undefined, callbacks: DictationCallbacks): () => void {
  const cfg = settings ?? FALLBACK_VOICE
  if (!cfg.enabled || !cfg.sttEnabled) { callbacks.onError?.('Voice input is disabled.'); callbacks.onEnd?.(); return () => {} }
  if (cfg.sttProvider !== 'browser') {
    if (!dictationSupported(cfg)) { callbacks.onError?.('API speech-to-text is not available in this runtime.'); callbacks.onEnd?.(); return () => {} }
    let stream: MediaStream | null = null
    let recorder: MediaRecorder | null = null
    let stopRequested = false
    const chunks: Blob[] = []
    const cleanup = () => {
      stream?.getTracks().forEach((track) => track.stop())
      stream = null
      recorder = null
    }
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then((nextStream) => {
        stream = nextStream
        if (stopRequested) { cleanup(); callbacks.onEnd?.(); return }
        recorder = new MediaRecorder(nextStream)
        recorder.ondataavailable = (event) => { if (event.data.size > 0) chunks.push(event.data) }
        recorder.onerror = (event: any) => callbacks.onError?.(event?.error?.message || 'Audio recording failed.')
        recorder.onstop = async () => {
          try {
            const mimeType = recorder?.mimeType || 'audio/webm'
            const blob = new Blob(chunks, { type: mimeType })
            const result = await api.transcribeVoice({ audioBase64: await blobToBase64(blob), mimeType })
            if (!result.ok) throw new Error(result.error || 'Speech transcription failed.')
            const text = String(result.text ?? '').trim()
            if (text) callbacks.onText(text)
          } catch (e: any) {
            callbacks.onError?.(e?.message ?? 'Speech transcription failed.')
          } finally {
            cleanup()
            callbacks.onEnd?.()
          }
        }
        recorder.start()
      })
      .catch((e: any) => {
        cleanup()
        callbacks.onError?.(e?.message ?? 'Microphone access failed.')
        callbacks.onEnd?.()
      })
    return () => {
      stopRequested = true
      if (recorder?.state === 'recording') recorder.stop()
      else cleanup()
    }
  }
  const SpeechRecognition = typeof window !== 'undefined' ? ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition) : null
  if (!SpeechRecognition) { callbacks.onError?.('Speech-to-text is not available in this runtime.'); callbacks.onEnd?.(); return () => {} }
  const recognition = new SpeechRecognition()
  recognition.continuous = false
  recognition.interimResults = false
  recognition.lang = navigator.language || 'en-GB'
  recognition.onresult = (event: any) => {
    const text = Array.from(event.results ?? [])
      .map((result: any) => result?.[0]?.transcript ?? '')
      .join(' ')
      .trim()
    if (text) callbacks.onText(text)
  }
  recognition.onerror = (event: any) => callbacks.onError?.(event?.error || 'Speech recognition failed.')
  recognition.onend = () => callbacks.onEnd?.()
  recognition.start()
  return () => recognition.stop()
}
