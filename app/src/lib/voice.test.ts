import assert from 'node:assert/strict'
import type { KokoroSynthesisInput, VoiceSettings, VoiceSynthesisResult } from './types'

const pending = new Map<string, (result: VoiceSynthesisResult) => void>()
const cancelled: string[] = []
const audios: FakeAudio[] = []
let lastRequest = ''
class FakeAudio {
  volume = 1
  onended: (() => void) | null = null
  onerror: (() => void) | null = null
  paused = false
  constructor(_url: string) { audios.push(this) }
  play() { return Promise.resolve() }
  pause() { this.paused = true }
}
Object.defineProperty(globalThis, 'Audio', { configurable: true, value: FakeAudio })
Object.defineProperty(globalThis, 'window', { configurable: true, value: {
  tensorfoldStudio: {
    synthesizeKokoroVoice: (input: KokoroSynthesisInput) => { lastRequest = input.requestId; return new Promise<VoiceSynthesisResult>((resolve) => pending.set(input.requestId, resolve)) },
    cancelKokoroVoice: async (requestId: string) => { cancelled.push(requestId); return { cancelled: true } },
  },
  speechSynthesis: { cancel() {}, getVoices() { return [] } },
  dispatchEvent: () => true,
} })
const { speakText, stopSpeaking } = await import('./voice')
const settings = { enabled: true, ttsEnabled: true, ttsProvider: 'kokoro', kokoroVoice: 'af_heart', rate: 1, volume: 1, autoSummarizeBeforeRead: false } as VoiceSettings
const success = { ok: true, audioBase64: Buffer.from('RIFF local test audio').toString('base64'), mimeType: 'audio/wav' }
const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

assert.equal(speakText('First read', settings), true)
const first = lastRequest
stopSpeaking()
assert.ok(cancelled.includes(first), 'Stop must reach the CPU synthesis runtime')
pending.get(first)!(success)
await flush()
assert.equal(audios.length, 0, 'a late synthesis result must never play after Stop')

speakText('Second read', settings)
const second = lastRequest
speakText('Third read', settings)
const third = lastRequest
pending.get(third)!(success)
await flush()
assert.equal(audios.length, 1)
pending.get(second)!(success)
await flush()
assert.equal(audios.length, 1, 'an older read must never replace the current playback')
const oldEnd = audios[0].onended
speakText('Fourth read', settings)
pending.get(lastRequest)!(success)
await flush()
assert.equal(audios.length, 2)
oldEnd?.()
assert.equal(audios[1].paused, false, 'a stale audio event must not stop a newer read')
stopSpeaking()
assert.ok(audios[0].paused)

let error = ''
speakText('Needs a model', settings, { onError: (message) => { error = message } })
pending.get(lastRequest)!({ ok: false, error: 'Download Kokoro in Voice settings.' })
await flush()
assert.match(error, /Download Kokoro/)
assert.equal(audios.length, 2, 'missing Kokoro must display the real error instead of silently switching providers')
console.log('Voice Stop, late-result, latest-read and missing-model feedback checks passed')
