import { randomUUID } from 'node:crypto'
import type { VoiceSynthesisResult } from './types/connectors'
import type { KokoroStatus, KokoroSynthesisInput } from './types/kokoro'

export const WATCH_AUDIO_PIECE_BYTES = 24_000
const MAX_WAV_BYTES = 3 * 1024 * 1024
const MAX_SPEECH_MS = 180_000
export interface WatchAudioPiece { speechId: string; sequence: number; state: 'pending' | 'audio' | 'done' | 'error'; audioBase64?: string; sampleRate?: number; error?: string }
export interface WatchSpeechBroker {
  isAllowed(sessionId: string, speechId?: string): boolean
  status(sessionId: string): Promise<{ available: boolean; provider: 'kokoro'; reason: string | null }>
  start(sessionId: string, input: {threadId: string; itemId: string}): Promise<{speechId: string; provider: 'kokoro'; sampleRate: number}>
  chunk(sessionId: string, speechId: string, sequence: number): Promise<WatchAudioPiece>
  cancel(sessionId: string, speechId: string): {cancelled: boolean}
  revoke(sessionId: string): void
  stop(): void
}
interface Speech {
  id: string; sessionId: string; threadId: string; itemId: string; text: string; createdAt: number;
  abort: AbortController; pieces: Buffer[]; next: number; consumed: number; done: boolean; error?: string;
  release?: () => void; previous?: WatchAudioPiece; timer: ReturnType<typeof setTimeout>;
}
function fail(message: string, code = 403): never { throw Object.assign(new Error(message), {code}) }
function pcmFromWav(wav: Buffer): Buffer {
  if (wav.length <= 44 || wav.length > MAX_WAV_BYTES || wav.toString('ascii', 0, 4) !== 'RIFF' || wav.readUInt32LE(4) !== wav.length - 8 || wav.toString('ascii', 8, 16) !== 'WAVEfmt '
    || wav.readUInt32LE(16) !== 16 || wav.readUInt16LE(20) !== 1 || wav.readUInt16LE(22) !== 1
    || wav.readUInt32LE(24) !== 24_000 || wav.readUInt16LE(34) !== 16 || wav.toString('ascii', 36, 40) !== 'data'
    || wav.readUInt32LE(40) !== wav.length - 44 || wav.length % 2 !== 0) throw new Error('Kokoro returned unsupported audio.')
  return wav.subarray(44)
}
export function createWatchSpeechBroker(deps: {
  canSpeak(sessionId: string): boolean; reply(threadId: string, itemId: string): string;
  kokoro: {status(): Promise<KokoroStatus>; stream(input: KokoroSynthesisInput, onChunk: (wav: Buffer, index: number) => Promise<void>, signal: AbortSignal): Promise<VoiceSynthesisResult>};
  voice(): {voice?: string; rate?: number}; now?: () => number;
}): WatchSpeechBroker {
  const sessions = new Map<string, Speech>(), now = deps.now ?? Date.now
  let lifecycle = 0
  function authorized(sessionId: string): void { if (!sessionId || !deps.canSpeak(sessionId)) fail('Enable Kokoro watch replies for this paired device on your Mac.') }
  function end(speech: Speech, error?: string): void {
    speech.abort.abort(); speech.error = error; speech.done = true; speech.pieces = []; speech.release?.(); speech.release = undefined; clearTimeout(speech.timer)
  }
  function lookup(sessionId: string, speechId: string): Speech {
    authorized(sessionId)
    const speech = sessions.get(speechId)
    if (!speech || speech.sessionId !== sessionId) fail('This speech session is unavailable.', 404)
    if (now() - speech.createdAt > MAX_SPEECH_MS) { end(speech, 'Speech expired; request playback again.'); fail('This speech session expired.', 410) }
    let text = ''
    try { text = deps.reply(speech.threadId, speech.itemId) } catch { end(speech, 'This reply is no longer available.'); fail('This reply is no longer available.', 404) }
    if (text !== speech.text) { end(speech, 'The reply changed; request playback again.'); fail('The reply changed; request playback again.', 409) }
    return speech
  }
  return {
    isAllowed(sessionId, speechId) {
      if (!deps.canSpeak(sessionId)) return false
      if (speechId === undefined) return true
      // Called synchronously at the final wire boundary, after prior awaits.
      // lookup rechecks exact current reply visibility/text and stream ownership.
      try { lookup(sessionId, speechId); return true } catch { return false }
    },
    async status(sessionId) {
      if (!deps.canSpeak(sessionId)) return {available: false, provider: 'kokoro', reason: 'Enable Kokoro watch replies for this paired device on your Mac.'}
      const status = await deps.kokoro.status()
      if (!deps.canSpeak(sessionId)) return {available: false, provider: 'kokoro', reason: 'Enable Kokoro watch replies for this paired device on your Mac.'}
      return {available: status.installed && status.state === 'ready', provider: 'kokoro', reason: status.installed && status.state === 'ready' ? null : 'Download or repair Kokoro in TensorFold Settings > Voice on your Mac.'}
    },
    async start(sessionId, input) {
      authorized(sessionId)
      const generation = lifecycle
      if (typeof input?.threadId !== 'string' || !input.threadId || input.threadId.length > 256 || typeof input.itemId !== 'string' || !input.itemId || input.itemId.length > 256) fail('Select an assistant reply to read.', 400)
      // Validate visibility and pin exact text before any synthesis or worker creation.
      const text = deps.reply(input.threadId, input.itemId)
      if (!text?.trim() || text.length > 5_000) fail('This reply is empty or exceeds the 5,000-character watch playback limit.', 400)
      for (const speech of sessions.values()) if (now() - speech.createdAt > MAX_SPEECH_MS || speech.done && !speech.pieces.length) { end(speech); sessions.delete(speech.id) }
      if ([...sessions.values()].some((speech) => !speech.done) || sessions.size >= 8) fail('Kokoro watch playback is busy; stop the current read first.', 409)
      const capability = await this.status(sessionId)
      authorized(sessionId)
      if (generation !== lifecycle) fail('The watch connection changed; request playback again.', 409)
      if (deps.reply(input.threadId, input.itemId) !== text) fail('The reply changed; request playback again.', 409)
      if (!capability.available) fail(capability.reason || 'Kokoro is unavailable.', 409)
      if ([...sessions.values()].some((speech) => !speech.done)) fail('Kokoro watch playback is busy; stop the current read first.', 409)
      const id = randomUUID(), abort = new AbortController()
      const speech: Speech = {id, sessionId, threadId: input.threadId, itemId: input.itemId, text, createdAt: now(), abort, pieces: [], next: 0, consumed: 0, done: false,
        timer: setTimeout(() => { end(speech, 'Speech expired; request playback again.'); sessions.delete(id) }, MAX_SPEECH_MS)}
      speech.timer.unref?.(); sessions.set(id, speech)
      const config = deps.voice()
      void deps.kokoro.stream({requestId: `watch-${id}`, text, voice: config.voice, rate: config.rate}, async (wav) => {
        if (abort.signal.aborted || !deps.canSpeak(sessionId)) throw new Error('Speech cancelled.')
        const pcm = pcmFromWav(wav)
        if (speech.pieces.length) throw new Error('Audio backpressure was not respected.')
        for (let offset = 0; offset < pcm.length; offset += WATCH_AUDIO_PIECE_BYTES) speech.pieces.push(pcm.subarray(offset, offset + WATCH_AUDIO_PIECE_BYTES))
        await new Promise<void>((resolve) => { speech.release = resolve; if (abort.signal.aborted) resolve() })
      }, abort.signal).then((result) => {
        speech.done = true
        if (!result.ok && !abort.signal.aborted) speech.error = result.error || 'Kokoro playback failed.'
      }).catch(() => { if (!abort.signal.aborted) { speech.error = 'Kokoro playback failed.'; speech.done = true } })
      return {speechId: id, provider: 'kokoro', sampleRate: 24_000}
    },
    async chunk(sessionId, speechId, sequence) {
      const speech = lookup(sessionId, speechId)
      if (!Number.isInteger(sequence) || sequence < 0 || sequence > 10_000) fail('Invalid audio sequence.', 400)
      if (speech.previous?.sequence === sequence) return speech.previous
      if (sequence !== speech.next) fail('Refresh playback; its audio sequence changed.', 409)
      if (speech.error) return {speechId, sequence, state: 'error', error: speech.error}
      const bytes = speech.pieces.shift()
      if (!bytes) return {speechId, sequence, state: speech.done ? 'done' : 'pending'}
      speech.next++; speech.consumed += bytes.length
      const result: WatchAudioPiece = {speechId, sequence, state: 'audio', audioBase64: bytes.toString('base64'), sampleRate: 24_000}
      speech.previous = result
      if (!speech.pieces.length) { const release = speech.release; speech.release = undefined; release?.() }
      return result
    },
    cancel(sessionId, speechId) { authorized(sessionId); const speech = sessions.get(speechId); if (!speech || speech.sessionId !== sessionId) return {cancelled: false}; end(speech); sessions.delete(speechId); return {cancelled: true} },
    revoke(sessionId) { lifecycle++; for (const speech of sessions.values()) if (speech.sessionId === sessionId) { end(speech); sessions.delete(speech.id) } },
    stop() { lifecycle++; for (const speech of sessions.values()) end(speech); sessions.clear() },
  }
}
