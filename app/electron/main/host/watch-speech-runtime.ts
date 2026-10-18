import type { KokoroService } from '../../voice/kokoro-service'
import { createWatchSpeechBroker } from '../../agent/watch-speech'
import { store } from '../state'
import { assertRemoteThreadVisible } from './remote-snapshot'
import { isActive } from './archive'

/** Speech permission exposes audio for public task replies, never the rest of the transcript. */
export function createWatchSpeechRuntime(kokoro: () => KokoroService, canSpeak: (sessionId: string) => boolean) {
  return createWatchSpeechBroker({kokoro: {status: () => kokoro().status(), stream: (input, onChunk, signal) => kokoro().stream(input, onChunk, signal)}, canSpeak, voice: () => ({voice: store.settings.voice.kokoroVoice, rate: store.settings.voice.rate}),
    reply(threadId, itemId) {
      assertRemoteThreadVisible(threadId)
      const thread = store.threads.find((entry) => entry.id === threadId)
      if (!thread || !isActive(thread) || thread.kind !== 'task' || thread.incognito || thread.phoneLocalOnly || thread.googleLocalOnly) throw Object.assign(new Error('This task reply stays on your Mac.'), {code: 404})
      const item = thread.timeline.find((entry) => entry.id === itemId && entry.kind === 'assistant')
      if (!item || typeof item.text !== 'string') throw Object.assign(new Error('This assistant reply is unavailable.'), {code: 404})
      return item.text
    },
  })
}
