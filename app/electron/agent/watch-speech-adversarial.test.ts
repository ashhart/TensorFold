import assert from 'node:assert/strict'
import { createWatchSpeechBroker } from './watch-speech'
import type { KokoroStatus } from './types/kokoro'

function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((yes) => { resolve = yes }); return {promise, resolve} }
const ready = {installed: true, state: 'ready'} as KokoroStatus
const input = {threadId: 'visible-job', itemId: 'assistant-reply'}
async function settle() { for (let i = 0; i < 15; i++) await Promise.resolve() }

for (const invalidation of ['stop', 'revoke', 'private', 'edited', 'permission'] as const) {
  const status = deferred<KokoroStatus>()
  let allowed = true, visible = true, text = 'Original reply', streams = 0
  const broker = createWatchSpeechBroker({canSpeak: () => allowed, reply: () => { if (!visible) throw new Error('Reply became private'); return text }, voice: () => ({}), kokoro: {
    status: () => status.promise,
    stream: async () => { streams++; return {ok: true} },
  }})
  try {
    const pending = broker.start('paired-session', input)
    const rejected = assert.rejects(() => pending, /cancel|change|unavailable|Enable|private/i)
    await settle()
    if (invalidation === 'stop') broker.stop()
    if (invalidation === 'revoke') broker.revoke('paired-session')
    if (invalidation === 'private') visible = false
    if (invalidation === 'edited') text = 'Revised reply'
    if (invalidation === 'permission') allowed = false
    status.resolve(ready)
    await rejected
    assert.equal(streams, 0, `${invalidation} while status is pending must refuse before creating a speech worker`)
  } finally { broker.stop() }
}

// Status itself must not report an available route after its consent was revoked in flight.
{
  const status = deferred<KokoroStatus>(); let allowed = true
  const broker = createWatchSpeechBroker({canSpeak: () => allowed, reply: () => 'reply', voice: () => ({}), kokoro: {status: () => status.promise, stream: async () => ({ok: true})}})
  const pending = broker.status('paired-session'); allowed = false; status.resolve(ready)
  assert.equal((await pending).available, false)
  broker.stop()
}
console.log('Watch speech deferred capability checks reject stop/revoke/private/edit/permission races before synthesis and recheck status consent')
