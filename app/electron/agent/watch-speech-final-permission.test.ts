import assert from 'node:assert/strict'
import {createWatchSpeechBroker} from './watch-speech'
import type {KokoroStatus} from './types/kokoro'

for (const change of ['private', 'edited', 'permission', 'expired', 'owner'] as const) {
  let visible = true, allowed = true, text = 'Original reply', clock = 100, aborted = false
  const broker = createWatchSpeechBroker({now: () => clock, canSpeak: session => allowed && session === 'paired-session', reply: () => {if (!visible) throw new Error('Private reply'); return text}, voice: () => ({}), kokoro: {
    status: async () => ({installed: true, state: 'ready'}) as KokoroStatus,
    stream: async (_input, _chunk, signal) => new Promise(resolve => {signal.addEventListener('abort', () => {aborted = true; resolve({ok: false})})}),
  }})
  try {
    const speech = await broker.start('paired-session', {threadId: 'task-test', itemId: 'assistant-test'})
    assert.equal(broker.isAllowed('paired-session'), true)
    assert.equal(broker.isAllowed('paired-session', speech.speechId), true)
    if (change === 'private') visible = false
    if (change === 'edited') text = 'Edited reply'
    if (change === 'permission') allowed = false
    if (change === 'expired') clock = 200_000
    const session = change === 'owner' ? 'other-session' : 'paired-session'
    assert.equal(broker.isAllowed(session, speech.speechId), false, `${change}: final synchronous stream permission refuses stale exact audio`)
    if (change === 'private' || change === 'edited' || change === 'expired') assert.equal(aborted, true, 'Invalid exact reply ends synthesis immediately')
  } finally { broker.stop() }
}
const cancelled = createWatchSpeechBroker({canSpeak: () => true, reply: () => 'Reply', voice: () => ({}), kokoro: {status: async () => ({installed: true, state: 'ready'}) as KokoroStatus, stream: async () => ({ok: true})}})
const speech = await cancelled.start('paired-session', {threadId: 'task-test', itemId: 'assistant-test'})
assert.equal(cancelled.cancel('paired-session', speech.speechId).cancelled, true)
assert.equal(cancelled.isAllowed('paired-session'), true, 'A completed cancel only needs current session permission')
assert.equal(cancelled.isAllowed('paired-session', speech.speechId), false, 'Deleted stream cannot be listened to again')
cancelled.stop()
console.log('Synchronous exact-stream final permission rechecks current visibility/text, permission, expiry and ownership; successful cancel remains confirmable')
