import assert from 'node:assert/strict'
import { createWatchSpeechBroker, WATCH_AUDIO_PIECE_BYTES } from './watch-speech'
import type { KokoroStatus } from './types/kokoro'
function wav(pcm: Buffer): Buffer {
  const data = Buffer.alloc(44 + pcm.length); data.write('RIFF'); data.writeUInt32LE(data.length - 8, 4); data.write('WAVEfmt ', 8); data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22); data.writeUInt32LE(24_000, 24); data.writeUInt32LE(48_000, 28); data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34); data.write('data', 36); data.writeUInt32LE(pcm.length, 40); pcm.copy(data, 44); return data
}
const ready = {installed: true, state: 'ready'} as KokoroStatus
let allowed = false, text = 'A short assistant reply.', generated = 0, aborted = false
const pcm = Buffer.alloc(WATCH_AUDIO_PIECE_BYTES * 2 + 42, 13)
const broker = createWatchSpeechBroker({canSpeak: session => allowed && session === 'session-one', reply: (thread, item) => {assert.equal(thread, 'job-one'); assert.equal(item, 'reply-one'); return text}, voice: () => ({voice: 'af_heart', rate: 1}), kokoro: {
  status: async () => ready,
  stream: async (input, chunk, signal) => {assert.equal(input.text, text); signal.addEventListener('abort', () => {aborted = true}); generated++; await chunk(wav(pcm), 0); if (signal.aborted) return {ok: false}; generated++; await chunk(wav(Buffer.alloc(40, 7)), 1); return {ok: true}},
}})
assert.equal((await broker.status('session-one')).available, false)
await assert.rejects(() => broker.start('session-one', {threadId: 'job-one', itemId: 'reply-one'}), /Enable Kokoro/)
allowed = true
const start = await broker.start('session-one', {threadId: 'job-one', itemId: 'reply-one'})
await new Promise(resolve => setTimeout(resolve, 0))
assert.equal(generated, 1)
const first = await broker.chunk('session-one', start.speechId, 0)
assert.equal(first.state, 'audio'); assert.equal(Buffer.from(first.audioBase64!, 'base64').length, WATCH_AUDIO_PIECE_BYTES)
assert.deepEqual(await broker.chunk('session-one', start.speechId, 0), first, 'one previous sequence may retry without consuming more data')
await assert.rejects(() => broker.chunk('session-one', start.speechId, 2), /sequence/)
await assert.rejects(() => broker.chunk('different-session', start.speechId, 1), /Enable/)
const second = await broker.chunk('session-one', start.speechId, 1)
assert.equal(generated, 1, 'the worker must not generate another passage while one remains buffered')
const third = await broker.chunk('session-one', start.speechId, 2)
assert.deepEqual(Buffer.concat([first, second, third].map(piece => Buffer.from(piece.audioBase64!, 'base64'))), pcm)
await new Promise(resolve => setTimeout(resolve, 0)); assert.equal(generated, 2, 'next passage streams only after the consumer drains the first')
const fourth = await broker.chunk('session-one', start.speechId, 3)
assert.equal(Buffer.from(fourth.audioBase64!, 'base64').length, 40)
await new Promise(resolve => setTimeout(resolve, 0)); assert.equal((await broker.chunk('session-one', start.speechId, 4)).state, 'done')
text = 'Changed reply'
await assert.rejects(() => broker.chunk('session-one', start.speechId, 4), /reply changed/)
assert.equal(aborted, true)
broker.stop()
let grants = true, clock = 1
const pending = createWatchSpeechBroker({now: () => clock, canSpeak: () => grants, reply: () => 'Example reply', voice: () => ({}), kokoro: {status: async () => ready, stream: async (_input, _chunk, signal) => new Promise(resolve => signal.addEventListener('abort', () => resolve({ok: false})))}})
const active = await pending.start('one', {threadId: 'job-one', itemId: 'reply-one'})
assert.equal((await pending.chunk('one', active.speechId, 0)).state, 'pending')
await assert.rejects(() => pending.start('one', {threadId: 'job-one', itemId: 'reply-one'}), /busy/)
pending.revoke('one'); grants = false
await assert.rejects(() => pending.chunk('one', active.speechId, 0), /Enable/)
grants = true
const later = await pending.start('one', {threadId: 'job-one', itemId: 'reply-one'}); clock = 200_000
await assert.rejects(() => pending.chunk('one', later.speechId, 0), /expired/)
pending.stop()
console.log('Watch speech broker verified exact-reply audio, bounded PCM, true passage backpressure, duplicate/order/ownership gates, changed-history refusal, revocation and expiry')
const empty = createWatchSpeechBroker({canSpeak: () => true, reply: () => 'Example reply', voice: () => ({}), kokoro: {status: async () => ready, stream: async (_input, chunk) => {try {await chunk(wav(Buffer.alloc(0)), 0); return {ok: true}} catch {return {ok: false, error: 'Invalid empty audio'}}}}})
const noAudio = await empty.start('one', {threadId: 'job-one', itemId: 'reply-one'})
await new Promise(resolve => setTimeout(resolve, 0))
assert.equal((await empty.chunk('one', noAudio.speechId, 0)).state, 'error', 'empty WAV must fail instead of waiting for impossible consumption')
empty.stop()
