import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { store } from '../state'
import type { Thread } from '../../agent/types'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-watch-runtime-privacy-'))
const {createWatchSpeechRuntime} = await import('./watch-speech-runtime')
const original = {settings: store.settings, threads: store.threads}
const fixture = (): Thread => ({id: 'visible-task', title: 'Synthetic', kind: 'task', cwd: '', providerId: null, model: null, status: 'idle', createdAt: 1, updatedAt: 1, apiMessages: [], timeline: [{id: 'assistant-reply', kind: 'assistant', text: 'Visible assistant reply'}]})
let workers = 0
const runtime = createWatchSpeechRuntime(() => { workers++; return {status: async () => ({installed: true, state: 'ready'}), stream: async () => ({ok: true})} as any }, () => true)
try {
  store.settings = {voice: {kokoroVoice: 'af_heart', rate: 1}} as any
  for (const patch of [{incognito: true}, {phoneLocalOnly: true}, {archivedAt: 2}, {kind: 'chat' as const}, {timeline: [{id: 'assistant-reply', kind: 'user' as const, text: 'User content'}]}]) {
    store.threads = [{...fixture(), ...patch}]
    await assert.rejects(() => runtime.start('paired-session', {threadId: 'visible-task', itemId: 'assistant-reply'}), /Mac|available|not_found/i)
  }
  store.threads = []
  await assert.rejects(() => runtime.start('paired-session', {threadId: 'missing', itemId: 'assistant-reply'}))
  assert.equal(workers, 0, 'private/archived/chat/non-assistant/missing requests refuse before creating or probing the Kokoro service')
  store.threads = [fixture()]
  const speech = await runtime.start('paired-session', {threadId: 'visible-task', itemId: 'assistant-reply'})
  assert.equal(speech.provider, 'kokoro'); assert.equal(speech.sampleRate, 24_000)
  assert.ok(workers > 0, 'a visible assistant task reply reaches the real runtime adapter')
  store.threads[0].phoneLocalOnly = true
  await assert.rejects(() => runtime.chunk('paired-session', speech.speechId, 0), /no longer available/i)
  console.log('Actual Mac watch speech runtime excludes private/local-only/archived/chat/user/missing content before Kokoro work and refuses a reply made local-only midstream')
} finally { runtime.stop(); Object.assign(store, original) }
