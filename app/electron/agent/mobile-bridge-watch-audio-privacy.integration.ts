import assert from 'node:assert/strict'
import {createServer as httpServer} from 'node:http'
import {request as httpsRequest} from 'node:https'
import {createServer as portProbe} from 'node:net'
import {mkdtemp, rm} from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import {createMobileBridge, mobileBridgeInterfaces} from './mobile-bridge'
import {loadMobileBridgeCertificate} from './mobile-bridge-certificate'
import {createWatchSpeechRuntime} from '../main/host/watch-speech-runtime'
import {store} from '../main/state'
import type {Thread} from './types'

const address = mobileBridgeInterfaces()[0]?.address
assert.ok(address, 'Audio privacy TLS regression requires a private IPv4 interface')
const profile = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-watch-audio-privacy-'))
const original = {...store}, token = 'fixture_' + 'a'.repeat(40), session = 'paired-session'
const caps = {taskStart: true, taskGuide: true, taskStop: true, approvalResolve: false, transcriptReview: false, changesReview: false, engineControl: false}
let allowed = true, mode = 'permission'
function wav() {
  const pcm = Buffer.alloc(100, 13), data = Buffer.alloc(44 + pcm.length)
  data.write('RIFF'); data.writeUInt32LE(data.length - 8, 4); data.write('WAVEfmt ', 8); data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22); data.writeUInt32LE(24_000, 24); data.writeUInt32LE(48_000, 28); data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34); data.write('data', 36); data.writeUInt32LE(pcm.length, 40); pcm.copy(data, 44); return data
}
const speech = createWatchSpeechRuntime(() => ({status: async () => ({installed: true, state: 'ready'}), stream: async (_input: unknown, chunk: (value: Buffer, index: number) => Promise<void>) => {await chunk(wav(), 0); return {ok: true}}}) as any, () => allowed)
const broker = {...speech, async chunk(id: string, stream: string, sequence: number) {
  const result = await speech.chunk(id, stream, sequence)
  assert.equal(result.state, 'audio', 'Fixture reaches actual runtime audio lookup before the late change')
  if (mode === 'permission') allowed = false
  if (mode === 'private') store.threads[0].incognito = true
  if (mode === 'phone') store.threads[0].phoneLocalOnly = true
  if (mode === 'edited') store.threads[0].timeline[0].text = 'Changed assistant reply'
  return result
}}
const core = httpServer((req, res) => {
  assert.equal(req.headers.authorization, `Bearer ${token}`)
  const value = req.url === '/v1/device-status' ? {scope: 'paired', capabilities: caps, engine: null}
    : {serverName: 'Synthetic host', activeThreadIds: [], threads: structuredClone(store.threads), work: {jobs: {running: 0, queued: 0}, approvals: {pending: 0}}}
  res.writeHead(200, {'content-type': 'application/json'}); res.end(JSON.stringify(value))
})
await new Promise<void>(resolve => core.listen(0, '127.0.0.1', resolve))
const corePort = (core.address() as {port: number}).port
const port = await new Promise<number>(resolve => {const probe = portProbe(); probe.listen(0, address, () => {const value = (probe.address() as {port: number}).port; probe.close(() => resolve(value))})})
const bridge = createMobileBridge({requireOwner() {}, loopbackUrl: () => `http://127.0.0.1:${corePort}`, ownerToken: () => 'fixture-owner-token', authorizeSessionToken: value => ({ok: value === token, sessionId: session, scope: 'paired'}), issuePairingLink: async () => ({credential: 'unused', expiresAt: 0}), loadCertificate: host => loadMobileBridgeCertificate(profile, host), watchSpeech: broker})
try {
  store.settings = {voice: {kokoroVoice: 'af_heart', rate: 1}} as any
  await bridge.configure({version: 1, enabled: true, host: address, port})
  const cert = await loadMobileBridgeCertificate(profile, address)
  async function command(action: string, extras: object = {}) {
    const bytes = Buffer.from(JSON.stringify({version: 1, id: `request-${mode}-${action}`, session, action, threadId: 'shared-task', itemId: 'assistant-test', generatedAt: Date.now(), ...extras}))
    return await new Promise<{status: number; value: any}>((resolve, reject) => {
      const req = httpsRequest({hostname: address, port, path: '/v1/watch/command', method: 'POST', ca: cert.cert, agent: false, headers: {authorization: `Bearer ${token}`, 'content-type': 'application/json', 'content-length': bytes.length}}, res => {
        const chunks: Buffer[] = []; res.on('data', part => chunks.push(part)); res.on('end', () => resolve({status: res.statusCode!, value: JSON.parse(Buffer.concat(chunks).toString())})); res.on('error', reject)
      }); req.on('error', reject); req.setTimeout(5_000, () => req.destroy(new Error('Synthetic audio deadline'))); req.end(bytes)
    })
  }
  for (mode of ['permission', 'private', 'phone', 'edited']) {
    speech.stop(); allowed = true
    store.threads = [{id: 'shared-task', title: 'Synthetic task', kind: 'task', cwd: '', providerId: 'example', model: 'example', createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [], timeline: [{id: 'assistant-test', kind: 'assistant', text: 'Original assistant reply'}]} as Thread]
    const start = await command('audio-start')
    assert.equal(start.status, 200); assert.equal(start.value.ok, true)
    const result = await command('audio-next', {speechId: start.value.speech.speechId, sequence: 0})
    assert.equal(result.status, 403, `${mode}: a credential can remain valid while speech becomes forbidden after lookup`)
    assert.equal(result.value.audio, undefined); assert.equal(result.value.error, 'watch_speech_permission_required')
  }
  speech.stop(); allowed = true; mode = 'cancel'
  store.threads = [{id: 'shared-task', title: 'Synthetic task', kind: 'task', cwd: '', providerId: 'example', model: 'example', createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [], timeline: [{id: 'assistant-test', kind: 'assistant', text: 'Original assistant reply'}]} as Thread]
  const cancelStart = await command('audio-start')
  const stopped = await command('audio-stop', {speechId: cancelStart.value.speech.speechId})
  assert.equal(stopped.status, 200); assert.equal(stopped.value.ok, true, 'Successful cancellation must not require the now-deleted speech ID')
  console.log('Actual TLS watch command audio uses production speech runtime; late speech opt-out, incognito, phone-only and edited reply all refuse before PCM serialization while bearer remains valid')
} finally {
  await bridge.stop(); speech.stop(); await new Promise<void>(resolve => core.close(() => resolve())); await rm(profile, {recursive: true, force: true}); Object.assign(store, original)
}
