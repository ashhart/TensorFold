// Explicit private-LAN integration, never auto-run by the normal test suite.
import assert from 'node:assert/strict'
import { request as httpsRequest } from 'node:https'
import { createServer as portProbe } from 'node:net'
import { checkServerIdentity } from 'node:tls'
import { createHash } from 'node:crypto'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createMobileBridge, mobileBridgeInterfaces } from './mobile-bridge'
import { loadMobileBridgeCertificate } from './mobile-bridge-certificate'
import { createPairingRegistry } from './pairing-sessions'
import { createWatchSpeechBroker, WATCH_AUDIO_PIECE_BYTES } from './watch-speech'
import { KokoroService } from '../voice/kokoro-service'
import { StudioClient } from '../../mobile/src/lib/client'
const assetRoot = process.env.TENSORFOLD_WATCH_QA_ASSETS, workerFile = process.env.TENSORFOLD_WATCH_QA_WORKER, evidence = process.env.TENSORFOLD_WATCH_QA_EVIDENCE
if (!assetRoot || !workerFile || !evidence || ![assetRoot, workerFile, evidence].every(path.isAbsolute)) throw new Error('Supply absolute isolated watch QA assets, worker and evidence paths.')
const address = mobileBridgeInterfaces()[0]?.address
if (!address) throw new Error('An active private IPv4 interface is required; no network path was tested.')
const profile = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-watch-tls-'))
const registry = createPairingRegistry(), ownerToken = 'owner_' + 'f'.repeat(40)
const device = registry.redeemPairingLink(registry.issuePairingLink('Example watch relay').credential).session!
let networkSynthesis = 0, workerStarted = 0
const service = new KokoroService({root: assetRoot, workerFile, fetchImpl: async () => {networkSynthesis++; throw new Error('Speech network is forbidden')}})
const text = 'Your example task is ready to review. The worker checked the sample files and prepared a short summary. You can send another instruction when your phone connection is available. This additional passage verifies that Kokoro sends audio from your Mac before the complete reply has finished generating.'
const speech = createWatchSpeechBroker({canSpeak: id => registry.isWatchSpeechAllowed(id), reply: (threadId, itemId) => {
  if (threadId !== 'example-job' || itemId !== 'example-assistant') throw Object.assign(new Error('Unavailable reply'), {code: 404}); return text
}, voice: () => ({voice: 'af_heart', rate: 1}), kokoro: {status: () => service.status(), stream: (input, callback, signal) => {workerStarted++; return service.stream(input, callback, signal)}}})
const port = await new Promise<number>((resolve, reject) => {const probe = portProbe(); probe.on('error', reject); probe.listen(0, address, () => {const port = (probe.address() as {port: number}).port; probe.close(() => resolve(port))})})
const cert = await loadMobileBridgeCertificate(profile, address)
const bridge = createMobileBridge({interfaces: () => [{name: 'qa-interface', address}], requireOwner: () => {}, loopbackUrl: () => 'http://127.0.0.1:8765', ownerToken: () => ownerToken,
  authorizeSessionToken: token => registry.authorizeSessionToken(token), issuePairingLink: async label => registry.issuePairingLink(label), loadCertificate: async () => cert, watchSpeech: speech})
function pinnedFetch(pin = cert.sha256): typeof fetch {
  return (async (input, init) => {
    const url = new URL(String(input)), body = typeof init?.body === 'string' ? Buffer.from(init.body) : undefined
    return new Promise<Response>((resolve, reject) => {
      const headers: Record<string, string> = {}; new Headers(init?.headers).forEach((value, key) => {headers[key] = value})
      const request = httpsRequest(url, {method: init?.method || 'GET', headers, ca: cert.cert, agent: false, signal: init?.signal ?? undefined,
        checkServerIdentity(host, peer) {const error = checkServerIdentity(host, peer); if (error) return error; if (createHash('sha256').update(peer.raw).digest('hex') !== pin) return new Error('certificate_pin_mismatch')},
      }, response => {const pieces: Buffer[] = []; let bytes = 0; response.on('data', piece => {bytes += piece.length; if (bytes > 64 * 1024) response.destroy(new Error('Oversize speech response')); else pieces.push(piece)}); response.on('error', reject); response.on('end', () => resolve(new Response(Buffer.concat(pieces), {status: response.statusCode, headers: {'content-type': String(response.headers['content-type'] || '')}})))})
      request.setTimeout(5_000, () => request.destroy(new Error('Private TLS deadline exceeded'))); request.on('error', reject); request.end(body)
    })
  }) as typeof fetch
}
const client = new StudioClient({baseUrl: `https://${address}:${port}`, token: device.token, sessionId: device.id, label: 'Example watch relay'}, pinnedFetch())
const begun = Date.now(); let firstAudioMs = 0, pieces = 0, bytes = 0, finished = false
try {
  await bridge.configure({version: 1, enabled: true, host: address, port})
  assert.equal((await client.watchSpeechStatus()).available, false)
  await assert.rejects(() => client.startWatchSpeech('example-job', 'example-assistant'))
  assert.equal(workerStarted, 0)
  registry.setWatchSpeechAllowed(device.id, true)
  assert.equal((await client.watchSpeechStatus()).available, true)
  await assert.rejects(() => client.startWatchSpeech('example-job', 'missing'))
  assert.equal(workerStarted, 0)
  const wrongPin = new StudioClient(client.session, pinnedFetch('0'.repeat(64)))
  await assert.rejects(() => wrongPin.watchSpeechStatus())
  const start = await client.startWatchSpeech('example-job', 'example-assistant')
  for (let sequence = 0; Date.now() - begun < 100_000;) {
    const piece = await client.readWatchSpeech(start.speechId, sequence)
    if (piece.state === 'pending') {await new Promise(resolve => setTimeout(resolve, 100)); continue}
    assert.notEqual(piece.state, 'error', piece.error)
    if (piece.state === 'done') {finished = true; break}
    assert.equal(piece.state, 'audio'); assert.equal(piece.sequence, sequence); assert.equal(piece.sampleRate, 24_000)
    const pcm = Buffer.from(piece.audioBase64!, 'base64'); assert.ok(pcm.length > 0 && pcm.length <= WATCH_AUDIO_PIECE_BYTES); assert.equal(pcm.length % 2, 0)
    if (!pieces) {firstAudioMs = Date.now() - begun; assert.deepEqual(await client.readWatchSpeech(start.speechId, sequence), piece)}
    pieces++; bytes += pcm.length; sequence++
  }
  assert.equal(finished, true); assert.ok(pieces > 1); assert.ok(bytes > 48_000)
  const stop = await client.startWatchSpeech('example-job', 'example-assistant')
  assert.equal((await client.cancelWatchSpeech(stop.speechId)).cancelled, true)
  await assert.rejects(() => client.readWatchSpeech(stop.speechId, 0))
  const revoked = await client.startWatchSpeech('example-job', 'example-assistant')
  registry.setWatchSpeechAllowed(device.id, false); speech.revoke(device.id)
  await assert.rejects(() => client.readWatchSpeech(revoked.speechId, 0))
  registry.revokeSession(device.id); await assert.rejects(() => client.watchSpeechStatus())
  assert.equal(networkSynthesis, 0)
  await writeFile(path.join(evidence, 'tls-kokoro-receipt.json'), JSON.stringify({passed: true, actualPinnedPrivateHttps: true, actualKokoroWorker: true, permissionDeniedByDefault: true, exactAssistantItemRequired: true, wrongCertificatePinRejected: true, firstAudioMs, pcmPieces: pieces, pcmBytes: bytes, sampleRate: 24_000, streamCompleted: finished, duplicateChunkSafe: true, cancellationVerified: true, audioPermissionRevocationVerified: true, deviceRevocationVerified: true, installerNetworkRequests: networkSynthesis, physicalWatchOrPhoneTested: false}, null, 2))
  console.log(JSON.stringify({passed: true, pcmPieces: pieces, pcmBytes: bytes, firstAudioMs, realPinnedHttps: true, actualKokoroWorker: true, physicalWatchOrPhoneTested: false}))
} finally {await bridge.stop(); speech.stop(); service.dispose(); await rm(profile, {recursive: true, force: true})}
