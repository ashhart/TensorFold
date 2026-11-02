import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { EventEmitter, once } from 'node:events'
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import type { Worker } from 'node:worker_threads'
import { KokoroService } from './kokoro-service'
import { KOKORO_ASSETS, KOKORO_REVISION, type KokoroAsset } from './kokoro-catalog'

const temp = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-kokoro-test-'))
const payloads = [Buffer.from('{"model_type":"style_text_to_speech_2"}'), Buffer.from('a small test model')]
const assets: KokoroAsset[] = payloads.map((bytes, i) => ({ path: i ? 'onnx/model_quantized.onnx' : 'config.json', bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'), url: `https://speech.example.test/${i}` }))
let requests: string[] = []
const fetchFixture: typeof fetch = async (url, options) => {
  assert.equal(options?.credentials, 'omit')
  requests.push(String(url))
  const payload = payloads[Number(String(url).split('/').pop())]
  return new Response(new Uint8Array(payload), { headers: { 'content-length': String(payload.length) } })
}

const workerEvents = new EventEmitter()
const nextWorkerInput = () => once(workerEvents, 'posted', { signal: AbortSignal.timeout(5_000) })
class FakeWorker extends EventEmitter {
  terminated = false
  input: unknown
  postMessage(input: unknown) { this.input = input; workerEvents.emit('posted', this) }
  terminate() { this.terminated = true; return Promise.resolve(0) }
}

try {
  for (const asset of KOKORO_ASSETS) {
    assert.match(asset.sha256, /^[a-f0-9]{64}$/)
    assert.ok(asset.url.includes(KOKORO_REVISION))
    assert.ok(asset.url.startsWith('https://huggingface.co/'))
  }
  const root = path.join(temp, 'managed')
  let spawned = 0
  let fake: FakeWorker
  const service = new KokoroService({ root, workerFile: 'test-worker.js', assets, fetchImpl: fetchFixture, makeWorker: () => { spawned++; fake = new FakeWorker(); return fake as unknown as Worker } })
  assert.equal((await service.status()).state, 'missing')
  assert.equal((await service.synthesize({ text: 'Hello', requestId: 'before-download' })).ok, false)
  assert.equal(spawned, 0, 'missing models must not spawn a runtime')
  assert.equal((await service.install()).state, 'ready')
  assert.equal(requests.length, 2)
  assert.deepEqual(await readFile(path.join(root, KOKORO_REVISION, assets[1].path)), payloads[1])

  const offline = new KokoroService({ root, workerFile: 'test-worker.js', assets, fetchImpl: async () => { throw new Error('Network is offline') }, makeWorker: () => { fake = new FakeWorker(); return fake as unknown as Worker } })
  assert.equal((await offline.status()).state, 'ready', 'a fresh process must recognize verified files offline')
  assert.equal((await offline.install()).state, 'ready', 'a ready offline installation must not fetch anything')
  const initialInput = nextWorkerInput()
  const synthesis = offline.synthesize({ text: 'Private local speech', requestId: 'read-one', voice: 'af_bella', rate: 100 })
  await initialInput
  assert.equal((fake!.input as { rate: number }).rate, 2.5)
  assert.equal(offline.cancelSpeech('another-read'), false)
  assert.equal(offline.cancelSpeech('read-one'), true)
  assert.equal((await synthesis).ok, false)
  assert.ok(fake!.terminated, 'Stop must terminate the inference worker')

  const firstInput = nextWorkerInput()
  const firstRead = offline.synthesize({ text: 'First request', requestId: 'reused-id' })
  const [oldWorker] = await firstInput
  const secondInput = nextWorkerInput()
  const secondRead = offline.synthesize({ text: 'Second request', requestId: 'reused-id' })
  const [newWorker] = await secondInput
  assert.equal((await firstRead).ok, false)
  oldWorker.emit('exit', 1)
  newWorker.emit('message', { requestId: 'reused-id', ok: true, audioBase64: 'UklGRg==', mimeType: 'audio/wav' })
  const reused = await Promise.race([secondRead, new Promise<{ ok: boolean }>((resolve) => setTimeout(() => resolve({ ok: false }), 40))])
  assert.equal(reused.ok, true, 'an old worker exit must not swallow a newer request when an ID is reused')

  offline.cancelSpeech('cancel-before-start')
  const early = await offline.synthesize({ text: 'Never start this request', requestId: 'cancel-before-start' })
  assert.match(early.error || '', /cancelled/)
  assert.equal((await offline.synthesize({ text: 'a'.repeat(5001), requestId: 'oversized' })).ok, false)
  assert.equal((await offline.synthesize({ text: 'Hello', requestId: 'bad/id' })).ok, false)
  assert.equal((await offline.synthesize({ text: 'Hello', requestId: 'bad-voice', voice: '../../other' })).ok, false)

  await writeFile(path.join(root, KOKORO_REVISION, assets[1].path), Buffer.alloc(payloads[1].length))
  assert.equal((await service.status()).state, 'error', 'same-size corruption must invalidate readiness')
  assert.equal((await service.synthesize({ text: 'Hello', requestId: 'corrupt' })).ok, false)
  requests = []
  assert.equal((await service.install()).state, 'ready')
  assert.deepEqual(requests, [assets[1].url], 'retry should retain verified complete assets')

  const corrupt = new KokoroService({ root: path.join(temp, 'bad-hash'), workerFile: '', assets, fetchImpl: async () => new Response('x'.repeat(payloads[0].length)) })
  assert.equal((await corrupt.install()).state, 'error')
  assert.match((await corrupt.status()).error || '', /integrity/)
  await assert.rejects(readFile(path.join(temp, 'bad-hash', KOKORO_REVISION, 'config.json.part')))

  const linked = path.join(temp, 'symlink-root')
  await mkdir(path.join(temp, 'outside'))
  await symlink(path.join(temp, 'outside'), linked)
  const unsafe = new KokoroService({ root: linked, workerFile: '', assets, fetchImpl: fetchFixture })
  assert.equal((await unsafe.install()).state, 'error', 'installer must reject symbolic-link destinations')
  assert.match((await unsafe.status()).error || '', /symbolic links/)

  let seenSignal: AbortSignal | undefined
  const interrupted = new KokoroService({ root: path.join(temp, 'interrupted'), workerFile: '', assets, fetchImpl: async (_url, options) => {
    seenSignal = options?.signal as AbortSignal
    return new Response(new ReadableStream({ start(controller) {
      controller.enqueue(new Uint8Array(payloads[0].subarray(0, 4)))
      seenSignal?.addEventListener('abort', () => controller.error(new Error('aborted')), { once: true })
    } }))
  } })
  const download = interrupted.install()
  while (!seenSignal) await new Promise((resolve) => setTimeout(resolve, 1))
  assert.equal((await interrupted.cancelInstall()).state, 'missing')
  assert.equal((await download).installed, false)
  await assert.rejects(readFile(path.join(temp, 'interrupted', KOKORO_REVISION, 'config.json.part')))
  await writeFile(path.join(temp, 'interrupted', KOKORO_REVISION, 'config.json.part'), 'bytes left by an interrupted process')
  const resumed = new KokoroService({ root: path.join(temp, 'interrupted'), workerFile: '', assets, fetchImpl: fetchFixture })
  assert.equal((await resumed.install()).state, 'ready', 'a fresh process must discard incomplete bytes and install verified assets')
  resumed.dispose()

  assert.equal((await service.remove()).state, 'missing')
  assert.equal((await new KokoroService({ root, workerFile: '', assets }).status()).installed, false)
  service.dispose(); offline.dispose(); corrupt.dispose(); unsafe.dispose(); interrupted.dispose()
  console.log('Kokoro installer integrity, retry, cancellation, offline reload, corruption and worker lifecycle checks passed')
} finally { await rm(temp, { recursive: true, force: true }) }
