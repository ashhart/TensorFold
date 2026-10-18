import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import type { Worker } from 'node:worker_threads'
import { KokoroService } from './kokoro-service'
const root = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-kokoro-stream-'))
const payload = Buffer.from('fixture model')
class FakeWorker extends EventEmitter {
  messages: any[] = []; terminated = false
  postMessage(input: any) {this.messages.push(input)}
  terminate() {this.terminated = true; return Promise.resolve(0)}
}
let worker!: FakeWorker
const service = new KokoroService({root, workerFile: 'fixture.js', assets: [{path: 'config.json', bytes: payload.length, sha256: createHash('sha256').update(payload).digest('hex'), url: 'https://model.example.test/config.json'}], fetchImpl: async () => new Response(payload), makeWorker: () => {worker = new FakeWorker(); return worker as unknown as Worker}})
try {
  await service.install()
  let release!: () => void, received = 0
  const abort = new AbortController()
  const stream = service.stream({requestId: 'watch-fixture', text: 'A real contract test.'}, async () => {received++; await new Promise<void>(resolve => {release = resolve})}, abort.signal)
  while (!worker) await new Promise(resolve => setTimeout(resolve, 1))
  assert.equal(worker.messages[0].stream, true)
  assert.equal((await service.stream({requestId: 'second-read', text: 'Do not interrupt.'}, async () => {}, new AbortController().signal)).ok, false)
  assert.equal(worker.terminated, false, 'a watch read cannot cancel another active read')
  const wav = Buffer.alloc(48); wav.write('RIFF')
  worker.emit('message', {requestId: 'watch-fixture', chunk: true, index: 0, audioBase64: wav.toString('base64'), ok: true})
  await new Promise(resolve => setTimeout(resolve, 0)); assert.equal(received, 1); assert.equal(worker.messages.length, 1, 'no worker acknowledgement before consumer completion')
  release(); await new Promise(resolve => setTimeout(resolve, 0)); assert.deepEqual(worker.messages[1], {requestId: 'watch-fixture', ack: 0})
  abort.abort(); assert.equal((await stream).ok, false); assert.equal(worker.terminated, true)
} finally {service.dispose(); await rm(root, {recursive: true, force: true})}
console.log('Kokoro stream consumer acknowledgement, cancellation and desktop-playback isolation checks passed')
