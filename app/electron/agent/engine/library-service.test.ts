import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineService } from './service'
import { makeHarness, seedModel } from './test-fixtures/harness'

const h = await makeHarness()
const source = await fs.realpath(h.root)
const folder = path.join(source, 'External Models')
const model = path.join(folder, 'My local model')
await fs.mkdir(model, { recursive: true })
const header = Buffer.from(JSON.stringify({ weight: { dtype: 'F32', shape: [2], data_offsets: [0, 8] } }))
const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length))
const file = path.join(model, 'model.safetensors')
await fs.writeFile(file, Buffer.concat([prefix, header, Buffer.alloc(8, 1)]))
await fs.writeFile(path.join(model, 'config.json'), JSON.stringify({ model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 } }))
await fs.writeFile(path.join(model, 'tokenizer.json'), '{}')
const hash = async () => createHash('sha256').update(await fs.readFile(file)).digest('hex')
const before = { hash: await hash(), stat: await fs.stat(file) }
let networkRequests = 0
const fresh = () => new EngineService({ binaryPath: '/fake', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 20, fetch: (async () => { networkRequests++; throw new Error('Local discovery must never request the network.') }) as typeof fetch, emit: () => {} })
const service = fresh()
let restarted: EngineService | null = null
try {
  await seedModel(h.hub, 'Example/ExternalCache')
  assert.equal((await service.librarySnapshot()).includeHuggingFace, false)
  const discovered = await service.addLibraryFolder(folder)
  const local = discovered.models.find((entry) => entry.path === model)!
  assert.equal(local.loadable, true, local.reason ?? '')
  const installed = await service.installed()
  assert.equal(installed.find((entry) => entry.repo === local.id)?.source, 'library')
  assert.equal(installed.find((entry) => entry.repo === local.id)?.title, 'My local model')
  assert.equal(installed.find((entry) => entry.repo === 'Example/ExternalCache')?.source, 'cache')
  await assert.rejects(() => service.remove('Example/ExternalCache'), /shared cache/)
  assert.equal((await service.readiness(local.id)).ready, true)
  assert.equal((await service.load(local.id)).state, 'ready')
  const serve = (await h.records()).filter((entry) => entry.kind === 'serve').at(-1)
  assert.equal(serve.args[0], model, 'the native engine receives the authorized existing folder as one argv value')
  assert.equal(serve.args[serve.args.indexOf('--name') + 1], local.id, 'API identity matches the chat model ID, not its private path')
  assert.deepEqual(serve.secretEnv, [])
  const connection = await service.connection()
  assert.equal(connection?.model, local.id)
  const response = await fetch(`${connection!.baseUrl}/models`, { headers: { Authorization: `Bearer ${connection!.apiKey}` } })
  assert.equal((await response.json()).data[0].id, local.id)
  const lease = await service.supervisor.acquire(local.id, { start: false })
  await assert.rejects(() => service.unload(), /active TensorFold requests/)
  await assert.rejects(() => service.remove(local.id), /read-only/)
  await assert.rejects(() => service.preflight(local.id), /load in place/)
  await assert.rejects(() => service.pull(local.id), /load in place/)
  const unlinked = await service.removeLibraryFolder(discovered.folders[0].id)
  assert.deepEqual(unlinked.models, [])
  assert.equal((await service.connection())?.apiKey, connection?.apiKey, 'unlinking a library folder preserves the active API and requests')
  assert.equal((await service.readiness(local.id)).ready, false)
  await assert.rejects(() => service.load(local.id), /no longer in the library/)
  lease.release()
  await service.unload()
  await assert.rejects(() => service.supervisor.acquire(local.id, { start: true }), /no longer in the library/)
  assert.equal(await hash(), before.hash)
  const after = await fs.stat(file)
  assert.deepEqual([after.ino, after.size, after.mtimeMs], [before.stat.ino, before.stat.size, before.stat.mtimeMs])
  assert.equal(networkRequests, 0)

  await service.addLibraryFolder(folder)
  restarted = fresh()
  assert.equal((await restarted.installed()).find((entry) => entry.repo === local.id)?.source, 'library', 'persisted folders and stable IDs restore after restart')
  await restarted.load(local.id)
  await restarted.unload()
  await fs.rename(model, `${model}-moved`)
  assert.equal((await restarted.readiness(local.id)).ready, false, 'stale installed caches do not pass readiness after a folder moves')
  await assert.rejects(() => restarted!.load(local.id), /ENOENT/)
  const pairRoot = path.join(source, 'Paired cache')
  const seedPair = async (repo: string, draft = false) => {
    const checkpoint = path.join(pairRoot, `models--${repo.replace('/', '--')}`, 'snapshots', 'a'.repeat(40))
    await fs.mkdir(checkpoint, { recursive: true })
    await fs.writeFile(path.join(checkpoint, 'config.json'), JSON.stringify(draft ? { model_type: 'qwen3', dflash_config: {} } : { model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 } }))
    await fs.writeFile(path.join(checkpoint, 'model.safetensors'), Buffer.concat([prefix, header, Buffer.alloc(8, 1)]))
    if (!draft) await fs.writeFile(path.join(checkpoint, 'tokenizer.json'), '{}')
    return checkpoint
  }
  const primaryRepo = 'TensorFold/Qwen3.8-27B-MLX-4bit'
  const draftRepo = 'z-lab/Qwen3.8-27B-DFlash2'
  const primaryPath = await seedPair(primaryRepo)
  const missingPair = await restarted.addLibraryFolder(pairRoot)
  const primaryId = missingPair.models.find((entry) => entry.repo === primaryRepo)!.id
  assert.equal(missingPair.models.find((entry) => entry.id === primaryId)?.loadable, false)
  assert.match(missingPair.models.find((entry) => entry.id === primaryId)!.reason!, /needs its drafter/)
  assert.equal((await restarted.readiness(primaryId)).ready, false)
  await assert.rejects(() => restarted!.load(primaryId), /needs its drafter/)
  const draftPath = await seedPair(draftRepo, true)
  const fullPair = await restarted.scanLibrary()
  assert.equal(fullPair.models.find((entry) => entry.id === primaryId)?.loadable, true)
  assert.equal(fullPair.models.find((entry) => entry.repo === draftRepo)?.loadable, false, 'a companion is never offered as a standalone chat model')
  await restarted.load(primaryId)
  const pairedServe = (await h.records()).filter((entry) => entry.kind === 'serve').at(-1)
  assert.equal(pairedServe.args[0], primaryPath)
  assert.equal(pairedServe.args[pairedServe.args.indexOf('--drafter') + 1], draftPath, 'an authorized local companion loads in place too')
  await restarted.unload()
  assert.equal(networkRequests, 0)
  console.log('local model library service and API lifecycle tests passed')
} finally { restarted?.shutdown(); service.shutdown(); await h.cleanup() }
