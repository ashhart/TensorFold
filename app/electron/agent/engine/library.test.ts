import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import type { EngineCapabilities } from '../types'
import { ModelLibrary, isLocalModelId } from './library'

const caps: EngineCapabilities = { version: '1.0.9', chip: null, backends: ['metal'], families: { qwen3_5: ['mlx-q4g64'] }, flags: [], env: [] }
const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tf-library-test-')))
const home = path.join(root, 'home')
const hub = path.join(home, '.cache', 'huggingface', 'hub')
const appHome = path.join(root, 'app')
const env = { HOME: home, HF_HOME: path.join(root, 'alternate-hf'), HF_HUB_CACHE: path.join(root, 'override-hub') }
await fs.mkdir(hub, { recursive: true })

function weights() {
  const header = Buffer.from(JSON.stringify({ weight: { dtype: 'F32', shape: [2], data_offsets: [0, 8] } }))
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length))
  return Buffer.concat([prefix, header, Buffer.alloc(8, 1)])
}
async function checkpoint(directory: string, modelType = 'qwen3_5') {
  await fs.mkdir(directory, { recursive: true })
  await fs.writeFile(path.join(directory, 'config.json'), JSON.stringify({ model_type: modelType, quantization: { bits: 4, group_size: 64 }, max_position_embeddings: 32768 }))
  await fs.writeFile(path.join(directory, 'tokenizer.json'), JSON.stringify({ model: { type: 'BPE', vocab: {} } }))
  await fs.writeFile(path.join(directory, 'model.safetensors'), weights())
}
async function cached(cache: string, repo: string, sha: string) {
  const repoRoot = path.join(cache, `models--${repo.replace('/', '--')}`)
  const snapshot = path.join(repoRoot, 'snapshots', sha)
  const blobs = path.join(repoRoot, 'blobs')
  await checkpoint(blobs)
  await fs.mkdir(snapshot, { recursive: true })
  for (const name of ['config.json', 'model.safetensors', 'tokenizer.json']) await fs.symlink(path.join(blobs, name), path.join(snapshot, name))
  return snapshot
}
const library = (extra: Partial<ConstructorParameters<typeof ModelLibrary>[0]> = {}) => new ModelLibrary({ appHome, env, capabilities: async () => caps, memoryLimitBytes: () => 16 * 1024 ** 3, ...extra })
const hash = async (file: string) => createHash('sha256').update(await fs.readFile(file)).digest('hex')

try {
  assert.equal(isLocalModelId(`local/${'a'.repeat(64)}`), true)
  for (const value of [`local/${'a'.repeat(63)}`, 'local/../../outside', '/tmp/checkpoint', null]) assert.equal(isLocalModelId(value), false)
  const snapshot = await cached(hub, 'Example/First', 'a'.repeat(40))
  const duplicate = path.join(path.dirname(snapshot), 'b'.repeat(40))
  await fs.mkdir(duplicate)
  for (const name of ['config.json', 'model.safetensors', 'tokenizer.json']) await fs.symlink(path.join(path.dirname(path.dirname(snapshot)), 'blobs', name), path.join(duplicate, name))
  const alternate = await cached(path.join(env.HF_HOME, 'hub'), 'Example/Second', 'c'.repeat(40))
  await cached(env.HF_HUB_CACHE, 'Example/Third', 'd'.repeat(40))
  const original = path.join(snapshot, 'model.safetensors')
  const before = { hash: await hash(original), stat: await fs.stat(original) }
  const lib = library()
  assert.deepEqual((await lib.snapshot()).models, [], 'no filesystem discovery before opt-in')
  assert.deepEqual((await lib.scan()).models, [], 'an ordinary scan does not enable HF discovery')
  const found = await lib.scan({ includeHuggingFace: true })
  assert.equal(found.models.length, 3, 'standard cache, HF_HOME and HF_HUB_CACHE all searched, duplicate revision deduplicated')
  assert.equal(found.scanning, false)
  assert.equal(found.models.every((model) => model.loadable && model.complete && model.compatible && model.source === 'huggingface'), true)
  assert.equal(found.models[0].format, 'mlx-q4g64')
  assert.equal(found.models[0].weightsBytes, weights().length)
  const first = found.models.find((model) => model.repo === 'Example/First')!
  assert.equal((await lib.resolve(first.id)).path, snapshot)
  assert.equal(await hash(original), before.hash)
  const after = await fs.stat(original)
  assert.deepEqual([after.ino, after.size, after.mtimeMs], [before.stat.ino, before.stat.size, before.stat.mtimeMs], 'discovery and resolution never modify source checkpoint files')
  assert.equal((await library().snapshot()).models.length, 3, 'library survives app restart')
  for (const [index, selectedPath] of [snapshot, path.dirname(snapshot), path.dirname(path.dirname(snapshot))].entries()) {
    const scoped = library({ appHome: path.join(root, `scoped-${index}`) })
    const directlySelected = await scoped.addFolder(selectedPath)
    assert.equal(directlySelected.includeHuggingFace, false)
    assert.equal(directlySelected.models.length, 1, 'native selection of a snapshot, snapshots parent, or repo root recognizes normal HF links')
    assert.equal(directlySelected.models[0].repo, 'Example/First')
    assert.equal((await scoped.resolve(directlySelected.models[0].id)).path, snapshot)
    assert.equal((await scoped.scan({ includeHuggingFace: false })).models.length, 1, 'explicit HF folder permission survives disabling automatic cache discovery')
    await scoped.removeFolder(directlySelected.folders[0].id)
    assert.ok(await fs.stat(original))
  }

  const modelsDir = path.join(root, 'My Models')
  const direct = path.join(modelsDir, 'My Chat Model')
  await checkpoint(direct)
  const foreign = path.join(modelsDir, 'unsupported')
  await checkpoint(foreign, 'unknown_family')
  const incomplete = path.join(modelsDir, 'incomplete')
  await checkpoint(incomplete)
  await fs.writeFile(path.join(incomplete, 'model.safetensors.index.json'), JSON.stringify({ weight_map: { a: 'model-00001-of-00002.safetensors', b: 'model-00002-of-00002.safetensors' } }))
  const invalid = path.join(modelsDir, 'truncated')
  await checkpoint(invalid)
  await fs.truncate(path.join(invalid, 'model.safetensors'), 12)
  const gguf = path.join(modelsDir, 'gguf')
  await fs.mkdir(gguf); await fs.writeFile(path.join(gguf, 'model.gguf'), 'GGUF')
  const outside = path.join(root, 'outside')
  await checkpoint(outside)
  await fs.symlink(outside, path.join(modelsDir, 'linked-folder'))
  const escape = path.join(modelsDir, 'escaped-file')
  await checkpoint(escape)
  await fs.unlink(path.join(escape, 'model.safetensors')); await fs.symlink(path.join(outside, 'model.safetensors'), path.join(escape, 'model.safetensors'))
  const selected = await lib.addFolder(modelsDir)
  assert.equal(selected.folders.length, 1)
  assert.equal(selected.models.filter((model) => model.path === outside || model.path.endsWith('linked-folder')).length, 0, 'directory symlinks are not traversed')
  assert.equal(selected.models.find((model) => model.path === direct)?.loadable, true)
  assert.match(selected.models.find((model) => model.path === foreign)!.reason!, /does not support unknown_family/)
  assert.equal(selected.models.find((model) => model.path === foreign)?.complete, true)
  assert.match(selected.models.find((model) => model.path === incomplete)!.reason!, /missing.*shards/)
  assert.match(selected.models.find((model) => model.path === invalid)!.reason!, /safetensors/)
  assert.equal(selected.models.find((model) => model.path === gguf)?.format, 'GGUF')
  assert.match(selected.models.find((model) => model.path === escape)!.reason!, /outside/)
  assert.equal((await lib.addFolder(modelsDir)).folders.length, 1, 'duplicate selected roots are ignored')
  await assert.rejects(() => lib.addFolder(home), /home folder/)
  await assert.rejects(() => lib.addFolder('/'), /entire disk/)
  await assert.rejects(() => lib.resolve(`local/${'f'.repeat(64)}`), /no longer in the library/)
  const id = selected.models.find((model) => model.path === direct)!.id
  await fs.rename(direct, `${direct}-moved`)
  await assert.rejects(() => lib.resolve(id), /ENOENT/)
  await fs.symlink(outside, direct)
  await assert.rejects(() => lib.resolve(id), /redirected/)
  await fs.unlink(direct); await fs.rename(`${direct}-moved`, direct)
  const unlinked = await lib.removeFolder(selected.folders[0].id)
  assert.equal(unlinked.models.some((model) => model.id === id), false)
  await fs.stat(path.join(direct, 'model.safetensors'))
  await assert.rejects(() => lib.resolve(id), /no longer in the library/)
  await assert.rejects(() => lib.removeFolder('../outside'), /saved model folder/)
  assert.equal((await lib.scan({ includeHuggingFace: false })).models.length, 0, 'turning opt-in off drops default cache authorizations')
  await fs.stat(alternate)

  // A snapshot may link only to its own blobs, never another repo or an arbitrary local file.
  await fs.unlink(original); await fs.symlink(path.join(outside, 'model.safetensors'), original)
  const escapedHF = await lib.scan({ includeHuggingFace: true })
  assert.equal(escapedHF.models.find((model) => model.repo === 'Example/First')?.loadable, false)
  assert.match(escapedHF.models.find((model) => model.repo === 'Example/First')!.reason!, /leaves its own blob cache/)

  // Abort retains the previous inventory instead of publishing a partial scan as complete.
  const slow = library({ capabilities: async () => { await new Promise((resolve) => setTimeout(resolve, 30)); return caps } })
  const previous = await slow.snapshot()
  const scanning = slow.scan()
  assert.equal((await slow.snapshot()).scanning, true)
  const cancelled = await slow.cancel()
  await scanning
  assert.equal(cancelled.cancelled, true)
  assert.equal(cancelled.scanning, false)
  assert.deepEqual(cancelled.models, previous.models)
  assert.equal(cancelled.scannedAt, previous.scannedAt)

  // A wide selected folder stops at a visible bound, and never traverses the remaining tree.
  const wide = path.join(root, 'wide')
  await fs.mkdir(wide)
  for (let index = 0; index < 150; index++) await fs.writeFile(path.join(wide, `file-${index}`), '')
  const bounded = library({ appHome: path.join(root, 'bounded-app'), limits: { entries: 100 } })
  const start = Date.now()
  const boundedResult = await bounded.addFolder(wide)
  assert.equal(boundedResult.truncated, true)
  assert.match(boundedResult.warnings.join(' '), /safety limit/)
  assert.ok(Date.now() - start < 2000)

  const lowMemory = library({ memoryLimitBytes: () => 1024 })
  const large = await lowMemory.scan()
  assert.equal(large.models.every((model) => !model.loadable), true)


  // Large checkpoint metadata: a shard index above the old 4 MiB cap now scans, resolves, and stays bounded.
  const bigModels = path.join(root, 'Big Metadata')
  const bigValid = path.join(bigModels, 'big-index')
  await fs.mkdir(bigValid, { recursive: true })
  await fs.writeFile(path.join(bigValid, 'config.json'), JSON.stringify({ model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 } }))
  await fs.writeFile(path.join(bigValid, 'tokenizer.json'), JSON.stringify({ model: { type: 'BPE', vocab: {} } }))
  const bigHeader = Buffer.from(JSON.stringify({ weight: { dtype: 'F32', shape: [2], data_offsets: [0, 8] } }))
  const bigPrefix = Buffer.alloc(8); bigPrefix.writeBigUInt64LE(BigInt(bigHeader.length))
  const bigWeight = Buffer.concat([bigPrefix, bigHeader, Buffer.alloc(8, 1)])
  await fs.writeFile(path.join(bigValid, 'model-00001-of-00001.safetensors'), bigWeight)
  const pad = 5 * 1024 * 1024
  const bigIndex = `{"metadata":{"total_size":${bigWeight.length},"pad":"${'x'.repeat(pad)}"},"weight_map":{"a":"model-00001-of-00001.safetensors","b":"model-00001-of-00001.safetensors"}}`
  await fs.writeFile(path.join(bigValid, 'model.safetensors.index.json'), bigIndex)
  const bigLib = library({ appHome: path.join(root, 'big-metadata-app') })
  const bigFound = await bigLib.addFolder(bigModels)
  const bigModel = bigFound.models.find((model) => model.path === bigValid)!
  assert.equal(bigModel.complete, true, bigModel.reason ?? '')
  assert.equal(bigModel.compatible, true)
  assert.equal(bigModel.loadable, true)
  assert.equal(bigModel.weightsBytes, bigWeight.length)
  assert.equal(bigModel.memoryFloorBytes, bigWeight.length + 2 * 1024 ** 3)
  assert.equal((await bigLib.resolve(bigModel.id)).path, bigValid, 'a checkpoint whose shard index exceeds the old cap resolves')

  // An oversized tokenizer.json alone was never blocked: it is checked through safeFile(), not the JSON cap.
  const bigTokenizer = path.join(bigModels, 'big-tokenizer')
  await checkpoint(bigTokenizer)
  await fs.truncate(path.join(bigTokenizer, 'tokenizer.json'), 5 * 1024 * 1024)
  const tokenizerFound = await library({ appHome: path.join(root, 'big-tokenizer-app') }).addFolder(bigModels)
  assert.equal(tokenizerFound.models.find((model) => model.path === bigTokenizer)?.loadable, true, 'an oversized tokenizer alone needs no limit change')

  // A file above the new per-file cap is still rejected; sparse files keep the cost to the stat only.
  const overLimit = path.join(bigModels, 'over-limit')
  await checkpoint(overLimit)
  const oversized = path.join(overLimit, 'model.safetensors.index.json')
  await fs.writeFile(oversized, '')
  await fs.truncate(oversized, 65 * 1024 * 1024)
  const overFound = await library({ appHome: path.join(root, 'over-limit-app') }).addFolder(bigModels)
  assert.match(overFound.models.find((model) => model.path === overLimit)!.reason!, /metadata is too large/)

  // Malformed JSON in an index is still rejected.
  const corrupt = path.join(bigModels, 'corrupt-index')
  await checkpoint(corrupt)
  await fs.writeFile(path.join(corrupt, 'model.safetensors.index.json'), '{"weight_map":')
  const corruptFound = await library({ appHome: path.join(root, 'corrupt-index-app') }).addFolder(bigModels)
  assert.match(corruptFound.models.find((model) => model.path === corrupt)!.reason!, /invalid JSON/)

  // The scan-wide metadata budget stays bounded and reports a visible truncation.
  const budgeted = library({ appHome: path.join(root, 'budget-app'), limits: { metadataBytes: 3 * 1024 * 1024 } })
  const budgetFound = await budgeted.addFolder(bigModels)
  assert.equal(budgetFound.truncated, true)
  assert.match(budgetFound.warnings.join(' '), /safety limit/)
  // The budget is the whole scan's: two checkpoints whose indexes each fit it stop the scan together.
  const scanWide = path.join(root, 'Scan Budget')
  const scanPad = `{"metadata":{"pad":"${'x'.repeat(4 * 1024 * 1024)}"},"weight_map":{"w":"model.safetensors"}}`
  for (const name of ['first', 'second']) {
    await checkpoint(path.join(scanWide, name))
    await fs.writeFile(path.join(scanWide, name, 'model.safetensors.index.json'), scanPad)
  }
  const scanWideFound = await library({ appHome: path.join(root, 'scan-budget-app'), limits: { metadataBytes: 6 * 1024 * 1024 } }).addFolder(scanWide)
  assert.equal(scanWideFound.truncated, true, 'two 4 MiB indexes pass a 6 MiB budget together')
  assert.match(scanWideFound.warnings.join(' '), /safety limit/)
  assert.equal(scanWideFound.models.length, 1, 'the first checkpoint fits the budget alone')

  const validRegistry = await fs.readFile(lib.file, 'utf8')
  await fs.writeFile(lib.file, '{')
  await assert.rejects(() => library().snapshot(), /library is invalid/)
  await assert.rejects(() => library().scan(), /library is invalid/)
  assert.equal(await fs.readFile(lib.file, 'utf8'), '{', 'invalid registry evidence is retained')
  await fs.writeFile(lib.file, validRegistry)
  console.log('local model library filesystem tests passed')
} finally { await fs.rm(root, { recursive: true, force: true }) }
