import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { ModelLibrary } from './library'

const MIB = 1024 ** 2
const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-library-metadata-')))
const caps = { version: '1.1.0', chip: null, backends: ['metal'], families: { qwen3_5: ['mlx-q4g64'], gemma4: ['mlx-q4g64'] }, flags: [], env: [] }
const library = (name: string, metadataBytes?: number) => new ModelLibrary({ appHome: path.join(root, 'profiles', name), env: () => ({ HOME: path.join(root, 'home') }), capabilities: async () => caps, memoryLimitBytes: () => 16 * 1024 ** 3, ...(metadataBytes === undefined ? {} : { limits: { metadataBytes } }) })
async function checkpoint(directory: string, marker: string, headerPadding = 0) {
  await fs.mkdir(directory, { recursive: true })
  await fs.writeFile(path.join(directory, 'config.json'), JSON.stringify({ model_type: 'qwen3_5', quantization: { bits: 4, group_size: 64 }, marker }))
  await fs.writeFile(path.join(directory, 'tokenizer.json'), '{}')
  const header = Buffer.from(JSON.stringify({ tensor: { dtype: 'F32', shape: [1], data_offsets: [0, 4] }, ...(headerPadding ? { __metadata__: { padding: 'x'.repeat(headerPadding) } } : {}) }))
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length))
  await fs.writeFile(path.join(directory, 'model.safetensors'), Buffer.concat([prefix, header, Buffer.alloc(4)]))
}
async function index(directory: string, bytes: number) { await fs.writeFile(path.join(directory, 'model.safetensors.index.json'), JSON.stringify({ weight_map: { tensor: 'model.safetensors' } }).padEnd(bytes, ' ')) }
try {
  const large = path.join(root, 'large-index'); await checkpoint(large, 'large-index'); await index(large, 12 * MIB)
  const largeLibrary = library('large-index'), found = await largeLibrary.addFolder(large)
  assert.equal(found.models[0]?.loadable, true, 'a valid twelve MiB shard index is within the bounded metadata budget')
  assert.equal((await largeLibrary.resolve(found.models[0].id)).path, large, 'load-time reinspection uses the same reviewed metadata limits')
  const header = path.join(root, 'large-header'); await checkpoint(header, 'large-header', 6 * MIB)
  assert.equal((await library('large-header').addFolder(header)).models[0]?.loadable, true, 'a supported safetensors header over four MiB remains readable')
  const oversized = path.join(root, 'oversized'); await checkpoint(oversized, 'oversized')
  await fs.writeFile(path.join(oversized, 'model.safetensors.index.json'), '{}'); await fs.truncate(path.join(oversized, 'model.safetensors.index.json'), 64 * MIB + 1)
  const refused = await library('oversized').addFolder(oversized)
  assert.equal(refused.models[0]?.loadable, false); assert.match(refused.models[0]?.reason ?? '', /metadata.*too large/i, 'size is rejected before allocating or parsing an oversized file')
  const parent = path.join(root, 'external-models'), child = path.join(parent, 'usable'); await checkpoint(child, 'companions')
  for (const directory of [parent, child]) for (const name of ['._model.safetensors', '._model.safetensors.index.json', '._config.json', '._tokenizer.json', '._download.ranges']) await fs.writeFile(path.join(directory, name), 'resource-fork-only')
  const companionLibrary = library('companions'), companions = await companionLibrary.addFolder(parent)
  assert.equal(companions.models.length, 1); assert.equal(companions.models[0].path, child); assert.equal(companions.models[0].loadable, true, 'AppleDouble files neither masquerade as weights nor suppress a real child checkpoint')
  assert.equal((await companionLibrary.resolve(companions.models[0].id)).path, child)
  const reported = path.join(root, 'engine-reported-family'); await checkpoint(reported, 'reported-family')
  await fs.writeFile(path.join(reported, 'config.json'), JSON.stringify({ model_type: 'gemma4', quantization: { bits: 4, group_size: 64 } }))
  assert.equal((await library('reported-family').addFolder(reported)).models[0]?.loadable, true, 'library admission follows engine capabilities rather than a fixed old-family list')
  const aggregate = path.join(root, 'aggregate')
  for (const name of ['one', 'two']) { const directory = path.join(aggregate, name); await checkpoint(directory, name); await index(directory, 8 * MIB) }
  const bounded = await library('aggregate', 14 * MIB).addFolder(aggregate)
  assert.equal(bounded.truncated, true); assert.equal(bounded.models.length, 1, 'aggregate metadata is bounded across individually valid checkpoints')
  assert.ok(bounded.warnings.some(warning => /metadata|limit/i.test(warning)))
  console.log('Model library metadata: valid large indexes and headers, oversized-file refusal, aggregate scan limit, AppleDouble filtering and load-time reinspection passed')
} finally { await fs.rm(root, { recursive: true, force: true }) }
