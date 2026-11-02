import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { NodeEngineAdapter } from './native-adapter'
import type { NodeAdmissionInput } from './types'
import { model } from './testing/fixture'

async function main() {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-native-node-adapter-'))), cacheDir = path.join(root, 'cache')
  const calls: string[][] = [], launches: string[][] = []
  let type = 'qwen3_5', memory = { total: 16 * 1024 ** 3, free: 12 * 1024 ** 3 }, stopped = 0, declaredDistributed: any = undefined
  const blockSize = 4096, stagingMinimumBlocks = 2 * 1024 ** 3 / blockSize
  let availableBlocks = stagingMinimumBlocks, statfsError: Error | null = null
  const config = { schema: 1 as const, dataDir: root, host: '127.0.0.1', port: 0, enginePath: '/synthetic/approved/tensorfold-native', cacheDir }
  const put = async (revision: string, modelType: string) => {
    const directory = path.join(cacheDir, 'models--Example--Small', 'snapshots', revision); await fs.mkdir(directory, { recursive: true })
    await fs.writeFile(path.join(directory, 'config.json'), JSON.stringify({ model_type: modelType, quantization: { bits: 4, group_size: 64 }, max_position_embeddings: 8192 }))
    await fs.writeFile(path.join(directory, 'tokenizer.json'), '{}'); await fs.writeFile(path.join(directory, 'model.safetensors'), Buffer.alloc(128))
  }
  const adapter = new NodeEngineAdapter({ config, passiveSysfsRoot: false, platform: process.platform as 'darwin' | 'linux', arch: process.arch as 'arm64' | 'x64', memory: () => memory,
    statfs: async directory => {
      assert.equal(directory, cacheDir)
      if (statfsError) throw statfsError
      return { bsize: blockSize, bavail: availableBlocks }
    },
    runCli: async (args, signal) => {
      if (signal.aborted) throw new Error('Synthetic cancellation'); calls.push(args)
      if (args[0] === 'capabilities') return { code: 0, stdout: JSON.stringify({ schema: 1, engine: 'zig', version: '1.0.12', backends: [process.platform === 'darwin' ? 'metal' : 'cuda'], families: { qwen3_5: ['mlx-q4g64'], unsupported_type: ['mlx-q4g64'], glm_moe_dsa: ['exlr8'] }, serve: { flags: Object.fromEntries(['--speed-up', '--host', '--port', '--name', '--api-key-file', '--context', '--max-tokens', '--temperature', '--parallel', '--prompt-cache-gib', '--no-thinking', '--no-update-check', '--dashboard'].map(flag => [flag, {}])) }, ...(declaredDistributed ? { distributed: declaredDistributed } : {}) }), stderr: '' }
      if (args[0] === 'pull') { const revision = args[1].split('@')[1]; await put(revision, type); return { code: 0, stdout: 'Synthetic immutable staging finished', stderr: '' } }
      if (args[0] === 'info') return { code: 0, stdout: `family: Synthetic family (${type})\nformat: 4-bit, group 64\ncontext: 8192\nweights: 0.00 GiB (128 bytes)\nmemory floor: 2.00 GiB (weights plus a 2.00 GiB runtime margin)\n`, stderr: '' }
      throw new Error('No other CPU/native command is authorized by this test')
    },
    sessionFactory: (_binary, args, env, alias, _key, keyFile) => {
      launches.push(args); assert.equal(env.TF_TP_LOCAL, undefined); assert.equal(env.TF_FLASHNEXT_TP, undefined); assert.equal(env.GLM_EXPERIMENTAL, undefined); assert.equal(env.TENSORFOLD_CORTHEON_UNREVIEWED, undefined); assert.equal(env.HF_HUB_CACHE, cacheDir)
      let live = true
      return { model: alias, alive: () => live, ready: async () => {}, request: async () => { throw new Error('No real or fake model inference is requested in the adapter test') }, stop: async () => { stopped++; live = false; await fs.unlink(keyFile).catch(() => {}) } }
    },
  })
  const input: NodeAdmissionInput = { expectedNodeId: 'synthetic-node', role: 'headless', model, profileId: 'single-native', generation: { contextTokens: 4096, maxOutputTokens: 64, temperature: 0 }, transport: 'tcp', rank: 0, worldSize: 1, links: [] }
  const previousLocal = process.env.TF_TP_LOCAL, previousPair = process.env.TF_FLASHNEXT_TP, previousGlm = process.env.GLM_EXPERIMENTAL, previousCortheon = process.env.TENSORFOLD_CORTHEON_UNREVIEWED
  try {
    const inspection = await adapter.inspect(new AbortController().signal)
    assert.equal(inspection.available, true); assert.deepEqual(inspection.profiles[0].modelTypes, ['qwen3_5']); assert.deepEqual(inspection.profiles.map(profile => profile.mode), ['single'])
    assert.equal(inspection.links[0].verified, false)
    assert.equal((await adapter.admission(input, new AbortController().signal)).cached, false)
    const unexpectedStagingProgress = async () => { assert.fail('Refused staging cannot announce a download') }
    availableBlocks = stagingMinimumBlocks - 1
    await assert.rejects(adapter.ensureModel(model, new AbortController().signal, unexpectedStagingProgress), /at least two GiB free/)
    assert.equal(calls.filter(args => args[0] === 'pull').length, 0)
    availableBlocks = stagingMinimumBlocks
    statfsError = new Error('Synthetic filesystem capacity failure')
    await assert.rejects(adapter.ensureModel(model, new AbortController().signal, unexpectedStagingProgress), error => error === statfsError)
    assert.equal(calls.filter(args => args[0] === 'pull').length, 0)
    statfsError = null
    await adapter.ensureModel(model, new AbortController().signal, async () => {})
    assert.deepEqual(calls.find(args => args[0] === 'pull'), ['pull', `${model.repo}@${model.revision}`])
    assert.equal((await adapter.admission(input, new AbortController().signal)).allowed, true)
    memory = { total: 1024 ** 3, free: 1024 ** 3 }
    assert.equal((await adapter.admission(input, new AbortController().signal)).allowed, false)
    memory = { total: 16 * 1024 ** 3, free: 12 * 1024 ** 3 }
    assert.equal((await adapter.admission({ ...input, generation: { ...input.generation, contextTokens: 16384 } }, new AbortController().signal)).allowed, false)
    assert.equal((await adapter.admission({ ...input, profileId: 'generic-cluster', worldSize: 2, transport: 'thunderbolt-rdma' }, new AbortController().signal)).allowed, false)
    process.env.TF_TP_LOCAL = '1'; process.env.TF_FLASHNEXT_TP = '/synthetic/unreviewed-pair.json'; process.env.GLM_EXPERIMENTAL = '1'; process.env.TENSORFOLD_CORTHEON_UNREVIEWED = '1'
    const handle = await adapter.start({ ...input, deploymentId: 'synthetic-deployment', commandId: 'synthetic-start-command' }, new AbortController().signal, async () => {})
    assert.equal(launches.length, 1); assert.ok(!launches[0].includes('--cluster') && !launches[0].includes('--speed-up'))
    assert.ok(launches[0].includes('--api-key-file') && launches[0].includes('--dashboard')); assert.ok(!launches[0].includes('--cortheon'))
    assert.equal(launches[0][1], path.join(cacheDir, 'models--Example--Small', 'snapshots', model.revision)); await handle.stop(); assert.equal(stopped, 1)
    await assert.rejects(adapter.start({ ...input, deploymentId: 'cancel-before-spawn', commandId: 'synthetic-failed-start' }, new AbortController().signal, async () => { throw new Error('Synthetic durable progress failure') }), /progress failure/)
    assert.equal(launches.length, 1); assert.ok(!(await fs.readdir(root)).some(name => name.startsWith('.engine-')), 'Pre-spawn failures remove newly created native credentials')
    type = 'unsupported_type'
    await assert.rejects(adapter.ensureModel({ ...model, revision: 'b'.repeat(40) }, new AbortController().signal, async () => {}), /not reported/)
    await put('c'.repeat(40), 'qwen3_5'); await fs.unlink(path.join(cacheDir, 'models--Example--Small', 'snapshots', 'c'.repeat(40), 'model.safetensors'))
    assert.equal((await adapter.admission({ ...input, model: { ...model, revision: 'c'.repeat(40) } }, new AbortController().signal)).allowed, false)
    await put('d'.repeat(40), 'qwen3_5'); const escaped = path.join(cacheDir, 'models--Example--Small', 'snapshots', 'd'.repeat(40), 'tokenizer.json'); await fs.unlink(escaped); await fs.writeFile(path.join(root, 'outside.json'), '{}'); await fs.symlink(path.join(root, 'outside.json'), escaped)
    assert.equal((await adapter.admission({ ...input, model: { ...model, revision: 'd'.repeat(40) } }, new AbortController().signal)).allowed, false)
    assert.ok(calls.every(args => !args.includes('--cluster') && args[0] !== 'serve'), 'The fake CLI fixture observes only CPU inspection and immutable staging, never actual native execution')
    console.log('PASS native adapter with fake CPU port: capabilities, known family gate, pinned cache path, immutable pull, incomplete/escaped file refusal, memory/context refusal, explicit distributed block, guarded arguments, environment cleanup and pre-spawn credential cleanup')
  } finally { if (previousLocal === undefined) delete process.env.TF_TP_LOCAL; else process.env.TF_TP_LOCAL = previousLocal; if (previousPair === undefined) delete process.env.TF_FLASHNEXT_TP; else process.env.TF_FLASHNEXT_TP = previousPair; if (previousGlm === undefined) delete process.env.GLM_EXPERIMENTAL; else process.env.GLM_EXPERIMENTAL = previousGlm; if (previousCortheon === undefined) delete process.env.TENSORFOLD_CORTHEON_UNREVIEWED; else process.env.TENSORFOLD_CORTHEON_UNREVIEWED = previousCortheon; await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
