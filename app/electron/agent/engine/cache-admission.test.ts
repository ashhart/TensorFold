import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineService } from './service'
import { ModelRevisions } from './model-revisions'
import { makeHarness, seedModel } from './test-fixtures/harness'
import { streamChat } from '../openai'
import { builtinEngineProvider } from './provider'

const h = await makeHarness(), model = 'Fixture/Owned'
await seedModel(h.hub, model)
await new ModelRevisions(h.root, () => h.hub).commit([{ repo: model, revision: 'a'.repeat(40) }])
const service = new EngineService({ binaryPath: '/fixture/engine', binary: h.binary, appHome: h.root, childEnv: h.env, totalMemoryBytes: 32 * 1024 ** 3, pollMs: 15, emit: () => {} })
let release = () => {}, reached = () => {}, held = false
const atInstalled = new Promise<void>((resolve) => { reached = resolve })
const proceed = new Promise<void>((resolve) => { release = resolve })
const installed = service.installed.bind(service)
service.installed = async (refresh = false) => { if (!held) { held = true; reached(); await proceed }; return installed(refresh) }
try {
  service.install()
  const deleting = service.remove(model)
  await atInstalled
  await assert.rejects(service.load(model), /cache cleanup/)
  assert.throws(() => service.tryModel(model), /cache cleanup/)
  assert.throws(() => service.applyModelSettings(model, async () => {}, async () => {}), /cache cleanup/)
  await assert.rejects(async () => {
    for await (const _chunk of streamChat({ provider: builtinEngineProvider(), model, messages: [{ role: 'user', content: 'Synthetic admission check' }] })) {}
  }, /cache cleanup/)
  release(); await deleting
  assert.equal(service.supervisor.status().state, 'stopped')
  assert.equal((await h.records()).filter((record) => record.kind === 'serve').length, 0, 'nothing launches from a cache being removed')
  assert.equal(await fs.lstat(path.join(h.hub, 'models--Fixture--Owned')).then(() => true, () => false), false)
} finally { release(); await service.unload().catch(() => {}); service.shutdown(); await h.cleanup() }
console.log('Cache deletion excludes actual loads, previews, profile edits and chat requests before child launch')
