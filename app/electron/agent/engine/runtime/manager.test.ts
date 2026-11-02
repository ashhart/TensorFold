import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineRuntimeManager } from './manager'
import { fixture, signed, document, publicKey } from './test-fixtures'
const f = await fixture()
try {
  let doc = document(f.pkg), bytes = f.bytes, failLoad = false, transitions = 0, block = false, catalogUnavailable = false
  const make = () => new EngineRuntimeManager({ root: path.join(f.root, 'installed'), bundledVersion: '1.1.6', appVersion: '1.0.0-beta.10',
    platform: 'darwin', arch: 'arm64', publicKey, probe: async () => {},
    transition: async (apply, rollback) => { transitions++; if (block) throw new Error('Active request'); try { await apply(); if (failLoad) throw new Error('Model load failed') } catch (e) { await rollback(); throw e } },
    network: { lookup: async () => [{ address: '93.184.216.34', family: 4 }], fetchImpl: (async (input) => {
      const url = String(input)
      if (catalogUnavailable && url.includes('runtimes.json')) return new Response('Unavailable', { status: 410 })
      return url.includes('runtimes.json') ? new Response(signed(doc), { headers: { 'content-type': 'application/json' } }) : new Response(bytes)
    }) as typeof fetch },
  })
  const manager = make()
  assert.equal((await manager.snapshot()).active, null)
  await manager.check(); await manager.install(f.pkg.id)
  assert.equal(manager.binaryPath(), null, 'download never activates automatically')
  block = true; await assert.rejects(manager.use(f.pkg.id), /Active request/); assert.equal(manager.binaryPath(), null)
  block = false; failLoad = true
  await assert.rejects(manager.use(f.pkg.id), /Model load failed/); assert.equal(manager.binaryPath(), null)
  failLoad = false; await manager.use(f.pkg.id)
  assert.ok(manager.binaryPath()?.endsWith('bin/tensorfold-native'))
  assert.equal((await make().snapshot()).active?.version, '1.1.7', 'selection survives an offline restart')
  catalogUnavailable = true
  await assert.rejects(manager.check(), /HTTP 410/)
  assert.equal((await manager.snapshot()).active?.version, '1.1.7', 'an unavailable catalog preserves the installed running engine')
  assert.match((await manager.snapshot()).error!, /HTTP 410/, 'catalog failure is never reported as up to date')
  catalogUnavailable = false
  await fs.writeFile(path.join(f.root, 'installed', '.selection.json'), JSON.stringify({ active: f.pkg.id, pending: true, previous: null }))
  assert.equal((await make().snapshot()).active, null, 'a crash during activation restores the last committed runtime')
  await manager.use(null); assert.equal(manager.binaryPath(), null)
  const before = transitions
  await fs.writeFile(path.join(f.root, 'installed', f.pkg.id, f.pkg.entrypoint), 'bad')
  await assert.rejects(manager.use(f.pkg.id), /integrity/)
  assert.equal(transitions, before, 'tampered files cannot enter activation')
  const restarted = await make().snapshot()
  assert.equal(restarted.active, null); assert.equal(restarted.installed.length, 0)
  doc = { ...doc, publishedAt: doc.publishedAt - 10 }
  await assert.rejects(manager.check(), /older runtime catalog/)
  doc = document({ ...f.pkg, id: 'bad-hash' }); bytes = Buffer.from('corrupt')
  await manager.check(); await assert.rejects(manager.install('bad-hash'), /SHA-256/)
  assert.equal((await manager.snapshot()).state, 'idle')
  assert.equal((await fs.readdir(path.join(f.root, 'installed', '.downloads'))).length, 0)
} finally { await f.cleanup() }
console.log('Runtime download, activation, busy denial, rollback, offline restore and corruption checks passed.')
