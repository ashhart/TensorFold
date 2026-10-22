import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import { createRequire } from 'node:module'
import ts from 'typescript'
import * as processTools from './mcp-process'
import * as catalog from './mcp-catalog'
import * as envelope from './mcp-envelope'
import * as lifecycle from './child-process-lifecycle'
import * as detection from './secret-detection'
import * as security from './security'

const source = await fs.readFile(new URL('./mcp-client.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const nodeRequire = createRequire(import.meta.url)
const base = processTools.createMcpServer({ id: 'synthetic-mcp', name: 'Synthetic MCP', command: '/bin/echo', args: ['safe'] })

function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>((done) => { resolve = done })
  return { promise, resolve }
}

function fixture(stage: 'manifest' | 'provenance') {
  const entered = deferred()
  const release = deferred()
  let launches = 0
  const pause = async () => { entered.resolve(); await release.promise }
  const dependencies: Record<string, unknown> = {
    './mcp-process': {
      ...processTools,
      verifyConfiguredManifest: async () => { if (stage === 'manifest') await pause(); return { ok: false, digest: null, error: null } },
      commandProvenance: async () => { if (stage === 'provenance') await pause(); return null },
      mcpSpawnSpec: () => { launches++; throw new Error('Synthetic process launch boundary reached') },
    },
    './mcp-catalog': catalog,
    './mcp-envelope': envelope,
    './child-process-lifecycle': lifecycle,
    './secret-detection': detection,
    './security': security,
  }
  const exports: Record<string, any> = {}
  new Function('require', 'exports', compiled)((name: string) => {
    if (name.startsWith('node:')) return nodeRequire(name)
    assert.ok(name in dependencies, `Unbound actual MCP dependency ${name}`)
    return dependencies[name]
  }, exports)
  const host = new exports.McpHost()
  host.setConfigs([base])
  return { host, entered: entered.promise, release: release.resolve, launches: () => launches }
}

let failed = 0
async function test(name: string, run: () => Promise<void>) {
  try { await run(); console.log(`PASS ${name}`) }
  catch (error) { failed++; console.error(`FAIL ${name}`, error) }
}

for (const stage of ['manifest', 'provenance'] as const) await test(`changed configuration cannot cross process launch boundary after ${stage} await`, async () => {
  const f = fixture(stage)
  const result = f.host.connect(base.id).catch((error: unknown) => error)
  await f.entered
  f.host.setConfigs([{ ...base, args: ['changed'] }])
  f.release()
  const error = await result
  assert.equal(f.launches(), 0, 'stale configuration must be rejected before process creation')
  assert.match(error.message, /changed|no longer matches|cancelled/i)
  await f.host.shutdown()
})

await test('disconnect cancels a launch still collecting process provenance', async () => {
  const f = fixture('provenance')
  const result = f.host.connect(base.id).catch((error: unknown) => error)
  await f.entered
  await f.host.disconnect(base.id)
  f.release()
  const error = await result
  assert.equal(f.launches(), 0, 'a stopped connection must not spawn later')
  assert.match(error.message, /changed|no longer matches|cancelled/i)
  await f.host.shutdown()
})

if (failed) process.exitCode = 1
