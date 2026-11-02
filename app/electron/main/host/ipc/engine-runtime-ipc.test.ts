import assert from 'node:assert/strict'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { runtimeId } from '../../../agent/engine/runtime/catalog'
const handlers = new Map<string, (...args: any[]) => any>()
const calls: string[] = []
let role = 'owner', idle = true, leases = 0
const deps = { reserve: () => { leases++; return { release: () => { leases-- } } }, assertIdle: () => { if (!idle) throw new Error('Active local conversation') } }
const runtime = { snapshot: async () => 'snapshot', check: async () => 'catalog', install: async (id: string) => { calls.push(id); return 'installed' },
  use: async (id: string | null) => { calls.push(String(id)); return 'switched' }, cancel: () => true }
const imports: Record<string, unknown> = {
  '../../../agent/engine/runtime/catalog': { runtimeId },
  '../audit': { requireSecurityRole: (required: string) => { if (required === 'owner' && role !== 'owner') throw new Error('Owner required') } },
  '../ipc': { handle: (name: string, fn: (...args: any[]) => any) => handlers.set(name, fn) },
}
const source = readFileSync(path.join(__dirname, 'engine-runtime-ipc.ts'), 'utf8')
const output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText
const module = { exports: {} as any }
new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports, id); return imports[id] }, module, module.exports)
module.exports.registerEngineRuntimeIpc(() => runtime, deps)
const invoke = (method: string, value?: unknown) => handlers.get(`engine:runtime${method}`)!({}, value)
assert.equal(await invoke('Status'), 'snapshot')
role = 'operator'
await assert.rejects(invoke('Check'), /Owner/)
await assert.rejects(invoke('Install', 'valid-runtime'), /Owner/)
await assert.rejects(invoke('Use', null), /Owner/)
role = 'owner'
for (const input of ['../escape', '/tmp/runtime', { id: 'runtime' }, undefined, 1]) await assert.rejects(invoke('Install', input), /verified catalog/)
idle = false; await assert.rejects(invoke('Use', 'valid-runtime'), /Active local/)
assert.deepEqual(calls, []); assert.equal(leases, 0)
idle = true; await invoke('Install', 'valid-runtime'); await invoke('Use', 'valid-runtime'); await invoke('Use', null)
assert.deepEqual(calls, ['valid-runtime', 'valid-runtime', 'null']); assert.equal(leases, 0)
assert.equal(await invoke('Cancel'), true)
console.log('Runtime IPC owner authorization, ID-only input, idle guard and operation lease cleanup passed.')
