import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'
import { isRepoId } from '../../../agent/engine/paths'
import { modelBenchmarkPreload } from '../../../preload/model-benchmark'
import { MODEL_BENCHMARK_API_SCHEMA } from '../../../../shared/model-benchmark-api'
import { createMockModelBenchmarkApi } from '../../../../src/lib/mock/model-benchmark'

const handlers = new Map<string, (...args: any[]) => Promise<any>>()
const roles: string[] = []
let denied = false; let busy = false; let available = true; let leaseCount = 0; let auditFailure = false
let current: any = null; let finish!: () => void
const drains = new Map<string, Promise<void>>()
const cancelled: string[] = []; const exports: any[] = []
const instance = {
  setBenchmarkAdmission: (_check: () => void) => {},
  startBenchmark: (input: any) => {
    if (current?.state === 'running') throw new Error('A benchmark is already active')
    current = { id: input.id, model: input.model, state: 'running', revision: 1 }
    drains.set(input.id, new Promise(resolve => { finish = () => { current = { ...current, state: 'complete', revision: 2 }; resolve() } }))
    return Promise.resolve(current)
  },
  benchmarkStatus: (id?: string) => !id || current?.id === id ? current : null,
  benchmarkHistory: async () => current ? [current] : [],
  cancelBenchmark: (id: string) => { if (current?.id !== id) return false; cancelled.push(id); finish(); return true },
  settleBenchmarks: (id: string) => drains.get(id) ?? Promise.resolve(),
  benchmarkReceiptForExport: async (id: string) => { assert.equal(id, current.id); return { schema_version: 1, fixture: 'synthetic-local-only' } },
}
let nativeSelection: any = { canceled: true }
const imports: Record<string, any> = {
  electron: { dialog: { showSaveDialog: async () => nativeSelection } }, '../../../agent/engine/paths': { isRepoId },
  '../ipc': { handle: (name: string, callback: any) => handlers.set(name, callback) },
  '../audit': { requireSecurityRole: (role: string) => { roles.push(role); if (denied) throw new Error('Synthetic role denied') }, auditEvent: async () => { if (auditFailure) throw new Error('Synthetic audit failure') } },
  '../model-benchmark-export': { saveBenchmarkExport: async (destination: string, json: string, check: () => void) => { check(); exports.push({ destination, json }) } },
}
const source = await readFile(new URL('./model-benchmark-ipc.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', compiled)((name: string) => { assert.ok(name in imports, `Reviewed benchmark IPC dependency ${name}`); return imports[name] }, loaded, loaded.exports)
loaded.exports.registerModelBenchmarkIpc(() => available ? instance : null, { reserveOperation: () => { leaseCount++; return { release: () => { leaseCount-- } } }, assertCanBenchmark: () => { if (busy) throw new Error('Synthetic active local work') } })
const api = modelBenchmarkPreload((channel, ...args) => { assert.ok(handlers.has(channel)); return handlers.get(channel)!({}, ...args) })
assert.deepEqual(Object.keys(api).sort(), Object.keys(MODEL_BENCHMARK_API_SCHEMA).sort())
const input = { id: 'benchmark_synthetic_0001', model: 'Fixture/Model', serial: false }
denied = true
for (const fn of Object.values(api)) await assert.rejects((fn as any)(input), /role denied/)
denied = false
await assert.rejects(api.engineStartBenchmark({ ...input, publish: true } as any), /installed model/)
await assert.rejects(api.engineStartBenchmark({ ...input, model: '--publish' }), /installed model/)
assert.equal(leaseCount, 0)
available = false; await assert.rejects(api.engineStartBenchmark(input), /unavailable/); assert.equal(leaseCount, 0)
available = true; busy = true; await assert.rejects(api.engineStartBenchmark(input), /active local/); busy = false
assert.equal((await api.engineStartBenchmark(input)).id, input.id)
assert.equal(leaseCount, 1, 'prompt start response retains the update/purge reservation until restoration settles')
assert.equal(await api.engineCancelBenchmark('benchmark_other_00000'), false)
assert.equal(leaseCount, 1)
await assert.rejects(api.engineStartBenchmark({ ...input, id: 'benchmark_duplicate_02' }), /already active/)
assert.equal(leaseCount, 1, 'failed duplicate admission releases only its own reservation')
finish(); await Promise.resolve(); await Promise.resolve(); assert.equal(leaseCount, 0)
assert.equal((await api.engineBenchmarkHistory()).length, 1)
assert.deepEqual(await api.engineExportBenchmark(input.id), { saved: false })
assert.equal(exports.length, 0)
nativeSelection = { canceled: false, filePath: '/synthetic-owner-picked/receipt.json' }
assert.deepEqual(await api.engineExportBenchmark(input.id), { saved: true })
assert.match(exports[0].json, /synthetic-local-only/)
auditFailure = true
await assert.rejects(api.engineStartBenchmark({ ...input, id: 'benchmark_audit_failure' }), /audit failure/)
assert.ok(cancelled.includes('benchmark_audit_failure')); assert.equal(leaseCount, 0)
assert.ok(roles.includes('owner') && roles.includes('operator'))
const preview = createMockModelBenchmarkApi()
await assert.rejects(preview.engineStartBenchmark(input), /desktop app/)
assert.deepEqual(await preview.engineBenchmarkHistory(), [])
assert.equal(await preview.engineBenchmarkStatus(), null)
console.log('Actual benchmark preload/IPC validation, role gates, exact-job cancellation, full-lifetime lease and native export passed')
