import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { comparisonId, comparisonInput } from '../../../agent/engine/model-comparison'
import { isRepoId, isRevision } from '../../../agent/engine/paths'
import { assertCortheonReleaseAllowed } from '../../../../shared/release-feature-policy'
import { createMockEngineApi } from '../../../../src/lib/mock/engine'

const handlers = new Map<string, (...args: any[]) => Promise<any>>()
const roles: string[] = [], events: any[] = []
let allow = true, busy = false, released = 0, cancelled: string[] = [], current: string | null = null
let auditRelease: (() => void) | null = null
let resultRelease: (() => void) | null = null
let auditFailure = false
const service = {
  setComparisonAdmission(check: (() => void) | null) { this.check = check }, check: null as (() => void) | null,
  compareModels(input: any) { this.check?.(); current = input.id; return new Promise<void>((resolve) => { resultRelease = resolve }) },
  comparisonStatus(id: string) { return id === current ? { id, state: 'running' } : null },
  cancelComparison(id: string) { if (id !== current) return false; cancelled.push(id); resultRelease?.(); return true },
}
const imports: Record<string, unknown> = {
  '../../../../shared/release-feature-policy': { assertCortheonReleaseAllowed },
  electron: { BrowserWindow: {}, dialog: {} },
  '../../../agent/engine/paths': { isRepoId, isRevision },
  '../../../agent/engine/model-comparison': { comparisonId, comparisonInput },
  '../audit': { requireSecurityRole(role: string) { roles.push(role); if (!allow) throw new Error('Synthetic role denied') }, auditEvent: async (name: string, value: any) => { events.push({ name, value }); await new Promise<void>((resolve) => { auditRelease = resolve }); if (auditFailure) throw new Error('Synthetic audit failure') } },
  '../ipc': { handle: (channel: string, handler: any) => handlers.set(channel, handler) },
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'engine-ipc.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText)((id: string) => { assert.ok(id in imports, `Reviewed main import ${id}`); return imports[id] }, loaded, loaded.exports)
loaded.exports.registerEngineIpc(() => service, { reserveOperation: () => ({ release() { released++ } }), assertCanCompare: () => { if (busy) throw new Error('Synthetic active local work') } })
const invoke = (channel: string, value: unknown) => handlers.get(channel)!({}, value)
const input = { id: 'comparison_ipc_test_1', models: ['Fixture/A', 'Fixture/B'], prompt: 'Private synthetic test prompt' }
allow = false; await assert.rejects(invoke('engine:compareModels', input), /role denied/); allow = true
await assert.rejects(invoke('engine:compareModels', { ...input, files: ['/tmp/private'] }), /Expected two/)
busy = true; await assert.rejects(invoke('engine:compareModels', input), /active local/); busy = false
const pending = invoke('engine:compareModels', input)
assert.equal((await invoke('engine:comparisonStatus', input.id)).id, input.id, 'job admission precedes async audit, so cleanup can always cancel startup')
assert.equal(await invoke('engine:cancelComparison', 'comparison_wrong_123'), false)
assert.equal(await invoke('engine:cancelComparison', input.id), true)
;(auditRelease as (() => void) | null)?.(); await pending
assert.deepEqual(events[0], { name: 'engine.comparison', value: { models: input.models } }); assert.ok(!JSON.stringify(events).includes(input.prompt), 'private prompt is excluded from audit')
assert.equal(released, 2, 'refused busy and completed operations release their global-start reservations')
auditFailure = true
const failed = invoke('engine:compareModels', { ...input, id: 'comparison_ipc_test_2' })
;(auditRelease as (() => void) | null)?.(); await assert.rejects(failed, /audit failure/)
assert.ok(cancelled.includes('comparison_ipc_test_2'), 'audit failure cancels and drains its admitted job')
assert.ok(roles.every((role) => role === 'operator'))
const demo = createMockEngineApi(); await assert.rejects(demo.engineCompareModels(input as any), /desktop app/); assert.equal(await demo.engineComparisonStatus(input.id), null); assert.equal(await demo.engineCancelComparison(input.id), false)
console.log('Actual comparison IPC handlers: strict input, role and local-work checks, synchronous cancellation admission, audit privacy/drain, operation release and honest demo refusal passed.')
