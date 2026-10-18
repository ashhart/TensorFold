import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { documentLibraryPreload } from '../preload/document-library'

const handlers = new Map<string, (...args: any[]) => any>(), calls: string[] = []
let role = 'operator', profile = '/synthetic/profile-one', leases = 0, release!: (value: any) => void
class FakeLibrary {
  constructor(_options: unknown) {}
  processing() { calls.push('processing'); return Promise.resolve({ jobs: [], concurrency: 1 }) }
  process(_request: unknown) { calls.push('process'); return new Promise(done => { release = done }) }
  readSource(_request: unknown) { calls.push('source'); return new Promise(done => { release = done }) }
  quiesce() { return Promise.resolve() }
}
const source = readFileSync('electron/main/host/document-library-runtime.ts', 'utf8')
const output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const imports: Record<string, any> = {
  '../../document-library': { DocumentLibrary: FakeLibrary }, '../../store': { vontraHomeDir: () => profile, vontraPath: () => profile + '/state' }, '../state': { store: { settings: { documents: { sources: [] } } } },
  './ipc': { handle: (name: string, callback: any) => handlers.set(name, callback) }, './audit': { requireSecurityRole: () => { if (role !== 'operator') throw new Error('Role revoked') } },
}
const module = { exports: {} as any }
new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports, id); return imports[id] }, module, module.exports)
module.exports.registerDocumentLibraryRuntime({ reserveOperation: () => { leases++; return { release: () => leases-- } } })
const invoke = (channel: string, ...args: unknown[]) => { calls.push(channel); return handlers.get(channel)!({}, ...args) }
const api = documentLibraryPreload(invoke)
assert.equal((await api.documentLibraryProcessing()).concurrency, 1); assert.equal(leases, 0)
let pending: Promise<unknown> = api.documentLibraryProcess({ action: 'pause' }); role = 'viewer'; release({ jobs: ['revoked-sensitive-file'] })
await assert.rejects(pending, /Role revoked/); assert.equal(leases, 0)
role = 'operator'
pending = api.documentLibraryReadSource({ documentId: 'synthetic', revision: 'a'.repeat(64), passageId: 'section' }); profile = '/synthetic/profile-two'; release({ status: 'ready', original: { text: 'old profile private text' } })
await assert.rejects(pending, /profile changed/); assert.equal(leases, 0)
assert.ok(calls.includes('documentLibrary:process') && calls.includes('documentLibrary:source') && calls.includes('documentLibrary:processing'))
console.log('PASS document queue/source IPC: preload routing, post-await role/profile fences and physical operation lease release')
