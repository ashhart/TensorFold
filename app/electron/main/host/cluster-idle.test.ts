import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
const file = ts.createSourceFile('cluster-runtime.ts', readFileSync('electron/main/host/cluster-runtime.ts', 'utf8'), ts.ScriptTarget.ES2022, true)
const names = ['observe', 'loadClusterWorkspaceIdleState', 'assertClusterWorkspaceIdle']
const selected = file.statements.filter(node => ts.isFunctionDeclaration(node) && names.includes(node.name?.text ?? ''))
assert.equal(selected.length, names.length)
const printed = selected.map(node => ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, file).replace(/\bexport (?=(?:async )?function)/g, '')).join('\n')
let root = '/synthetic/one', exists = true, bad = false, pending: (() => void) | null = null
let value: any = { jobs: [], deployments: [], hosts: [] }
let reads = 0
class ProtectedStore {
  constructor(readonly options: { assertCurrent(): void }) {}
  async get() { reads++; this.options.assertCurrent(); pending?.(); if (bad) throw new Error('Synthetic unreadable protected state'); this.options.assertCurrent(); return structuredClone(value) }
}
const module = ts.transpileModule('let unresolved=false,checkedRoot=null,active=null;\n' + printed + '\nreturn { loadClusterWorkspaceIdleState,assertClusterWorkspaceIdle,setRunning(value){active=value?{manager:{running:new Map([["synthetic",true]])}}:null} }', { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const api = new Function('ClusterWorkspaceStore', 'vontraHomeDir', 'existsSync', 'path', module)(ProtectedStore, () => root, () => exists, { join: (...parts: string[]) => parts.join('/') })
assert.throws(() => api.assertClusterWorkspaceIdle(), /reconcile/, 'Uninspected saved receipts cannot be discarded during update or purge')
await api.loadClusterWorkspaceIdleState(); assert.equal(reads, 1); api.assertClusterWorkspaceIdle()
value = { jobs: [], deployments: [], hosts: [{ state: 'ready' }] }
await api.loadClusterWorkspaceIdleState(); api.assertClusterWorkspaceIdle()
for (const phase of ['queued', 'running', 'reconciling', 'needs-review']) {
  value = { jobs: [{ phase }], deployments: [], hosts: [] }; await api.loadClusterWorkspaceIdleState(); assert.throws(() => api.assertClusterWorkspaceIdle(), /reconcile/)
}
value = { jobs: [], deployments: [{ state: 'ready' }], hosts: [] }; await api.loadClusterWorkspaceIdleState(); assert.throws(() => api.assertClusterWorkspaceIdle(), /stop/)
value = { jobs: [], deployments: [{ state: 'stopped' }], hosts: [] }; await api.loadClusterWorkspaceIdleState(); api.assertClusterWorkspaceIdle()
bad = true; await api.loadClusterWorkspaceIdleState(); assert.throws(() => api.assertClusterWorkspaceIdle(), /reconcile/, 'Corrupt receipts leave the app available but forbid forgetting remote work')
bad = false; await api.loadClusterWorkspaceIdleState(); api.setRunning(true); assert.throws(() => api.assertClusterWorkspaceIdle(), /reconcile/); api.setRunning(false)
pending = () => { root = '/synthetic/two' }; await api.loadClusterWorkspaceIdleState(); pending = null
assert.throws(() => api.assertClusterWorkspaceIdle(), /reconcile/, 'An old profile read cannot declare a new profile idle')
exists = false; api.assertClusterWorkspaceIdle()
console.log('Startup compute idle barrier: metadata-only workspaces allow updates, active/uncertain/corrupt receipts block erasure, and profile drift cannot publish another account’s idle state; no network or model effects')
