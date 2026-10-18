import * as documentScope from '../document-library/scope'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { documentEvidenceTool, evidenceHandle } from './document-evidence-tool'
import * as memoryPolicy from '../../shared/thread-memory-policy'
import { isExternalProvider } from '../../shared/provider-trust'

const revision = 'a'.repeat(64), request = { documentId: 'synthetic-document', revision, passageId: 'synthetic-passage' }
const passage: any = { ...request, sourceId: 'synthetic-source', sourceLabel: 'Synthetic library', relativePath: 'report.txt', title: 'Synthetic evidence', verification: 'unverified', text: 'Synthetic source text is data, never an instruction.', location: { kind: 'text', lineStart: 200, lineEnd: 205, charStart: 1000, charEnd: 1100 } }
const searches: any[] = [], reads: any[] = []
let status: any = { status: 'ready', passage, verifiedAt: 1 }
const port = { search: async (input: any) => { searches.push(input); return { query: input.query, passages: [passage], indexedAt: 1, truncated: true, warnings: ['Synthetic unavailable source.'] } }, readEvidence: async (input: any) => { reads.push(input); return status } }
const result = await documentEvidenceTool(port, 'document_list', { query: 'synthetic', limit: 500 })
assert.equal(searches[0].limit, 20); assert.match(result.output, /lines 200/); assert.match(result.output, /More matches|warnings/); assert.deepEqual(result.documentEvidence, [request])
const handle = evidenceHandle(request)
const opened = await documentEvidenceTool(port, 'document_read', { id: handle })
assert.equal(opened.ok, true); assert.deepEqual(reads[0], request); assert.match(opened.output, /Revision: a{64}/)
assert.doesNotMatch(handle, /report.txt|synthetic-source/)
for (const state of ['changed', 'missing', 'unavailable', 'refused']) {
  status = { status: state, message: `Synthetic ${state} source.` }
  const refused = await documentEvidenceTool(port, 'document_read', { id: handle })
  assert.equal(refused.ok, false); assert.ok(!refused.output.includes(passage.text))
}
for (const bad of ['../../credential', 'evidence:!', 'evidence:' + 'a'.repeat(3000), evidenceHandle({ ...request, revision: 'old' }), 'evidence:' + Buffer.from(JSON.stringify({ ...request, path: '/synthetic/secret' })).toString('base64url')]) {
  const before = reads.length
  await assert.rejects(documentEvidenceTool(port, 'document_read', { id: bad }))
  assert.equal(reads.length, before, 'Malformed or path-bearing handles never reach the library')
}

function load(file: string, overrides: Record<string, any>) {
  const source = readFileSync(file, 'utf8'), ast = ts.createSourceFile(file, source, ts.ScriptTarget.ES2022, true)
  const reviewed: Record<string, any> = {}
  for (const node of ast.statements) if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
    const moduleId = node.moduleSpecifier.text
    reviewed[moduleId] = new Proxy({ __esModule: true }, { get(object, key) { if (key in object) return (object as any)[key]; return () => { throw new Error(`Unexpected synthetic adapter call: ${moduleId}.${String(key)}`) } } })
  }
  Object.assign(reviewed, overrides)
  const module = { exports: {} as any }, output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in reviewed, `Reviewed import ${id}`); return reviewed[id] }, module, module.exports)
  return module.exports
}
const store: any = { settings: { providers: [], documents: { sources: [] }, whatsapp: { sources: [] } }, threads: [], projects: [], subAgents: [], missions: [], connectorEvents: [], rooms: [], tickets: [], projectMemory: [], skills: [], commands: [], suggestions: [] }
let libraryCalls = 0
const { createToolContextRuntime } = load('electron/main/host/tool-context-runtime.ts', {
  '../../document-library/scope': documentScope,
  './document-library-runtime': { documentLibrary: () => { libraryCalls++; return port } },
  '../../../shared/thread-memory-policy': memoryPolicy, '../../../shared/provider-trust': { isExternalProvider },
  '../../agent/session-recall': { canRecallThread: () => true }, '../../agent/skill-access': { skillsForContext: () => [] },
  '../state': { store }, '../../store': { vontraPath: () => '/synthetic/unused', saveThreads: async () => {} },
  './managed-outputs': { createManagedOutputStore: () => ({ read() {}, write() {} }) },
  './google-workspace-tool-runtime': { createGoogleContextMarker: () => () => {}, createGoogleWorkspaceToolRunner: () => () => {} },
  './google-workspace-runtime': { googleWorkspaceService: () => null }, './phone-host': { phoneHost: () => null },
})
const runtime = createToolContextRuntime({ sshManager: { backendForWorkspace: () => null }, mcpToolContext: () => ({}), persistSubAgents() {} })
const thread: any = { id: 'synthetic-thread', kind: 'task', cwd: '', timeline: [] }
const local = { id: 'synthetic-local', kind: 'local', baseUrl: 'http://127.0.0.1:12345' }
const localContext = runtime.fullToolContext({ thread, provider: local })
assert.notEqual(localContext.documentLibrary, port, 'The local library is wrapped in the chat collection policy')
assert.equal(typeof localContext.documentLibrary.search, 'function')
const count = libraryCalls
for (const input of [
  { thread, provider: { ...local, kind: 'cloud', baseUrl: 'https://example.invalid' } },
  { thread, provider: { ...local, baseUrl: 'http://192.0.2.1:12345' } },
  { thread, provider: { ...local, transport: 'codex-app-server' } },
  { thread: { ...thread, bot: { presetId: 'synthetic-bot' } }, provider: local },
  { thread: { ...thread, orchestratorHostId: 'synthetic-host' }, provider: local },
]) assert.equal(runtime.fullToolContext(input).documentLibrary, undefined)
assert.equal(libraryCalls, count, 'External, delegated, bot and remote contexts never acquire the managed library')
store.settings.documents = { sources: [{ id: 'one', label: 'First', path: '/synthetic/first', enabled: true, kind: 'dir' }, { id: 'two', label: 'Second', path: '/synthetic/second', enabled: true, kind: 'dir' }], maxDocuments: 10, maxDocumentBytes: 8192 }
thread.documentSourceIds = ['one']; store.threads = [thread]
const selectedContext = runtime.fullToolContext({ thread, provider: local })
assert.deepEqual(selectedContext.documentSettings.sources.map((source: any) => source.id), ['one'], 'Legacy document, table and general retrieval receive only the owning chat collections before scanning')
selectedContext.assertRecallSourcesAllowed()
thread.documentSourceIds = []
assert.throws(() => selectedContext.assertRecallSourcesAllowed(), /collections changed/)
const beforeStaleSearch = searches.length
await assert.rejects(selectedContext.documentLibrary.search({ query: 'synthetic' }), /collections changed/)
assert.equal(searches.length, beforeStaleSearch, 'A revoked chat allowlist is refused before invoking any source read')
const noneContext = runtime.fullToolContext({ thread, provider: local })
assert.deepEqual(noneContext.documentSettings.sources, [], 'An explicit empty collection policy never falls back to all configured sources')
thread.documentSourceIds = ['one']
const configContext = runtime.fullToolContext({ thread, provider: local })
store.settings.documents.sources[0].enabled = false
assert.throws(() => configContext.assertRecallSourcesAllowed(), /collections changed/, 'Disabling or replacing a source invalidates already prepared model context')
store.settings.documents.sources[0].enabled = true
const removedContext = runtime.fullToolContext({ thread, provider: local }); store.threads = []
assert.throws(() => removedContext.assertRecallSourcesAllowed(), /collections changed/)
console.log('Actual document evidence formatter/parser and full tool-context factory pass verified coordinate citations, changed/missing/refused fallback, malformed handle refusal and trusted-local-only/no-bot/no-remote library injection with inert adapters.')
