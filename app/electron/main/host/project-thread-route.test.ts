import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { providerWithAccount, resolveProviderModel, selectProviderAccount } from '../../agent/provider-accounts'

function readSource(file: string): ts.SourceFile {
  return ts.createSourceFile(file, readFileSync(path.join(__dirname, file), 'utf8'), ts.ScriptTarget.ES2022, true)
}

function find<T extends ts.Node>(root: ts.Node, match: (node: ts.Node) => node is T): T {
  let found: T | undefined
  const visit = (node: ts.Node) => {
    if (match(node)) found = node
    if (!found) ts.forEachChild(node, visit)
  }
  visit(root)
  assert.ok(found, 'production route source exists')
  return found
}

function compile(source: string, exports: string[], dependencies: Record<string, unknown>) {
  const output = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText
  return new Function(...Object.keys(dependencies), `${output}\nreturn { ${exports.join(', ')} }`)(...Object.values(dependencies))
}

const printer = ts.createPrinter()
const providerSource = readSource('provider-runtime.ts')
const ipcSource = readSource('ipc/workspace-ipc.ts')
const routeNames = ['selectedProviderAccountId', 'runtimeProvider', 'resolveProjectProvider']
const functions = routeNames.map((name) => {
  const declaration = find(providerSource, (node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === name)
  return printer.printNode(ts.EmitHint.Unspecified, declaration, providerSource).replace(/^export /, '')
}).join('\n')
const call = find(ipcSource, (node): node is ts.CallExpression =>
  ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'handle'
  && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text === 'threads:create')
const handler = printer.printNode(ts.EmitHint.Expression, call.arguments[1], ipcSource)

const provider = {
  id: 'fixture', name: 'Fixture', kind: 'local', models: ['base-model'], defaultAccountId: 'named',
  accounts: [{ id: 'named', name: 'Named fixture', models: ['named-model'] }],
}
const project = { id: 'project', defaultProviderId: provider.id, defaultProviderAccountId: null, defaultModel: 'base-model' }
const store = {
  settings: { providers: [provider], activeProviderId: provider.id, activeProviderAccountId: 'named', activeModel: 'named-model' },
  projects: [project], threads: [] as Array<{ providerAccountId: string | null; model: string }>,
}
const route = compile(functions, routeNames, { store, providerWithAccount, resolveProviderModel, selectProviderAccount })
let sequence = 0
const persisted: Array<{ providerAccountId: string | null; model: string }> = []
const { create } = compile(`const create = ${handler}`, ['create'], {
  store, ...route, resolveProviderModel,
  requireSecurityRole() {},
  workspaceMutationGuard: { reserveStart: () => ({ release() {} }) },
  spine: () => null,
  privateCreateCommands: new Map(),
  randomUUID: () => String(++sequence),
  dirExists: async () => true,
  persistThreads: () => { persisted.push(structuredClone(store.threads[0])) },
  remoteServer: { publish() {} },
  assertThreadMemoryTransport() {},
})

for (const kind of ['chat', 'task']) {
  const thread = await create(null, { cwd: '/fixture', kind, projectId: project.id })
  assert.deepEqual([thread.providerAccountId, thread.model], [null, 'base-model'])
  const record = persisted.at(-1)
  assert.deepEqual([record?.providerAccountId, record?.model], [null, 'base-model'])
}

const noAccount = route.resolveProjectProvider({ ...project, defaultProviderAccountId: undefined })
assert.deepEqual([noAccount.providerAccountId, noAccount.model], ['named', 'named-model'])
const noProjectProvider = route.resolveProjectProvider({ ...project, defaultProviderId: null })
assert.equal(noProjectProvider.providerAccountId, 'named')
store.projects.push({ ...project, id: 'missing-provider', defaultProviderId: 'missing' })
const inherited = await create(null, { cwd: '/fixture', kind: 'chat', projectId: 'missing-provider' })
assert.deepEqual([inherited.providerAccountId, inherited.model], ['named', 'named-model'])
const explicitOverride = route.resolveProjectProvider(project, 'named')
assert.deepEqual([explicitOverride.providerAccountId, explicitOverride.model], ['named', 'named-model'])

console.log('Project Chat and Task persist the selected provider key and retain inherited or explicitly named accounts')
