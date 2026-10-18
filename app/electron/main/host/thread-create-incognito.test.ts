import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import ts from 'typescript'
import { createEventSpineHost } from '../../agent/event-spine-host'
import { resolveProviderModel } from '../../agent/provider-accounts'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-private-create-'))
process.env.VONTRA_HOME = home
const { saveThreads, loadThreads, defaultSettings } = await import('../../store')
const { writeComposerDraft, readComposerDraft, purgeComposerDraft, composerDraftFile } = await import('../../store/composer-drafts')
const source = await readFile(new URL('./ipc/workspace-ipc.ts', import.meta.url), 'utf8')
const tree = ts.createSourceFile('workspace-ipc.ts', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS)
const printer = ts.createPrinter()
function find<T extends ts.Node>(predicate: (node: ts.Node) => node is T): T {
  let result: T | null = null
  const visit = (node: ts.Node) => { if (!result && predicate(node)) result = node; if (!result) ts.forEachChild(node, visit) }
  visit(tree)
  assert.ok(result, 'required production source node exists')
  return result
}
function ipcHandler(channel: string) {
  const call = find((node): node is ts.CallExpression => ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'handle' && !!node.arguments[0] && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text === channel)
  return printer.printNode(ts.EmitHint.Expression, call.arguments[1], tree)
}
const keyStatement = find((node): node is ts.VariableStatement => ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === 'RENDERER_THREAD_PATCH_KEYS'))
const sanitizer = find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'sanitizeRendererThreadPatch')
function compile(channel: string, dependencies: Record<string, unknown>) {
  const body = `${printer.printNode(ts.EmitHint.Unspecified, keyStatement, tree)}\n${printer.printNode(ts.EmitHint.Unspecified, sanitizer, tree).replace(/\bexport\s+(?=function\s+sanitizeRendererThreadPatch)/, '')}\nconst actualHandler = ${ipcHandler(channel)};`
  const compiled = ts.transpileModule(body, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  return new Function(...Object.keys(dependencies), `${compiled}\nreturn actualHandler;`)(...Object.values(dependencies)) as (event: unknown, input: any) => Promise<any>
}

// This executes the exact production handler AST under inert Electron/role/routing
// collaborators, with the real protected thread store and real in-memory spine host.
const provider: any = { id: 'synthetic-local', kind: 'local', models: ['synthetic'] }
const calls = { persist: 0, local: [] as any[], remote: [] as any[], spineLookups: 0, folders: 0 }
const writes: Promise<void>[] = []
const privateCreateCommands = new Map<string, string>()
const aborters = new Map<string, AbortController>()
const spineFiles = new Map<string, string>()
const spineHost = await createEventSpineHost({ journalFile: '/inert-spine.jsonl', fs: {
  readFile: async (file) => spineFiles.get(file) ?? '',
  appendFile: async (file, value) => { spineFiles.set(file, (spineFiles.get(file) ?? '') + value) },
  writeFile: async (file, value) => { spineFiles.set(file, value) },
  rename: async (from, to) => { spineFiles.set(to, spineFiles.get(from) ?? ''); spineFiles.delete(from) },
} })
const ordinaryControl: any = { id: 'ordinary-control', title: 'Ordinary fixture history', kind: 'chat', cwd: home, providerId: provider.id, model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Keep this synthetic ordinary history.' }], timeline: [] }
const store: any = { threads: [ordinaryControl], projects: [], settings: { ...defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'synthetic', defaultCwd: home }, win: { webContents: { send: (_channel: string, thread: any) => { calls.local.push({ id: thread.id, incognito: thread.incognito === true }) } } } }
let folderGate: Promise<void> = Promise.resolve()
const dependencies = {
  store, privateCreateCommands, aborters, randomUUID,
  requireSecurityRole: (role: string) => { assert.equal(role, 'operator') },
  spine: () => { calls.spineLookups++; return spineHost },
  dirExists: async () => { calls.folders++; await folderGate; return true },
  resolveProjectProvider: () => ({ baseProvider: provider, providerAccountId: null, model: 'synthetic' }),
  selectedProviderAccountId: () => null, resolveProviderModel,
  persistThreads: () => { calls.persist++; writes.push(saveThreads(store.threads)) },
  remoteServer: { publish: ({ data }: any) => { calls.remote.push({ id: data.id, incognito: data.incognito === true }) } },
  saveThreads, purgeComposerDraft,
  workspaceMutationGuard: { reserveStart: () => ({ release() {} }) },
  sendToRenderer: (channel: string, value: any) => { assert.equal(channel, 'composer-drafts:reset'); assert.equal(value.incognito, true) },
}
const create = compile('threads:create', dependencies)
const patch = compile('threads:patch', dependencies)
try {
  await saveThreads(store.threads)
  let release!: () => void
  folderGate = new Promise<void>((resolve) => { release = resolve })
  const input = { cwd: home, kind: 'task', commandId: 'synthetic-private-command', incognito: true }
  const first = create(null, input), duplicate = create(null, input)
  assert.equal(calls.folders, 2, 'both concurrent retries reach the asynchronous folder check')
  assert.equal(store.threads.length, 1, 'no creation is committed before folder readiness')
  release()
  const [a, b] = await Promise.all([first, duplicate])
  assert.equal(a.id, b.id, 'the post-await retry check prevents duplicate private records')
  assert.equal(a.incognito, true, 'the record is private at its initial construction')
  assert.equal(store.threads.length, 2)
  assert.equal(calls.persist, 1)
  assert.deepEqual(calls.local, [{ id: a.id, incognito: true }])
  assert.deepEqual(calls.remote, [{ id: a.id, incognito: true }], 'the emitter sees the private flag before it applies its separately tested projection policy')
  assert.equal(calls.spineLookups, 0, 'private creation never even obtains the durable spine')
  assert.equal(spineHost.receipt(input.commandId), null)
  await Promise.all(writes); await spineHost.flush()
  assert.deepEqual((await loadThreads()).map((thread) => thread.id), [ordinaryControl.id], 'the actual protected file preserves ordinary history and omits private creation')
  assert.ok([...spineFiles.values()].every((raw) => !raw.includes(a.id)), 'no private creation event or receipt reaches the real spine journal adapter')
  assert.equal((await create(null, input)).id, a.id, 'lost-reply retry returns the same live private record')
  assert.equal(calls.persist, 1)

  const beforeInvalid = store.threads.length
  for (const incognito of ['true', 1, null, {}]) await assert.rejects(() => create(null, { cwd: home, incognito }), /boolean/)
  assert.equal(store.threads.length, beforeInvalid)
  assert.equal(calls.persist, 1, 'invalid privacy input causes no persistence/publication')

  folderGate = new Promise<void>((resolve) => { release = resolve })
  const normalFirst = create(null, { cwd: home, kind: 'task', commandId: 'synthetic-normal-command' })
  const normalRetry = create(null, { cwd: home, kind: 'task', commandId: 'synthetic-normal-command' })
  release()
  const [normal, normalDuplicate] = await Promise.all([normalFirst, normalRetry])
  assert.equal(normal.id, normalDuplicate.id, 'ordinary concurrent retries also retain exactly-once creation')
  assert.notEqual(normal.incognito, true)
  assert.equal((await create(null, { cwd: home, commandId: 'synthetic-normal-command' })).id, normal.id)
  await Promise.all(writes); await spineHost.flush()
  assert.ok(spineHost.receipt('synthetic-normal-command'), 'normal creation retains durable exactly-once receipts')
  assert.ok((await loadThreads()).some((thread) => thread.id === normal.id), 'ordinary creation still persists')

  const persistBeforePatch = calls.persist
  aborters.set(normal.id, new AbortController())
  await assert.rejects(() => patch(null, { id: normal.id, patch: { incognito: true } }), /before a thread starts/, 'startup reservation blocks a late toggle even before messages/status exist')
  aborters.delete(normal.id); normal.status = 'running'
  await assert.rejects(() => patch(null, { id: normal.id, patch: { incognito: true } }), /before a thread starts/)
  normal.status = 'idle'
  await assert.rejects(() => patch(null, { id: a.id, patch: { incognito: false } }), /cannot be converted/)
  await assert.rejects(() => patch(null, { id: normal.id, patch: { incognito: 'true' } }), /boolean/)
  assert.equal(calls.persist, persistBeforePatch, 'refused late or invalid toggles make no durable mutation')
  const ordinaryDraftScope = { surface: 'chat' as const, conversationId: normal.id, hostId: null, incognito: false }
  await writeComposerDraft({ scope: ordinaryDraftScope, expectedRevision: 0, text: 'Unsent normal draft before choosing Incognito', attachments: [] })
  assert.equal((await readComposerDraft(ordinaryDraftScope)).text, 'Unsent normal draft before choosing Incognito')
  await patch(null, { id: normal.id, patch: { incognito: true } })
  assert.equal(normal.incognito, true)
  await assert.rejects(() => readFile(composerDraftFile(ordinaryDraftScope)), /ENOENT/, 'Choosing Incognito removes the previously persisted normal draft')
  assert.equal((await loadThreads()).some((thread) => thread.id === normal.id), false, 'Transition completes protected thread omission before returning')
  console.log('Exact native creation/patch handlers: atomic private flag, async retry dedup, real protected-store omission/no spine receipt, ordinary persistence and late/invalid privacy refusals pass; Electron sender validation remains separately tested.')
} finally {
  await Promise.allSettled(writes); await spineHost.flush()
  await rm(home, { recursive: true, force: true })
}
