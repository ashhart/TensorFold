import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { assertGmailInboxRoute } from '../../../../shared/gmail-inbox'

const source = readFileSync(new URL('./remote-run-ipc.ts', import.meta.url), 'utf8')
const tree = ts.createSourceFile('remote-run-ipc.ts', source, ts.ScriptTarget.ES2022, true)
const registration = tree.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'registerRemoteRunIpc')
assert.ok(registration)
const body = ts.createPrinter().printNode(ts.EmitHint.Unspecified, registration, tree).replace(/^export\s+/, '')
const code = ts.transpileModule(body, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const handlers = new Map<string, (...args: any[]) => Promise<any>>()
let calls = 0, streams = 0
const register = new Function('handle', 'requireSecurityRole', 'assertGmailInboxRoute', `${code}; return registerRemoteRunIpc`)((channel: string, fn: any) => handlers.set(channel, fn), () => { }, assertGmailInboxRoute)
register({ remoteHostForId: () => ({ id: 'synthetic-remote' }), ensureRemoteEventStream: () => { streams++ }, fetchRemoteOrchestratorJson: async () => { calls++; return { data: { runId: 'ordinary-run' } } }, remoteOrchestratorConnectionError: (error: Error) => error.message })
await assert.rejects(handlers.get('remote:startThread')!({}, { payload: { threadId: 'task', userText: 'Synthetic private Gmail', sourceThreadKey: 'gmail:account-a:1:dd11' } }), /Gmail Inbox tasks run on this computer/)
assert.equal(calls, 0); assert.equal(streams, 0, 'No network stream or run request is admitted for Gmail')
await assert.rejects(handlers.get('remote:startThread')!({}, { payload: { threadId: 'task', userText: 'Synthetic mail', sourceThreadKey: 'gmail:invalid' } }), /current Gmail conversation/)
assert.equal(calls, 0)
assert.deepEqual(await handlers.get('remote:startThread')!({}, { payload: { threadId: 'task', userText: 'Ordinary task' } }), { runId: 'ordinary-run' })
assert.equal(calls, 1); assert.equal(streams, 1)
console.log('Native remote-start rejects Gmail and malformed Gmail keys before network access; ordinary remote execution remains available.')
