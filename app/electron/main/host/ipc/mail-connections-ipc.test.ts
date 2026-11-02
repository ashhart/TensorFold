import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { mailConnectionsPreload } from '../../../preload/mail-connections'
import { gmailInboxPreload } from '../../../preload/gmail-inbox'
import { MAIL_CONNECTIONS_API_SCHEMA } from '../../../../shared/mail-connections'
import { GMAIL_INBOX_API_SCHEMA } from '../../../../shared/gmail-inbox'

const handlers = new Map<string, (...args: any[]) => Promise<any>>()
let denied = false, calls = 0
const service: any = new Proxy({}, { get: (_target, method) => async (...args: unknown[]) => { calls++; return { method, args } } })
const owner = (role: string) => { assert.equal(role, 'owner'); if (denied) throw new Error('Synthetic owner denied') }
for (const [file, name, dependency] of [['mail-connections-ipc.ts', 'registerMailConnectionsIpc', 'mailConnectionsService'], ['gmail-inbox-ipc.ts', 'registerGmailInboxIpc', 'googleWorkspaceService']]) {
  const source = readFileSync(new URL(file, import.meta.url), 'utf8')
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.ES2022, true)
  const declaration = ast.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === name)
  assert.ok(declaration)
  const printed = ts.createPrinter().printNode(ts.EmitHint.Unspecified, declaration, ast).replace(/^export\s+/, '')
  const compiled = ts.transpileModule(printed, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  const register = new Function('requireSecurityRole', 'defaultHandle', dependency, `${compiled}; return ${name}`)(owner, (channel: string, fn: (...args: any[]) => Promise<any>) => handlers.set(channel, fn), () => service)
  register()
}
const invoke = (channel: string, ...args: unknown[]) => { assert.ok(handlers.has(channel), channel); return handlers.get(channel)!({}, ...args) }
const api = { ...mailConnectionsPreload(invoke), ...gmailInboxPreload(invoke) }
assert.deepEqual(Object.keys(api).sort(), [...Object.keys(MAIL_CONNECTIONS_API_SCHEMA), ...Object.keys(GMAIL_INBOX_API_SCHEMA)].sort())
denied = true
for (const method of Object.values(api)) await assert.rejects((method as any)({}, 'body'), /owner denied/)
assert.equal(calls, 0, 'Every channel refuses before reading state or probing a provider')
denied = false
const target = { provider: 'gmail' as const, id: 'account-a', revision: '1' }
assert.deepEqual(await api.testMailConnection(target), { method: 'test', args: [target] })
assert.deepEqual(await api.gmailInboxReply('gmail:account-a:1:dd11', 'Exact text'), { method: 'inboxReply', args: ['gmail:account-a:1:dd11', 'Exact text'] })
const draft = { key: 'gmail:account-a:1:dd11', expectedRevision: 3, text: 'Keep my draft' }
assert.deepEqual(await api.gmailInboxSaveDraft(draft), { method: 'saveInboxDraft', args: [draft] })
assert.deepEqual(await api.gmailInboxEnable({ accountId: 'account-a', revision: 1, enabled: false }), { method: 'enableInbox', args: [{ accountId: 'account-a', revision: 1, enabled: false }] })
console.log('All health and Gmail Inbox preload routes forward exact inputs and require the host owner before state or network access.')
