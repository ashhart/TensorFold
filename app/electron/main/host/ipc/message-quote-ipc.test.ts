import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { prepareMessageQuote } from '../../../agent/conversation-quotation'
const handlers = new Map<string, (...args: any[]) => any>()
let role = 'operator', leases = 0
const thread: any = { id: 'chat', timeline: [{ id: 'answer', kind: 'assistant', text: 'Synthetic amber source' }] }
const store: any = { threads: [thread], settings: { providers: [{ id: 'local', transport: 'chat-completions' }] } }; thread.providerId = 'local'
const imports: Record<string, any> = {
  '../../../agent/conversation-quotation': { prepareMessageQuote }, '../../../store': { vontraHomeDir: () => '/synthetic/profile' }, '../../state': { store },
  '../audit': { requireSecurityRole() { if (role !== 'operator') throw new Error('Synthetic denied role') } },
  '../ipc': { handle(channel: string, callback: any) { handlers.set(channel, callback) } },
}
const module = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync('electron/main/host/ipc/message-quote-ipc.ts', 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText)((id: string) => { assert.ok(id in imports, id); return imports[id] }, module, module.exports)
module.exports.registerMessageQuoteIpc({ reserveOperation() { leases++; return { release() { leases-- } } } })
const call = handlers.get('threads:prepareMessageQuote')!, input = { threadId: 'chat', itemId: 'answer' }
assert.equal((await call({}, input)).sourceText, 'Synthetic amber source'); assert.equal(leases, 0)
role = 'viewer'; await assert.rejects(call({}, input), /denied role/); assert.equal(leases, 0)
role = 'operator'; thread.archivedAt = 1; await assert.rejects(call({}, input), /active conversation/); assert.equal(leases, 0)
delete thread.archivedAt; thread.orchestratorHostId = 'remote'; await assert.rejects(call({}, input), /owning host/); assert.equal(leases, 0)
delete thread.orchestratorHostId; store.settings.providers[0].transport = 'codex-cli'; await assert.rejects(call({}, input), /chat API/); assert.equal(leases, 0)
console.log('Production quote IPC requires the visible owning local chat, supported transport, operator role and released operation lease on every refusal')
