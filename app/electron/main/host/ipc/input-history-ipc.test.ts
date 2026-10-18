import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
const handlers = new Map<string, (...args: any[]) => any>()
let denied = false, held = 0, calls = 0, refused = false
const action = async () => { calls++; assert.equal(held, 1); if (refused) throw new Error('Synthetic protected storage refusal'); return 'synthetic result' }
const imports: Record<string, unknown> = {
  '../input-history': { getInputHistory: action, recallInputHistory: action, clearInputHistory: action },
  '../audit': { requireSecurityRole(role: string) { assert.equal(role, 'operator'); if (denied) throw new Error('Synthetic denied role') } },
  '../ipc': { handle(channel: string, callback: (...args: any[]) => any) { handlers.set(channel, callback) } },
}
const module = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync('electron/main/host/ipc/input-history-ipc.ts', 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText)((id: string) => { assert.ok(id in imports); return imports[id] }, module, module.exports)
module.exports.registerInputHistoryIpc({ reserveOperation() { held++; return { release() { held-- } } } })
for (const channel of ['get', 'recall', 'clear']) {
  const handler = handlers.get('input-history:' + channel)!
  assert.equal(await handler({}, {}), 'synthetic result'); assert.equal(held, 0)
  refused = true; await assert.rejects(() => handler({}, {}), /storage refusal/); assert.equal(held, 0)
  refused = false; denied = true; const previous = calls; await assert.rejects(() => handler({}, {}), /denied role/)
  assert.equal(calls, previous); assert.equal(held, 0); denied = false
}
console.log('Input history IPC requires an operator and holds/releases the workspace mutation lease through reads, recall, clearing and failures')
