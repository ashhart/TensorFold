import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

const handlers = new Map<string, (...args: any[]) => Promise<any>>()
const required: string[] = []
let denied = false
let restarted = 0
let configured: unknown
const status = { enabled: true, scope: 'selected', threadIds: ['synthetic'], roomIds: [], revision: 1, eligibleThreads: [], eligibleRooms: [] }
const dependencies = {
  handle: (name: string, callback: (...args: any[]) => Promise<any>) => handlers.set(name, callback),
  requireSecurityRole: (role: string) => { required.push(role); if (denied) throw new Error('Synthetic role denied') },
  registerBotRecipeIpc: () => {},
  getTelegramRemoteRuntime: () => ({ verifyTelegram: async () => ({ ok: true, botId: '123456', username: 'synthetic_bot', error: null }),
    configureRemote: async (input: any) => { configured = input; if (!input) throw new Error('Synthetic invalid input'); return status }, listTelegramRemoteTargets: () => status }),
}
const source = await readFile(new URL('./bot-ipc.ts', import.meta.url), 'utf8')
const ast = ts.createSourceFile('bot-ipc.ts', source, ts.ScriptTarget.ES2022, true)
const declaration = ast.statements.find((item): item is ts.FunctionDeclaration => ts.isFunctionDeclaration(item) && item.name?.text === 'registerBotIpc')
assert.ok(declaration)
const functionSource = ts.createPrinter().printNode(ts.EmitHint.Unspecified, declaration, ast).replace(/^export\s+/, '')
const compiled = ts.transpileModule(functionSource, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const register = new Function(...Object.keys(dependencies), `${compiled}; return registerBotIpc;`)(...Object.values(dependencies))
register({ pollTelegramUpdates: async () => ({ ok: true }), startBotControlPolling: () => { restarted++ } })
const call = (name: string, input?: unknown) => handlers.get(`bot:${name}`)!({}, input)
assert.equal((await call('verifyTelegram')).ok, true)
assert.equal(await call('listTelegramRemoteTargets'), status)
const input = { enabled: true, scope: 'selected', threadIds: ['synthetic'], roomIds: [] }
assert.equal(await call('configureTelegramRemote', input), status)
assert.equal(configured, input)
assert.equal(restarted, 1, 'only a committed configuration change restarts polling')
await assert.rejects(() => call('configureTelegramRemote'), /invalid input/)
assert.equal(restarted, 1)
assert.ok(required.every((role) => role === 'owner'))
denied = true
for (const name of ['verifyTelegram', 'listTelegramRemoteTargets', 'configureTelegramRemote']) await assert.rejects(() => call(name, input), /role denied/)
assert.equal(restarted, 1)
console.log('Actual Telegram IPC handlers enforce owner access and restart polling only after committed configuration')
