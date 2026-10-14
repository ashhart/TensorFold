import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import vm from 'node:vm'

const root = path.resolve(import.meta.dirname, '..')
const require = createRequire(path.join(root, 'package.json'))
const { createJiti } = require('jiti')
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-cortheon-boundary-'))
const state = { handlers: new Map(), audit: async () => {} }
globalThis.__engineCortheonBoundaryFixture = state
try {
  const fixture = path.join(temp, 'fixture.mjs')
  await fs.writeFile(fixture, `export const handle=(channel,handler)=>globalThis.__engineCortheonBoundaryFixture.handlers.set(channel,handler);export const auditEvent=()=>globalThis.__engineCortheonBoundaryFixture.audit();export const requireSecurityRole=()=>{};export const BrowserWindow={};export const dialog={};`)
  const jiti = createJiti(path.join(root, 'package.json'), { moduleCache: false, fsCache: false, alias: { '../ipc': fixture, '../audit': fixture, electron: fixture } })
  const { registerEngineIpc } = await jiti.import(path.join(root, 'electron/main/host/ipc/engine-ipc.ts'))
  const calls = []
  let reserved = 0, released = 0, privateRunActive = false, guardChecks = 0
  registerEngineIpc(() => ({ setCortheon: async (enabled) => { calls.push(enabled); return { enabled } } }), {
    reserveOperation: () => { reserved++; return { release() { released++ } } },
    assertCanEnableCortheon: () => { guardChecks++; if (privateRunActive) throw new Error('A private phone or bot VM run is active.') },
  })
  const set = (enabled) => state.handlers.get('engine:setCortheon')({}, enabled)
  let finishAudit
  state.audit = () => new Promise((resolve) => { finishAudit = resolve })
  const pending = set(true)
  privateRunActive = true
  finishAudit()
  await assert.rejects(() => pending, /private phone or bot VM/, 'privacy is checked after asynchronous audit, immediately before the service change')
  assert.deepEqual(calls, [])
  assert.equal(guardChecks, 1)
  assert.equal(reserved, released, 'a refused enable releases its operation lease')
  state.audit = async () => {}
  await set(false)
  assert.deepEqual(calls, [false], 'turning external research off remains possible during a private run')
  assert.equal(guardChecks, 1)
  privateRunActive = false
  await set(true)
  assert.deepEqual(calls, [false, true])
  await assert.rejects(() => set('true'), /on or off/)
  assert.equal(reserved, released)
  // Execute the actual main-app callback rather than a second copy of its policy.
  const ts = require('typescript')
  const main = await fs.readFile(path.join(root, 'electron/main-app.ts'), 'utf8')
  const tree = ts.createSourceFile('main-app.ts', main, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  let callback
  const visit = (node) => {
    if (ts.isCallExpression(node) && node.expression.getText(tree) === 'registerEngineIpc') {
      callback = node.arguments[1].properties.find((property) => property.name?.getText(tree) === 'assertCanEnableCortheon')?.initializer.getText(tree)
    }
    ts.forEachChild(node, visit)
  }
  visit(tree); assert.ok(callback)
  const { isRestrictedBot } = await jiti.import(path.join(root, 'shared/bot-tool-policy.ts'))
  const threads = [{ id: 'scoped-fixture', bot: { toolScope: undefined } }]
  const active = new Map([['scoped-fixture', new AbortController()]])
  const actualGuard = vm.runInNewContext(`(${callback})`, { store: { threads }, aborters: active, isRestrictedBot })
  registerEngineIpc(() => ({ setCortheon: async (enabled) => { calls.push(enabled); return { enabled } } }), { assertCanEnableCortheon: actualGuard })
  await set(true)
  state.audit = () => new Promise((resolve) => { finishAudit = resolve })
  const racedScope = set(true)
  threads[0].bot.toolScope = { mode: 'selected', names: [] }
  finishAudit()
  const beforeRefused = calls.length
  await assert.rejects(() => racedScope, /restricted bot|execution boundaries/)
  assert.equal(calls.length, beforeRefused, 'an active scoped bot appearing during audit refuses the real engine mutation')
  state.audit = async () => {}
  await set(false)
  active.clear(); await set(true)
  console.log('Registered Cortheon handler checks passed: post-audit privacy admission, refusal before service mutation, disabling allowed and leases released.')
} finally {
  delete globalThis.__engineCortheonBoundaryFixture
  await fs.rm(temp, { recursive: true, force: true })
}
