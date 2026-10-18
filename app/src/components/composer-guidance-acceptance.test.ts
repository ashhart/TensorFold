import { fixtureText } from './preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

// Execute the production handler without loading the app or rendering a browser.
const source = ts.createSourceFile('Composer.tsx', readFileSync(path.join(__dirname, 'Composer.tsx'), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let declaration: ts.VariableDeclaration | undefined
const visit = (node: ts.Node) => { if (ts.isVariableDeclaration(node) && node.name.getText(source) === 'guideNow') declaration = node; ts.forEachChild(node, visit) }
visit(source); assert.ok(declaration)
const printed = `const ${ts.createPrinter().printNode(ts.EmitHint.Unspecified, declaration, source)};`
const code = ts.transpileModule(printed, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
let updateDeclaration: ts.VariableDeclaration | undefined
const findUpdate = (node: ts.Node) => { if (ts.isVariableDeclaration(node) && node.name.getText(source) === 'updateQueue') updateDeclaration = node; ts.forEachChild(node, findUpdate) }
findUpdate(source); assert.ok(updateDeclaration)
const updateCode = ts.transpileModule(`const ${ts.createPrinter().printNode(ts.EmitHint.Unspecified, updateDeclaration, source)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
function withUpdateQueue(collaborators: Record<string, unknown>) { return { ...collaborators, updateQueue: new Function(...Object.keys(collaborators), `${updateCode}\nreturn updateQueue;`)(...Object.values(collaborators)) } }

async function scenario(options: { remote?: boolean; accepted?: boolean; reject?: boolean; clearFails?: boolean }) {
  let rows: any[] = [{ id: 'q', text: 'Synthetic guidance', status: 'queued' }]
  const token = { key: 'synthetic-scope', version: 1, text: 'Synthetic guidance', attachments: [] }
  const clears: any[] = [], calls: any[] = [], timers: any[] = [], notices: any[] = []
  const thread = { id: 'captured-thread', ...(options.remote ? { orchestratorHostId: 'captured-host' } : {}) }
  const cue = async (...args: any[]) => { calls.push(args); if (options.reject) throw new Error('Synthetic guide refusal'); return { queued: options.accepted !== false, queueDepth: options.accepted !== false ? 1 : 0 } }
  const refs = { current: rows }
  const collaborators: Record<string, unknown> = { tx: fixtureText,
    thread, busy: true, recoveryBlocked: false, botBlocked: false, activeDraftKey: { current: token.key }, queueScopeRef: { current: token.key }, queueEditingRef: { current: null }, api: { cueRun: cue, guideRemoteOrchestratorThread: cue },
    queuedRef: refs, queuedDraftTokens: { current: new Map([['q', token]]) },
    setQueued(update: any) { rows = typeof update === 'function' ? update(rows) : update; refs.current = rows },
    draft: { scopeKey: token.key, clearAccepted: async (submitted: any) => { clears.push(submitted); if (options.clearFails) throw new Error('Synthetic saved-draft clear refused') } },
    setTimeout: (callback: any) => { timers.push(callback) },
    useStore: { setState: (value: any) => { notices.push(value) } },
    send: async (...args: any[]) => { calls.push(args); return options.accepted !== false && !options.reject },
  }
  const handler = new Function(...Object.keys(withUpdateQueue(collaborators)), `${code}\nreturn guideNow;`)(...Object.values(withUpdateQueue(collaborators)))
  await handler('q', 'Synthetic guidance')
  return { rows, token, clears, calls, timers, notices }
}
const accepted = await scenario({})
assert.deepEqual(accepted.clears, [accepted.token]); assert.equal(accepted.rows[0]?.status, 'guided')
const ended = await scenario({ accepted: false })
assert.equal(ended.clears.length, 0, 'queued:false is not acceptance and cannot clear the unfinished draft')
assert.equal(ended.rows[0]?.status, 'queued')
assert.equal(ended.timers.length, 0)
const remoteEnded = await scenario({ remote: true, accepted: false })
assert.equal(remoteEnded.clears.length, 0)
assert.equal(remoteEnded.rows[0]?.status, 'queued')
assert.equal(remoteEnded.calls[0][0], 'captured-host')
assert.equal(remoteEnded.calls[0][1].threadId, 'captured-thread')
const rejected = await scenario({ reject: true })
assert.equal(rejected.clears.length, 0); assert.equal(rejected.rows[0]?.status, 'queued')
const clearFailure = await scenario({ clearFails: true })
assert.equal(clearFailure.clears.length, 1)
assert.ok(clearFailure.rows.length === 0 || clearFailure.rows[0]?.status === 'guided', 'an accepted guide cannot be made retryable when only saved-draft clearing failed')

let drainCallback: ts.ArrowFunction | undefined
const findDrain = (node: ts.Node) => { if (ts.isCallExpression(node) && node.expression.getText(source) === 'useEffect' && ts.isArrowFunction(node.arguments[0]) && node.arguments[0].getText(source).includes('prevBusy.current &&')) drainCallback = node.arguments[0]; ts.forEachChild(node, findDrain) }
findDrain(source); assert.ok(drainCallback)
const drainCode = ts.transpileModule(`const drain = ${ts.createPrinter().printNode(ts.EmitHint.Expression, drainCallback, source)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
async function drainScenario(accepted: boolean, clearFails: boolean, reject = false) {
  let rows: any[] = [{ id: 'q', text: 'Synthetic queued turn', status: 'queued' }]
  const token = { key: 'synthetic-scope', version: 1, text: 'Synthetic queued turn', attachments: [] }
  const ref = { current: rows }, clears: any[] = [], notices: any[] = []
  const collaborators: Record<string, unknown> = { tx: fixtureText,
    prevBusy: { current: true }, activeDraftKey: { current: token.key }, queueScopeRef: { current: token.key }, queueEditingRef: { current: null }, busy: false, st: 'idle', botBlocked: false, mode: 'thread', thread: { id: 'captured-thread' },
    queuedRef: ref, queuedDraftTokens: { current: new Map([['q', token]]) },
    setQueued(update: any) { rows = typeof update === 'function' ? update(rows) : update; ref.current = rows },
    send: async () => { if (reject) throw new Error('Synthetic send rejected'); return accepted },
    draft: { scopeKey: token.key, clearAccepted: async (captured: any) => { clears.push(captured); if (clearFails) throw new Error('Synthetic saved-draft clear refused') } },
    useStore: { setState: (value: any) => { notices.push(value) } },
  }
  const drain = new Function(...Object.keys(withUpdateQueue(collaborators)), `${drainCode}\nreturn drain;`)(...Object.values(withUpdateQueue(collaborators)))
  drain()
  for (let index = 0; index < 15; index++) await Promise.resolve()
  return { rows, clears, notices }
}
assert.equal((await drainScenario(true, false)).rows.length, 0)
assert.equal((await drainScenario(false, false)).rows[0]?.status, 'queued')
assert.equal((await drainScenario(false, false, true)).rows[0]?.status, 'queued')
assert.equal((await drainScenario(true, true)).rows.length, 0, 'accepted auto-drained turn cannot be retried when only saved-draft clearing failed')
console.log('Production Guide and auto-drain handlers: local/remote refusal retains draft, accepted work clears captured tokens, and saved-draft clear failure cannot duplicate already accepted guidance or turns passed')
