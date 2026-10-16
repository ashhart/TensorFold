import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import ts from 'typescript'
import { createRunAttention, type AttentionNotice } from './run-attention'

// Execute the actual main-process canonical predicate, rather than duplicating it
// in a test while the direct approval callback can bypass the event emitter.
const filename = path.join(process.cwd(), 'electron/main-app.ts')
const source = ts.createSourceFile(filename, await readFile(filename, 'utf8'), ts.ScriptTarget.Latest, true)
let predicate: ts.Expression | undefined
function visit(node: ts.Node) {
  if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'createRunAttention') {
    assert.ok(ts.isObjectLiteralExpression(node.arguments[0]))
    const field = node.arguments[0].properties.find((entry) => ts.isPropertyAssignment(entry) && entry.name.getText(source) === 'isIncognito')
    assert.ok(field && ts.isPropertyAssignment(field))
    assert.equal(predicate, undefined, 'one canonical native-attention admission predicate')
    predicate = field.initializer
  }
  ts.forEachChild(node, visit)
}
visit(source)
assert.ok(predicate)
const js = ts.transpileModule(`const admission = ${predicate.getText(source)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const store: { threads: Array<{ id: string; incognito?: boolean; phoneLocalOnly?: boolean; title: string }> } = { threads: [] }
const suppressed = new Function('store', `${js}; return admission;`)(store) as (id: string) => boolean
for (const mode of ['phone', 'dynamic-phone', 'incognito', 'unknown', 'ordinary'] as const) {
  const marker = mode === 'ordinary' ? 'PUBLIC_SYNTHETIC_NATIVE_TITLE' : 'PRIVATE_SYNTHETIC_NATIVE_TITLE'
  const thread = { id: `synthetic-${mode}`, title: marker, incognito: mode === 'incognito', phoneLocalOnly: mode === 'phone' }
  store.threads = mode === 'unknown' ? [] : [thread]
  const notices: AttentionNotice[] = [], badges: string[] = []
  const attention = createRunAttention({ isIncognito: suppressed, enabled: () => true, focused: () => false, supported: () => true, threadTitle: (id) => store.threads.find((entry) => entry.id === id)?.title, notify: (notice) => notices.push(notice), setBadge: (value) => badges.push(value) })
  if (mode === 'dynamic-phone') thread.phoneLocalOnly = true
  // createApprovalRuntime invokes approvalPending directly; it does not pass
  // through createEmitHelpers' automatic projection guard.
  attention.approvalPending({ threadId: thread.id, tool: 'phone_control' })
  attention.observe({ threadId: thread.id, runId: 'synthetic-run', type: 'item_update', id: 'synthetic-question', patch: { question: { id: 'synthetic-question', question: marker, choices: [], state: 'pending' } as any } })
  attention.observe({ threadId: thread.id, runId: 'synthetic-run', type: 'error', message: marker })
  attention.observe({ threadId: thread.id, runId: 'synthetic-done', type: 'done' })
  if (mode === 'ordinary') {
    assert.ok(notices.some((notice) => notice.body.includes(marker)))
    assert.deepEqual(notices.map((notice) => notice.kind), ['approval', 'question', 'failed', 'finished'])
    assert.ok(badges.length > 0)
  } else {
    assert.deepEqual(notices, [], `${mode}: canonical private or unresolved titles and error text must not enter native notification history through direct approvals or events`)
    assert.deepEqual(badges, [])
    assert.equal(attention.needsYouCount(), 0)
  }
}
console.log('Actual main native-attention admission suppresses phone, dynamic-phone, Incognito and unresolved-owner direct approvals/questions/errors, with ordinary positive controls')
