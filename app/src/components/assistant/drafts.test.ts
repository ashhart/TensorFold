import { fixtureText } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import { assistantDraftScope } from '../../../shared/assistant-drafts'
import { ConversationDrafts } from '../../lib/conversation-drafts'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-assistant-drafts-'))
const persistence = await import('../../../electron/store/composer-drafts')
const port = { getComposerDraft: persistence.readComposerDraft, saveComposerDraft: persistence.writeComposerDraft, clearComposerDraft: persistence.clearComposerDraft }
const accountA = { id: 'account-a', kind: 'imap', host: 'mail.example.test', user: 'a@example.test', password: 'SYNTHETIC-SECRET-EXCLUDED', port: 993, tls: true }
const accountB = { ...accountA, id: 'account-b', user: 'b@example.test' }
const settings: any = { defaultCwd: '/synthetic-project', approvalMode: 'approve-edits', email: { sources: [accountA, accountB] } }
const eventA = { id: 'a', connectorId: 'email', threadKey: 'email:same-external-id', sources: [{ connectorId: 'email', sourceId: accountA.id }] }
const eventB = { ...eventA, id: 'b', sources: [{ connectorId: 'email', sourceId: accountB.id }] }
const a = assistantDraftScope('email', eventA, settings), b = assistantDraftScope('all', eventB, settings)
assert.notEqual(a.conversationId, b.conversationId, 'The external conversation key alone cannot cross account boundaries')
assert.equal(a.conversationId, assistantDraftScope('all', { ...eventA, id: 'new-message' }, settings).conversationId, 'Same account conversation retains draft across new messages and source filters')
const changed = structuredClone(settings); changed.email.sources[0].user = 'new-account@example.test'
assert.notEqual(a.conversationId, assistantDraftScope('email', eventA, changed).conversationId)
assert.notEqual(assistantDraftScope('email', null, settings).conversationId, assistantDraftScope('email', null, changed).conversationId)
const modelOnly = { ...settings, activeModel: 'different-model', activeProviderAccountId: 'different-provider-account' }
assert.equal(a.conversationId, assistantDraftScope('email', eventA, modelOnly).conversationId)
assert.ok(!a.conversationId!.includes(accountA.password))
const drafts = new ConversationDrafts(port, 60_000)
const tick = async () => { for (let index = 0; index < 20; index++) await Promise.resolve() }
drafts.get(a); drafts.get(b); await tick()
drafts.update(a, { text: 'Assistant draft A' }); drafts.update(b, { text: 'Assistant draft B' }); await drafts.flushAll()
const restarted = new ConversationDrafts(port, 60_000); restarted.get(a); restarted.get(b); await restarted.flushAll()
assert.equal(restarted.get(a).text, 'Assistant draft A'); assert.equal(restarted.get(b).text, 'Assistant draft B')

const source = readFileSync(path.join(process.cwd(), 'src/components/Assistant.tsx'), 'utf8')
const tree = ts.createSourceFile('Assistant.tsx', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
function action(name: string, dependencies: Record<string, unknown>): () => Promise<void> {
  let expression: ts.Expression | null = null
  const visit = (node: ts.Node) => { if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text === name && node.initializer) expression = node.initializer; ts.forEachChild(node, visit) }
  visit(tree); assert.ok(expression, `Actual ${name} action exists`)
  const text = ts.createPrinter().printNode(ts.EmitHint.Expression, expression, tree)
  const compiled = ts.transpileModule(`const actual = ${text};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  return new Function('tx', ...Object.keys(dependencies), `${compiled};return actual;`)(fixtureText, ...Object.values(dependencies))
}
const facade = { ready: true, capture: () => drafts.capture(a), clearAccepted: (token: any) => drafts.clearAccepted(token) }
let sends = 0, launches = 0, notices: string[] = [], accepted = false, confirmation = true
let finish!: (value: any) => void
const dependencies = () => ({
  taskText: drafts.get(a).text, draft: facade, selectedItem: { from: 'Example contact', source: 'email', threadKey: 'email:example' },
  selectedSource: 'email', selectedEvent: eventA, filteredItems: [], snapshot: { briefing: '' }, settings,
  actionPending: { current: false }, setSendingReply: () => {}, setTaskBusy: () => {}, setNotice: (value: string) => notices.push(value), sourceName: () => 'Email', confirm: () => confirmation,
  api: { replyToThread: async () => { sends++; return new Promise((resolve) => { finish = resolve }) }, readEmailThread: async () => ({}) },
  refresh: async () => {}, resolveAssistantModel: () => ({ providerId: 'synthetic-provider', providerAccountId: null, model: 'synthetic-model' }), assistantTaskContext: () => 'Synthetic context', emailThreadContext: () => 'Synthetic thread context',
  launch: async () => { launches++; return { ok: accepted, error: 'Synthetic launch refused' } },
})
confirmation = false
await action('sendReply', dependencies())()
assert.equal(sends, 0); assert.equal(drafts.get(a).text, 'Assistant draft A', 'Cancelled confirmation leaves draft intact')
confirmation = true
const sending = action('sendReply', dependencies())(); await tick()
drafts.update(a, { text: 'New A while reply is sending' }); drafts.update(b, { text: 'New B while A sends' })
finish({ ok: true }); await sending
assert.equal(sends, 1); assert.equal(drafts.get(a).text, 'New A while reply is sending'); assert.equal(drafts.get(b).text, 'New B while A sends')
const failedReply = action('sendReply', dependencies())(); await tick(); finish({ ok: false, error: 'Synthetic reply failed' }); await failedReply
assert.equal(drafts.get(a).text, 'New A while reply is sending', 'A failed connector delivery never clears the draft')
accepted = false; await action('startTask', dependencies())()
assert.equal(drafts.get(a).text, 'New A while reply is sending', 'Refused task launch retains the originating draft')
accepted = true; await action('startTask', dependencies())()
assert.equal(drafts.get(a).text, ''); assert.equal(drafts.get(b).text, 'New B while A sends')
await drafts.flushAll()
assert.equal((await persistence.readComposerDraft(a)).text, '', 'Actual accepted launch clears only its captured durable revision')
assert.equal((await persistence.readComposerDraft(b)).text, 'New B while A sends')
assert.equal(launches, 2)
drafts.dispose(); restarted.dispose()
console.log('Actual Assistant reply/launch actions preserve account-scoped protected drafts across navigation, restart, confirmation, failure and newer typing')
