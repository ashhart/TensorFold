import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import * as crypto from 'node:crypto'
import * as shared from '../../../../shared/composer-drafts'
import * as assistantDrafts from '../../../../shared/assistant-drafts'
import { connectorSourceBinding } from '../../../agent/connector-source-binding'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-draft-ipc-'))
const persistence = await import('../../../store/composer-drafts')
const handlers = new Map<string, (...args: any[]) => any>()
const state: any = { settings: { remoteOrchestrators: [{ id: 'remote-fixture' }] }, connectorEvents: [], threads: [{ id: 'local-fixture', incognito: false }, { id: 'private-fixture', incognito: true }] }
let denied = false
const imports: Record<string, unknown> = {
  'node:crypto': crypto,
  '../../../../shared/composer-drafts': shared,
  '../../../../shared/assistant-drafts': assistantDrafts,
  '../../../agent/connector-source-binding': { connectorSourceBinding },
  '../../../store/composer-drafts': persistence,
  '../../state': { store: state },
  '../audit': { requireSecurityRole(role: string) { assert.equal(role, 'operator'); if (denied) throw new Error('Synthetic role denied.') } },
  '../ipc': { handle(channel: string, handler: (...args: any[]) => any) { handlers.set(channel, handler) } },
}
const module = { exports: {} as any }
const compiled = ts.transpileModule(readFileSync(path.join(process.cwd(), 'electron/main/host/ipc/composer-draft-ipc.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
new Function('require', 'module', 'exports', compiled)((id: string) => { assert.ok(id in imports, `Reviewed production import ${id}`); return imports[id] }, module, module.exports)
const { registerComposerDraftIpc, requestRendererDraftFlush } = module.exports
registerComposerDraftIpc()
const call = (name: string, input: unknown, sender = {}) => Promise.resolve().then(() => handlers.get(`composer-drafts:${name}`)!({ sender }, input))
const local = { surface: 'chat' as const, conversationId: 'local-fixture', hostId: null, incognito: false }
await call('save', { scope: local, expectedRevision: 0, text: 'Synthetic local unfinished message', attachments: [] })
assert.equal((await call('get', local)).text, 'Synthetic local unfinished message')
await assert.rejects(() => call('get', { ...local, conversationId: 'unknown-local' }), /not available/)
await assert.rejects(() => call('save', { scope: { ...local, conversationId: 'private-fixture' }, expectedRevision: 0, text: 'must not reach disk', attachments: [] }), /Incognito/)
await assert.rejects(() => call('get', { ...local, hostId: 'unconfigured-host' }), /not configured/)
const remote = { ...local, hostId: 'remote-fixture' }
assert.equal((await call('get', remote)).text, '', 'Same conversation identity on another configured host has a separate local draft')
await call('save', { scope: remote, expectedRevision: 0, text: 'remote-owned local draft', attachments: [] })
assert.equal((await call('get', local)).text, 'Synthetic local unfinished message')
denied = true
await assert.rejects(() => call('get', local), /role denied/)
await assert.rejects(() => call('save', { scope: local, expectedRevision: 1, text: 'denied', attachments: [] }), /role denied/)
denied = false

// A real IPC save begins before ACK; quit drains the write after renderer flush completes.
let requestId = ''
const sender = { isDestroyed: () => false, getURL: () => 'file:///synthetic-app/index.html', send(channel: string, value: string) { assert.equal(channel, 'composer-drafts:flush'); requestId = value } }
const quitting = requestRendererDraftFlush(sender)
await assert.rejects(() => call('flushed', { requestId }, { wrong: true }), /Unknown/)
const saving = call('save', { scope: local, expectedRevision: 1, text: 'last keystroke before Quit', attachments: [] }, sender)
await saving
await call('flushed', { requestId }, sender)
await quitting
assert.equal((await call('get', local)).text, 'last keystroke before Quit')
const failed = requestRendererDraftFlush(sender)
await call('flushed', { requestId, error: 'Synthetic disk error' }, sender)
await assert.rejects(() => failed, /could not be saved/)
await assert.rejects(() => requestRendererDraftFlush(sender, 1), /could not confirm/)
await assert.rejects(() => call('flushed', { requestId }, sender), /Unknown/)
await requestRendererDraftFlush(null)
// A boot exception can leave an allocated WebContents whose renderer never loaded.
// There can be no unsaved input there, so Quit must drain writes without a renderer ACK.
const unloaded = { isDestroyed: () => false, getURL: () => '', on() { return this }, send() { throw new Error('An unloaded renderer cannot acknowledge a draft flush') } }
module.exports.observeComposerDraftRenderer(unloaded)
const bootScope = { surface: 'home' as const, conversationId: null, hostId: null, incognito: false }
const bootWrite = persistence.writeComposerDraft({ scope: bootScope, expectedRevision: 0, text: 'Host write still drains when renderer never loaded', attachments: [] })
await requestRendererDraftFlush(unloaded, 1)
await bootWrite
assert.equal((await persistence.readComposerDraft(bootScope)).text, 'Host write still drains when renderer never loaded')
let navigationListener!: (event: unknown, url: string, inPlace: boolean, isMainFrame: boolean) => void
let currentUrl = 'about:blank'
let blankRequests = 0
const blank = { isDestroyed: () => false, getURL: () => currentUrl, on(event: string, callback: typeof navigationListener) { if (event === 'did-start-navigation') navigationListener = callback; return this }, send() { blankRequests++ } }
module.exports.observeComposerDraftRenderer(blank)
await requestRendererDraftFlush(blank, 1)
assert.equal(blankRequests, 0, 'An observed initial about:blank renderer cannot have editable app input')
navigationListener({}, 'https://synthetic.example.test/child', false, false)
await requestRendererDraftFlush(blank, 1)
assert.equal(blankRequests, 0, 'A subframe does not count as the editable app document')
currentUrl = 'file:///synthetic-app/index.html'
navigationListener({}, currentUrl, false, true)
currentUrl = 'about:blank'
await assert.rejects(() => requestRendererDraftFlush(blank, 1), /could not confirm/, 'A previously loaded renderer that becomes blank retains its ACK fence')
assert.equal(blankRequests, 1)
const unobserved = { isDestroyed: () => false, getURL: () => '', send() {} }
await assert.rejects(() => requestRendererDraftFlush(unobserved, 1), /could not confirm/, 'A late-unobserved renderer is not assumed safe to skip')
const unreadableUrl = { isDestroyed: () => false, getURL() { throw new Error('Synthetic inspection failure') }, on() { return this }, send() {} }
module.exports.observeComposerDraftRenderer(unreadableUrl)
await assert.rejects(() => requestRendererDraftFlush(unreadableUrl, 1), /could not confirm/, 'URL inspection failure retains the flush fence')
// A privacy transition while the command waits in its transaction is checked again.
const pending = call('save', { scope: local, expectedRevision: 2, text: 'must not persist after privacy transition', attachments: [] })
state.threads[0].incognito = true
await assert.rejects(() => pending, /Incognito/)
state.threads[0].incognito = false
assert.equal((await call('get', local)).text, 'last keystroke before Quit')
const sourceA = { id: 'account-a', kind: 'maildir', path: `/synthetic/${'long-location/'.repeat(90)}`, user: 'account-a@example.test', password: 'SYNTHETIC-SECRET-EXCLUDED' }
const sourceB = { ...sourceA, id: 'account-b', user: 'account-b@example.test' }
state.settings.email = { sources: [sourceA, sourceB] }
const eventA = { id: 'event-a', connectorId: 'email', threadKey: 'email:shared-external-id', sources: [{ connectorId: 'email', sourceId: sourceA.id, sourceRevision: connectorSourceBinding(sourceA) }] }
const eventB = { ...eventA, id: 'event-b', sources: [{ connectorId: 'email', sourceId: sourceB.id, sourceRevision: connectorSourceBinding(sourceB) }] }
state.connectorEvents = [eventA, eventB]
const assistantA = assistantDrafts.assistantDraftScope('email', eventA, state.settings)
const assistantB = assistantDrafts.assistantDraftScope('email', eventB, state.settings)
assert.ok(assistantA.conversationId!.length > 320, 'Long real account metadata passes through shared validation rather than breaking drafts')
assert.ok(!assistantA.conversationId!.includes(sourceA.password), 'Credentials never enter draft identity')
await call('save', { scope: assistantA, expectedRevision: 0, text: 'Account A assistant unfinished reply', attachments: [] })
await call('save', { scope: assistantB, expectedRevision: 0, text: 'Account B assistant unfinished reply', attachments: [] })
assert.equal((await call('get', assistantA)).text, 'Account A assistant unfinished reply')
assert.equal((await call('get', assistantB)).text, 'Account B assistant unfinished reply')
assert.notEqual(persistence.composerDraftFile(assistantA), persistence.composerDraftFile(assistantB))
await assert.rejects(() => call('get', { ...assistantA, conversationId: local.conversationId }), /conversation\/account/, 'External assistant scope is never treated as a local task Thread')
const overview = assistantDrafts.assistantDraftScope('email', null, state.settings)
await call('save', { scope: overview, expectedRevision: 0, text: 'Overview under original configured accounts', attachments: [] })
sourceA.user = 'replaced-account@example.test'
await assert.rejects(() => call('get', assistantA), /conversation\/account/, 'A reused source ID with a different account cannot read the old draft')
await assert.rejects(() => call('get', overview), /conversation\/account/, 'An account replacement does not inherit the prior overview draft')
const replacedOverview = assistantDrafts.assistantDraftScope('email', null, state.settings)
assert.equal((await call('get', replacedOverview)).text, '')
console.log('Actual draft IPC handlers enforce role, thread/host ownership, incognito, quit ACK identity, save-drain and bounded failure')
