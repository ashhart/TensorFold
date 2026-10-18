import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import { ConversationDrafts } from '../../../../src/lib/conversation-drafts'
import type { ComposerDraftPort, ComposerDraftScope } from '../../../../shared/composer-drafts'

// Invoke the actual production registrations without Electron's renderer process.
// Only their outer host dependencies are injected; no privacy handler is reimplemented.
function productionRegistration(channel: string, dependencies: Record<string, unknown>) {
  const filename = path.join(process.cwd(), 'electron/main/host/ipc/workspace-ipc.ts')
  const text = readFileSync(filename, 'utf8')
  const source = ts.createSourceFile(filename, text, ts.ScriptTarget.Latest, true)
  let registration: ts.CallExpression | undefined
  let sanitizer: ts.FunctionDeclaration | undefined
  let patchKeys: ts.VariableStatement | undefined
  function visit(node: ts.Node) {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'handle' && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text === channel) registration = node
    if (ts.isFunctionDeclaration(node) && node.name?.text === 'sanitizeRendererThreadPatch') sanitizer = node
    if (ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === 'RENDERER_THREAD_PATCH_KEYS')) patchKeys = node
    ts.forEachChild(node, visit)
  }
  visit(source)
  assert.ok(registration && sanitizer && patchKeys, 'Production privacy registration and patch validator must exist')
  let handler!: (...args: any[]) => Promise<any>
  const compiled = ts.transpileModule(`${patchKeys.getText(source)}\n${sanitizer.getText(source).replace(/^export\s+/, '')}\n${registration.getText(source)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  const names = Object.keys(dependencies)
  new Function(...names, 'handle', compiled)(...Object.values(dependencies), (name: string, callback: typeof handler) => { assert.equal(name, channel); handler = callback })
  return (input: unknown) => handler({}, input)
}

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-draft-privacy-'))
const persistence = await import('../../../store/composer-drafts')
const { store } = await import('../../state')
const { defaultSettings, saveThreads } = await import('../../../store')
const { sendToRenderer } = await import('../renderer-events')
const { archiveEntity } = await import('../archive')
const scope: ComposerDraftScope = { surface: 'chat', conversationId: 'synthetic-private-transition', hostId: null, incognito: false }
const secondScope: ComposerDraftScope = { ...scope, conversationId: 'synthetic-retained-chat' }
const thread = (id: string) => ({ id, kind: 'chat', status: 'idle', projectId: null, timeline: [], apiMessages: [], createdAt: 1, updatedAt: 1, title: 'Synthetic privacy fixture', cwd: '', workspace: null } as any)
store.settings = defaultSettings()
store.threads = [thread(scope.conversationId!), thread(secondScope.conversationId!)]
const port: ComposerDraftPort = {
  getComposerDraft: persistence.readComposerDraft,
  saveComposerDraft: persistence.writeComposerDraft,
  clearComposerDraft: persistence.clearComposerDraft,
}
const drafts = new ConversationDrafts(port, 60_000)
drafts.update(scope, { text: 'Synthetic normal draft before private mode' })
drafts.update(secondScope, { text: 'Synthetic unrelated draft remains' })
await drafts.flushAll()
const oldSend = drafts.capture(scope)
const file = persistence.composerDraftFile(scope)
writeFileSync(`${file}.undecryptable-fixture.bak`, 'Synthetic retired draft recovery bytes')
const events: Array<{ channel: string; payload: any }> = []
store.win = { isDestroyed: () => false, webContents: { isDestroyed: () => false, send(channel: string, payload: any) { events.push({ channel, payload }); if (channel === 'composer-drafts:reset') { if (payload?.incognito) drafts.makeIncognito(payload); else drafts.forget(payload) } } } } as any
let releases = 0
const aborters = new Map()
const dependencies = {
  requireSecurityRole(role: string) { assert.equal(role, 'operator') },
  store, aborters, saveThreads, purgeComposerDraft: persistence.purgeComposerDraft, sendToRenderer,
  workspaceMutationGuard: { reserveStart: () => ({ release() { releases++ } }) },
  remoteServer: { publish() {} }, persistThreads() { throw new Error('Privacy transition must await its durable write') },
}
const patch = productionRegistration('threads:patch', dependencies)
let ownerChecks = 0
// Admit a normal save, then switch privacy synchronously before its queued write executes.
const lateSave = persistence.writeComposerDraft({ scope, expectedRevision: 1, text: 'Must not resurrect a private chat', attachments: [] }, () => {
  ownerChecks++
  if (store.threads[0].incognito) throw new Error('Incognito owner refuses old normal draft')
})
const rejectedLateSave = assert.rejects(lateSave, /Incognito/)
await patch({ id: scope.conversationId, patch: { incognito: true } })
await rejectedLateSave
assert.ok(ownerChecks > 0, 'Queued writes check current privacy ownership')
assert.equal(store.threads[0].incognito, true)
assert.equal(releases, 1, 'Privacy transition releases the workspace reservation')
assert.equal(existsSync(file), false, 'Private transition removes the old protected normal draft')
assert.equal(existsSync(`${file}.undecryptable-fixture.bak`), false, 'Private transition also removes draft recovery copies')
assert.equal(drafts.get(scope).text, '', 'Host reset clears only the retired normal draft memory')
assert.equal(await drafts.clearAccepted(oldSend), false, 'A sent token captured before privacy reset cannot clear new work')
assert.equal(drafts.get(secondScope).text, 'Synthetic unrelated draft remains')
assert.equal((await persistence.readComposerDraft(secondScope)).text, 'Synthetic unrelated draft remains')
assert.deepEqual(events.filter((event) => event.channel === 'composer-drafts:reset').map((event) => event.payload), [{ ...scope, incognito: true }])
await assert.rejects(() => patch({ id: scope.conversationId, patch: { incognito: false } }), /cannot be converted/)
const privateScope = { ...scope, incognito: true }
assert.equal(drafts.get(privateScope).text, 'Synthetic normal draft before private mode', 'First-turn privacy selection preserves unfinished text only in session memory')
drafts.update(privateScope, { text: 'Synthetic session-only private draft' })
await drafts.flushAll()
assert.equal(existsSync(file), false, 'Private draft never restores the retired normal file')

// Archive is recoverable navigation, not Forget: its real host path retains the draft.
const archive = productionRegistration('threads:delete', {
  requireSecurityRole(role: string) { assert.equal(role, 'operator') }, store, aborters,
  spine: () => null, archiveEntity, abortMissionsForArchive: async () => {}, stopLoopsForArchive: async () => {},
  remoteServer: { publish() {} },
})
await archive(secondScope.conversationId)
assert.ok(store.threads.find((candidate) => candidate.id === secondScope.conversationId)?.archivedAt)
assert.equal((await persistence.readComposerDraft(secondScope)).text, 'Synthetic unrelated draft remains', 'Archiving a thread must preserve its protected draft')
assert.equal(drafts.get(secondScope).text, 'Synthetic unrelated draft remains', 'Archive does not emit a Forget reset')
drafts.dispose()
store.win = null
console.log('Actual thread privacy handler fences stale normal saves, purges protected drafts/recovery bytes, resets only that conversation, retains incognito in memory, and preserves archived drafts')
