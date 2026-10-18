import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import { projectMemoryAutoCandidateTurn, projectMemoryAutoMessages, projectMemoryAutoSourceSuppressed } from './project-memory-auto-policy'
import { authoredMemoryMessages, harvestPostRunMemoryCandidates } from '../../agent/memory-harvest'
import { createProjectMemoryCandidate, scopeProjectMemoryEntries } from '../../agent/project-memory'
import { canContributeThreadMemory } from '../../../shared/thread-memory-policy'
import { threadTurnNumber } from './checkpoint-runtime'
import type { ProjectMemoryEntry, Thread } from '../../agent/types'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-trash-consumers-'))
process.env.VONTRA_HOME = home
const { store } = await import('../state')
const { defaultSettings } = await import('../../store')
const { createToolContextRuntime } = await import('./tool-context-runtime')
const { projectMemorySourceSnapshots, renderProjectMemoryBucket } = await import('./project-memory-renderer')
const { executeTool } = await import('../../agent/tools')
const saved = { ...store }
const entry = (id: string, text: string, extra: Partial<ProjectMemoryEntry> = {}): ProjectMemoryEntry => ({
  id, projectId: 'synthetic-alpha', text, kind: 'preference', sourceType: 'manual', sourceRef: null,
  confidence: 'high', privacyScope: 'project', owner: null, lastVerifiedAt: null, createdAt: 1, updatedAt: 1, ...extra,
})
const active = entry('shared-id', 'SYNTHETIC_ACTIVE_ALPHA_EXAMPLE_INVALID')
const hidden = entry('hidden', 'SYNTHETIC_TRASH_TOKEN_EXAMPLE_INVALID', { trashedAt: 10, sourceThreadId: 'synthetic-owner', sourceTurn: 1 })
const beta = entry('shared-id', 'SYNTHETIC_OTHER_PROJECT_EXAMPLE_INVALID', { projectId: 'synthetic-beta' })
const provider: any = { id: 'synthetic-local', name: 'Synthetic local', kind: 'local', models: ['synthetic'] }
const project: any = { id: active.projectId, name: 'Synthetic project', cwd: home, memory: 'SYNTHETIC_USER_CHARTER_CONTROL', team: '', createdAt: 1, updatedAt: 1 }
const owner = (kind: 'chat' | 'task', route: string): Thread => ({
  id: `synthetic-${route}`, title: 'Synthetic owner', kind, projectId: project.id, cwd: home,
  providerId: provider.id, model: 'synthetic', status: 'idle', createdAt: 1, updatedAt: 1,
  memoryPolicy: { useExisting: true, contribute: true }, timeline: [], apiMessages: [],
})
const runtime = createToolContextRuntime({
  sshManager: { backendForWorkspace: () => null }, persistSubAgents() {}, persistThreads() {}, applyEventToThread() {},
  emitToRenderer() {}, emitPluginEvent() {}, renderProjectMemoryBucket, recordEmailDraft() {}, recordEmailEvent() {},
  recordCalendarEvent() {}, recordMessageEvent() {}, runBrowserAutomation() {}, mcpToolContext: () => ({}), cancelThreadRun() {},
} as any)

// Execute the production nested functions with inert persistence collaborators, avoiding a model run.
async function productionMemoryCallbacks(thread: Thread) {
  const source = await readFile(new URL('./thread-runtime.ts', import.meta.url), 'utf8')
  const tree = ts.createSourceFile('thread-runtime.ts', source, ts.ScriptTarget.Latest, true)
  let postRun: ts.FunctionDeclaration | undefined, compaction: ts.ArrowFunction | undefined
  function visit(node: ts.Node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === 'proposePostRunMemory') postRun = node
    if (ts.isPropertyAssignment(node) && node.name.getText(tree) === 'onMemoryHarvest' && ts.isArrowFunction(node.initializer)) compaction = node.initializer
    ts.forEachChild(node, visit)
  }
  visit(tree)
  assert.ok(postRun && compaction, 'production memory filing seams exist')
  const printer = ts.createPrinter()
  const compiled = ts.transpileModule(`${printer.printNode(ts.EmitHint.Unspecified, postRun, tree)}\nconst onMemoryHarvest = ${printer.printNode(ts.EmitHint.Expression, compaction, tree)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
  const controller = new AbortController()
  const deps = { store, thread, memoryPolicy: thread.memoryPolicy, memoryRevision: 0, controller, canContributeThreadMemory,
    projectMemoryAutoCandidateTurn, projectMemoryAutoMessages, scopeProjectMemoryEntries, authoredMemoryMessages, harvestPostRunMemoryCandidates,
    createProjectMemoryCandidate, threadTurnNumber, persistProjectMemoryCandidates() {},
  }
  return new Function(...Object.keys(deps), `${compiled}\nreturn { proposePostRunMemory, onMemoryHarvest };`)(...Object.values(deps)) as any
}

try {
  store.settings = { ...defaultSettings(), providers: [provider] }
  store.projects = [project]; store.projectMemory = [active, hidden, beta]; store.projectMemoryCandidates = []
  store.threads = []; store.missions = []; store.rooms = []; store.connectorEvents = []; store.skills = []; store.commands = []; store.suggestions = []
  assert.match(renderProjectMemoryBucket(project), /SYNTHETIC_USER_CHARTER_CONTROL/)
  assert.match(renderProjectMemoryBucket(project), /SYNTHETIC_ACTIVE_ALPHA/)
  assert.doesNotMatch(renderProjectMemoryBucket(project), /SYNTHETIC_(?:TRASH|OTHER_PROJECT)/)
  const unreadableTrash = entry('unreadable', '', { trashedAt: 0, sourceType: 'file', sourceRef: 'hidden.txt' })
  Object.defineProperty(unreadableTrash, 'text', { get: () => { throw new Error('Trash body was read') } })
  assert.deepEqual(projectMemorySourceSnapshots(project, [unreadableTrash]), {}, 'source warnings do not inspect Trash')

  for (const [kind, route] of [['chat', 'native-chat'], ['task', 'native-task'], ['task', 'room-task'], ['task', 'mission-task']] as const) {
    const thread = owner(kind, route)
    store.threads = [thread]
    store.projectMemory = [active, hidden, beta]
    const ctx = runtime.fullToolContext({ thread, provider, model: 'synthetic' })
    assert.deepEqual(ctx.projectMemory.map((memory: ProjectMemoryEntry) => memory.id), [active.id])
    store.projectMemory = [{ ...active, trashedAt: 30 }, hidden, beta]
    assert.equal(ctx.recallProjectMemory().length, 0, `${route} re-reads Trash after context capture`)
    const duringTrash = await executeTool('retrieval_search', { query: 'SYNTHETIC', kinds: ['summary'] }, { cwd: home, currentThread: thread, ...ctx })
    assert.doesNotMatch(duringTrash.output, /SYNTHETIC_(?:ACTIVE_ALPHA|TRASH|OTHER_PROJECT)/)
    store.projectMemory = [{ ...active, trashedAt: null }, hidden, beta]
    const restored = await executeTool('retrieval_search', { query: 'SYNTHETIC', kinds: ['summary'] }, { cwd: home, currentThread: thread, ...ctx })
    assert.match(restored.output, /SYNTHETIC_ACTIVE_ALPHA/)
    assert.doesNotMatch(restored.output, /SYNTHETIC_(?:TRASH|OTHER_PROJECT)/)
    thread.memoryPolicy = { useExisting: false, contribute: false }
    assert.equal(ctx.recallProjectMemory().length, 0, 'a captured callback cannot widen a later saved choice')
    thread.memoryPolicy = { useExisting: true, contribute: true }
    store.threads = [{ ...thread }]
    assert.equal(ctx.recallProjectMemory().length, 0, 'owner/profile replacement invalidates an older callback even with matching ids')
  }
  const thread = owner('chat', 'policy-control')
  store.threads = [thread]; store.projectMemory = [active, hidden, beta]
  assert.equal(runtime.fullToolContext({ thread, provider: { ...provider, kind: 'cloud' }, model: 'synthetic' }).recallProjectMemory().length, 0)
  thread.incognito = true
  thread.memoryPolicy = { useExisting: false, contribute: false }
  assert.equal(runtime.fullToolContext({ thread, provider, model: 'synthetic' }).recallProjectMemory().length, 0, 'private chat memory choices also narrow the live reader')
  thread.memoryPolicy = { useExisting: true, contribute: true }
  assert.equal(runtime.fullToolContext({ thread, provider, model: 'synthetic' }).recallProjectMemory().length, 1, 'private chat preserves its explicit existing recall choice')
  thread.incognito = false
  const child = owner('task', 'owned-child')
  thread.memoryPolicy = { useExisting: false, contribute: false }
  assert.equal(runtime.fullToolContext({ thread: child, provider, model: 'synthetic', memoryOwner: { id: thread.id, revision: 0 } }).recallProjectMemory().length, 0, 'a child cannot widen its host-owned parent policy')
  thread.memoryPolicy = { useExisting: true, contribute: true }
  thread.bot = { executionTarget: 'vm' } as any
  assert.equal(runtime.fullToolContext({ thread, provider, model: 'synthetic' }).recallProjectMemory().length, 0)
  thread.bot = undefined
  const beforeProfileReplacement = runtime.fullToolContext({ thread, provider, model: 'synthetic' })
  store.projects = [{ ...project }]
  assert.equal(beforeProfileReplacement.recallProjectMemory().length, 0)
  store.projects = [project]

  const fact = 'I always prefer synthetic lavender examples.'
  const candidateOwner = owner('task', 'owner')
  candidateOwner.timeline = [{ id: 'turn-1', kind: 'user', memorySource: 'user', text: fact }]
  store.threads = [candidateOwner]
  store.projectMemory = [entry('deleted-source', '', { trashedAt: 10, sourceThreadId: candidateOwner.id, sourceTurn: 1 })]
  Object.defineProperty(store.projectMemory[0], 'text', { get: () => { throw new Error('Trash body entered automatic dedup') } })
  const stalePending = createProjectMemoryCandidate({ id: 'pending-deleted-target', projectId: project.id, targetEntryId: 'deleted-source', text: fact })
  Object.defineProperty(stalePending, 'text', { get: () => { throw new Error('Pending Trash target body entered automatic dedup') } })
  store.projectMemoryCandidates = [stalePending]
  const callbacks = await productionMemoryCallbacks(candidateOwner)
  callbacks.proposePostRunMemory(candidateOwner, candidateOwner.memoryPolicy, 0, new AbortController().signal)
  callbacks.onMemoryHarvest([fact])
  assert.deepEqual(store.projectMemoryCandidates, [stalePending], 'exact deleted-origin turn cannot replay through post-run or compaction; a pending Trash target cannot enter dedup')
  store.projectMemoryCandidates = []
  candidateOwner.timeline.push({ id: 'turn-2', kind: 'user', memorySource: 'user', text: 'I always prefer synthetic copper controls.' })
  callbacks.proposePostRunMemory(candidateOwner, candidateOwner.memoryPolicy, 0, new AbortController().signal)
  assert.deepEqual(store.projectMemoryCandidates.map((candidate) => [candidate.text, candidate.sourceTurn]), [['I always prefer synthetic copper controls.', 2]], 'later unrelated turns remain eligible while the deleted originating turn stays excluded')
  callbacks.onMemoryHarvest([fact, 'I always prefer synthetic copper controls.'])
  assert.ok(store.projectMemoryCandidates.every((candidate) => candidate.text !== fact && candidate.sourceTurn === 2), 'later compaction cannot relabel the deleted origin as a new fact')
  candidateOwner.timeline.push({ id: 'turn-3', kind: 'user', memorySource: 'user', text: fact })
  assert.equal(projectMemoryAutoCandidateTurn(projectMemoryAutoMessages(store.projectMemory, candidateOwner), fact), 3, 'an explicit later user statement remains eligible for review')
  assert.equal(projectMemoryAutoSourceSuppressed(store.projectMemory, { projectId: 'synthetic-beta', sourceThreadId: candidateOwner.id, sourceTurn: 1 }), false)
  assert.equal(projectMemoryAutoSourceSuppressed(store.projectMemory, { projectId: project.id, sourceThreadId: candidateOwner.id }), false, 'missing origin turn cannot mute an entire source')
  const connectorTrash = entry('connector-trash', '', { trashedAt: 10, sourceType: 'connector', sourceRef: 'connector:synthetic-event' })
  assert.equal(projectMemoryAutoSourceSuppressed([connectorTrash], { projectId: project.id, sourceRef: connectorTrash.sourceRef }), true)
  for (const sourceRef of ['', null, 'connector:later-event', 'notes.md']) assert.equal(projectMemoryAutoSourceSuppressed([connectorTrash], { projectId: project.id, sourceRef }), false)
  assert.equal(projectMemoryAutoSourceSuppressed([{ ...connectorTrash, sourceRef: 'notes.md' }], { projectId: project.id, sourceRef: 'notes.md' }), false, 'file names cannot suppress new revisions')
  console.log('Production host recall, renderer, chat/task/room/mission scopes, live Trash/Restore, profile isolation and exact-origin automatic replay checks passed')
} finally {
  Object.assign(store, saved)
  await rm(home, { recursive: true, force: true })
}
