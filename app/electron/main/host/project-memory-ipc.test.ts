import assert from 'node:assert/strict'
import { promises as fs, mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import type { Project, ProjectMemoryCandidate, ProjectMemoryEntry } from '../../agent/types'
import { changeProjectMemoryTrash, createProjectMemoryCandidate, formatProjectMemoryEntryForPrompt, projectMemoryCandidateDiff, rollbackProjectMemoryEntry, scopeProjectMemoryEntries } from '../../agent/project-memory'

const home = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-memory-trash-'))
process.env.VONTRA_HOME = home
process.env.VONTRA_SECURITY_ROLE = 'owner'
process.env.VONTRA_AUDIT_ACTOR = 'synthetic-memory-test'
const persistence = await import('../../store/project-memory')
const { registerProjectMemoryIpc } = await import('./project-memory-ipc')
const { requireSecurityRole, flushAuditEvents } = await import('./audit')
const { closeSessionSearchIndexes } = await import('../../store')
const file = path.join(home, 'state', 'project-memory.json')
const project = (id: string): Project => ({ id, name: 'Synthetic project', cwd: '/synthetic', memory: '', team: '', createdAt: 10, updatedAt: 10 })
const entry: ProjectMemoryEntry = { id: 'entry-a', projectId: 'project-a', text: 'FICTIONAL-MEMORY-BODY-ALPHA', kind: 'decision', sourceType: 'file', sourceRef: 'synthetic.md', sourceIds: ['synthetic.md'], confidence: 'high', privacyScope: 'project', owner: 'fictional-owner', sourceThreadId: 'thread-a', sourceTurn: 3, lastVerifiedAt: 20, createdAt: 10, updatedAt: 20, revisions: [{ text: 'FICTIONAL-OLD-REVISION', kind: 'summary', confidence: 'low', privacyScope: 'project', sourceRef: null, replacedAt: 20, replacedBy: 'fictional-owner' }] }
const linked = createProjectMemoryCandidate({ projectId: entry.projectId, targetEntryId: entry.id, text: 'FICTIONAL-REVIEW-BODY' }, 30, 'candidate-a')
const store = { projects: [project('project-a'), project('project-b')], projectMemory: [] as ProjectMemoryEntry[], projectMemoryCandidates: [] as ProjectMemoryCandidate[] }
const handlers = new Map<string, (...args: any[]) => any>()
registerProjectMemoryIpc({ store, handle: ((channel: string, handler: any) => handlers.set(channel, handler)) as any, requireSecurityRole, auditEvent: async () => {}, auditPreview: (value) => value })
const invoke = (channel: string, ...args: any[]) => handlers.get(channel)!({}, ...args)
let checks = 0
const check = async (name: string, action: () => Promise<void>) => { await action(); checks++; console.log(`PASS ${name}`) }
const originalOpen = fs.open
const originalRename = fs.rename
try {
  const { writeProtectedJson } = await import('../../store/write-infra')
  await writeProtectedJson(file, [entry])
  await writeProtectedJson(path.join(home, 'state', 'project-memory-candidates.json'), [linked])
  store.projectMemory = await persistence.loadProjectMemory()
  store.projectMemoryCandidates = await persistence.loadProjectMemoryCandidates()
  await check('legacy entries migrate without changing provenance or revision ledger and remain encrypted', async () => {
    assert.equal(store.projectMemory[0].stateRevision, 1)
    assert.equal(store.projectMemory[0].trashedAt, null)
    assert.deepEqual(store.projectMemoryCandidates, JSON.parse(JSON.stringify([linked])))
    assert.equal((await fs.readdir(path.dirname(file))).includes('project-memory-candidates.json'), false)
    assert.deepEqual(store.projectMemory[0].revisions, entry.revisions)
    assert.ok(!readFileSync(file, 'utf8').includes(entry.text))
    assert.equal(statSync(file).mode & 0o777, 0o600)
    assert.equal(statSync(path.dirname(file)).mode & 0o777, 0o700)
  })
  await check('viewer may inspect but cannot change Trash and active project scopes never cross', async () => {
    process.env.VONTRA_SECURITY_ROLE = 'viewer'
    assert.equal((await invoke('memory:get', 'project-a')).length, 1)
    assert.deepEqual(await invoke('memory:get', 'project-b'), [])
    await assert.rejects(() => invoke('memory:trash', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 1 }), /requires operator/)
    process.env.VONTRA_SECURITY_ROLE = 'owner'
  })
  await check('Trash commits durably, removes prompt/candidate visibility, and preserves content history', async () => {
    const trashed = await invoke('memory:trash', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 1 })
    assert.equal(trashed.stateRevision, 2); assert.ok(trashed.trashedAt > 0)
    assert.equal(trashed.updatedAt, entry.updatedAt)
    assert.deepEqual(trashed.revisions, entry.revisions)
    assert.equal(trashed.sourceThreadId, entry.sourceThreadId)
    assert.deepEqual(await invoke('memory:get', 'project-a'), [])
    assert.deepEqual(await invoke('memory:candidates', 'project-a'), [])
    assert.equal(formatProjectMemoryEntryForPrompt(trashed), '')
    assert.deepEqual(scopeProjectMemoryEntries(await persistence.loadProjectMemory(), 'project-a'), [])
    assert.equal((await invoke('memory:trashList', 'project-a')).length, 1)
    assert.deepEqual(await invoke('memory:trashList', 'project-b'), [])
    await assert.rejects(() => invoke('memory:save', { id: 'entry-a', projectId: 'project-a', text: 'replacement' }), /Restore memory/)
    await assert.rejects(() => invoke('memory:approveCandidate', 'candidate-a'), /Restore memory/)
    await assert.rejects(() => invoke('memory:candidateDiff', 'candidate-a'), /Restore memory/)
    await assert.rejects(() => invoke('memory:rollbackEntry', 'entry-a'), /Restore memory/)
    assert.throws(() => projectMemoryCandidateDiff([trashed], linked), /Restore memory/)
    assert.throws(() => rollbackProjectMemoryEntry([trashed], [linked], 'entry-a'), /Restore memory/)
  })
  await check('wrong project, stale/missing CAS, forged host state, and unknown targets fail without writes', async () => {
    const before = readFileSync(file)
    await assert.rejects(() => invoke('memory:restore', { projectId: 'project-b', entryId: 'entry-a', expectedRevision: 2 }), /not found/)
    await assert.rejects(() => invoke('memory:restore', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 1 }), /changed/)
    await assert.rejects(() => invoke('memory:restore', { projectId: 'project-a', entryId: 'entry-a' }), /expectedRevision/)
    await assert.rejects(() => invoke('memory:save', { id: 'entry-a', trashedAt: null }), /host-owned/)
    await assert.rejects(() => invoke('memory:save', { id: 'unknown', projectId: 'project-a', text: 'unknown edit' }), /not found/)
    await assert.rejects(() => invoke('memory:propose', { projectId: 'project-b', targetEntryId: 'entry-a', text: 'retarget' }), /unavailable/)
    await assert.rejects(() => invoke('memory:save', { id: 'entry-a', projectId: 'project-b', text: 'retarget' }), /different project/)
    assert.deepEqual(readFileSync(file), before)
  })
  await check('real disk rename failure leaves live state and durable ciphertext unchanged', async () => {
    const before = readFileSync(file); const live = JSON.stringify(store)
    fs.rename = async (from, to) => { if (String(to) === file) throw new Error('Synthetic atomic rename failure'); return originalRename(from, to) }
    await assert.rejects(() => invoke('memory:restore', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 2 }), /atomic rename failure/)
    fs.rename = originalRename
    assert.equal(JSON.stringify(store), live)
    assert.deepEqual(readFileSync(file), before)
    assert.equal((await fs.readdir(path.dirname(file))).some((name) => name.includes('.tmp-')), false)
  })
  await check('concurrent restore CAS has one durable winner and a fresh process reads it', async () => {
    const result = await Promise.allSettled([invoke('memory:restore', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 2 }), invoke('memory:restore', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 2 })])
    assert.deepEqual(result.map((item) => item.status).sort(), ['fulfilled', 'rejected'])
    const active = (await invoke('memory:get', 'project-a'))[0]
    assert.equal(active.stateRevision, 3); assert.equal(active.trashedAt, null)
    assert.deepEqual(active.revisions, entry.revisions); assert.equal(active.updatedAt, entry.updatedAt)
    const restart = path.join(home, 'restart-fixture.ts')
    writeFileSync(restart, `import assert from 'node:assert/strict'; import { loadProjectMemory } from ${JSON.stringify(path.resolve(__dirname, '../../store/project-memory.ts'))}; const entries=await loadProjectMemory(); assert.equal(entries[0].stateRevision,3); assert.equal(entries[0].trashedAt,null); console.log('Fresh protected memory read passed');`)
    const stdout = execFileSync(process.execPath, [path.resolve('node_modules/jiti/lib/jiti-cli.mjs'), restart], { env: process.env, encoding: 'utf8' })
    assert.match(stdout, /Fresh protected memory read passed/)
  })
  for (const mode of ['role', 'project', 'profile', 'purge'] as const) await check(`${mode} changes during disk flush invalidate the transaction before rename`, async () => {
    const before = readFileSync(file); const live = JSON.stringify(store.projectMemory)
    let reach!: () => void; let release!: () => void
    const reached = new Promise<void>((resolve) => { reach = resolve })
    const released = new Promise<void>((resolve) => { release = resolve })
    fs.open = (async (...args: Parameters<typeof fs.open>) => {
      const handle = await originalOpen(...args)
      if (String(args[0]).startsWith(`${file}.tmp-`)) {
        const sync = handle.datasync.bind(handle)
        handle.datasync = async () => { await sync(); reach(); await released }
      }
      return handle
    }) as typeof fs.open
    const pending = invoke('memory:trash', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 3 })
    const rejection = assert.rejects(pending, mode === 'role' ? /requires operator/ : mode === 'project' ? /active local project/ : mode === 'profile' ? /profile changed/ : /being cleared/)
    await reached
    let purging: Promise<void> | undefined
    if (mode === 'role') process.env.VONTRA_SECURITY_ROLE = 'viewer'
    if (mode === 'project') store.projects[0].archivedAt = 200
    if (mode === 'profile') process.env.VONTRA_HOME = path.join(home, 'other-profile')
    if (mode === 'purge') purging = persistence.beginProjectMemoryPurge()
    release(); await rejection; await purging
    fs.open = originalOpen
    process.env.VONTRA_SECURITY_ROLE = 'owner'; process.env.VONTRA_HOME = home; store.projects[0].archivedAt = null
    if (mode === 'purge') persistence.endProjectMemoryPurge()
    assert.equal(JSON.stringify(store.projectMemory), live)
    assert.deepEqual(readFileSync(file), before)
  })
  await check('stale general saves cannot resurrect Trash or replay approved candidate content', async () => {
    const staleEntries = JSON.parse(JSON.stringify(store.projectMemory))
    const staleCandidates = JSON.parse(JSON.stringify(store.projectMemoryCandidates))
    await invoke('memory:trash', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 3 })
    await assert.rejects(() => persistence.saveProjectMemory(staleEntries), /Stale project memory snapshot/)
    await persistence.saveProjectMemoryCandidates(staleCandidates)
    assert.ok((await persistence.loadProjectMemory())[0].trashedAt)
    assert.deepEqual(await invoke('memory:get', 'project-a'), [])
  })
  await check('permanent Forget removes Trash, revisions, candidate payloads, and recovery copies with scoped CAS', async () => {
    const staleCandidates = JSON.parse(JSON.stringify(store.projectMemoryCandidates))
    await assert.rejects(() => invoke('memory:delete', 'entry-a', { projectId: 'project-b', expectedRevision: 4 }), /changed/)
    writeFileSync(`${file}.synthetic.bak`, readFileSync(file))
    await invoke('memory:delete', 'entry-a', { projectId: 'project-a', expectedRevision: 4 })
    assert.deepEqual(store.projectMemory, []); assert.deepEqual(store.projectMemoryCandidates, [])
    assert.deepEqual((await persistence.loadProjectMemoryState()).entries, [])
    assert.deepEqual((await persistence.loadProjectMemoryState()).candidates, [])
    await persistence.saveProjectMemoryCandidates(staleCandidates)
    assert.deepEqual((await persistence.loadProjectMemoryState()).candidates, [])
    store.projectMemoryCandidates = staleCandidates
    await assert.rejects(() => invoke('memory:approveCandidate', 'candidate-a'), /candidate not found/)
    store.projectMemoryCandidates = []
    assert.deepEqual((await persistence.loadProjectMemoryState()).candidates, [])
    assert.equal((await fs.readdir(path.dirname(file))).some((name) => name.endsWith('.bak')), false)
    await assert.rejects(() => invoke('memory:restore', { projectId: 'project-a', entryId: 'entry-a', expectedRevision: 4 }), /not found/)
  })
  await check('stale unlinked create proposals cannot return after approval and permanent Forget', async () => {
    const proposal = await invoke('memory:propose', { projectId: 'project-a', text: 'FICTIONAL-CREATED-CANDIDATE' })
    const oldPending = JSON.parse(JSON.stringify(store.projectMemoryCandidates))
    const created = await invoke('memory:approveCandidate', proposal.id)
    await invoke('memory:delete', created.id, { projectId: created.projectId, expectedRevision: created.stateRevision })
    await persistence.saveProjectMemoryCandidates(oldPending)
    assert.deepEqual((await persistence.loadProjectMemoryState()).candidates, [])
    store.projectMemoryCandidates = oldPending
    await assert.rejects(() => invoke('memory:approveCandidate', proposal.id), /candidate not found/)
    store.projectMemoryCandidates = []
    assert.deepEqual((await persistence.loadProjectMemoryState()).entries, [])
  })
  await check('forget replay protection preserves unrelated project pending proposals', async () => {
    const unrelated = createProjectMemoryCandidate({ projectId: 'project-b', text: 'FICTIONAL-UNRELATED-PENDING' }, 25, 'unrelated-b')
    await persistence.saveProjectMemoryCandidates([unrelated])
    assert.deepEqual((await persistence.loadProjectMemoryState()).candidates.map((candidate) => candidate.id), ['unrelated-b'])
  })
  await check('invalid persisted Trash metadata fails closed and keeps original encrypted state recoverable', async () => {
    const { writeProtectedJson } = await import('../../store/write-infra')
    await writeProtectedJson(file, { version: 1, entries: [{ ...entry, trashedAt: 0 }], candidates: [] })
    const before = readFileSync(file)
    await assert.rejects(() => persistence.loadProjectMemoryState(), /Invalid memory Trash state/)
    assert.deepEqual(readFileSync(file), before)
    assert.equal(formatProjectMemoryEntryForPrompt({ ...entry, trashedAt: 0 }), '')
    assert.throws(() => changeProjectMemoryTrash([{ ...entry, trashedAt: -1 }], { projectId: entry.projectId, entryId: entry.id, expectedRevision: 1 }, false), /Invalid memory Trash state/)
    await persistence.beginProjectMemoryPurge()
    await persistence.purgeProjectMemoryState()
    persistence.endProjectMemoryPurge()
    assert.deepEqual((await persistence.loadProjectMemoryState()).entries, [], 'Explicit security purge can clear quarantined malformed Trash')
    assert.equal((await fs.readdir(path.dirname(file))).some((name) => name.endsWith('.bak')), false)
  })
  console.log(`Project memory host and protected persistence passed ${checks} cases`)
} finally {
  fs.open = originalOpen; fs.rename = originalRename
  persistence.endProjectMemoryPurge()
  await flushAuditEvents(); closeSessionSearchIndexes()
  await fs.rm(home, { recursive: true, force: true })
}
