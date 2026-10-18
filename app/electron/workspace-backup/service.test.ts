import assert from 'node:assert/strict'
import type { WorkspaceBackupData, WorkspaceBackupSelection } from '../../shared/workspace-backup'
import { backupDigest, parseArchive, type BackupSnapshot } from './archive'
import { decryptWorkspaceBackup, encryptWorkspaceBackup, MAX_BACKUP_BYTES } from './crypto'
import { createWorkspaceBackupService, WorkspaceRestoreError, type WorkspaceBackupRepository, type WorkspaceRestorePlan } from './service'

const password = 'synthetic backup passphrase'
const original = Buffer.from('Synthetic managed original document bytes')
const parsed = Buffer.from('Synthetic full parsed document content')
const originalId = backupDigest(original), parsedId = backupDigest(parsed)
const selection: WorkspaceBackupSelection = { settings: true, conversationIds: ['chat', 'private'], projectIds: ['project'], botIds: ['bot'], drafts: true, attachments: true }
function data(): WorkspaceBackupData {
  return {
    settings: { theme: 'dark', apiKey: 'synthetic-secret-slot', profileKey: 'synthetic-profile-key', providers: [{ id: 'local', apiKey: 'synthetic-provider-secret' }] },
    conversations: [{ id: 'chat', title: 'Synthetic chat', kind: 'chat', createdAt: 1, updatedAt: 2, cwd: '/synthetic/source-folder', status: 'running', phoneLocalOnly: true, googleLocalOnly: true,
      apiMessages: [{ role: 'user', content: 'Synthetic question' }, { role: 'assistant', content: 'Synthetic answer' }],
      timeline: [{ id: 'message', kind: 'user', text: 'Synthetic question', appAction: { kind: 'create_project' }, question: { pending: true } }], codexResumeCursor: { threadId: 'active-external-session' } },
      { id: 'private', title: 'Memory-only synthetic chat', incognito: true, apiMessages: [], timeline: [] }],
    projects: [{ id: 'project', name: 'Synthetic project', cwd: '/synthetic/source-folder', memory: 'Synthetic project notes' }],
    bots: [{ id: 'bot', name: 'Synthetic bot', instructions: 'Synthetic helper', executionTarget: 'vm', apiKey: 'synthetic-bot-secret' }],
    drafts: [{ id: 'draft', scope: { surface: 'chat', conversationId: 'chat', hostId: null, incognito: false }, snapshot: { revision: 7, text: 'Unsent synthetic draft', attachments: [], updatedAt: 2 } },
      { id: 'private-draft', scope: { incognito: true }, snapshot: { text: 'Memory-only draft' } }],
    attachments: [{ id: 'file', name: 'Synthetic document (1).txt', mediaType: 'text/plain', originalBlobId: originalId, parsedBlobId: parsedId },
      { id: 'missing', name: 'Missing.pdf', mediaType: 'application/pdf', originalBlobId: null, parsedBlobId: null }],
  }
}
interface Stage { plan: WorkspaceRestorePlan; before: BackupSnapshot }
class Repository implements WorkspaceBackupRepository<Stage> {
  state = data()
  blobs = new Map<string, Uint8Array>([[originalId, Buffer.from(original)], [parsedId, Buffer.from(parsed)]])
  serial = 1
  snapshots = 0
  stages = 0
  rollbacks = 0
  discards = 0
  failCommit = false
  failRollback = false
  duringCommit: (() => void) | undefined
  async revision() { return `synthetic-profile:${this.serial}` }
  async snapshot(): Promise<BackupSnapshot> { this.snapshots++; return { revision: await this.revision(), data: structuredClone(this.state), blobs: new Map([...this.blobs].map(([id, bytes]) => [id, Buffer.from(bytes)])) } }
  async stage(plan: WorkspaceRestorePlan): Promise<Stage> { this.stages++; return { plan, before: await this.snapshot() } }
  async commit(stage: Stage, revision: string) {
    assert.equal(await this.revision(), revision)
    const { data: incoming, mode } = stage.plan
    this.state.settings = mode === 'merge' ? { ...incoming.settings, ...this.state.settings } : incoming.settings
    for (const collection of ['conversations', 'projects', 'bots', 'drafts', 'attachments'] as const) {
      const existing = new Map(this.state[collection].map((item) => [String(item.id), item]))
      for (const item of incoming[collection]) if (mode === 'replace' || !existing.has(String(item.id))) existing.set(String(item.id), item)
      this.state[collection] = [...existing.values()] as any
      if (this.failCommit) throw new Error('Synthetic midpoint apply failure')
    }
    for (const [id, bytes] of stage.plan.blobs) this.blobs.set(id, Buffer.from(bytes))
    this.serial++
    this.duringCommit?.()
  }
  async rollback(stage: Stage) {
    this.rollbacks++
    if (this.failRollback) throw new Error('Synthetic rollback unavailable')
    this.state = structuredClone(stage.before.data)
    this.blobs = new Map([...stage.before.blobs].map(([id, bytes]) => [id, Buffer.from(bytes)]))
    this.serial++
  }
  async discard() { this.discards++ }
}

const source = new Repository()
const exportService = createWorkspaceBackupService({ repository: source })
const exportPreview = await exportService.previewExport(selection)
assert.equal(exportPreview.counts.conversations, 1)
assert.equal(exportPreview.counts.drafts, 1)
assert.equal(exportPreview.counts.blobs, 2)
assert.ok(exportPreview.warnings.some((warning) => warning.includes('Missing.pdf') && warning.includes('original')))
assert.ok(exportPreview.warnings.some((warning) => warning.includes('incognito')))
const encrypted = await exportService.exportEncrypted({ approvalId: exportPreview.approvalId, password })
assert.equal(Buffer.from(encrypted).includes(Buffer.from('Synthetic question')), false)
await assert.rejects(exportService.exportEncrypted({ approvalId: exportPreview.approvalId, password }), /unavailable/)
const decrypted = await decryptWorkspaceBackup(encrypted, password)
const decoded = parseArchive(decrypted)
const visible = JSON.stringify(decoded.archive)
for (const excluded of ['synthetic-secret-slot', 'synthetic-profile-key', 'synthetic-provider-secret', 'synthetic-bot-secret', 'active-external-session', 'appAction', 'Memory-only']) assert.equal(visible.includes(excluded), false)
assert.equal(decoded.archive.data.conversations[0].status, 'idle')
assert.equal(decoded.archive.data.conversations[0].cwd, '')
assert.equal(decoded.archive.data.conversations[0].phoneLocalOnly, true)
assert.equal(decoded.archive.data.conversations[0].googleLocalOnly, true)
assert.equal(decoded.archive.data.bots[0].paused, true)
assert.equal(decoded.archive.data.bots[0].executionTarget, 'vm', 'restoring a paused VM bot must not widen confinement to the host workspace')
assert.deepEqual(decoded.blobs.get(originalId), original)
assert.deepEqual(decoded.blobs.get(parsedId), parsed)

const destination = new Repository()
destination.state = { settings: {}, conversations: [], projects: [], bots: [], drafts: [], attachments: [] }
destination.blobs = new Map()
const service = createWorkspaceBackupService({ repository: destination })
const restore = await service.previewRestore({ fileId: 'synthetic-file', bytes: encrypted, password, mode: 'replace' })
assert.equal(destination.stages, 0, 'preview never stages or mutates the working profile')
const restored = await service.applyRestore({ approvalId: restore.approvalId, fileId: 'synthetic-file', bytes: encrypted })
assert.equal(restored.restored, true)
assert.equal(destination.state.drafts[0].snapshot && (destination.state.drafts[0].snapshot as any).text, 'Unsent synthetic draft')
assert.deepEqual(destination.blobs.get(originalId), original, 'complete managed original bytes survive restore and approval cleanup')
assert.deepEqual(destination.blobs.get(parsedId), parsed)
await assert.rejects(service.applyRestore({ approvalId: restore.approvalId, fileId: 'synthetic-file', bytes: encrypted }), /unavailable/)

const snapshotReads = destination.snapshots
await assert.rejects(service.previewRestore({ fileId: 'synthetic-file', bytes: encrypted, password: 'wrong synthetic password', mode: 'merge' }), /password or integrity/)
const damaged = Uint8Array.from(encrypted); damaged[damaged.length - 1] ^= 1
await assert.rejects(service.previewRestore({ fileId: 'synthetic-file', bytes: damaged, password, mode: 'merge' }), /password or integrity/)
await assert.rejects(service.previewRestore({ fileId: 'synthetic-file', bytes: new Uint8Array(MAX_BACKUP_BYTES + 1), password, mode: 'merge' }), /oversized/)
assert.equal(destination.snapshots, snapshotReads, 'invalid archives cannot access or mutate the working profile')

const merge = await service.previewRestore({ fileId: 'synthetic-file', bytes: encrypted, password, mode: 'merge' })
assert.ok(merge.conflicts.some((conflict) => conflict.collection === 'drafts' && conflict.action === 'keep-existing'))
destination.state.drafts[0] = { ...destination.state.drafts[0], snapshot: { text: 'Newest unsent draft' } }
destination.serial++
await assert.rejects(service.applyRestore({ approvalId: merge.approvalId, fileId: 'synthetic-file', bytes: encrypted }), /drafts changed/)
assert.equal((destination.state.drafts[0].snapshot as any).text, 'Newest unsent draft')
const changedFile = await service.previewRestore({ fileId: 'synthetic-file', bytes: encrypted, password, mode: 'merge' })
await assert.rejects(service.applyRestore({ approvalId: changedFile.approvalId, fileId: 'another-file', bytes: encrypted }), /file changed/)

const rollback = await service.previewRestore({ fileId: 'synthetic-file', bytes: encrypted, password, mode: 'replace' })
const beforeFailure = structuredClone(destination.state)
destination.failCommit = true
await assert.rejects(service.applyRestore({ approvalId: rollback.approvalId, fileId: 'synthetic-file', bytes: encrypted }), (error: unknown) => error instanceof WorkspaceRestoreError && error.rolledBack)
assert.deepEqual(destination.state, beforeFailure, 'partial commit rollback restores conversations, settings and the newest unsent draft')
assert.equal(destination.rollbacks, 1)
destination.failCommit = false

const abort = new AbortController()
const cancellation = await service.previewRestore({ fileId: 'synthetic-file', bytes: encrypted, password, mode: 'replace' })
destination.duringCommit = () => abort.abort()
const cancelled = await service.applyRestore({ approvalId: cancellation.approvalId, fileId: 'synthetic-file', bytes: encrypted, signal: abort.signal })
assert.equal(cancelled.cancelled, true)
assert.deepEqual(destination.state, beforeFailure, 'even an uncooperative commit is rolled back after cancellation')
destination.duringCommit = undefined

const recovery = await service.previewRestore({ fileId: 'synthetic-file', bytes: encrypted, password, mode: 'replace' })
destination.failCommit = true; destination.failRollback = true
const priorDiscards = destination.discards
await assert.rejects(service.applyRestore({ approvalId: recovery.approvalId, fileId: 'synthetic-file', bytes: encrypted }), (error: unknown) => error instanceof WorkspaceRestoreError && !error.rolledBack)
assert.equal(destination.discards, priorDiscards, 'failed rollback retains its recovery checkpoint')

const malformed = structuredClone(decoded.archive)
malformed.data.attachments[0].name = '../escape.txt'
const hostile = await encryptWorkspaceBackup(Buffer.from(JSON.stringify(malformed)), password)
const priorStages = source.stages
await assert.rejects(exportService.previewRestore({ fileId: 'malicious', bytes: hostile, password, mode: 'replace' }), /Invalid|unsafe|unsupported/)
assert.equal(source.stages, priorStages)

let time = 1
const expiring = createWorkspaceBackupService({ repository: source, now: () => time })
const old = await expiring.previewExport(selection)
time += 10 * 60_000 + 1
await assert.rejects(expiring.exportEncrypted({ approvalId: old.approvalId, password }), /expired/)
const stale = await exportService.previewExport(selection)
source.serial++
await assert.rejects(exportService.exportEncrypted({ approvalId: stale.approvalId, password }), /Workspace changed/)
decrypted.fill(0)
exportService.dispose(); service.dispose(); expiring.dispose()
console.log('Encrypted workspace backup roundtrip, exclusion, bounds, approvals, draft safety and rollback tests passed')
