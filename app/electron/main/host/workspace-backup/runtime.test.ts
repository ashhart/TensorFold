import assert from 'node:assert/strict'
import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { Settings, Thread } from '../../../agent/types'
import { createWorkspaceBackupRepository, type BackupHostState } from './repository'
import { createWorkspaceBackupService } from '../../../workspace-backup/service'
import { decryptWorkspaceBackup } from '../../../workspace-backup/crypto'
import { parseArchive } from '../../../workspace-backup/archive'
import { recoverWorkspaceBackupCheckpoints, writeWorkspaceRecoveryMarker } from './recovery'
import { chosenDestination, registerWorkspaceBackupRuntime, readChosenBackup, writeChosenBackup } from './runtime'

const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-backup-host-')))
const key = Buffer.alloc(32, 37)
async function writeProtected(file: string, value: unknown) {
  const iv = randomBytes(12), cipher = createCipheriv('aes-256-gcm', key, iv)
  const ciphertext = Buffer.concat([cipher.update(JSON.stringify(value), 'utf8'), cipher.final()])
  await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 })
  await fs.writeFile(file, Buffer.concat([iv, cipher.getAuthTag(), ciphertext]), { mode: 0o600 })
}
async function readProtected<T>(file: string, fallback: T, _migrate?: boolean, encrypted?: boolean): Promise<T> {
  assert.equal(encrypted, true)
  const bytes = await fs.readFile(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
  if (!bytes) return fallback
  const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12)); decipher.setAuthTag(bytes.subarray(12, 28))
  return JSON.parse(Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString('utf8')) as T
}
const image = 'data:image/png;base64,' + Buffer.from('SYNTHETIC_RETAINED_IMAGE').toString('base64')
const conversation: Thread = { id: 'chat', title: 'Synthetic host chat', kind: 'chat', cwd: '', providerId: null, model: null, createdAt: 1, updatedAt: 2, status: 'idle',
  timeline: [{ id: 'user', kind: 'user', text: 'Synthetic prompt', attachments: [{ id: 'image', kind: 'image', name: 'Synthetic.png', mime: 'image/png', dataUrl: image }, { id: 'document', kind: 'document', name: 'Synthetic.txt', mime: 'text/plain', text: 'Complete synthetic parsed file text' }] }],
  apiMessages: [{ role: 'user', content: [{ type: 'text', text: 'Synthetic prompt' }, { type: 'image_url', image_url: { url: image } }] }] }
let state: BackupHostState = { settings: { theme: 'dark', agentPresets: [], apiKey: 'synthetic-private-setting' } as unknown as Settings, threads: [conversation], projects: [] }
let owner = true, paused = false, leased = false, flushes = 0, published = 0, fail = false, failPermanently = false
const draftId = createHash('sha256').update('synthetic-composer-scope').digest('hex')
const draftFile = path.join(root, 'state', 'composer-drafts', `${draftId}.json`)
await writeProtected(draftFile, { revision: 2, text: 'Newest unsent synthetic draft', attachments: [], updatedAt: 3 })
const deps = { profileRoot: () => root, getState: () => state, setState: (next: BackupHostState) => { state = next },
  persist: async (next: BackupHostState) => {
    await writeProtected(path.join(root, 'synthetic-settings.json'), next.settings)
    if (fail || failPermanently) { fail = false; throw new Error('Synthetic partial persist failure') }
    await writeProtected(path.join(root, 'synthetic-threads.json'), next.threads.filter(thread => !thread.incognito))
    await writeProtected(path.join(root, 'synthetic-projects.json'), next.projects)
  }, readProtectedJson: readProtected, writeProtectedJson: writeProtected,
  validateOwner: () => { if (!owner) throw new Error('Owner required') } }

try {
  const repository = createWorkspaceBackupRepository(deps)
  const first = await repository.snapshot()
  assert.equal(first.data.drafts[0].id, draftId)
  assert.equal(first.data.attachments.length, 2)
  assert.equal(first.blobs.size, 2)
  assert.equal(conversation.timeline[0].attachments?.[0].dataUrl, image, 'snapshot never mutates live conversation attachments')
  conversation.timeline[0].attachments![0].dataUrl = 'data:image/png;base64,' + Buffer.from('DIFFERENT_SYNTHETIC_IMAGE').toString('base64')
  assert.notEqual(await repository.revision(), first.revision, 'retained attachment bytes participate in revision binding')
  conversation.timeline[0].attachments![0].dataUrl = image

  const core = createWorkspaceBackupService({ repository })
  const preview = await core.previewExport({ settings: true, conversationIds: ['chat'], drafts: true, attachments: true })
  const archive = await core.exportEncrypted({ approvalId: preview.approvalId, password: 'synthetic host password' })
  state.threads = []
  const restore = await core.previewRestore({ fileId: 'synthetic-chosen', bytes: archive, password: 'synthetic host password', mode: 'replace' })
  const result = await core.applyRestore({ approvalId: restore.approvalId, fileId: 'synthetic-chosen', bytes: archive })
  assert.equal(result.restored, true)
  assert.equal(state.threads[0].timeline[0].attachments?.[0].dataUrl, image, 'retained image payload is rehydrated through a scoped manifest')
  assert.equal(state.threads[0].timeline[0].attachments?.[1].text, 'Complete synthetic parsed file text')
  assert.ok((await readProtected<any>(draftFile, null, false, true)).revision > 2, 'restored drafts invalidate delayed pre-restore CAS callbacks')
  const afterRestart = createWorkspaceBackupRepository(deps)
  assert.equal((await afterRestart.snapshot()).blobs.size, 2, 'managed bytes survive a repository restart')
  const rollbackPreview = await core.previewRestore({ fileId: 'synthetic-chosen', bytes: archive, password: 'synthetic host password', mode: 'replace' })
  fail = true
  await assert.rejects(core.applyRestore({ approvalId: rollbackPreview.approvalId, fileId: 'synthetic-chosen', bytes: archive }), /previous workspace.*drafts were restored/)
  assert.equal((await readProtected<any>(draftFile, null, false, true)).text, 'Newest unsent synthetic draft')
  assert.equal((await fs.readFile(draftFile)).includes(Buffer.from('Newest unsent')), false)

  const handlers = new Map<string, (event: any, ...args: any[]) => any>()
  const file = path.join(root, 'synthetic-backup.tfbackup')
  const event = { sender: { once: () => undefined } }
  registerWorkspaceBackupRuntime({ ...deps,
    handle: (channel, listener) => handlers.set(channel, listener), authorize: () => deps.validateOwner(),
    assertIdle: () => {}, acquireLease: async () => { assert.equal(leased, false); leased = true; return { release: () => { leased = false } } },
    flushDrafts: async () => { assert.equal(leased, true); assert.equal(paused, false); flushes++ },
    pauseDraftWrites: async () => { paused = true }, resumeDraftWrites: () => { paused = false },
    chooseOpen: async () => file, chooseSave: async () => file,
    publishRestored: () => { assert.equal(paused, true); published++ },
  })
  const invoke = (channel: string, ...args: unknown[]) => handlers.get(`workspace-backup:${channel}`)!(event, ...args)
  const nativePreview = await invoke('preview-export', { conversationIds: ['chat'], drafts: true, attachments: true })
  assert.equal(paused, false); assert.equal(leased, false)
  const saved = await invoke('export', { approvalId: nativePreview.approvalId, password: 'synthetic native password' })
  assert.equal(saved.saved, true)
  assert.equal((await fs.stat(file)).mode & 0o777, 0o600)
  const chosen = await invoke('preview-restore', { password: 'synthetic native password', mode: 'merge' })
  assert.equal((await invoke('apply-restore', { approvalId: chosen.approvalId })).restored, true)
  assert.equal(published, 1)
  assert.ok(flushes >= 4, 'every preview/export/restore crosses the renderer draft flush barrier')
  const existingDestination = await chosenDestination(file)
  await fs.writeFile(file, 'INTERVENING_SYNTHETIC_FILE')
  await assert.rejects(writeChosenBackup(existingDestination, archive, () => {}), /destination changed/)
  assert.equal(await fs.readFile(file, 'utf8'), 'INTERVENING_SYNTHETIC_FILE', 'an intervening native destination is never overwritten')
  const absent = path.join(root, 'absent.tfbackup'), absentDestination = await chosenDestination(absent)
  await fs.writeFile(absent, 'NEW_SYNTHETIC_FILE')
  await assert.rejects(writeChosenBackup(absentDestination, archive, () => {}), /destination changed/)
  const hardLink = path.join(root, 'hard-linked.tfbackup')
  await fs.link(file, hardLink)
  await assert.rejects(chosenDestination(file), /unlinked/)
  await fs.unlink(hardLink)
  owner = false
  await assert.rejects(invoke('preview-export', {}), /Owner required/)
  await assert.rejects(invoke('list'), /Owner required/)
  owner = true
  const link = path.join(root, 'linked-backup.tfbackup')
  await fs.symlink(file, link)
  await assert.rejects(readChosenBackup(link), /ELOOP|symbolic|link/iu)

  // Simulate process exit after a partial persist, without invoking in-process rollback.
  const parsedBackup = parseArchive(await decryptWorkspaceBackup(archive, 'synthetic host password'))
  const crashRepository = createWorkspaceBackupRepository(deps)
  const beforeCrash = await crashRepository.snapshot()
  const crashData = structuredClone(parsedBackup.archive.data)
  crashData.settings.theme = 'light'
  crashData.conversations[0].title = 'Synthetic replacement after recovery review'
  const plan = { mode: 'replace' as const, expectedRevision: beforeCrash.revision, sourceDigest: 'a'.repeat(64), data: crashData, blobs: parsedBackup.blobs, selection: parsedBackup.archive.selection }
  const crashStage = await crashRepository.stage(plan)
  fail = true
  await assert.rejects(crashRepository.commit(crashStage, beforeCrash.revision), /partial persist/)
  assert.equal((await readProtected<any>(path.join(root, 'synthetic-settings.json'), null, false, true)).theme, 'light')
  const recovered = await recoverWorkspaceBackupCheckpoints(deps)
  assert.equal(recovered.rolledBack, 1)
  assert.equal((await readProtected<any>(path.join(root, 'synthetic-settings.json'), null, false, true)).theme, 'dark')
  assert.equal((await readProtected<any>(draftFile, null, false, true)).text, 'Newest unsent synthetic draft')
  const freshRepository = createWorkspaceBackupRepository(deps)
  const committedBeforeRestart = await freshRepository.snapshot()
  const completedStage = await freshRepository.stage({ ...plan, expectedRevision: committedBeforeRestart.revision })
  await freshRepository.commit(completedStage, committedBeforeRestart.revision)
  // A durable completed marker survives failure/exit before staging cleanup.
  const completed = await recoverWorkspaceBackupCheckpoints(deps)
  assert.equal(completed.completed, 1)
  assert.equal(completed.rolledBack, 0)
  assert.equal((await readProtected<any>(path.join(root, 'synthetic-settings.json'), null, false, true)).theme, 'light', 'startup never rolls back a completed restore')
  assert.equal(state.threads[0].title, 'Synthetic replacement after recovery review')

  const rollbackBeforeRestart = await freshRepository.snapshot()
  const interruptedRollback = await freshRepository.stage({ ...plan, expectedRevision: rollbackBeforeRestart.revision })
  await freshRepository.commit(interruptedRollback, rollbackBeforeRestart.revision)
  await writeWorkspaceRecoveryMarker(deps, interruptedRollback.directory, root, 'rolling-back')
  await writeProtected(path.join(root, 'synthetic-settings.json'), { theme: 'synthetic interrupted rollback' })
  assert.equal((await recoverWorkspaceBackupCheckpoints(deps)).rolledBack, 1, 'a crash during rollback cannot be mistaken for a completed restore')
  assert.equal((await readProtected<any>(path.join(root, 'synthetic-settings.json'), null, false, true)).theme, 'light')

  const blockedPreview = await invoke('preview-restore', { password: 'synthetic native password', mode: 'merge' }).catch(() => null)
  // The previous chosen file was deliberately replaced in the destination-race test.
  assert.equal(blockedPreview, null)
  const freshNative = await invoke('preview-export', { conversationIds: ['chat'], drafts: true, attachments: true })
  await invoke('export', { approvalId: freshNative.approvalId, password: 'synthetic native password' })
  const failedNativeRestore = await invoke('preview-restore', { password: 'synthetic native password', mode: 'replace' })
  failPermanently = true
  await assert.rejects(invoke('apply-restore', { approvalId: failedNativeRestore.approvalId }), /rollback needs recovery/)
  assert.equal(leased, true, 'failed rollback retains the global workspace admission barrier')
  assert.equal(paused, true, 'failed rollback keeps draft writes paused')
  failPermanently = false
  // A new process recovers before admitting state or drafts again.
  assert.equal((await recoverWorkspaceBackupCheckpoints(deps)).rolledBack, 1)
  leased = false; paused = false
  const finalSnapshot = await freshRepository.snapshot()
  const corruptStage = await freshRepository.stage({ ...plan, expectedRevision: finalSnapshot.revision })
  await writeProtected(path.join(corruptStage.directory, 'status.json'), { schema: 'invalid-marker' })
  await assert.rejects(recoverWorkspaceBackupCheckpoints(deps), /status is invalid/)
  assert.equal((await fs.stat(corruptStage.directory)).isDirectory(), true, 'invalid recovery status blocks startup and retains evidence')
  core.dispose()
  console.log('Synthetic protected host repository, draft barrier, scoped attachments, native path and rollback tests passed')
} finally { await fs.rm(root, { recursive: true, force: true }); key.fill(0) }
