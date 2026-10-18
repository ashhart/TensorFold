import { randomUUID } from 'node:crypto'
import type { WorkspaceBackupData, WorkspaceBackupPreview, WorkspaceBackupSelection, WorkspaceRestoreMode, WorkspaceRestoreResult } from '../../shared/workspace-backup'
import { backupCounts, backupDigest, buildBackupArchive, encodeArchive, parseArchive, validateData, type BackupArchive, type BackupSnapshot } from './archive'
import { decryptWorkspaceBackup, encryptWorkspaceBackup, MAX_BACKUP_BYTES } from './crypto'

export interface WorkspaceRestorePlan {
  mode: WorkspaceRestoreMode
  expectedRevision: string
  sourceDigest: string
  /** Selected incoming logical records; merge preserves colliding existing identities. */
  data: WorkspaceBackupData
  blobs: ReadonlyMap<string, Uint8Array>
  selection: WorkspaceBackupSelection
}

export interface WorkspaceBackupRepository<Stage> {
  /** Serialized host snapshot including all unsent drafts after the host's draft-flush barrier. */
  snapshot(): Promise<BackupSnapshot>
  /** Namespace + monotonic state/draft revision; changes on every relevant mutation/profile change. */
  revision(): Promise<string>
  /** Isolated staging only; own independent bytes plus a recoverable pre-restore snapshot/drafts. */
  stage(plan: WorkspaceRestorePlan): Promise<Stage>
  /** Host must enforce expectedRevision atomically and never replay active jobs/actions. */
  commit(stage: Stage, expectedRevision: string, signal?: AbortSignal): Promise<void>
  /** Restore the staged pre-restore checkpoint, including unsent drafts, even after partial commit. */
  rollback(stage: Stage): Promise<void>
  discard(stage: Stage): Promise<void>
}

interface ApprovalBase { id: string; revision: string; expiresAt: number; preview: WorkspaceBackupPreview }
type Approval = ApprovalBase & ({ operation: 'export'; payload: Buffer } | {
  operation: 'restore'; archive: BackupArchive; blobs: Map<string, Uint8Array>; fileId: string; digest: string; mode: WorkspaceRestoreMode
})

export class WorkspaceRestoreError extends Error {
  constructor(message: string, readonly rolledBack: boolean) { super(message); this.name = 'WorkspaceRestoreError' }
}

function checkFileId(fileId: string): void {
  if (typeof fileId !== 'string' || !fileId || fileId.length > 256 || /\0/u.test(fileId)) throw new Error('Invalid backup file identity.')
}
function cancelled(signal?: AbortSignal): void { if (signal?.aborted) throw new Error('Restore cancelled.') }

export function createWorkspaceBackupService<Stage>(deps: { repository: WorkspaceBackupRepository<Stage>; now?: () => number }) {
  const repository = deps.repository, now = deps.now ?? Date.now
  const approvals = new Map<string, Approval>()
  let busy = false, cryptoBusy = false, disposed = false
  function active() { if (disposed) throw new Error('Workspace backup session closed; reopen the preview.') }
  function remove(approval: Approval) {
    approvals.delete(approval.id)
    if (approval.operation === 'export') approval.payload.fill(0)
    else for (const blob of approval.blobs.values()) blob.fill(0)
  }
  function reserve(approval: Approval) {
    for (const entry of approvals.values()) if (entry.expiresAt <= now()) remove(entry)
    while (approvals.size >= 2) remove(approvals.values().next().value!)
    approvals.set(approval.id, approval)
  }
  function take(id: string, operation: Approval['operation']): Approval {
    const approval = approvals.get(id)
    if (!approval || approval.operation !== operation) throw new Error('Backup preview approval is unavailable; preview again.')
    approvals.delete(id)
    if (approval.expiresAt <= now()) { remove(approval); throw new Error('Backup preview approval expired; preview again.') }
    return approval
  }
  function makePreview(operation: 'export' | 'restore', archive: BackupArchive, blobs: ReadonlyMap<string, Uint8Array>, mode?: WorkspaceRestoreMode): WorkspaceBackupPreview {
    return { approvalId: randomUUID(), operation, ...(mode ? { mode } : {}), counts: backupCounts(archive.data, blobs),
      warnings: [...archive.warnings], conflicts: [], expiresAt: now() + 10 * 60_000 }
  }
  return {
    async previewExport(selection: WorkspaceBackupSelection): Promise<WorkspaceBackupPreview> {
      active()
      const snapshot = await repository.snapshot()
      const archive = buildBackupArchive(snapshot, selection, now())
      const payload = encodeArchive(archive)
      const selectedBlobs = new Map(archive.blobs.map((blob) => [blob.id, snapshot.blobs.get(blob.id)!]))
      const preview = makePreview('export', archive, selectedBlobs)
      if (await repository.revision() !== snapshot.revision) { payload.fill(0); throw new Error('Workspace changed; preview the backup again.') }
      active()
      reserve({ id: preview.approvalId, operation: 'export', revision: snapshot.revision, expiresAt: preview.expiresAt, preview, payload })
      return structuredClone(preview)
    },
    async exportEncrypted(input: { approvalId: string; password: string }): Promise<Uint8Array> {
      active()
      if (cryptoBusy) throw new Error('Backup encryption is busy; try again after it finishes.')
      const approval = take(input.approvalId, 'export')
      if (approval.operation !== 'export') throw new Error('Invalid backup operation.')
      try {
        cryptoBusy = true
        if (await repository.revision() !== approval.revision) throw new Error('Workspace changed; preview the backup again.')
        const bytes = await encryptWorkspaceBackup(approval.payload, input.password)
        if (disposed) { bytes.fill(0); throw new Error('Workspace backup session closed; reopen the preview.') }
        if (await repository.revision() !== approval.revision) { bytes.fill(0); throw new Error('Workspace changed while encrypting; preview again.') }
        return bytes
      } finally { approval.payload.fill(0); cryptoBusy = false }
    },
    async previewRestore(input: { fileId: string; bytes: Uint8Array; password: string; mode: WorkspaceRestoreMode }): Promise<WorkspaceBackupPreview> {
      active()
      if (cryptoBusy) throw new Error('Backup encryption is busy; try again after it finishes.')
      checkFileId(input.fileId)
      if (input.mode !== 'merge' && input.mode !== 'replace') throw new Error('Invalid restore mode.')
      // Copy before asynchronous KDF work so caller mutation cannot change the approved file.
      if (!(input.bytes instanceof Uint8Array) || input.bytes.byteLength > MAX_BACKUP_BYTES) throw new Error('Invalid or oversized backup archive.')
      const bytes = Buffer.from(input.bytes), digest = backupDigest(bytes)
      let plaintext: Buffer | undefined
      try {
        cryptoBusy = true
        plaintext = await decryptWorkspaceBackup(bytes, input.password)
        const { archive, blobs } = parseArchive(plaintext)
        const current = await repository.snapshot()
        const existing = validateData(current.data, true)
        const preview = makePreview('restore', archive, blobs, input.mode)
        for (const collection of ['conversations', 'projects', 'bots', 'drafts', 'attachments'] as const) {
          const ids = new Set(existing[collection].map((item) => String(item.id)))
          for (const item of archive.data[collection]) if (ids.has(String(item.id))) preview.conflicts.push({ collection, id: String(item.id), action: input.mode === 'merge' ? 'keep-existing' : 'replace' })
        }
        if (preview.conflicts.length) preview.warnings.push(input.mode === 'merge' ? 'Existing records with matching identities will be kept.' : 'Matching selected records will be replaced after a rollback checkpoint is staged.')
        if (await repository.revision() !== current.revision) throw new Error('Workspace changed; preview the restore again.')
        active()
        reserve({ id: preview.approvalId, operation: 'restore', revision: current.revision, expiresAt: preview.expiresAt,
          preview, archive, blobs, fileId: input.fileId, digest, mode: input.mode })
        return structuredClone(preview)
      } finally { plaintext?.fill(0); bytes.fill(0); cryptoBusy = false }
    },
    async applyRestore(input: { approvalId: string; fileId: string; bytes: Uint8Array; signal?: AbortSignal }): Promise<WorkspaceRestoreResult> {
      active()
      if (busy) throw new Error('A workspace restore is already in progress.')
      busy = true
      let approval: Approval | undefined, stage: Stage | undefined, hasStage = false, committed = false, commitStarted = false, rollbackFailed = false
      try {
        approval = take(input.approvalId, 'restore')
        if (approval.operation !== 'restore') throw new Error('Invalid restore operation.')
        checkFileId(input.fileId)
        if (!(input.bytes instanceof Uint8Array) || input.bytes.byteLength > MAX_BACKUP_BYTES
          || input.fileId !== approval.fileId || backupDigest(input.bytes) !== approval.digest) throw new Error('Backup file changed; preview the restore again.')
        if (await repository.revision() !== approval.revision) throw new Error('Workspace or drafts changed; preview the restore again.')
        cancelled(input.signal)
        stage = await repository.stage({ mode: approval.mode, expectedRevision: approval.revision, sourceDigest: approval.digest,
          data: structuredClone(approval.archive.data), blobs: new Map([...approval.blobs].map(([id, bytes]) => [id, Buffer.from(bytes)])), selection: structuredClone(approval.archive.selection) })
        hasStage = true
        cancelled(input.signal)
        if (await repository.revision() !== approval.revision) throw new Error('Workspace or drafts changed before restore commit.')
        commitStarted = true
        await repository.commit(stage, approval.revision, input.signal)
        cancelled(input.signal)
        committed = true
        return { restored: true, cancelled: false, counts: approval.preview.counts, warnings: approval.preview.warnings }
      } catch (error) {
        if (hasStage && commitStarted) {
          try { await repository.rollback(stage!) }
          catch { rollbackFailed = true; throw new WorkspaceRestoreError('Restore failed and rollback needs recovery; the staged checkpoint must be retained.', false) }
        }
        if (input.signal?.aborted && approval) return { restored: false, cancelled: true, counts: approval.preview.counts, warnings: approval.preview.warnings }
        if (hasStage && commitStarted) throw new WorkspaceRestoreError('Restore failed; the previous workspace and unsent drafts were restored.', true)
        throw error
      } finally {
        // Never discard the only recovery checkpoint when rollback itself failed.
        if (hasStage && !rollbackFailed) {
          try { await repository.discard(stage!) } catch { approval?.preview.warnings.push('Staged recovery cleanup needs attention; the recovery checkpoint was retained.') }
        }
        if (approval) remove(approval)
        busy = false
      }
    },
    dispose() { disposed = true; for (const approval of approvals.values()) remove(approval) },
  }
}
