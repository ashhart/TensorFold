import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { validateComposerDraftSnapshot } from '../../../../shared/composer-drafts'
import type { BackupRecord, WorkspaceBackupAttachment } from '../../../../shared/workspace-backup'
import { backupDigest, MAX_BLOB_BYTES, MAX_TOTAL_BLOB_BYTES } from '../../../workspace-backup/archive'
import type { BackupHostState } from './repository'

const HASH = /^[a-f0-9]{64}$/u
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/u
export const MAX_RECOVERY_JSON_BYTES = 64 * 1024 * 1024
export interface WorkspaceRecoveryCheckpoint {
  schema: 'tensorfold.workspace-recovery/1'
  profile: string
  checkpointId: string
  before: BackupHostState
  drafts: BackupRecord[]
  attachments: WorkspaceBackupAttachment[]
  blobs: Array<{ id: string; base64: string }>
  plannedDraftIds: string[]
  plannedBlobIds: string[]
  existingBlobFiles: string[]
}
interface RecoveryIo {
  profileRoot(): string
  readProtectedJson<T>(file: string, fallback: T, migratePlaintext?: boolean, requireEncrypted?: boolean): Promise<T>
  writeProtectedJson(file: string, value: unknown): Promise<void>
}
interface RecoveryMarker { schema: 'tensorfold.workspace-recovery-marker/1'; profile: string; checkpointId: string; state: 'committed' | 'rolling-back' | 'rolled-back' }

export async function syncWorkspaceRecoveryFile(file: string): Promise<void> {
  const handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW)
  try {
    const stat = await handle.stat()
    if (!stat.isFile() || stat.nlink !== 1) throw new Error('Recovery write is not a regular private file.')
    await handle.sync()
  } finally { await handle.close() }
  const parent = await fs.open(path.dirname(file), constants.O_RDONLY | constants.O_NOFOLLOW)
  try { await parent.sync() } finally { await parent.close() }
}

/** Flush the original canonical owners before persisting the terminal restore marker. */
export async function syncWorkspaceRecoveryState(root: string): Promise<void> {
  const files = ['settings.json', 'models.json', path.join('state', 'threads.json'), path.join('state', 'projects.json')].map((part) => path.join(root, part))
  for (const namespace of ['composer-drafts', 'workspace-managed-attachments']) {
    const location = await directory(root, ['state', namespace])
    if (location) for (const file of await fs.readdir(location)) if (/^[a-f0-9]{64}\.json$/u.test(file) || namespace === 'workspace-managed-attachments' && file === 'index.json') files.push(path.join(location, file))
  }
  for (const file of files) {
    const exists = await fs.lstat(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (exists) await syncWorkspaceRecoveryFile(file)
  }
}

export async function writeWorkspaceRecoveryMarker(deps: Pick<RecoveryIo, 'writeProtectedJson'>, directory: string, root: string, state: RecoveryMarker['state']): Promise<void> {
  await deps.writeProtectedJson(path.join(directory, 'status.json'), { schema: 'tensorfold.workspace-recovery-marker/1', profile: backupDigest(Buffer.from(root)), checkpointId: path.basename(directory), state } satisfies RecoveryMarker)
  await syncWorkspaceRecoveryFile(path.join(directory, 'status.json'))
}

async function directory(root: string, parts: string[], create = false): Promise<string | null> {
  let current = path.resolve(root)
  const base = await fs.lstat(current)
  if (!base.isDirectory() || base.isSymbolicLink() || await fs.realpath(current) !== current) throw new Error('Workspace recovery profile is linked or unavailable.')
  for (const part of parts) {
    if (!/^[A-Za-z0-9_-]+$/u.test(part)) throw new Error('Invalid workspace recovery namespace.')
    current = path.join(current, part)
    let stat = await fs.lstat(current).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (!stat && create) { await fs.mkdir(current, { mode: 0o700 }); stat = await fs.lstat(current) }
    if (!stat) return null
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('Workspace recovery namespace is linked or unavailable.')
  }
  return current
}

async function protectedRead<T>(deps: RecoveryIo, file: string): Promise<T | undefined> {
  const stat = await fs.lstat(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
  if (!stat) return undefined
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > MAX_RECOVERY_JSON_BYTES * 2) throw new Error('Workspace recovery file is invalid or oversized.')
  const value = await deps.readProtectedJson<T | undefined>(file, undefined, false, true)
  if (value === undefined) throw new Error('Workspace recovery data could not be decrypted; its checkpoint is retained.')
  return value
}

function validateCheckpoint(input: unknown, root: string, checkpointId: string): WorkspaceRecoveryCheckpoint {
  const value = input as WorkspaceRecoveryCheckpoint
  if (!value || typeof value !== 'object' || value.schema !== 'tensorfold.workspace-recovery/1' || value.profile !== backupDigest(Buffer.from(root)) || value.checkpointId !== checkpointId
    || !value.before || typeof value.before.settings !== 'object' || !Array.isArray(value.before.threads) || !Array.isArray(value.before.projects)
    || value.before.threads.some((thread) => !thread || typeof thread.id !== 'string' || thread.incognito)
    || !Array.isArray(value.drafts) || value.drafts.length > 10_000 || !Array.isArray(value.attachments) || value.attachments.length > 10_000
    || !Array.isArray(value.blobs) || value.blobs.length > 512) throw new Error('Workspace recovery checkpoint is unsupported; the original is retained.')
  for (const draft of value.drafts) {
    if (typeof draft.id !== 'string' || !HASH.test(draft.id)) throw new Error('Workspace recovery draft identity is invalid.')
    validateComposerDraftSnapshot(draft.snapshot)
  }
  for (const list of [value.plannedDraftIds, value.plannedBlobIds, value.existingBlobFiles]) if (!Array.isArray(list) || list.length > 10_000 || list.some((id) => typeof id !== 'string' || !HASH.test(id))) throw new Error('Workspace recovery target allowlist is invalid.')
  let total = 0
  const seen = new Set<string>()
  for (const blob of value.blobs) {
    if (!blob || !HASH.test(blob.id) || seen.has(blob.id) || typeof blob.base64 !== 'string' || blob.base64.length > Math.ceil(MAX_BLOB_BYTES / 3) * 4
      || blob.base64.length % 4 || !/^[A-Za-z0-9+/]*={0,2}$/u.test(blob.base64)) throw new Error('Workspace recovery attachment is invalid.')
    const decoded = Buffer.from(blob.base64, 'base64'); total += decoded.length; seen.add(blob.id)
    if (decoded.length > MAX_BLOB_BYTES || total > MAX_TOTAL_BLOB_BYTES || decoded.toString('base64') !== blob.base64 || backupDigest(decoded) !== blob.id) throw new Error('Workspace recovery attachment integrity failed.')
  }
  for (const attachment of value.attachments) if (!attachment || typeof attachment.id !== 'string'
    || [attachment.originalBlobId, attachment.parsedBlobId].some((id) => id !== null && (typeof id !== 'string' || !HASH.test(id)))) throw new Error('Workspace recovery attachment manifest is invalid.')
  return value
}

/** Must run before loading settings/threads, starting workers, or admitting any renderer writes. */
export async function recoverWorkspaceBackupCheckpoints(deps: RecoveryIo & { persist(state: BackupHostState): Promise<void> }): Promise<{ rolledBack: number; completed: number; warnings: string[] }> {
  const root = path.resolve(deps.profileRoot()), staging = await directory(root, ['state', 'workspace-restore-staging'])
  const result = { rolledBack: 0, completed: 0, warnings: [] as string[] }
  if (!staging) return result
  const entries = await fs.readdir(staging, { withFileTypes: true })
  if (entries.length > 20) throw new Error('Too many workspace recovery checkpoints; startup remains blocked for controlled recovery.')
  const pending: Array<{ folder: string; checkpoint: WorkspaceRecoveryCheckpoint }> = []
  for (const entry of entries) {
    if (!UUID.test(entry.name) || !entry.isDirectory() || entry.isSymbolicLink()) throw new Error('Unexpected workspace recovery entry; startup remains blocked.')
    const folder = (await directory(root, ['state', 'workspace-restore-staging', entry.name]))!
    const marker = await protectedRead<RecoveryMarker>(deps, path.join(folder, 'status.json'))
    if (marker) {
      if (marker.schema !== 'tensorfold.workspace-recovery-marker/1' || marker.profile !== backupDigest(Buffer.from(root)) || marker.checkpointId !== entry.name || !['committed', 'rolling-back', 'rolled-back'].includes(marker.state)) throw new Error('Workspace recovery status is invalid; startup remains blocked.')
      if (marker.state !== 'rolling-back') {
        if (marker.state === 'committed') result.completed++
        await fs.rm(folder, { recursive: true, force: true })
        continue
      }
    }
    const checkpoint = await protectedRead<unknown>(deps, path.join(folder, 'checkpoint.json'))
    // No live write begins until a canonical protected checkpoint has been committed.
    if (checkpoint === undefined) { await fs.rm(folder, { recursive: true, force: true }); continue }
    pending.push({ folder, checkpoint: validateCheckpoint(checkpoint, root, entry.name) })
  }
  // Multiple nonterminal restores cannot be admitted by the runtime global fence.
  if (pending.length > 1) throw new Error('Multiple incomplete workspace restores need controlled recovery; startup remains blocked.')
  for (const { folder, checkpoint } of pending) {
    if (path.resolve(deps.profileRoot()) !== root) throw new Error('Workspace recovery profile changed; startup remains blocked.')
    const before = { ...checkpoint.before, threads: checkpoint.before.threads.map((thread) => ({ ...thread, status: 'idle' as const })) }
    await deps.persist(before)
    const draftDirectory = (await directory(root, ['state', 'composer-drafts'], true))!
    const oldDraftIds = new Set(checkpoint.drafts.map((draft) => String(draft.id)))
    for (const draft of checkpoint.drafts) {
      const snapshot = validateComposerDraftSnapshot(draft.snapshot), file = path.join(draftDirectory, `${draft.id}.json`)
      const current = await protectedRead<unknown>(deps, file)
      const revision = Math.max(snapshot.revision, current ? validateComposerDraftSnapshot(current).revision : 0) + 1
      if (!Number.isSafeInteger(revision)) throw new Error('Workspace recovery draft revision limit reached; startup remains blocked.')
      await deps.writeProtectedJson(file, { ...snapshot, revision })
    }
    for (const id of checkpoint.plannedDraftIds) if (!oldDraftIds.has(id)) await fs.unlink(path.join(draftDirectory, `${id}.json`)).catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error })
    const managed = (await directory(root, ['state', 'workspace-managed-attachments'], true))!
    for (const blob of checkpoint.blobs) await deps.writeProtectedJson(path.join(managed, `${blob.id}.json`), { base64: blob.base64 })
    await deps.writeProtectedJson(path.join(managed, 'index.json'), { attachments: checkpoint.attachments })
    const originalFiles = new Set(checkpoint.existingBlobFiles)
    for (const id of checkpoint.plannedBlobIds) if (!originalFiles.has(id)) await fs.unlink(path.join(managed, `${id}.json`)).catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error })
    await syncWorkspaceRecoveryState(root)
    await writeWorkspaceRecoveryMarker(deps, folder, root, 'rolled-back')
    await fs.rm(folder, { recursive: true, force: true })
    result.rolledBack++
  }
  if (result.rolledBack) result.warnings.push('An interrupted workspace restore was rolled back before startup; the previous workspace and unsent drafts were recovered.')
  if (result.completed) result.warnings.push('A completed restore was preserved; its retained staging checkpoint was cleaned up.')
  return result
}
