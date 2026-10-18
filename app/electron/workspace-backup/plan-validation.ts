import type { WorkspaceRestorePlan } from './service'
import { backupDigest, MAX_BLOB_BYTES, MAX_TOTAL_BLOB_BYTES, validateData, validateSelection } from './archive'
import { portableImportedRecovery } from './portable-records'

/** Recheck the complete admitted plan independently before staging or any canonical write. */
export function validatedRestorePlan(plan: WorkspaceRestorePlan): WorkspaceRestorePlan {
  if (!plan || !['merge', 'replace'].includes(plan.mode) || typeof plan.expectedRevision !== 'string' || !plan.expectedRevision
    || typeof plan.sourceDigest !== 'string' || !/^[a-f0-9]{64}$/u.test(plan.sourceDigest)) throw new Error('Invalid workspace restore plan.')
  const data = portableImportedRecovery(validateData(plan.data)), selection = validateSelection(plan.selection)
  if (!(plan.blobs instanceof Map) || plan.blobs.size > 512) throw new Error('Invalid workspace restore blob plan.')
  const referenced = new Set<string>()
  for (const attachment of data.attachments) for (const id of [attachment.originalBlobId, attachment.parsedBlobId]) if (id) referenced.add(id)
  let total = 0
  for (const [id, bytes] of plan.blobs) {
    if (!/^[a-f0-9]{64}$/u.test(id) || !(bytes instanceof Uint8Array) || bytes.byteLength > MAX_BLOB_BYTES || backupDigest(bytes) !== id || !referenced.has(id)) throw new Error('Invalid workspace restore blob identity.')
    total += bytes.byteLength
    if (total > MAX_TOTAL_BLOB_BYTES) throw new Error('Workspace restore blob size limit exceeded.')
  }
  for (const id of referenced) if (!plan.blobs.has(id)) throw new Error('A referenced restore blob is missing.')
  return { ...plan, data, selection }
}
