import { constants as fsConstants, promises as fs } from 'node:fs'
import { randomBytes } from 'node:crypto'

// Shared mutable recovery state. Backups/quarantines are process-wide: once a protected file is
// quarantined (undecryptable / malformed), every later write to that path is blocked until an
// explicit purge or a successful re-read lifts the block. The maps are imported by store.ts so
// the write guards (writeJson / writeProtectedJson / overwriteAndUnlink) all share one view.
export const recoveryBackups = new Map<string, string>()
export const quarantinedProtectedFiles = new Map<string, { backupPath: string }>()
export const legacySecretBackups = new Map<string, string>()

// Back up an undecryptable protected file alongside itself so the user can attempt
// recovery later (re-pair the key, restore from backup). Returns the backup path.
async function backupExistingFile(p: string): Promise<string | null> {
  try {
    const raw = await fs.readFile(p)
    const existing = recoveryBackups.get(p)
    if (existing) {
      const backedUp = await fs.readFile(existing).catch(() => null)
      if (backedUp?.equals(raw)) return existing
      // The live path was explicitly restored/replaced (or the old backup was
      // removed). Never cite a stale recovery copy for different current bytes.
      recoveryBackups.delete(p)
    }
    const suffix = `${new Date().toISOString().replace(/[:.]/g, '-')}-${randomBytes(4).toString('hex')}`
    const backup = `${p}.undecryptable-${suffix}.bak`
    const noFollow = typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0
    const handle = await fs.open(backup, fsConstants.O_WRONLY | fsConstants.O_CREAT | fsConstants.O_EXCL | noFollow, 0o600)
    try {
      await handle.writeFile(raw)
      await handle.sync().catch(() => {})
    } finally {
      await handle.close().catch(() => {})
    }
    recoveryBackups.set(p, backup)
    return backup
  } catch {
    return null
  }
}

async function backupLegacySecretFile(p: string): Promise<string | null> {
  const existing = legacySecretBackups.get(p)
  if (existing) return existing
  try {
    const raw = await fs.readFile(p)
    const suffix = `${new Date().toISOString().replace(/[:.]/g, '-')}-${randomBytes(4).toString('hex')}`
    const backup = `${p}.legacy-safe-storage-${suffix}.bak`
    const noFollow = typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0
    const handle = await fs.open(backup, fsConstants.O_WRONLY | fsConstants.O_CREAT | fsConstants.O_EXCL | noFollow, 0o600)
    try {
      await handle.writeFile(raw)
      await handle.sync().catch(() => {})
    } finally {
      await handle.close().catch(() => {})
    }
    legacySecretBackups.set(p, backup)
    return backup
  } catch {
    return null
  }
}

async function backupUndecryptableFile(p: string): Promise<string> {
  const backup = await backupExistingFile(p)
  if (backup) {
    // Honest, recovery-accurate message: the blob may be (a) app-key encrypted with a
    // rotated/missing ~/.vontra/.secretkey, OR (b) a pre-refactor file encrypted via the
    // macOS Keychain by an older build, which this build intentionally never reads (no
    // Keychain access by policy) and therefore cannot auto-migrate. Either way the
    // ORIGINAL is preserved at the backup path — the live state is never destroyed (the
    // boot re-write also refuses to overwrite it; see writeTensorFoldStudioStateJson).
    console.error(`[vontra] Could not decrypt ${p}: it was written with a key this build can't access — a profile/external key may be missing or rotated, or the file may predate the current format and use the macOS Keychain (which this build no longer reads). The original is preserved at ${backup}; the live path is quarantined and write-blocked. Recover by restoring the backup with the matching key available, or explicitly purge the quarantined state.`)
    quarantinedProtectedFiles.set(p, { backupPath: backup })
    return backup
  }
  throw new Error(`Could not decrypt or create a recovery backup for ${p}; refusing to continue because a later write could destroy the only copy.`)
}

async function quarantineMalformedProtectedFile(p: string, problem: string): Promise<string> {
  const backup = await backupExistingFile(p)
  if (!backup) {
    throw new Error(`Could not ${problem} or create a recovery backup for ${p}; refusing to continue because a later write could destroy the only copy.`)
  }
  quarantinedProtectedFiles.set(p, { backupPath: backup })
  console.error(`[vontra] Could not ${problem} ${p}. The original is preserved at ${backup}; the live path is quarantined and write-blocked. Restore a valid file or explicitly purge the quarantined state.`)
  return backup
}

export async function overwriteAndUnlink(p: string): Promise<void> {
  const stat = await fs.lstat(p).catch((error: NodeJS.ErrnoException) => {
    if (error?.code === 'ENOENT') return null
    throw error
  })
  if (!stat) {
    quarantinedProtectedFiles.delete(p)
    recoveryBackups.delete(p)
    return
  }
  // Never follow a symlink while shredding. A compromised state path must not be able to turn
  // "forget local memory" into an overwrite of an arbitrary user file outside ~/.vontra.
  if (stat.isSymbolicLink()) {
    await fs.unlink(p)
    quarantinedProtectedFiles.delete(p)
    recoveryBackups.delete(p)
    return
  }
  if (!stat.isFile()) return
  // lstat/open is otherwise a symlink-swap race. O_NOFOLLOW makes the kernel reject a path that
  // changed into a symlink between those operations; failing the purge is safer than following it.
  const noFollow = typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0
  const handle = await fs.open(p, fsConstants.O_RDWR | noFollow)
  try {
    const opened = await handle.stat()
    if (!opened.isFile()) throw new Error(`Refusing to shred non-regular file: ${p}`)
    let offset = 0
    while (offset < opened.size) {
      const len = Math.min(64 * 1024, opened.size - offset)
      await handle.write(randomBytes(len), 0, len, offset)
      offset += len
    }
    await handle.sync().catch(() => {})
  } finally {
    await handle.close().catch(() => {})
  }
  await fs.unlink(p).catch((error: NodeJS.ErrnoException) => {
    if (error?.code !== 'ENOENT') throw error
  })
  quarantinedProtectedFiles.delete(p)
  recoveryBackups.delete(p)
}

// Internal helpers re-exported so store.ts (and other store submodules) can call the same
// backup/quarantine routines the protected-JSON read/write path relies on.
export {
  backupExistingFile,
  backupLegacySecretFile,
  backupUndecryptableFile,
  quarantineMalformedProtectedFile,
}
