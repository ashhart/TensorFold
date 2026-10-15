import { promises as fs } from 'node:fs'
import path from 'node:path'
import { vontraPath, legacyPath, statePath, settingsPath, readJson, fileExists, safeSegment } from './paths'
import { recoveryBackups, quarantinedProtectedFiles, overwriteAndUnlink, backupExistingFile, backupLegacySecretFile, backupUndecryptableFile, quarantineMalformedProtectedFile } from './recovery-registry'
import { protectJsonPayload, revealJsonPayload, isProtectedJsonEnvelope, isUndecryptableEnvelope, isLegacyProtectedEnvelope, isProfileFallbackProtectedEnvelope, hasMalformedDecryptedProtectedPayload, matchesProtectedFallbackShape, containsLegacySecret } from './app-crypto'
import type { Settings } from '../agent/types'

// Atomic write machinery + protected-json read/write. Everything between the path/recovery/crypto
// foundation (paths.ts / recovery-registry.ts / app-crypto.ts) and the per-collection load/save
// pairs that still live in store.ts. The per-path serialization state (writeChains / writeCounter)
// is process-wide mutable state owned HERE so writeJson / writeJsonState / writeProtectedJson all
// share one queue per file.

// Per-path write chain: serialize writes to the SAME file so concurrent fire-and-forget persist*()
// calls can't interleave or clobber each other (last write still wins, but atomically, never partial).
const writeChains = new Map<string, Promise<void>>()
let writeCounter = 0

// Crash-safe write: stage to a temp file, fdatasync it, then rename over the target. rename is atomic
// on the same filesystem, so a crash/full-disk mid-write leaves the OLD file intact instead of a
// truncated/empty one — replacing the previous truncate-in-place that could zero a whole collection.
async function atomicWriteJson(p: string, data: unknown): Promise<void> {
  await fs.mkdir(path.dirname(p), { recursive: true, mode: 0o700 })
  await fs.chmod(path.dirname(p), 0o700).catch(() => {})
  const tmp = `${p}.tmp-${process.pid}-${++writeCounter}`
  const json = JSON.stringify(data, null, 2)
  let handle: import('node:fs/promises').FileHandle | null = null
  try {
    handle = await fs.open(tmp, 'w', 0o600)
    await handle.writeFile(json, 'utf8')
    // Flush bytes to disk before the rename. A flush failure must ABORT the rename (the catch
    // unlinks the tmp and preserves the old file) instead of renaming possibly-unflushed bytes
    // over good data — so don't swallow it.
    await handle.datasync()
    await handle.close()
    handle = null
    await fs.rename(tmp, p)
    await fs.chmod(p, 0o600).catch(() => {})
  } catch (e) {
    if (handle) await handle.close().catch(() => {})
    await fs.unlink(tmp).catch(() => {}) // don't leave a stray temp file on failure
    throw e
  }
}

function writeJson(p: string, data: unknown): Promise<void> {
  const prev = writeChains.get(p) ?? Promise.resolve()
  const next = prev.catch(() => {}).then(() => {
    const quarantine = quarantinedProtectedFiles.get(p)
    if (quarantine) {
      throw new Error(`Refusing to overwrite quarantined state ${p}. Restore or explicitly purge it first. Recovery copy: ${quarantine.backupPath}`)
    }
    return atomicWriteJson(p, data)
  })
  writeChains.set(p, next)
  void next.catch(() => {}).finally(() => { if (writeChains.get(p) === next) writeChains.delete(p) })
  return next
}

export async function writeTensorFoldStudioJson(parts: string[], data: unknown): Promise<void> {
  await writeJson(vontraPath(...parts), data)
}

async function readProtectedState<T>(name: string, fallback: T): Promise<T> {
  const primary = statePath(name)
  if (await fileExists(primary)) return readProtectedJson<T>(primary, fallback)

  const legacy = legacyPath(name)
  if (!(await fileExists(legacy))) return fallback
  // Probe with `undefined` so a legacy file that FAILED to decode (returns undefined) is told
  // apart from a genuinely empty one. Only promote REAL decoded data to primary — never write
  // an authoritative empty primary because the legacy envelope was transiently undecryptable
  // (that would strand recoverable data behind a now-empty primary that future loads trust).
  const probe = await readProtectedJson<T | undefined>(legacy, undefined, false)
  if (probe === undefined) return fallback
  await writeProtectedJson(primary, probe)
  return probe
}

async function readJsonState<T>(name: string, fallback: T): Promise<T> {
  const primary = statePath(name)
  if (await fileExists(primary)) return readJson<T>(primary, fallback)

  const legacy = legacyPath(name)
  if (!(await fileExists(legacy))) return fallback
  // A failed read is not authoritative empty state: promoting its fallback
  // would hide a later repaired legacy file behind an empty primary forever.
  const imported = await readJson<T | undefined>(legacy, undefined)
  if (imported === undefined) return fallback
  await writeJson(primary, imported)
  return imported
}

function writeJsonState(name: string, data: unknown): Promise<void> {
  const p = statePath(name)
  // Reserve this write's place in the per-path serialization queue SYNCHRONOUSLY (no await before
  // this point) — exactly like writeJson()'s own registration below. The empty-overwrite guard needs
  // async work (stat + read) before it knows whether to skip; if that ran BEFORE joining the queue
  // (the old shape), a concurrent POPULATED write — which has zero awaits before reaching its own
  // registration — could register and run FIRST even when issued AFTER this empty write, so the
  // empty write then ran LAST and silently clobbered the newer data. Registering up front preserves
  // call order regardless of how much async work either side needs once it's already queued.
  const prev = writeChains.get(p) ?? Promise.resolve()
  const next = prev.catch(() => {}).then(async () => {
    // Guard against an empty in-memory value clobbering a populated-but-UNREADABLE file: readJsonState
    // falls back to [] when the file is corrupt/transiently unreadable, so an unconditional write would
    // bake that loss in permanently. A readable file —
    // populated or not — still writes through, so a legitimate "delete all" is honoured.
    const empty = Array.isArray(data) ? data.length === 0 : data == null
    if (empty) {
      const st = await fs.stat(p).catch(() => null)
      if (st?.isFile()) {
        let readable = false
        try { JSON.parse(await fs.readFile(p, 'utf8')); readable = true } catch { readable = false }
        if (!readable) {
          const backup = await backupExistingFile(p).catch(() => null)
          console.error(`[vontra] Skipped overwriting unreadable ${p} with an empty value — kept on disk for recovery${backup ? ` (backed up at ${backup})` : ''}.`)
          return
        }
      }
    }
    await atomicWriteJson(p, data)
  })
  writeChains.set(p, next)
  void next.catch(() => {}).finally(() => { if (writeChains.get(p) === next) writeChains.delete(p) })
  return next
}

export async function writeArchiveRecord(kind: 'project' | 'thread' | 'room' | 'ticket', id: string, data: unknown): Promise<void> {
  const folder = kind === 'project' ? 'projects' : kind === 'thread' ? 'threads' : kind === 'room' ? 'rooms' : 'tickets'
  await writeTensorFoldStudioProtectedJson(['archives', folder, `${safeSegment(id)}.json`], data)
}

export async function readProtectedJson<T>(p: string, fallback: T, migratePlaintext = true): Promise<T> {
  let rawText = ''
  try {
    rawText = await fs.readFile(p, 'utf8')
  } catch (error: any) {
    if (error?.code === 'ENOENT') return fallback
    await quarantineMalformedProtectedFile(p, 'read')
    return fallback
  }

  let raw: unknown
  try {
    raw = JSON.parse(rawText)
  } catch {
    // A truncated/otherwise malformed protected-state file must enter the same
    // fail-closed recovery posture as a valid envelope whose key was lost. Returning
    // a fallback without recording quarantine lets the next non-empty save replace
    // the only recoverable bytes.
    await quarantineMalformedProtectedFile(p, 'parse')
    return fallback
  }
  // If this is a protected envelope we genuinely cannot decode (lost/rotated app key,
  // not a recoverable legacy safeStorage blob), back it up before we fall through to
  // defaults — otherwise the re-save below would encrypt defaults over it and make the
  // loss permanent. This is the "settings lost on every reload" recovery path.
  if (isUndecryptableEnvelope(raw)) {
    await backupUndecryptableFile(p)
    return fallback
  }
  if (hasMalformedDecryptedProtectedPayload(raw)) {
    await quarantineMalformedProtectedFile(p, 'parse the decrypted payload from')
    return fallback
  }
  const revealed = revealJsonPayload(raw, fallback)
  if (!matchesProtectedFallbackShape(revealed, fallback)) {
    await quarantineMalformedProtectedFile(p, 'validate the top-level data in')
    return fallback
  }
  // A matching key/file may have been restored since a prior read. Successful
  // decode + shape classification automatically lifts the write quarantine.
  const wasQuarantined = quarantinedProtectedFiles.delete(p)
  if (wasQuarantined) recoveryBackups.delete(p)
  // Migrate legacy plaintext or a profile-fallback-readable envelope to the active
  // key on first read. Truly unknown safeStorage/lost-key envelopes returned above.
  const isEnvelope = !!raw && typeof raw === 'object' && !Array.isArray(raw) && typeof (raw as { encrypted?: unknown }).encrypted === 'string'
  const needsResave = migratePlaintext && (!isEnvelope || isLegacyProtectedEnvelope(raw) || isProfileFallbackProtectedEnvelope(raw))
  if (needsResave) {
    // A plaintext legacy structure may contain per-field safeStorage ciphertext.
    // Back up its exact bytes before the structural whole-file migration; the
    // ciphertext itself is preserved inside the new envelope for later re-entry.
    if (containsLegacySecret(revealed)) await backupLegacySecretFile(p)
    await writeProtectedJson(p, revealed)
  }
  return revealed
}

// `opts.isEmpty` lets a caller whose payload is a non-array OBJECT (e.g. ContactDirectory) state
// emptiness explicitly: the default `Array.isArray(data) ? data.length===0 : data==null` check can
// never be true for a non-null object, so the guard below silently never engaged for any
// object-shaped protected collection — a freshly-constructed "empty" object (e.g. from a failed
// load) could clobber a real, merely-transiently-unreadable file with zero protection.
export async function writeProtectedJson(
  p: string,
  data: unknown,
  opts: {
    isEmpty?: boolean
    purgeLegacyPlaintext?: boolean
    /** Stage a detached payload inside this file's serialized write slot. */
    prepareData?: () => unknown
    /** Publish the staged state synchronously after rename, before later writes advance. */
    onCommitted?: () => void
  } = {},
): Promise<void> {
  // Reserve this write's place in the per-path queue SYNCHRONOUSLY — see writeJsonState's identical
  // fix for why: the guard's async stat+read must not delay joining the queue relative to a
  // concurrent populated write, or call order (last write wins) breaks for the empty-vs-populated race.
  const prev = writeChains.get(p) ?? Promise.resolve()
  const next = prev.catch(() => {}).then(async () => {
    const empty = opts.isEmpty ?? (Array.isArray(data) ? data.length === 0 : data == null)
    const quarantine = quarantinedProtectedFiles.get(p)
    if (quarantine) {
      if (empty) {
        console.error(`[vontra] Skipped overwriting quarantined protected state ${p} with an empty value. Recovery copy: ${quarantine.backupPath}`)
        return
      }
      throw new Error(`Refusing to overwrite quarantined protected state ${p}. Restore the matching key/file or explicitly purge it. Recovery copy: ${quarantine.backupPath}`)
    }
    // Do not rely on a loader having run first. Any caller can issue a populated save
    // directly, so probe an existing file inside this path's serialized write slot.
    // readProtectedJson(..., false) quarantines malformed JSON, malformed decrypted
    // payloads, and unknown-key envelopes without attempting a migration write.
    const st = await fs.stat(p).catch((error: NodeJS.ErrnoException) => {
      if (error?.code === 'ENOENT') return null
      throw error
    })
    if (st?.isFile()) {
      await readProtectedJson(p, data, false)
      const detected = quarantinedProtectedFiles.get(p)
      if (detected) {
        if (empty) {
          console.error(`[vontra] Skipped overwriting newly quarantined protected state ${p} with an empty value. Recovery copy: ${detected.backupPath}`)
          return
        }
        throw new Error(`Refusing to overwrite quarantined protected state ${p}. Restore the matching key/file or explicitly purge it. Recovery copy: ${detected.backupPath}`)
      }
    }
    if (opts.purgeLegacyPlaintext) {
      // Legacy plaintext shredding belongs inside the same serialized slot and only
      // after the existing file has passed syntax/decryption/top-level-shape checks.
      // purgeLegacyPlaintextJson re-validates the bytes before unlinking to narrow
      // the probe-to-purge race with an external writer.
      await purgeLegacyPlaintextJson(p, data)
      const detected = quarantinedProtectedFiles.get(p)
      if (detected) {
        if (empty) {
          console.error(`[vontra] Skipped overwriting newly quarantined protected state ${p} with an empty value. Recovery copy: ${detected.backupPath}`)
          return
        }
        throw new Error(`Refusing to overwrite quarantined protected state ${p}. Restore the matching key/file or explicitly purge it. Recovery copy: ${detected.backupPath}`)
      }
    }
    const prepared = opts.prepareData ? opts.prepareData() : data
    await atomicWriteJson(p, protectJsonPayload(prepared))
    opts.onCommitted?.()
  })
  writeChains.set(p, next)
  void next.catch(() => {}).finally(() => { if (writeChains.get(p) === next) writeChains.delete(p) })
  return next
}

export async function writeTensorFoldStudioProtectedJson(parts: string[], data: unknown): Promise<void> {
  const p = vontraPath(...parts)
  await writeProtectedJson(p, data, { purgeLegacyPlaintext: true })
}

// Persist a boot-time state collection, but REFUSE to overwrite a populated on-disk file
// with an empty in-memory value. The loaders silently fall back to [] when a state file is
// corrupt or transiently unreadable (readJson catch); the unconditional boot re-write would
// then make that recoverable situation a permanent wipe. When the in-memory value is empty
// yet the file on disk has real content, we back it up and leave it in place instead. An
// already-empty file ("[]", 2 bytes) or an absent file (fresh install) writes normally.
export async function writeTensorFoldStudioStateJson(parts: string[], data: unknown): Promise<void> {
  const empty = Array.isArray(data) ? data.length === 0 : data == null
  if (empty) {
    const p = vontraPath(...parts)
    // Only guard an EXISTING file (an absent file = fresh install, where writing [] is correct).
    const st = await fs.stat(p).catch(() => null)
    if (st?.isFile()) {
      // State collections are encrypted envelopes, so decode to compare. `undefined` means the
      // file exists but couldn't be decoded (corrupt / undecryptable) — also a case where we
      // must NOT clobber it with empty.
      const probe = await readProtectedJson<unknown>(p, data, false)
      const hasData = Array.isArray(probe) ? probe.length > 0 : probe != null
      if (hasData) {
        const backup = await backupExistingFile(p).catch(() => null)
        console.error(`[vontra] Skipped overwriting ${p} with an empty collection — its on-disk data was kept (a load likely fell back to empty).${backup ? ` Backed up for recovery at ${backup}.` : ' Existing backup already recorded for this process.'}`)
        return
      }
    }
  }
  await writeTensorFoldStudioProtectedJson(parts, data)
}

async function purgeLegacyPlaintextJson(p: string, expectedShape: unknown): Promise<void> {
  let raw = ''
  try {
    raw = await fs.readFile(p, 'utf8')
  } catch (error: any) {
    if (error?.code !== 'ENOENT') await quarantineMalformedProtectedFile(p, 'read')
    return
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    await quarantineMalformedProtectedFile(p, 'parse')
    return
  }
  if (isProtectedJsonEnvelope(parsed)) return
  if (!matchesProtectedFallbackShape(parsed, expectedShape)) {
    await quarantineMalformedProtectedFile(p, 'validate the top-level data in')
    return
  }
  await overwriteAndUnlink(p)
}

// writeModelsJson is defined in store.ts (it belongs to the settings load/save flow there). Injecting
// it as a callback avoids a circular import: readSettingsJson lives here, writeModelsJson lives there.
export async function readSettingsJson(
  p: string,
  protectedAtRest: boolean,
  writeModelsJson: (s: Settings) => Promise<void>,
): Promise<Partial<Settings> | null> {
  let rawText = ''
  try {
    rawText = await fs.readFile(p, 'utf8')
  } catch (error: any) {
    // File absent = fresh install (backupExistingFile no-ops). File present but unreadable
    // (lock/EIO/permission flap) = preserve a recovery copy before we fall back to defaults, so a
    // later save can't overwrite the still-recoverable original UNRECOVERABLY (the silent wipe of
    // every provider + API key). Mirrors the undecryptable-envelope recovery path below.
    if (error?.code === 'ENOENT') return null
    const backup = await backupExistingFile(p)
    if (!backup) throw new Error(`Could not read or back up ${p}; refusing to start with defaults because that could destroy the only settings copy.`)
    throw new Error(`Could not read ${p}. A recovery copy is preserved at ${backup}; fix file access before changing settings.`)
  }

  let raw: unknown
  try {
    raw = JSON.parse(rawText)
  } catch {
    // Corrupt / truncated settings.json — back up the original bytes before defaults overwrite them.
    const backup = await backupExistingFile(p)
    if (!backup) throw new Error(`Could not parse or back up ${p}; refusing to overwrite the only settings copy.`)
    quarantinedProtectedFiles.set(p, { backupPath: backup })
    return null
  }

  // A protected settings envelope we can't decode means the key was lost/rotated
  // (or it is legacy safeStorage). Quarantine it after a verified recovery backup;
  // all later writes to the live path fail until recovery or an explicit purge.
  if (protectedAtRest && isUndecryptableEnvelope(raw)) {
    await backupUndecryptableFile(p)
    return null
  }

  const revealed = revealJsonPayload<Partial<Settings> | null>(raw, null)
  if (!revealed || typeof revealed !== 'object' || Array.isArray(revealed)) {
    await quarantineMalformedProtectedFile(p, 'validate the top-level settings data in')
    return null
  }
  const wasQuarantined = quarantinedProtectedFiles.delete(p)
  if (wasQuarantined) recoveryBackups.delete(p)
  // Migrate an active-key or profile-fallback-readable settings envelope to the
  // readable per-field-protected form. Unknown safeStorage/lost-key envelopes were
  // quarantined above. This security migration is required and must surface failure.
  if (protectedAtRest && isProtectedJsonEnvelope(raw)) {
    await writeJson(p, revealed)
    if (p === settingsPath() && (revealed as Settings).providers) await writeModelsJson(revealed as Settings)
  }
  return revealed
}

// Internal helpers re-exported so store.ts (and other store submodules) can route the same
// serialized state writes through this module's queue.
export {
  writeChains,
  writeCounter,
  atomicWriteJson,
  writeJson,
  readProtectedState,
  readJsonState,
  writeJsonState,
  purgeLegacyPlaintextJson,
}
