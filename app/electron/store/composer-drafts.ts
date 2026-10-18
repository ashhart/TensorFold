import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { composerDraftKey, emptyComposerDraft, validateComposerDraftScope, validateComposerDraftSnapshot, validateComposerDraftWrite } from '../../shared/composer-drafts'
import type { ComposerDraftScope, ComposerDraftSnapshot, ComposerDraftWrite, ComposerDraftWriteResult } from '../../shared/composer-drafts'
import { vontraPath } from './paths'
import { readProtectedJson, writeProtectedJson } from './write-infra'
import { overwriteAndUnlink } from './recovery-registry'

// Independent files and queues prevent drafts in different chats from overwriting one another.
// The transaction queue covers READ + CAS + WRITE, not just the atomic write operation.
const transactions = new Map<string, Promise<unknown>>()
let epoch = 0
let purging = false
let admissionPaused = false

export function composerDraftFile(scope: ComposerDraftScope): string {
  return vontraPath('state', 'composer-drafts', `${createHash('sha256').update(composerDraftKey(scope)).digest('hex')}.json`)
}

function requirePersistentScope(scope: ComposerDraftScope): ComposerDraftScope {
  const valid = validateComposerDraftScope(scope)
  if (valid.incognito) throw new Error('Incognito drafts are memory-only.')
  return valid
}

async function read(file: string): Promise<ComposerDraftSnapshot> {
  let exists = false
  try {
    const directory = await fs.lstat(vontraPath('state', 'composer-drafts'))
    if (!directory.isDirectory()) throw new Error('Unexpected draft directory.')
    const entry = await fs.lstat(file)
    if (!entry.isFile()) throw new Error('Unexpected draft file.')
    exists = true
  }
  catch (error: any) { if (error?.code !== 'ENOENT') throw new Error('Draft storage is unavailable; your unsaved text remains in this window.') }
  if (!exists) return emptyComposerDraft()
  const loaded = await readProtectedJson<unknown>(file, undefined, false)
  // A recovery fallback is not an empty draft, and must never be saved over the original.
  if (loaded === undefined) throw new Error('Saved draft is damaged or unavailable; restore its protected recovery copy before saving.')
  try { return validateComposerDraftSnapshot(loaded) }
  catch { throw new Error('Saved draft is damaged; its original bytes have been preserved.') }
}

function transaction<T>(file: string, action: () => Promise<T>): Promise<T> {
  const previous = transactions.get(file) ?? Promise.resolve()
  const next = previous.catch(() => {}).then(action)
  transactions.set(file, next)
  void next.catch(() => {}).finally(() => { if (transactions.get(file) === next) transactions.delete(file) })
  return next
}

export function readComposerDraft(scope: ComposerDraftScope): Promise<ComposerDraftSnapshot> {
  try {
    const file = composerDraftFile(requirePersistentScope(scope))
    return transaction(file, () => read(file))
  } catch (error) { return Promise.reject(error) }
}

export function writeComposerDraft(input: ComposerDraftWrite, validateCurrentOwner: () => void = () => {}): Promise<ComposerDraftWriteResult> {
  try {
    if (purging || admissionPaused) throw new Error('Draft storage is being cleared.')
    const admittedEpoch = epoch
    const valid = validateComposerDraftWrite(input)
    const file = composerDraftFile(requirePersistentScope(valid.scope))
    return transaction(file, async () => {
      if (admittedEpoch !== epoch || purging || admissionPaused) throw new Error('Draft storage was cleared before this save; the old draft was not restored.')
      validateCurrentOwner()
      const previous = await read(file)
      if (previous.revision !== valid.expectedRevision) return { saved: false, draft: previous }
      if (previous.revision === Number.MAX_SAFE_INTEGER) throw new Error('Draft revision limit reached.')
      const draft = { revision: previous.revision + 1, text: valid.text, attachments: valid.attachments, ...(valid.quotes?.length ? { quotes: valid.quotes } : {}), updatedAt: Date.now() }
      // An empty tombstone still carries a revision, preventing a late write from resurrecting sent text.
      await writeProtectedJson(file, draft, { isEmpty: false, prepareData: () => {
        if (admittedEpoch !== epoch || purging || admissionPaused) throw new Error('Draft storage was cleared before this save; the old draft was not restored.')
        validateCurrentOwner()
        return draft
      } })
      return { saved: true, draft }
    })
  } catch (error) { return Promise.reject(error) }
}

export function clearComposerDraft(input: { scope: ComposerDraftScope; expectedRevision: number }, validateCurrentOwner: () => void = () => {}): Promise<ComposerDraftWriteResult> {
  return writeComposerDraft({ ...input, text: '', attachments: [] }, validateCurrentOwner)
}

export async function drainComposerDraftWrites(): Promise<void> {
  while (transactions.size) await Promise.all([...transactions.values()])
}

/** Synchronous admission fence lasts until the enclosing security purge finishes. */
export async function beginComposerDraftPurge(): Promise<void> {
  if (admissionPaused) throw new Error('Draft storage is already paused for clearing.')
  admissionPaused = true
  epoch++
  await Promise.allSettled([...transactions.values()])
}

export function endComposerDraftPurge(): void { admissionPaused = false }

/** Forget controls must also clear coordinator memory before it can schedule another save. */
export async function purgeComposerDrafts(): Promise<void> {
  if (purging) throw new Error('Draft storage is already being cleared.')
  const ownsFence = !admissionPaused
  if (ownsFence) admissionPaused = true
  purging = true
  epoch++
  try {
    await Promise.allSettled([...transactions.values()])
    const directory = vontraPath('state', 'composer-drafts')
    const directoryStat = await fs.lstat(directory).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (directoryStat?.isSymbolicLink()) {
      // Forget removes the link itself, never data in an unrelated target directory.
      await fs.unlink(directory)
      return
    }
    if (directoryStat && !directoryStat.isDirectory()) throw new Error('Unexpected draft storage directory; clearing was not completed.')
    const entries = await fs.readdir(directory, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return []; throw error })
    for (const entry of entries) {
      if (!entry.isFile() || !/^[a-f0-9]{64}\.json(?:\..+)?$/.test(entry.name)) throw new Error('Unexpected draft storage entry; clearing was not completed.')
      await overwriteAndUnlink(vontraPath('state', 'composer-drafts', entry.name))
    }
    await fs.rmdir(directory).catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error })
  } finally { purging = false; if (ownsFence) admissionPaused = false }
}

/** Serialize forgetting with every older save of this exact conversation. */
export function purgeComposerDraft(scope: ComposerDraftScope): Promise<void> {
  const file = composerDraftFile(requirePersistentScope({ ...scope, incognito: false }))
  return transaction(file, async () => {
    const directory = vontraPath('state', 'composer-drafts')
    const directoryStat = await fs.lstat(directory).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (directoryStat && !directoryStat.isDirectory()) throw new Error('Unexpected draft storage directory; clearing was not completed.')
    const filename = file.slice(directory.length + 1)
    const entries = await fs.readdir(directory, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return []; throw error })
    for (const entry of entries) if (entry.name === filename || entry.name.startsWith(`${filename}.`)) {
      if (!entry.isFile()) throw new Error('Unexpected draft storage entry; clearing was not completed.')
      await overwriteAndUnlink(vontraPath('state', 'composer-drafts', entry.name))
    }
  })
}
