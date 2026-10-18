import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { ProjectMemoryCandidate, ProjectMemoryEntry } from '../agent/types'
import { normalizeProjectMemoryEntry } from '../agent/project-memory'
import { statePath, legacyPath, vontraHomeDir } from './paths'
import { readProtectedJson, writeProtectedJson } from './write-infra'
import { overwriteAndUnlink, quarantineMalformedProtectedFile } from './recovery-registry'

export interface ProjectMemoryState { entries: ProjectMemoryEntry[]; candidates: ProjectMemoryCandidate[] }
interface ProtectedMemoryState extends ProjectMemoryState { version: 1; candidateCutoff: number; candidateCutoffs: Record<string, number> }
const queues = new Map<string, Promise<unknown>>()
let epoch = 0
let purging = false
interface MemoryPaths { profile: string; file: string; candidates: string; legacyEntries: string; legacyCandidates: string }
function memoryPaths(): MemoryPaths {
  return { profile: vontraHomeDir(), file: statePath('project-memory.json'), candidates: statePath('project-memory-candidates.json'), legacyEntries: legacyPath('project-memory.json'), legacyCandidates: legacyPath('project-memory-candidates.json') }
}
const detached = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

function validateState(input: unknown): ProtectedMemoryState {
  const value = input as Partial<ProtectedMemoryState> | null
  if (!value || value.version !== 1 || !Array.isArray(value.entries) || !Array.isArray(value.candidates)) throw new Error('Invalid protected memory state')
  const entries = value.entries.map((entry) => {
    if (!entry || typeof entry.id !== 'string' || !entry.id.trim() || typeof entry.projectId !== 'string' || !entry.projectId.trim() || typeof entry.text !== 'string') throw new Error('Invalid protected memory entry')
    return normalizeProjectMemoryEntry(entry)
  })
  const candidateIds = new Set<string>()
  for (const candidate of value.candidates) {
    if (!candidate || typeof candidate.id !== 'string' || !candidate.id.trim() || typeof candidate.projectId !== 'string' || !candidate.projectId.trim() || typeof candidate.text !== 'string' || !['pending', 'approved', 'rejected', 'rolled_back'].includes(candidate.status) || candidateIds.has(candidate.id)) throw new Error('Invalid protected memory candidate')
    candidateIds.add(candidate.id)
  }
  if (new Set(entries.map((entry) => entry.id)).size !== entries.length) throw new Error('Duplicate protected memory identity')
  const candidateCutoff = value.candidateCutoff ?? 0
  if (!Number.isFinite(candidateCutoff) || candidateCutoff < 0) throw new Error('Invalid memory candidate cutoff')
  const candidateCutoffs = value.candidateCutoffs ?? {}
  if (typeof candidateCutoffs !== 'object' || Array.isArray(candidateCutoffs) || Object.values(candidateCutoffs).some((cutoff) => !Number.isFinite(cutoff) || cutoff < 0)) throw new Error('Invalid project memory candidate cutoffs')
  return { version: 1, entries, candidates: value.candidates, candidateCutoff, candidateCutoffs }
}

async function present(file: string): Promise<boolean> {
  try { const stat = await fs.lstat(file); if (!stat.isFile() || stat.isSymbolicLink()) throw new Error('Unexpected protected memory file'); return true }
  catch (error: any) { if (error?.code === 'ENOENT') return false; throw error }
}

async function readRequired(file: string): Promise<unknown> {
  const raw = await readProtectedJson<unknown>(file, undefined, false)
  if (raw === undefined) throw new Error('Protected memory is unavailable; restore its recovery copy before changing memory')
  return raw
}

async function cleanObsoleteCopies(paths: MemoryPaths): Promise<void> {
  const targets = [paths.candidates, paths.legacyEntries, paths.legacyCandidates].filter(Boolean)
  for (const file of targets) await overwriteAndUnlink(file)
}

async function loadState(paths: MemoryPaths, validate: () => void): Promise<ProtectedMemoryState> {
  validate()
  const file = paths.file
  const source = await present(file) ? file : paths.legacyEntries
  const raw = source && await present(source) ? await readRequired(source) : []
  if (!Array.isArray(raw)) {
    let value: ProtectedMemoryState
    try { value = validateState(raw) }
    catch (error) { await quarantineMalformedProtectedFile(file, 'validate memory lifecycle state in'); throw error }
    validate()
    await cleanObsoleteCopies(paths)
    return value
  }
  const candidatesSource = await present(paths.candidates) ? paths.candidates : paths.legacyCandidates
  const candidates = candidatesSource && await present(candidatesSource) ? await readRequired(candidatesSource) : []
  let value: ProtectedMemoryState
  try { value = validateState({ version: 1, entries: raw, candidates }) }
  catch (error) { if (source && await present(source)) await quarantineMalformedProtectedFile(source, 'validate memory lifecycle state in'); throw error }
  // One encrypted file commits entries and review history together. The old split file is
  // never a second source of truth after migration, and is removed before returning.
  validate()
  await writeProtectedJson(file, raw, { isEmpty: false, prepareData: () => { validate(); return value }, validateCommit: validate })
  await cleanObsoleteCopies(paths)
  return value
}

function transaction<T>(file: string, action: () => Promise<T>): Promise<T> {
  const previous = queues.get(file) ?? Promise.resolve()
  const next = previous.catch(() => {}).then(action)
  queues.set(file, next)
  void next.catch(() => {}).finally(() => { if (queues.get(file) === next) queues.delete(file) })
  return next
}

function admission(profile: string, admittedEpoch: number): void {
  if (purging || epoch !== admittedEpoch) throw new Error('Project memory is being cleared; the old memory was not restored')
  if (profile !== vontraHomeDir()) throw new Error('Project memory profile changed before the save completed')
}

export async function loadProjectMemoryState(): Promise<ProjectMemoryState> {
  const paths = memoryPaths(), admittedEpoch = epoch
  admission(paths.profile, admittedEpoch)
  return transaction(paths.file, () => loadState(paths, () => admission(paths.profile, admittedEpoch)))
}

export async function loadProjectMemory(): Promise<ProjectMemoryEntry[]> { return (await loadProjectMemoryState()).entries }
export async function loadProjectMemoryCandidates(): Promise<ProjectMemoryCandidate[]> { return (await loadProjectMemoryState()).candidates }

/** Only explicit host transactions may change durable entries after migration. */
export function saveProjectMemory(entries: ProjectMemoryEntry[]): Promise<void> {
  const snapshot = detached(entries).map(normalizeProjectMemoryEntry)
  const paths = memoryPaths(), admittedEpoch = epoch, { profile, file } = paths
  return transaction(file, async () => {
    admission(profile, admittedEpoch)
    const existed = await present(file)
    const state = await loadState(paths, () => admission(profile, admittedEpoch))
    if (same(snapshot, state.entries)) return
    if (existed) throw new Error('Stale project memory snapshot refused; use a host memory transaction')
    await writeProtectedJson(file, { ...state, entries: snapshot }, { isEmpty: false, validateCommit: () => admission(profile, admittedEpoch) })
  })
}

function backgroundCandidates(state: ProtectedMemoryState, snapshot: ProjectMemoryCandidate[]): ProjectMemoryCandidate[] {
  const ids = new Set(state.candidates.map((candidate) => candidate.id))
  // The cutoff contains only a timestamp, so Forget needs no retained entry/candidate IDs.
  const additions = snapshot.filter((candidate) => !ids.has(candidate.id) && candidate.status === 'pending' && Number.isFinite(candidate.createdAt) && candidate.createdAt > Math.max(state.candidateCutoff, Object.prototype.hasOwnProperty.call(state.candidateCutoffs, candidate.projectId) ? state.candidateCutoffs[candidate.projectId] : 0) && !candidate.durableEntryId && (!candidate.targetEntryId || state.entries.some((entry) => entry.id === candidate.targetEntryId && entry.projectId === candidate.projectId)))
  return [...additions, ...state.candidates]
}

/** Background producers may add pending proposals, never replay reviews or removed targets. */
export function saveProjectMemoryCandidates(candidates: ProjectMemoryCandidate[]): Promise<void> {
  const snapshot = detached(candidates)
  const paths = memoryPaths(), admittedEpoch = epoch, { profile, file } = paths
  return transaction(file, async () => {
    admission(profile, admittedEpoch)
    const state = await loadState(paths, () => admission(profile, admittedEpoch))
    const candidates = backgroundCandidates(state, snapshot)
    if (candidates.length === state.candidates.length) return
    const next = validateState({ ...state, candidates })
    await writeProtectedJson(file, next, { isEmpty: false, validateCommit: () => admission(profile, admittedEpoch) })
  })
}

/** Stage and publish detached live state in the same encrypted atomic commit. */
export function commitProjectMemoryMutation<T>(getState: () => ProjectMemoryState, stage: (state: ProjectMemoryState) => ProjectMemoryState & { result: T }, validate: () => void, publish: (state: ProjectMemoryState) => void, beforeCommit?: () => Promise<void>): Promise<T> {
  const paths = memoryPaths(), admittedEpoch = epoch, { profile, file } = paths
  admission(profile, admittedEpoch)
  validate()
  return transaction(file, async () => {
    admission(profile, admittedEpoch); validate()
    const disk = await loadState(paths, () => { admission(profile, admittedEpoch); validate() })
    admission(profile, admittedEpoch); validate()
    const live = getState()
    const before = detached(live)
    // Failed earlier saves must never turn a stale renderer/runtime projection into truth.
    if (!same(before.entries.map(normalizeProjectMemoryEntry), disk.entries)) throw new Error('Project memory changed on disk; reload before retrying')
    const staged = stage({ entries: detached(disk.entries), candidates: detached(backgroundCandidates(disk, before.candidates)) })
    const removed = disk.candidates.filter((candidate) => !staged.candidates.some((item) => item.id === candidate.id))
    const cutoffs = new Map(Object.entries(disk.candidateCutoffs))
    for (const candidate of removed) cutoffs.set(candidate.projectId, Math.max(cutoffs.get(candidate.projectId) ?? 0, candidate.createdAt))
    const payload = validateState({ version: 1, entries: staged.entries, candidates: staged.candidates, candidateCutoff: disk.candidateCutoff, candidateCutoffs: Object.fromEntries(cutoffs) })
    await beforeCommit?.()
    const validateCommit = () => {
      admission(profile, admittedEpoch); validate()
      if (!same(getState(), before)) throw new Error('Project memory changed during save; refresh before retrying')
    }
    let committed = false
    await writeProtectedJson(file, payload, { isEmpty: false, prepareData: () => { validateCommit(); return payload }, validateCommit, onCommitted: () => {
      publish(payload)
      committed = true
    } })
    if (!committed) throw new Error('Protected memory commit did not complete')
    return staged.result
  })
}

/** Close admission synchronously; wait for admitted writers before security shredding. */
export async function beginProjectMemoryPurge(): Promise<void> {
  purging = true; epoch++
  while (queues.size) await Promise.allSettled([...queues.values()])
}
export function endProjectMemoryPurge(): void { purging = false }

export async function purgeProjectMemoryState(): Promise<void> {
  if (!purging) throw new Error('Project memory purge requires closed admission')
  const paths = memoryPaths()
  await removeProjectMemoryRecoveryCopies(paths)
  await overwriteAndUnlink(paths.file)
  await cleanObsoleteCopies(paths)
  await writeProtectedJson(paths.file, { version: 1, entries: [], candidates: [], candidateCutoff: Date.now() }, { isEmpty: false })
}

/** Forget removes obsolete recovery copies of this collection as well as live payloads. */
export async function removeProjectMemoryRecoveryCopies(paths = memoryPaths()): Promise<void> {
  for (const base of [paths.file, paths.candidates, paths.legacyEntries, paths.legacyCandidates].filter(Boolean)) {
    const names = await fs.readdir(path.dirname(base)).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return []; throw error })
    for (const name of names) if (name.startsWith(`${path.basename(base)}.`)) await overwriteAndUnlink(path.join(path.dirname(base), name))
  }
}
