import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { BackupRecord, WorkspaceBackupData } from '../../../../shared/workspace-backup'
import { validateComposerDraftSnapshot } from '../../../../shared/composer-drafts'
import type { Project, Settings, Thread } from '../../../agent/types'
import { backupDigest, MAX_BLOB_BYTES, MAX_TOTAL_BLOB_BYTES, type BackupSnapshot } from '../../../workspace-backup/archive'
import { validatedRestorePlan } from '../../../workspace-backup/plan-validation'
import { portableRunRecovery } from '../../../workspace-backup/recovery-validation'
import { parseThreadMemoryPolicy } from '../../../../shared/thread-memory-policy'
import { portableBot } from '../../../workspace-backup/portable-records'
import { mergedBackupPrivacyTerms } from '../../../workspace-backup/record-validation'
import type { WorkspaceBackupRepository, WorkspaceRestorePlan } from '../../../workspace-backup/service'
import { captureManagedAttachments, hydrateManagedAttachments, serializeBackupRecord, type ManagedAttachmentSnapshot } from './attachments'
import { MAX_RECOVERY_JSON_BYTES, syncWorkspaceRecoveryFile, syncWorkspaceRecoveryState, writeWorkspaceRecoveryMarker, type WorkspaceRecoveryCheckpoint } from './recovery'

export interface BackupHostState { settings: Settings; threads: Thread[]; projects: Project[] }
export interface BackupHostRepositoryDependencies {
  profileRoot(): string
  getState(): BackupHostState
  setState(state: BackupHostState): void
  persist(state: BackupHostState): Promise<void>
  readProtectedJson<T>(file: string, fallback: T, migratePlaintext?: boolean, requireEncrypted?: boolean): Promise<T>
  writeProtectedJson(file: string, value: unknown): Promise<void>
  validateOwner(): void
}
export interface BackupHostStage {
  root: string
  directory: string
  before: BackupHostState
  drafts: BackupRecord[]
  managed: ManagedAttachmentSnapshot
  plan: WorkspaceRestorePlan
  introducedBlobs: Set<string>
}
const HASH = /^[a-f0-9]{64}$/u
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

/** Profile-contained, no-follow directories only; filenames below are fixed or hash/UUID allowlists. */
export async function backupDirectory(root: string, parts: string[], create = false): Promise<string | null> {
  let current = path.resolve(root)
  const base = await fs.lstat(current)
  if (!base.isDirectory() || base.isSymbolicLink() || await fs.realpath(current) !== current) throw new Error('Backup profile directory is unavailable or linked.')
  for (const part of parts) {
    if (!/^[a-zA-Z0-9_-]+$/u.test(part)) throw new Error('Invalid backup storage namespace.')
    current = path.join(current, part)
    let stat = await fs.lstat(current).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (!stat && create) { await fs.mkdir(current, { mode: 0o700 }); stat = await fs.lstat(current) }
    if (!stat) return null
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('Backup storage namespace is unavailable or linked.')
  }
  return current
}

export function createWorkspaceBackupRepository(deps: BackupHostRepositoryDependencies): WorkspaceBackupRepository<BackupHostStage> {
  const rootNow = () => path.resolve(deps.profileRoot())
  async function readProtected<T>(file: string, fallback: T): Promise<T> {
    const stat = await fs.lstat(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (!stat) return fallback
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 70 * 1024 * 1024) throw new Error('Protected backup data is unavailable or oversized.')
    return deps.readProtectedJson(file, fallback, false, true)
  }
  async function drafts(root: string): Promise<BackupRecord[]> {
    const directory = await backupDirectory(root, ['state', 'composer-drafts'])
    if (!directory) return []
    const files = (await fs.readdir(directory)).filter((file) => /^[a-f0-9]{64}\.json$/u.test(file))
    if (files.length > 10_000) throw new Error('There are too many saved drafts for one backup.')
    const result: BackupRecord[] = []
    for (const file of files.sort()) {
      const snapshot = validateComposerDraftSnapshot(await readProtected<unknown>(path.join(directory, file), undefined))
      result.push({ id: file.slice(0, -5), snapshot: serializeBackupRecord(snapshot) })
    }
    return result
  }
  async function managed(root: string): Promise<ManagedAttachmentSnapshot> {
    const directory = await backupDirectory(root, ['state', 'workspace-managed-attachments'])
    if (!directory) return { attachments: [], blobs: new Map() }
    const metadata = await readProtected<{ attachments: WorkspaceBackupData['attachments'] }>(path.join(directory, 'index.json'), { attachments: [] })
    if (!Array.isArray(metadata.attachments) || metadata.attachments.length > 10_000) throw new Error('Managed backup attachment metadata is invalid.')
    const blobs = new Map<string, Uint8Array>()
    let total = 0
    for (const attachment of metadata.attachments) for (const id of [attachment.originalBlobId, attachment.parsedBlobId]) {
      if (!id || blobs.has(id)) continue
      if (!HASH.test(id)) throw new Error('Managed attachment identity is invalid.')
      const value = await readProtected<{ base64: string } | null>(path.join(directory, `${id}.json`), null)
      if (!value) continue
      if (typeof value.base64 !== 'string' || value.base64.length > Math.ceil(MAX_BLOB_BYTES / 3) * 4 || value.base64.length % 4 || !/^[A-Za-z0-9+/]*={0,2}$/u.test(value.base64)) throw new Error('Managed attachment data is invalid.')
      const bytes = Buffer.from(value.base64, 'base64')
      total += bytes.length
      if (bytes.length > MAX_BLOB_BYTES || total > MAX_TOTAL_BLOB_BYTES || bytes.toString('base64') !== value.base64 || backupDigest(bytes) !== id) throw new Error('Managed attachment integrity check failed.')
      blobs.set(id, bytes)
    }
    return { attachments: metadata.attachments, blobs }
  }
  async function capture(): Promise<BackupSnapshot> {
    deps.validateOwner()
    const root = rootNow(), state = clone(deps.getState()), stateRevisionPayload = JSON.stringify(state)
    const savedDrafts = await drafts(root), stored = await managed(root)
    const threads = state.threads.filter((thread) => !thread.incognito && !thread.orchestratorHostId)
    const lifted = captureManagedAttachments(threads, stored)
    const data: WorkspaceBackupData = { settings: serializeBackupRecord(state.settings), conversations: threads.map(serializeBackupRecord), projects: state.projects.map(serializeBackupRecord),
      bots: (state.settings.agentPresets ?? []).map(serializeBackupRecord), drafts: savedDrafts, attachments: lifted.attachments }
    const revision = createHash('sha256').update(root).update('\0').update(stateRevisionPayload).update('\0').update(JSON.stringify(savedDrafts))
      .update('\0').update(JSON.stringify(stored.attachments)).digest('hex')
    if (rootNow() !== root) throw new Error('Profile changed during backup snapshot.')
    deps.validateOwner()
    return { revision, data, blobs: lifted.blobs }
  }
  async function writeDrafts(root: string, incoming: BackupRecord[], bump = true): Promise<void> {
    if (!incoming.length) return
    const directory = (await backupDirectory(root, ['state', 'composer-drafts'], true))!
    for (const item of incoming) {
      if (typeof item.id !== 'string' || !HASH.test(item.id)) throw new Error('Saved draft identity is invalid.')
      const snapshot = validateComposerDraftSnapshot(item.snapshot)
      const previous = await readProtected<unknown>(path.join(directory, `${item.id}.json`), undefined)
      const previousRevision = previous ? validateComposerDraftSnapshot(previous).revision : 0
      const revision = bump ? Math.max(snapshot.revision, previousRevision) + 1 : snapshot.revision
      if (!Number.isSafeInteger(revision)) throw new Error('Saved draft revision limit reached.')
      await deps.writeProtectedJson(path.join(directory, `${item.id}.json`), { ...snapshot, revision })
    }
  }
  async function writeManaged(root: string, value: ManagedAttachmentSnapshot, stage?: BackupHostStage): Promise<void> {
    if (!value.attachments.length && !value.blobs.size) return
    const directory = (await backupDirectory(root, ['state', 'workspace-managed-attachments'], true))!
    for (const [id, bytes] of value.blobs) {
      if (!HASH.test(id) || backupDigest(bytes) !== id || bytes.byteLength > MAX_BLOB_BYTES) throw new Error('Managed attachment identity is invalid.')
      const file = path.join(directory, `${id}.json`)
      const exists = await fs.lstat(file).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
      if (exists && (!exists.isFile() || exists.isSymbolicLink())) throw new Error('Managed attachment destination is unavailable.')
      if (!exists) stage?.introducedBlobs.add(id)
      await deps.writeProtectedJson(file, { base64: Buffer.from(bytes).toString('base64') })
    }
    await deps.writeProtectedJson(path.join(directory, 'index.json'), { attachments: value.attachments })
  }
  return {
    snapshot: capture,
    async revision() { return (await capture()).revision },
    async stage(plan) {
      deps.validateOwner()
      const root = rootNow()
      plan = validatedRestorePlan(plan)
      mergedBackupPrivacyTerms(deps.getState().settings.confidentialTerms, plan.data.settings.confidentialTerms)
      const privateIds = new Set(deps.getState().threads.filter(thread => thread.incognito).map(thread => thread.id))
      if (plan.data.conversations.some(thread => privateIds.has(String(thread.id)))) throw new Error('A selected identity belongs to an incognito conversation; its privacy boundary cannot be replaced.')
      for (const draft of plan.data.drafts) {
        if (typeof draft.id !== 'string' || !HASH.test(draft.id)) throw new Error('Saved draft identity is invalid.')
        validateComposerDraftSnapshot(draft.snapshot)
      }
      if ((await capture()).revision !== plan.expectedRevision) throw new Error('Workspace or drafts changed before staging.')
      const directory = (await backupDirectory(root, ['state', 'workspace-restore-staging', randomUUID()], true))!
      const before = clone(deps.getState()), savedDrafts = await drafts(root), stored = await managed(root)
      const durableBefore = { ...before, threads: before.threads.filter((thread) => !thread.incognito) }
      const managedDirectory = await backupDirectory(root, ['state', 'workspace-managed-attachments'])
      const existingBlobFiles = managedDirectory ? (await fs.readdir(managedDirectory)).filter((file) => /^[a-f0-9]{64}\.json$/u.test(file)).map((file) => file.slice(0, -5)) : []
      const checkpoint: WorkspaceRecoveryCheckpoint = { schema: 'tensorfold.workspace-recovery/1', profile: backupDigest(Buffer.from(root)), checkpointId: path.basename(directory),
        before: durableBefore, drafts: savedDrafts, attachments: stored.attachments, blobs: [...stored.blobs].map(([id, bytes]) => ({ id, base64: Buffer.from(bytes).toString('base64') })),
        plannedDraftIds: plan.data.drafts.map((draft) => String(draft.id)), plannedBlobIds: [...plan.blobs.keys()], existingBlobFiles }
      try {
        if (Buffer.byteLength(JSON.stringify(checkpoint), 'utf8') > MAX_RECOVERY_JSON_BYTES) throw new Error('The complete recovery checkpoint exceeds the supported restore size.')
        await deps.writeProtectedJson(path.join(directory, 'checkpoint.json'), checkpoint)
        await syncWorkspaceRecoveryFile(path.join(directory, 'checkpoint.json'))
      } catch (error) { await fs.rm(directory, { recursive: true, force: true }); throw error }
      return { root, directory, before, drafts: savedDrafts, managed: stored, plan, introducedBlobs: new Set<string>() }
    },
    async commit(stage, expectedRevision, signal) {
      deps.validateOwner()
      const incoming = validatedRestorePlan(stage.plan).data
      if (signal?.aborted || rootNow() !== stage.root || (await capture()).revision !== expectedRevision) throw new Error('Workspace changed before restore commit.')
      const current = clone(deps.getState()), replace = stage.plan.mode === 'replace'
      const confidentialTerms = mergedBackupPrivacyTerms(current.settings.confidentialTerms, incoming.settings.confidentialTerms)
      if (incoming.conversations.some(thread => current.threads.some(existing => existing.id === thread.id && existing.incognito))) throw new Error('A selected identity became incognito; restore was refused.')
      const merge = <T extends { id: string }>(existing: T[], additions: T[]): T[] => {
        const merged = new Map(existing.map((item) => [item.id, item]))
        for (const item of additions) if (replace || !merged.has(item.id)) merged.set(item.id, item)
        return [...merged.values()]
      }
      const restoredThreads = incoming.conversations.map((record) => {
        if ((record.kind !== 'chat' && record.kind !== 'task') || typeof record.title !== 'string' || !Array.isArray(record.timeline) || !Array.isArray(record.apiMessages)) throw new Error('A saved conversation has an unsupported structure.')
        const previous = current.threads.find(thread => thread.id === record.id)
        const oldMemory = parseThreadMemoryPolicy(previous?.memoryPolicy), restoredMemory = parseThreadMemoryPolicy(record.memoryPolicy)
        const preserveReview = previous?.runRecovery?.retry === 'requires_review' && previous.runRecoveryReviewedId !== previous.runRecovery.runId
        const savedBot = record.bot ?? previous?.bot
        const guardedBot = savedBot === undefined ? undefined : portableBot(savedBot, true)
        if (guardedBot && previous?.bot?.executionTarget === 'vm') guardedBot.executionTarget = 'vm'
        return { ...record, documentSourceIds: [], id: String(record.id), cwd: '', providerId: typeof record.providerId === 'string' ? record.providerId : null, model: typeof record.model === 'string' ? record.model : null,
          ...(guardedBot ? { bot: guardedBot } : {}),
          ...(previous?.phoneLocalOnly ? { phoneLocalOnly: true } : {}), ...(previous?.googleLocalOnly ? { googleLocalOnly: true } : {}),
          memoryPolicy: { useExisting: oldMemory.useExisting && restoredMemory.useExisting, contribute: oldMemory.contribute && restoredMemory.contribute },
          ...(preserveReview ? { runRecovery: portableRunRecovery(previous!.runRecovery) } : {}),
          createdAt: typeof record.createdAt === 'number' ? record.createdAt : Date.now(), updatedAt: Date.now(), status: 'idle' } as unknown as Thread
      })
      const restoredProjects = incoming.projects.map((record) => ({ memory: '', team: '', mode: 'solo', ...record, ...(current.projects.find(project => project.id === record.id)?.localOnly ? { localOnly: true } : {}), id: String(record.id), cwd: '', createdAt: typeof record.createdAt === 'number' ? record.createdAt : Date.now(), updatedAt: Date.now() })) as Project[]
      const settings = { ...current.settings, ...incoming.settings, ...(confidentialTerms !== undefined ? { confidentialTerms } : {}), agentPresets: merge(current.settings.agentPresets ?? [], incoming.bots as unknown as Settings['agentPresets']) } as Settings
      const restoredManaged = { attachments: merge(stage.managed.attachments, incoming.attachments), blobs: new Map([...stage.managed.blobs, ...stage.plan.blobs]) }
      hydrateManagedAttachments(restoredThreads, restoredManaged)
      const next: BackupHostState = { settings, threads: merge(current.threads, restoredThreads), projects: merge(current.projects, restoredProjects) }
      const existingDrafts = new Map(stage.drafts.map((item) => [String(item.id), item]))
      const changedDrafts = incoming.drafts.filter((item) => replace || !existingDrafts.has(String(item.id)))
      deps.validateOwner()
      await deps.persist(next)
      await writeManaged(stage.root, restoredManaged, stage)
      await writeDrafts(stage.root, changedDrafts)
      deps.validateOwner()
      if (signal?.aborted || rootNow() !== stage.root) throw new Error('Restore cancelled or profile changed during commit.')
      deps.setState(next)
      await syncWorkspaceRecoveryState(stage.root)
      await writeWorkspaceRecoveryMarker(deps, stage.directory, stage.root, 'committed')
    },
    async rollback(stage) {
      if (rootNow() !== stage.root) throw new Error('Restore recovery belongs to a different profile.')
      await writeWorkspaceRecoveryMarker(deps, stage.directory, stage.root, 'rolling-back')
      await deps.persist(stage.before)
      await writeDrafts(stage.root, stage.drafts)
      const directory = await backupDirectory(stage.root, ['state', 'workspace-managed-attachments'])
      if (directory) {
        await deps.writeProtectedJson(path.join(directory, 'index.json'), { attachments: stage.managed.attachments })
        for (const id of stage.introducedBlobs) if (HASH.test(id)) await fs.unlink(path.join(directory, `${id}.json`)).catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error })
      }
      // Newly introduced draft identities are removed; older unsent draft bodies are restored.
      const beforeDrafts = new Set(stage.drafts.map((item) => String(item.id)))
      const draftDirectory = await backupDirectory(stage.root, ['state', 'composer-drafts'])
      if (draftDirectory) for (const item of stage.plan.data.drafts) if (typeof item.id === 'string' && HASH.test(item.id) && !beforeDrafts.has(item.id)) {
        await fs.unlink(path.join(draftDirectory, `${item.id}.json`)).catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error })
      }
      deps.setState(stage.before)
      await syncWorkspaceRecoveryState(stage.root)
      await writeWorkspaceRecoveryMarker(deps, stage.directory, stage.root, 'rolled-back')
    },
    async discard(stage) {
      if (rootNow() !== stage.root || path.dirname(stage.directory) !== path.join(stage.root, 'state', 'workspace-restore-staging') || !/^[a-f0-9-]{36}$/u.test(path.basename(stage.directory))) throw new Error('Invalid restore staging identity.')
      await backupDirectory(stage.root, ['state', 'workspace-restore-staging'])
      await fs.rm(stage.directory, { recursive: true, force: true })
    },
  }
}
