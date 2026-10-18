import { randomUUID } from 'node:crypto'
import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import type { WorkspaceBackupInventory, WorkspaceBackupSelection, WorkspaceRestoreMode } from '../../../../shared/workspace-backup'
import { createWorkspaceBackupService, WorkspaceRestoreError } from '../../../workspace-backup/service'
import { MAX_BACKUP_BYTES } from '../../../workspace-backup/crypto'
import { atomicExport } from '../../../store/atomic-export'
import { createWorkspaceBackupRepository, type BackupHostRepositoryDependencies } from './repository'

interface Sender { once?(event: 'destroyed', callback: () => void): unknown }
export interface WorkspaceBackupRuntimeDependencies<Event extends { sender: Sender }> extends Omit<BackupHostRepositoryDependencies, 'validateOwner'> {
  handle(channel: string, listener: (event: Event, ...args: any[]) => unknown): void
  /** Owner role plus the host's trusted-frame gate; never expose these channels on paired REST. */
  authorize(event: Event): void
  assertIdle(): void
  acquireLease(): Promise<{ release(): void }>
  flushDrafts(event: Event): Promise<void>
  pauseDraftWrites(): Promise<void>
  resumeDraftWrites(): void
  chooseOpen(event: Event): Promise<string | null>
  chooseSave(event: Event): Promise<string | null>
  publishRestored(): void
}

export async function readChosenBackup(file: string): Promise<Uint8Array> {
  const handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW)
  try {
    const stat = await handle.stat()
    if (!stat.isFile() || stat.size < 1 || stat.size > MAX_BACKUP_BYTES) throw new Error('Choose a supported backup file within the archive size limit.')
    const bytes = Buffer.alloc(stat.size)
    let read = 0
    while (read < bytes.length) {
      const result = await handle.read(bytes, read, bytes.length - read, read)
      if (!result.bytesRead) throw new Error('Backup file changed while reading; choose it again.')
      read += result.bytesRead
    }
    const final = await handle.stat()
    if (final.size !== stat.size || final.mtimeMs !== stat.mtimeMs || final.ino !== stat.ino) { bytes.fill(0); throw new Error('Backup file changed while reading; choose it again.') }
    return bytes
  } finally { await handle.close() }
}

export async function chosenDestination(file: string) {
  const parent = await fs.realpath(path.dirname(file))
  const destination = path.join(parent, path.basename(file))
  const existing = await fs.lstat(destination).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
  if (existing && (!existing.isFile() || existing.isSymbolicLink() || existing.nlink !== 1)) throw new Error('Choose a regular, unlinked backup file destination.')
  return { destination, existing }
}

export async function writeChosenBackup(chosen: Awaited<ReturnType<typeof chosenDestination>>, bytes: Uint8Array, authorize: () => void): Promise<void> {
  const { destination, existing } = chosen
  const current = (await chosenDestination(destination)).existing
  if (existing ? !current || current.dev !== existing.dev || current.ino !== existing.ino || current.mtimeMs !== existing.mtimeMs || current.size !== existing.size : current !== null) throw new Error('The backup destination changed; choose it again.')
  authorize()
  await atomicExport(destination, bytes, MAX_BACKUP_BYTES, authorize)
}

export function registerWorkspaceBackupRuntime<Event extends { sender: Sender }>(deps: WorkspaceBackupRuntimeDependencies<Event>) {
  type Session = { profile: string; service: ReturnType<typeof createWorkspaceBackupService>; files: Map<string, { path: string; fileId: string }>; abort?: AbortController }
  const sessions = new WeakMap<Sender, Session>()
  const sessionFor = (event: Event): Session => {
    deps.authorize(event)
    const profile = path.resolve(deps.profileRoot())
    let session = sessions.get(event.sender)
    if (session?.profile !== profile) { session?.abort?.abort(); session?.service.dispose(); session = undefined }
    if (!session) {
      const repository = createWorkspaceBackupRepository({ ...deps, validateOwner: () => {
        deps.authorize(event)
        if (path.resolve(deps.profileRoot()) !== profile) throw new Error('Backup profile changed; reopen the preview.')
      } })
      session = { profile, service: createWorkspaceBackupService({ repository }), files: new Map() }
      sessions.set(event.sender, session)
      const owned = session
      event.sender.once?.('destroyed', () => { owned.abort?.abort(); owned.service.dispose(); owned.files.clear(); sessions.delete(event.sender) })
    }
    return session
  }
  const stable = async <T>(event: Event, work: (session: Session) => Promise<T>): Promise<T> => {
    const session = sessionFor(event)
    deps.assertIdle()
    const lease = await deps.acquireLease()
    let paused = false, retainFence = false
    try {
      deps.authorize(event); deps.assertIdle()
      await deps.flushDrafts(event)
      await deps.pauseDraftWrites(); paused = true
      deps.authorize(event)
      if (path.resolve(deps.profileRoot()) !== session.profile) throw new Error('Backup profile changed; reopen the preview.')
      return await work(session)
    } catch (error) {
      if (error instanceof WorkspaceRestoreError && !error.rolledBack) retainFence = true
      throw error
    } finally { if (!retainFence) { if (paused) deps.resumeDraftWrites(); lease.release() } }
  }
  deps.handle('workspace-backup:list', async (event): Promise<WorkspaceBackupInventory> => {
    deps.authorize(event)
    const state = deps.getState()
    return { available: true, conversations: state.threads.filter((thread) => !thread.incognito && !thread.orchestratorHostId).map((thread) => ({ id: thread.id, title: thread.title, archived: !!thread.archivedAt })),
      projects: state.projects.map((project) => ({ id: project.id, name: project.name })), bots: (state.settings.agentPresets ?? []).map((bot) => ({ id: bot.id, name: bot.name })) }
  })
  deps.handle('workspace-backup:preview-export', (event, selection: WorkspaceBackupSelection) => stable(event, session => session.service.previewExport(selection)))
  deps.handle('workspace-backup:export', (event, input: { approvalId: string; password: string }) => stable(event, async (session) => {
    const file = await deps.chooseSave(event)
    if (!file) return { saved: false, cancelled: true }
    deps.authorize(event)
    const chosen = await chosenDestination(file)
    const bytes = await session.service.exportEncrypted(input)
    try {
      await writeChosenBackup(chosen, bytes, () => {
        deps.authorize(event)
        if (sessions.get(event.sender) !== session || path.resolve(deps.profileRoot()) !== session.profile) throw new Error('Backup session changed before saving.')
      })
      return { saved: true, cancelled: false, bytes: bytes.byteLength }
    } finally { bytes.fill(0) }
  }))
  deps.handle('workspace-backup:preview-restore', (event, input: { password: string; mode: WorkspaceRestoreMode }) => stable(event, async (session) => {
    const file = await deps.chooseOpen(event)
    if (!file) return null
    deps.authorize(event)
    const bytes = await readChosenBackup(file), fileId = randomUUID()
    try {
      const preview = await session.service.previewRestore({ ...input, fileId, bytes })
      session.files.set(preview.approvalId, { path: file, fileId })
      while (session.files.size > 2) session.files.delete(session.files.keys().next().value!)
      return preview
    } finally { bytes.fill(0) }
  }))
  deps.handle('workspace-backup:apply-restore', (event, input: { approvalId: string }) => stable(event, async (session) => {
    const file = session.files.get(input?.approvalId)
    session.files.delete(input?.approvalId)
    if (!file) throw new Error('Restore preview is unavailable; choose the file again.')
    const bytes = await readChosenBackup(file.path)
    session.abort = new AbortController()
    try {
      const result = await session.service.applyRestore({ approvalId: input.approvalId, fileId: file.fileId, bytes, signal: session.abort.signal })
      deps.authorize(event)
      if (result.restored) {
        try { deps.publishRestored() } catch { result.warnings.push('The workspace was restored; reopen the window to refresh its display.') }
      }
      return result
    } finally { bytes.fill(0); session.abort = undefined }
  }))
  deps.handle('workspace-backup:cancel', async (event) => {
    deps.authorize(event)
    const session = sessions.get(event.sender)
    session?.abort?.abort(); session?.service.dispose(); session?.files.clear(); sessions.delete(event.sender)
  })
}
