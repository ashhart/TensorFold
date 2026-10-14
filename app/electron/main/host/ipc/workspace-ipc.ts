import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import { spawn } from 'node:child_process'
import path from 'node:path'
import { estimateTokens, kfmt } from '../../../agent/compaction'
import { compactThread } from '../../../agent/loop'
import { discoverOmpBinary } from '../../../agent/omp-rpc'
import { cliSpawnEnv } from '../../../agent/cli-spawn-env'
import { detachedProcessGroup, terminateChildProcess } from '../../../agent/child-process-lifecycle'
import type { EventSpineHost } from '../../../agent/event-spine-host'
import { resolveProviderModel } from '../../../agent/provider-accounts'
import type { ArchiveKind, RendererThreadPatch, Thread, WorkspaceLocation } from '../../../agent/types'
import type { RemoteControlEvent } from '../../../agent/types'
import { auditPreview } from '../../../agent/security'
import { store } from '../../state'
import { archiveEntity, archiveSnapshot, isActive, restoreEntity } from '../archive'
import { auditEvent, requireSecurityRole } from '../audit'
import { dirExists } from '../workspace'
import { handle } from '../ipc'
import { persistThreads } from '../persistence'
import { runtimeProviderWithFreshModels, resolveProjectProvider, selectedProviderAccountId } from '../provider-runtime'
import { assertSshThreadTransportSupported } from '../thread-runtime'
import type { WorkspaceMutationGuard } from '../workspace-mutation-guard'
import { captureBotConfig } from '../../../../shared/bot-policy'
import { saveThreads } from '../../../store'
import { assertBotLifecycleIdle } from '../bot-run-boundary'

const RENDERER_THREAD_PATCH_KEYS = new Set([
  'goal', 'goalStatus', 'plan', 'notepad', 'annotations', 'diffReviewedAt', 'incognito',
])

/**
 * The renderer owns only user-authored presentation/planning fields. Runtime state,
 * workspace identity, transcripts, timeline mutation evidence, resume cursors, and
 * provider state are main-process owned and cannot cross the generic patch seam.
 */
export function sanitizeRendererThreadPatch(input: unknown): RendererThreadPatch {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Thread patch must be an object.')
  const raw = input as Record<string, unknown>
  const rejected = Object.keys(raw).filter((key) => !RENDERER_THREAD_PATCH_KEYS.has(key))
  if (rejected.length) throw new Error(`Thread patch contains host-owned fields: ${rejected.sort().join(', ')}`)
  const out: RendererThreadPatch = {}
  const optionalString = (key: 'goal' | 'notepad' | 'annotations', max: number) => {
    const value = raw[key]
    if (value === undefined) { if (Object.prototype.hasOwnProperty.call(raw, key)) out[key] = undefined; return }
    if (typeof value !== 'string') throw new Error(`${key} must be text.`)
    out[key] = value.slice(0, max)
  }
  optionalString('goal', 8_000)
  optionalString('notepad', 200_000)
  optionalString('annotations', 200_000)
  if (Object.prototype.hasOwnProperty.call(raw, 'goalStatus')) {
    if (raw.goalStatus !== undefined && !['active', 'complete', 'paused'].includes(String(raw.goalStatus))) throw new Error('Invalid goal status.')
    out.goalStatus = raw.goalStatus as RendererThreadPatch['goalStatus']
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'plan')) {
    if (raw.plan !== undefined && !Array.isArray(raw.plan)) throw new Error('Plan must be an array.')
    out.plan = raw.plan === undefined ? undefined : raw.plan.slice(0, 200).map((step, index) => {
      if (!step || typeof step !== 'object') throw new Error(`Invalid plan step ${index + 1}.`)
      const candidate = step as Record<string, unknown>
      if (typeof candidate.id !== 'string' || typeof candidate.text !== 'string' || typeof candidate.done !== 'boolean') {
        throw new Error(`Invalid plan step ${index + 1}.`)
      }
      return { id: candidate.id.slice(0, 200), text: candidate.text.slice(0, 8_000), done: candidate.done }
    })
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'diffReviewedAt')) {
    const value = raw.diffReviewedAt
    if (value !== undefined && value !== null && (typeof value !== 'number' || !Number.isFinite(value) || value < 0)) throw new Error('Invalid diff review timestamp.')
    out.diffReviewedAt = value as number | null | undefined
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'incognito')) {
    if (typeof raw.incognito !== 'boolean') throw new Error('incognito must be boolean.')
    out.incognito = raw.incognito
  }
  return out
}

interface RemoteServerLike {
  publish(event: Pick<RemoteControlEvent, 'kind' | 'data'>): void
}

export interface WorkspaceIpcDeps {
  remoteServer: RemoteServerLike
  spine: () => EventSpineHost | null
  aborters: Map<string, AbortController>
  roomAborters: Map<string, AbortController>
  abortMissionsForArchive(target: { projectId?: string | null; threadId?: string | null }): void
  stopLoopsForArchive(kind: ArchiveKind, id: string): Promise<unknown>
  killProjectTerminals(projectId: string): void
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  emitToRenderer(event: any): void
}

export function registerWorkspaceIpc(deps: WorkspaceIpcDeps) {
  const privateCreateCommands = new Map<string, string>()
  const {
    remoteServer,
    spine,
    aborters,
    roomAborters,
    abortMissionsForArchive,
    stopLoopsForArchive,
    killProjectTerminals,
    workspaceMutationGuard,
    emitToRenderer,
  } = deps

handle('threads:get', async () => store.threads.filter(isActive))
handle('archives:get', async () => archiveSnapshot())
handle('archives:archive', async (_e, { kind, id }: { kind: 'project' | 'thread' | 'room' | 'ticket'; id: string }) => {
  requireSecurityRole('operator', 'archive workspace records')
  return archiveEntity(kind, id, {
    abortProjectTerminals: killProjectTerminals,
    abortMissions: abortMissionsForArchive,
    stopLoops: stopLoopsForArchive,
    abortThread: (threadId) => aborters.get(threadId)?.abort(),
    abortRoom: (roomId) => roomAborters.get(roomId)?.abort(),
    publish: (recordKind, data) => remoteServer.publish({ kind: recordKind, data }),
  })
})
handle('archives:restore', async (_e, { kind, id }: { kind: 'project' | 'thread' | 'room' | 'ticket'; id: string }) => {
  requireSecurityRole('operator', 'restore workspace records')
  return restoreEntity(kind, id, { publish: (recordKind, data) => remoteServer.publish({ kind: recordKind, data }) })
})

handle('threads:create', async (_e, { cwd, kind = 'task', projectId = null, commandId, incognito = false }: { cwd: string; kind?: 'task' | 'chat'; projectId?: string | null; commandId?: string; incognito?: boolean }) => {
  requireSecurityRole('operator', 'create threads')
  if (typeof incognito !== 'boolean') throw new Error('Incognito must be selected as a boolean option.')
  // Exactly-once via the spine (VST-034): a retried create with the same commandId (lost
  // IPC reply, reconnect replay) returns the thread the accepted receipt points at instead
  // of minting a duplicate.
  const spineHost = incognito ? null : spine()
  const cmdId = typeof commandId === 'string' && commandId.trim() ? commandId.trim() : null
  if (cmdId && cmdId.length > 160) throw new Error('Thread creation command is too long.')
  const priorCreatedThread = () => {
    if (!cmdId) return null
    if (incognito) {
      const existingId = privateCreateCommands.get(cmdId)
      return existingId ? store.threads.find((thread) => thread.id === existingId && thread.incognito) ?? null : null
    }
    const prior = spineHost?.receipt(cmdId)
    const existing = prior?.status === 'accepted' ? store.threads.find((thread) => thread.id === prior.streamId) : null
    if (existing?.incognito) throw new Error('That creation command belongs to a private chat.')
    return existing ?? null
  }
  const prior = priorCreatedThread()
  if (prior) return prior
  const project = projectId ? store.projects.find((p) => p.id === projectId) : null
  if (incognito && project?.executionTarget === 'orchestrator') throw new Error('Incognito tasks run in this desktop app; choose This Mac before starting a private task.')
  if (projectId && (!project || project.archivedAt)) throw new Error('Choose an active project before creating a thread.')
  const projectWorkspace = project?.workspace ?? null
  const candidateCwd = cwd || project?.cwd || (projectWorkspace?.kind === 'ssh' ? projectWorkspace.path : '') || store.settings.defaultCwd || ''
  // Don't carry a stale/missing folder (e.g. a deleted default workspace) into a new
  // thread — fall back to "no folder" so the composer prompts to choose one.
  const selectedCwd = project?.executionTarget === 'orchestrator'
    ? candidateCwd
    : projectWorkspace?.kind === 'ssh' && candidateCwd === projectWorkspace.path
    ? candidateCwd
    : candidateCwd && (await dirExists(candidateCwd)) ? candidateCwd : ''
  const resolved = project ? resolveProjectProvider(project) : null
  const baseProvider = resolved?.baseProvider ?? store.settings.providers.find((p) => p.id === store.settings.activeProviderId)
  const providerAccountId = resolved?.providerAccountId ?? (baseProvider ? selectedProviderAccountId(baseProvider, store.settings.activeProviderAccountId ?? undefined) : null)
  const model = resolved?.model ?? (baseProvider ? resolveProviderModel(baseProvider, providerAccountId, store.settings.activeModel) : store.settings.activeModel)
  // Folder checks above yield; a simultaneous retry may have completed meanwhile.
  const concurrent = priorCreatedThread()
  if (concurrent) return concurrent
  const t: Thread = {
    id: randomUUID(),
    ...(incognito ? { incognito: true } : {}),
    title: kind === 'chat' ? 'New chat' : 'New task',
    kind,
    projectId: projectId ?? null,
    cwd: selectedCwd,
    workspace: projectWorkspace,
    providerId: baseProvider?.id ?? store.settings.activeProviderId,
    providerAccountId,
    model,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: 'idle',
    apiMessages: [],
    timeline: [],
  }
  store.threads.unshift(t)
  if (incognito && cmdId) {
    privateCreateCommands.set(cmdId, t.id)
    if (privateCreateCommands.size > 500) privateCreateCommands.delete(privateCreateCommands.keys().next().value as string)
  }
  persistThreads()
  spineHost?.executeCommand(
    { commandId: cmdId ?? randomUUID(), aggregate: 'thread', streamId: t.id, kind: 'thread.create', actor: 'user' },
    () => ({ aggregate: 'thread', streamId: t.id, type: 'thread.created', payload: { kind: t.kind, projectId: t.projectId ?? null } }),
  )
  store.win?.webContents.send('thread:update', t)
  remoteServer.publish({ kind: 'thread', data: t })
  return t
})

handle('bots:createChat', async (_e, { presetId }: { presetId: string }) => {
  requireSecurityRole('operator', 'start bot chats')
  const preset = store.settings.agentPresets.find((candidate) => candidate.id === presetId)
  if (!preset) throw new Error('Bot not found.')
  assertBotLifecycleIdle(preset.id)
  const bot = captureBotConfig(preset, store.settings.providers)
  const t: Thread = {
    id: randomUUID(), title: `${bot.name} chat`, kind: 'chat', projectId: null,
    cwd: bot.executionTarget === 'vm' ? '/workspace' : '', workspace: null, bot,
    providerId: bot.providerId, providerAccountId: bot.providerAccountId, model: bot.model,
    createdAt: Date.now(), updatedAt: Date.now(), status: 'idle', apiMessages: [], timeline: [],
  }
  const lease = workspaceMutationGuard.reserveStart({ threadId: t.id, workspaceKeys: [] })
  try {
    store.threads.unshift(t)
    try { await saveThreads(store.threads) } catch (error) { store.threads = store.threads.filter((candidate) => candidate.id !== t.id); throw error }
    store.win?.webContents.send('thread:update', t)
    remoteServer.publish({ kind: 'thread', data: t })
    return t
  } finally { lease.release() }
})

handle('threads:delete', async (_e, id: string) => {
  requireSecurityRole('operator', 'delete threads')
  const deletingThread = store.threads.find((thread) => thread.id === id)
  if (!deletingThread?.incognito) spine()?.executeCommand(
    { commandId: randomUUID(), aggregate: 'thread', streamId: id, kind: 'thread.archive', actor: 'user' },
    () => ({ aggregate: 'thread', streamId: id, type: 'thread.archived', payload: {} }),
  )
  await archiveEntity('thread', id, {
    abortThread: (threadId) => aborters.get(threadId)?.abort(),
    abortMissions: abortMissionsForArchive,
    stopLoops: stopLoopsForArchive,
    publish: (recordKind, data) => remoteServer.publish({ kind: recordKind, data }),
  })
})

handle('threads:rename', async (_e, { id, title }: { id: string; title: string }) => {
  requireSecurityRole('operator', 'rename threads')
  const finalTitle = String(title ?? '').trim() || 'Untitled'
  const t = store.threads.find((t) => t.id === id)
  if (t?.archivedAt) throw new Error('Restore the archived thread before renaming it.')
  if (t) {
    t.title = finalTitle
    t.updatedAt = Date.now()
    persistThreads()
    store.win?.webContents.send('thread:update', t)
    remoteServer.publish({ kind: 'thread', data: t })
  }
})

handle('threads:setEngine', async (_e, { id, engine }: { id: string; engine: 'native' | 'omp' }) => {
  requireSecurityRole('operator', 'configure threads')
  if (engine !== 'native' && engine !== 'omp') throw new Error('Engine must be "native" or "omp".')
  const t = store.threads.find((t) => t.id === id)
  if (t?.archivedAt) throw new Error('Restore the archived thread before changing its engine.')
  if (t) {
    if (t.bot) throw new Error('Bot chats use TensorFold’s permission checks; start a new chat to change engines.')
    if (engine === 'omp') {
      const binary = discoverOmpBinary()
      if (!existsSync(binary)) throw new Error('The omp CLI is not installed. Install @oh-my-pi/pi-coding-agent first.')
    }
    t.engine = engine === 'native' ? undefined : engine
    t.updatedAt = Date.now()
    persistThreads()
    store.win?.webContents.send('thread:update', t)
    remoteServer.publish({ kind: 'thread', data: t })
  }
})

handle('omp:status', async () => {
  const binary = discoverOmpBinary()
  if (!existsSync(binary)) return { available: false as const, binary }
  const version = await new Promise<string>((resolve) => {
    const child = spawn(binary, ['--version'], { env: cliSpawnEnv(), detached: detachedProcessGroup() })
    let out = ''
    const timer = setTimeout(() => terminateChildProcess(child), 5000)
    child.stdout?.setEncoding('utf8')
    child.stdout?.on('data', (chunk: string) => { out += chunk })
    child.on('error', () => { clearTimeout(timer); resolve('') })
    child.on('close', () => { clearTimeout(timer); resolve(out.trim().split('\n')[0] ?? '') })
  })
  return { available: true as const, binary, version }
})

handle('threads:configure', async (_e, { id, providerId, model, providerAccountId }: { id: string; providerId: string; model: string; providerAccountId?: string | null }) => {
  requireSecurityRole('operator', 'configure threads')
  const t = store.threads.find((t) => t.id === id)
  if (t?.archivedAt) throw new Error('Restore the archived thread before configuring it.')
  if (t) {
    if (t.bot) throw new Error('This chat keeps its bot’s saved model; edit the bot and start a new chat to change it.')
    const provider = store.settings.providers.find((p) => p.id === providerId)
    const accountId = provider ? selectedProviderAccountId(provider, providerAccountId) : providerAccountId ?? null
    t.providerId = providerId
    t.providerAccountId = accountId
    t.model = provider ? (resolveProviderModel(provider, accountId, model) ?? model) : model
    t.modelRoute = null
    persistThreads()
    store.win?.webContents.send('thread:update', t)
    remoteServer.publish({ kind: 'thread', data: t })
  }
})

// Generic thread patch is deliberately narrow: only renderer-owned planning/notes
// fields cross this seam. Workspace/runtime/timeline state has dedicated host APIs.
handle('threads:patch', async (_e, { id, patch }: { id: string; patch: RendererThreadPatch }) => {
  requireSecurityRole('operator', 'patch threads')
  const t = store.threads.find((t) => t.id === id)
  if (t?.archivedAt) throw new Error('Restore the archived thread before patching it.')
  if (t) {
    const safe = sanitizeRendererThreadPatch(patch)
    if (safe.incognito === false && t.incognito) throw new Error('An incognito thread cannot be converted into a persisted thread.')
    if (safe.incognito === true && (t.timeline.length > 0 || t.apiMessages.length > 0 || t.status === 'running' || aborters.has(t.id))) throw new Error('Incognito mode must be selected before a thread starts.')
    Object.assign(t, safe, { updatedAt: Date.now() })
    persistThreads()
    store.win?.webContents.send('thread:update', t)
    remoteServer.publish({ kind: 'thread', data: t })
  }
  return t
})

handle('threads:setWorkspace', async (_e, input: { id: string; workspace?: WorkspaceLocation | null; cwd?: string | null }) => {
  requireSecurityRole('operator', 'change a thread workspace')
  const t = store.threads.find((candidate) => candidate.id === String(input?.id ?? ''))
  if (!t) throw new Error('Thread not found.')
  if (t.archivedAt) throw new Error('Restore the archived thread before changing its workspace.')
  if (t.bot) throw new Error('Bot chats keep their captured execution location; start a new bot chat to change it.')
  if (aborters.has(t.id)) throw new Error('Stop the running thread before changing its workspace.')
  const targetLease = workspaceMutationGuard.reserveStart({ threadId: t.id })
  try {
  const requested = input?.workspace ?? null
  if (requested && requested.kind !== 'local' && requested.kind !== 'ssh') throw new Error('Invalid workspace kind.')
  let workspace: WorkspaceLocation | null = null
  let cwd = ''
  if (requested?.kind === 'ssh') {
    const hostId = String(requested.hostId ?? '').trim()
    const remotePath = String(requested.path ?? '').trim()
    if (!store.settings.sshHosts.some((host) => host.id === hostId)) throw new Error('Choose a configured SSH host.')
    if (!path.posix.isAbsolute(remotePath) || remotePath.includes('\0')) throw new Error('Choose an absolute remote workspace path.')
    workspace = { kind: 'ssh', hostId, path: path.posix.normalize(remotePath) }
    cwd = workspace.path
  } else {
    const localPath = String(requested?.path ?? input?.cwd ?? t.cwd ?? '').trim()
    if (localPath) {
      if (!await dirExists(localPath)) throw new Error('Choose an existing local workspace folder.')
      cwd = await fs.realpath(localPath)
    }
    workspace = requested?.kind === 'local' && cwd ? { kind: 'local', path: cwd } : null
  }
  t.cwd = cwd
  t.workspace = workspace
  t.updatedAt = Date.now()
  persistThreads()
  store.win?.webContents.send('thread:update', t)
  remoteServer.publish({ kind: 'thread', data: t })
  await auditEvent('thread.workspace_changed', { threadId: auditPreview(t.id), kind: workspace?.kind ?? 'local', cwd: auditPreview(cwd) })
  return t
  } finally {
    targetLease.release()
  }
})

// Append a single notice to a thread's timeline IN MAIN (the timeline owner) and echo it via
// item_add. Used for things like the "Model changed to X" note, so the renderer never has to
// round-trip its whole timeline back (which would clobber main's copy with a mid-stream snapshot).
handle('threads:appendNotice', async (_e, { id, text }: { id: string; text: string }) => {
  requireSecurityRole('operator', 'annotate threads')
  const t = store.threads.find((t) => t.id === id)
  if (!t) return
  if (t.archivedAt) throw new Error('Restore the archived thread before annotating it.')
  const item = { id: randomUUID(), kind: 'notice' as const, text, createdAt: Date.now() }
  t.timeline.push(item); t.updatedAt = Date.now(); persistThreads()
  emitToRenderer({ runId: '', threadId: id, type: 'item_add', item })
  remoteServer.publish({ kind: 'thread', data: t })
})

// Manual /compact: summarize older turns now and drop a notice in the timeline.
handle('thread:compact', async (_e, id: string) => {
  requireSecurityRole('operator', 'compact thread history')
  const thread = store.threads.find((t) => t.id === id)
  if (!thread) throw new Error('Thread not found')
  if (thread.archivedAt) throw new Error('Restore the archived thread before compacting it.')
  // compactThread writes back a snapshot taken before its model call, which would drop
  // whatever a live run appended meanwhile (and can orphan tool calls).
  if (aborters.has(thread.id)) throw new Error('Stop the running thread before compacting it.')
  const baseProvider = store.settings.providers.find((p) => p.id === (thread.providerId ?? store.settings.activeProviderId))
  if (!baseProvider) throw new Error('Configure a provider and model first.')
  const compactWorkspace = thread.workspace ?? store.projects.find((project) => project.id === thread.projectId)?.workspace ?? null
  assertSshThreadTransportSupported(compactWorkspace, baseProvider)
  const provider = await runtimeProviderWithFreshModels(baseProvider, thread.providerAccountId)
  const model = resolveProviderModel(provider, null, thread.model, store.settings.activeModel)
  if (!model) throw new Error('Configure a provider and model first.')
  const before = estimateTokens(thread.apiMessages)
  const entry = await compactThread(thread, provider, model, new AbortController().signal, 'manual')
  const after = estimateTokens(thread.apiMessages)
  const item = { id: randomUUID(), kind: 'notice' as const, text: entry ? `Context compacted. Folded ${entry.messagesFolded} earlier messages (~${kfmt(before)} -> ~${kfmt(after)} tokens).` : 'Nothing to compact yet.', createdAt: Date.now() }
  thread.timeline.push(item); thread.updatedAt = Date.now(); persistThreads()
  emitToRenderer({ runId: '', threadId: id, type: 'item_add', item })
  remoteServer.publish({ kind: 'thread', data: thread })
  return { ok: true, thread }
})

handle('threads:setProject', async (_e, { id, projectId }: { id: string; projectId: string | null }) => {
  requireSecurityRole('operator', 'assign threads to projects')
  const t = store.threads.find((t) => t.id === id)
  if (t?.archivedAt) throw new Error('Restore the archived thread before assigning its project.')
  if (t) {
    if (t.bot) throw new Error('Bot chats keep their captured execution location; start a new bot chat to choose a project.')
    if (aborters.has(t.id)) throw new Error('Stop the running thread before assigning its project.')
    const targetLease = workspaceMutationGuard.reserveStart({ threadId: t.id })
    try {
    const project = projectId ? store.projects.find((p) => p.id === projectId) : null
    if (projectId && (!project || project.archivedAt)) throw new Error('Threads can only move to an active project.')
    t.projectId = projectId
    if (project) t.cwd = project.cwd || (project.workspace?.kind === 'ssh' ? project.workspace.path : '')
    t.workspace = project?.workspace ?? null
    t.updatedAt = Date.now()
    persistThreads()
    store.win?.webContents.send('thread:update', t)
    remoteServer.publish({ kind: 'thread', data: t })
    } finally {
      targetLease.release()
    }
  }
})

}
