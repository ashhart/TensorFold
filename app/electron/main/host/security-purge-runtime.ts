import path from 'node:path'
import { promises as fs } from 'node:fs'
import type { JobApprovalDecision, Project, SecurityPurgeResult } from '../../agent/types'
import { ensureTensorFoldStudioHome, overwriteAndUnlink, saveArtifacts, saveAutomations, saveBotControlSessions, saveCliSessions, closeSessionSearchIndexes, saveConnectorEvents, saveContactDirectory, saveIncidents, saveJobs, saveMissions, saveProjectMemory, saveProjectMemoryCandidates, saveProjects, saveRooms, saveSettings, saveSubAgents, saveSuggestions, saveThreads, saveTickets, shredPurgedStateFiles, vontraPath } from '../../store'
import { emptyContactDirectory } from '../../agent/contact-directory'
import { store } from '../state'
import { auditEvent, runAuditBarrier } from './audit'
import { localMemoryCounts } from './security-controls'
import { auditPreview } from '../../agent/security'
import { inspectGitRepository, qq, sh, shStrict } from './workspace'
import { killAllTerminals } from './terminal-runtime'
import { stopAllWhatsAppBridges } from '../../agent/whatsapp-bridge'
import { beginWhatsAppAuthStatePurge, endWhatsAppAuthStatePurge, quiesceWhatsAppAuthStateWrites } from './whatsapp-auth-state'
import type { WorkspaceMutationGuard, WorkspaceMutationLease } from './workspace-mutation-guard'
import type { EventSpineHost } from '../../agent/event-spine-host'
import type { PairingRegistry } from '../../agent/pairing-sessions'
import { purgeBrowserAutomationSessions } from './browser-automation'
import { quiesceOrchestrationJournal, enableOrchestrationJournal } from '../../agent/orchestration-journal'
import { sendToRenderer } from './renderer-events'
import { durablePairingMutation } from './durable-pairing'

interface SecurityPurgeRuntimeDeps {
  aborters: Map<string, AbortController>
  roomAborters: Map<string, AbortController>
  missionAborters: Map<string, AbortController>
  pendingApprovals: Map<string, (d: JobApprovalDecision) => void>
  pendingApprovalMeta: Map<string, unknown>
  missionApprovalResolvers: Map<string, (d: JobApprovalDecision) => void>
  abortAutomations(): void
  clearThreadCues(): void
  checkpointRecordCount(): number
  pairingRecordCount(): number
  eventSpineRecordCount(): number
  providerStatusRecordCount(): number
  clearCheckpoints(): Promise<void>
  clearPairingSessions(): Promise<void>
  clearEventSpine(): Promise<void>
  clearProviderStatus(): Promise<void>
  clearUsageMemory(): void | Promise<void>
  purgeBrowserAutomation(): Promise<void>
  quiesceOrchestrationJournal(): Promise<void>
  resumeOrchestrationJournal(): void
  quiesceProviderStatus(): Promise<void>
  resumeProviderStatus(): void
  quiesceLoops(): Promise<void>
  resumeLoops(): void
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'acquireGlobal'>
}

const PRESERVED_CREDENTIALS_AND_SETTINGS = [
  'App settings and provider, cloud-connector, remote-control, and remote-Orchestrator credentials (linked WhatsApp device credentials are removed)',
  'MCP server configuration and protected environment secrets',
  'Webhook endpoint configuration and signing secrets',
] as const

// Git cleanup below is host-local. Remote project paths can have unrelated local
// lookalikes, so they must never be inspected or passed to `git worktree prune`.
export function localProjectGitCandidates(items: Project[]): Project[] {
  return items.filter((project) => !!project.cwd && project.workspace?.kind !== 'ssh')
}

export function createSecurityPurgeRuntime(deps: SecurityPurgeRuntimeDeps) {
  async function purgeLocalMemory(): Promise<SecurityPurgeResult> {
    let lifecycleQuiesced = false
    let whatsappAuthPurgeActive = false
    let purgeLease: WorkspaceMutationLease | null = null
    // Close admission before yielding, then abort every long-lived owner that may be
    // holding a workspace lease. The global lease resolves only after those owners have
    // released, so no terminal/run/restore can recreate state while the purge shreds it.
    const purgeLeasePending = deps.workspaceMutationGuard.acquireGlobal()
    for (const controller of deps.aborters.values()) controller.abort()
    for (const controller of deps.roomAborters.values()) controller.abort()
    for (const controller of deps.missionAborters.values()) controller.abort()
    deps.abortAutomations()
    killAllTerminals({ purge: true })
    // Cached sqlite handles keep writing to the deleted inode after the purge below
    // unlinks the file; close them first so the index genuinely disappears and the
    // next saveThreads rebuilds it instead of resurrecting pre-purge rows.
    closeSessionSearchIndexes()
    const loopsQuiescing = deps.quiesceLoops()
    try {
      purgeLease = await purgeLeasePending
      const auditTargets = [
      vontraPath('security-events.jsonl'),
      vontraPath('security-events.anchor.json'),
    ]
      const residualTargets = [
      vontraPath('archives'),
      vontraPath('rooms'),
      vontraPath('pr-loops'),
      vontraPath('cache'),
      vontraPath('thread-journals'),
      vontraPath('browser-automation'),
      vontraPath('outputs'),
      vontraPath('updates'),
      vontraPath('email-outbox'),
      vontraPath('usage'),
      vontraPath('event-spine.jsonl'),
      vontraPath('journals'),
      vontraPath('loops.json'),
      vontraPath('state', 'provider-status.json'),
      vontraPath('state', 'session-search.sqlite'),
      vontraPath('worktrees'),
      vontraPath('mission-attempts'),
      vontraPath('whatsapp-auth'),
    ]
      const preservedResiduals = await collectManagedWorktreeResiduals()
      const preservedByPath = new Map(preservedResiduals.map((item) => [path.resolve(item.path), item]))
      const residualFileCounts = await Promise.all([...residualTargets, ...auditTargets].map(countFiles))
      const preservedFileCounts = await Promise.all([...preservedByPath.keys()].map(countFiles))
      const counts = {
      ...localMemoryCounts(),
      turnCheckpoints: deps.checkpointRecordCount(),
      pairingSessions: deps.pairingRecordCount(),
      eventSpine: deps.eventSpineRecordCount(),
      providerStatus: deps.providerStatusRecordCount(),
        residualFiles: Math.max(
          0,
          residualFileCounts.reduce((sum, count) => sum + count, 0)
            - preservedFileCounts.reduce((sum, count) => sum + count, 0),
        ),
    }
    const projectRoots = await collectProjectGitRoots(store.projects)
    const beforePurge = {
      artifacts: store.artifacts,
      threads: store.threads,
      projects: store.projects,
      rooms: store.rooms,
      tickets: store.tickets,
      automations: store.automations,
      connectorEvents: store.connectorEvents,
      missions: store.missions,
      jobs: store.jobs,
      subAgents: store.subAgents,
      botSessions: store.botSessions,
      projectMemory: store.projectMemory,
      projectMemoryCandidates: store.projectMemoryCandidates,
      incidents: store.incidents,
      // Snapshot contacts too: it is wiped at line 68 and its save is in the batch below, so the
      // save-failure rollback must restore it or a failed purge silently empties the in-memory
      // contact directory while telling the operator the purge failed and data may remain.
      contacts: store.contacts,
      cliSessions: store.cliSessions,
      suggestions: store.suggestions,
    }
    deps.aborters.clear()
    deps.roomAborters.clear()
    deps.missionAborters.clear()
    deps.clearThreadCues()
    deps.pendingApprovals.clear()
    deps.pendingApprovalMeta.clear()
    deps.missionApprovalResolvers.clear()
    lifecycleQuiesced = true
    await Promise.all([
      deps.quiesceOrchestrationJournal(),
      deps.quiesceProviderStatus(),
      loopsQuiescing,
    ])

    store.artifacts = []
    store.threads = []
    store.projects = []
    store.rooms = []
    store.tickets = []
    store.automations = []
    store.connectorEvents = []
    store.missions = []
    store.jobs = []
    store.subAgents = []
    store.botSessions = []
    store.projectMemory = []
    store.projectMemoryCandidates = []
    store.incidents = []
    store.contacts = emptyContactDirectory()
    // Kill any live terminals and clear the persisted "was live" set so a purge leaves no PTYs and no
    // restorable sessions. killAllTerminals() sets the shutdown flag (so the kills don't re-prune the
    // file we're about to shred) and tears down the PTYs; shredPurgedStateFiles() wipes the bytes.
    store.cliSessions = []
    store.suggestions = []

    // Persist the now-empty collections. Promise.allSettled never throws and silently
    // swallows rejections — so we MUST inspect every result. If any save rejected
    // (disk full, EACCES, encryption error) the on-disk file still holds the old,
    // supposedly-purged data; reporting success here would tell the operator their
    // sensitive data was wiped when it was not. Record a FAILURE audit and throw so the
    // failure surfaces instead of being masked by { ok: true }.
    const saves: Array<[string, Promise<void>]> = [
      ['artifacts', saveArtifacts(store.artifacts)],
      ['threads', saveThreads(store.threads)],
      ['projects', saveProjects(store.projects)],
      ['rooms', saveRooms(store.rooms)],
      ['tickets', saveTickets(store.tickets)],
      ['automations', saveAutomations(store.automations)],
      ['connectorEvents', saveConnectorEvents(store.connectorEvents)],
      ['missions', saveMissions(store.missions)],
      ['jobs', saveJobs(store.jobs)],
      ['subAgents', saveSubAgents(store.subAgents)],
      ['botSessions', saveBotControlSessions(store.botSessions)],
      ['projectMemory', saveProjectMemory(store.projectMemory)],
      ['projectMemoryCandidates', saveProjectMemoryCandidates(store.projectMemoryCandidates)],
      ['incidents', saveIncidents(store.incidents)],
      ['contacts', saveContactDirectory(store.contacts)],
      ['cliSessions', saveCliSessions(store.cliSessions)],
      ['suggestions', saveSuggestions(store.suggestions)],
    ]
    const saveResults = await Promise.allSettled(saves.map(([, promise]) => promise))
    const failed = saveResults
      .map((result, index) => ({ result, label: saves[index][0] }))
      .filter((entry): entry is { result: PromiseRejectedResult; label: string } => entry.result.status === 'rejected')
    if (failed.length > 0) {
      Object.assign(store, beforePurge)
      const failedLabels = failed.map((entry) => entry.label)
      await auditEvent('security.purge_local_memory_failed', {
        counts,
        failedLabels,
        errors: failed.map((entry) => auditPreview(String((entry.result.reason as any)?.message ?? entry.result.reason))),
      })
      throw new Error(`Local memory purge failed to persist: ${failedLabels.join(', ')}. Sensitive data may remain on disk.`)
    }

    // Checkpoints and paired-device tokens live outside the central store object. Clear both
    // in memory AND await their protected-state writes; fire-and-forget here would allow the
    // UI to report success before a crash could leave the credentials/snapshots recoverable.
    // End Baileys sockets before draining writes so a late Signal-key update cannot recreate
    // protected auth state after the credential-wipe phase removes it.
    // Block new protected auth handles and permanently invalidate every existing writer before
    // stopping sockets or awaiting the drain. A late Baileys key/creds event cannot recreate the
    // auth directory after the residual-path shred.
    beginWhatsAppAuthStatePurge()
    whatsappAuthPurgeActive = true
    stopAllWhatsAppBridges()
    const externalClears: Array<[string, Promise<void>]> = [
      ['turnCheckpoints', deps.clearCheckpoints()],
      ['pairingSessions', deps.clearPairingSessions()],
      ['eventSpine', deps.clearEventSpine()],
      ['providerStatus', deps.clearProviderStatus()],
      ['usageMemory', Promise.resolve(deps.clearUsageMemory())],
      ['browserAutomation', deps.purgeBrowserAutomation()],
      ['whatsappAuthWrites', quiesceWhatsAppAuthStateWrites()],
    ]
    const externalResults = await Promise.allSettled(externalClears.map(([, promise]) => promise))
    const externalFailed = externalResults
      .map((result, index) => ({ result, label: externalClears[index][0] }))
      .filter((entry): entry is { result: PromiseRejectedResult; label: string } => entry.result.status === 'rejected')
    if (externalFailed.length > 0) {
      const failedLabels = externalFailed.map((entry) => entry.label)
      await auditEvent('security.purge_local_memory_failed', {
        counts,
        failedLabels,
        errors: externalFailed.map((entry) => auditPreview(String((entry.result.reason as any)?.message ?? entry.result.reason))),
      })
      throw new Error(`Local memory purge failed to clear: ${failedLabels.join(', ')}. Sensitive data may remain on disk.`)
    }

    // Force-shred the on-disk collections too. saveX([]) above skips (and backs up) any UNDECRYPTABLE
    // file, so without this a purge run during a key rotation would leave + duplicate sensitive bytes
    // while still reporting success.
    try {
      await shredPurgedStateFiles()
    } catch (error) {
      await auditEvent('security.purge_local_memory_failed', {
        counts,
        failedLabels: ['protectedStateShred'],
        errors: [auditPreview(error instanceof Error ? error.message : String(error))],
      })
      throw new Error('Local memory purge failed while shredding protected state. Sensitive data may remain on disk.')
    }

    // A local-state purge is also the paired-device credential-wipe boundary. Persist an
    // honest unpaired status before shredding; ordinary settings/provider credentials remain.
    // If this status write fails, abort before touching auth bytes rather than leaving durable
    // settings claiming credentials were removed when they were not.
    const priorSettings = store.settings
    const clearedWhatsAppSources = store.settings.whatsapp.sources.map((source) => ({
      ...source,
      paired: false,
      phoneNumber: null,
      credentialRotatedAt: null,
      lastError: 'Linked-device credentials were removed by the local-state purge; pair again.',
      updatedAt: Date.now(),
    }))
    if (clearedWhatsAppSources.length) {
      store.settings = {
        ...store.settings,
        whatsapp: { ...store.settings.whatsapp, sources: clearedWhatsAppSources },
      }
      try {
        await saveSettings(store.settings)
      } catch (error) {
        store.settings = priorSettings
        await auditEvent('security.purge_local_memory_failed', {
          counts,
          failedLabels: ['whatsappCredentialStatus'],
          errors: [auditPreview(error instanceof Error ? error.message : String(error))],
        })
        throw new Error('Local memory purge failed to persist the WhatsApp credential-wipe status. Credentials were not removed.')
      }
    }

    const removedPaths: string[] = []
    // These paths carry plaintext or derived conversational state outside the protected
    // collections: transcripts, browser screenshots/DOM, email bodies, usage caches, event
    // receipts, and the audit trail. Securely overwrite regular files before unlinking and
    // never follow symlinks while walking a directory tree.
    for (const target of residualTargets) {
      const exists = await fs.lstat(target).catch(() => null)
      if (!exists) continue
      try {
        const removed = await shredPath(target, preservedByPath, {
          preserveGitDirectories: isManagedWorktreeRoot(target),
        })
        if (removed) removedPaths.push(target)
      } catch (error) {
        await auditEvent('security.purge_local_memory_failed', {
          counts,
          failedLabels: ['residualPaths'],
          path: auditPreview(target),
          errors: [auditPreview(error instanceof Error ? error.message : String(error))],
        })
        throw new Error(`Local memory purge failed while removing ${target}. Sensitive data may remain on disk.`)
      }
    }
    // The audit writer is an asynchronous queue. Delete the old log/anchor *inside* that
    // queue so every pre-purge event lands before erasure and every event captured later
    // lands after it; otherwise an already-queued append can resurrect erased content.
    try {
      await runAuditBarrier(async () => {
        for (const target of auditTargets) {
          const exists = await fs.lstat(target).catch(() => null)
          if (!exists) continue
          if (await shredPath(target, preservedByPath)) removedPaths.push(target)
        }
      })
    } catch (error) {
      await auditEvent('security.purge_local_memory_failed', {
        counts,
        failedLabels: ['auditTrail'],
        errors: [auditPreview(error instanceof Error ? error.message : String(error))],
      })
      throw new Error('Local memory purge failed while removing the audit trail. Sensitive data may remain on disk.')
    }
    for (const root of projectRoots) {
      await sh(`git -C ${qq(root)} worktree prune`, root, 10_000).catch(() => '')
    }
    await ensureTensorFoldStudioHome()
    const finalPreservedResiduals = [...preservedByPath.values()]
    await auditEvent('security.purge_local_memory', {
      counts,
      removedPaths: removedPaths.map((item) => auditPreview(item)),
      preservedResiduals: finalPreservedResiduals.map((item) => ({
        ...item,
        path: auditPreview(item.path),
      })),
      preservedCredentialsAndSettings: PRESERVED_CREDENTIALS_AND_SETTINGS,
    })
    return {
      ok: true,
      purgedAt: Date.now(),
      counts,
      removedPaths,
      preservedCredentialsAndSettings: [...PRESERVED_CREDENTIALS_AND_SETTINGS],
      preservedResiduals: finalPreservedResiduals,
    }
    } finally {
      if (whatsappAuthPurgeActive) endWhatsAppAuthStatePurge()
      purgeLease?.release()
      if (lifecycleQuiesced) {
        deps.resumeOrchestrationJournal()
        deps.resumeProviderStatus()
        deps.resumeLoops()
      }
    }
  }

  async function collectProjectGitRoots(items: Project[]): Promise<string[]> {
    const roots = new Set<string>()
    for (const project of localProjectGitCandidates(items)) {
      const repo = await inspectGitRepository(project.cwd).catch(() => null)
      if (repo?.topLevel) roots.add(repo.topLevel)
    }
    return Array.from(roots)
  }

  return { purgeLocalMemory }
}

async function countFiles(target: string): Promise<number> {
  const stat = await fs.lstat(target).catch(() => null)
  if (!stat) return 0
  if (!stat.isDirectory() || stat.isSymbolicLink()) return 1
  const names = await fs.readdir(target).catch(() => [] as string[])
  const counts = await Promise.all(names.map((name) => countFiles(path.join(target, name))))
  return counts.reduce((sum, count) => sum + count, 0)
}

interface PreservedWorktreeResidual {
  path: string
  kind: 'managed_worktree'
  reason: string
}

function managedWorktreeRoots(): string[] {
  return [
    vontraPath('rooms'),
    vontraPath('pr-loops'),
    vontraPath('worktrees'),
  ]
}

function isManagedWorktreeRoot(target: string): boolean {
  const absolute = path.resolve(target)
  return managedWorktreeRoots().some((root) => path.resolve(root) === absolute)
}

async function isGitDirectory(target: string): Promise<boolean> {
  const marker = await fs.lstat(path.join(target, '.git')).catch(() => null)
  return Boolean(marker && !marker.isSymbolicLink() && (marker.isFile() || marker.isDirectory()))
}

async function describeManagedWorktree(target: string): Promise<PreservedWorktreeResidual> {
  let status: string | null = null
  try {
    status = await shStrict('git status --porcelain --untracked-files=normal', target, 5_000)
  } catch {
    // An unreadable/corrupt .git marker is still a reason to preserve, never a reason to delete.
  }
  return {
    path: path.resolve(target),
    kind: 'managed_worktree',
    reason: status === null
      ? 'Managed Git worktree preserved because its status could not be verified safely.'
      : status.trim()
      ? 'Dirty managed Git worktree preserved; confirm and remove it separately when its changes are safe.'
      : 'Managed Git worktree preserved; a clean status does not prove its commits were merged or are otherwise safe to delete.',
  }
}

async function collectManagedWorktreeResiduals(): Promise<PreservedWorktreeResidual[]> {
  const found = new Map<string, PreservedWorktreeResidual>()

  async function visit(target: string, depth: number): Promise<void> {
    if (depth > 5) return
    const stat = await fs.lstat(target).catch(() => null)
    if (!stat || stat.isSymbolicLink() || !stat.isDirectory()) return
    if (await isGitDirectory(target)) {
      const residual = await describeManagedWorktree(target)
      found.set(residual.path, residual)
      return
    }
    const names = await fs.readdir(target).catch(() => [] as string[])
    await Promise.all(names.map((name) => visit(path.join(target, name), depth + 1)))
  }

  await Promise.all(managedWorktreeRoots().map((root) => visit(root, 0)))
  return [...found.values()]
}

async function shredPath(
  target: string,
  preservedByPath = new Map<string, PreservedWorktreeResidual>(),
  options: { preserveGitDirectories?: boolean } = {},
): Promise<boolean> {
  const absolute = path.resolve(target)
  if (preservedByPath.has(absolute)) return false
  const stat = await fs.lstat(target).catch(() => null)
  if (!stat) return true
  if (stat.isSymbolicLink()) {
    await fs.unlink(target)
    return true
  }
  if (!stat.isDirectory()) {
    await overwriteAndUnlink(target)
    return true
  }
  if (options.preserveGitDirectories && await isGitDirectory(target)) {
    const residual = await describeManagedWorktree(target)
    preservedByPath.set(residual.path, residual)
    return false
  }
  let fullyRemoved = true
  for (const name of await fs.readdir(target)) {
    const removed = await shredPath(path.join(target, name), preservedByPath, options)
    if (!removed) fullyRemoved = false
  }
  if (!fullyRemoved) return false
  try {
    await fs.rmdir(target)
  } catch (error: any) {
    if (error?.code !== 'ENOENT') throw error
  }
  return true
}

export interface SecurityPurgeConfigDeps {
  aborters: Map<string, AbortController>
  roomAborters: Map<string, AbortController>
  missionAborters: Map<string, AbortController>
  pendingApprovals: Map<string, (d: JobApprovalDecision) => void>
  pendingApprovalMeta: Map<string, unknown>
  missionApprovalResolvers: Map<string, (d: JobApprovalDecision) => void>
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'acquireGlobal'>
  // Late-bound collaborators — the purge happens at runtime, well after boot assigns these.
  automationRuntime(): { abortAll(): void }
  clearThreadCues(): void
  checkpoints(): { recordCount(): number; clearAll(): Promise<void> }
  pairingRegistry: PairingRegistry
  persistPairingSessions(): Promise<void>
  spineHost(): EventSpineHost | null
  providerStatusRuntime(): { status(): { statuses: Record<string, unknown> }; clear(): Promise<void>; quiesce(): Promise<void>; resume(): void }
  loopHost(): { quiesceAndClear(): Promise<void>; resumeAfterPurge(): void }
}

/**
 * Builds the `SecurityPurgeRuntimeDeps` config consumed by `createSecurityPurgeRuntime`.
 *
 * Pure relocation of the ~39-line inline blob from main-app.ts. The computed entries
 * (record counters, clear closures that wrap collaborators) are assembled here from the same
 * collaborators the inline closures used; the collection references are forwarded verbatim.
 */
export function buildSecurityPurgeConfig(deps: SecurityPurgeConfigDeps): SecurityPurgeRuntimeDeps {
  const {
    aborters,
    roomAborters,
    missionAborters,
    pendingApprovals,
    pendingApprovalMeta,
    missionApprovalResolvers,
    workspaceMutationGuard,
    automationRuntime,
    clearThreadCues,
    checkpoints,
    pairingRegistry,
    persistPairingSessions,
    spineHost,
    providerStatusRuntime,
    loopHost,
  } = deps

  return {
    aborters,
    roomAborters,
    missionAborters,
    pendingApprovals,
    pendingApprovalMeta,
    missionApprovalResolvers,
    abortAutomations: () => automationRuntime().abortAll(),
    clearThreadCues,
    checkpointRecordCount: () => checkpoints().recordCount(),
    pairingRecordCount: () => {
      const snapshot = pairingRegistry.snapshot()
      return snapshot.links.length + snapshot.sessions.length
    },
    eventSpineRecordCount: () => spineHost()?.recordCount() ?? 0,
    providerStatusRecordCount: () => Object.keys(providerStatusRuntime().status().statuses).length,
    clearCheckpoints: () => checkpoints().clearAll(),
    clearPairingSessions: async () => {
      await durablePairingMutation(
        pairingRegistry,
        () => pairingRegistry.clearAll(),
        persistPairingSessions,
        { rollbackOnFailure: false },
      )
    },
    clearEventSpine: () => spineHost()?.purge() ?? Promise.resolve(),
    clearProviderStatus: () => providerStatusRuntime().clear(),
    clearUsageMemory: () => {
      sendToRenderer('usage:cleared')
    },
    purgeBrowserAutomation: () => purgeBrowserAutomationSessions(),
    quiesceOrchestrationJournal,
    resumeOrchestrationJournal: () => enableOrchestrationJournal(true),
    quiesceProviderStatus: () => providerStatusRuntime().quiesce(),
    resumeProviderStatus: () => providerStatusRuntime().resume(),
    quiesceLoops: () => loopHost().quiesceAndClear(),
    resumeLoops: () => loopHost().resumeAfterPurge(),
    workspaceMutationGuard,
  }
}
