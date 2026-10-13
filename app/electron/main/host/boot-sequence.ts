import type { app as AppT, BrowserWindow } from 'electron'
import type { promises as FsT } from 'node:fs'
import type path from 'node:path'
import type { OrchestrationEvent } from './orchestration-events'
import type { EventSpineHost, createEventSpineHost } from '../../agent/event-spine-host'
import type { PairingRegistry } from '../../agent/pairing-sessions'

// The boot body's collaborator surface is large; every symbol it closes over is passed in via
// this deps object. Loaders/savers/accessors keep their original identities, and the two mutable
// late-assigned bindings (`spineHost`, `bootSchedulerLease`) are exposed as setters so the boot
// body can assign them exactly as the inline closure did.
export interface BootStateDeps {
  app: typeof AppT
  BrowserWindow: typeof BrowserWindow
  fs: typeof FsT
  path: typeof path
  HEADLESS_MODE: boolean

  // Forwarders / helpers wired earlier in the bootstrap.
  ensureCliPath(): void
  migrateLegacyUserData(): Promise<void>
  ensureTensorFoldStudioHome(): Promise<void>
  enableOrchestrationJournal(): void
  wireOrchestrationEventsToWindow(forward: (event: OrchestrationEvent) => void): void
  sendToRenderer(channel: string, payload: unknown): void
  createEventSpineHost: typeof createEventSpineHost
  vontraHomeDir(): string

  // Store + loaders/savers (imported from ../store).
  store: any
  loadSettings(): Promise<any>
  loadConfiguredPlugins(settings: any): Promise<void>
  saveSettings(settings: any): Promise<void>
  loadThreads(): Promise<any[]>
  loadProjects(): Promise<any[]>
  loadRooms(): Promise<any[]>
  loadTickets(): Promise<any[]>
  loadAutomations(): Promise<any[]>
  loadConnectorEvents(): Promise<any[]>
  loadContactDirectory(): Promise<any>
  loadCliSessions(): Promise<any[]>
  loadMissions(): Promise<any[]>
  loadJobs(): Promise<any[]>
  loadSubAgents(): Promise<any[]>
  loadSkills(): Promise<any[]>
  loadCommands(): Promise<any[]>
  loadSuggestions(): Promise<any>
  loadBotControlSessions(): Promise<any[]>
  loadProjectMemory(): Promise<any[]>
  loadProjectMemoryCandidates(): Promise<any[]>
  loadWebhooks(): Promise<any[]>
  loadIncidents(): Promise<any[]>
  loadMcpServers(): Promise<any[]>
  loadArtifacts(): Promise<any[]>
  loadPairingSessions(): Promise<any>
  loadProviderStatus(): Promise<any>
  loadSchedulerLease(): Promise<any>
  recordPersistenceFailure(label: string, error: unknown): void
  auditEvent(kind: string, data: unknown): Promise<void> | void
  writeTensorFoldStudioStateJson(parts: string[], data: unknown): Promise<void>
  purgeLegacyUserDataState(): Promise<void>
  vontraPath(segment: string): string

  normalizeServerConfig(server: any): any
  normalizeArtifactRecord(artifact: any): any

  // Runtimes + collaborators assigned before the boot body runs.
  pairingRegistry: PairingRegistry
  persistPairingSessions(): Promise<void>
  durablePairingMutation<T>(registry: PairingRegistry, mutate: () => T, persist: () => Promise<void>, options?: { changed?: (r: T) => boolean; rollbackOnFailure?: boolean }): Promise<T>
  checkpoints: { restorePersisted(): Promise<unknown> }
  providerStatusRuntime: { setStatus(state: any): void }
  mcpHost: { setConfigs(servers: any[]): void; connectEnabled(): Promise<void> }
  remoteServer: { syncFromSettings(): Promise<void>; status(): { listening: boolean; url: string | null; lastError: string | null } }
  webhookServer: { syncFromEndpoints(): Promise<void> }
  prepareHeadlessRemoteControl(settings: any, opts: { newToken: () => string }): any
  makeRemoteControlToken(): string
  headlessBootstrapMessages(input: any): string[]
  cleanupMediaCache(dir: string, maxAgeMs?: number, now?: number): Promise<unknown>
  reconcileJobsAndMissionsAfterRestart(deps: { persistJobs(snap: any): void; persistMissions(snap: any): void }): void
  reconcileIncidentsAfterRestart(deps: { persistIncidents(snap: any): void }): void
  persistJobs(snap: any): void
  persistMissions(snap: any): void
  persistIncidents(snap: any): void
  upgradeGeneratedClaudeGuidance(project: any): Promise<boolean>
  reconcileStuckJobs(): void
  installSessionHardening(): unknown
  createWindow(): unknown
  applyTheme(): void
  applyPowerSettings(): void
  startScheduler(): void
  startUpdateChecks(): void
  startBotControlPolling(): void
  resumeAutoJobsAfterRestart(): Promise<void>

  // Late-assigned bindings — exposed as setters so the boot body can mutate them.
  getSpineHost(): EventSpineHost | null
  setSpineHost(host: EventSpineHost): void
  setBootSchedulerLease(lease: any): void
}

/**
 * Runs the app-ready boot sequence: PATH recovery, orchestration journal + spine restore,
 * state loading + reconciliation, optional headless bootstrap, and the scheduler/update/bot
 * poll kick-off.
 *
 * Pure relocation of the ~165-line `app.whenReady().then(async () => { ... })` body from
 * main-app.ts. No behavior changes; every collaborator is forwarded verbatim through `deps`.
 */
export async function bootAppState(deps: BootStateDeps): Promise<void> {
  const {
    app,
    BrowserWindow,
    fs,
    path,
    HEADLESS_MODE,
    ensureCliPath,
    migrateLegacyUserData,
    ensureTensorFoldStudioHome,
    enableOrchestrationJournal,
    wireOrchestrationEventsToWindow,
    sendToRenderer,
    createEventSpineHost,
    vontraHomeDir,
    store,
    loadSettings,
    loadConfiguredPlugins,
    saveSettings,
    loadThreads,
    loadProjects,
    loadRooms,
    loadTickets,
    loadAutomations,
    loadConnectorEvents,
    loadContactDirectory,
    loadCliSessions,
    loadMissions,
    loadJobs,
    loadSubAgents,
    loadSkills,
    loadCommands,
    loadSuggestions,
    loadBotControlSessions,
    loadProjectMemory,
    loadProjectMemoryCandidates,
    loadWebhooks,
    loadIncidents,
    loadMcpServers,
    loadArtifacts,
    loadPairingSessions,
    loadProviderStatus,
    loadSchedulerLease,
    recordPersistenceFailure,
    auditEvent,
    writeTensorFoldStudioStateJson,
    purgeLegacyUserDataState,
    vontraPath,
    normalizeServerConfig,
    normalizeArtifactRecord,
    pairingRegistry,
    persistPairingSessions,
    durablePairingMutation,
    checkpoints,
    providerStatusRuntime,
    mcpHost,
    remoteServer,
    webhookServer,
    prepareHeadlessRemoteControl,
    makeRemoteControlToken,
    headlessBootstrapMessages,
    cleanupMediaCache,
    reconcileJobsAndMissionsAfterRestart,
    reconcileIncidentsAfterRestart,
    persistJobs,
    persistMissions,
    persistIncidents,
    upgradeGeneratedClaudeGuidance,
    reconcileStuckJobs,
    installSessionHardening,
    createWindow,
    applyTheme,
    applyPowerSettings,
    startScheduler,
    startUpdateChecks,
    startBotControlPolling,
    resumeAutoJobsAfterRestart,
    setSpineHost,
    setBootSchedulerLease,
  } = deps

  // GUI-launched apps get a stripped PATH; recover the login-shell PATH so spawned CLIs (codex,
  // claude, git, MCP stdio servers) are findable — must run before anything spawns a child process.
  ensureCliPath()
  await migrateLegacyUserData()
  await ensureTensorFoldStudioHome()
  // Turn on the durable on-disk orchestration journal now that we're in Electron with a
  // writable home dir. Off by default so pure unit tests never write to the real ~/.vontra.
  enableOrchestrationJournal()
  // Bridge the in-process orchestration event broker to the renderer (mirrors the
  // onLoopUpdate push). Additive: missions/rooms/jobs still snapshot-poll as the
  // reconciling fallback — this just delivers live lifecycle events so the Command
  // Centre updates without waiting for the next poll. A destroyed window is skipped.
  wireOrchestrationEventsToWindow((event: OrchestrationEvent) => {
    sendToRenderer('orchestration:event', event)
  })
  // Ordered event spine (VST-034): restore from its crash-safe journal before any runtime
  // can record. Failures degrade to an empty in-memory spine — the app must boot even if
  // the journal is unreadable (the journal parser already skips torn lines).
  const spineJournalFile = path.join(vontraHomeDir(), 'event-spine.jsonl')
  const spineFs = {
    appendFile: (file: string, data: string) => fs.appendFile(file, data),
    readFile: (file: string) => fs.readFile(file, 'utf8'),
    writeFile: (file: string, data: string) => fs.writeFile(file, data),
    rename: (from: string, to: string) => fs.rename(from, to),
    // Durability seam: lets rewriteSpineJournal datasync the staged temp before the rename.
    open: (file: string, flags: string) => fs.open(file, flags),
  }
  try {
    setSpineHost(await createEventSpineHost({ journalFile: spineJournalFile, fs: spineFs, onError: (message) => console.error('[event-spine]', message) }))
  } catch (error) {
    console.error('[event-spine] boot restore failed; continuing with an empty spine:', error)
    setSpineHost(await createEventSpineHost({ journalFile: spineJournalFile, fs: { ...spineFs, readFile: async () => '' }, onError: (message) => console.error('[event-spine]', message) }))
  }
  store.settings = await loadSettings()
  await loadConfiguredPlugins(store.settings)
  const headlessBootstrap = HEADLESS_MODE
    ? prepareHeadlessRemoteControl(store.settings.remoteControl, { newToken: makeRemoteControlToken })
    : null
  if (headlessBootstrap) {
    store.settings = { ...store.settings, remoteControl: headlessBootstrap.remoteControl }
    if (headlessBootstrap.changed) await saveSettings(store.settings)
  }
  store.threads = await loadThreads()
  // Revive turn-checkpoint ledgers AFTER threads load so the persistable filter can prune
  // ledgers of threads that no longer exist. Fire-and-forget: restore failure only means
  // revert/fork history starts empty, never a boot failure.
  void checkpoints.restorePersisted().catch(() => {})
  // Paired device sessions survive restarts (VST-035): merge-restore (live wins) so a
  // session revoked after the save can't come back.
  const restoredPairingSessions = await loadPairingSessions().catch((error) => {
    recordPersistenceFailure('pairingSessions.load', error)
    return null
  })
  if (restoredPairingSessions) pairingRegistry.restore(restoredPairingSessions)
  if (headlessBootstrap?.resetPairingSessions) {
    pairingRegistry.clearAll()
    await persistPairingSessions()
  }
  // Provider rate-limit status (VST-037): restore last-known windows so the picker/doctor
  // show something before the next codex turn refreshes them.
  void loadProviderStatus().then((state) => { if (state && state.version === 1 && state.statuses) providerStatusRuntime.setStatus(state) }).catch(() => {})
  // Automation scheduler lease (VST-038): load BEFORE the scheduler's first tick so it sees
  // any lease another host still holds and stands down instead of double-firing.
  setBootSchedulerLease(await loadSchedulerLease().catch(() => null))
  store.projects = await loadProjects()
  void Promise.allSettled(store.projects.map(upgradeGeneratedClaudeGuidance)).then((results) => {
    if (results.some((result) => result.status === 'fulfilled' && result.value)) {
      void auditEvent('project.instructions.claudeImportUpgrade', { count: results.filter((result) => result.status === 'fulfilled' && result.value).length })
    }
  })
  store.rooms = await loadRooms()
  store.tickets = await loadTickets()
  store.automations = await loadAutomations()
  store.connectorEvents = await loadConnectorEvents()
  store.contacts = await loadContactDirectory()
  // The terminals that were live at the last (clean) quit — terminal:restore re-spawns + resumes them.
  store.cliSessions = await loadCliSessions()
  store.missions = await loadMissions()
  store.jobs = await loadJobs()
  store.subAgents = await loadSubAgents()
  // Orphan reconciliation (jobs + missions): reap lease owners from the prior process, then
  // demote any mission left mid-flight (running / waiting_approval / stalled) by a prior
  // crash/quit — it has no live executor or approval resolver, so pause it for deliberate
  // resume. reconcileStuckJobs runs between the two, exactly as the original inline sequence.
  reconcileJobsAndMissionsAfterRestart({ persistJobs, persistMissions })
  reconcileStuckJobs()
  setInterval(reconcileStuckJobs, 60_000).unref()
  store.skills = await loadSkills()
  store.artifacts = (await loadArtifacts()).map(normalizeArtifactRecord).filter((a): a is NonNullable<typeof a> => a != null)
  store.commands = await loadCommands()
  store.suggestions = await loadSuggestions()
  store.botSessions = await loadBotControlSessions()
  store.projectMemory = await loadProjectMemory()
  store.projectMemoryCandidates = await loadProjectMemoryCandidates()
  store.webhooks = await loadWebhooks()
  store.incidents = await loadIncidents()
  // Reconcile incident state orphaned by a crash/restart: the in-process scan timer AND the mission
  // completion hook both die with the previous process, so a persisted 'running' scan or a mitigation
  // still pointing at a now-dead mission would otherwise be stuck in the UI forever. (missions were
  // already orphan-reconciled above, so a previously-running mission now reads 'paused'.)
  reconcileIncidentsAfterRestart({ persistIncidents })
  store.mcpServers = (await loadMcpServers()).map(normalizeServerConfig)
  await Promise.allSettled([
    writeTensorFoldStudioStateJson(['state', 'threads.json'], store.threads),
    writeTensorFoldStudioStateJson(['state', 'projects.json'], store.projects),
    writeTensorFoldStudioStateJson(['state', 'rooms.json'], store.rooms),
    writeTensorFoldStudioStateJson(['state', 'tickets.json'], store.tickets),
    writeTensorFoldStudioStateJson(['state', 'automations.json'], store.automations),
    writeTensorFoldStudioStateJson(['state', 'missions.json'], store.missions),
    writeTensorFoldStudioStateJson(['state', 'jobs.json'], store.jobs),
    writeTensorFoldStudioStateJson(['state', 'subagents.json'], store.subAgents),
  ])
  await purgeLegacyUserDataState().catch(() => {})
  // Periodic media-cache prune: drop cached inbound attachments older than the max age so
  // ~/.vontra/cache doesn't grow without bound. Run once shortly after boot, then daily.
  const pruneMediaCache = () => { void cleanupMediaCache(vontraPath('cache')).catch(() => {}) }
  setTimeout(pruneMediaCache, 60_000).unref()
  setInterval(pruneMediaCache, 24 * 60 * 60_000).unref()
  mcpHost.setConfigs(store.mcpServers)
  applyTheme()
  applyPowerSettings()
  if (!HEADLESS_MODE) {
    installSessionHardening()
    createWindow()
  }
  void mcpHost.connectEnabled()
  void webhookServer.syncFromEndpoints() // self-disables when no inbound endpoint is enabled
  if (headlessBootstrap) {
    await remoteServer.syncFromSettings()
    const status = remoteServer.status()
    if (!status.listening || !status.url) {
      throw new Error(status.lastError || 'Headless Orchestrator failed to start its remote-control listener.')
    }
    let pairingCredential: string | null = null
    let pairingExpiresAt: number | null = null
    if (headlessBootstrap.revealCredentials) {
      const link = await durablePairingMutation(
        pairingRegistry,
        () => pairingRegistry.issuePairingLink('Headless bootstrap browser'),
        persistPairingSessions,
      )
      pairingCredential = link.credential
      pairingExpiresAt = link.expiresAt
      store.settings = {
        ...store.settings,
        remoteControl: {
          ...store.settings.remoteControl,
          headlessBootstrappedAt: headlessBootstrap.completionMarker,
        },
      }
      await saveSettings(store.settings)
    }
    for (const line of headlessBootstrapMessages({
      url: status.url,
      token: headlessBootstrap.revealCredentials ? store.settings.remoteControl.authToken : null,
      pairingCredential,
      pairingExpiresAt,
    })) console.log(line)
  } else {
    void remoteServer.syncFromSettings()
  }
  startScheduler()
  startUpdateChecks()
  startBotControlPolling()
  void resumeAutoJobsAfterRestart()
  app.on('activate', () => { if (!HEADLESS_MODE && BrowserWindow.getAllWindows().length === 0) createWindow() })
}
