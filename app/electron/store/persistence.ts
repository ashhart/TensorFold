import { validateDocumentSourceIds } from '../../shared/document-collections'
import { normalizeStoredHelperDefaults } from './model-helper-defaults'
import { normalizeStoredUiPreferences } from './ui-preferences'
import { interruptedRunRecovery } from '../../shared/run-recovery'
import { loadEngineModelProfiles, validateEngineModelProfileSets } from '../../shared/model-profiles'
import { parseThreadMemoryPolicy, threadMemoryPolicyRevision } from '../../shared/thread-memory-policy'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { vontraPath, LEGACY_STATE_FILES, legacyPath, statePath, settingsPath, modelsPath, legacySettingsPath, threadsPath, fileExists } from './paths'
import { overwriteAndUnlink, backupLegacySecretFile } from './recovery-registry'
import { SECRET_PREFIX, appKeys, blindSearchToken, hasLegacySecret, countLegacySecrets, hasProfileFallbackSecret, countMcpLegacySecrets, countWebhookLegacySecrets, mergeUnresolvedLegacySecrets, protectSettings, revealSettings, normalizeCliSubscriptionModels, protectMcpServers, revealMcpServers, protectWebhooks, revealWebhooks, getLegacySecretTemplate, setLegacySecretTemplate, getLegacySettingsBackupPath, setLegacySettingsBackupPath, getLegacyMcpTemplate, setLegacyMcpTemplate, getLegacyMcpBackupPath, setLegacyMcpBackupPath, getLegacyWebhookTemplate, setLegacyWebhookTemplate, getLegacyWebhookBackupPath, setLegacyWebhookBackupPath } from './app-crypto'
import { syncSessionSearchIndex } from './session-search-index'

export { closeSessionSearchIndexes } from './session-search-index'
import { writeJson, readProtectedState, readJsonState, writeJsonState, writeProtectedJson, readSettingsJson } from './write-infra'
import { defaultSettings, seedProviders, persistSettingsMigration } from './defaults'
import { dropUnusedSeededCliAccounts } from './seeded-cli-accounts'
import { migrateThemePreset } from './theme-migration'
import { withBuiltinEngineProvider } from '../agent/engine/provider'
import type { Artifact, Settings, Thread, TimelineItem, UserQuestionRun, Project, Room, Ticket, Automation, ConnectorEvent, Mission, Job, SubAgent, Skill, Command, SkillSuggestion, McpServerConfig, BotControlSession, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookEndpoint, Incident, CliSession } from '../agent/types'
import { stripIncognitoThreads } from '../agent/incognito'
import type { PersistedTurnCheckpoints } from '../agent/turn-checkpoints'
import type { PairingRegistryState } from '../agent/pairing-sessions'
import type { PersistedProviderStatus } from '../agent/provider-status'
import type { SchedulerLease } from '../agent/automation-scheduler'
import { normalizeMissions } from '../agent/missions'
import { normalizeJobs } from '../agent/jobs'
import { normalizeSubAgents } from '../agent/subagents'
import { normalizeTickets } from '../agent/tickets'
import type { ContactDirectory } from '../agent/contact-directory'
import { emptyContactDirectory } from '../agent/contact-directory'
import { normalizeAgentPresets } from '../../shared/bot-policy'

// Settings load/save + every per-collection load/save pair + the two purge routines.
// Depends on all five other store/ modules: paths / recovery-registry / app-crypto / write-infra /
// defaults. writeModelsJson is a hoisted function declaration so loadSettings/saveSettings above
// its definition can reference it (readSettingsJson takes it as a callback to avoid a cycle).

export async function loadSettings(): Promise<Settings> {
  // Validate/capture the configured key policy even on a fresh profile with no
  // ciphertext yet. A managed require-external launch must fail now, not silently
  // boot and discover the missing key only after the first secret is saved.
  appKeys()
  const def = defaultSettings()
  const primary = await readSettingsJson(settingsPath(), true, writeModelsJson)
  const legacy = primary ? null : await readSettingsJson(legacySettingsPath(), false, writeModelsJson)
  const loaded = primary ?? legacy ?? {}
  const unresolvedLegacyCount = countLegacySecrets(loaded)
  if (unresolvedLegacyCount > 0) {
    // Preserve the exact ciphertext before revealing unknown legacy blobs as empty.
    // The template is also merged into subsequent unrelated saves until the user
    // replaces each value, preventing the old destructive empty-overwrite migration.
    setLegacySecretTemplate(JSON.parse(JSON.stringify(loaded)) as Partial<Settings>)
    const sourcePath = primary ? settingsPath() : legacy ? legacySettingsPath() : settingsPath()
    setLegacySettingsBackupPath(await backupLegacySecretFile(sourcePath))
  } else {
    setLegacySecretTemplate(null)
    setLegacySettingsBackupPath(null)
  }
  const revealed = revealSettings({
    ...def,
    ...loaded,
    ...loadEngineModelProfiles(loaded.engineModelProfiles),
    modelHelperDefaults: normalizeStoredHelperDefaults(loaded.modelHelperDefaults),
    uiPreferences: normalizeStoredUiPreferences(loaded.uiPreferences),
    desktopMode: loaded.desktopMode === 'minimal' ? 'minimal' : 'full',
    desktopOnboardingComplete: loaded.desktopOnboardingComplete === true,
    agentPresets: normalizeAgentPresets(loaded.agentPresets ?? def.agentPresets),
    providers: withBuiltinEngineProvider(seedProviders(def.providers, loaded.providers), loaded.engineProviderSeeded === true),
    engineProviderSeeded: true,
    email: {
      ...def.email,
      ...(loaded.email ?? {}),
      sources: loaded.email?.sources ?? def.email.sources,
    },
    slack: {
      ...def.slack,
      ...(loaded.slack ?? {}),
      sources: loaded.slack?.sources ?? def.slack.sources,
    },
    calendar: {
      ...def.calendar,
      ...(loaded.calendar ?? {}),
      sources: loaded.calendar?.sources ?? def.calendar.sources,
    },
    documents: {
      ...def.documents,
      ...(loaded.documents ?? {}),
      sources: loaded.documents?.sources ?? def.documents.sources,
    },
    browser: {
      ...def.browser,
      ...(loaded.browser ?? {}),
      sources: loaded.browser?.sources ?? def.browser.sources,
    },
    imessage: {
      ...def.imessage,
      ...(loaded.imessage ?? {}),
    },
    whatsapp: {
      ...def.whatsapp,
      ...(loaded.whatsapp ?? {}),
    },
    discord: {
      ...def.discord,
      ...(loaded.discord ?? {}),
    },
    mattermost: {
      ...def.mattermost,
      ...(loaded.mattermost ?? {}),
    },
    matrix: {
      ...def.matrix,
      ...(loaded.matrix ?? {}),
    },
    googlechat: {
      ...def.googlechat,
      ...(loaded.googlechat ?? {}),
    },
    botControl: {
      telegram: {
        ...def.botControl.telegram,
        ...(loaded.botControl?.telegram ?? {}),
      },
      whatsapp: {
        ...def.botControl.whatsapp,
        ...(loaded.botControl?.whatsapp ?? {}),
      },
    },
    capabilities: {
      ...def.capabilities,
      ...(loaded.capabilities ?? {}),
      disabledCapabilityIds: loaded.capabilities?.disabledCapabilityIds ?? def.capabilities.disabledCapabilityIds,
      health: loaded.capabilities?.health ?? def.capabilities.health,
      localScripts: loaded.capabilities?.localScripts ?? def.capabilities.localScripts,
      packagedPlugins: loaded.capabilities?.packagedPlugins ?? def.capabilities.packagedPlugins,
    },
    controller: {
      ...def.controller,
      ...(loaded.controller ?? {}),
    },
    power: {
      ...def.power,
      ...(loaded.power ?? {}),
    },
    voice: {
      ...def.voice,
      ...(loaded.voice ?? {}),
    },
    remoteControl: {
      ...def.remoteControl,
      ...(loaded.remoteControl ?? {}),
      authToken: loaded.remoteControl?.authToken || def.remoteControl.authToken,
      secretRotations: loaded.remoteControl ? (loaded.remoteControl.secretRotations ?? {}) : def.remoteControl.secretRotations,
    },
    remoteClient: {
      ...def.remoteClient,
      ...(loaded.remoteClient ?? {}),
    },
    remoteOrchestrators: (loaded.remoteOrchestrators ?? []).map((host) => ({
      ...host,
      enabled: host.enabled ?? true,
      lastConnectedAt: host.lastConnectedAt ?? null,
      lastError: host.lastError ?? null,
      createdAt: host.createdAt ?? Date.now(),
      updatedAt: host.updatedAt ?? Date.now(),
    })),
  })
  const { settings: normalized, changed: normalizedCliModels } = normalizeCliSubscriptionModels(revealed)
  const seeded = dropUnusedSeededCliAccounts(normalized.providers)
  const withAccounts = seeded.removedIds.length
    ? {
        ...normalized,
        providers: seeded.providers,
        activeProviderAccountId: seeded.removedIds.includes(normalized.activeProviderAccountId ?? '') ? null : normalized.activeProviderAccountId,
      }
    : normalized
  const { settings: result, changed: rebranded } = migrateThemePreset(withAccounts)
  // Migrate any legacy plaintext secrets to encrypted-at-rest on first load.
  const loadedBot = loaded.botControl
  const hasPlaintextProviderSecret = (loaded.providers ?? []).some((p) => {
    const accountSecrets = (p.accounts ?? []).some((account) =>
      !!(account.apiKey && !account.apiKey.startsWith(SECRET_PREFIX)) ||
      Object.values(account.env ?? {}).some((value) => !!value && !value.startsWith(SECRET_PREFIX))
    )
    return !!(p.apiKey && !p.apiKey.startsWith(SECRET_PREFIX)) ||
      !!(p.serverPassword && !p.serverPassword.startsWith(SECRET_PREFIX)) ||
      Object.values(p.cliEnv ?? {}).some((value) => !!value && !value.startsWith(SECRET_PREFIX)) ||
      accountSecrets
  })
  const hasPlaintextSecret = hasPlaintextProviderSecret ||
    !!(loadedBot?.telegram?.botToken && !loadedBot.telegram.botToken.startsWith(SECRET_PREFIX)) ||
    !!(loadedBot?.whatsapp?.accessToken && !loadedBot.whatsapp.accessToken.startsWith(SECRET_PREFIX)) ||
    !!(loadedBot?.whatsapp?.verifyToken && !loadedBot.whatsapp.verifyToken.startsWith(SECRET_PREFIX)) ||
    !!(loadedBot?.whatsapp?.appSecret && !loadedBot.whatsapp.appSecret.startsWith(SECRET_PREFIX)) ||
    !!(loaded.voice?.apiKey && !loaded.voice.apiKey.startsWith(SECRET_PREFIX)) ||
    !!(loaded.voice?.elevenLabsApiKey && !loaded.voice.elevenLabsApiKey.startsWith(SECRET_PREFIX)) ||
    !!(loaded.remoteControl?.authToken && !loaded.remoteControl.authToken.startsWith(SECRET_PREFIX)) ||
    !!(loaded.remoteClient?.authToken && !loaded.remoteClient.authToken.startsWith(SECRET_PREFIX)) ||
    (loaded.remoteOrchestrators ?? []).some((host) => !!host.authToken && !host.authToken.startsWith(SECRET_PREFIX)) ||
    // Connector source secrets were missing from this migration check entirely — a plaintext
    // email/Slack/Discord/Mattermost/Matrix/Google Chat secret never triggered the re-encrypt-on-load.
    (loaded.email?.sources ?? []).some((s) => !!s.password && !s.password.startsWith(SECRET_PREFIX)) ||
    (loaded.slack?.sources ?? []).some((s) => !!s.token && !s.token.startsWith(SECRET_PREFIX)) ||
    (loaded.discord?.sources ?? []).some((s) => !!s.token && !s.token.startsWith(SECRET_PREFIX)) ||
    (loaded.mattermost?.sources ?? []).some((s) => !!s.token && !s.token.startsWith(SECRET_PREFIX)) ||
    (loaded.matrix?.sources ?? []).some((s) => !!s.token && !s.token.startsWith(SECRET_PREFIX)) ||
    (loaded.googlechat?.sources ?? []).some((s) => (!!s.webhookUrl && !s.webhookUrl.startsWith(SECRET_PREFIX)) || (!!s.accessToken && !s.accessToken.startsWith(SECRET_PREFIX)))
  // Unknown legacy macOS-Keychain/lost-key blobs share our prefix but cannot be
  // decoded under the active keys. They remain backed up and preserved verbatim;
  // never auto-save the empty revealed values over them. A profile-key fallback,
  // however, is readable and can be safely rewrapped under an external key.
  const hasLegacySafeStorageSecret = hasLegacySecret(loaded)
  const hasProfileKeyMigrationSecret = hasProfileFallbackSecret(loaded)
  // Migration is a security write, not optional housekeeping. If it fails, reject startup
  // rather than continue with known plaintext/legacy credentials while only logging nothing.
  await persistSettingsMigration(
    !hasLegacySafeStorageSecret && Boolean(legacy || normalizedCliModels || seeded.removedIds.length || rebranded || hasPlaintextSecret || hasProfileKeyMigrationSecret),
    () => saveSettings(result),
  )
  return result
}

export async function saveSettings(s: Settings, options: { validateCommit?: () => void } = {}): Promise<void> {
  const authoritativePath = settingsPath(), cataloguePath = modelsPath()
  options.validateCommit?.()
  validateEngineModelProfileSets(s.engineModelProfiles)
  // Readable JSON: protectSettings already encrypts secret fields in-place (safe:v1:), so we skip the
  // whole-file envelope — settings.json is now human-readable/editable, with only API keys/tokens
  // opaque. The discovered model catalogue per provider goes in a sibling readable models.json.
  const protectedSettings = protectSettings({ ...s, uiPreferences: normalizeStoredUiPreferences(s.uiPreferences), modelHelperDefaults: normalizeStoredHelperDefaults(s.modelHelperDefaults) })
  const merged = getLegacySecretTemplate()
    ? mergeUnresolvedLegacySecrets(getLegacySecretTemplate(), protectedSettings) as Settings
    : protectedSettings
  // The catalogue is derived; prepare it before the authoritative configuration
  // commit so a rejected cache write cannot leave settings.json secretly changed.
  await writeModelsJson(s, cataloguePath, options.validateCommit)
  options.validateCommit?.()
  await writeJson(authoritativePath, merged, options.validateCommit)
  const remainingLegacy = countLegacySecrets(merged)
  setLegacySecretTemplate(remainingLegacy > 0 ? JSON.parse(JSON.stringify(merged)) as Settings : null)
  if (remainingLegacy === 0) setLegacySettingsBackupPath(null)
}

// Function declaration (hoisted) so loadSettings/saveSettings above can reference it. readSettingsJson
// takes it as a callback to avoid a circular dependency between this module and write-infra.
function writeModelsJson(s: Settings, target = modelsPath(), validateCommit?: () => void): Promise<void> {
  const catalogue: Record<string, string[]> = {}
  for (const p of s.providers) {
    const models = p.discoveredModels?.length ? p.discoveredModels : p.models
    if (models?.length) catalogue[p.id] = models
  }
  return writeJson(target, catalogue, validateCommit)
}

export async function loadThreads(): Promise<Thread[]> {
  const threads = (await readProtectedState<Thread[]>('threads.json', [])).map(thread => {
    let sourceIds: string[] | undefined
    try { sourceIds = validateDocumentSourceIds(thread.documentSourceIds) } catch { sourceIds = [] }
    return { ...thread, ...(sourceIds !== undefined ? { documentSourceIds: sourceIds } : {}), ...(thread.runRecovery ? { runRecovery: interruptedRunRecovery(thread.runRecovery) } : {}) }
  })
  try {
    syncSessionSearchIndex(vontraPath('state', 'session-search.sqlite'), stripIncognitoThreads(threads), blindSearchToken)
  } catch (error) {
    console.warn('[vontra] session search index rebuild failed:', (error as Error)?.message ?? error)
  }
  return threads
}

export async function saveThreads(threads: Thread[]): Promise<void> {
  // Incognito threads are memory-only: never let them touch threads.json.
  for (const thread of threads) validateDocumentSourceIds(thread.documentSourceIds)
  const persisted = stripIncognitoThreads(threads)
  // An empty new thread can become Incognito while an earlier save is queued.
  // Re-evaluate privacy in the serialized slot, immediately before encryption.
  await writeProtectedJson(threadsPath(), persisted, { prepareData: () => { for (const thread of threads) validateDocumentSourceIds(thread.documentSourceIds); return stripIncognitoThreads(threads) } })
  // The search DB is disposable and stores keyed token digests, not transcript text.
  // Failure never invalidates the canonical protected JSON save.
  try {
    syncSessionSearchIndex(vontraPath('state', 'session-search.sqlite'), stripIncognitoThreads(threads), blindSearchToken)
  } catch (error) {
    console.warn('[vontra] session search index sync failed:', (error as Error)?.message ?? error)
  }
}

/** Keep question acceptance out of live IPC snapshots until the encrypted write commits. */
export async function saveThreadQuestion(
  getThreads: () => Thread[],
  run: Pick<UserQuestionRun, 'threadId' | 'runId' | 'itemId'>,
  patch: Pick<TimelineItem, 'question' | 'state' | 'output'>,
): Promise<void> {
  const resolveTarget = (threads: Thread[]) => {
    const thread = threads.find((candidate) => candidate.id === run.threadId)
    const item = thread?.timeline.find((candidate) => candidate.id === run.itemId && candidate.tool === 'ask_user')
    const question = patch.question
    if (!thread || !item || !question || question.threadId !== run.threadId || question.runId !== run.runId || question.itemId !== run.itemId) throw new Error('This question no longer belongs to the saved chat turn.')
    return { thread, item }
  }
  const initial = resolveTarget(getThreads())
  if (initial.thread.incognito) {
    // Incognito questions are memory-only even if the normal thread store is
    // quarantined; never require, skip or rewrite its unrelated protected file.
    Object.assign(initial.item, patch)
    return
  }
  let target: TimelineItem | null = null
  await writeProtectedJson(threadsPath(), stripIncognitoThreads(getThreads()), {
    prepareData: () => {
      const threads = getThreads()
      const { thread, item } = resolveTarget(threads)
      target = item
      const stagedItem = { ...target, ...patch }
      const stagedThread = { ...thread, timeline: thread.timeline.map((item) => item === target ? stagedItem : item) }
      return stripIncognitoThreads(threads.map((candidate) => candidate === thread ? stagedThread : candidate))
    },
    // Generic thread saves retain shallow references. Publish within the same
    // serialized slot so an already queued save sees the newly durable answer.
    onCommitted: () => { if (target) Object.assign(target, patch) },
  })
  try { syncSessionSearchIndex(vontraPath('state', 'session-search.sqlite'), stripIncognitoThreads(getThreads()), blindSearchToken) }
  catch (error) { console.warn('[vontra] session search index sync failed:', (error as Error)?.message ?? error) }
}

// Provider rate-limit status snapshots (VST-037): non-secret usage percentages/resets, so
// the plain state envelope (not protected) is fine — the same tier as connector events.
export async function loadProviderStatus(): Promise<PersistedProviderStatus | null> {
  return readJsonState<PersistedProviderStatus | null>('provider-status.json', null)
}
export async function saveProviderStatus(data: PersistedProviderStatus): Promise<void> {
  await writeJsonState('provider-status.json', data)
}

// Automation scheduler lease (VST-038): the single-runner lease persisted so a restart /
// shared host sees who currently owns scheduling and stands down. Non-secret coordination
// state → plain envelope.
export async function loadSchedulerLease(): Promise<SchedulerLease | null> {
  return readJsonState<SchedulerLease | null>('scheduler-lease.json', null)
}
export async function saveSchedulerLease(lease: SchedulerLease): Promise<void> {
  await writeJsonState('scheduler-lease.json', lease)
}

// Paired device sessions (VST-035) carry live bearer tokens — protected envelope, always.
const pairingSessionsPath = () => statePath('pairing-sessions.json')
export async function loadPairingSessions(): Promise<PairingRegistryState | null> {
  return readProtectedState<PairingRegistryState | null>('pairing-sessions.json', null)
}
export async function savePairingSessions(data: PairingRegistryState): Promise<void> {
  await writeProtectedJson(pairingSessionsPath(), data)
}

// Turn-checkpoint ledgers (VST-036) hold full conversation snapshots, so they get the same
// protected envelope as threads.json. The service filters incognito threads before save.
const turnCheckpointsPath = () => statePath('turn-checkpoints.json')
export async function loadTurnCheckpoints(): Promise<PersistedTurnCheckpoints | null> {
  return readProtectedState<PersistedTurnCheckpoints | null>('turn-checkpoints.json', null)
}
export async function saveTurnCheckpoints(data: PersistedTurnCheckpoints): Promise<void> {
  await writeProtectedJson(turnCheckpointsPath(), data)
}

const projectsPath = () => statePath('projects.json')
const roomsPath = () => statePath('rooms.json')
const botSessionsPath = () => statePath('bot-control-sessions.json')

export async function loadProjects(): Promise<Project[]> { return readProtectedState<Project[]>('projects.json', []) }
export async function saveProjects(p: Project[]): Promise<void> {
  await writeProtectedJson(projectsPath(), p)
}
export async function loadRooms(): Promise<Room[]> { return readProtectedState<Room[]>('rooms.json', []) }
export async function saveRooms(r: Room[]): Promise<void> {
  await writeProtectedJson(roomsPath(), r)
}
export async function loadBotControlSessions(): Promise<BotControlSession[]> { return readProtectedState<BotControlSession[]>('bot-control-sessions.json', []) }
export async function saveBotControlSessions(s: BotControlSession[]): Promise<void> { await writeProtectedJson(botSessionsPath(), s) }

const ticketsPath = () => statePath('tickets.json')
export async function loadTickets(): Promise<Ticket[]> { return normalizeTickets(await readProtectedState<Ticket[]>('tickets.json', [])) }
export async function saveTickets(t: Ticket[]): Promise<void> {
  await writeProtectedJson(ticketsPath(), normalizeTickets(t))
}

const automationsPath = () => statePath('automations.json')
export async function loadAutomations(): Promise<Automation[]> { return readProtectedState<Automation[]>('automations.json', []) }
export async function saveAutomations(a: Automation[]): Promise<void> {
  await writeProtectedJson(automationsPath(), a)
}

const connectorEventsPath = () => statePath('assistant-events.json')
export async function loadConnectorEvents(): Promise<ConnectorEvent[]> { return readProtectedState<ConnectorEvent[]>('assistant-events.json', []) }
export async function saveConnectorEvents(e: ConnectorEvent[]): Promise<void> { await writeProtectedJson(connectorEventsPath(), e) }

// Learned contact directory (platform handle -> display name). Holds names + phone
// numbers, so it is stored ENCRYPTED at rest like the other PII-bearing state files.
const contactsPath = () => statePath('contacts.json')
export async function loadContactDirectory(): Promise<ContactDirectory> { return readProtectedState<ContactDirectory>('contacts.json', emptyContactDirectory()) }
export async function saveContactDirectory(d: ContactDirectory): Promise<void> { await writeProtectedJson(contactsPath(), d, { isEmpty: Object.keys(d.records).length === 0 }) }

// The set of project terminals (codex/claude/shell/…) that were LIVE — persisted so they survive an
// app restart and can be resumed to their prior CLI session. Holds spawn cwds (filesystem paths), so
// it is stored ENCRYPTED at rest like the other path/PII-bearing state files. The file represents
// "sessions that were running": a finished/closed terminal is removed; the clean-quit set is kept.
const cliSessionsPath = () => statePath('cli-sessions.json')
export async function loadCliSessions(): Promise<CliSession[]> { return readProtectedState<CliSession[]>('cli-sessions.json', []) }
export async function saveCliSessions(list: CliSession[]): Promise<void> { await writeProtectedJson(cliSessionsPath(), list) }

const missionsPath = () => statePath('missions.json')
export async function loadMissions(): Promise<Mission[]> { return normalizeMissions(await readProtectedState<Mission[]>('missions.json', [])) }
export async function saveMissions(m: Mission[]): Promise<void> {
  const normalized = normalizeMissions(m)
  await writeProtectedJson(missionsPath(), normalized)
}

const jobsPath = () => statePath('jobs.json')
export async function loadJobs(): Promise<Job[]> { return normalizeJobs(await readProtectedState<Job[]>('jobs.json', [])).filter((job) => job.incognito !== true) }
export async function saveJobs(j: Job[]): Promise<void> {
  await writeProtectedJson(jobsPath(), normalizeJobs(j).filter((job) => job.incognito !== true))
}

const subAgentsPath = () => statePath('subagents.json')
export async function loadSubAgents(): Promise<SubAgent[]> { return normalizeSubAgents(await readProtectedState<SubAgent[]>('subagents.json', [])) }
export async function saveSubAgents(a: SubAgent[]): Promise<void> {
  await writeProtectedJson(subAgentsPath(), normalizeSubAgents(a))
}

const skillsPath = () => statePath('skills.json')
export async function loadSkills(): Promise<Skill[]> { return readProtectedState<Skill[]>('skills.json', []) }
export async function saveSkills(s: Skill[]): Promise<void> { await writeProtectedJson(skillsPath(), s) }

// First-class artifacts (VST-011): titles/paths/inline text can be sensitive — protected.
const artifactsPath = () => statePath('artifacts.json')
export async function loadArtifacts(): Promise<Artifact[]> { return readProtectedState<Artifact[]>('artifacts.json', []) }
export async function saveArtifacts(artifacts: Artifact[]): Promise<void> { await writeProtectedJson(artifactsPath(), artifacts) }

// Project-skill trust registry (VST-010): content hashes + timestamps only — plain state.
export async function loadSkillTrust(): Promise<unknown> { return readJsonState<unknown>('skill-trust.json', null) }
export async function saveSkillTrust(data: unknown): Promise<void> { await writeJsonState('skill-trust.json', data) }

const commandsPath = () => statePath('commands.json')
export async function loadCommands(): Promise<Command[]> { return readProtectedState<Command[]>('commands.json', []) }
export async function saveCommands(c: Command[]): Promise<void> { await writeProtectedJson(commandsPath(), c) }

const suggestionsPath = () => statePath('skill-suggestions.json')
export async function loadSuggestions(): Promise<SkillSuggestion[]> { return readProtectedState<SkillSuggestion[]>('skill-suggestions.json', []) }
export async function saveSuggestions(s: SkillSuggestion[]): Promise<void> { await writeProtectedJson(suggestionsPath(), s) }

export { loadProjectMemory, saveProjectMemory, loadProjectMemoryCandidates, saveProjectMemoryCandidates } from './project-memory'


// Keep the existing per-field secret wrapping for compatibility, but protect the whole file too:
// MCP commands, arguments, server names, and env-var names are sensitive operational metadata.
// readProtectedState transparently reads the former plaintext-JSON shape and rewrites it into a
// safe-json envelope on first load.
export async function loadMcpServers(): Promise<McpServerConfig[]> {
  const stored = await readProtectedState<McpServerConfig[]>('mcp-servers.json', [])
  if (countMcpLegacySecrets(stored) > 0) {
    setLegacyMcpTemplate(JSON.parse(JSON.stringify(stored)) as McpServerConfig[])
    setLegacyMcpBackupPath(await backupLegacySecretFile(statePath('mcp-servers.json')))
  } else {
    setLegacyMcpTemplate(null)
    setLegacyMcpBackupPath(null)
  }
  return revealMcpServers(stored)
}
export async function saveMcpServers(s: McpServerConfig[]): Promise<void> {
  const protectedServers = protectMcpServers(s)
  const merged = getLegacyMcpTemplate()
    ? mergeUnresolvedLegacySecrets(getLegacyMcpTemplate(), protectedServers) as McpServerConfig[]
    : protectedServers
  await writeProtectedJson(statePath('mcp-servers.json'), merged)
  setLegacyMcpTemplate(countMcpLegacySecrets(merged) > 0 ? JSON.parse(JSON.stringify(merged)) as McpServerConfig[] : null)
  if (!getLegacyMcpTemplate()) setLegacyMcpBackupPath(null)
}

// As above, preserve the inner secretRef format while hiding endpoint names, paths/URLs, and the
// surrounding integration structure in a whole-file envelope. Legacy per-field files migrate on
// read without losing compatibility with their safe:v1 secret values.
export async function loadWebhooks(): Promise<WebhookEndpoint[]> {
  const stored = await readProtectedState<WebhookEndpoint[]>('webhooks.json', [])
  if (countWebhookLegacySecrets(stored) > 0) {
    setLegacyWebhookTemplate(JSON.parse(JSON.stringify(stored)) as WebhookEndpoint[])
    setLegacyWebhookBackupPath(await backupLegacySecretFile(statePath('webhooks.json')))
  } else {
    setLegacyWebhookTemplate(null)
    setLegacyWebhookBackupPath(null)
  }
  return revealWebhooks(stored)
}
export async function saveWebhooks(w: WebhookEndpoint[]): Promise<void> {
  const protectedWebhooks = protectWebhooks(w)
  const merged = getLegacyWebhookTemplate()
    ? mergeUnresolvedLegacySecrets(getLegacyWebhookTemplate(), protectedWebhooks) as WebhookEndpoint[]
    : protectedWebhooks
  await writeProtectedJson(statePath('webhooks.json'), merged)
  setLegacyWebhookTemplate(countWebhookLegacySecrets(merged) > 0 ? JSON.parse(JSON.stringify(merged)) as WebhookEndpoint[] : null)
  if (!getLegacyWebhookTemplate()) setLegacyWebhookBackupPath(null)
}

// Incidents can carry log/config excerpts from the read-only scan, so persist the
// whole record encrypted at rest (protected JSON), like missions/project-memory.
const incidentsPath = () => statePath('incidents.json')
export async function loadIncidents(): Promise<Incident[]> { return readProtectedState<Incident[]>('incidents.json', []) }
export async function saveIncidents(i: Incident[]): Promise<void> { await writeProtectedJson(incidentsPath(), i) }

export async function purgeLegacyUserDataState(): Promise<void> {
  for (const name of LEGACY_STATE_FILES) {
    await overwriteAndUnlink(legacyPath(name)).catch(() => {})
  }
  await overwriteAndUnlink(statePath('model-scorecards.json')).catch(() => {})
  await fs.rm(legacyPath('usage'), { recursive: true, force: true }).catch(() => {})
  const legacyAudit = legacyPath('security-events.jsonl')
  const auditTarget = vontraPath('security-events.jsonl')
  if (!(await fileExists(auditTarget)) && await fileExists(legacyAudit)) {
    await fs.copyFile(legacyAudit, auditTarget).catch(() => {})
    await fs.chmod(auditTarget, 0o600).catch(() => {})
  }
  await overwriteAndUnlink(legacyAudit).catch(() => {})
}

// The memory collections the security purge clears (filenames under ~/.vontra/state/). Kept here so
// the purge can FORCE-shred them: saveX([]) routes through writeProtectedJson, whose empty-over-
// unreadable guard SKIPS the write for an UNDECRYPTABLE file (key rotated/missing) and backs the
// bytes up to a .bak — leaving AND duplicating the very data the purge must destroy. MCP servers /
// webhooks are deliberately NOT here: the purge does not clear that config.
const PURGED_STATE_FILES = [
  'threads.json', 'projects.json', 'rooms.json', 'tickets.json', 'automations.json',
  'assistant-events.json', 'missions.json', 'jobs.json', 'subagents.json',
  'bot-control-sessions.json', 'project-memory.json', 'project-memory-candidates.json',
  'incidents.json', 'contacts.json', 'cli-sessions.json', 'artifacts.json',
  'turn-checkpoints.json', 'pairing-sessions.json', 'skill-suggestions.json',
]

// Overwrite-and-unlink every purged collection file AND any recovery .bak (bypassing the
// empty-over-unreadable write guard so a purge truly leaves no sensitive bytes), then leave a CLEAN
// empty placeholder so the cleared state is actually persisted on disk (not merely absent).
export async function shredPurgedStateFiles(): Promise<void> {
  const dir = vontraPath('state')
  const present = await fs.readdir(dir).catch(() => [] as string[])
  for (const base of PURGED_STATE_FILES) {
    const p = statePath(base)
    await overwriteAndUnlink(p)
    for (const name of present) {
      if (name !== base && name.startsWith(`${base}.`)) await overwriteAndUnlink(path.join(dir, name))
    }
    // The file is now absent, so this empty write goes through the guard cleanly. Preserve each
    // store's real wire shape; boot code should never have to interpret an array as a checkpoint
    // ledger or pairing registry merely because the operator used the forget control.
    if (base === 'contacts.json') continue
    const empty: unknown = base === 'turn-checkpoints.json'
      ? { version: 1, ledgers: {}, planLinks: [] }
      : base === 'pairing-sessions.json'
        ? { links: [], sessions: [] }
        : base === 'project-memory.json'
          ? { version: 1, entries: [], candidates: [], candidateCutoff: Date.now() }
        : []
    await writeProtectedJson(p, empty)
  }
}

/** Stage choices without exposing them through threads:get before durable acceptance. */
export async function saveThreadMemoryPolicy(getThreads: () => Thread[], threadId: string, value: unknown, expectedRevision: number, validate: (thread: Thread) => void = () => {}): Promise<Thread> {
  const policy = parseThreadMemoryPolicy(value)
  if (!Number.isSafeInteger(expectedRevision) || expectedRevision < 0 || expectedRevision >= Number.MAX_SAFE_INTEGER) throw new Error('Invalid memory-choice revision.')
  const resolve = () => {
    const thread = getThreads().find((t) => t.id === threadId)
    if (!thread || thread.archivedAt) throw new Error('Choose an active chat before changing memory choices.')
    if (threadMemoryPolicyRevision(thread) !== expectedRevision) throw new Error('Memory choices changed in another window. Reopen this menu and try again.')
    validate(thread)
    return thread
  }
  const initial = resolve()
  const patch = { memoryPolicy: policy, memoryPolicyRevision: expectedRevision + 1, updatedAt: Date.now() }
  if (initial.incognito) { Object.assign(initial, patch); return initial }
  let target: Thread | undefined
  await writeProtectedJson(threadsPath(), stripIncognitoThreads(getThreads()), {
    prepareData: () => {
      target = resolve()
      return stripIncognitoThreads(getThreads().map((t) => t === target ? { ...t, ...patch } : t))
    },
    onCommitted: () => { if (target) Object.assign(target, patch) },
  })
  return target!
}
