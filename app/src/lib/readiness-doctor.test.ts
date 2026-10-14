import assert from 'node:assert/strict'
import { buildReadinessDoctor, readinessProbeKey, summarizeDoctor, worstDoctorStatus, type RemoteSnapshotProbeValue } from './readiness-doctor'
import type { CliToolStatus, ConnectorDefinition, RemoteControlWorkSnapshot, Settings } from './types'

const now = Date.now()

function baseSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    providers: [],
    activeProviderId: null,
    activeProviderAccountId: null,
    activeModel: null,
    agentPresets: [],
    theme: 'system',
    approvalMode: 'suggest',
    approvalRules: [],
    defaultCwd: null,
    temperature: 0.2,
    reasoningEffort: 'medium',
    autoCompact: true,
    contextLimit: 128000,
    localContextLimit: 32000,
    autoCheckUpdates: false,
    updateFeedUrl: '',
    power: {},
    email: { sources: [], maxMessages: 50, lastSyncAt: null },
    slack: { sources: [], maxMessages: 50, lastSyncAt: null },
    calendar: { sources: [], maxEvents: 50, lastSyncAt: null },
    documents: { sources: [], maxDocuments: 100, maxDocumentBytes: 1_000_000, lastSyncAt: null },
    browser: { sources: [], maxItems: 100, lastSyncAt: null },
    imessage: { enabled: false, allowedContacts: '', lastSyncAt: null, lastError: null },
    whatsapp: { sources: [], lastSyncAt: null },
    discord: { sources: [], lastSyncAt: null },
    mattermost: { sources: [], lastSyncAt: null },
    matrix: { sources: [], lastSyncAt: null },
    googlechat: { sources: [], lastSyncAt: null },
    botControl: {
      telegram: { enabled: false, botToken: '', allowedChatIds: '', allowedUserIds: '', polling: false, lastUpdateId: null, lastPollAt: null, lastError: null },
      whatsapp: { enabled: false, accessToken: '', phoneNumberId: '', verifyToken: '', appSecret: '', allowedChatIds: '', lastEventAt: null, lastError: null },
    },
    capabilities: { definitions: [], sets: [], activeSetId: null, audit: [] },
    controller: { enabled: false, captureMode: 'manual', allowedApps: '', lastCapturedAt: null, lastError: null },
    voice: { enabled: false, inputEnabled: false, outputEnabled: false, provider: 'system', voiceName: '', localBaseUrl: '', localVoice: '', localEmotion: '', lastError: null },
    remoteControl: { enabled: false, serverName: 'This Mac', bindHost: '127.0.0.1', allowLan: false, port: 8765, authToken: '', lastStartedAt: null, lastClientAt: null, lastError: null },
    remoteClient: { enabled: false, serverUrl: '', authToken: '', lastConnectedAt: null, lastError: null },
    remoteOrchestrators: [],
    preset: 'graphite',
    light: {},
    dark: {},
    pointerCursors: true,
    sshHosts: [],
    trustedSshHostKeys: [],
    ...overrides,
  } as unknown as Settings
}

function source(extra: Record<string, unknown> = {}) {
  return { id: 'src', label: 'Source', enabled: true, createdAt: now, updatedAt: now, lastSyncAt: null, lastError: null, ...extra }
}

function cliStatus(id: string, extra: Partial<CliToolStatus> = {}): CliToolStatus {
  return {
    id,
    name: id[0].toUpperCase() + id.slice(1),
    short: id,
    command: id,
    resumeCommand: null,
    color: '#fff',
    icon: 'terminal',
    builtin: true,
    installed: false,
    version: null,
    effectiveCommand: id,
    configuredPath: null,
    providerIds: [],
    ...extra,
  }
}

function connectorDef(id: ConnectorDefinition['id'], extra: Partial<ConnectorDefinition> = {}): ConnectorDefinition {
  return {
    id,
    name: id,
    kind: 'communication',
    status: 'connected',
    description: '',
    transport: null,
    tools: [],
    unread: 0,
    pendingApprovals: 0,
    lastSyncAt: null,
    lastEventAt: null,
    permissionStatus: 'granted',
    permissionDetail: '',
    ...extra,
  }
}

function workSnapshot(extra: Partial<RemoteControlWorkSnapshot> = {}): RemoteControlWorkSnapshot {
  return {
    jobs: { total: 0, visible: 0, queued: 0, running: 0, waitingApproval: 0, paused: 0, stalled: 0, failed: 0, deadLetter: 0, retryable: 0 },
    leases: { active: 0, expired: 0, expiringSoon: 0, missingRunning: 0 },
    approvals: { pending: 0, tool: 0, connector: 0, mission: 0 },
    nextWakeAt: null,
    nextWakeJobId: null,
    generatedAt: now,
    terminals: { total: 0, running: 0, attention: 0 },
    ...extra,
  }
}

function slackSettings(sourceExtra: Record<string, unknown> = {}) {
  return baseSettings({
    slack: { sources: [source({ kind: 'api', token: 'xoxb-token', ...sourceExtra })], maxMessages: 50, lastSyncAt: null } as any,
  })
}

const find = (checks: ReturnType<typeof buildReadinessDoctor>, id: string) => checks.find((item) => item.id === id)

// ---------------------------------------------------------------- baseline

{
  const checks = buildReadinessDoctor({ settings: baseSettings(), localClis: { status: 'ok', value: [] }, now })
  assert.equal(find(checks, 'models:none')?.status, 'error')
  assert.equal(find(checks, 'terminal:clis')?.status, 'error')
  assert.ok(summarizeDoctor(checks).error >= 2)
}

{
  const settings = baseSettings({
    providers: [{ id: 'p1', name: 'OpenAI', kind: 'cloud', baseUrl: 'https://api.openai.com/v1', apiKey: 'sk-test', models: ['gpt-4o'] } as any],
    activeProviderId: 'p1',
    activeModel: 'gpt-4o',
    slack: { sources: [source({ kind: 'api', token: '' })], maxMessages: 50, lastSyncAt: null } as any,
    remoteOrchestrators: [{ id: 'mini', name: 'Mini', serverUrl: 'http://127.0.0.1:8765', authToken: 'tok', enabled: true, lastConnectedAt: null, lastError: null, createdAt: now, updatedAt: now }],
  })
  const checks = buildReadinessDoctor({
    settings,
    providerReadiness: { [readinessProbeKey('p1')]: { status: 'ok', value: { ready: true, reason: 'ok', message: '' } } },
    localClis: { status: 'ok', value: [cliStatus('pi', { installed: true })] },
    remoteProbes: { mini: { status: 'error', message: 'connection refused' } },
    now,
  })
  assert.equal(find(checks, 'models:active')?.status, 'ok')
  assert.equal(find(checks, 'connector:slack')?.status, 'error', 'missing token is a distinct error')
  assert.equal(find(checks, 'remote:mini')?.status, 'error', 'unreachable host is a distinct error')
  assert.equal(find(checks, 'remote:mini:heartbeat'), undefined, 'no subchecks for an unreachable host')
  assert.equal(find(checks, 'terminal:clis')?.status, 'ok')
}

// ---------------------------------------------------------------- CLI inventory distinctness

{
  const value = [
    cliStatus('shell', { command: null, installed: true, effectiveCommand: null }),
    cliStatus('claude', { name: 'Claude Code', installed: true, version: '2.1.0', configuredPath: '/opt/claude', effectiveCommand: '/opt/claude', providerIds: ['acct-1'] }),
    cliStatus('codex', { name: 'Codex', installed: false }),
    cliStatus('pi', { name: 'Pi', installed: true, version: 'v0.9' }),
    cliStatus('opencode', { name: 'OpenCode', installed: false }),
    cliStatus('hermes', { name: 'Hermes', installed: false }),
    cliStatus('custom-1', { name: 'My Tool', builtin: false, command: 'mytool --serve', effectiveCommand: 'mytool --serve', installed: true }),
    cliStatus('grok', { name: 'Grok', installed: true, state: 'broken', problem: '~/bin/grok: line 2: node: not found' }),
  ]
  const checks = buildReadinessDoctor({ settings: baseSettings(), localClis: { status: 'ok', value }, now })
  assert.equal(find(checks, 'terminal:cli:grok')?.status, 'warning', 'a binary whose --version fails is not ready')
  assert.ok(find(checks, 'terminal:cli:grok')?.detail.includes('node: not found'), 'the probe error is shown, never as a version')

  assert.equal(find(checks, 'terminal:cli:shell')?.status, 'ok', 'plain shell is always launchable')
  const claude = find(checks, 'terminal:cli:claude')
  assert.equal(claude?.status, 'ok')
  assert.ok(claude?.detail.includes('v2.1.0') && claude.detail.includes('configured path') && claude.detail.includes('1 provider account'), 'installed CLI proves version, path source, and provider link')
  assert.equal(find(checks, 'terminal:cli:codex')?.status, 'warning', 'missing binary is a distinct non-ready state')
  assert.ok(find(checks, 'terminal:cli:codex')?.detail.includes('not found'), 'missing binary says so explicitly')
  assert.equal(find(checks, 'terminal:cli:pi')?.status, 'ok')
  assert.ok(find(checks, 'terminal:cli:pi')?.detail.includes('v0.9'), 'version prefix is normalized')
  assert.equal(find(checks, 'terminal:cli:opencode')?.status, 'warning')
  assert.equal(find(checks, 'terminal:cli:hermes')?.status, 'warning', 'hermes renders distinctly even when missing')
  const custom = find(checks, 'terminal:cli:custom-1')
  assert.equal(custom?.status, 'ok')
  assert.ok(custom?.title.includes('· custom'), 'custom entries are labeled distinctly')
  assert.ok(find(checks, 'terminal:clis')?.detail.startsWith('4 of 8'), 'aggregate counts launchable entries, not broken ones')
}

{
  // Plain registry definitions (no probe data) must not show ok for command-backed CLIs.
  const checks = buildReadinessDoctor({
    settings: baseSettings(),
    localClis: { status: 'ok', value: [{ id: 'codex', name: 'Codex', short: 'codex', command: 'codex', color: '#fff', icon: 'code', builtin: true }] },
    now,
  })
  assert.equal(find(checks, 'terminal:cli:codex')?.status, 'checking', 'unprobed binary renders as checking, never ok')
}

{
  // Version advisory: newer version available warns (informational, with the update command);
  // up-to-date stays ok; a dev build ahead of latest stays ok with a note.
  const value = [
    cliStatus('codex', { name: 'Codex', installed: true, version: '0.142.5', advisory: { status: 'update_available', currentVersion: '0.142.5', latestVersion: '0.150.0', updateCommand: 'npm i -g @openai/codex', canUpdate: true, message: '' } }),
    cliStatus('claude', { name: 'Claude Code', installed: true, version: '2.1.170', advisory: { status: 'current', currentVersion: '2.1.170', latestVersion: '2.1.170', updateCommand: null, canUpdate: false, message: '' } }),
    cliStatus('pi', { name: 'Pi', installed: true, version: '0.81.0', advisory: { status: 'ahead', currentVersion: '0.81.0', latestVersion: '0.80.3', updateCommand: null, canUpdate: false, message: '' } }),
  ]
  const checks = buildReadinessDoctor({ settings: baseSettings(), localClis: { status: 'ok', value }, now })
  const codex = find(checks, 'terminal:cli:codex')
  assert.equal(codex?.status, 'warning', 'a newer version available warns')
  assert.ok(codex?.detail.includes('Update available: 0.150.0') && codex.detail.includes('npm i -g @openai/codex'), 'shows latest version + update command')
  assert.equal(codex?.action, 'Update')
  assert.equal(find(checks, 'terminal:cli:claude')?.status, 'ok', 'up-to-date stays ok')
  const pi = find(checks, 'terminal:cli:pi')
  assert.equal(pi?.status, 'ok', 'a dev build ahead of latest stays ok')
  assert.ok(pi?.detail.includes('dev build'), 'ahead build is annotated')
}

// ---------------------------------------------------------------- remote host subchecks

const remoteHostSettings = baseSettings({
  remoteOrchestrators: [{ id: 'mini', name: 'Mini', serverUrl: 'http://127.0.0.1:8765', authToken: 'tok', enabled: true, lastConnectedAt: null, lastError: null, createdAt: now, updatedAt: now }],
})

function remoteChecks(value: RemoteSnapshotProbeValue) {
  return buildReadinessDoctor({ settings: remoteHostSettings, remoteProbes: { mini: { status: 'ok', value } }, now })
}

{
  const checks = remoteChecks({ ok: true, generatedAt: now - 5_000, work: workSnapshot({ jobs: { ...workSnapshot().jobs, running: 2, queued: 1 }, terminals: { total: 3, running: 2, attention: 0 } }) })
  assert.equal(find(checks, 'remote:mini')?.status, 'ok')
  assert.equal(find(checks, 'remote:mini:heartbeat')?.status, 'ok')
  assert.equal(find(checks, 'remote:mini:work')?.status, 'ok')
  assert.ok(find(checks, 'remote:mini:work')?.detail.includes('2 running'), 'active work is explicit')
  assert.equal(find(checks, 'remote:mini:approvals')?.status, 'ok')
  assert.equal(find(checks, 'remote:mini:terminals')?.status, 'ok')
  assert.ok(find(checks, 'remote:mini:terminals')?.detail.includes('2 running of 3'), 'terminal health is explicit')
}

{
  const checks = remoteChecks({ ok: true, generatedAt: now - 10 * 60_000, work: workSnapshot() })
  assert.equal(find(checks, 'remote:mini:heartbeat')?.status, 'warning', 'stale heartbeat is a distinct warning')
  assert.ok(find(checks, 'remote:mini:heartbeat')?.detail.includes('stale'))
}

{
  const checks = remoteChecks({
    ok: true,
    generatedAt: now,
    work: workSnapshot({
      jobs: { ...workSnapshot().jobs, failed: 1, stalled: 2, running: 1 },
      leases: { active: 1, expired: 1, expiringSoon: 0, missingRunning: 1 },
      approvals: { pending: 3, tool: 2, connector: 1, mission: 0 },
      terminals: { total: 2, running: 1, attention: 1 },
    }),
  })
  const work = find(checks, 'remote:mini:work')
  assert.equal(work?.status, 'warning')
  assert.ok(work?.detail.includes('1 failed') && work.detail.includes('2 stalled') && work.detail.includes('1 expired lease') && work.detail.includes('running without a lease'), 'job + lease trouble is itemized')
  const approvals = find(checks, 'remote:mini:approvals')
  assert.equal(approvals?.status, 'warning')
  assert.ok(approvals?.detail.includes('3 approvals') && approvals.detail.includes('2 tool'), 'approval breakdown is explicit')
  assert.equal(find(checks, 'remote:mini:terminals')?.status, 'warning', 'terminals needing attention warn')
}

{
  const checks = remoteChecks({ ok: true, generatedAt: now })
  assert.equal(find(checks, 'remote:mini:work')?.status, 'warning', 'missing work summary is unknown, not ok')
  assert.equal(find(checks, 'remote:mini:approvals'), undefined)
}

{
  const checks = remoteChecks({ ok: true, work: workSnapshot() })
  assert.equal(find(checks, 'remote:mini:heartbeat')?.status, 'warning', 'missing heartbeat timestamp warns')
}

{
  const noTerm = { ...workSnapshot() } as RemoteControlWorkSnapshot
  delete (noTerm as any).terminals
  const checks = remoteChecks({ ok: true, generatedAt: now, work: noTerm })
  assert.equal(find(checks, 'remote:mini:terminals')?.status, 'warning', 'older hosts without a terminal summary render unknown, never ok')
}

{
  const settings = baseSettings({
    remoteOrchestrators: [
      { id: 'off', name: 'Off', serverUrl: 'http://127.0.0.1:1', authToken: 'tok', enabled: false, lastConnectedAt: null, lastError: null, createdAt: now, updatedAt: now },
      { id: 'naked', name: 'Naked', serverUrl: '', authToken: '', enabled: true, lastConnectedAt: null, lastError: null, createdAt: now, updatedAt: now },
    ],
  })
  const checks = buildReadinessDoctor({ settings, now })
  assert.equal(find(checks, 'remote:off')?.status, 'warning', 'disabled host is a distinct state')
  assert.equal(find(checks, 'remote:naked')?.status, 'error', 'missing url/token is a distinct state')
}

// ---------------------------------------------------------------- connector proof

{
  const checks = buildReadinessDoctor({ settings: slackSettings(), now })
  assert.equal(find(checks, 'connector:slack')?.status, 'warning', 'configured connector without proof caps at warning')
  assert.ok(find(checks, 'connector:slack')?.detail.includes('Proof unavailable'))
}

{
  const checks = buildReadinessDoctor({ settings: slackSettings(), connectors: { status: 'checking' }, now })
  assert.equal(find(checks, 'connector:slack')?.status, 'checking')
}

{
  const checks = buildReadinessDoctor({ settings: slackSettings(), connectors: { status: 'ok', value: [] }, now })
  assert.equal(find(checks, 'connector:slack')?.status, 'warning', 'no live state yet → unproven')
  assert.ok(find(checks, 'connector:slack')?.detail.includes('sync or setup test'))
}

{
  const checks = buildReadinessDoctor({
    settings: slackSettings(),
    connectors: { status: 'ok', value: [connectorDef('slack', { lastSyncAt: null, lastEventAt: null })] },
    now,
  })
  assert.equal(find(checks, 'connector:slack')?.status, 'warning', 'connected but never synced → unproven')
}

{
  const checks = buildReadinessDoctor({
    settings: slackSettings(),
    connectors: { status: 'ok', value: [connectorDef('slack', { lastSyncAt: now - 3 * 60_000, lastEventAt: now - 60_000, tools: [{ name: 'slack_send' } as any, { name: 'slack_read' } as any], pendingApprovals: 1 })] },
    connectorEvents: [{ connectorId: 'slack', direction: 'outbound', createdAt: now - 30_000, approvalStatus: 'approved' }],
    now,
  })
  const slack = find(checks, 'connector:slack')
  assert.equal(slack?.status, 'ok', 'sync + sample proof unlocks ok')
  assert.ok(slack?.detail.includes('last sync 3m ago'), 'sync recency is shown')
  assert.ok(slack?.detail.includes('sample item'), 'sample item proof is shown')
  assert.ok(slack?.detail.includes('2 capabilities'), 'capability list is summarized')
  assert.ok(slack?.detail.includes('1 approval pending'), 'approval posture is visible')
  assert.ok(slack?.detail.includes('outbound ok'), 'successful outbound proof is visible')
}

{
  const checks = buildReadinessDoctor({
    settings: slackSettings(),
    connectors: { status: 'ok', value: [connectorDef('slack', { lastSyncAt: now - 3 * 60_000, lastEventAt: now - 60_000 })] },
    now,
  })
  assert.equal(find(checks, 'connector:slack')?.status, 'warning', 'read proof alone cannot prove a write-capable connector')
  assert.ok(find(checks, 'connector:slack')?.detail.includes('write unproven'))
}

{
  const checks = buildReadinessDoctor({
    settings: slackSettings(),
    connectors: { status: 'ok', value: [connectorDef('slack', { status: 'error', error: 'invalid_auth' })] },
    now,
  })
  assert.equal(find(checks, 'connector:slack')?.status, 'error', 'failed auth is a distinct error')
  assert.ok(find(checks, 'connector:slack')?.detail.includes('invalid_auth'))
}

{
  const checks = buildReadinessDoctor({
    settings: slackSettings(),
    connectors: { status: 'ok', value: [connectorDef('slack', { status: 'mock', lastSyncAt: now - 1000 })] },
    now,
  })
  assert.equal(find(checks, 'connector:slack')?.status, 'warning', 'mock/demo data never proves readiness')
  assert.ok(find(checks, 'connector:slack')?.detail.includes('simulated'))
}

{
  const denied = buildReadinessDoctor({
    settings: slackSettings(),
    connectors: { status: 'ok', value: [connectorDef('slack', { permissionStatus: 'denied', permissionDetail: 'workspace admin blocked the bot' })] },
    now,
  })
  assert.equal(find(denied, 'connector:slack')?.status, 'error', 'denied permissions are an error')
  const degraded = buildReadinessDoctor({
    settings: slackSettings(),
    connectors: { status: 'ok', value: [connectorDef('slack', { permissionStatus: 'degraded', permissionDetail: 'missing channels:history', lastSyncAt: now - 1000 })] },
    now,
  })
  assert.equal(find(degraded, 'connector:slack')?.status, 'warning', 'degraded permissions warn even with sync proof')
}

{
  const checks = buildReadinessDoctor({ settings: slackSettings({ enabled: false }), connectors: { status: 'ok', value: [connectorDef('slack', { lastSyncAt: now })] }, now })
  assert.equal(find(checks, 'connector:slack')?.status, 'warning', 'disabled source is a distinct state even with live proof')
  assert.ok(find(checks, 'connector:slack')?.detail.includes('none are enabled'))
}

{
  // Telegram + WhatsApp flow through the same proof gate.
  const settings = baseSettings({
    botControl: {
      telegram: { enabled: true, botToken: 'tok', allowedChatIds: '123', allowedUserIds: '123', polling: false, lastUpdateId: null, lastPollAt: null, lastError: null },
      whatsapp: { enabled: false, accessToken: '', phoneNumberId: '', verifyToken: '', appSecret: '', allowedChatIds: '', lastEventAt: null, lastError: null },
    },
    whatsapp: { sources: [source({ kind: 'bridge' })], lastSyncAt: null } as any,
  })
  const unproven = buildReadinessDoctor({ settings, connectors: { status: 'ok', value: [] }, now })
  assert.equal(find(unproven, 'connector:telegram')?.status, 'warning')
  assert.equal(find(unproven, 'connector:whatsapp')?.status, 'warning')
  const proven = buildReadinessDoctor({
    settings,
    connectors: { status: 'ok', value: [connectorDef('telegram', { lastEventAt: now - 5_000 }), connectorDef('whatsapp', { lastSyncAt: now - 5_000 })] },
    connectorEvents: [
      { connectorId: 'telegram', direction: 'outbound', createdAt: now - 2_000, approvalStatus: 'approved' },
      { connectorId: 'whatsapp', direction: 'outbound', createdAt: now - 1_000, approvalStatus: 'approved' },
    ],
    now,
  })
  assert.equal(find(proven, 'connector:telegram')?.status, 'ok')
  assert.equal(find(proven, 'connector:whatsapp')?.status, 'ok')
}

{
  const base = baseSettings()
  const settings = baseSettings({
    botControl: {
      telegram: { ...base.botControl.telegram, enabled: true, botToken: 'tok', allowedChatIds: '-100', allowedUserIds: '' },
      whatsapp: { ...base.botControl.whatsapp, enabled: true, accessToken: 'access', phoneNumberId: 'phone', verifyToken: 'verify', appSecret: '', allowedChatIds: '447700900123' },
    },
  })
  const checks = buildReadinessDoctor({ settings, connectors: { status: 'ok', value: [] }, now })
  assert.match(find(checks, 'connector:telegram')?.detail ?? '', /user-ID allowlist/)
  assert.equal(find(checks, 'connector:whatsapp-cloud')?.status, 'error')
  assert.match(find(checks, 'connector:whatsapp-cloud')?.detail ?? '', /Meta app secret/)
}

{
  const base = baseSettings()
  const settings = baseSettings({
    remoteControl: { ...base.remoteControl, enabled: true, authToken: 'remote-token', secretRotations: { authToken: now - 91 * 24 * 60 * 60_000 } },
    whatsapp: {
      sources: [{
        id: 'wa_old', label: 'Personal', enabled: true, phoneNumber: null, paired: true,
        lastQrAt: null, lastEventAt: null, lastError: null,
        credentialRotatedAt: now - 91 * 24 * 60 * 60_000, createdAt: 1, updatedAt: 1,
      }],
      lastSyncAt: null,
    },
  })
  const webhook: any = { id: 'w1', name: 'Deploy hook', enabled: true, secretRef: 'hook-secret', secretRotations: { secretRef: null } }
  const checks = buildReadinessDoctor({ settings, webhooks: [webhook], now })
  assert.match(find(checks, 'secret-age:remote-control')?.detail ?? '', /older than 90 days/)
  assert.match(find(checks, 'secret-age:whatsapp-bridge:wa_old')?.detail ?? '', /older than 90 days/, 'paired Baileys credentials warn at 90 days')
  assert.match(find(checks, 'secret-age:webhook:w1')?.detail ?? '', /age is unknown/)
}

// ---------------------------------------------------------------- security posture (unchanged behavior)

{
  const settings = baseSettings({
    approvalMode: 'auto',
    allowTaskNetwork: true,
    providers: [{ id: 'local', name: 'Local', kind: 'local', baseUrl: 'http://127.0.0.1:11434/v1', apiKey: '', models: ['qwen'] } as any],
    activeProviderId: 'local',
    activeModel: 'qwen',
  })
  const checks = buildReadinessDoctor({
    settings,
    providerReadiness: { [readinessProbeKey('local')]: { status: 'ok', value: { ready: true, reason: 'ok', message: '' } } },
    localClis: { status: 'ok', value: [cliStatus('shell', { command: null, installed: true, effectiveCommand: null })] },
    now,
  })
  assert.equal(find(checks, 'security:approval')?.status, 'warning')
  assert.match(find(checks, 'security:approval')?.detail ?? '', /read-only, Git-read, test, and build/)
  assert.match(find(checks, 'security:approval')?.detail ?? '', /mutations, network\/package work, destructive actions, and unknown/)
  assert.equal(find(checks, 'security:network')?.status, 'warning')
  assert.equal(worstDoctorStatus(checks), 'warning', 'ready provider + unproven connectors roll up to warning, not error')
}

{
  const healthy = buildReadinessDoctor({ settings: baseSettings(), now })
  assert.equal(find(healthy, 'security:persistence')?.status, 'ok')

  const failed = buildReadinessDoctor({
    settings: baseSettings(),
    persistenceFailure: { label: 'threads', message: 'Disk is full.', at: now - 5_000 },
    now,
  })
  assert.equal(find(failed, 'security:persistence')?.status, 'error')
  assert.match(find(failed, 'security:persistence')?.detail ?? '', /Recent changes may not survive restart/)
  assert.match(find(failed, 'security:persistence')?.detail ?? '', /Disk is full/)
}

{
  const profileKey = buildReadinessDoctor({
    settings: baseSettings(),
    securityControls: {
      localStateKeyMode: 'profile-key',
      externalSecretKeyRequired: false,
      copiedProfileDecryptable: true,
      legacySafeStorageSecrets: 0,
      legacySafeStorageBackupPath: null,
      quarantinedProtectedFiles: 0,
      quarantinedProtectedPaths: [],
      quarantineBackupPaths: [],
    },
    now,
  })
  assert.equal(find(profileKey, 'security:local-secrets')?.status, 'warning')
  assert.match(find(profileKey, 'security:local-secrets')?.detail ?? '', /complete TensorFold profile copy may include a decryption key/)

  const externalKey = buildReadinessDoctor({
    settings: baseSettings(),
    securityControls: {
      localStateKeyMode: 'external-env-key',
      externalSecretKeyRequired: true,
      copiedProfileDecryptable: false,
      legacySafeStorageSecrets: 0,
      legacySafeStorageBackupPath: null,
      quarantinedProtectedFiles: 0,
      quarantinedProtectedPaths: [],
      quarantineBackupPaths: [],
    },
    now,
  })
  assert.equal(find(externalKey, 'security:local-secrets')?.status, 'ok')

  const legacy = buildReadinessDoctor({
    settings: baseSettings(),
    securityControls: {
      localStateKeyMode: 'profile-key',
      externalSecretKeyRequired: false,
      copiedProfileDecryptable: true,
      legacySafeStorageSecrets: 2,
      legacySafeStorageBackupPath: '/recovery/settings.bak',
      quarantinedProtectedFiles: 0,
      quarantinedProtectedPaths: [],
      quarantineBackupPaths: [],
    },
    now,
  })
  assert.equal(find(legacy, 'security:local-secrets')?.status, 'error')
  assert.match(find(legacy, 'security:local-secrets')?.detail ?? '', /2 credentials could not be decrypted/)
  assert.match(find(legacy, 'security:local-secrets')?.detail ?? '', /\/recovery\/settings\.bak/)

  const quarantined = buildReadinessDoctor({
    settings: baseSettings(),
    securityControls: {
      localStateKeyMode: 'profile-key',
      externalSecretKeyRequired: false,
      copiedProfileDecryptable: true,
      legacySafeStorageSecrets: 0,
      legacySafeStorageBackupPath: null,
      quarantinedProtectedFiles: 1,
      quarantinedProtectedPaths: ['/state/threads.json'],
      quarantineBackupPaths: ['/state/threads.json.recovery.bak'],
    },
    now,
  })
  assert.equal(find(quarantined, 'security:local-secrets')?.status, 'error')
  assert.match(find(quarantined, 'security:local-secrets')?.detail ?? '', /write-blocked/)
}

console.log('readiness-doctor.test.ts passed')
