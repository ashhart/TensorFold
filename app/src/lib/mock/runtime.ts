import type { AgentEvent, ArchiveKind, ArchiveSnapshot, DiffLine, Settings, StartRunPayload, Thread, TimelineItem, Project, Room, RoomEventItem, RoomMemoryEntry, RoomStreamEvent, Ticket, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitStatus, TokenUsageSnapshot, TokenUsageWindow, TokenUsageWindowKey, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, AppUpdateStatus, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSyncResult, EmailThreadResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, Skill, Command, McpServerConfig, McpServerSnapshot, McpSnapshot, McpToolCallResult, McpToolDefinition, ControllerSnapshot, ControllerStatus, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookDeliveryRecord, WebhookEndpoint, WebhookSnapshot, RemoteControlPairingInfo, RemoteControlStatus, RemoteControlWorkSnapshot, SecurityControlSnapshot, SecurityPurgeResult } from '../types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DiscordSource, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, GoogleChatSource, MattermostSource, MatrixSource, SlackMessage, SlackListOptions, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from '../types'
import type { Incident, RunExportBundle } from '../types'
import { ago, mockState, uid } from './state'
import { simulate, thread } from './fixtures-core'

export function mockMissionSnapshot(): MissionRuntimeSnapshot {
  return {
    generatedAt: Date.now(),
    missions: mockState.mockMissions,
    activeMissionIds: mockState.mockMissions.filter((m) => ['queued', 'running', 'waiting_approval', 'paused', 'stalled'].includes(m.status)).map((m) => m.id),
    pendingApprovalCount: mockState.mockMissions.reduce((sum, m) => sum + m.pendingApprovals.length, 0),
  }
}

export function mockMissionInspection(id: string): MissionInspectionSnapshot {
  const mission = mockState.mockMissions.find((m) => m.id === id)
  if (!mission) throw new Error('Mission not found')
  const latestCheckpoint = mission.checkpoints.at(-1)
  const latestJournal = mission.journal.at(-1)
  return {
    generatedAt: Date.now(),
    mission,
    terminal: ['completed', 'failed', 'cancelled'].includes(mission.status),
    active: ['queued', 'running', 'waiting_approval', 'paused', 'stalled'].includes(mission.status),
    rerunRequired: false,
    checkpointCount: mission.checkpoints.length,
    journalCount: mission.journal.length,
    scratchMemoryCount: mission.scratchMemory.length,
    pendingApprovalCount: mission.pendingApprovals.length,
    latestCheckpointId: latestCheckpoint?.id ?? null,
    latestJournalId: latestJournal?.id ?? null,
  }
}

const MOCK_VISIBLE_JOB_STATUSES = new Set<Job['status']>(['queued', 'running', 'waiting_approval', 'paused', 'stalled', 'failed', 'dead_letter'])
const MOCK_RETRYABLE_JOB_STATUSES = new Set<Job['status']>(['failed', 'stalled', 'dead_letter'])
const SECRET_MASK = '••••••••'

function maskSecret(value?: string): string {
  return value ? SECRET_MASK : ''
}

function redactEmailSource(source: EmailSourceConfig): EmailSourceConfig {
  return { ...source, password: source.password ? SECRET_MASK : source.password }
}

function redactSlackSource(source: SlackSourceConfig): SlackSourceConfig {
  return { ...source, token: source.token ? SECRET_MASK : source.token }
}

function redactDiscordSource(source: DiscordSource): DiscordSource {
  return { ...source, token: maskSecret(source.token) }
}

function redactMattermostSource(source: MattermostSource): MattermostSource {
  return { ...source, token: maskSecret(source.token) }
}

function redactMatrixSource(source: MatrixSource): MatrixSource {
  return { ...source, token: maskSecret(source.token) }
}

function redactGoogleChatSource(source: GoogleChatSource): GoogleChatSource {
  return {
    ...source,
    webhookUrl: maskSecret(source.webhookUrl),
    accessToken: source.accessToken ? SECRET_MASK : source.accessToken,
  }
}

export function mockRemoteWorkSnapshot(now = Date.now()): RemoteControlWorkSnapshot {
  const jobs = {
    total: mockState.mockJobs.length,
    visible: 0,
    queued: 0,
    running: 0,
    waitingApproval: 0,
    paused: 0,
    stalled: 0,
    failed: 0,
    deadLetter: 0,
    retryable: 0,
  }
  const leases = { active: 0, expired: 0, expiringSoon: 0, missingRunning: 0 }
  const approvals = {
    pending: 0,
    tool: 0,
    connector: mockState.mockConnectorEvents.filter((event) => event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending').length,
    mission: mockState.mockMissions.reduce((sum, mission) => sum + mission.pendingApprovals.length, 0),
  }
  let nextWakeAt: number | null = null
  let nextWakeJobId: string | null = null

  for (const job of mockState.mockJobs) {
    if (MOCK_VISIBLE_JOB_STATUSES.has(job.status)) jobs.visible += 1
    if (job.status === 'queued') jobs.queued += 1
    if (job.status === 'running') jobs.running += 1
    if (job.status === 'waiting_approval') jobs.waitingApproval += 1
    if (job.status === 'paused') jobs.paused += 1
    if (job.status === 'stalled') jobs.stalled += 1
    if (job.status === 'failed') jobs.failed += 1
    if (job.status === 'dead_letter') jobs.deadLetter += 1
    if (MOCK_RETRYABLE_JOB_STATUSES.has(job.status)) jobs.retryable += 1
    approvals.tool += (job.pendingApprovals ?? []).filter((approval) => !approval.resolvedAt).length
    if (job.nextWakeAt && (nextWakeAt == null || job.nextWakeAt < nextWakeAt)) {
      nextWakeAt = job.nextWakeAt
      nextWakeJobId = job.id
    }
    if (job.leaseExpiresAt) {
      if (job.leaseExpiresAt <= now) leases.expired += 1
      else {
        leases.active += 1
        if (job.leaseExpiresAt - now <= 2 * 60_000) leases.expiringSoon += 1
      }
    } else if (job.status === 'running') {
      leases.missingRunning += 1
    }
  }

  approvals.pending = approvals.tool + approvals.connector + approvals.mission
  // Demo terminal health: one running session + one needing attention keeps the readiness
  // doctor's terminal subcheck exercised in the browser demo.
  const terminals = { total: 2, running: 1, attention: 1 }
  return { jobs, leases, approvals, nextWakeAt, nextWakeJobId, generatedAt: now, terminals }
}

export function updateMockMission(id: string, fn: (mission: Mission) => Mission): Mission {
  const mission = mockState.mockMissions.find((m) => m.id === id)
  if (!mission) throw new Error('Mission not found')
  const next = fn(mission)
  mockState.mockMissions = mockState.mockMissions.map((m) => m.id === id ? next : m)
  return next
}

export function createMockLocalPrLoop(input: CreatePrLoopInput): CreatePrLoopResult {
  const now = Date.now()
  const project = input.projectId ? mockState.projects.find((candidate) => candidate.id === input.projectId) : mockState.projects[0]
  const localPrId = `lpr_${uid().slice(0, 8)}`
  const title = (input.title || input.objective || 'Local PR loop').trim()
  const branch = `vontra/pr/${localPrId}`
  const baseBranch = input.baseBranch || 'main'
  const worktree = `/Users/example/.vontra/pr-loops/${localPrId}/worktree`
  const requiredApprovals = Math.max(1, Math.min(6, Math.round(Number(input.requiredApprovals ?? 1) || 1)))
  const plan = (steps: string[]) => steps.map((text) => ({ id: text.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40), text, done: false }))
  const implementationThread: Thread = {
    id: uid(),
    title: `Build local PR: ${title}`.slice(0, 96),
    kind: 'task',
    projectId: project?.id ?? input.projectId ?? null,
    cwd: worktree,
    providerId: mockState.settings.activeProviderId,
    providerAccountId: mockState.settings.activeProviderAccountId,
    model: mockState.settings.activeModel,
    createdAt: now,
    updatedAt: now,
    status: input.autoStart === false ? 'idle' : 'running',
    apiMessages: [],
    timeline: [{ id: uid(), kind: 'notice', text: `Local PR ${localPrId} is ready on ${branch}.`, createdAt: now }],
    goal: `Build the requested change on ${branch}, summarize the local diff, and address review comments until approved.`,
    goalStatus: 'active',
    plan: plan(['Confirm local PR branch and worktree', 'Implement the requested change', 'Run relevant checks', 'Post local PR summary for review', 'Address reviewer comments until approved']),
  }
  const reviewThread: Thread = {
    id: uid(),
    title: `Review local PR: ${title}`.slice(0, 96),
    kind: 'task',
    projectId: project?.id ?? input.projectId ?? null,
    cwd: worktree,
    providerId: mockState.settings.activeProviderId,
    providerAccountId: mockState.settings.activeProviderAccountId,
    model: mockState.settings.activeModel,
    createdAt: now,
    updatedAt: now,
    status: 'idle',
    apiMessages: [],
    timeline: [{ id: uid(), kind: 'notice', text: `Waiting to review ${branch} after the builder posts a summary.`, createdAt: now }],
    goal: `Review ${branch} against ${baseBranch}. Do not edit files. Leave comments or approval.`,
    goalStatus: 'active',
    plan: plan(['Wait for implementation ready signal', 'Review local diff against base', 'Leave actionable comments or approval']),
  }
  const mission: Mission = {
    id: uid(),
    title: `Local PR loop: ${title}`,
    objective: input.objective || title,
    stopCondition: `The local PR branch ${branch} has ${requiredApprovals} evidence-bound review pass${requiredApprovals === 1 ? '' : 'es'} and relevant checks are complete.`,
    repeatMode: 'until_condition',
    contextMode: 'build_on_previous',
    roleProfileId: 'builder',
    fanOutBudget: 'paired',
    maxRuns: 15,
    costCapUsd: null,
    expiresAt: null,
    destructiveApprovalRequired: true,
    status: 'queued',
    runBudgetClass: 'mission',
    budget: { executionClass: 'mission', label: 'Mission', maxWallSeconds: 3600, heartbeatIntervalSeconds: 30, stallGraceSeconds: 120, requiresHeartbeat: true, startedAt: null, deadlineAt: null },
    heartbeat: { intervalSeconds: 30, stallGraceSeconds: 120, required: true, count: 0, lastBeatAt: null, nextDueAt: null, missedSince: null },
    checkpoints: [
      { id: uid(), sequence: 1, label: 'Local PR worktree', summary: `Created ${branch} at ${worktree}.`, status: 'queued', createdAt: now, data: { localPrId, branch, worktree } },
    ],
    journal: [{ id: uid(), kind: 'created', actor: 'system', message: 'Local PR loop queued.', createdAt: now, status: 'queued' }],
    scratchMemory: [],
    pendingApprovals: [],
    threadId: implementationThread.id,
    projectId: project?.id ?? input.projectId ?? null,
    cwd: worktree,
    providerId: mockState.settings.activeProviderId,
    providerAccountId: mockState.settings.activeProviderAccountId,
    model: mockState.settings.activeModel,
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    finishedAt: null,
    metadata: { workflow: 'local_pr_loop', localPrId, implementationThreadId: implementationThread.id, reviewThreadId: reviewThread.id, branch, baseBranch, worktree, requiredApprovals, remotePublish: { requested: false, approved: false, allowed: false, reason: 'Remote publish/push/open-PR is outside the local PR loop and requires a separate explicit approval.' } },
  }
  mockState.threads = [implementationThread, reviewThread, ...mockState.threads]
  mockState.mockMissions = [mission, ...mockState.mockMissions]
  if (input.autoStart !== false) simulate(implementationThread.id, implementationThread.goal || input.objective)
  return { localPrId, branch, baseBranch, worktree, mission, implementationThread, reviewThread }
}

mockState.mockSkills = [
  { id: 's_review', name: 'code-review', description: 'Review a diff for bugs, security, and clarity', body: 'Review the changes for correctness, security, edge cases, and clarity. Be specific and cite file:line. Flag anything risky before approving.', enabled: true, createdAt: Date.now(), updatedAt: Date.now() },
  { id: 's_commit', name: 'commit-message', description: 'Write a clean conventional commit', body: 'Write a concise conventional-commit message for the staged changes: a `type(scope): summary` line under 72 chars, then a short body explaining the why.', enabled: true, createdAt: Date.now(), updatedAt: Date.now() },
]
// First-class artifacts (VST-011 demo): one textual report (redactable), one screenshot.
// Attached to the seeded dark-mode thread (ids are minted at load).
const demoArtifactThreadId = mockState.threads[0]?.id ?? 'demo-thread'
mockState.mockArtifacts = [
  { id: 'art_report', title: 'Dark mode rollout report', kind: 'report', path: null, inlineText: 'Rollout summary for maya@example.com: toggle shipped behind the header, persisted via localStorage; deploy token sk-live-a1b2c3d4e5f6g7h8i9j0k1l2 rotated. All 3 files updated.', inlineTruncated: false, mimeType: null, sizeBytes: 152, provenance: { threadId: demoArtifactThreadId, runId: null, turn: 1, tool: 'save_artifact', sourceRef: 'npm test output' }, redaction: 'unredacted', redactionReport: [], createdAt: Date.now(), updatedAt: Date.now() },
  { id: 'art_shot', title: 'Header toggle screenshot', kind: 'screenshot', path: '~/screens/toggle.png', inlineText: null, inlineTruncated: false, mimeType: 'image/png', sizeBytes: 48211, provenance: { threadId: demoArtifactThreadId, runId: null, turn: 1, tool: 'save_artifact', sourceRef: null }, redaction: 'unredacted', redactionReport: [], createdAt: Date.now(), updatedAt: Date.now() },
]
// Repo-authored project skills (VST-010 demo): one untrusted, one trusted, one changed.
mockState.mockProjectSkills = [
  { contentHash: 'demo-hash-a', trust: 'untrusted', skill: { id: 'ps_proj-demo_deploy-checklist', name: 'Deploy checklist', description: 'Use when shipping this service to production.', body: 'Run the release script, verify /healthz, watch error rates for 10 minutes.\n\n## Tests\ncurl /healthz expects 200.', enabled: false, scope: 'project', projectId: 'proj-demo', createdAt: Date.now(), updatedAt: Date.now() } },
  { contentHash: 'demo-hash-b', trust: 'trusted', skill: { id: 'ps_proj-demo_db-migrations', name: 'DB migrations', description: 'Use when altering the schema.', body: 'Write reversible migrations; test down-migrations locally before review.', enabled: true, scope: 'project', projectId: 'proj-demo', createdAt: Date.now(), updatedAt: Date.now() } },
  { contentHash: 'demo-hash-c', trust: 'changed', skill: { id: 'ps_proj-demo_incident-runbook', name: 'Incident runbook', description: 'Use during a production incident.', body: 'Page the on-call, open an incident, mitigate before root-causing.', enabled: false, scope: 'project', projectId: 'proj-demo', createdAt: Date.now(), updatedAt: Date.now() } },
]
// A pending in-place improvement (VST-010 demo): the review card shows the body diff.
mockState.mockSuggestions = [
  { id: 'sg_demo_upd', kind: 'skill', name: 'code-review', description: 'Review a diff for bugs, security, and clarity', body: 'Review the changes for correctness, security, edge cases, and clarity. Cite file:line for every finding, check error handling on new paths, and flag anything risky before approving.', rationale: 'Recent reviews kept missing unhandled promise rejections — the added error-handling pass catches them.', sourceThreadId: null, targetSkillId: 's_review', status: 'pending', createdAt: Date.now(), updatedAt: Date.now() },
]
mockState.mockProjectMemory = [
  { id: 'm_arch', projectId: 'proj-demo', kind: 'architecture', text: 'Electron main owns all tools/IPC; renderer is context-isolated and talks only via window.tensorfoldStudio.', sourceType: 'manual', sourceRef: 'CLAUDE.md', confidence: 'high', privacyScope: 'project', owner: null, lastVerifiedAt: Date.now(), createdAt: Date.now(), updatedAt: Date.now() },
  { id: 'm_conv', projectId: 'proj-demo', kind: 'convention', text: 'Use semantic Tailwind tokens (bg-card/text-ink-2), never raw colors.', sourceType: 'manual', sourceRef: null, confidence: 'medium', privacyScope: 'project', owner: null, lastVerifiedAt: null, createdAt: Date.now(), updatedAt: Date.now() },
  { id: 'm_pref', projectId: 'proj-demo', kind: 'preference', text: 'Maya prefers concise roadmap updates with exact completion percentages.', sourceType: 'manual', sourceRef: null, confidence: 'high', privacyScope: 'personal', owner: 'Maya', lastVerifiedAt: null, createdAt: Date.now(), updatedAt: Date.now() },
]
mockState.mockProjectMemoryCandidates = [
  { id: 'mc_demo', projectId: 'proj-demo', kind: 'person', text: 'Tom handles planning in rooms when assigned by the user.', sourceType: 'manual', sourceRef: null, confidence: 'medium', privacyScope: 'team', owner: 'demo', reason: 'Demo candidate awaiting approval.', status: 'pending', targetEntryId: null, reviewedAt: null, reviewedBy: null, durableEntryId: null, createdAt: Date.now(), updatedAt: Date.now() },
]
mockState.mockWebhooks = [
  { id: 'wh_demo', name: 'GitHub CI', direction: 'inbound', enabled: false, scope: {}, path: '/hooks/github', secretRef: 'demo-secret', signatureMode: 'hmac-sha256', replayWindowSeconds: 300, capturePolicy: 'redacted_payload', triggerPolicy: 'notify', runAutomationId: null, createdAt: Date.now(), updatedAt: Date.now() },
  { id: 'wh_out_demo', name: 'Ops notifier', direction: 'outbound', enabled: false, scope: {}, url: 'https://example.com/vontra', secretRef: 'demo-secret', signatureMode: 'hmac-sha256', replayWindowSeconds: 300, capturePolicy: 'redacted_payload', triggerPolicy: 'notify', runAutomationId: null, eventKinds: ['mission', 'ticket', 'automation'], approvalRequired: true, approvedAt: null, deliveries: [], createdAt: Date.now(), updatedAt: Date.now() },
]
export const mockWebhookSnapshot = (): WebhookSnapshot => ({
  endpoints: mockState.mockWebhooks.map((w) => ({ ...w, secretRef: w.secretRef ? '••••••••' : '' })),
  listener: {
    listening: mockState.mockWebhooks.some((w) => w.direction === 'inbound' && w.enabled),
    port: mockState.mockWebhooks.some((w) => w.direction === 'inbound' && w.enabled) ? 7717 : null,
    routes: mockState.mockWebhooks.filter((w) => w.direction === 'inbound' && w.enabled).length,
    lastDeliveryAt: mockState.mockWebhooks.flatMap((w) => w.deliveries ?? []).reduce<number | null>((latest, d) => Math.max(latest ?? 0, d.finishedAt ?? d.createdAt), null),
  },
})
mockState.mockIncidents = [
  {
    id: 'inc_demo', title: 'Checkout 500s spike', severity: 'sev2', status: 'investigating', cwd: '/repo',
    summary: 'Error rate on /checkout jumped to ~12% after the 14:02 deploy.', createdAt: ago(30), updatedAt: ago(4),
    timeline: [
      { id: 'ie1', kind: 'symptom', text: '5xx rate on /checkout climbed from 0.2% to 12%.', at: ago(30), source: 'user' },
      { id: 'ie2', kind: 'scan', text: '## Symptoms\n- 500s localized to /checkout.\n\n## Likely cause\n- New payment client timeout (2s) below upstream p99 (3.4s).', at: ago(20), source: 'scan' },
    ],
    mitigations: [
      { id: 'im1', text: 'Raise payment client timeout to 5s and redeploy.', rationale: 'Upstream p99 is 3.4s; 2s is too aggressive.', status: 'proposed', proposedAt: ago(18) },
    ],
    scan: { status: 'done', findings: '## Symptoms\n- 500s localized to /checkout.\n\n## Likely cause\n- New payment client timeout (2s) below upstream p99 (3.4s).', ranAt: ago(20) },
  },
]
export function mockLocalMemoryCounts(): Record<string, number> {
  return {
    threads: mockState.threads.length,
    projects: mockState.projects.length,
    rooms: mockState.mockRooms.length,
    tickets: mockState.mockTickets.length,
    automations: mockState.mockAutomations.length,
    connectorEvents: mockState.mockConnectorEvents.length,
    missions: mockState.mockMissions.length,
    jobs: mockState.mockJobs.length,
    botSessions: 0,
    projectMemory: mockState.mockProjectMemory.length,
    incidents: mockState.mockIncidents.length,
  }
}

export function mockSecurityControls(): SecurityControlSnapshot {
  return {
    actor: 'browser-demo@local',
    role: 'owner',
    commandSandboxAvailable: true,
    mcpSandboxAvailable: true,
    signedUpdatesRequired: true,
    updatePublicKeyConfigured: true,
    updateTrustRootProductionReady: false,
    signedMcpRequired: true,
    trustedMcpExplicitlyAllowed: false,
    localStateKeyMode: 'profile-key',
    externalSecretKeyRequired: false,
    copiedProfileDecryptable: true,
    legacySafeStorageSecrets: 0,
    legacySafeStorageBackupPath: null,
    quarantinedProtectedFiles: 0,
    quarantinedProtectedPaths: [],
    quarantineBackupPaths: [],
    unencryptedLocalStoreAllowed: false,
    strictAuditEnabled: false,
    auditHealthy: true,
    lastAuditError: null,
    lastPersistenceFailure: null,
    localMemoryCounts: mockLocalMemoryCounts(),
  }
}

mockState.mockCommands = [
  { id: 'c_test', name: 'test', description: 'Run the test suite and fix failures', prompt: 'Run the test suite, then diagnose and fix any failures you find.', createdAt: Date.now(), updatedAt: Date.now() },
  { id: 'c_explain', name: 'explain', description: 'Explain a file', argHint: '<file>', prompt: 'Explain what this file does and how it fits into the wider codebase:', createdAt: Date.now(), updatedAt: Date.now() },
]

mockState.mockMcpServers = [
  { id: 'filesystem_demo', name: 'Filesystem demo', transport: 'stdio', command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '/Users/example/Projects'], cwd: null, env: {}, enabled: true, autoConnect: false, approvalMode: 'ask', createdAt: ago(120), updatedAt: ago(10) },
]
mockState.mockMcpStopped = new Set<string>()
mockState.mockGitHygieneSanitized = false

export function mockMcpTools(server: McpServerConfig): McpToolDefinition[] {
  if (!server.enabled) return []
  return [
    {
      serverId: server.id,
      serverName: server.name,
      approvalMode: server.approvalMode,
      name: 'list_allowed_directories',
      callName: `mcp__${server.id}__list_allowed_directories`,
      title: 'List allowed directories',
      description: 'List the filesystem roots this MCP server can access.',
      inputSchema: { type: 'object', properties: {} },
      readOnlyHint: true,
    },
    {
      serverId: server.id,
      serverName: server.name,
      approvalMode: server.approvalMode,
      name: 'write_file',
      callName: `mcp__${server.id}__write_file`,
      title: 'Write file',
      description: 'Write a file through the MCP filesystem server.',
      inputSchema: { type: 'object', properties: { path: { type: 'string' }, content: { type: 'string' } }, required: ['path', 'content'] },
      destructiveHint: true,
    },
  ]
}

export function mockMcpResources(server: McpServerConfig) {
  if (!server.enabled) return []
  return [
    {
      serverId: server.id,
      serverName: server.name,
      uri: 'file:///Users/example/Projects/README.md',
      name: 'README.md',
      title: 'Project README',
      description: 'Demo MCP resource exposed by the filesystem server.',
      mimeType: 'text/markdown',
    },
  ]
}

export function mockMcpPrompts(server: McpServerConfig) {
  if (!server.enabled) return []
  return [
    {
      serverId: server.id,
      serverName: server.name,
      name: 'summarize_project',
      title: 'Summarize project',
      description: 'Summarize the selected project resource.',
      arguments: [{ name: 'uri', description: 'Resource URI', required: true }],
    },
  ]
}

export function mockMcpSnapshot(): McpSnapshot {
  const servers: McpServerSnapshot[] = mockState.mockMcpServers.map((server) => {
    const connected = server.enabled && !mockState.mockMcpStopped.has(server.id)
    const tools = connected ? mockMcpTools(server) : []
    const resources = connected ? mockMcpResources(server) : []
    const prompts = connected ? mockMcpPrompts(server) : []
    return {
      ...server,
      env: Object.fromEntries(Object.keys(server.env ?? {}).map((key) => [key, '[set]'])),
      runtime: {
        id: server.id,
        status: connected ? 'connected' : 'stopped',
        pid: connected ? 4242 : null,
        provenance: connected ? { commandPath: '/usr/local/bin/npx', commandSha256: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', checkedAt: ago(8) } : null,
        serverInfo: connected ? { name: server.name, version: 'demo' } : null,
        capabilities: connected ? { tools: { listChanged: true }, resources: { listChanged: true }, prompts: { listChanged: true } } : null,
        tools,
        resources,
        prompts,
        lastConnectedAt: connected ? ago(8) : null,
        lastError: null,
        logs: connected ? ['Demo MCP server connected.', `${tools.length} tools discovered.`] : [],
      },
    }
  })
  return {
    servers,
    tools: servers.flatMap((server) => server.runtime.tools),
    resources: servers.flatMap((server) => server.runtime.resources),
    prompts: servers.flatMap((server) => server.runtime.prompts),
    updatedAt: Date.now(),
  }
}

export function mockBlockedCommandReason(command: string): string | null {
  const words = String(command || '').match(/"[^"]*"|'[^']*'|`[^`]*`|\S+/g)?.map((word) => word.replace(/^['"`]|['"`]$/g, '')) ?? []
  for (let i = 0; i < words.length; i++) {
    if (words[i].toLowerCase() !== 'git') continue
    for (let j = i + 1; j < words.length; j++) {
      const word = words[j]
      if (['-C', '-c', '--git-dir', '--work-tree', '--namespace', '--exec-path', '--config-env'].includes(word)) { j++; continue }
      if (/^(?:-C|-c|--git-dir=|--work-tree=|--namespace=|--exec-path=|--config-env=)/.test(word)) continue
      if (word.startsWith('-')) continue
      if (word === 'commit') return 'use TensorFold commit so Git hygiene and AI co-author stripping run first'
      if (word === 'push') return 'run Git hygiene before pushing to remote version control'
      break
    }
  }
  return null
}

export function mockGitHygieneReport(cwd: string, message = ''): GitHygieneReport {
  const findings: GitHygieneFinding[] = mockState.mockGitHygieneSanitized || message.toLowerCase().includes('clean')
    ? []
    : [
        {
          id: 'mock_email',
          source: 'working_tree',
          kind: 'email',
          severity: 'medium',
          file: 'README.md',
          line: 18,
          commit: null,
          excerpt: 'Contact: a***@example.com',
          match: 'a***@example.com',
          suggestion: 'Replace personal email addresses with a role alias or documentation placeholder.',
          autoFixable: true,
        },
        {
          id: 'mock_path',
          source: 'working_tree',
          kind: 'absolute_path',
          severity: 'medium',
          file: 'docs/setup.md',
          line: 7,
          commit: null,
          excerpt: 'Workspace path: /Volumes/example/…',
          match: '/Volumes/example/…',
          suggestion: 'Replace absolute local paths with repo-relative paths or placeholders.',
          autoFixable: true,
        },
      ]
  const summary = {
    critical: findings.filter((f) => f.severity === 'critical').length,
    high: findings.filter((f) => f.severity === 'high').length,
    medium: findings.filter((f) => f.severity === 'medium').length,
    low: findings.filter((f) => f.severity === 'low').length,
  }
  return {
    cwd,
    scannedAt: Date.now(),
    scannedFiles: 42,
    scannedBytes: 220_000,
    scannedCommits: 8,
    findings,
    blocked: summary.critical > 0 || summary.high > 0 || summary.medium > 0,
    summary,
    notes: ['Browser demo report. Desktop mode scans real files and Git history.'],
  }
}

export const activeOnly = <T extends { archivedAt?: number | null }>(items: T[]) => items.filter((item) => !item.archivedAt)
export function mockArchiveSnapshot(): ArchiveSnapshot {
  const projectArchiveAt = new Map(mockState.projects.filter((project) => project.archivedAt).map((project) => [project.id, project.archivedAt!]))
  const roomArchiveAt = new Map(mockState.mockRooms.filter((room) => room.archivedAt).map((room) => [room.id, room.archivedAt!]))
  return {
    projects: mockState.projects.filter((p) => !!p.archivedAt),
    threads: mockState.threads.filter((t) => !!t.archivedAt && !projectArchiveAt.has(t.projectId ?? '')),
    rooms: mockState.mockRooms.filter((r) => !!r.archivedAt && !projectArchiveAt.has(r.projectId)),
    tickets: mockState.mockTickets.filter((t) => !!t.archivedAt && !projectArchiveAt.has(t.projectId) && (!t.roomId || !roomArchiveAt.has(t.roomId))),
  }
}
export function mockArchive(kind: ArchiveKind, id: string): ArchiveSnapshot {
  const now = Date.now()
  if (kind === 'project') {
    const p = mockState.projects.find((x) => x.id === id)
    const batchAt = p?.archivedAt ?? now
    if (p && !p.archivedAt) { p.archivedAt = batchAt; p.updatedAt = now }
    for (const t of mockState.threads.filter((x) => x.projectId === id && !x.archivedAt)) { t.archivedAt = batchAt; t.updatedAt = now }
    for (const r of mockState.mockRooms.filter((x) => x.projectId === id && !x.archivedAt)) { r.archivedAt = batchAt; r.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.projectId === id && !x.archivedAt)) { t.archivedAt = batchAt; t.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.projectId === id)) { t.archivedAt = now; t.updatedAt = now }
  } else if (kind === 'thread') {
    const t = mockState.threads.find((x) => x.id === id)
    if (t) { t.archivedAt = now; t.updatedAt = now }
  } else if (kind === 'room') {
    const r = mockState.mockRooms.find((x) => x.id === id)
    const batchAt = r?.archivedAt ?? now
    if (r && !r.archivedAt) { r.archivedAt = batchAt; r.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.roomId === id && !x.archivedAt)) { t.archivedAt = batchAt; t.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.roomId === id)) { t.archivedAt = now; t.updatedAt = now }
  } else {
    const t = mockState.mockTickets.find((x) => x.id === id)
    if (t) { t.archivedAt = now; t.updatedAt = now }
  }
  return mockArchiveSnapshot()
}
export function mockRestore(kind: ArchiveKind, id: string): ArchiveSnapshot {
  const now = Date.now()
  if (kind === 'project') {
    const p = mockState.projects.find((x) => x.id === id)
    const batchAt = p?.archivedAt
    if (p) { p.archivedAt = null; p.updatedAt = now }
    for (const t of mockState.threads.filter((x) => x.projectId === id && x.archivedAt === batchAt)) { t.archivedAt = null; t.updatedAt = now }
    for (const r of mockState.mockRooms.filter((x) => x.projectId === id && x.archivedAt === batchAt)) { r.archivedAt = null; r.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.projectId === id && x.archivedAt === batchAt)) { t.archivedAt = null; t.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.projectId === id)) { t.archivedAt = null; t.updatedAt = now }
  } else if (kind === 'thread') {
    const t = mockState.threads.find((x) => x.id === id)
    if (t) { t.archivedAt = null; t.updatedAt = now }
  } else if (kind === 'room') {
    const r = mockState.mockRooms.find((x) => x.id === id)
    const batchAt = r?.archivedAt
    if (r) { r.archivedAt = null; r.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.roomId === id && x.archivedAt === batchAt)) { t.archivedAt = null; t.updatedAt = now }
    for (const t of mockState.mockTickets.filter((x) => x.roomId === id)) { t.archivedAt = null; t.updatedAt = now }
  } else {
    const t = mockState.mockTickets.find((x) => x.id === id)
    if (t) { t.archivedAt = null; t.updatedAt = now }
  }
  return mockArchiveSnapshot()
}

export function mockRemoteStatus(): RemoteControlStatus {
  const bindHost = mockState.settings.remoteControl.allowLan ? mockState.settings.remoteControl.bindHost : '127.0.0.1'
  return {
    enabled: mockState.settings.remoteControl.enabled,
    listening: mockState.settings.remoteControl.enabled,
    serverName: mockState.settings.remoteControl.serverName,
    bindHost,
    port: mockState.settings.remoteControl.enabled ? mockState.settings.remoteControl.port : null,
    url: mockState.settings.remoteControl.enabled ? `http://${bindHost}:${mockState.settings.remoteControl.port}` : null,
    clients: mockState.settings.remoteControl.enabled ? 1 : 0,
    lastStartedAt: mockState.settings.remoteControl.lastStartedAt,
    lastClientAt: mockState.settings.remoteControl.lastClientAt,
    lastError: mockState.settings.remoteControl.lastError,
  }
}

export function mockRemotePairingInfo(): RemoteControlPairingInfo {
  return {
    url: mockRemoteStatus().url,
    token: mockState.settings.remoteControl.authToken || 'demo_remote_token',
    status: mockRemoteStatus(),
  }
}

export function redactedMockSettings(): Settings {
  return {
    ...mockState.settings,
    providers: mockState.settings.providers.map((p) => ({
      ...p,
      apiKey: maskSecret(p.apiKey),
      cliEnv: p.cliEnv ? Object.fromEntries(Object.entries(p.cliEnv).map(([key, value]) => [key, maskSecret(value)])) : p.cliEnv,
      accounts: p.accounts?.map((account) => ({
        ...account,
        apiKey: maskSecret(account.apiKey),
        env: account.env ? Object.fromEntries(Object.entries(account.env).map(([key, value]) => [key, maskSecret(value)])) : account.env,
      })),
    })),
    email: { ...mockState.settings.email, sources: mockState.settings.email.sources.map(redactEmailSource) },
    slack: { ...mockState.settings.slack, sources: mockState.settings.slack.sources.map(redactSlackSource) },
    discord: { ...mockState.settings.discord, sources: mockState.settings.discord.sources.map(redactDiscordSource) },
    mattermost: { ...mockState.settings.mattermost, sources: mockState.settings.mattermost.sources.map(redactMattermostSource) },
    matrix: { ...mockState.settings.matrix, sources: mockState.settings.matrix.sources.map(redactMatrixSource) },
    googlechat: { ...mockState.settings.googlechat, sources: mockState.settings.googlechat.sources.map(redactGoogleChatSource) },
    botControl: {
      ...mockState.settings.botControl,
      telegram: { ...mockState.settings.botControl.telegram, botToken: maskSecret(mockState.settings.botControl.telegram.botToken) },
      whatsapp: {
        ...mockState.settings.botControl.whatsapp,
        accessToken: maskSecret(mockState.settings.botControl.whatsapp.accessToken),
        verifyToken: maskSecret(mockState.settings.botControl.whatsapp.verifyToken),
        appSecret: maskSecret(mockState.settings.botControl.whatsapp.appSecret),
      },
    },
    voice: {
      ...mockState.settings.voice,
      apiKey: maskSecret(mockState.settings.voice.apiKey),
      elevenLabsApiKey: maskSecret(mockState.settings.voice.elevenLabsApiKey),
    },
    remoteControl: {
      ...mockState.settings.remoteControl,
      authToken: maskSecret(mockState.settings.remoteControl.authToken),
    },
    remoteClient: {
      ...mockState.settings.remoteClient,
      authToken: maskSecret(mockState.settings.remoteClient.authToken),
    },
    remoteOrchestrators: mockState.settings.remoteOrchestrators.map((host) => ({
      ...host,
      authToken: maskSecret(host.authToken),
    })),
  }
}
