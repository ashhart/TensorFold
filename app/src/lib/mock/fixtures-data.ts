import type { AgentEvent, ArchiveKind, ArchiveSnapshot, DiffLine, Settings, StartRunPayload, Thread, TimelineItem, Project, Room, RoomEventItem, RoomMemoryEntry, RoomStreamEvent, Ticket, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitStatus, TokenUsageSnapshot, TokenUsageWindow, TokenUsageWindowKey, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, AppUpdateStatus, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSyncResult, EmailThreadResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, Skill, Command, McpServerConfig, McpServerSnapshot, McpSnapshot, McpToolCallResult, McpToolDefinition, ControllerSnapshot, ControllerStatus, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookDeliveryRecord, WebhookEndpoint, WebhookSnapshot, RemoteControlPairingInfo, RemoteControlStatus, RemoteControlWorkSnapshot, SecurityControlSnapshot, SecurityPurgeResult } from '../types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, SlackMessage, SlackListOptions, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from '../types'
import type { Incident, RunExportBundle } from '../types'
import { ago, mockState, startOfMockDay, uid } from './state'

// ---- Tickets (demo) ----
export function mt(projectId: string, id: string, title: string, body: string, type: Ticket['type'], status: Ticket['status'], priority: Ticket['priority'], effort: Ticket['effort'], tags: string[], parentId: string | null, roomId: string | null = null): Ticket {
  return { id, projectId, roomId, title, body, type, status, priority, effort, tags, parentId, assignee: null, threadId: null, runState: status === 'in_progress' ? 'running' : null, order: Date.now() + Math.random(), createdAt: ago(120), updatedAt: ago(8) }
}
const EID = 'tk-epic'
mockState.mockTickets = [
  mt('proj-demo', EID, 'Epic: Local marketplace opening-hours comparison', 'Build a small local-first TypeScript/Effect app that fetches Foodhub, Uber Eats, and Just Eat opening hours and compares them.', 'epic', 'todo', 'urgent', 'large', ['scraping', 'local-first'], null),
  mt('proj-demo', 'tk1', 'Generate simple static HTML dashboard', 'Create the local report the user can open after a run.', 'subticket', 'todo', 'high', 'medium', ['dashboard', 'html'], EID),
  mt('proj-demo', 'tk2', 'Port minimal provider listing fetchers + snapshot CLI', 'First working manual fetch command that pulls raw snapshots.', 'subticket', 'ready', 'high', 'medium', ['scraping', 'providers'], EID),
  mt('proj-demo', 'tk3', 'Add provider detail fetching + opening-status extraction', 'Fetch or parse matched provider detail sources by provider.', 'subticket', 'ready', 'high', 'large', ['scraping', 'parsing'], EID),
  mt('proj-demo', 'tk4', 'Implement manual match loading + comparison JSON', 'Use the human-curated provider match file as the only source.', 'subticket', 'ready', 'high', 'medium', ['matching', 'comparison'], EID),
  mt('proj-demo', 'tk5', 'Scaffold minimal local Effect app + data contracts', 'Create the empty repo app foundation so later tasks can build on it.', 'subticket', 'in_progress', 'high', 'medium', ['setup', 'effect'], EID, 'room-demo'),
]
// Demo: show the room supervisor panel a seat actively holding a ticket.
;(() => { const t = mockState.mockTickets.find((x) => x.id === 'tk5'); if (t) t.assignee = 'architect' })()

mockState.ticketListener = null
export const ticketEmit = (t: Ticket) => mockState.ticketListener?.(t)
mockState.mockJobs = []
mockState.jobListener = null
export const jobEmit = (j: Job) => mockState.jobListener?.(j)
export function mockJob(kind: Job['kind'], owner: string, patch: Partial<Job> = {}): Job {
  const terminal = new Set<Job['status']>(['completed', 'failed', 'cancelled', 'dead_letter'])
  const existing = mockState.mockJobs.find((job) => job.kind === kind && job.owner === owner && !terminal.has(job.status))
  const now = Date.now()
  const job: Job = {
    id: existing?.id ?? uid(),
    kind,
    status: patch.status ?? existing?.status ?? 'running',
    attempt: patch.attempt ?? existing?.attempt ?? 1,
    owner,
    projectId: patch.projectId ?? existing?.projectId ?? null,
    missionId: patch.missionId ?? existing?.missionId ?? null,
    roomId: patch.roomId ?? existing?.roomId ?? null,
    threadId: patch.threadId ?? existing?.threadId ?? null,
    ticketId: patch.ticketId ?? existing?.ticketId ?? null,
    automationId: patch.automationId ?? existing?.automationId ?? null,
    connectorEventId: patch.connectorEventId ?? existing?.connectorEventId ?? null,
    currentRunId: patch.currentRunId ?? existing?.currentRunId ?? null,
    currentThreadId: patch.currentThreadId ?? existing?.currentThreadId ?? patch.threadId ?? null,
    nextWakeAt: patch.nextWakeAt ?? existing?.nextWakeAt ?? null,
    lastUsefulProgressAt: patch.lastUsefulProgressAt ?? now,
    blockedReason: patch.blockedReason ?? existing?.blockedReason ?? null,
    resumePlan: patch.resumePlan ?? existing?.resumePlan ?? null,
    resumeAfterRestart: patch.resumeAfterRestart ?? existing?.resumeAfterRestart ?? 'ask',
    pendingApprovals: patch.pendingApprovals ?? existing?.pendingApprovals ?? [],
    leaseOwner: patch.leaseOwner ?? existing?.leaseOwner ?? 'demo',
    leaseExpiresAt: patch.leaseExpiresAt ?? existing?.leaseExpiresAt ?? now + 5 * 60_000,
    startedAt: patch.startedAt ?? existing?.startedAt ?? now,
    finishedAt: patch.finishedAt ?? null,
    metadata: { ...(existing?.metadata ?? {}), ...(patch.metadata ?? {}) },
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  }
  mockState.mockJobs = [job, ...mockState.mockJobs.filter((candidate) => candidate.id !== job.id)]
  jobEmit(job)
  return job
}
export function replaceMockJob(id: string, patch: Partial<Job>, allowTerminal = false): Job {
  const terminal = new Set<Job['status']>(['completed', 'failed', 'cancelled', 'dead_letter'])
  const existing = mockState.mockJobs.find((job) => job.id === id)
  if (!existing) throw new Error('Job not found')
  if (!allowTerminal && terminal.has(existing.status)) throw new Error('Job cannot be changed from its current state.')
  const next: Job = {
    ...existing,
    ...patch,
    metadata: { ...(existing.metadata ?? {}), ...(patch.metadata ?? {}) },
    updatedAt: Date.now(),
  }
  mockState.mockJobs = [next, ...mockState.mockJobs.filter((job) => job.id !== id)]
  jobEmit(next)
  return next
}

// ---- Automations (demo) ----
export function mkAuto(p: Partial<Automation>): Automation {
  return {
    id: p.id || uid(), name: p.name || 'Untitled automation', prompt: p.prompt || '',
    scheduleKind: p.scheduleKind || 'manual', everyMinutes: p.everyMinutes ?? 60, atHour: p.atHour ?? 9, atMinute: p.atMinute ?? 0,
    providerId: p.providerId ?? 'openai', model: p.model ?? 'gpt-4o', mode: p.mode || 'chat', cwd: p.cwd ?? null,
    deliverTo: p.deliverTo || 'inbox', enabled: p.enabled ?? false,
    lastRunAt: p.lastRunAt ?? null, nextRunAt: p.nextRunAt ?? null, lastStatus: p.lastStatus ?? null,
    runs: p.runs ?? [], createdAt: ago(2000), updatedAt: ago(40),
    // Industrial fields (VST-038) carried through so a template / NL-built demo draft keeps them.
    ...(p.misfire ? { misfire: p.misfire } : {}),
    ...(p.requiredRisks ? { requiredRisks: p.requiredRisks } : {}),
    ...(p.acknowledgedRisks ? { acknowledgedRisks: p.acknowledgedRisks } : {}),
    ...(p.completionPolicy ? { completionPolicy: p.completionPolicy } : {}),
  }
}
mockState.mockAutomations = [
  mkAuto({
    name: 'Daily AI News Brief', prompt: 'Search the web for the most important AI developments in the last 24 hours and summarize the top 5 with one-line takeaways and links.',
    scheduleKind: 'daily', atHour: 9, atMinute: 0, providerId: 'openai', model: 'gpt-4o', mode: 'chat', deliverTo: 'telegram', enabled: true,
    lastRunAt: ago(180), lastStatus: 'ok', nextRunAt: Date.now() + 6 * 3600_000,
    runs: [{ id: uid(), startedAt: ago(180), finishedAt: ago(179), status: 'ok', output: '1. OpenAI ships GPT-5.5 with native tool-use — faster agentic loops.\n2. Nous Research ships new open agent weights — strong local agent perf.\n3. Anthropic adds memory to Claude Code — persistent project context.\n4. Mistral raises Series C — pushing on-device inference.\n5. Meta releases Llama 4 Scout — 10M-token context.' }],
  }),
  mkAuto({
    name: 'Inbox triage', prompt: 'Scan new email since the last run, label anything urgent, and draft concise replies for messages that need one. Summarize what you did.',
    scheduleKind: 'interval', everyMinutes: 30, providerId: 'ollama', model: 'qwen2.5-coder:14b', mode: 'agent', deliverTo: 'email', enabled: false,
  }),
  mkAuto({
    name: 'Standup digest', prompt: "Summarize yesterday's commits and open PRs across the team repos into a short standup digest.",
    scheduleKind: 'daily', atHour: 8, atMinute: 30, providerId: 'zai', model: 'glm-4.6', mode: 'chat', deliverTo: 'inbox', enabled: true,
    nextRunAt: Date.now() + 14 * 3600_000,
  }),
]
mockState.autoListener = null
export const autoEmit = (a: Automation) => mockState.autoListener?.({ ...a, runs: [...a.runs] })

mockState.updateListener = null
mockState.mockUpdateStatus = {
  state: 'available',
  currentVersion: '0.1.0',
  latestVersion: '0.1.1',
  releaseName: 'TensorFold 0.1.1',
  releaseNotes: 'Demo update feed: restart-safe update detection, release notes, and a download action.',
  downloadUrl: 'https://example.com/vontra/releases/0.1.1',
  checksum: { algorithm: 'sha256', value: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' },
  signature: 'demo-signature',
  signatureVerified: true,
  integrity: 'signature_verified',
  feedUrl: 'https://example.com/vontra/update.json',
  checkedAt: ago(12),
  error: null,
}
export const updateEmit = (s: AppUpdateStatus) => mockState.updateListener?.(s)

export function mockUsageWindow(key: TokenUsageWindowKey, label: string, multiplier: number): TokenUsageWindow {
  const accounts = [
    {
      id: 'claude' as const,
      name: 'Claude',
      latestAt: ago(4),
      total: Math.round(1_820_000 * multiplier),
      input: Math.round(540_000 * multiplier),
      output: Math.round(190_000 * multiplier),
      cacheCreation: Math.round(120_000 * multiplier),
      cacheRead: Math.round(970_000 * multiplier),
      cached: 0,
      reasoning: 0,
      models: [
        { model: 'claude-sonnet-4-6', total: Math.round(1_280_000 * multiplier), input: Math.round(420_000 * multiplier), output: Math.round(150_000 * multiplier), cacheCreation: Math.round(92_000 * multiplier), cacheRead: Math.round(618_000 * multiplier), cached: 0, reasoning: 0 },
        { model: 'claude-opus-4-1', total: Math.round(540_000 * multiplier), input: Math.round(120_000 * multiplier), output: Math.round(40_000 * multiplier), cacheCreation: Math.round(28_000 * multiplier), cacheRead: Math.round(352_000 * multiplier), cached: 0, reasoning: 0 },
      ],
    },
    {
      id: 'codex' as const,
      name: 'Codex',
      latestAt: ago(1),
      total: Math.round(2_340_000 * multiplier),
      input: Math.round(1_020_000 * multiplier),
      output: Math.round(310_000 * multiplier),
      cacheCreation: 0,
      cacheRead: 0,
      cached: Math.round(790_000 * multiplier),
      reasoning: Math.round(220_000 * multiplier),
      models: [
        { model: 'gpt-5.5-codex-high', total: Math.round(1_750_000 * multiplier), input: Math.round(810_000 * multiplier), output: Math.round(220_000 * multiplier), cacheCreation: 0, cacheRead: 0, cached: Math.round(560_000 * multiplier), reasoning: Math.round(160_000 * multiplier) },
        { model: 'gpt-5.5-extra-high', total: Math.round(590_000 * multiplier), input: Math.round(210_000 * multiplier), output: Math.round(90_000 * multiplier), cacheCreation: 0, cacheRead: 0, cached: Math.round(230_000 * multiplier), reasoning: Math.round(60_000 * multiplier) },
      ],
    },
    {
      id: 'opencode' as const,
      name: 'OpenCode',
      latestAt: ago(18),
      total: Math.round(420_000 * multiplier),
      input: Math.round(170_000 * multiplier),
      output: Math.round(80_000 * multiplier),
      cacheCreation: Math.round(20_000 * multiplier),
      cacheRead: Math.round(120_000 * multiplier),
      cached: 0,
      reasoning: Math.round(30_000 * multiplier),
      models: [
        { model: 'qwen3-coder-30b', total: Math.round(420_000 * multiplier), input: Math.round(170_000 * multiplier), output: Math.round(80_000 * multiplier), cacheCreation: Math.round(20_000 * multiplier), cacheRead: Math.round(120_000 * multiplier), cached: 0, reasoning: Math.round(30_000 * multiplier) },
      ],
    },
  ]
  const modelMap = new Map<string, any>()
  for (const account of accounts) {
    for (const model of account.models) {
      const existing = modelMap.get(model.model) || { model: model.model, total: 0, input: 0, output: 0, cacheCreation: 0, cacheRead: 0, cached: 0, reasoning: 0 }
      for (const field of ['total', 'input', 'output', 'cacheCreation', 'cacheRead', 'cached', 'reasoning'] as const) existing[field] += model[field]
      modelMap.set(model.model, existing)
    }
  }
  return {
    key,
    label,
    startAt: Date.now() - 30 * 24 * 60 * 60_000,
    total: accounts.reduce((sum, account) => sum + account.total, 0),
    accounts,
    models: [...modelMap.values()].sort((a, b) => b.total - a.total),
  }
}

export function mockTokenUsage(): TokenUsageSnapshot {
  const windows = {
    five_hour: mockUsageWindow('five_hour', '5h', 0.04),
    today: mockUsageWindow('today', 'Today', 0.12),
    week: mockUsageWindow('week', 'Week', 0.48),
    month: mockUsageWindow('month', 'Month', 1),
    all: mockUsageWindow('all', 'All', 2.8),
  }
  return {
    generatedAt: Date.now(),
    latestAt: ago(1),
    active: true,
    windows,
    daily: Array.from({ length: 14 }, (_, index) => ({
      key: `d-${index}`,
      ts: Date.now() - (13 - index) * 24 * 60 * 60_000,
      total: 90_000 + index * 18_000,
      accounts: { claude: 30_000 + index * 5_000, codex: 48_000 + index * 11_000, opencode: 12_000 + index * 2_000 },
    })),
    hourly: Array.from({ length: 12 }, (_, index) => ({
      key: `h-${index}`,
      ts: Date.now() - (11 - index) * 60 * 60_000,
      total: 8_000 + index * 2_800,
      accounts: { claude: 2_000 + index * 700, codex: 5_000 + index * 1_700, opencode: 1_000 + index * 400 },
    })),
  }
}

export function mockContextPressure(): ContextPressureSnapshot {
  const registry: ContextPressureSnapshot['registry'] = [
    {
      id: 'ollama:qwen2.5-coder:14b',
      providerId: 'ollama',
      providerName: 'Ollama (local)',
      model: 'qwen2.5-coder:14b',
      kind: 'local',
      nominalContextTokens: 32_768,
      recommendedWorkingTokens: 24_000,
      kvCacheTokens: 6_000,
      source: 'known',
      note: 'Qwen-style local coding model; keep long tool output in summaries before the working set gets hot.',
    },
    {
      id: 'lmstudio:local-model',
      providerId: 'lmstudio',
      providerName: 'LM Studio (local)',
      model: 'local-model',
      kind: 'local',
      nominalContextTokens: 24_000,
      recommendedWorkingTokens: 24_000,
      kvCacheTokens: 6_000,
      source: 'settings',
      note: 'Unknown local model; using the configured local threshold.',
    },
  ]
  const pressureThreads: ContextPressureSnapshot['threads'] = [
    {
      id: 'ctx-local-run',
      title: 'Refactor file explorer index',
      cwd: '/Users/example/Projects/web-dashboard',
      providerId: 'ollama',
      providerName: 'Ollama (local)',
      model: 'qwen2.5-coder:14b',
      kind: 'local',
      estimatedTokens: 21_600,
      contextWindowTokens: 32_768,
      workingLimitTokens: 24_000,
      pressureRatio: 0.9,
      status: 'pressured',
      updatedAt: ago(3),
      guidance: {
        pinnedTokens: 4_200,
        recentTokens: 7_400,
        summarizableTokens: 10_000,
        toolTokens: 7_200,
        recommendedAction: 'summarize_next',
        reason: 'Pressure is near the working limit; summarize the older middle before the next long turn.',
        cacheHint: 'Summarize older turns before the next large tool result so the KV prefix stays stable.',
        pinned: ['System and project instructions', 'Active goal', 'Latest 6 messages'],
        summarizable: ['18 older messages', 'Older tool calls and outputs'],
      },
    },
    {
      id: 'ctx-lmstudio-chat',
      title: 'Explain async/await, simply',
      cwd: '',
      providerId: 'lmstudio',
      providerName: 'LM Studio (local)',
      model: 'local-model',
      kind: 'local',
      estimatedTokens: 3_900,
      contextWindowTokens: 24_000,
      workingLimitTokens: 24_000,
      pressureRatio: 0.163,
      status: 'healthy',
      updatedAt: ago(30),
      guidance: {
        pinnedTokens: 900,
        recentTokens: 3_000,
        summarizableTokens: 0,
        toolTokens: 0,
        recommendedAction: 'continue',
        reason: 'Working set is inside the local context budget.',
        cacheHint: 'Keep appending turns without rewriting the stable prompt prefix.',
        pinned: ['System and project instructions', 'Latest 4 messages'],
        summarizable: [],
      },
    },
  ]
  return {
    generatedAt: Date.now(),
    activeProviderId: mockState.settings.activeProviderId,
    activeModel: mockState.settings.activeModel,
    activeProfile: null,
    registry,
    threads: pressureThreads,
    totals: {
      localThreads: pressureThreads.length,
      estimatedTokens: pressureThreads.reduce((sum, thread) => sum + thread.estimatedTokens, 0),
      pinnedTokens: pressureThreads.reduce((sum, thread) => sum + thread.guidance.pinnedTokens + thread.guidance.recentTokens, 0),
      summarizableTokens: pressureThreads.reduce((sum, thread) => sum + thread.guidance.summarizableTokens, 0),
      maxPressureRatio: Math.max(...pressureThreads.map((thread) => thread.pressureRatio)),
    },
  }
}

mockState.mockConnectorEvents = [
  { id: uid(), connectorId: 'email', pluginName: 'Email', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'high', trustLevel: 'suggest', requiresApproval: false, approvalId: null, title: 'Renewal thread needs a reply', summary: 'A client asked for pricing clarification and a rollout date.', actor: 'Maya Chen', threadKey: 'email:renewal', payloadKeys: ['subject', 'folder'], createdAt: ago(22) },
  { id: uid(), connectorId: 'slack', pluginName: 'Slack', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify', requiresApproval: false, approvalId: null, title: 'Launch room risk callouts', summary: 'Auth edge cases, staging data drift, and release-note ownership came up since standup.', actor: '#launch-room', threadKey: 'slack:launch-room', payloadKeys: ['channel', 'message'], createdAt: ago(48) },
  { id: uid(), connectorId: 'calendar', pluginName: 'Calendar', action: 'event_seen', direction: 'inbound', classification: 'schedule', priority: 'normal', trustLevel: 'notify', requiresApproval: false, approvalId: null, title: 'Launch review today', summary: 'Calendar source has a project review and follow-up block for today.', actor: 'Demo Calendar', threadKey: 'calendar:launch-review', payloadKeys: ['event', 'source'], createdAt: ago(58) },
  { id: uid(), connectorId: 'browser', pluginName: 'Browser', action: 'browser_source_seen', direction: 'inbound', classification: 'document', priority: 'low', trustLevel: 'notify', requiresApproval: false, approvalId: null, title: 'Launch review tab', summary: 'Explicit current-tab source mentions auth edge cases, QA evidence, and release-note ownership.', actor: 'Demo current tab', threadKey: 'browser:browser_tab_demo:demo-browser-1', payloadKeys: ['source', 'url', 'title'], createdAt: ago(60) },
  { id: uid(), connectorId: 'whatsapp', pluginName: 'WhatsApp', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify', requiresApproval: false, approvalId: null, title: 'Travel plan changed', summary: 'Pickup time moved and two people asked whether the calendar invite is accurate.', actor: 'Family', threadKey: 'whatsapp:family', payloadKeys: ['chat', 'message'], createdAt: ago(70) },
  { id: uid(), connectorId: 'docs', pluginName: 'Documents', action: 'document_indexed', direction: 'inbound', classification: 'document', priority: 'high', trustLevel: 'suggest', requiresApproval: false, approvalId: null, title: 'Contract clause differs from email promise', summary: 'The vendor PDF has a 30-day termination window while email says 14 days.', actor: 'Q3 vendor pack', threadKey: 'docs:q3-vendor-pack', payloadKeys: ['path', 'title'], createdAt: ago(120) },
]

mockState.mockEmailMessages = [
  { id: 'demo-email-1', sourceId: 'email_demo', sourceLabel: 'Demo Maildir', path: '/Users/example/Mail/demo/new/1.eml', subject: 'Renewal pricing and rollout date', from: 'Maya Chen <maya@example.com>', to: 'alex@example.com', date: ago(22), unread: true, snippet: 'Could you confirm the rollout date and whether the renewal pricing includes the new reporting tier?', size: 1842, headers: { subject: 'Renewal pricing and rollout date' }, body: 'Could you confirm the rollout date and whether the renewal pricing includes the new reporting tier?\n\nWe need to reply before close of business.' },
  { id: 'demo-email-2', sourceId: 'email_demo', sourceLabel: 'Demo Maildir', path: '/Users/example/Mail/demo/cur/2.eml', subject: 'Invoice pack uploaded', from: 'Finance <finance@example.com>', to: 'alex@example.com', date: ago(95), unread: false, snippet: 'The invoice pack is uploaded. Please check the VAT line before approving payment.', size: 2201, headers: { subject: 'Invoice pack uploaded' }, body: 'The invoice pack is uploaded. Please check the VAT line before approving payment.' },
]

mockState.mockSlackMessages = [
  {
    id: 'demo-slack-1',
    sourceId: 'slack_demo',
    sourceLabel: 'Demo Slack export',
    path: '/Users/example/Slack/demo-export/launch-room/2026-06-20.json',
    channelId: 'CDEMO1',
    channelName: 'launch-room',
    senderId: 'UDEMO1',
    senderName: 'Maya Chen',
    ts: '1781969700.000100',
    threadTs: null,
    date: ago(48),
    text: 'Auth edge cases and staging data drift came up since standup. Can someone own the release-note callout?',
    snippet: 'Auth edge cases and staging data drift came up since standup. Can someone own the release-note callout?',
    raw: { type: 'message', user: 'UDEMO1', ts: '1781969700.000100' },
  },
  {
    id: 'demo-slack-2',
    sourceId: 'slack_demo',
    sourceLabel: 'Demo Slack export',
    path: '/Users/example/Slack/demo-export/launch-room/2026-06-20.json',
    channelId: 'CDEMO1',
    channelName: 'launch-room',
    senderId: 'UDEMO2',
    senderName: 'Alex Rivera',
    ts: '1781970600.000200',
    threadTs: '1781969700.000100',
    date: ago(42),
    text: 'I can own release notes. Please cite the calendar approval event when you summarize this.',
    snippet: 'I can own release notes. Please cite the calendar approval event when you summarize this.',
    raw: { type: 'message', user: 'UDEMO2', ts: '1781970600.000200', thread_ts: '1781969700.000100' },
  },
  {
    id: 'demo-slack-3',
    sourceId: 'slack_demo',
    sourceLabel: 'Demo Slack export',
    path: '/Users/example/Slack/demo-export/ops/2026-06-20.json',
    channelId: 'CDEMO2',
    channelName: 'ops',
    senderId: 'UDEMO3',
    senderName: 'Sam Patel',
    ts: '1781972400.000300',
    threadTs: null,
    date: ago(18),
    text: 'Ops note: vendor review PDF and metrics table both need source-backed evidence before the rollout brief.',
    snippet: 'Ops note: vendor review PDF and metrics table both need source-backed evidence before the rollout brief.',
    raw: { type: 'message', user: 'UDEMO3', ts: '1781972400.000300' },
  },
]

mockState.mockCalendarEvents = [
  {
    id: 'demo-event-1',
    uid: 'launch-review@example.com',
    sourceId: 'cal_demo',
    sourceLabel: 'Demo Calendar',
    path: '/Users/example/Calendars/demo.ics',
    title: 'Launch review',
    startAt: startOfMockDay(10, 30),
    endAt: startOfMockDay(11, 0),
    allDay: false,
    location: 'Project room',
    description: 'Review launch blockers, pending approvals, and owner handoffs.',
    organizer: 'ops@example.com',
    attendees: ['alex@example.com', 'maya@example.com'],
    status: 'CONFIRMED',
    raw: { UID: 'launch-review@example.com', SUMMARY: 'Launch review' },
  },
  {
    id: 'demo-event-2',
    uid: 'follow-up@example.com',
    sourceId: 'cal_demo',
    sourceLabel: 'Demo Calendar',
    path: '/Users/example/Calendars/demo.ics',
    title: 'Follow-up block',
    startAt: startOfMockDay(14, 0),
    endAt: startOfMockDay(15, 0),
    allDay: false,
    location: '',
    description: 'Reserved focus time for the review follow-ups.',
    organizer: 'alex@example.com',
    attendees: [],
    status: 'CONFIRMED',
    raw: { UID: 'follow-up@example.com', SUMMARY: 'Follow-up block' },
  },
]

mockState.mockDocuments = [
  {
    id: 'demo-doc-1',
    sourceId: 'docs_demo',
    sourceLabel: 'Demo Documents',
    path: '/Users/example/Documents/TensorFold/launch-brief.md',
    relativePath: 'launch-brief.md',
    title: 'Launch brief',
    excerpt: 'The launch gate needs reviewer evidence, final sign-off, and a calendar owner before rollout.',
    size: 912,
    modifiedAt: ago(120),
    extension: '.md',
    content: '# Launch brief\n\nThe launch gate needs reviewer evidence, final sign-off, and a calendar owner before rollout.',
  },
  {
    id: 'demo-doc-2',
    sourceId: 'docs_demo',
    sourceLabel: 'Demo Documents',
    path: '/Users/example/Documents/TensorFold/vendor-notes.txt',
    relativePath: 'vendor-notes.txt',
    title: 'Vendor notes',
    excerpt: 'The contract clause has a 30-day termination window while the email promise says 14 days.',
    size: 684,
    modifiedAt: ago(144),
    extension: '.txt',
    content: 'Vendor notes\n\nThe contract clause has a 30-day termination window while the email promise says 14 days.',
  },
  {
    id: 'demo-doc-3',
    sourceId: 'docs_demo',
    sourceLabel: 'Demo Documents',
    path: '/Users/example/Documents/TensorFold/vendor-review.pdf',
    relativePath: 'vendor-review.pdf',
    title: 'Vendor PDF review',
    excerpt: 'Renewal terms require source-backed approval before the calendar update is requested.',
    size: 142_336,
    modifiedAt: ago(180),
    extension: '.pdf',
    content: 'Vendor PDF review\n\nRenewal terms require source-backed approval before the calendar update is requested.',
  },
  {
    id: 'demo-doc-4',
    sourceId: 'docs_demo',
    sourceLabel: 'Demo Documents',
    path: '/Users/example/Documents/TensorFold/metrics.csv',
    relativePath: 'metrics.csv',
    title: 'metrics.csv',
    excerpt: 'Team | Revenue | Cost | Status; Alpha | 1200 | 800 | green; Beta | 950 | 500 | yellow',
    size: 112,
    modifiedAt: ago(210),
    extension: '.csv',
    content: 'Team,Revenue,Cost,Status\nAlpha,1200,800,green\nBeta,950,500,yellow\n"Gamma, East",700,710,red',
  },
  {
    id: 'demo-doc-5',
    sourceId: 'docs_demo',
    sourceLabel: 'Demo Documents',
    path: '/Users/example/Documents/TensorFold/budget.xlsx',
    relativePath: 'budget.xlsx',
    title: 'budget.xlsx',
    excerpt: 'Team | Budget | Actual; Alpha | 1000 | 900; Beta | 800 | 850',
    size: 8_192,
    modifiedAt: ago(240),
    extension: '.xlsx',
    content: 'Team\tBudget\tActual\nAlpha\t1000\t900\nBeta\t800\t850',
  },
]

mockState.mockBrowserItems = [
  {
    id: 'demo-browser-1',
    sourceId: 'browser_tab_demo',
    sourceLabel: 'Demo current tab',
    kind: 'current_tab',
    url: 'https://example.com/launch-review',
    title: 'Launch review tab',
    excerpt: 'The foreground browser tab mentions auth edge cases, QA evidence, and release-note ownership.',
    capturedAt: ago(11),
    content: 'ID: demo-browser-1\nSource ID: browser:browser_tab_demo:demo-browser-1\nSource: Demo current tab\nKind: Current tab\nURL: https://example.com/launch-review\n\nThe foreground browser tab mentions auth edge cases, QA evidence, and release-note ownership.\n\nSources:\n- browser:browser_tab_demo:demo-browser-1 | Launch review tab',
  },
  {
    id: 'demo-browser-2',
    sourceId: 'browser_history_demo',
    sourceLabel: 'Demo history export',
    kind: 'history_export',
    url: 'https://example.com/qa-evidence',
    title: 'QA evidence checklist',
    excerpt: 'History export page about screenshot evidence, console errors, and final verification.',
    capturedAt: ago(35),
    path: '/Users/example/Browser/history-export.json',
    content: 'ID: demo-browser-2\nSource ID: browser:browser_history_demo:demo-browser-2\nSource: Demo history export\nKind: History export\nURL: https://example.com/qa-evidence\n\nHistory export page about screenshot evidence, console errors, and final verification.\n\nSources:\n- browser:browser_history_demo:demo-browser-2 | QA evidence checklist',
  },
]

mockState.mockMissions = [
  {
    id: 'mission_demo',
    title: 'Morning operations sweep',
    objective: 'Check email, collect urgent signals, and prepare reply drafts for approval.',
    stopCondition: 'No urgent unread email remains and reply drafts are ready for approval.',
    repeatMode: 'until_condition',
    contextMode: 'build_on_previous',
    roleProfileId: 'operator',
    fanOutBudget: 'single',
    maxRuns: 15,
    costCapUsd: 25,
    expiresAt: Date.now() + 3 * 24 * 60 * 60_000,
    destructiveApprovalRequired: true,
    status: 'running',
    runBudgetClass: 'mission',
    budget: { executionClass: 'mission', label: 'Mission', maxWallSeconds: 3600, heartbeatIntervalSeconds: 30, stallGraceSeconds: 120, requiresHeartbeat: true, startedAt: ago(14), deadlineAt: ago(14) + 3600_000 },
    heartbeat: { intervalSeconds: 30, stallGraceSeconds: 120, required: true, count: 3, lastBeatAt: ago(1), nextDueAt: Date.now() + 29_000, missedSince: null },
    checkpoints: [{ id: 'chk_demo', sequence: 1, label: 'Inbox scanned', summary: 'Found one renewal email needing a reply.', status: 'running', createdAt: ago(10) }],
    journal: [{ id: 'j_demo', kind: 'started', actor: 'system', message: 'Mission started.', createdAt: ago(14), status: 'running' }],
    scratchMemory: [],
    pendingApprovals: [],
    threadId: null,
    projectId: null,
    cwd: null,
    providerId: 'openai',
    model: 'gpt-4o',
    createdAt: ago(15),
    updatedAt: ago(1),
    startedAt: ago(14),
    finishedAt: null,
  },
]
