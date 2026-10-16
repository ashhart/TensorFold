import type { AgentEvent, ArchiveKind, ArchiveSnapshot, DiffLine, Settings, StartRunPayload, Thread, TimelineItem, Project, Room, RoomEventItem, RoomMemoryEntry, RoomStreamEvent, Ticket, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitStatus, TokenUsageSnapshot, TokenUsageWindow, TokenUsageWindowKey, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, AppUpdateStatus, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSyncResult, EmailThreadResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, Skill, Command, McpServerConfig, McpServerSnapshot, McpSnapshot, McpToolCallResult, McpToolDefinition, ControllerSnapshot, ControllerStatus, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookDeliveryRecord, WebhookEndpoint, WebhookSnapshot, RemoteControlPairingInfo, RemoteControlStatus, RemoteControlWorkSnapshot, SecurityControlSnapshot, SecurityPurgeResult } from '../types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, SlackMessage, SlackListOptions, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from '../types'
import type { Incident, RunExportBundle } from '../types'
import { ago, mockState, uid } from './state'

export function mockAssistantSnapshot(): AssistantControlSnapshot {
  const tool = (name: string, description: string, approvalRequired = false) => ({ name, description, approvalRequired, parameters: { type: 'object', properties: {} } })
  const emailTools = [tool('email_list', 'List emails with optional filters.'), tool('email_read', 'Read a specific email.'), tool('email_thread', 'Read the local conversation around an email.'), tool('email_compose', 'Compose, draft, reply to, or send email.', true), tool('email_summary', 'Summarise email activity.')]
  const calendarTools = [tool('calendar_list', 'List upcoming calendar events.'), tool('calendar_read', 'Read a specific calendar event.'), tool('calendar_summary', 'Summarise a calendar day.'), tool('calendar_create', 'Create events.', true), tool('calendar_update', 'Update events.', true)]
  const messageTools = [tool('message_list', 'List recent messages.'), tool('message_read', 'Read a message thread.'), tool('message_compose', 'Draft or send a message.', true), tool('message_summary', 'Summarise unread messages.')]
  const slackTools = [tool('message_list', 'List recent Slack export messages.'), tool('message_read', 'Read a Slack export message.'), tool('message_summary', 'Summarise Slack export messages.')]
  const docTools = [tool('document_list', 'List indexed documents and PDFs.'), tool('document_read', 'Read a document or PDF excerpt.'), tool('document_summary', 'Summarise document/PDF sets.'), tool('table_list', 'List indexed CSV/TSV/XLSX tables.'), tool('table_read', 'Read a table preview.'), tool('table_summary', 'Summarise tables with source ids.')]
  const browserTools = [tool('browser_list', 'List explicit browser sources.'), tool('browser_read', 'Read an explicit browser source item.'), tool('browser_summary', 'Summarise explicit browser current-tab/history sources.')]
  const emailSources = mockState.settings.email.sources.filter((s) => s.enabled)
  const emailErrors = emailSources.filter((s) => s.lastError)
  const emailStatus = emailSources.length ? 'connected' : 'pending'
  const emailPermissionStatus = emailSources.length === 0 ? 'not_configured' : emailErrors.length === 0 ? 'granted' : emailErrors.length === emailSources.length ? 'denied' : 'degraded'
  const emailPermissionDetail = emailSources.length === 0
    ? 'No enabled email sources.'
    : emailErrors.length === 0
      ? `${emailSources.length} enabled email source${emailSources.length === 1 ? '' : 's'} available.`
      : `${emailErrors.length}/${emailSources.length} email source${emailSources.length === 1 ? '' : 's'} failing: ${emailErrors[0].lastError ?? 'unknown error'}`
  const slackSources = mockState.settings.slack.sources.filter((s) => s.enabled)
  const slackErrors = slackSources.filter((s) => s.lastError)
  const slackStatus = slackSources.length ? 'connected' : 'pending'
  const slackPermissionStatus = slackSources.length === 0 ? 'not_configured' : slackErrors.length === 0 ? 'granted' : slackErrors.length === slackSources.length ? 'denied' : 'degraded'
  const slackPermissionDetail = slackSources.length === 0
    ? 'No enabled Slack export sources.'
    : slackErrors.length === 0
      ? `${slackSources.length} enabled Slack export source${slackSources.length === 1 ? '' : 's'} available.`
      : `${slackErrors.length}/${slackSources.length} Slack export source${slackSources.length === 1 ? '' : 's'} failing: ${slackErrors[0].lastError ?? 'unknown error'}`
  const calendarSources = mockState.settings.calendar.sources.filter((s) => s.enabled)
  const calendarErrors = calendarSources.filter((s) => s.lastError)
  const calendarStatus = calendarSources.length ? 'connected' : 'pending'
  const calendarPermissionStatus = calendarSources.length === 0 ? 'not_configured' : calendarErrors.length === 0 ? 'granted' : calendarErrors.length === calendarSources.length ? 'denied' : 'degraded'
  const calendarPermissionDetail = calendarSources.length === 0
    ? 'No enabled calendar sources.'
    : calendarErrors.length === 0
      ? `${calendarSources.length} enabled calendar source${calendarSources.length === 1 ? '' : 's'} available.`
      : `${calendarErrors.length}/${calendarSources.length} calendar source${calendarSources.length === 1 ? '' : 's'} failing: ${calendarErrors[0].lastError ?? 'unknown error'}`
  const documentSources = mockState.settings.documents.sources.filter((s) => s.enabled)
  const documentErrors = documentSources.filter((s) => s.lastError)
  const documentStatus = documentSources.length ? 'connected' : 'pending'
  const documentPermissionStatus = documentSources.length === 0 ? 'not_configured' : documentErrors.length === 0 ? 'granted' : documentErrors.length === documentSources.length ? 'denied' : 'degraded'
  const documentPermissionDetail = documentSources.length === 0
    ? 'No enabled document sources.'
    : documentErrors.length === 0
      ? `${documentSources.length} enabled document source${documentSources.length === 1 ? '' : 's'} available.`
      : `${documentErrors.length}/${documentSources.length} document source${documentSources.length === 1 ? '' : 's'} failing: ${documentErrors[0].lastError ?? 'unknown error'}`
  const browserSources = mockState.settings.browser.sources.filter((s) => s.enabled)
  const browserErrors = browserSources.filter((s) => s.lastError)
  const browserStatus = browserSources.length ? 'connected' : 'pending'
  const browserPermissionStatus = browserSources.length === 0 ? 'not_configured' : browserErrors.length === 0 ? 'granted' : browserErrors.length === browserSources.length ? 'denied' : 'degraded'
  const browserPermissionDetail = browserSources.length === 0
    ? 'No enabled browser sources.'
    : browserErrors.length === 0
      ? `${browserSources.length} enabled browser source${browserSources.length === 1 ? '' : 's'} available.`
      : `${browserErrors.length}/${browserSources.length} browser source${browserSources.length === 1 ? '' : 's'} failing: ${browserErrors[0].lastError ?? 'unknown error'}`
  const controllerTools = [tool('controller_capture', 'Attach current window context.'), tool('controller_action', 'Approval-gated local Mac app action.', true)]
  const controllerStatus = mockState.settings.controller.enabled ? (mockState.settings.controller.lastError ? 'error' : 'connected') : 'pending'
  const allowedCount = (raw: string) => String(raw || '').split(/[\s,;]+/).filter(Boolean).length
  const telegramAllowed = allowedCount(mockState.settings.botControl.telegram.allowedChatIds)
  const telegramUsers = allowedCount(mockState.settings.botControl.telegram.allowedUserIds)
  const telegramOptedIn = mockState.settings.botControl.telegram.enabled && Boolean(mockState.settings.botControl.telegram.botToken.trim()) && telegramAllowed > 0 && telegramUsers > 0
  const telegramPermissionDetail = telegramOptedIn
    ? mockState.settings.botControl.telegram.lastError
      ? `Telegram bridge is allowlisted for ${telegramAllowed} chat${telegramAllowed === 1 ? '' : 's'}, but reports: ${mockState.settings.botControl.telegram.lastError}`
      : `Telegram bridge is opted in for ${telegramAllowed} allowed Telegram chat${telegramAllowed === 1 ? '' : 's'}. Outbound replies require approval.`
    : 'Telegram requires explicit opt-in: add enabled bridge, bot token, allowed Telegram chat IDs, and allowed Telegram user IDs.'
  const whatsappAllowed = allowedCount(mockState.settings.botControl.whatsapp.allowedChatIds)
  const whatsappOptedIn = mockState.settings.botControl.whatsapp.enabled
    && Boolean(mockState.settings.botControl.whatsapp.accessToken.trim())
    && Boolean(mockState.settings.botControl.whatsapp.phoneNumberId.trim())
    && Boolean(mockState.settings.botControl.whatsapp.verifyToken.trim())
    && Boolean(mockState.settings.botControl.whatsapp.appSecret.trim())
    && whatsappAllowed > 0
  const whatsappPermissionDetail = whatsappOptedIn
    ? mockState.settings.botControl.whatsapp.lastError
      ? `WhatsApp bridge is allowlisted for ${whatsappAllowed} sender${whatsappAllowed === 1 ? '' : 's'}, but reports: ${mockState.settings.botControl.whatsapp.lastError}`
      : `WhatsApp bridge is opted in for ${whatsappAllowed} allowed WhatsApp sender${whatsappAllowed === 1 ? '' : 's'}. Outbound replies require approval.`
    : 'WhatsApp requires explicit opt-in: add enabled bridge, access token, phone number ID, verify token, Meta app secret, and allowed sender IDs.'
  const connectors: AssistantControlSnapshot['connectors'] = [
    { id: 'email', name: 'Email', kind: 'communication', status: emailStatus, description: `${emailSources.length || 0} local email source${emailSources.length === 1 ? '' : 's'} connected for inbox triage, summaries, drafts, replies, and approval-gated send requests.`, transport: 'email', tools: emailTools, unread: mockState.mockEmailMessages.filter((m) => m.unread).length, pendingApprovals: mockState.mockConnectorEvents.filter((e) => e.connectorId === 'email' && e.requiresApproval && (e.approvalStatus ?? 'pending') === 'pending').length, lastSyncAt: mockState.settings.email.lastSyncAt, lastEventAt: mockState.settings.email.lastSyncAt ?? mockState.mockConnectorEvents[0]?.createdAt ?? null, permissionStatus: emailPermissionStatus, permissionDetail: emailPermissionDetail, error: emailErrors[0]?.lastError ?? undefined },
    { id: 'slack', name: 'Slack', kind: 'communication', status: slackStatus, description: `${slackSources.length || 0} local Slack export source${slackSources.length === 1 ? '' : 's'} connected for source-backed message reads and summaries.`, transport: 'slack', tools: slackTools, unread: filterMockSlack({ limit: mockState.settings.slack.maxMessages }).length, pendingApprovals: 0, lastSyncAt: mockState.settings.slack.lastSyncAt, lastEventAt: mockState.settings.slack.lastSyncAt ?? mockState.mockConnectorEvents.find((event) => event.connectorId === 'slack')?.createdAt ?? null, permissionStatus: slackPermissionStatus, permissionDetail: slackPermissionDetail, error: slackErrors[0]?.lastError ?? undefined },
    { id: 'telegram', name: 'Telegram', kind: 'communication', status: telegramOptedIn ? mockState.settings.botControl.telegram.lastError ? 'error' : 'connected' : 'pending', description: 'Opt-in Telegram bot command and delivery channel with allowlisted chats.', transport: 'telegram', tools: telegramOptedIn ? messageTools : [], unread: 0, pendingApprovals: 0, lastSyncAt: mockState.settings.botControl.telegram.lastPollAt, lastEventAt: mockState.settings.botControl.telegram.lastPollAt, permissionStatus: telegramOptedIn ? mockState.settings.botControl.telegram.lastError ? 'degraded' : 'granted' : 'not_configured', permissionDetail: telegramPermissionDetail, error: mockState.settings.botControl.telegram.lastError ?? undefined },
    { id: 'whatsapp', name: 'WhatsApp', kind: 'communication', status: whatsappOptedIn ? mockState.settings.botControl.whatsapp.lastError ? 'error' : 'connected' : 'pending', description: 'Opt-in personal WhatsApp bridge with allowlisted senders and approval-gated replies.', transport: 'whatsapp', tools: whatsappOptedIn ? messageTools : [], unread: whatsappOptedIn ? 5 : 0, pendingApprovals: 0, lastSyncAt: mockState.settings.botControl.whatsapp.lastEventAt, lastEventAt: mockState.settings.botControl.whatsapp.lastEventAt ?? mockState.mockConnectorEvents.find((event) => event.connectorId === 'whatsapp')?.createdAt ?? null, permissionStatus: whatsappOptedIn ? mockState.settings.botControl.whatsapp.lastError ? 'degraded' : 'granted' : 'not_configured', permissionDetail: whatsappPermissionDetail, error: mockState.settings.botControl.whatsapp.lastError ?? undefined },
    { id: 'imessage', name: 'iMessage', kind: 'communication', status: 'pending', description: 'Opt-in local iMessage bridge contract; unavailable until an explicit bridge and contact allowlist exist.', transport: null, tools: [], unread: 0, pendingApprovals: 0, lastSyncAt: null, lastEventAt: null, permissionStatus: 'not_configured', permissionDetail: 'iMessage requires an explicit local bridge configuration before messages can enter or leave TensorFold.' },
    { id: 'controller', name: 'Controller', kind: 'system', status: controllerStatus, description: mockState.settings.controller.enabled ? 'Current-window context capture and approval-gated app actions are enabled.' : 'Enable Controller to attach current window context or request scoped app actions.', transport: null, tools: controllerTools, unread: 0, pendingApprovals: 0, lastSyncAt: mockState.settings.controller.lastCapturedAt, lastEventAt: mockState.settings.controller.lastCapturedAt, permissionStatus: mockState.settings.controller.enabled ? (mockState.settings.controller.lastError ? 'denied' : 'granted') : 'not_configured', permissionDetail: mockState.settings.controller.enabled ? (mockState.settings.controller.lastError ?? 'Controller capture and scoped actions are enabled.') : 'Controller is disabled.', error: mockState.settings.controller.lastError ?? undefined },
    { id: 'docs', name: 'Documents', kind: 'document', status: documentStatus, description: `${documentSources.length || 0} local document/PDF/table source${documentSources.length === 1 ? '' : 's'} connected for source-backed reads and citation-aware summaries.`, transport: null, tools: docTools, unread: filterMockDocuments({ limit: mockState.settings.documents.maxDocuments }).length, pendingApprovals: 0, lastSyncAt: mockState.settings.documents.lastSyncAt, lastEventAt: mockState.settings.documents.lastSyncAt ?? mockState.mockConnectorEvents.find((event) => event.connectorId === 'docs')?.createdAt ?? null, permissionStatus: documentPermissionStatus, permissionDetail: documentPermissionDetail, error: documentErrors[0]?.lastError ?? undefined },
    { id: 'browser', name: 'Browser', kind: 'document', status: browserStatus, description: `${browserSources.length || 0} explicit browser source${browserSources.length === 1 ? '' : 's'} connected for current-tab/history reads and summaries.`, transport: null, tools: browserTools, unread: filterMockBrowser({ limit: mockState.settings.browser.maxItems }).length, pendingApprovals: 0, lastSyncAt: mockState.settings.browser.lastSyncAt, lastEventAt: mockState.settings.browser.lastSyncAt ?? mockState.mockConnectorEvents.find((event) => event.connectorId === 'browser')?.createdAt ?? null, permissionStatus: browserPermissionStatus, permissionDetail: browserPermissionDetail, error: browserErrors[0]?.lastError ?? undefined },
    { id: 'calendar', name: 'Calendar', kind: 'calendar', status: calendarStatus, description: `${calendarSources.length || 0} local .ics source${calendarSources.length === 1 ? '' : 's'} connected for event reads and day summaries.`, transport: 'calendar', tools: calendarTools, unread: 0, pendingApprovals: 0, lastSyncAt: mockState.settings.calendar.lastSyncAt, lastEventAt: mockState.settings.calendar.lastSyncAt ?? mockState.mockConnectorEvents.find((event) => event.connectorId === 'calendar')?.createdAt ?? null, permissionStatus: calendarPermissionStatus, permissionDetail: calendarPermissionDetail, error: calendarErrors[0]?.lastError ?? undefined },
  ]
  const events = [...mockState.mockConnectorEvents].sort((a, b) => b.createdAt - a.createdAt)
  return {
    generatedAt: Date.now(),
    connectors,
    events,
    attention: mockConnectorAttention(events),
    briefing: mockConnectorBriefing(events),
    memoryCandidates: mockConnectorMemoryCandidateDrafts(events),
    missionTriggers: mockConnectorMissionTriggers(events),
    gaps: connectors.filter((c) => c.status === 'pending').map((c) => ({ id: `connector:${c.id}:not_connected`, severity: 'warning', title: `${c.name} is not connected`, detail: 'The connector contract exists, but live ingress and transport are not configured yet.', subject: c.id })),
    transports: connectors.filter((c) => c.transport).map((c) => c.transport!),
    tools: connectors.flatMap((c) => c.tools),
    budgets: [
      { executionClass: 'interactive', label: 'Interactive', maxWallSeconds: 30, heartbeatIntervalSeconds: 0, stallGraceSeconds: 0, requiresHeartbeat: false },
      { executionClass: 'background', label: 'Background', maxWallSeconds: 300, heartbeatIntervalSeconds: 60, stallGraceSeconds: 120, requiresHeartbeat: true },
      { executionClass: 'mission', label: 'Mission', maxWallSeconds: 3600, heartbeatIntervalSeconds: 30, stallGraceSeconds: 120, requiresHeartbeat: true },
      { executionClass: 'overnight', label: 'Overnight', maxWallSeconds: 28800, heartbeatIntervalSeconds: 60, stallGraceSeconds: 300, requiresHeartbeat: true },
    ],
    pendingApprovals: mockState.mockConnectorEvents.filter((event) => event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending').length,
  }
}

export function mockConnectorAttention(events: ConnectorEvent[]): AssistantControlSnapshot['attention'] {
  return events.slice(0, 24).map((event) => ({
    id: `attention:${event.id}`,
    eventId: event.id,
    connectorId: event.connectorId,
    threadKey: event.threadKey,
    connectorAction: event.action,
    title: event.title,
    summary: event.summary,
    actor: event.actor || event.pluginName,
    action: mockConnectorAction(event),
    priority: event.priority,
    requiresApproval: event.requiresApproval,
    sourceIds: mockConnectorSourceIds(event),
    sources: event.sources ?? [],
    createdAt: event.createdAt,
  }))
}

export function mockConnectorBriefing(events: ConnectorEvent[]): AssistantControlSnapshot['briefing'] {
  const items = events.slice(0, 8)
  const sourceIds = [...new Set(items.flatMap(mockConnectorSourceIds))]
  return {
    generatedAt: Date.now(),
    summary: `${items.length} connector signal${items.length === 1 ? '' : 's'}${items.some((event) => event.requiresApproval) ? ', approval review needed' : ''}.`,
    lines: items.map((event) => `${mockConnectorSourceIds(event).map((id) => `[${id}]`).join(' ')} ${event.title}: ${event.summary}`),
    sourceIds,
    sources: items.flatMap((event) => event.sources ?? []).filter((source, index, all) => all.findIndex((candidate) => candidate.id === source.id) === index),
    eventCount: items.length,
    pendingApprovalCount: items.filter((event) => event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending').length,
    latestEventAt: items[0]?.createdAt ?? null,
  }
}

export function mockConnectorMemoryCandidateDrafts(events: ConnectorEvent[]): AssistantControlSnapshot['memoryCandidates'] {
  const project = mockState.settings.defaultCwd
    ? mockState.projects.find((item) => !item.archivedAt && item.cwd === mockState.settings.defaultCwd) ?? mockState.projects.find((item) => !item.archivedAt)
    : mockState.projects.find((item) => !item.archivedAt)
  if (!project) return []
  const existing = new Set(mockState.mockProjectMemoryCandidates.map((candidate) => `${candidate.projectId}:${candidate.sourceRef ?? ''}`))
  return events
    .filter((event) => event.requiresApproval || event.priority === 'high' || event.priority === 'critical' || event.classification === 'document' || event.classification === 'schedule')
    .map((event) => {
      const sourceRef = mockConnectorSourceIds(event)[0]
      const kind: AssistantControlSnapshot['memoryCandidates'][number]['kind'] = event.classification === 'schedule' ? 'decision' : event.classification === 'document' ? 'summary' : 'gotcha'
      return {
        id: `connector-memory:${project.id}:${event.id}`,
        eventId: event.id,
        connectorId: event.connectorId,
        projectId: project.id,
        kind,
        text: `${event.pluginName || event.connectorId} ${event.classification}: ${event.title}. ${event.summary}`,
        sourceType: 'connector' as const,
        sourceRef,
        confidence: 'medium' as const,
        privacyScope: 'project' as const,
        reason: `Source-backed connector event from ${event.pluginName || event.connectorId}.`,
        createdAt: event.createdAt,
      }
    })
    .filter((draft) => {
      const key = `${draft.projectId}:${draft.sourceRef}`
      if (existing.has(key)) return false
      existing.add(key)
      return true
    })
    .slice(0, 8)
}

export function mockConnectorMissionTriggers(events: ConnectorEvent[]): AssistantControlSnapshot['missionTriggers'] {
  const active = mockState.mockMissions.filter((mission) => ['queued', 'running', 'waiting_approval', 'paused', 'stalled'].includes(mission.status))
  const triggers: AssistantControlSnapshot['missionTriggers'] = []
  for (const event of events) {
    if (!(event.requiresApproval || event.priority === 'high' || event.priority === 'critical' || event.classification === 'document' || event.classification === 'schedule')) continue
    for (const mission of active) {
      const root = mission.cwd?.trim().replace(/\/+$/, '')
      const pathMatch = Boolean(root && (event.sources ?? []).some((source) => {
        const candidate = source.path?.trim().replace(/\/+$/, '')
        return candidate === root || candidate?.startsWith(`${root}/`)
      }))
      // Mirror the host security boundary: connector free text is untrusted and cannot
      // trigger a mission by keyword. Only a structural workspace-path match or explicit
      // connector-feed opt-in is eligible.
      if (!pathMatch && mission.metadata?.connectorEventFeed !== true) continue
      triggers.push({
        id: `mission-trigger:${mission.id}:${event.id}`,
        missionId: mission.id,
        missionTitle: mission.title,
        eventId: event.id,
        connectorId: event.connectorId,
        title: event.title,
        summary: `${event.title}: ${event.summary}`,
        action: event.requiresApproval ? 'Review' : 'Attach',
        sourceIds: mockConnectorSourceIds(event),
        sources: event.sources ?? [],
        createdAt: event.createdAt,
      })
      if (triggers.length >= 12) return triggers
    }
  }
  return triggers
}

export function mockConnectorAction(event: ConnectorEvent): AssistantControlSnapshot['attention'][number]['action'] {
  if (event.requiresApproval) return 'Review'
  if (event.classification === 'document') return 'Compare'
  if (event.classification === 'schedule') return 'Plan'
  return 'Summarize'
}

export function mockConnectorSourceIds(event: ConnectorEvent): string[] {
  const ids = [...new Set((event.sources ?? []).map((source) => source.id).filter(Boolean))]
  return ids.length ? ids : [`connector:${event.id}`]
}

export function mockModelRouterSnapshot(): ModelRouterSnapshot {
  const provider = mockState.settings.providers.find((p) => p.id === 'codex-cli')!
  const account = provider.accounts?.find((a) => a.id === 'codex-personal')!
  return {
    generatedAt: Date.now(),
    activeCooldowns: [{
      providerId: provider.id,
      providerName: provider.name,
      providerKind: provider.kind ?? 'cloud',
      providerTransport: provider.transport ?? null,
      accountId: account.id,
      accountName: account.name,
      accountKind: account.kind,
      model: mockState.settings.activeModel ?? provider.models[0] ?? 'gpt-5.4',
      kind: 'rate_limit',
      observedAt: ago(6),
      resetAt: Date.now() + 54 * 60_000,
      source: 'mock-429',
    }],
    recentlyReset: [],
    history: [{
      id: 'mock-route-history-1',
      scope: 'thread',
      subjectId: 'mock-thread',
      subjectLabel: 'Daily build review',
      providerId: provider.id,
      providerName: provider.name,
      fromAccountId: account.id,
      fromAccountName: account.name,
      toAccountId: 'codex-work',
      toAccountName: 'Work Codex',
      model: mockState.settings.activeModel ?? provider.models[0] ?? 'gpt-5.4',
      reason: 'Selected Work Codex because Personal Codex is cooling down for this model.',
      exhaustionKind: 'rate_limit',
      resetAt: Date.now() + 54 * 60_000,
      createdAt: ago(5),
    }],
  }
}

export function publicEmail(message: EmailReadResult): EmailMessage {
  const { headers: _headers, body: _body, ...rest } = message
  return rest
}

export function filterMockEmail(opts: EmailListOptions = {}): EmailMessage[] {
  let messages = mockState.mockEmailMessages.map(publicEmail).sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
  if (opts.unreadOnly) messages = messages.filter((m) => m.unread)
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    messages = messages.filter((m) => [m.subject, m.from, m.to, m.snippet].some((v) => v.toLowerCase().includes(q)))
  }
  return messages.slice(0, opts.limit ?? 50)
}

export function mockEmailSourceRef(message: EmailReadResult) {
  return {
    id: `email:${message.sourceId}:${message.id}`,
    connectorId: 'email' as const,
    sourceId: message.sourceId,
    itemId: message.id,
    title: message.subject || '(no subject)',
    path: message.path,
    excerpt: message.snippet,
    createdAt: message.date,
  }
}

export function mockEmailThread(id: string): EmailThreadResult {
  const root = mockState.mockEmailMessages.find((m) => m.id === id.replace(/^email:/, '')) ?? mockState.mockEmailMessages[0]
  return {
    root,
    messages: mockState.mockEmailMessages.filter((m) => m.subject === root.subject || m.id === root.id).slice(0, 5),
    summary: `Demo thread for ${root.subject || '(no subject)'}.`,
  }
}

export function publicSlackMessage(message: SlackReadResult): SlackMessage {
  const { raw: _raw, ...rest } = message
  return rest
}

export function mockSlackSourceId(message: SlackMessage | SlackReadResult): string {
  return `slack:${message.sourceId}:${message.id}`
}

export function mockSlackSourceRef(message: SlackMessage | SlackReadResult) {
  return {
    id: mockSlackSourceId(message),
    connectorId: 'slack' as const,
    sourceId: message.sourceId,
    itemId: message.id,
    title: `#${message.channelName} ${message.senderName || 'Unknown'}`,
    path: message.path,
    excerpt: message.snippet,
    createdAt: message.date,
  }
}

export function filterMockSlack(opts: SlackListOptions = {}): SlackMessage[] {
  const enabledSourceIds = new Set(mockState.settings.slack.sources.filter((source) => source.enabled).map((source) => source.id))
  let messages = mockState.mockSlackMessages
    .filter((message) => enabledSourceIds.has(message.sourceId))
    .map(publicSlackMessage)
    .sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
  if (opts.channel?.trim()) {
    const channel = opts.channel.trim().replace(/^#/, '').toLowerCase()
    messages = messages.filter((message) => message.channelName.toLowerCase() === channel || message.channelId.toLowerCase() === channel)
  }
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    messages = messages.filter((message) => [message.channelName, message.senderName, message.text, message.snippet].some((value) => value.toLowerCase().includes(q)))
  }
  return messages.slice(0, Math.max(1, Math.min(opts.limit ?? 50, mockState.settings.slack.maxMessages || 300)))
}

export function mockSlackSummary(opts: SlackListOptions = {}): SlackSummaryResult {
  const messages = filterMockSlack({ ...opts, limit: opts.limit ?? 20 })
  const sources = messages.slice(0, 12).map(mockSlackSourceRef)
  if (!messages.length) return { summary: 'No Slack export messages found for the current filters.\n\nSources: none.', sources: [] }
  const byChannel = messages.reduce<Record<string, number>>((acc, message) => {
    acc[message.channelName] = (acc[message.channelName] ?? 0) + 1
    return acc
  }, {})
  const channelText = Object.entries(byChannel).map(([channel, count]) => `#${channel}: ${count}`).join(', ')
  return {
    summary: `${messages.length} demo Slack export message${messages.length === 1 ? '' : 's'} matched (${channelText}).\n\n${messages.slice(0, 5).map((message) => `- [${mockSlackSourceId(message)}] #${message.channelName} ${message.senderName || 'Unknown'}: ${message.snippet}`).join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  }
}

export function startOfMockDay(hour = 0, minute = 0): number {
  const date = new Date()
  date.setHours(hour, minute, 0, 0)
  return date.getTime()
}

export function mockCalendarSourceId(event: CalendarEvent): string {
  return `calendar:${event.sourceId}:${event.id}`
}

export function mockCalendarSourceRef(event: CalendarEvent) {
  return {
    id: mockCalendarSourceId(event),
    connectorId: 'calendar' as const,
    sourceId: event.sourceId,
    itemId: event.id,
    title: event.title || '(untitled)',
    path: event.path,
    excerpt: [event.location, event.description].filter(Boolean).join(' | ').slice(0, 240),
    createdAt: event.startAt,
  }
}

export function mockCalendarWindow(opts: CalendarListOptions = {}): { from: number | null; to: number | null } {
  if (opts.day) {
    const start = new Date(`${opts.day}T00:00:00`)
    const from = new Date(start.getFullYear(), start.getMonth(), start.getDate()).getTime()
    return { from, to: from + 24 * 60 * 60_000 }
  }
  return { from: mockCalendarBoundary(opts.from), to: mockCalendarBoundary(opts.to) }
}

export function mockCalendarBoundary(value: CalendarListOptions['from']): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return new Date(`${value}T00:00:00`).getTime()
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function mockCalendarDayLabel(value: string | number | null | undefined = Date.now()): string {
  const date = typeof value === 'string' ? new Date(value.length === 10 ? `${value}T00:00:00` : value) : new Date(value ?? Date.now())
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

export function filterMockCalendar(opts: CalendarListOptions = {}): CalendarEvent[] {
  const window = mockCalendarWindow(opts)
  const enabledSourceIds = new Set(mockState.settings.calendar.sources.filter((source) => source.enabled).map((source) => source.id))
  let events: CalendarEvent[] = mockState.mockCalendarEvents.filter((event) => enabledSourceIds.has(event.sourceId)).sort((a, b) => (a.startAt ?? 0) - (b.startAt ?? 0))
  if (window.from != null) events = events.filter((event) => (event.endAt ?? event.startAt ?? 0) >= window.from!)
  if (window.to != null) events = events.filter((event) => (event.startAt ?? 0) < window.to!)
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    events = events.filter((event) => [event.title, event.location, event.description, event.organizer, event.attendees.join(' ')].some((value) => value.toLowerCase().includes(q)))
  }
  return events.slice(0, Math.max(1, Math.min(opts.limit ?? 50, mockState.settings.calendar.maxEvents || 500)))
}

export function mockCalendarSummary(opts: CalendarListOptions = {}): CalendarSummaryResult {
  const day = opts.day ?? mockCalendarDayLabel(opts.from)
  const events = filterMockCalendar({ ...opts, day, limit: opts.limit ?? 40 })
  const sources = events.map(mockCalendarSourceRef)
  if (!events.length) return { summary: `No calendar events found for ${day}.\n\nSources: none.`, sources: [] }
  return {
    summary: `${events.length} demo event${events.length === 1 ? '' : 's'} on ${day}.\n\n${events.map((event) => `- [${mockCalendarSourceId(event)}] ${event.title}${event.location ? ` @ ${event.location}` : ''}`).join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  }
}

export function publicDocument(doc: DocumentReadResult): DocumentItem {
  const { content: _content, ...rest } = doc
  return rest
}

export function mockDocumentSourceId(doc: DocumentItem | DocumentReadResult): string {
  return `docs:${doc.sourceId}:${doc.id}`
}

export function mockDocumentSourceRef(doc: DocumentItem | DocumentReadResult) {
  return {
    id: mockDocumentSourceId(doc),
    connectorId: 'docs' as const,
    sourceId: doc.sourceId,
    itemId: doc.id,
    title: doc.title || doc.relativePath,
    path: doc.path,
    excerpt: doc.excerpt,
    createdAt: doc.modifiedAt,
  }
}

export function filterMockDocuments(opts: DocumentListOptions = {}): DocumentItem[] {
  const enabledSourceIds = new Set(mockState.settings.documents.sources.filter((source) => source.enabled).map((source) => source.id))
  let docs: DocumentItem[] = mockState.mockDocuments.filter((doc) => enabledSourceIds.has(doc.sourceId)).map(publicDocument).sort((a, b) => (b.modifiedAt ?? 0) - (a.modifiedAt ?? 0))
  if (opts.path?.trim()) {
    const needle = opts.path.trim().toLowerCase()
    docs = docs.filter((doc) => doc.relativePath.toLowerCase().includes(needle) || doc.path.toLowerCase().includes(needle))
  }
  if (opts.extension?.trim()) {
    const ext = opts.extension.trim().startsWith('.') ? opts.extension.trim().toLowerCase() : `.${opts.extension.trim().toLowerCase()}`
    docs = docs.filter((doc) => doc.extension === ext)
  }
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    docs = docs.filter((doc) => [doc.title, doc.relativePath, doc.excerpt].some((value) => value.toLowerCase().includes(q)))
  }
  return docs.slice(0, Math.max(1, Math.min(opts.limit ?? 50, mockState.settings.documents.maxDocuments || 500)))
}

export function mockDocumentSummary(opts: DocumentListOptions = {}): DocumentSummaryResult {
  const docs = filterMockDocuments({ ...opts, limit: opts.limit ?? 12 })
  const sources = docs.map(mockDocumentSourceRef)
  if (!docs.length) return { summary: 'No documents found for the current filters.\n\nSources: none.', sources: [] }
  return {
    summary: `${docs.length} demo document${docs.length === 1 ? '' : 's'} matched.\n\n${docs.map((doc) => `- [${mockDocumentSourceId(doc)}] ${mockDocumentSummaryLabel(doc)}: ${doc.excerpt}`).join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  }
}

export function mockDocumentSummaryLabel(doc: DocumentItem): string {
  if (doc.title && doc.title !== doc.relativePath) return `${doc.title} (${doc.relativePath})`
  return doc.title || doc.relativePath
}

export function publicBrowserItem(item: BrowserReadResult): BrowserItem {
  const { content: _content, ...rest } = item
  return rest
}

export function mockBrowserSourceId(item: BrowserItem | BrowserReadResult): string {
  return `browser:${item.sourceId}:${item.id}`
}

export function mockBrowserSourceRef(item: BrowserItem | BrowserReadResult) {
  return {
    id: mockBrowserSourceId(item),
    connectorId: 'browser' as const,
    sourceId: item.sourceId,
    itemId: item.id,
    title: item.title || item.url,
    path: item.url,
    excerpt: item.excerpt,
    createdAt: item.capturedAt,
  }
}

export function filterMockBrowser(opts: BrowserListOptions = {}): BrowserItem[] {
  const enabledSourceIds = new Set(mockState.settings.browser.sources.filter((source) => source.enabled).map((source) => source.id))
  let items = mockState.mockBrowserItems
    .filter((item) => enabledSourceIds.has(item.sourceId))
    .map(publicBrowserItem)
    .sort((a, b) => (b.capturedAt ?? 0) - (a.capturedAt ?? 0))
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    items = items.filter((item) => [item.title, item.url, item.excerpt, item.sourceLabel].some((value) => value.toLowerCase().includes(q)))
  }
  return items.slice(0, Math.max(1, Math.min(opts.limit ?? 50, mockState.settings.browser.maxItems || 300)))
}

export function mockBrowserSummary(opts: BrowserListOptions = {}): BrowserSummaryResult {
  const items = filterMockBrowser({ ...opts, limit: opts.limit ?? 12 })
  const sources = items.map(mockBrowserSourceRef)
  if (!items.length) return { summary: 'No browser sources found for the current filters.\n\nSources: none.', sources: [] }
  return {
    summary: `${items.length} demo browser source${items.length === 1 ? '' : 's'} matched.\n\n${items.map((item) => `- [${mockBrowserSourceId(item)}] ${item.title || item.url}: ${item.excerpt}`).join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  }
}

export function mockBrowserCurrentTabSource(input: BrowserSnapshotInput): { source: BrowserSourceConfig; item: BrowserReadResult } {
  const url = new URL(String(input.url || '').trim())
  url.username = ''
  url.password = ''
  const now = Date.now()
  const source: BrowserSourceConfig = {
    id: uid(),
    label: input.title?.trim() || url.hostname,
    kind: 'current_tab',
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
    url: url.href,
    title: input.title?.trim() || url.hostname,
    excerpt: input.excerpt?.replace(/\s+/g, ' ').trim().slice(0, 360) || '',
    capturedAt: input.capturedAt ?? now,
  }
  const item: BrowserReadResult = {
    id: `item-${source.id}`,
    sourceId: source.id,
    sourceLabel: source.label,
    kind: 'current_tab',
    url: url.href,
    title: source.title || url.hostname,
    excerpt: source.excerpt || '',
    capturedAt: source.capturedAt ?? now,
    content: `ID: item-${source.id}\nSource ID: browser:${source.id}:item-${source.id}\nSource: ${source.label}\nKind: Current tab\nURL: ${url.href}\n\n${source.excerpt || source.title || url.href}\n\nSources:\n- browser:${source.id}:item-${source.id} | ${source.title || url.href}`,
  }
  return { source, item }
}

export function mockCalendarMutationEvent(input: CalendarMutationInput, existing?: CalendarEvent | null): ConnectorEvent {
  const mode = input.mode === 'update' ? 'update' : 'create'
  const title = input.title?.trim() || existing?.title || 'Calendar event'
  const sourceId = input.sourceId || existing?.sourceId || mockState.settings.calendar.sources.find((source) => source.enabled)?.id || 'calendar'
  const attendees = input.attendees?.filter(Boolean) ?? existing?.attendees ?? []
  const sources = existing ? [mockCalendarSourceRef(existing)] : undefined
  return {
    id: uid(),
    connectorId: 'calendar',
    pluginName: 'Calendar',
    action: mode === 'update' ? 'calendar_update_requested' : 'calendar_create_requested',
    direction: 'outbound',
    classification: 'approval',
    priority: 'high',
    trustLevel: 'approve',
    requiresApproval: true,
    approvalId: `approval-${uid().slice(0, 8)}`,
    approvalStatus: 'pending',
    resolvedAt: null,
    resolvedBy: null,
    title,
    summary: `${mode === 'update' ? 'Update' : 'Create'} "${title}"${input.location ? ` at ${input.location}` : ''}. Calendar file is not modified until approval.`,
    actor: attendees[0] || input.organizer || existing?.organizer || 'Calendar',
    threadKey: mode === 'update'
      ? mockCalendarSourceId(existing ?? { id: input.id || 'event', sourceId, title, path: '', sourceLabel: 'Calendar', startAt: null, endAt: null, allDay: false, location: '', description: '', organizer: '', attendees: [], status: '' })
      : `calendar:create:${sourceId}:${uid().slice(0, 8)}`,
    payloadKeys: ['title', 'start_at', 'end_at', 'all_day', 'location', 'description', 'attendees', ...(mode === 'update' ? ['event_id'] : [])],
    sources,
    createdAt: Date.now(),
  }
}
