import type { ConnectorEvent, ConnectorInteractionInput, EmailDraftInput, EmailResolveInput, EmailSourceConfig, EmailSyncResult, EmailThreadResult, McpServerConfig, McpToolCallResult, ControllerSnapshot, ControllerStatus } from '../types'
import type { BrowserConnectorSettings, BrowserSourceConfig, BrowserSyncResult, CalendarSourceConfig, CalendarSyncResult, DiscordSource, DocumentSourceConfig, DocumentSyncResult, GoogleChatSource, MattermostSource, MatrixSource, SlackSourceConfig, SlackSyncResult } from '../types'
import { mockState, uid } from './state'
import { filterMockBrowser, filterMockCalendar, filterMockDocuments, filterMockEmail, filterMockSlack, mockBrowserCurrentTabSource, mockBrowserSourceId, mockBrowserSourceRef, mockBrowserSummary, mockCalendarMutationEvent, mockCalendarSummary, mockDocumentSourceId, mockDocumentSourceRef, mockDocumentSummary, mockEmailSourceRef, mockEmailThread, mockSlackSourceId, mockSlackSourceRef, mockSlackSummary } from './connectors'
import { mockMcpSnapshot, redactedMockSettings } from './runtime'
import type { TensorFoldStudioApi } from '../ipc'

function mockDiscordSource(label: string): DiscordSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Discord server', token: '', allowedChannels: '', enabled: false, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function mockMattermostSource(label: string): MattermostSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Mattermost server', baseUrl: '', token: '', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function mockMatrixSource(label: string): MatrixSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Matrix account', baseUrl: 'https://matrix.org', token: '', allowedRooms: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function mockGoogleChatSource(label: string): GoogleChatSource {
  const now = Date.now()
  return { id: uid(), label: label || 'Google Chat space', webhookUrl: '', accessToken: '', baseUrl: '', allowedChannels: '', enabled: true, lastSyncAt: null, lastError: null, createdAt: now, updatedAt: now }
}

function patchSource<T extends { id: string; updatedAt: number }>(sources: T[], id: string, patch: Partial<T>): T[] {
  return sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source)
}

export function createMockConnectionsApi(): Pick<TensorFoldStudioApi, 'recordConnectorInteraction' | 'replyToThread' | 'requestChannelApproval' | 'resolveConnectorApproval' | 'getControllerStatus' | 'enableController' | 'captureController' | 'getMcpSnapshot' | 'saveMcpServer' | 'deleteMcpServer' | 'connectMcpServer' | 'disconnectMcpServer' | 'refreshMcpServer' | 'testMcpTool' | 'addEmailSource' | 'updateEmailSource' | 'removeEmailSource' | 'syncEmail' | 'listEmail' | 'readEmail' | 'readEmailThread' | 'summarizeEmail' | 'addSlackSource' | 'updateSlackSource' | 'removeSlackSource' | 'syncSlack' | 'listSlack' | 'readSlack' | 'summarizeSlack' | 'verifySlack' | 'listSlackChannels' | 'sendSlack' | 'addDiscordSource' | 'removeDiscordSource' | 'verifyDiscord' | 'listDiscordChannels' | 'syncDiscord' | 'sendDiscord' | 'addMattermostSource' | 'updateMattermostSource' | 'removeMattermostSource' | 'verifyMattermost' | 'listMattermostChannels' | 'syncMattermost' | 'sendMattermost' | 'addMatrixSource' | 'updateMatrixSource' | 'removeMatrixSource' | 'verifyMatrix' | 'listMatrixChannels' | 'syncMatrix' | 'sendMatrix' | 'addGoogleChatSource' | 'updateGoogleChatSource' | 'removeGoogleChatSource' | 'verifyGoogleChat' | 'listGoogleChatChannels' | 'syncGoogleChat' | 'sendGoogleChat' | 'sendIMessage' | 'composeEmail' | 'resolveEmailThread' | 'addCalendarSource' | 'updateCalendarSource' | 'removeCalendarSource' | 'syncCalendar' | 'listCalendar' | 'readCalendar' | 'summarizeCalendar' | 'createCalendarEvent' | 'updateCalendarEvent' | 'addDocumentSource' | 'updateDocumentSource' | 'removeDocumentSource' | 'syncDocuments' | 'listDocuments' | 'readDocument' | 'summarizeDocuments' | 'addBrowserCurrentTabSource' | 'addBrowserHistorySource' | 'updateBrowserSource' | 'removeBrowserSource' | 'syncBrowser' | 'listBrowser' | 'readBrowser' | 'summarizeBrowser' | 'microsoftMailBegin' | 'microsoftMailPoll' | 'microsoftMailCancel' | 'imessageAccess' | 'syncImessage' | 'listImessage' | 'readImessage' | 'openFullDiskAccess' | 'pollTelegram'> {

  return {
    recordConnectorInteraction: async (input: ConnectorInteractionInput) => {
      const event: ConnectorEvent = {
        id: uid(), connectorId: input.connectorId, pluginName: input.pluginName || input.connectorId,
        action: input.action, direction: input.direction || 'internal', classification: 'communication',
        priority: 'normal', trustLevel: input.direction === 'outbound' ? 'approve' : 'suggest',
        requiresApproval: input.direction === 'outbound', approvalId: input.direction === 'outbound' ? `approval-${uid()}` : null,
        approvalStatus: input.direction === 'outbound' ? 'pending' : null,
        resolvedAt: null,
        resolvedBy: null,
        title: String(input.input?.title || input.input?.subject || input.action),
        summary: String(input.input?.summary || input.input?.body || 'Connector interaction recorded.'),
        actor: String(input.target || input.connectorId), threadKey: String(input.input?.threadKey || input.target || input.connectorId),
        payloadKeys: Object.keys(input.input || {}).slice(0, 10), createdAt: Date.now(),
      }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    replyToThread: async (_threadKey: string, text: string) => ({ ok: !!text.trim(), error: text.trim() ? undefined : 'Refusing to send an empty reply.' }),
    requestChannelApproval: async (_threadKey, _request) => ({ ok: false as const, error: 'In-channel approval needs the desktop app.', fallbackToApp: true }),
    resolveConnectorApproval: async (eventId, decision) => {
      const event = mockState.mockConnectorEvents.find((item) => item.id === eventId || item.approvalId === eventId)
      if (!event) throw new Error('Connector approval not found')
      event.approvalStatus = decision === 'approve' ? 'approved' : 'denied'
      event.requiresApproval = false
      event.resolvedAt = Date.now()
      event.resolvedBy = 'demo'
      if (decision === 'approve' && event.connectorId === 'email' && ['send_requested', 'reply_requested'].includes(event.action)) {
        event.delivery = {
          transport: 'local_outbox',
          status: 'delivered',
          path: `/Users/example/.vontra/email-outbox/${event.id}.eml`,
          deliveredAt: event.resolvedAt,
          error: null,
        }
      }
      const resolvedEvent: ConnectorEvent = {
        id: uid(),
        connectorId: event.connectorId,
        pluginName: event.pluginName,
        action: decision === 'approve' ? 'approval_approved' : 'approval_denied',
        direction: 'internal',
        classification: 'approval',
        priority: 'normal',
        trustLevel: 'auto_logged',
        requiresApproval: false,
        approvalId: null,
        approvalStatus: null,
        resolvedAt: null,
        resolvedBy: null,
        title: decision === 'approve' ? 'Approval allowed' : 'Approval denied',
        summary: event.title,
        actor: 'demo',
        threadKey: event.threadKey,
        payloadKeys: ['event_id', 'decision'],
        createdAt: Date.now(),
      }
      mockState.mockConnectorEvents = [
        resolvedEvent,
        ...mockState.mockConnectorEvents,
      ].slice(0, 80)
      return event
    },
    getControllerStatus: async (): Promise<ControllerStatus> => ({
      platform: 'browser',
      supported: true,
      enabled: mockState.settings.controller.enabled,
      accessibilityTrusted: true,
      canCapture: mockState.settings.controller.enabled,
      message: mockState.settings.controller.enabled ? 'Controller demo capture is ready.' : 'Controller is disabled.',
    }),
    enableController: async (): Promise<ControllerStatus> => {
      mockState.settings.controller = { ...mockState.settings.controller, enabled: true, lastError: null }
      return {
        platform: 'browser',
        supported: true,
        enabled: true,
        accessibilityTrusted: true,
        canCapture: true,
        message: 'Controller demo capture is ready.',
      }
    },
    captureController: async (): Promise<ControllerSnapshot> => {
      const fullText = [
        'Create an intelligent Gomoku game where the AI can explain why it chose each move.',
        'Goal: implement player turns, computer strategy, restart, and review changed files.',
        'Progress: board rendering complete; heuristic scoring needs refinement.',
      ].join('\n')
      const text = mockState.settings.controller.includeAccessibilityText ? fullText.slice(0, mockState.settings.controller.maxTextChars || 4_000) : ''
      const snapshot: ControllerSnapshot = {
        appName: 'Demo Browser',
        bundleId: 'dev.tensorfold.studio.demo',
        windowTitle: 'Gomoku task window',
        capturedAt: Date.now(),
        platform: 'browser',
        text,
        textCharCount: mockState.settings.controller.includeAccessibilityText ? fullText.length : 0,
        truncated: mockState.settings.controller.includeAccessibilityText && text.length < fullText.length,
        source: 'demo',
        warning: mockState.settings.controller.includeAccessibilityText ? null : 'Text capture is off; only app/window metadata was captured.',
      }
      mockState.settings.controller = { ...mockState.settings.controller, enabled: true, lastCapturedAt: snapshot.capturedAt, lastError: null }
      return snapshot
    },
    getMcpSnapshot: async () => mockMcpSnapshot(),
    saveMcpServer: async (input) => {
      const now = Date.now()
      const existing = input.id ? mockState.mockMcpServers.find((server) => server.id === input.id) : null
      const next: McpServerConfig = {
        id: existing?.id || input.id || uid(),
        name: input.name || existing?.name || 'New MCP server',
        transport: 'stdio',
        command: input.command ?? existing?.command ?? '',
        args: input.args ?? existing?.args ?? [],
        cwd: input.cwd ?? existing?.cwd ?? null,
        env: input.env ?? existing?.env ?? {},
        enabled: input.enabled ?? existing?.enabled ?? true,
        autoConnect: input.autoConnect ?? existing?.autoConnect ?? false,
        approvalMode: input.approvalMode ?? existing?.approvalMode ?? 'ask',
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
      }
      mockState.mockMcpServers = existing ? mockState.mockMcpServers.map((server) => server.id === next.id ? next : server) : [next, ...mockState.mockMcpServers]
      return mockMcpSnapshot()
    },
    deleteMcpServer: async (id) => {
      mockState.mockMcpServers = mockState.mockMcpServers.filter((server) => server.id !== id)
      mockState.mockMcpStopped.delete(id)
      return mockMcpSnapshot()
    },
    connectMcpServer: async (id) => {
      mockState.mockMcpServers = mockState.mockMcpServers.map((server) => server.id === id ? { ...server, enabled: true, updatedAt: Date.now() } : server)
      mockState.mockMcpStopped.delete(id)
      return mockMcpSnapshot()
    },
    disconnectMcpServer: async (id) => {
      mockState.mockMcpStopped.add(id)
      mockState.mockMcpServers = mockState.mockMcpServers.map((server) => server.id === id ? { ...server, updatedAt: Date.now() } : server)
      return mockMcpSnapshot()
    },
    refreshMcpServer: async () => mockMcpSnapshot(),
    testMcpTool: async (callName, args): Promise<McpToolCallResult> => {
      if (callName.endsWith('list_allowed_directories')) return { ok: true, output: '/Users/example/Projects', title: 'List allowed directories', meta: 'Filesystem demo MCP' }
      return { ok: true, output: `Demo MCP tool call ${callName}\n${JSON.stringify(args ?? {}, null, 2)}`, title: callName, meta: 'Demo MCP' }
    },
    addEmailSource: async (dir) => {
      const source: EmailSourceConfig = { id: uid(), label: dir.split('/').filter(Boolean).pop() || 'Email', path: dir, kind: 'eml_dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.email = { ...mockState.settings.email, sources: [source, ...mockState.settings.email.sources] }
      return mockState.settings.email
    },
    updateEmailSource: async (id, patch) => {
      mockState.settings.email = { ...mockState.settings.email, sources: mockState.settings.email.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) }
      return mockState.settings.email
    },
    removeEmailSource: async (id) => {
      mockState.settings.email = { ...mockState.settings.email, sources: mockState.settings.email.sources.filter((source) => source.id !== id) }
      return mockState.settings.email
    },
    syncEmail: async (): Promise<EmailSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.email = { ...mockState.settings.email, lastSyncAt: syncedAt, sources: mockState.settings.email.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const event: ConnectorEvent = { id: uid(), connectorId: 'email', pluginName: 'Email', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'high', trustLevel: 'suggest', requiresApproval: false, approvalId: null, title: 'Demo email sync completed', summary: 'Local email source scanned and recent messages are available to Assistant.', actor: 'Email', threadKey: `email:sync:${syncedAt}`, payloadKeys: ['source', 'message_count'], createdAt: syncedAt }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events: [event], messages: mockState.mockEmailMessages.length, sources: mockState.settings.email.sources, syncedAt }
    },
    listEmail: async (opts) => filterMockEmail(opts),
    readEmail: async (id) => {
      const message = mockState.mockEmailMessages.find((m) => m.id === id.replace(/^email:/, ''))
      if (!message) throw new Error('Email message not found')
      return message
    },
    readEmailThread: async (id): Promise<EmailThreadResult> => {
      return mockEmailThread(id)
    },
    summarizeEmail: async (opts) => {
      const messages = filterMockEmail(opts)
      const recent = messages.slice(0, 3)
      const sources = recent.map((message) => ({
        id: `email:${message.sourceId}:${message.id}`,
        connectorId: 'email' as const,
        sourceId: message.sourceId,
        itemId: message.id,
        title: message.subject || '(no subject)',
        path: message.path,
        excerpt: message.snippet,
        createdAt: message.date,
      }))
      return {
        summary: `${messages.length} demo messages, ${messages.filter((m) => m.unread).length} unread. Recent: ${recent.map((m) => `[email:${m.sourceId}:${m.id}] ${m.subject}`).join(', ')}.\n\nSources:\n${sources.length ? sources.map((source) => `- ${source.id} | ${source.title}`).join('\n') : 'none.'}`,
        sources,
      }
    },
    addSlackSource: async (dir) => {
      const source: SlackSourceConfig = { id: uid(), label: dir.split('/').filter(Boolean).pop() || 'Slack export', path: dir, kind: 'export_dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.slack = { ...mockState.settings.slack, sources: [source, ...mockState.settings.slack.sources] }
      return mockState.settings.slack
    },
    updateSlackSource: async (id, patch) => {
      mockState.settings.slack = { ...mockState.settings.slack, sources: mockState.settings.slack.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, kind: 'export_dir', updatedAt: Date.now() } : source) }
      return mockState.settings.slack
    },
    removeSlackSource: async (id) => {
      mockState.settings.slack = { ...mockState.settings.slack, sources: mockState.settings.slack.sources.filter((source) => source.id !== id) }
      return mockState.settings.slack
    },
    syncSlack: async (id?: string): Promise<SlackSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.slack = { ...mockState.settings.slack, lastSyncAt: syncedAt, sources: mockState.settings.slack.sources.map((source) => !id || source.id === id ? { ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null } : source) }
      const messages = filterMockSlack({ limit: mockState.settings.slack.maxMessages }).filter((message) => !id || message.sourceId === id)
      const events: ConnectorEvent[] = messages.slice(0, 6).map((message) => ({ id: uid(), connectorId: 'slack', pluginName: 'Slack', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'normal', trustLevel: 'notify', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: `#${message.channelName} ${message.senderName || 'Unknown'}`, summary: message.snippet, actor: `#${message.channelName}`, threadKey: mockSlackSourceId(message), payloadKeys: ['channel', 'message', 'source'], sources: [mockSlackSourceRef(message)], createdAt: syncedAt }))
      mockState.mockConnectorEvents = [...events, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events, messages: messages.length, sources: mockState.settings.slack.sources.filter((source) => !id || source.id === id), syncedAt }
    },
    listSlack: async (opts) => filterMockSlack(opts),
    readSlack: async (id) => {
      const target = id.replace(/^slack:[^:]+:/, '')
      const message = mockState.mockSlackMessages.find((item) => item.id === target || mockSlackSourceId(item) === id)
      if (!message) throw new Error('Slack message not found')
      return message
    },
    summarizeSlack: async (opts) => mockSlackSummary(opts),
    verifySlack: async () => ({ ok: false, error: 'Slack live API is unavailable in browser demo mode.' }),
    listSlackChannels: async () => [],
    sendSlack: async () => { throw new Error('Slack send is unavailable in browser demo mode.') },
    addDiscordSource: async (label) => {
      mockState.settings.discord = { ...mockState.settings.discord, sources: [...mockState.settings.discord.sources, mockDiscordSource(label)] }
      return redactedMockSettings().discord
    },
    removeDiscordSource: async (id) => {
      mockState.settings.discord = { ...mockState.settings.discord, sources: mockState.settings.discord.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().discord
    },
    verifyDiscord: async () => ({ ok: false, error: 'Discord live API is unavailable in browser demo mode.' }),
    listDiscordChannels: async () => [],
    syncDiscord: async () => ({ messages: 0, events: 0 }),
    sendDiscord: async () => { throw new Error('Discord send is unavailable in browser demo mode.') },
    addMattermostSource: async (label) => {
      mockState.settings.mattermost = { ...mockState.settings.mattermost, sources: [...mockState.settings.mattermost.sources, mockMattermostSource(label)] }
      return redactedMockSettings().mattermost
    },
    updateMattermostSource: async (id, patch) => {
      mockState.settings.mattermost = { ...mockState.settings.mattermost, sources: patchSource(mockState.settings.mattermost.sources, id, patch) }
      return redactedMockSettings().mattermost
    },
    removeMattermostSource: async (id) => {
      mockState.settings.mattermost = { ...mockState.settings.mattermost, sources: mockState.settings.mattermost.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().mattermost
    },
    verifyMattermost: async () => ({ ok: false, error: 'Mattermost live API is unavailable in browser demo mode.' }),
    listMattermostChannels: async () => [],
    syncMattermost: async () => ({ messages: 0, events: 0 }),
    sendMattermost: async () => { throw new Error('Mattermost send is unavailable in browser demo mode.') },
    addMatrixSource: async (label) => {
      mockState.settings.matrix = { ...mockState.settings.matrix, sources: [...mockState.settings.matrix.sources, mockMatrixSource(label)] }
      return redactedMockSettings().matrix
    },
    updateMatrixSource: async (id, patch) => {
      mockState.settings.matrix = { ...mockState.settings.matrix, sources: patchSource(mockState.settings.matrix.sources, id, patch) }
      return redactedMockSettings().matrix
    },
    removeMatrixSource: async (id) => {
      mockState.settings.matrix = { ...mockState.settings.matrix, sources: mockState.settings.matrix.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().matrix
    },
    verifyMatrix: async () => ({ ok: false, error: 'Matrix live API is unavailable in browser demo mode.' }),
    listMatrixChannels: async () => [],
    syncMatrix: async () => ({ messages: 0, events: 0 }),
    sendMatrix: async () => { throw new Error('Matrix send is unavailable in browser demo mode.') },
    addGoogleChatSource: async (label) => {
      mockState.settings.googlechat = { ...mockState.settings.googlechat, sources: [...mockState.settings.googlechat.sources, mockGoogleChatSource(label)] }
      return redactedMockSettings().googlechat
    },
    updateGoogleChatSource: async (id, patch) => {
      mockState.settings.googlechat = { ...mockState.settings.googlechat, sources: patchSource(mockState.settings.googlechat.sources, id, patch) }
      return redactedMockSettings().googlechat
    },
    removeGoogleChatSource: async (id) => {
      mockState.settings.googlechat = { ...mockState.settings.googlechat, sources: mockState.settings.googlechat.sources.filter((source) => source.id !== id) }
      return redactedMockSettings().googlechat
    },
    verifyGoogleChat: async () => ({ ok: false, mode: 'webhook' as const, error: 'Google Chat live API is unavailable in browser demo mode.' }),
    listGoogleChatChannels: async () => [],
    syncGoogleChat: async () => ({ messages: 0, events: 0 }),
    sendGoogleChat: async () => { throw new Error('Google Chat send is unavailable in browser demo mode.') },
    sendIMessage: async () => { throw new Error('iMessage send is unavailable in browser demo mode.') },
    composeEmail: async (input: EmailDraftInput) => {
      const wantsSend = input.mode === 'send'
      const wantsReply = Boolean(input.inReplyTo || input.threadKey)
      const action = wantsSend ? wantsReply ? 'reply_requested' : 'send_requested' : wantsReply ? 'reply_draft_created' : 'draft_created'
      const thread = wantsReply ? mockEmailThread(input.threadKey || input.inReplyTo || '') : null
      const sources = thread?.messages.map(mockEmailSourceRef)
      const event: ConnectorEvent = { id: uid(), connectorId: 'email', pluginName: 'Email', action, direction: 'outbound', classification: 'approval', priority: wantsSend ? 'high' : 'normal', trustLevel: wantsSend ? 'approve' : 'suggest', requiresApproval: wantsSend, approvalId: wantsSend ? `approval-${uid()}` : null, approvalStatus: wantsSend ? 'pending' : null, resolvedAt: null, resolvedBy: null, title: input.subject, summary: `${wantsSend ? wantsReply ? 'Reply send request' : 'Send request' : wantsReply ? 'Reply draft' : 'Draft'} to ${input.to}: ${input.body.slice(0, 180)}`, actor: input.to, threadKey: thread ? `email:${thread.root.id}` : input.threadKey || `email:draft:${uid()}`, payloadKeys: ['to', 'subject', 'body', ...(input.threadKey ? ['thread_key'] : []), ...(input.inReplyTo ? ['in_reply_to'] : [])], payload: { mode: input.mode ?? 'draft', to: input.to, subject: input.subject, body: input.body, ...(input.threadKey ? { threadKey: input.threadKey } : {}), ...(input.inReplyTo ? { inReplyTo: input.inReplyTo } : {}) }, sources, createdAt: Date.now() }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    resolveEmailThread: async (input: EmailResolveInput) => {
      const thread = mockEmailThread(input.id)
      const event: ConnectorEvent = { id: uid(), connectorId: 'email', pluginName: 'Email', action: 'thread_resolved', direction: 'internal', classification: 'communication', priority: 'normal', trustLevel: 'auto_logged', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: Date.now(), resolvedBy: 'user', title: thread.root.subject || 'Email thread resolved', summary: `${input.note || 'Marked resolved locally.'} ${thread.summary}`, actor: thread.root.from || 'Email', threadKey: `email:${thread.root.id}`, payloadKeys: ['message_id', ...(input.note ? ['note'] : [])], sources: thread.messages.map(mockEmailSourceRef), createdAt: Date.now() }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    addCalendarSource: async (target) => {
      const source: CalendarSourceConfig = { id: uid(), label: target.split('/').filter(Boolean).pop() || 'Calendar', path: target, kind: target.toLowerCase().endsWith('.ics') ? 'ics_file' : 'ics_dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.calendar = { ...mockState.settings.calendar, sources: [source, ...mockState.settings.calendar.sources] }
      return mockState.settings.calendar
    },
    updateCalendarSource: async (id, patch) => {
      mockState.settings.calendar = { ...mockState.settings.calendar, sources: mockState.settings.calendar.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) }
      return mockState.settings.calendar
    },
    removeCalendarSource: async (id) => {
      mockState.settings.calendar = { ...mockState.settings.calendar, sources: mockState.settings.calendar.sources.filter((source) => source.id !== id) }
      return mockState.settings.calendar
    },
    syncCalendar: async (): Promise<CalendarSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.calendar = { ...mockState.settings.calendar, lastSyncAt: syncedAt, sources: mockState.settings.calendar.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const event: ConnectorEvent = { id: uid(), connectorId: 'calendar', pluginName: 'Calendar', action: 'event_seen', direction: 'inbound', classification: 'schedule', priority: 'normal', trustLevel: 'notify', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: 'Demo calendar sync completed', summary: 'Local .ics source scanned and calendar events are available to Assistant.', actor: 'Calendar', threadKey: `calendar:sync:${syncedAt}`, payloadKeys: ['source', 'event_count'], createdAt: syncedAt }
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events: filterMockCalendar({ limit: mockState.settings.calendar.maxEvents }), sources: mockState.settings.calendar.sources, syncedAt }
    },
    listCalendar: async (opts) => filterMockCalendar(opts),
    readCalendar: async (id) => {
      const target = id.replace(/^calendar:[^:]+:/, '')
      const event = mockState.mockCalendarEvents.find((item) => item.id === target)
      if (!event) throw new Error('Calendar event not found')
      return event
    },
    summarizeCalendar: async (opts) => mockCalendarSummary(opts),
    createCalendarEvent: async (input) => {
      const enabledSource = input.sourceId
        ? mockState.settings.calendar.sources.find((source) => source.id === input.sourceId && source.enabled)
        : mockState.settings.calendar.sources.find((source) => source.enabled)
      if (!enabledSource) throw new Error('No enabled calendar source is available for the create request.')
      const event = mockCalendarMutationEvent({ ...input, mode: 'create', sourceId: enabledSource.id })
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    updateCalendarEvent: async (input) => {
      const target = input.id.replace(/^calendar:[^:]+:/, '')
      const existing = mockState.mockCalendarEvents.find((item) => item.id === target)
      if (!existing) throw new Error('Calendar event not found')
      const event = mockCalendarMutationEvent({ ...input, mode: 'update', sourceId: existing.sourceId }, existing)
      mockState.mockConnectorEvents = [event, ...mockState.mockConnectorEvents].slice(0, 80)
      return event
    },
    addDocumentSource: async (target) => {
      const source: DocumentSourceConfig = { id: uid(), label: target.split('/').filter(Boolean).pop() || 'Documents', path: target, kind: /\.(md|mdx|txt|rst|csv|tsv|xlsx|json|jsonl|ya?ml|toml|xml|html?|css|jsx?|tsx?|py|rb|go|rs|java|c|cpp|h|hpp|pdf)$/i.test(target) ? 'file' : 'dir', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
      mockState.settings.documents = { ...mockState.settings.documents, sources: [source, ...mockState.settings.documents.sources] }
      return mockState.settings.documents
    },
    updateDocumentSource: async (id, patch) => {
      mockState.settings.documents = { ...mockState.settings.documents, sources: mockState.settings.documents.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, updatedAt: Date.now() } : source) }
      return mockState.settings.documents
    },
    removeDocumentSource: async (id) => {
      mockState.settings.documents = { ...mockState.settings.documents, sources: mockState.settings.documents.sources.filter((source) => source.id !== id) }
      return mockState.settings.documents
    },
    syncDocuments: async (): Promise<DocumentSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.documents = { ...mockState.settings.documents, lastSyncAt: syncedAt, sources: mockState.settings.documents.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const docs = filterMockDocuments({ limit: mockState.settings.documents.maxDocuments })
      const events: ConnectorEvent[] = docs.map((doc) => ({ id: uid(), connectorId: 'docs', pluginName: 'Documents', action: 'document_indexed', direction: 'inbound', classification: 'document', priority: 'normal', trustLevel: 'suggest', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: doc.title || doc.relativePath, summary: `${doc.relativePath}: ${doc.excerpt}`, actor: doc.sourceLabel, threadKey: mockDocumentSourceId(doc), payloadKeys: ['path', 'title', 'excerpt'], sources: [mockDocumentSourceRef(doc)], createdAt: syncedAt }))
      mockState.mockConnectorEvents = [...events, ...mockState.mockConnectorEvents].slice(0, 80)
      return { documents: docs, sources: mockState.settings.documents.sources, syncedAt }
    },
    listDocuments: async (opts) => filterMockDocuments(opts),
    readDocument: async (id) => {
      const target = id.replace(/^docs:[^:]+:/, '')
      const doc = mockState.mockDocuments.find((item) => item.id === target || item.relativePath === id || item.path === id)
      if (!doc) throw new Error('Document not found')
      return doc
    },
    summarizeDocuments: async (opts) => mockDocumentSummary(opts),
    addBrowserCurrentTabSource: async (input): Promise<BrowserConnectorSettings> => {
      const parsed = new URL(String(input.url || '').trim())
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Browser source URLs must use http or https.')
      const { source, item } = mockBrowserCurrentTabSource(input)
      mockState.settings.browser = { ...mockState.settings.browser, sources: [source, ...mockState.settings.browser.sources] }
      mockState.mockBrowserItems = [item, ...mockState.mockBrowserItems]
      return mockState.settings.browser
    },
    addBrowserHistorySource: async (target): Promise<BrowserConnectorSettings> => {
      const now = Date.now()
      const source: BrowserSourceConfig = { id: uid(), label: target.split('/').filter(Boolean).pop() || 'Browser history export', kind: 'history_export', enabled: true, createdAt: now, updatedAt: now, lastSyncAt: null, lastError: null, path: target }
      mockState.settings.browser = { ...mockState.settings.browser, sources: [source, ...mockState.settings.browser.sources] }
      return mockState.settings.browser
    },
    updateBrowserSource: async (id, patch): Promise<BrowserConnectorSettings> => {
      mockState.settings.browser = { ...mockState.settings.browser, sources: mockState.settings.browser.sources.map((source) => source.id === id ? { ...source, ...patch, id: source.id, kind: source.kind, updatedAt: Date.now() } : source) }
      return mockState.settings.browser
    },
    removeBrowserSource: async (id): Promise<BrowserConnectorSettings> => {
      mockState.settings.browser = { ...mockState.settings.browser, sources: mockState.settings.browser.sources.filter((source) => source.id !== id) }
      return mockState.settings.browser
    },
    syncBrowser: async (): Promise<BrowserSyncResult> => {
      const syncedAt = Date.now()
      mockState.settings.browser = { ...mockState.settings.browser, lastSyncAt: syncedAt, sources: mockState.settings.browser.sources.map((source) => ({ ...source, lastSyncAt: syncedAt, updatedAt: syncedAt, lastError: null })) }
      const items = filterMockBrowser({ limit: mockState.settings.browser.maxItems })
      const events: ConnectorEvent[] = items.map((item) => ({ id: uid(), connectorId: 'browser', pluginName: 'Browser', action: 'browser_source_seen', direction: 'inbound', classification: 'document', priority: 'low', trustLevel: 'notify', requiresApproval: false, approvalId: null, approvalStatus: null, resolvedAt: null, resolvedBy: null, title: item.title || item.url, summary: item.excerpt || item.url, actor: item.sourceLabel, threadKey: mockBrowserSourceId(item), payloadKeys: ['source', 'url', 'title'], sources: [mockBrowserSourceRef(item)], createdAt: item.capturedAt ?? syncedAt }))
      mockState.mockConnectorEvents = [...events, ...mockState.mockConnectorEvents].slice(0, 80)
      return { events, items, sources: mockState.settings.browser.sources, syncedAt }
    },
    listBrowser: async (opts) => filterMockBrowser(opts),
    readBrowser: async (id) => {
      const target = id.replace(/^browser:[^:]+:/, '')
      const item = mockState.mockBrowserItems.find((candidate) => candidate.id === target || candidate.url === id || mockBrowserSourceId(candidate) === id)
      if (!item) throw new Error('Browser source item not found')
      return item
    },
    summarizeBrowser: async (opts) => mockBrowserSummary(opts),
    // iMessage is a local-only connector; browser demo mode reports it unavailable.
    microsoftMailBegin: async () => { throw new Error('Microsoft sign-in requires the desktop app.') },
    microsoftMailPoll: async () => { throw new Error('Microsoft sign-in requires the desktop app.') },
    microsoftMailCancel: async () => { },
    imessageAccess: async () => ({ state: 'no_macos', detail: 'iMessage local bridge is unavailable in browser demo mode.' }),
    syncImessage: async () => ({ events: [], messages: 0, syncedAt: Date.now() }),
    listImessage: async () => [],
    readImessage: async () => null,
    openFullDiskAccess: async () => { },
    pollTelegram: async () => {
      mockState.settings.botControl.telegram = { ...mockState.settings.botControl.telegram, lastPollAt: Date.now(), lastError: null }
      return { ok: true, processed: 0, skipped: 0, error: null, lastUpdateId: mockState.settings.botControl.telegram.lastUpdateId }
    },
  }
}
