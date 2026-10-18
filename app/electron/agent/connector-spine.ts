import { randomUUID } from 'node:crypto'
import { CONNECTOR_CATALOG } from '../../shared/connector-catalog'
import { isWhatsAppEventInCurrentScope } from '../../shared/whatsapp-event-scope'
import type {
  AssistantControlSnapshot,
  ConnectorDefinition,
  ConnectorEvent,
  ConnectorId,
  ConnectorInteractionInput,
  ConnectorPermissionStatus,
  ConnectorPriority,
  ConnectorToolSchema,
  ConnectorTrustLevel,
  ControlPlaneGap,
  BrowserConnectorSettings,
  BotControlSettings,
  ControllerSettings,
  DocumentConnectorSettings,
  EmailConnectorSettings,
  IMessageConnectorSettings,
  CalendarConnectorSettings,
  SlackConnectorSettings,
  WhatsAppBridgeConnectorSettings,
  WhatsAppBridgeRuntimeSnapshot,
  DiscordConnectorSettings,
  MattermostConnectorSettings,
  MatrixConnectorSettings,
  GoogleChatConnectorSettings,
  RunBudgetDefinition,
  Mission,
  Project,
  ProjectMemoryCandidate,
} from './types'
import { botControlBridgePolicy, personalBridgePolicy, type PersonalBridgePolicy } from './bot-control'
import { traceForConnectorEvent } from './trace'
import { buildConnectorBriefing, connectorAttentionItems, connectorMemoryCandidateDrafts, connectorMissionTriggers } from './connector-feed'

const EMAIL_TOOLS: ConnectorToolSchema[] = [
  tool('email_list', 'List emails with filters for sender, folder, unread, search, and time window.'),
  tool('email_read', 'Read a specific email by message id.'),
  tool('email_thread', 'Read the local conversation around a message id before drafting a source-backed reply.'),
  tool('email_compose', 'Compose, draft, reply to, or send an email.', true),
  tool('email_resolve', 'Mark an email thread resolved in TensorFold after review or reply drafting.'),
  tool('email_summary', 'Summarise email activity with counts, topics, and top senders.'),
]

const MESSAGE_TOOLS: ConnectorToolSchema[] = [
  tool('message_list', 'List recent messages by channel, person, unread state, or time window.'),
  tool('message_read', 'Read a specific thread or message by id.'),
  tool('message_compose', 'Draft or send a reply through the connector transport.', true),
  tool('message_summary', 'Summarise unread messages, decisions, blockers, and requested replies.'),
]

const SLACK_TOOLS: ConnectorToolSchema[] = [
  tool('message_list', 'List recent Slack export messages by channel, sender, or query.'),
  tool('message_read', 'Read a specific Slack export message by source id.'),
  tool('message_summary', 'Summarise Slack export messages with cited source ids.'),
]

const DOCUMENT_TOOLS: ConnectorToolSchema[] = [
  tool('document_list', 'List indexed local documents and PDFs by path, tag, source, or modified time.'),
  tool('document_read', 'Read an indexed document, PDF, or excerpt by id/path.'),
  tool('document_summary', 'Summarise a document/PDF set with cited source ids.'),
  tool('table_list', 'List indexed CSV/TSV/XLSX tables with row and column metadata.'),
  tool('table_read', 'Read a CSV/TSV/XLSX table preview by id/path.'),
  tool('table_summary', 'Summarise local tables with source ids and numeric column stats.'),
]

const BROWSER_TOOLS: ConnectorToolSchema[] = [
  tool('browser_list', 'List explicitly saved current-tab snapshots and user-provided browser history exports.'),
  tool('browser_read', 'Read one explicit browser source item by source id, URL, or item id.'),
  tool('browser_summary', 'Summarise explicit browser current-tab/history sources with cited source ids.'),
]

const CONTROLLER_TOOLS: ConnectorToolSchema[] = [
  tool('controller_capture', 'Attach the current foreground window as readable context for the active chat or task.'),
  tool('controller_action', 'Approval-gated local Mac app actions through Accessibility.', true),
]

export const RUN_BUDGETS: RunBudgetDefinition[] = [
  budget('interactive', 'Interactive', 30, 0, 0, false),
  budget('backlog', 'Backlog', 60, 0, 0, false),
  budget('background', 'Background', 300, 60, 120, true),
  budget('mission', 'Mission', 3600, 30, 120, true),
  budget('overnight', 'Overnight', 28_800, 60, 300, true),
  budget('unbounded_with_heartbeat', 'Unbounded', 0, 30, 180, true),
]

const BASE_CONNECTORS: ConnectorDefinition[] = [
  connector('email', 'communication', 'pending', 'Local inbox triage, summaries, drafts, replies, and send requests with approval.', 'email', EMAIL_TOOLS),
  connector('slack', 'communication', 'pending', 'Local Slack export or live workspace channels and direct messages.', 'slack', SLACK_TOOLS),
  connector('discord', 'communication', 'pending', 'Live Discord channels and threads via bot token.', 'discord', []),
  connector('mattermost', 'communication', 'pending', 'Live Mattermost channels and threads via Personal Access / bot token.', 'mattermost', MESSAGE_TOOLS),
  connector('matrix', 'communication', 'pending', 'Live Matrix rooms and threads via a homeserver access token.', 'matrix', MESSAGE_TOOLS),
  connector('googlechat', 'communication', 'pending', 'Google Chat spaces via an incoming webhook (send) and an optional service-account token (read).', 'googlechat', MESSAGE_TOOLS),
  connector('whatsapp', 'communication', 'pending', 'Opt-in personal WhatsApp bridge with allowlisted senders and approval-gated replies.', 'whatsapp', []),
  connector('imessage', 'communication', 'pending', 'Opt-in local iMessage bridge contract; unavailable until an explicit bridge and contact allowlist exist.', null, []),
  connector('telegram', 'communication', 'pending', 'Opt-in Telegram bot command and delivery channel with allowlisted chats.', 'telegram', []),
  connector('controller', 'system', 'pending', 'Attach current app/window context through local Accessibility capture.', null, CONTROLLER_TOOLS),
  connector('webhook', 'communication', 'pending', 'Inbound signed localhost routes normalized into connector events.', null, []),
  connector('docs', 'document', 'mock', 'Local files, PDFs, notes, and source-backed summaries.', null, DOCUMENT_TOOLS),
  connector('browser', 'document', 'pending', 'Explicit current-tab snapshots and user-provided browser history exports for source-backed reads and summaries.', null, BROWSER_TOOLS),
  connector('calendar', 'calendar', 'pending', 'Schedule changes, meeting reminders, and planning signals.', 'calendar', [
    tool('calendar_list', 'List upcoming calendar events.'),
    tool('calendar_summary', 'Summarise schedule changes and commitments.'),
    tool('calendar_create', 'Create a calendar event.', true),
    tool('calendar_update', 'Update a calendar event.', true),
  ]),
]

export function recordConnectorInteraction(input: ConnectorInteractionInput): ConnectorEvent {
  const connector = BASE_CONNECTORS.find((c) => c.id === input.connectorId)
  const direction = input.direction ?? 'internal'
  const payload = { ...(input.input ?? {}), ...(input.output ?? {}) }
  const classification = classify(input.connectorId, input.action)
  const trustLevel = trust(input.action, direction)
  const priority = priorityFor(input.action, trustLevel)
  const requiresApproval = trustLevel === 'approve'
  // Inbound webhook payloads are fully attacker-controlled — never copy raw payload values
  // into human/agent-visible fields (title/summary/actor/threadKey). Derive them from the
  // redacted key names only, so untrusted text can't reach the UI or the model's context.
  const keys = safeKeys(payload)
  const untrusted = input.connectorId === 'webhook'
  return event({
    connectorId: input.connectorId,
    action: input.action,
    title: untrusted ? `${input.pluginName ?? connector?.name ?? 'Webhook'} · ${input.action}` : titleFrom(input, connector?.name ?? input.connectorId),
    summary: untrusted ? (keys.length ? `${keys.length} field(s): ${keys.join(', ')}` : 'No fields.') : summaryFrom(input),
    actor: untrusted ? String(input.pluginName ?? connector?.name ?? 'webhook') : String(input.target || payload.sender || payload.from || connector?.name || input.connectorId),
    threadKey: untrusted ? String(input.target || input.connectorId) : String(payload.threadKey || payload.thread_id || payload.message_id || input.target || input.connectorId),
    createdAt: Date.now(),
    priority,
    trustLevel,
    requiresApproval,
    classification,
    pluginName: input.pluginName ?? connector?.name ?? input.connectorId,
    direction,
    payloadKeys: keys,
  })
}

export interface AssistantControlSnapshotOptions {
  projects?: Project[]
  missions?: Mission[]
  existingMemoryCandidates?: ProjectMemoryCandidate[]
  defaultProjectId?: string | null
  now?: number
  whatsappRuntime?: readonly WhatsAppBridgeRuntimeSnapshot[]
}

export function buildAssistantControlSnapshot(events: ConnectorEvent[], email?: EmailConnectorSettings, botControl?: BotControlSettings, controllerSettings?: ControllerSettings, calendar?: CalendarConnectorSettings, documents?: DocumentConnectorSettings, slack?: SlackConnectorSettings, browser?: BrowserConnectorSettings, imessage?: IMessageConnectorSettings, whatsapp?: WhatsAppBridgeConnectorSettings, discord?: DiscordConnectorSettings, mattermost?: MattermostConnectorSettings, matrix?: MatrixConnectorSettings, googlechat?: GoogleChatConnectorSettings, options: AssistantControlSnapshotOptions = {}): AssistantControlSnapshot {
  // Reapply current QR source grants to cached incoming content. Pending exact
  // outbound review cards remain visible so their author can approve or discard.
  const latest = events.filter((event) => event.connectorId !== 'whatsapp' || !event.threadKey.startsWith('whatsapp:') || event.direction !== 'inbound' || isWhatsAppEventInCurrentScope(event, whatsapp?.sources ?? [])).sort((a, b) => b.createdAt - a.createdAt)
  const snapshotEvents = latest.slice(0, 40)
  const connectors = connectorDefinitions(email, botControl, controllerSettings, calendar, documents, slack, browser, imessage, whatsapp, discord, mattermost, matrix, googlechat, options.whatsappRuntime).map((connector) => {
    const sourceEvents = latest.filter((event) => event.connectorId === connector.id)
    return {
      ...connector,
      lastEventAt: sourceEvents[0]?.createdAt ?? connector.lastEventAt,
      unread: sourceEvents.filter((event) => event.direction !== 'outbound').length,
      pendingApprovals: sourceEvents.filter((event) => event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending').length,
    }
  })
  const gaps = detectGaps(connectors, latest)
  return {
    generatedAt: options.now ?? Date.now(),
    connectors,
    events: snapshotEvents,
    attention: connectorAttentionItems(snapshotEvents),
    briefing: buildConnectorBriefing(snapshotEvents, options.now),
    memoryCandidates: connectorMemoryCandidateDrafts(snapshotEvents, {
      projects: options.projects,
      defaultProjectId: options.defaultProjectId,
      existingCandidates: options.existingMemoryCandidates,
      now: options.now,
    }),
    missionTriggers: connectorMissionTriggers(snapshotEvents, options.missions),
    gaps,
    transports: connectors.filter((c) => c.transport).map((c) => c.transport!),
    tools: connectors.flatMap((c) => c.tools),
    budgets: RUN_BUDGETS,
    pendingApprovals: latest.filter((event) => event.requiresApproval && (event.approvalStatus ?? 'pending') === 'pending').length,
  }
}

function connector(id: ConnectorId, kind: ConnectorDefinition['kind'], status: ConnectorDefinition['status'], description: string, transport: string | null, tools: ConnectorToolSchema[]): ConnectorDefinition {
  return {
    id,
    name: CONNECTOR_CATALOG[id].name,
    kind,
    status,
    description,
    transport,
    tools,
    unread: 0,
    pendingApprovals: 0,
    lastSyncAt: null,
    lastEventAt: null,
    permissionStatus: permissionFromStatus(status),
    permissionDetail: status === 'pending' ? 'Connector is not configured.' : status === 'mock' ? 'Demo/mock connector contract.' : 'Connector permission is available.',
  }
}

function personalBridgeConnector(connector: ConnectorDefinition, policy: PersonalBridgePolicy, detail: string): ConnectorDefinition {
  return {
    ...connector,
    status: policy.status,
    description: `${connector.description} ${detail}`,
    tools: policy.optedIn ? MESSAGE_TOOLS : [],
    error: policy.lastError ?? undefined,
    lastSyncAt: policy.lastActivityAt,
    lastEventAt: policy.lastActivityAt,
    permissionStatus: policy.permissionStatus,
    permissionDetail: policy.permissionDetail,
  }
}

function connectorDefinitions(email?: EmailConnectorSettings, botControl?: BotControlSettings, controllerSettings?: ControllerSettings, calendar?: CalendarConnectorSettings, documents?: DocumentConnectorSettings, slack?: SlackConnectorSettings, browser?: BrowserConnectorSettings, imessage?: IMessageConnectorSettings, whatsapp?: WhatsAppBridgeConnectorSettings, discord?: DiscordConnectorSettings, mattermost?: MattermostConnectorSettings, matrix?: MatrixConnectorSettings, googlechat?: GoogleChatConnectorSettings, whatsappRuntime: readonly WhatsAppBridgeRuntimeSnapshot[] = []): ConnectorDefinition[] {
  const emailSettings = email ?? { sources: [], maxMessages: 0, lastSyncAt: null }
  const enabledSources = emailSettings.sources.filter((source) => source.enabled)
  const sourceErrors = enabledSources.filter((source) => source.lastError)
  const slackSettings = slack ?? { sources: [], maxMessages: 0, lastSyncAt: null }
  const enabledSlackSources = slackSettings.sources.filter((source) => source.enabled)
  const slackErrors = enabledSlackSources.filter((source) => source.lastError)
  const calendarSettings = calendar ?? { sources: [], maxEvents: 0, lastSyncAt: null }
  const enabledCalendarSources = calendarSettings.sources.filter((source) => source.enabled)
  const calendarErrors = enabledCalendarSources.filter((source) => source.lastError)
  const documentSettings = documents ?? { sources: [], maxDocuments: 0, maxDocumentBytes: 0, lastSyncAt: null }
  const enabledDocumentSources = documentSettings.sources.filter((source) => source.enabled)
  const documentErrors = enabledDocumentSources.filter((source) => source.lastError)
  const browserSettings = browser ?? { sources: [], maxItems: 0, lastSyncAt: null }
  const enabledBrowserSources = browserSettings.sources.filter((source) => source.enabled)
  const browserErrors = enabledBrowserSources.filter((source) => source.lastError)
  const telegramPolicy = botControl
    ? botControlBridgePolicy(botControl, 'telegram')
    : personalBridgePolicy({ channel: 'telegram', label: 'Telegram', enabled: false, allowedTargets: '', allowedTargetLabel: 'allowed Telegram chat IDs', requiredFields: [{ label: 'bot token', value: '' }, { label: 'allowed Telegram user IDs', value: '' }] })
  const whatsappPolicy = botControl
    ? botControlBridgePolicy(botControl, 'whatsapp')
    : personalBridgePolicy({ channel: 'whatsapp', label: 'WhatsApp', enabled: false, allowedTargets: '', allowedTargetLabel: 'allowed sender IDs', requiredFields: [{ label: 'access token', value: '' }, { label: 'phone number ID', value: '' }, { label: 'verify token', value: '' }, { label: 'Meta app secret', value: '' }] })
  const imessagePolicy = personalBridgePolicy({
    channel: 'imessage',
    label: 'iMessage',
    enabled: imessage?.enabled ?? false,
    allowedTargets: imessage?.allowedChatIds ?? '',
    allowedTargetLabel: 'allowed contacts',
    requiredFields: [{ label: 'full disk access', value: imessage?.enabled ? 'granted' : '' }],
  })
  return BASE_CONNECTORS.map((connector) => {
    if (connector.id === 'telegram') return personalBridgeConnector(connector, telegramPolicy, botControl?.telegram.polling ? 'Local polling is enabled.' : 'Local polling is off.')
    if (connector.id === 'whatsapp') {
      const enabled = (whatsapp?.sources ?? []).filter((source) => source.enabled)
      const runtime = whatsappRuntime.filter((state) => enabled.some((source) => source.id === state.sourceId))
      const connected = runtime.filter((state) => state.phase === 'connected')
      if (connected.length) {
        const degraded = runtime.some((state) => state.phase === 'error' || state.phase === 'reconnecting')
        return {
          ...connector, status: 'connected', transport: 'whatsapp', tools: MESSAGE_TOOLS.filter((item) => item.name === 'message_compose'),
          permissionStatus: degraded ? 'degraded' : 'granted',
          permissionDetail: 'Linked device connected. Reads and sends are limited to the selected chats.',
          description: `${connected.length} linked WhatsApp account${connected.length === 1 ? '' : 's'} connected. Experimental phone bridge.`,
          lastEventAt: Math.max(...enabled.map((source) => source.lastEventAt ?? 0)) || null,
        }
      }
      if (whatsappPolicy.optedIn) return { ...personalBridgeConnector(connector, whatsappPolicy, 'Business Cloud bot-control route configured; the phone bridge is separate.'), tools: [] }
      const error = runtime.find((state) => state.phase === 'error')
      const connecting = runtime.some((state) => ['connecting', 'pairing', 'reconnecting'].includes(state.phase))
      return {
        ...connector, status: error ? 'error' : 'pending', tools: [],
        permissionStatus: error ? 'degraded' : 'not_configured',
        permissionDetail: error?.lastError || (connecting ? 'Connecting the linked device.' : enabled.some((source) => source.paired) ? 'Saved device credentials; reconnect to use this account.' : 'Link your phone in Connections.'),
        description: 'Link WhatsApp through Linked Devices without a Business API key. Experimental.',
        error: error?.lastError ?? undefined,
      }
    }
    if (connector.id === 'imessage') {
      const detail = imessage?.enabled
        ? (imessage.lastError ? `iMessage bridge error: ${imessage.lastError}` : 'Local bridge active — reading your iMessage database.')
        : 'Enable the iMessage bridge and grant Full Disk Access to read your messages.'
      return personalBridgeConnector(connector, imessagePolicy, detail)
    }
    if (connector.id === 'slack') {
      const status: ConnectorDefinition['status'] = enabledSlackSources.length === 0 ? 'pending' : slackErrors.length === enabledSlackSources.length ? 'error' : 'connected'
      const sourceLabel = enabledSlackSources.length === 0
        ? 'No local Slack export source connected.'
        : `${enabledSlackSources.length} enabled Slack source${enabledSlackSources.length === 1 ? '' : 's'} connected${slackErrors.length ? `, ${slackErrors.length} with errors` : ''}.`
      return {
        ...connector,
        status,
        description: `Read-only local Slack exports for source-backed message reads and summaries. ${sourceLabel}`,
        error: slackErrors[0]?.lastError ?? undefined,
        lastSyncAt: slackSettings.lastSyncAt,
        lastEventAt: slackSettings.lastSyncAt,
        permissionStatus: emailPermissionStatus(enabledSlackSources.length, slackErrors.length),
        permissionDetail: emailPermissionDetail(enabledSlackSources.length, slackErrors, 'Slack source'),
      }
    }
    if (connector.id === 'discord') {
      const discordSources = discord?.sources ?? []
      const enabledDiscord = discordSources.filter((s) => s.enabled)
      const discordErrors = enabledDiscord.filter((s) => s.lastError)
      const status: ConnectorDefinition['status'] = enabledDiscord.length === 0 ? 'pending' : discordErrors.length === enabledDiscord.length ? 'error' : 'connected'
      return {
        ...connector,
        status,
        description: enabledDiscord.length === 0
          ? 'Connect a Discord bot token to read channels/threads and reply.'
          : `${enabledDiscord.length} Discord source${enabledDiscord.length === 1 ? '' : 's'} connected${discordErrors.length ? `, ${discordErrors.length} with errors` : ''}.`,
        error: discordErrors[0]?.lastError ?? undefined,
        lastSyncAt: discord?.lastSyncAt ?? null,
        lastEventAt: discord?.lastSyncAt ?? null,
        permissionStatus: emailPermissionStatus(enabledDiscord.length, discordErrors.length),
        permissionDetail: emailPermissionDetail(enabledDiscord.length, discordErrors, 'Discord source'),
      }
    }
    if (connector.id === 'mattermost') {
      const mmSources = mattermost?.sources ?? []
      const enabledMm = mmSources.filter((s) => s.enabled)
      const mmErrors = enabledMm.filter((s) => s.lastError)
      const status: ConnectorDefinition['status'] = enabledMm.length === 0 ? 'pending' : mmErrors.length === enabledMm.length ? 'error' : 'connected'
      return {
        ...connector,
        status,
        description: enabledMm.length === 0
          ? 'Connect a Mattermost server URL + access token to read channels/threads and reply.'
          : `${enabledMm.length} Mattermost source${enabledMm.length === 1 ? '' : 's'} connected${mmErrors.length ? `, ${mmErrors.length} with errors` : ''}.`,
        error: mmErrors[0]?.lastError ?? undefined,
        lastSyncAt: mattermost?.lastSyncAt ?? null,
        lastEventAt: mattermost?.lastSyncAt ?? null,
        permissionStatus: emailPermissionStatus(enabledMm.length, mmErrors.length),
        permissionDetail: emailPermissionDetail(enabledMm.length, mmErrors, 'Mattermost source'),
      }
    }
    if (connector.id === 'matrix') {
      const matrixSources = matrix?.sources ?? []
      const enabledMatrix = matrixSources.filter((s) => s.enabled)
      const matrixErrors = enabledMatrix.filter((s) => s.lastError)
      const status: ConnectorDefinition['status'] = enabledMatrix.length === 0 ? 'pending' : matrixErrors.length === enabledMatrix.length ? 'error' : 'connected'
      return {
        ...connector,
        status,
        description: enabledMatrix.length === 0
          ? 'Connect a Matrix homeserver URL + access token to read rooms/threads and reply.'
          : `${enabledMatrix.length} Matrix source${enabledMatrix.length === 1 ? '' : 's'} connected${matrixErrors.length ? `, ${matrixErrors.length} with errors` : ''}.`,
        error: matrixErrors[0]?.lastError ?? undefined,
        lastSyncAt: matrix?.lastSyncAt ?? null,
        lastEventAt: matrix?.lastSyncAt ?? null,
        permissionStatus: emailPermissionStatus(enabledMatrix.length, matrixErrors.length),
        permissionDetail: emailPermissionDetail(enabledMatrix.length, matrixErrors, 'Matrix source'),
      }
    }
    if (connector.id === 'googlechat') {
      const gcSources = googlechat?.sources ?? []
      const enabledGc = gcSources.filter((s) => s.enabled)
      const gcErrors = enabledGc.filter((s) => s.lastError)
      const status: ConnectorDefinition['status'] = enabledGc.length === 0 ? 'pending' : gcErrors.length === enabledGc.length ? 'error' : 'connected'
      return {
        ...connector,
        status,
        description: enabledGc.length === 0
          ? 'Add a Google Chat incoming-webhook URL to send (and a service-account token to read).'
          : `${enabledGc.length} Google Chat source${enabledGc.length === 1 ? '' : 's'} connected${gcErrors.length ? `, ${gcErrors.length} with errors` : ''}.`,
        error: gcErrors[0]?.lastError ?? undefined,
        lastSyncAt: googlechat?.lastSyncAt ?? null,
        lastEventAt: googlechat?.lastSyncAt ?? null,
        permissionStatus: emailPermissionStatus(enabledGc.length, gcErrors.length),
        permissionDetail: emailPermissionDetail(enabledGc.length, gcErrors, 'Google Chat source'),
      }
    }
    if (connector.id === 'controller') {
      return {
        ...connector,
        status: controllerSettings?.enabled ? (controllerSettings.lastError ? 'error' : 'connected') : 'pending',
        description: controllerSettings?.enabled
          ? 'Controller can attach foreground app/window context on demand.'
          : 'Enable Controller to attach current window context to chat or code.',
        error: controllerSettings?.lastError ?? undefined,
        lastSyncAt: controllerSettings?.lastCapturedAt ?? null,
        lastEventAt: controllerSettings?.lastCapturedAt ?? null,
        permissionStatus: controllerSettings?.enabled ? (controllerSettings.lastError ? 'denied' : 'granted') : 'not_configured',
        permissionDetail: controllerSettings?.enabled ? (controllerSettings.lastError ?? 'Controller capture is enabled.') : 'Controller is disabled.',
      }
    }
    if (connector.id === 'calendar') {
      const status: ConnectorDefinition['status'] = enabledCalendarSources.length === 0 ? 'pending' : calendarErrors.length === enabledCalendarSources.length ? 'error' : 'connected'
      const sourceLabel = enabledCalendarSources.length === 0
        ? 'No local calendar source connected.'
        : `${enabledCalendarSources.length} local .ics source${enabledCalendarSources.length === 1 ? '' : 's'} connected${calendarErrors.length ? `, ${calendarErrors.length} with errors` : ''}.`
      return {
        ...connector,
        status,
        description: `Local .ics calendar events, event reads, and day summaries. ${sourceLabel}`,
        error: calendarErrors[0]?.lastError ?? undefined,
        lastSyncAt: calendarSettings.lastSyncAt,
        lastEventAt: calendarSettings.lastSyncAt,
        permissionStatus: emailPermissionStatus(enabledCalendarSources.length, calendarErrors.length),
        permissionDetail: emailPermissionDetail(enabledCalendarSources.length, calendarErrors, 'calendar source'),
      }
    }
    if (connector.id === 'docs') {
      const status: ConnectorDefinition['status'] = enabledDocumentSources.length === 0 ? 'pending' : documentErrors.length === enabledDocumentSources.length ? 'error' : 'connected'
      const sourceLabel = enabledDocumentSources.length === 0
        ? 'No local document source connected.'
        : `${enabledDocumentSources.length} local document/PDF/table source${enabledDocumentSources.length === 1 ? '' : 's'} connected${documentErrors.length ? `, ${documentErrors.length} with errors` : ''}.`
      return {
        ...connector,
        status,
        description: `Local documents, PDFs, and tables with source-backed reads and citation-aware summaries. ${sourceLabel}`,
        error: documentErrors[0]?.lastError ?? undefined,
        lastSyncAt: documentSettings.lastSyncAt,
        lastEventAt: documentSettings.lastSyncAt,
        permissionStatus: emailPermissionStatus(enabledDocumentSources.length, documentErrors.length),
        permissionDetail: emailPermissionDetail(enabledDocumentSources.length, documentErrors, 'document source'),
      }
    }
    if (connector.id === 'browser') {
      const status: ConnectorDefinition['status'] = enabledBrowserSources.length === 0 ? 'pending' : browserErrors.length === enabledBrowserSources.length ? 'error' : 'connected'
      const sourceLabel = enabledBrowserSources.length === 0
        ? 'No explicit browser source connected.'
        : `${enabledBrowserSources.length} enabled browser source${enabledBrowserSources.length === 1 ? '' : 's'} connected${browserErrors.length ? `, ${browserErrors.length} with errors` : ''}.`
      return {
        ...connector,
        status,
        description: `Explicit current-tab snapshots and user-provided history exports for source-backed browser reads and summaries. ${sourceLabel}`,
        error: browserErrors[0]?.lastError ?? undefined,
        lastSyncAt: browserSettings.lastSyncAt,
        lastEventAt: browserSettings.lastSyncAt,
        permissionStatus: emailPermissionStatus(enabledBrowserSources.length, browserErrors.length),
        permissionDetail: emailPermissionDetail(enabledBrowserSources.length, browserErrors, 'browser source'),
      }
    }
    if (connector.id !== 'email') return connector
    const status: ConnectorDefinition['status'] = enabledSources.length === 0 ? 'pending' : sourceErrors.length === enabledSources.length ? 'error' : 'connected'
    const sourceLabel = enabledSources.length === 0
      ? 'No local email source connected.'
      : `${enabledSources.length} local source${enabledSources.length === 1 ? '' : 's'} connected${sourceErrors.length ? `, ${sourceErrors.length} with errors` : ''}.`
    return {
      ...connector,
      status,
      description: `Local .eml/Maildir inbox triage, summaries, drafts, and approval-gated send requests. ${sourceLabel}`,
      error: sourceErrors[0]?.lastError ?? undefined,
      lastSyncAt: emailSettings.lastSyncAt,
      lastEventAt: emailSettings.lastSyncAt,
      permissionStatus: emailPermissionStatus(enabledSources.length, sourceErrors.length),
      permissionDetail: emailPermissionDetail(enabledSources.length, sourceErrors),
    }
  })
}

function tool(name: string, description: string, approvalRequired = false): ConnectorToolSchema {
  return { name, description, approvalRequired, parameters: { type: 'object', properties: {} } }
}

function budget(executionClass: RunBudgetDefinition['executionClass'], label: string, maxWallSeconds: number, heartbeatIntervalSeconds: number, stallGraceSeconds: number, requiresHeartbeat: boolean): RunBudgetDefinition {
  return { executionClass, label, maxWallSeconds, heartbeatIntervalSeconds, stallGraceSeconds, requiresHeartbeat }
}

function permissionFromStatus(status: ConnectorDefinition['status']): ConnectorPermissionStatus {
  if (status === 'connected' || status === 'mock') return 'granted'
  if (status === 'error') return 'denied'
  return 'not_configured'
}

function emailPermissionStatus(enabledSources: number, sourceErrors: number): ConnectorPermissionStatus {
  if (enabledSources === 0) return 'not_configured'
  if (sourceErrors === 0) return 'granted'
  return sourceErrors === enabledSources ? 'denied' : 'degraded'
}

function emailPermissionDetail(enabledSources: number, sourceErrors: Array<{ lastError?: string | null }>, label = 'email source'): string {
  if (enabledSources === 0) return `No enabled ${label}s.`
  if (sourceErrors.length === 0) return `${enabledSources} enabled ${label}${enabledSources === 1 ? '' : 's'} available.`
  return `${sourceErrors.length}/${enabledSources} ${label}${enabledSources === 1 ? '' : 's'} failing: ${sourceErrors[0].lastError ?? 'unknown error'}`
}

function event(input: Partial<ConnectorEvent> & Pick<ConnectorEvent, 'connectorId' | 'action' | 'title' | 'summary' | 'actor' | 'threadKey' | 'createdAt' | 'priority' | 'trustLevel'>): ConnectorEvent {
  const requiresApproval = input.requiresApproval ?? input.trustLevel === 'approve'
  const id = input.id ?? randomUUID()
  const pluginName = input.pluginName ?? BASE_CONNECTORS.find((c) => c.id === input.connectorId)?.name ?? input.connectorId
  const next: ConnectorEvent = {
    id,
    connectorId: input.connectorId,
    pluginName,
    action: input.action,
    direction: input.direction ?? 'inbound',
    classification: input.classification ?? classify(input.connectorId, input.action),
    priority: input.priority,
    trustLevel: input.trustLevel,
    requiresApproval,
    approvalId: input.approvalId ?? (requiresApproval ? `approval-${randomUUID().slice(0, 8)}` : null),
    approvalStatus: input.approvalStatus ?? (requiresApproval ? 'pending' : null),
    resolvedAt: input.resolvedAt ?? null,
    resolvedBy: input.resolvedBy ?? null,
    title: input.title,
    summary: input.summary,
    actor: input.actor,
    threadKey: input.threadKey,
    payloadKeys: input.payloadKeys ?? [],
    trace: input.trace,
    createdAt: input.createdAt,
  }
  next.trace ??= traceForConnectorEvent(next)
  return next
}

function classify(connectorId: ConnectorId, action: string): ConnectorEvent['classification'] {
  if (action.includes('approval') || action.includes('send') || action.includes('delete')) return 'approval'
  if (connectorId === 'docs') return 'document'
  if (connectorId === 'browser') return 'document'
  if (connectorId === 'calendar') return 'schedule'
  if (connectorId === 'controller') return 'system'
  if (['email', 'slack', 'discord', 'mattermost', 'matrix', 'googlechat', 'whatsapp', 'imessage', 'telegram', 'webhook'].includes(connectorId)) return 'communication'
  return 'unknown'
}

function trust(action: string, direction: ConnectorEvent['direction']): ConnectorTrustLevel {
  if (direction === 'outbound' || /send|delete|create|update/i.test(action)) return 'approve'
  if (/summary|summarise|classify|index/i.test(action)) return 'auto_logged'
  if (/received|message|document/i.test(action)) return 'suggest'
  return 'notify'
}

function priorityFor(action: string, trustLevel: ConnectorTrustLevel): ConnectorPriority {
  if (trustLevel === 'approve') return 'high'
  if (/urgent|overdue|security|meeting/i.test(action)) return 'critical'
  if (/received|changed|risk/i.test(action)) return 'normal'
  return 'low'
}

function titleFrom(input: ConnectorInteractionInput, fallback: string) {
  const payload = { ...(input.input ?? {}), ...(input.output ?? {}) }
  return String(payload.title || payload.subject || payload.summary || `${fallback} ${input.action.replace(/_/g, ' ')}`)
}

function summaryFrom(input: ConnectorInteractionInput) {
  const payload = { ...(input.input ?? {}), ...(input.output ?? {}) }
  return String(payload.summary || payload.body || payload.text || payload.message || `Connector action ${input.action} was recorded.`).slice(0, 320)
}

function safeKeys(payload: Record<string, unknown>) {
  const sensitive = /api|auth|bearer|cookie|email|jid|key|password|phone|secret|session|token|url|user/i
  return Object.keys(payload).filter((key) => !sensitive.test(key)).slice(0, 20)
}

function detectGaps(connectors: ConnectorDefinition[], events: ConnectorEvent[]): ControlPlaneGap[] {
  const gaps: ControlPlaneGap[] = []
  for (const connector of connectors) {
    if (connector.status === 'pending') {
      gaps.push({ id: `connector:${connector.id}:not_connected`, severity: 'warning', title: `${connector.name} is not connected`, detail: 'The connector contract exists, but live ingress and transport are not configured yet.', subject: connector.id })
    }
    if (connector.kind === 'communication' && !connector.transport) {
      gaps.push({ id: `connector:${connector.id}:missing_transport`, severity: 'error', title: `${connector.name} has no transport`, detail: 'Outbound replies need a registered transport before Assistant can act.', subject: connector.id })
    }
  }
  if (!events.length) {
    gaps.push({ id: 'events:empty', severity: 'info', title: 'No connector events recorded', detail: 'The Assistant event spine is ready but has no recorded inbound events.', subject: 'events' })
  }
  return gaps.sort((a, b) => a.id.localeCompare(b.id))
}
