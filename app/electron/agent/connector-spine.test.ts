import assert from 'node:assert/strict'
import { buildAssistantControlSnapshot, recordConnectorInteraction } from './connector-spine'
import type { BotControlSettings, BrowserConnectorSettings, CalendarConnectorSettings, ConnectorDefinition, ConnectorId, ControllerSettings, DocumentConnectorSettings, EmailConnectorSettings, SlackConnectorSettings } from './types'

function connector(snapshot: ReturnType<typeof buildAssistantControlSnapshot>, id: ConnectorId): ConnectorDefinition {
  const found = snapshot.connectors.find((item) => item.id === id)
  assert.ok(found, `expected connector ${id}`)
  return found
}

function emailSettings(sources: EmailConnectorSettings['sources'], lastSyncAt: number | null): EmailConnectorSettings {
  return { sources, maxMessages: 50, lastSyncAt }
}

function emailSource(overrides: Partial<EmailConnectorSettings['sources'][number]> = {}): EmailConnectorSettings['sources'][number] {
  return {
    id: 'mail',
    label: 'Mail',
    path: '/tmp/mail',
    kind: 'eml_dir',
    enabled: true,
    createdAt: 1,
    updatedAt: 1,
    lastSyncAt: null,
    lastError: null,
    ...overrides,
  }
}

function slackSettings(sources: SlackConnectorSettings['sources'], lastSyncAt: number | null): SlackConnectorSettings {
  return { sources, maxMessages: 50, lastSyncAt }
}

function slackSource(overrides: Partial<SlackConnectorSettings['sources'][number]> = {}): SlackConnectorSettings['sources'][number] {
  return {
    id: 'slack',
    label: 'Slack Export',
    path: '/tmp/slack-export',
    kind: 'export_dir',
    enabled: true,
    createdAt: 1,
    updatedAt: 1,
    lastSyncAt: null,
    lastError: null,
    ...overrides,
  }
}

function calendarSettings(sources: CalendarConnectorSettings['sources'], lastSyncAt: number | null): CalendarConnectorSettings {
  return { sources, maxEvents: 100, lastSyncAt }
}

function calendarSource(overrides: Partial<CalendarConnectorSettings['sources'][number]> = {}): CalendarConnectorSettings['sources'][number] {
  return {
    id: 'cal',
    label: 'Calendar',
    path: '/tmp/calendar.ics',
    kind: 'ics_file',
    enabled: true,
    createdAt: 1,
    updatedAt: 1,
    lastSyncAt: null,
    lastError: null,
    ...overrides,
  }
}

function documentSettings(sources: DocumentConnectorSettings['sources'], lastSyncAt: number | null): DocumentConnectorSettings {
  return { sources, maxDocuments: 100, maxDocumentBytes: 1_000_000, lastSyncAt }
}

function documentSource(overrides: Partial<DocumentConnectorSettings['sources'][number]> = {}): DocumentConnectorSettings['sources'][number] {
  return {
    id: 'docs',
    label: 'Project Docs',
    path: '/tmp/docs',
    kind: 'dir',
    enabled: true,
    createdAt: 1,
    updatedAt: 1,
    lastSyncAt: null,
    lastError: null,
    ...overrides,
  }
}

function browserSettings(sources: BrowserConnectorSettings['sources'], lastSyncAt: number | null): BrowserConnectorSettings {
  return { sources, maxItems: 100, lastSyncAt }
}

function browserSource(overrides: Partial<BrowserConnectorSettings['sources'][number]> = {}): BrowserConnectorSettings['sources'][number] {
  return {
    id: 'browser',
    label: 'Current Tab',
    kind: 'current_tab',
    enabled: true,
    createdAt: 1,
    updatedAt: 1,
    lastSyncAt: null,
    lastError: null,
    url: 'https://example.com/current',
    title: 'Current Tab',
    excerpt: 'Explicitly saved browser context.',
    capturedAt: 1,
    ...overrides,
  }
}

function botSettings(overrides: Partial<BotControlSettings> = {}): BotControlSettings {
  return {
    telegram: {
      enabled: false,
      botToken: '',
      allowedChatIds: '',
      allowedUserIds: '',
      polling: false,
      lastUpdateId: null,
      lastPollAt: null,
      lastError: null,
    },
    whatsapp: {
      enabled: false,
      accessToken: '',
      phoneNumberId: '',
      verifyToken: '',
      appSecret: '',
      allowedChatIds: '',
      lastEventAt: null,
      lastError: null,
    },
    ...overrides,
  }
}

function controllerSettings(overrides: Partial<ControllerSettings> = {}): ControllerSettings {
  return {
    enabled: false,
    includeAccessibilityText: false,
    maxTextChars: 24_000,
    lastCapturedAt: null,
    lastError: null,
    ...overrides,
  }
}

{
  const syncAt = 1_234_000
  const snapshot = buildAssistantControlSnapshot([], emailSettings([emailSource({ lastSyncAt: syncAt })], syncAt))
  const email = connector(snapshot, 'email')
  assert.equal(email.status, 'connected')
  assert.equal(email.lastSyncAt, syncAt)
  assert.equal(email.lastEventAt, syncAt)
  assert.equal(email.permissionStatus, 'granted')
  assert.match(email.permissionDetail, /1 enabled email source/)
}

{
  const snapshot = buildAssistantControlSnapshot([], emailSettings([
    emailSource({ id: 'personal', label: 'Personal', lastError: 'mailbox locked' }),
    emailSource({ id: 'work', label: 'Work', lastError: null }),
  ], 2_000))
  const email = connector(snapshot, 'email')
  assert.equal(email.status, 'connected')
  assert.equal(email.permissionStatus, 'degraded')
  assert.equal(email.error, 'mailbox locked')
  assert.match(email.permissionDetail, /1\/2 email sources failing: mailbox locked/)
}

{
  const snapshot = buildAssistantControlSnapshot([], emailSettings([
    emailSource({ id: 'personal', label: 'Personal', lastError: 'not authorized' }),
    emailSource({ id: 'work', label: 'Work', lastError: 'maildir missing' }),
  ], 3_000))
  const email = connector(snapshot, 'email')
  assert.equal(email.status, 'error')
  assert.equal(email.permissionStatus, 'denied')
  assert.equal(email.error, 'not authorized')
}

{
  const syncAt = 3_250
  const snapshot = buildAssistantControlSnapshot([], undefined, undefined, undefined, undefined, undefined, slackSettings([slackSource({ lastSyncAt: syncAt })], syncAt))
  const slack = connector(snapshot, 'slack')
  assert.equal(slack.status, 'connected')
  assert.equal(slack.lastSyncAt, syncAt)
  assert.equal(slack.permissionStatus, 'granted')
  assert.match(slack.permissionDetail, /1 enabled Slack source/)
  assert.ok(slack.tools.some((tool) => tool.name === 'message_summary'))
  assert.ok(!slack.tools.some((tool) => tool.name === 'message_compose'))
}

{
  const syncAt = 3_500
  const snapshot = buildAssistantControlSnapshot([], undefined, undefined, undefined, calendarSettings([calendarSource({ lastSyncAt: syncAt })], syncAt))
  const calendar = connector(snapshot, 'calendar')
  assert.equal(calendar.status, 'connected')
  assert.equal(calendar.lastSyncAt, syncAt)
  assert.equal(calendar.permissionStatus, 'granted')
  assert.match(calendar.permissionDetail, /1 enabled calendar source/)
  assert.ok(calendar.tools.some((tool) => tool.name === 'calendar_summary'))
  assert.ok(calendar.tools.some((tool) => tool.name === 'calendar_update' && tool.approvalRequired))
}

{
  const snapshot = buildAssistantControlSnapshot([], undefined, undefined, undefined, calendarSettings([
    calendarSource({ id: 'personal', label: 'Personal', lastError: 'ics missing' }),
    calendarSource({ id: 'work', label: 'Work', lastError: null }),
  ], 3_600))
  const calendar = connector(snapshot, 'calendar')
  assert.equal(calendar.status, 'connected')
  assert.equal(calendar.permissionStatus, 'degraded')
  assert.equal(calendar.error, 'ics missing')
  assert.match(calendar.permissionDetail, /1\/2 calendar sources failing: ics missing/)
}

{
  const syncAt = 4_000
  const bots = botSettings({
    telegram: {
      enabled: true,
      botToken: 'token',
      allowedChatIds: '42',
      allowedUserIds: '7',
      polling: true,
      lastUpdateId: 7,
      lastPollAt: syncAt,
      lastError: null,
    },
  })
  const snapshot = buildAssistantControlSnapshot([], undefined, bots)
  const telegram = connector(snapshot, 'telegram')
  assert.equal(telegram.status, 'connected')
  assert.equal(telegram.lastSyncAt, syncAt)
  assert.equal(telegram.permissionStatus, 'granted')
  assert.match(telegram.permissionDetail, /1 allowed Telegram chat/)
  assert.ok(telegram.tools.some((tool) => tool.name === 'message_compose' && tool.approvalRequired))
}

{
  const bots = botSettings({
    telegram: {
      enabled: true,
      botToken: 'token',
      allowedChatIds: '',
      allowedUserIds: '7',
      polling: true,
      lastUpdateId: null,
      lastPollAt: null,
      lastError: null,
    },
  })
  const snapshot = buildAssistantControlSnapshot([], undefined, bots)
  const telegram = connector(snapshot, 'telegram')
  assert.equal(telegram.status, 'pending')
  assert.equal(telegram.permissionStatus, 'not_configured')
  assert.match(telegram.permissionDetail, /allowed Telegram chat IDs/)
  assert.equal(telegram.tools.length, 0)
}

{
  const bots = botSettings({
    whatsapp: {
      enabled: true,
      accessToken: 'token',
      phoneNumberId: 'phone',
      verifyToken: 'verify',
      appSecret: 'app-secret',
      allowedChatIds: '42',
      lastEventAt: 5_000,
      lastError: 'webhook signature failed',
    },
  })
  const snapshot = buildAssistantControlSnapshot([], undefined, bots)
  const whatsapp = connector(snapshot, 'whatsapp')
  assert.equal(whatsapp.status, 'error')
  assert.equal(whatsapp.lastSyncAt, 5_000)
  assert.equal(whatsapp.permissionStatus, 'degraded')
  assert.equal(whatsapp.error, 'webhook signature failed')
  assert.equal(whatsapp.tools.length, 0, 'Cloud bot-control ingress must not advertise the separate linked-device send adapter')
}

{
  const snapshot = buildAssistantControlSnapshot([])
  const imessage = connector(snapshot, 'imessage')
  assert.equal(imessage.status, 'pending')
  assert.equal(imessage.permissionStatus, 'not_configured')
  assert.match(imessage.permissionDetail, /explicit opt-in/i)
  assert.equal(imessage.tools.length, 0)
}

{
  const disabledSnapshot = buildAssistantControlSnapshot([], undefined, undefined, controllerSettings())
  const disabledController = connector(disabledSnapshot, 'controller')
  assert.equal(disabledController.status, 'pending')
  assert.equal(disabledController.permissionStatus, 'not_configured')
  assert.match(disabledController.permissionDetail, /disabled/)

  const deniedSnapshot = buildAssistantControlSnapshot([], undefined, undefined, controllerSettings({
    enabled: true,
    lastCapturedAt: 6_000,
    lastError: 'Accessibility permission denied',
  }))
  const deniedController = connector(deniedSnapshot, 'controller')
  assert.equal(deniedController.status, 'error')
  assert.equal(deniedController.lastSyncAt, 6_000)
  assert.equal(deniedController.permissionStatus, 'denied')
  assert.equal(deniedController.error, 'Accessibility permission denied')
}

{
  const event = recordConnectorInteraction({
    connectorId: 'docs',
    action: 'document_indexed',
    output: { path: '/tmp/contract.pdf' },
  })
  const snapshot = buildAssistantControlSnapshot([event])
  const docs = connector(snapshot, 'docs')
  assert.equal(docs.lastSyncAt, null)
  assert.equal(docs.lastEventAt, event.createdAt)
  assert.equal(docs.unread, 1)
}

{
  const syncAt = 4_500
  const snapshot = buildAssistantControlSnapshot([], undefined, undefined, undefined, undefined, documentSettings([documentSource({ lastSyncAt: syncAt })], syncAt))
  const docs = connector(snapshot, 'docs')
  assert.equal(docs.status, 'connected')
  assert.equal(docs.lastSyncAt, syncAt)
  assert.equal(docs.permissionStatus, 'granted')
  assert.match(docs.permissionDetail, /1 enabled document source/)
  assert.ok(docs.tools.some((tool) => tool.name === 'document_summary'))
  assert.ok(docs.tools.some((tool) => tool.name === 'table_summary'))
}

{
  const syncAt = 4_750
  const snapshot = buildAssistantControlSnapshot([], undefined, undefined, undefined, undefined, undefined, undefined, browserSettings([browserSource({ lastSyncAt: syncAt })], syncAt))
  const browser = connector(snapshot, 'browser')
  assert.equal(browser.status, 'connected')
  assert.equal(browser.lastSyncAt, syncAt)
  assert.equal(browser.permissionStatus, 'granted')
  assert.match(browser.permissionDetail, /1 enabled browser source/)
  assert.match(browser.description, /explicit/i)
  assert.ok(browser.tools.some((tool) => tool.name === 'browser_summary'))
  assert.ok(!browser.tools.some((tool) => tool.name.includes('submit')))
}

console.log('connector spine tests passed')
