import { createSerialTaskQueue } from '../../agent/serial-task-queue'
import { discordEventFromMessage } from '../../agent/discord-connector'
import { advanceSyncCursor, isNewerThanCursor } from '../../agent/connector-sync'
import { requireNativeConfirmation, type ConfirmSensitiveAction } from './sensitive-confirmation'
import { createConnectorEventSpine, connectorPolicyForEvent, resolveSavedConnectorApprovalPolicy } from './connector-event-spine'
import { createEmailTransport } from './connector-email-transport'
import { createConnectorChannels } from './connector-channels'
import { registerAllConnectorIpc } from './connector-ipc'
import { resumeSavedWhatsAppAccounts } from './connector-ipc/whatsapp'
import type { ApprovalPolicyScope } from '../../agent/approval-policy'
import type { ChannelApprovalRequest } from '../../agent/channel-adapter'
import type { ConnectorApprovalDecision } from '../../agent/types'
import type {
  ConnectorEvent,
  EmailDraftInput,
  SyncCursor,
} from '../../agent/types'

// Dedup Discord sync output by stable message id. fetchDiscordMessages always
// returns the most recent ~50 messages per channel, and discordEventFromMessage
// assigns a fresh UUID each time, so without this every poll re-creates the same
// messages as new events (the threadKey is per-channel and cannot dedup messages).
// We stamp the source message id into the event payload so later syncs recognise it.
//
// INCREMENTAL: when a per-source cursor is supplied, skip anything not newer than it
// (position = message date in ms — snowflakes exceed 2^53 so we sort by the reliable
// timestamp and keep the snowflake as the boundary id) BEFORE the id-set dedup, which
// stays as the exact-correctness fallback. The advanced cursor is returned to persist.
export function dedupeDiscordEvents(
  messages: { id: string; date?: number }[],
  existingEvents: ConnectorEvent[],
  cursor?: SyncCursor | null,
): { events: ConnectorEvent[]; cursor: SyncCursor | undefined } {
  const known = new Set(
    existingEvents
      .filter((e) => e.connectorId === 'discord')
      .map((e) => (typeof e.payload?.messageId === 'string' ? (e.payload.messageId as string) : null))
      .filter((id): id is string => Boolean(id)),
  )
  const events: ConnectorEvent[] = []
  for (const m of messages) {
    if (!isNewerThanCursor({ position: m.date ?? null, id: m.id }, cursor)) continue
    if (known.has(m.id)) continue
    known.add(m.id)
    const event = discordEventFromMessage(m as Parameters<typeof discordEventFromMessage>[0])
    event.payload = { ...(event.payload ?? {}), messageId: m.id }
    if (!event.payloadKeys.includes('messageId')) event.payloadKeys = [...event.payloadKeys, 'messageId']
    events.push(event)
  }
  const advanced = advanceSyncCursor(cursor, messages.map((m) => ({ position: m.date ?? null, id: m.id })), Date.now()) ?? undefined
  return { events, cursor: advanced }
}

export async function reconcileTelegramDraft(
  draftMessageId: string | null | undefined,
  firstChunk: string,
  ops: {
    edit(messageId: string, text: string): Promise<boolean>
    remove(messageId: string): Promise<boolean>
    send(text: string): Promise<string | null>
  },
): Promise<{ sentId: string | null; error?: string }> {
  if (!draftMessageId) return { sentId: await ops.send(firstChunk) }
  if (await ops.edit(draftMessageId, firstChunk)) return { sentId: draftMessageId }

  // Never post a fresh final beside an unreconciled "…" placeholder. Delete the failed
  // draft first; if deletion also fails, surface the failure and leave one visible message
  // instead of manufacturing the confusing placeholder+duplicate pair.
  if (!await ops.remove(draftMessageId)) {
    return { sentId: null, error: 'Telegram draft could not be reconciled after its final edit failed; no duplicate final message was posted.' }
  }
  return { sentId: await ops.send(firstChunk) }
}

export interface ConnectorRuntimeCallbacks {
  findApprovalRule: (scope: 'tool' | 'connector', key: string, context?: { projectId?: string | null; workspace?: string | null }) => { decision: 'allow' | 'deny' } | null | undefined | Promise<{ decision: 'allow' | 'deny' } | null | undefined>
  rememberApprovalRule: (
    scope: 'tool' | 'connector',
    key: string,
    decision: 'allow' | 'deny',
    duration?: 'always',
    meta?: { label?: string; explanation?: string; legacyTool?: string; projectId?: string | null; workspace?: string | null },
  ) => Promise<unknown | null>
  remotePublish: (event: ConnectorEvent) => void
  emitOutboundWebhook: (event: ConnectorEvent) => void
  confirmSensitiveAction?: ConfirmSensitiveAction
  reserveOperation?: () => { release(): void }
}

export async function removeWhatsAppSourceWithNativeConfirmation(
  source: { id: string; label?: string | null },
  confirm: ConfirmSensitiveAction | undefined,
  operations: {
    stop(id: string): void
    shred(id: string): Promise<void>
    remove(id: string): Promise<void>
  },
): Promise<void> {
  await requireNativeConfirmation(confirm, {
    title: 'Remove WhatsApp account?',
    message: `Remove ${source.label?.trim() || 'this WhatsApp account'} and its linked-device credentials?`,
    detail: 'This permanently shreds the locally stored WhatsApp authentication state. Pair the device again to reconnect it.',
    confirmLabel: 'Remove account',
  })
  operations.stop(source.id)
  await operations.shred(source.id)
  await operations.remove(source.id)
}

export interface ConnectorRuntime {
  resumeWhatsAppAccounts(): Promise<void>
  connectorPolicyForEvent: (event: ConnectorEvent) => ApprovalPolicyScope
  rememberConnectorEvents: (events: ConnectorEvent[]) => Promise<void>
  recordEmailDraft: (input: EmailDraftInput, sources?: ConnectorEvent['sources']) => Promise<ConnectorEvent>
  recordEmailEvent: (event: ConnectorEvent) => Promise<ConnectorEvent>
  recordCalendarEvent: (event: ConnectorEvent) => Promise<ConnectorEvent>
  recordMessageEvent: (event: ConnectorEvent) => Promise<ConnectorEvent>
  registerConnectorIpc: () => void
  // In-channel approval gate (Feature A): prompt in the channel and block until answered;
  // resolvers let inbound button callbacks / "/approve" replies unblock a parked request.
  requestChannelApproval: (threadKey: string, request: ChannelApprovalRequest) => Promise<import('../../agent/channel-adapter').ChannelApprovalResult>
  resolveChannelApproval: (token: string, decision: ConnectorApprovalDecision) => boolean
  resolveChannelApprovalByThread: (threadKey: string, decision: ConnectorApprovalDecision) => boolean
}

// The shape returned by createSerialTaskQueue(); declared here so the IPC module can type its
// `syncQueue` dependency without importing the (value-returning) factory into a type position.
export interface SerialTaskQueue {
  run<T>(key: string, task: () => Promise<T>): Promise<T>
  pendingKeys(): number
}

export function createConnectorRuntime(callbacks: ConnectorRuntimeCallbacks): ConnectorRuntime {
  // Sync calls can be triggered by polling and UI actions at the same time. Serialize each
  // connector/source independently; different connectors still run in parallel.
  const connectorSyncQueue: SerialTaskQueue = createSerialTaskQueue()

  const spine = createConnectorEventSpine({ callbacks, syncQueue: connectorSyncQueue })
  const emailTransport = createEmailTransport({ spine })
  const channels = createConnectorChannels({ spine })

  function registerConnectorIpc(): void {
    registerAllConnectorIpc({ spine, emailTransport, channels, syncQueue: connectorSyncQueue, callbacks })
  }

  return {
    connectorPolicyForEvent,
    rememberConnectorEvents: spine.rememberConnectorEvents,
    recordEmailDraft: emailTransport.recordEmailDraft,
    recordEmailEvent: emailTransport.recordEmailEvent,
    recordCalendarEvent: spine.recordCalendarEvent,
    recordMessageEvent: spine.recordMessageEvent,
    registerConnectorIpc,
    resumeWhatsAppAccounts: () => resumeSavedWhatsAppAccounts({ channels }),
    requestChannelApproval: channels.requestChannelApproval,
    resolveChannelApproval: channels.resolveChannelApproval,
    resolveChannelApprovalByThread: channels.resolveChannelApprovalByThread,
  }
}

// Re-export the pure helpers from their new homes so external consumers
// (main-app.ts, connector-runtime.test.ts, whatsapp-source-removal.test.ts) keep importing
// them from './connector-runtime' unchanged.
export { connectorPolicyForEvent, resolveSavedConnectorApprovalPolicy } from './connector-event-spine'
