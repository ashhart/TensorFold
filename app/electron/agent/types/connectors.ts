import type { Job, Mission } from './jobs'
import type { Thread, TraceContext } from './timeline'
import type { SecretRotationMetadata } from './base'
import type {
  Automation,
  Project,
  ProjectMemoryConfidence,
  ProjectMemoryKind,
  ProjectMemoryPrivacyScope,
  Room,
  Ticket,
} from './workspace'

// ---- Assistant connector spine (Zora-inspired local contracts) ----
export type ConnectorId = 'email' | 'slack' | 'discord' | 'mattermost' | 'matrix' | 'googlechat' | 'whatsapp' | 'imessage' | 'docs' | 'calendar' | 'telegram' | 'controller' | 'webhook' | 'browser'

// Inbound/outbound webhooks (build.md Phase 2), inbound-first with a loopback signed listener.
export type WebhookDirection = 'inbound' | 'outbound'
export type WebhookSignatureMode = 'hmac-sha256'
export type WebhookCapturePolicy = 'metadata' | 'redacted_payload' | 'ttl_payload'
export type WebhookTriggerPolicy = 'notify' | 'suggest' | 'approval' | 'run_automation'
export type WebhookEventKind = 'agent' | 'room' | 'ticket' | 'automation' | 'mission' | 'connector' | 'test'
export interface WebhookDeliveryRecord {
  id: string
  direction: WebhookDirection
  eventKind: WebhookEventKind
  urlOrPath: string
  status: 'queued' | 'delivered' | 'failed' | 'deduped'
  attempts: number
  httpStatus: number | null
  error: string | null
  idempotencyKey: string
  payloadPreview: string
  createdAt: number
  finishedAt: number | null
}
export interface WebhookEndpoint {
  id: string
  name: string
  direction: WebhookDirection
  enabled: boolean
  scope: { projectId?: string | null; workspaceId?: string | null }
  url?: string // outbound only
  path?: string // inbound route, e.g. '/hooks/github'
  secretRef: string // HMAC secret (safeStorage-protected at rest, masked to the renderer)
  signatureMode: WebhookSignatureMode
  replayWindowSeconds: number
  capturePolicy: WebhookCapturePolicy
  triggerPolicy: WebhookTriggerPolicy
  runAutomationId?: string | null // which automation 'run_automation' runs (never built from payload)
  eventKinds?: WebhookEventKind[] // outbound filter; empty means all event kinds
  approvalRequired?: boolean
  approvedAt?: number | null
  deliveries?: WebhookDeliveryRecord[]
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}
export interface WebhookServerStatus {
  listening: boolean
  port: number | null
  routes: number
  lastDeliveryAt: number | null
}
export interface WebhookSnapshot {
  endpoints: WebhookEndpoint[]
  listener: WebhookServerStatus
}
// ---- Incremental sync cursor (per source) ----
// High-water mark of the newest item a source has already ingested, so a sync only
// fetches/emits items NEWER than the cursor instead of re-scanning all history each
// poll. `position` is the monotonic sort key (IMAP UID, slack/imessage message ts in
// ms, discord snowflake as a number); `lastId` is the stable id of the item at that
// position (boundary tie-break). `lastImapUid` is the IMAP-specific UID floor used to
// fetch only UID > floor. A missing/null cursor means a FULL resync (the fallback).
export interface SyncCursor {
  position: number
  lastId: string | null
  lastImapUid?: number
  updatedAt: number
}

// ---- Reply-to / quoted context on an inbound message ----
// When an inbound message is a reply, the connector resolves the quoted message's
// author + text so the agent's context shows what is being replied to. Bounded/clipped
// upstream (makeReplyContext) before it ever reaches an event.
export interface ConnectorReplyContext {
  id: string | null
  author: string | null
  text: string | null
}

// ---- Per-channel ephemeral system-prompt override ----
// A source/channel can carry a `channelPrompt` that is injected into the agent's context
// ONLY for messages originating from that channel (resolved by selectChannelPrompt in
// channel-prompt.ts). It is an EPHEMERAL system message — it is never written into
// thread.apiMessages nor folded into the persisted system-prompt snapshot, so switching
// channels mid-thread can't leak one channel's instructions into another's history.
export interface ConnectorChannelPrompt {
  connectorId: ConnectorId
  sourceId: string | null
  channelPrompt?: string
}

// ---- In-channel approval (Feature A) ----
// A connector-action approval surfaced IN the channel (Slack/Discord/Telegram inline
// controls; plain-text "/approve" for the rest), mirroring the in-app approval gate.
export interface ChannelApprovalRequest {
  title: string
  summary: string
  command?: string | null
}
// Plain-text channels can't block synchronously, so they post the prompt and return
// fallbackToApp: the in-app gate stays authoritative for the actual decision.
export type ChannelApprovalResult =
  | { ok: true; decision: ConnectorApprovalDecision }
  | { ok: false; error?: string; fallbackToApp?: boolean }

// ---- Per-platform outbound message length limits (Feature: message split) ----
// Each transport caps a single outbound message; a longer reply must be split into
// multiple messages. Limits are measured in UTF-16 code units (String.length / what the
// platform actually counts), NOT code points — Telegram counts UTF-16 units explicitly,
// and the others are byte/char caps we keep on the safe side of. `null` means no practical
// cap (email), so the reply is sent whole. Conservative on purpose: better one extra split
// than a rejected send. Used by splitForPlatform + the channel adapter's multi-send path.
export interface MessageLengthLimit {
  // Hard cap, in UTF-16 code units, of a single outbound message for this platform.
  // null = effectively unlimited (sent as one message).
  maxUnits: number | null
}
export type MessageLengthLimits = Record<string, MessageLengthLimit>

export type ConnectorStatus = 'connected' | 'mock' | 'pending' | 'error'
export type ConnectorPermissionStatus = 'granted' | 'degraded' | 'denied' | 'not_configured' | 'unknown'
export type ConnectorDirection = 'inbound' | 'outbound' | 'internal'
export type ConnectorPriority = 'critical' | 'high' | 'normal' | 'low'
export type ConnectorTrustLevel = 'notify' | 'suggest' | 'approve' | 'auto_logged' | 'auto_silent'
export type ConnectorClassification = 'communication' | 'document' | 'schedule' | 'approval' | 'system' | 'unknown'
export type ConnectorApprovalStatus = 'pending' | 'approved' | 'denied'
export type ConnectorApprovalDecision = 'approve' | 'deny'
export type RunBudgetClass = 'interactive' | 'backlog' | 'background' | 'mission' | 'overnight' | 'unbounded_with_heartbeat'

export interface ConnectorToolSchema {
  name: string
  description: string
  approvalRequired?: boolean
  parameters: Record<string, unknown>
}

export interface ConnectorDefinition {
  id: ConnectorId
  name: string
  kind: 'communication' | 'document' | 'calendar' | 'system'
  status: ConnectorStatus
  description: string
  transport: string | null
  tools: ConnectorToolSchema[]
  unread: number
  pendingApprovals: number
  lastSyncAt: number | null
  lastEventAt: number | null
  permissionStatus: ConnectorPermissionStatus
  permissionDetail: string
  error?: string
}

export interface ConnectorEvent {
  id: string
  connectorId: ConnectorId
  pluginName: string
  action: string
  direction: ConnectorDirection
  classification: ConnectorClassification
  priority: ConnectorPriority
  trustLevel: ConnectorTrustLevel
  requiresApproval: boolean
  approvalId: string | null
  approvalStatus?: ConnectorApprovalStatus | null
  resolvedAt?: number | null
  resolvedBy?: string | null
  title: string
  summary: string
  actor: string
  threadKey: string
  payloadKeys: string[]
  payload?: Record<string, unknown>
  // Quoted/replied-to context for an inbound reply. Captured at ingest from the
  // message's reply identifier (email In-Reply-To, slack thread_ts, discord
  // referenced_message) and also folded into `summary` so the agent's context shows it.
  reply?: ConnectorReplyContext
  // Local file paths of inbound media (images/documents/audio) downloaded into the media
  // cache (~/.vontra/cache/...) at ingest. Remote attachment URLs expire and the vision
  // tool reads paths, not URLs — so the assistant reads the attachment by this local path.
  // Absent/empty when the message had no media or the downloads failed/were unsafe (SSRF).
  mediaPaths?: string[]
  delivery?: {
    transport: string
    status: 'delivered' | 'failed'
    path?: string | null
    deliveredAt?: number | null
    error?: string | null
  }
  sources?: ConnectorSourceRef[]
  trace?: TraceContext
  createdAt: number
}

export interface ConnectorSourceRef {
  id: string
  connectorId: ConnectorId
  sourceId?: string | null
  itemId?: string | null
  title: string
  path?: string | null
  excerpt?: string | null
  createdAt?: number | null
}

export type ConnectorFeedAction = 'Review' | 'Compare' | 'Plan' | 'Summarize' | 'Attach'

export interface ConnectorAttentionItem {
  id: string
  eventId: string
  connectorId: ConnectorId
  threadKey: string
  connectorAction: string
  title: string
  summary: string
  actor: string
  action: ConnectorFeedAction
  priority: ConnectorPriority
  requiresApproval: boolean
  sourceIds: string[]
  sources: ConnectorSourceRef[]
  createdAt: number
  // Burst coalescing (thread-address.coalesceThreadBursts): how many messages this
  // conversation collapsed into (1 when not coalesced), and whether several arrived in a
  // tight window. Lets a chatty thread render as one "N messages" row instead of N rows.
  count?: number
  burst?: boolean
}

export interface ConnectorBriefing {
  generatedAt: number
  summary: string
  lines: string[]
  sourceIds: string[]
  sources: ConnectorSourceRef[]
  eventCount: number
  pendingApprovalCount: number
  latestEventAt: number | null
}

export interface ConnectorMemoryCandidateDraft {
  id: string
  eventId: string
  connectorId: ConnectorId
  projectId: string
  kind: ProjectMemoryKind
  text: string
  sourceType: 'connector'
  sourceRef: string
  sourceIds?: string[]
  confidence: ProjectMemoryConfidence
  privacyScope: ProjectMemoryPrivacyScope
  reason: string
  createdAt: number
}

export interface ConnectorMissionTrigger {
  id: string
  missionId: string
  missionTitle: string
  eventId: string
  connectorId: ConnectorId
  title: string
  summary: string
  action: ConnectorFeedAction
  sourceIds: string[]
  sources: ConnectorSourceRef[]
  createdAt: number
}

export interface ConnectorInteractionInput {
  connectorId: ConnectorId
  pluginName?: string
  action: string
  direction?: ConnectorDirection
  target?: string
  input?: Record<string, unknown>
  output?: Record<string, unknown>
}

export type EmailSourceKind = 'eml_dir' | 'maildir' | 'imap' | 'outlook_local'

export interface EmailSourceConfig {
  id: string
  label: string
  path: string
  kind: EmailSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
  // Incremental-sync high-water mark for this source: a sync only emits items newer
  // than this (and IMAP fetches only UID > cursor.lastImapUid). Absent = full resync.
  cursor?: SyncCursor
  // IMAP account (#multi-account email). For kind: 'imap' only. `password` is a
  // protected secret (safe:v1: at rest, masked to the renderer), resolved in the
  // main host before the connector sees it.
  host?: string
  port?: number
  tls?: boolean
  user?: string
  password?: string
  mailbox?: string
  // The account's actual email address, used as the SMTP From/envelope-from when sending.
  // Many providers use a login `user` that is NOT a full email address (e.g. 'maya', a
  // domain\login, or an alias), so the From header must come from here when present; we
  // fall back to `user` only when it itself parses as an email address.
  fromAddress?: string
  // Optional per-source ephemeral system-prompt override (see ConnectorChannelPrompt).
  channelPrompt?: string
  secretRotations?: SecretRotationMetadata
}

export interface EmailConnectorSettings {
  sources: EmailSourceConfig[]
  maxMessages: number
  lastSyncAt: number | null
  /** Look for Outlook for Mac's local store on every scan. Off by default: reading another app's
      container makes macOS ask for "data from other apps" and blocks the scan until answered. */
  detectOutlook?: boolean
}

// ---- Phone bot control (Telegram now, WhatsApp bridge next) ----
export type BotControlChannel = 'telegram' | 'whatsapp'

export interface TelegramBotControlSettings {
  enabled: boolean
  botToken: string
  allowedChatIds: string
  /** Telegram user IDs allowed to act, independently of the chat allowlist. */
  allowedUserIds: string
  polling: boolean
  lastUpdateId: number | null
  lastPollAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface WhatsAppBotControlSettings {
  enabled: boolean
  accessToken: string
  phoneNumberId: string
  verifyToken: string
  /** Meta App Secret used to verify X-Hub-Signature-256 over the raw webhook body. */
  appSecret: string
  allowedChatIds: string
  lastEventAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface BotControlSettings {
  telegram: TelegramBotControlSettings
  whatsapp: WhatsAppBotControlSettings
}

// ---- iMessage local bridge (macOS, Full-Disk-Access-gated) ----
// Reads the user's own ~/Library/Messages/chat.db read-only. Opt-in; off by
// default. isFromMe drives the bubble direction (your messages blue, theirs gray).
export interface IMessageItem {
  id: string
  handle: string
  isFromMe: boolean
  text: string
  date: number | null
  hasAttachments: boolean
  chatId: string
  chatName: string
  // Apple message GUID (chat.db message.guid). Stable identity used to resolve a
  // reply's quoted parent within a scan batch. Empty when unavailable.
  guid?: string | null
  // The GUID this message replies to (chat.db threaded reply, derived from
  // associated_message_guid). null when the message is not a threaded reply.
  replyToId?: string | null
}
export type IMessageReadResult = IMessageItem
export interface IMessageConnectorSettings {
  enabled: boolean
  allowedChatIds: string
  maxMessages: number
  lastSyncAt: number | null
  lastError: string | null
  // Incremental-sync high-water mark (newest message ROWID/date already ingested).
  // chat.db is single-source, so the cursor lives on the settings. Absent = full resync.
  cursor?: SyncCursor
}
export interface IMessageSyncResult {
  events: ConnectorEvent[]
  messages: number
  syncedAt: number
  // Advanced incremental cursor to persist back onto IMessageConnectorSettings.cursor.
  cursor?: SyncCursor
}

// ---- Discord live connector (bot token, REST) ----
// One source = one bot token across its servers. Allowed channels filter scope.
export interface DiscordChannel {
  id: string
  name: string
  guildId: string
  guildName: string
  threadId: string | null
  parentId?: string | null
}
export interface DiscordSource {
  id: string
  label: string
  token: string // protected secret
  allowedChannels: string
  enabled: boolean
  lastSyncAt: number | null
  lastError: string | null
  // Incremental-sync high-water mark (newest Discord snowflake id already ingested,
  // as a number); a sync only emits messages newer than this. Absent = full resync.
  cursor?: SyncCursor
  // Optional per-source ephemeral system-prompt override (see ConnectorChannelPrompt).
  channelPrompt?: string
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}
export interface DiscordConnectorSettings {
  sources: DiscordSource[]
  lastSyncAt: number | null
}
export interface DiscordMessage {
  id: string
  sourceId: string
  sourceLabel: string
  channelId: string
  channelName: string
  threadId: string | null
  threadName: string | null
  guildName: string
  senderId: string
  senderName: string
  isBot: boolean
  date: number
  text: string
  snippet: string
  replyTo: string | null
  // Quoted author + text from the referenced (replied-to) message, when Discord returns
  // the referenced_message inline. null when the message is not a reply.
  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}
export type DiscordReadResult = DiscordMessage & { raw: Record<string, unknown> }

// ---- Mattermost live connector (Personal Access / bot token, REST v4) ----
// One source = one server (base URL) + token. Self-hosted Slack-alike; the base URL is
// per-source (not a fixed host), so every request is built off `source.baseUrl`.

// A channel the bot can see, scoped to a team. Mattermost channel `type`:
//   'O' open/public, 'P' private, 'D' direct message, 'G' group message.
export interface MattermostChannel {
  id: string
  name: string // url-safe slug (channel.name)
  displayName: string // human label (channel.display_name)
  type: 'O' | 'P' | 'D' | 'G' | string
  teamId: string
  teamName: string
}

// One Mattermost source = one server + token. `baseUrl` is the server origin
// (e.g. https://mm.example.com) — Mattermost is NOT a fixed host, so this is required.
// `allowedChannels` is a csv allowlist of channel ids and/or names/slugs TensorFold may read
// and reply in (blank = every channel the bot can see). The token is a protected secret.
export interface MattermostSource {
  id: string
  label: string
  baseUrl: string // server origin, e.g. https://mm.example.com (no /api/v4 suffix)
  token: string // protected secret (Personal Access Token or bot token)
  // Explicit opt-in for a deliberately self-hosted RFC1918/loopback server. Metadata and
  // link-local addresses remain blocked even when enabled.
  allowPrivateNetwork?: boolean
  allowedChannels: string // csv allowlist of channel ids and/or slugs; blank = all visible
  enabled: boolean
  lastSyncAt: number | null
  lastError: string | null
  // Incremental-sync high-water mark (newest Mattermost post create_at, in ms, already
  // ingested); a sync only emits posts newer than this. Absent = full resync.
  cursor?: SyncCursor
  // Optional per-source ephemeral system-prompt override (see ConnectorChannelPrompt).
  channelPrompt?: string
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}

export interface MattermostConnectorSettings {
  sources: MattermostSource[]
  lastSyncAt: number | null
}

export interface MattermostMessage {
  id: string // post id
  sourceId: string
  sourceLabel: string
  channelId: string
  channelName: string // display name when known, else slug/id
  teamName: string
  senderId: string
  senderName: string
  isBot: boolean
  date: number // create_at (ms)
  text: string
  snippet: string
  // The thread root post id (Mattermost root_id). null when the post is a thread root
  // itself (top-level post). A reply's quoted parent is resolved to its author + text.
  rootId: string | null
  replyTo: string | null
  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}

export type MattermostReadResult = MattermostMessage & { raw: Record<string, unknown> }

export interface MattermostListOptions {
  query?: string
  channel?: string
  limit?: number
}

export interface MattermostSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface MattermostSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: MattermostSource[]
  syncedAt: number
}

// ---- Matrix live connector (homeserver access token, client-server API) ----
// One source = one homeserver (base URL) + access token. The allowlist is ROOM
// ids/aliases (allowedRooms), not channels.

// One joined Matrix room the source can read/post in. `roomId` is the canonical
// "!opaque:server" id; `alias` is the human "#room:server" alias when the homeserver
// exposes one (best-effort, may be null). `name` is the room's display name (m.room.name)
// when resolvable, else falls back to the alias or id. The allowlist matches on roomId OR alias.
export interface MatrixRoom {
  roomId: string
  alias: string | null
  name: string
}
export interface MatrixSource {
  id: string
  label: string
  // Homeserver base URL, e.g. "https://matrix.org" (no trailing /_matrix). Not a fixed
  // host — the user supplies it per source, like an IMAP server.
  baseUrl: string
  token: string // protected secret (access token)
  // Explicit opt-in for a deliberately self-hosted RFC1918/loopback homeserver.
  allowPrivateNetwork?: boolean
  // Comma/space/semicolon-separated room IDs and/or aliases this source may read + reply
  // in. Blank = every room the user account is joined to.
  allowedRooms: string
  enabled: boolean
  lastSyncAt: number | null
  lastError: string | null
  // Incremental-sync high-water mark (newest event origin_server_ts in ms already
  // ingested); a sync only emits events newer than this. Absent = full resync.
  cursor?: SyncCursor
  // Optional per-source ephemeral system-prompt override (see ConnectorChannelPrompt).
  channelPrompt?: string
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}
export interface MatrixConnectorSettings {
  sources: MatrixSource[]
  lastSyncAt: number | null
}
export interface MatrixMessage {
  id: string // Matrix event_id ("$opaque")
  sourceId: string
  sourceLabel: string
  roomId: string
  roomName: string
  roomAlias: string | null
  senderId: string // Matrix user_id ("@user:server")
  senderName: string // resolved display name, falls back to the user_id
  isSelf: boolean // true when sender === the source account's own user_id
  date: number // origin_server_ts (ms)
  text: string
  snippet: string
  // Root event_id of the thread this message belongs to (m.relates_to rel_type
  // 'm.thread'), else null when it is a top-level room message.
  threadRootId: string | null
  // event_id this message replies to (m.relates_to → m.in_reply_to), independent of
  // threading. null when not a reply.
  replyTo: string | null
  // Best-effort quoted author + text of the replied-to event, resolved within the same
  // sync batch (Matrix does not inline the parent). null when not a reply / unresolved.
  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}
export type MatrixReadResult = MatrixMessage & { raw: Record<string, unknown> }
export interface MatrixListOptions {
  query?: string
  limit?: number
}
export interface MatrixSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}
export interface MatrixSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: MatrixSource[]
  syncedAt: number
}

// ---- Google Chat live connector (chat.googleapis.com REST) ----
// v1 = one incoming WEBHOOK URL per space (Option A, send-only). Optional accessToken is
// the Option B seam (service-account / OAuth2 bearer) that enables inbound reads.

// One Google Chat space the connector can reach. `name` is the API resource name
// (`spaces/AAAA...`); `displayName` is the human label when known (Option B list only).
export interface GoogleChatSpace {
  // API resource name, e.g. "spaces/AAAAt3pNxxx". This is the allowlist + threadKey unit.
  name: string
  displayName: string
  // 'SPACE' (named room) | 'DIRECT_MESSAGE' (DM) | 'GROUP_CHAT' — best-effort from the API.
  spaceType: string | null
}

// One Google Chat source. v1 = one incoming WEBHOOK URL per space (Option A). The optional
// `accessToken` is the Option B seam (service-account / OAuth2 bearer for inbound + REST send).
export interface GoogleChatSource {
  id: string
  label: string
  // OPTION A secret: the incoming-webhook URL for a space, including its ?key=&token=.
  // protected secret (carries an auth token in its query string). Empty until configured.
  webhookUrl: string
  // OPTION B secret (optional): a service-account / OAuth2 bearer access token granting
  // chat.messages + chat.spaces. When present, reads/sends go through the REST API instead
  // of (or in addition to) the webhook. protected secret. Token minting is out of v1 scope.
  accessToken?: string
  // Override of the REST base host. Defaults to https://chat.googleapis.com when blank —
  // Google Chat is a fixed host, so this is only for testing / a future regional endpoint.
  baseUrl?: string
  // csv allowlist of SPACE names (e.g. "spaces/AAAA, spaces/BBBB") this source may read +
  // post in. Bare ids and a leading "spaces/" are both accepted. Blank = every reachable space.
  allowedChannels: string
  enabled: boolean
  lastSyncAt: number | null
  lastError: string | null
  // Incremental-sync high-water mark (newest message createTime already ingested, as ms);
  // a sync only emits messages newer than this. Absent = full resync. (Option B only.)
  cursor?: SyncCursor
  // Optional per-source ephemeral system-prompt override (see ConnectorChannelPrompt).
  channelPrompt?: string
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}

export interface GoogleChatConnectorSettings {
  sources: GoogleChatSource[]
  lastSyncAt: number | null
}

export interface GoogleChatMessage {
  // API message resource name, e.g. "spaces/AAAA/messages/MMMM.MMMM". Stable id.
  id: string
  sourceId: string
  sourceLabel: string
  // Space (the "channel"): resource name + best-effort display name.
  spaceName: string
  spaceDisplayName: string
  // Thread resource name within the space, e.g. "spaces/AAAA/threads/TTTT". null at top level.
  threadName: string | null
  senderId: string
  senderName: string
  senderEmail: string | null
  // True when the message came from a Chat app/bot (sender.type === 'BOT') — our own
  // outbound included. Sync skips these to avoid reply-to-self loops.
  isBot: boolean
  date: number
  text: string
  snippet: string
  // Quoted/replied-to context. Google Chat threads don't expose a per-message parent over
  // REST, so this is populated only when a future caller resolves it; null otherwise.
  replyTo: string | null
  replyToAuthor?: string | null
  replyToText?: string | null
  threadKey: string
}

export type GoogleChatReadResult = GoogleChatMessage & { raw: Record<string, unknown> }

export interface GoogleChatListOptions {
  query?: string
  // Restrict to a single space resource name (or bare id). Blank = all allowed spaces.
  space?: string
  limit?: number
}

export interface GoogleChatSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface GoogleChatSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: GoogleChatSource[]
  syncedAt: number
  // Advanced incremental cursor to persist back onto the originating source.cursor.
  cursor?: SyncCursor
}

// ---- WhatsApp multidevice bridge (multi-account, QR-pairing) ----
// Each source is one paired number with its own app-key protected auth state. Off by
// default and experimental (uses the unofficial WhatsApp multidevice protocol).
export interface WhatsAppBridgeSource {
  id: string
  label: string
  enabled: boolean
  phoneNumber: string | null
  paired: boolean
  lastQrAt: number | null
  lastEventAt: number | null
  lastError: string | null
  /** Main-process-authored timestamp of the last actual paired-credential update. */
  credentialRotatedAt?: number | null
  createdAt: number
  updatedAt: number
}
export interface WhatsAppBridgeConnectorSettings {
  sources: WhatsAppBridgeSource[]
  lastSyncAt: number | null
}

// ---- Controller: attach current app/window context on demand ----
export interface ControllerSettings {
  enabled: boolean
  includeAccessibilityText: boolean
  maxTextChars: number
  lastCapturedAt: number | null
  lastError: string | null
}

export interface ControllerStatus {
  platform: NodeJS.Platform | string
  supported: boolean
  enabled: boolean
  accessibilityTrusted: boolean
  canCapture: boolean
  message: string
}

export interface ControllerSnapshot {
  appName: string
  bundleId: string | null
  windowTitle: string
  capturedAt: number
  platform: NodeJS.Platform | string
  text: string
  textCharCount: number
  truncated: boolean
  source: 'accessibility' | 'demo'
  warning: string | null
}

// ---- Voice bridge: local/browser first, API/custom ready ----
// 'browser'    Web Speech (no network), 'openai'/'custom' OpenAI-compatible /audio/speech,
// 'elevenlabs' ElevenLabs cloud TTS (character/emotion via stability+style),
// 'local'      a downloadable expressive engine (Kokoro/Piper-class) served over a local
//              HTTP endpoint the user runs — falls back to Web Speech if unreachable.
export type VoiceProvider = 'browser' | 'openai' | 'custom' | 'elevenlabs' | 'local'

export interface VoiceSettings {
  enabled: boolean
  ttsEnabled: boolean
  sttEnabled: boolean
  ttsProvider: VoiceProvider
  sttProvider: VoiceProvider
  voiceName: string
  rate: number
  pitch: number
  volume: number
  autoSummarizeBeforeRead: boolean
  summaryMaxChars: number
  autoReadRoomUpdates: boolean
  // OpenAI-compatible cloud voice endpoint + protected secret. The key is encrypted at
  // rest, masked to the renderer, and resolved only inside the main-process voice runtime.
  apiBaseUrl: string
  apiKey: string
  ttsModel: string
  sttModel: string
  // ElevenLabs cloud TTS. apiKey is a protected secret (safe:v1: at rest, masked to the
  // renderer); resolved unmasked in the main host before the request. stability/style drive
  // emotion/character (0..1). Empty apiKey => provider disabled with a hint.
  elevenLabsApiKey: string
  elevenLabsVoiceId: string
  elevenLabsModelId: string
  elevenLabsStability: number
  elevenLabsStyle: number
  // Local expressive TTS engine (downloadable Kokoro/Piper-class server, e.g. an
  // OpenAI-/Kokoro-compatible /v1/audio/speech on 127.0.0.1). voice + emotion are passed to
  // the engine; if unreachable, speakText() gracefully falls back to the Web Speech voice.
  localTtsBaseUrl: string
  localTtsVoice: string
  localTtsEmotion: string
  secretRotations?: SecretRotationMetadata
}

// ---- Voice IPC (main-process cloud/local TTS + cloud STT) ----
// The main host performs every provider HTTP call so protected keys never cross to the
// renderer and endpoint/redirect/SSRF checks happen at the privileged network boundary.
// Audio comes back base64-encoded with its mime type; the renderer turns it into a Blob.
export interface VoiceSynthesisResult {
  ok: boolean
  audioBase64?: string
  mimeType?: string
  error?: string
  // 'fallback' tells the renderer to use the local Web Speech voice instead (engine down).
  fallback?: boolean
}

export interface VoiceTranscriptionInput {
  audioBase64: string
  mimeType?: string
}

export interface VoiceTranscriptionResult {
  ok: boolean
  text?: string
  error?: string
}

export interface LocalTtsDiscovery {
  found: boolean
  baseUrl: string
  voices: string[]
  message: string
}

// ---- TensorFold Orchestrator host / remote client bridge ----
export type RemoteControlBindHost = '127.0.0.1' | '0.0.0.0'
export type RemoteControlEventKind = 'agent' | 'project' | 'thread' | 'room' | 'mission' | 'ticket' | 'job' | 'automation' | 'connector' | 'snapshot' | 'terminal' | 'stream_gap'

export interface RemoteControlSettings {
  enabled: boolean
  serverName: string
  bindHost: RemoteControlBindHost
  port: number
  authToken: string
  allowLan: boolean
  // Optional always-on public reachability via a supervised tunnel (cloudflared/ngrok). Opt-in;
  // default stays 127.0.0.1-only. tunnelUrl is the last-resolved public https URL.
  tunnelEnabled?: boolean
  tunnelProvider?: 'cloudflared' | 'ngrok'
  tunnelUrl?: string | null
  // Set after the one-time headless credential bootstrap is persisted. Later starts never
  // print bearer/pairing secrets unless VONTRA_HEADLESS_BOOTSTRAP=1 explicitly rotates them.
  headlessBootstrappedAt?: number | null
  lastStartedAt: number | null
  lastClientAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface RemoteClientSettings {
  enabled: boolean
  serverUrl: string
  authToken: string
  lastConnectedAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface RemoteOrchestratorProfile {
  id: string
  name: string
  serverUrl: string
  authToken: string
  enabled: boolean
  lastConnectedAt: number | null
  lastError: string | null
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}

export interface RemoteControlStatus {
  enabled: boolean
  listening: boolean
  serverName: string
  bindHost: RemoteControlBindHost
  port: number | null
  url: string | null
  clients: number
  lastStartedAt: number | null
  lastClientAt: number | null
  lastError: string | null
}

export interface RemoteControlWorkSnapshot {
  jobs: {
    total: number
    visible: number
    queued: number
    running: number
    waitingApproval: number
    paused: number
    stalled: number
    failed: number
    deadLetter: number
    retryable: number
  }
  leases: {
    active: number
    expired: number
    expiringSoon: number
    missingRunning: number
  }
  approvals: {
    pending: number
    tool: number
    connector: number
    mission: number
  }
  nextWakeAt: number | null
  nextWakeJobId: string | null
  generatedAt: number
  // Terminal health hosted by this orchestrator (optional → older hosts omit it and the
  // readiness doctor renders "terminal health unknown" instead of a false 'ok').
  terminals?: {
    total: number
    running: number
    attention: number
  }
}

export type RemoteControlCommandEffect = 'read' | 'write'

export interface RemoteControlCommandDescriptor {
  id: string
  title: string
  method: 'GET' | 'POST'
  path: string
  effect: RemoteControlCommandEffect
  description: string
}

export interface RemoteControlDoctorSnapshot {
  generatedAt: number
  serverName: string
  appVersion: string
  headless: boolean
  counts: {
    threads: number
    projects: number
    rooms: number
    tickets: number
    automations: number
    missions: number
    jobs: number
    connectorEvents: number
    activeThreads: number
    activeRooms: number
    activeMissions: number
  }
  work: RemoteControlWorkSnapshot
}

// A provider as a paired device may see it (VST-035): identity + models for pickers and
// labels, NEVER credentials (no apiKey/cliEnv/serverPassword/serverUrl/headers).
export interface RedactedProvider {
  id: string
  name: string
  kind: string
  transport?: string | null
  models: string[]
  defaultModel?: string | null
}

export interface RemoteControlSnapshot {
  generatedAt: number
  serverName: string
  appVersion: string
  headless: boolean
  threads: Thread[]
  projects: Project[]
  rooms: Room[]
  tickets: Ticket[]
  automations: Automation[]
  missions: Mission[]
  jobs: Job[]
  connectorEvents: ConnectorEvent[]
  work: RemoteControlWorkSnapshot
  activeThreadIds: string[]
  activeRoomIds: string[]
  activeMissionIds: string[]
  // Redacted provider identities so a paired client can label thread models and populate
  // pickers. Optional: older hosts don't send it.
  providers?: RedactedProvider[]
}

export interface RemoteControlEvent {
  id: string
  kind: RemoteControlEventKind
  createdAt: number
  data: unknown
}

export interface RemoteControlActionResult {
  ok: boolean
  error?: string
}

export interface RemoteConnectionTestResult extends RemoteControlActionResult {
  status?: RemoteControlStatus
  snapshot?: Pick<RemoteControlSnapshot, 'serverName' | 'generatedAt' | 'activeThreadIds' | 'activeRoomIds' | 'activeMissionIds' | 'work'>
}

export interface RemoteDirectoryEntry {
  name: string
  path: string
  isDirectory: boolean
}

export interface RemoteDirectoryListResult extends RemoteControlActionResult {
  path: string
  parent: string | null
  entries: RemoteDirectoryEntry[]
}

export interface RemoteControlPairingInfo {
  url: string | null
  token: string
  status: RemoteControlStatus
}

export interface RemoteOrchestratorSnapshotResult extends RemoteControlActionResult {
  hostId: string | null
  name: string
  serverUrl: string
  generatedAt?: number
  activeThreadIds?: string[]
  activeRoomIds?: string[]
  activeMissionIds?: string[]
  work?: RemoteControlWorkSnapshot
  totalProjects?: number
  totalThreads?: number
  totalRooms?: number
  totalMissions?: number
  projects?: Project[]
  threads?: Thread[]
  rooms?: Room[]
  tickets?: Ticket[]
  jobs?: Job[]
}

export interface BotControlIngestInput {
  channel: BotControlChannel
  chatId: string
  /** Authenticated channel principal (Telegram from.id or Meta messages[].from). */
  senderId?: string
  /** Stable, channel-issued delivery identifier used to reject replays. */
  eventId?: string
  text: string
  senderName?: string
  payload?: Record<string, unknown>
}

export interface BotControlResult {
  ok: boolean
  channel: BotControlChannel
  chatId: string
  reply: string
  action: string
  roomId: string | null
}

export interface BotControlPollResult {
  ok: boolean
  processed: number
  skipped: number
  error: string | null
  lastUpdateId: number | null
}

export interface BotControlSessionMessage {
  route: 'central' | 'room'
  text: string
  roomId: string | null
  action: string
  createdAt: number
}

export interface BotControlSessionRoom {
  roomId: string | null
  action: string
  createdAt: number
}

export interface BotControlPendingConfirmation {
  code: string
  action: 'close' | 'remove' | 'cleanup'
  roomRef?: string | null
  /** Confirmation is bound to the authenticated principal that requested it. */
  principalId: string
  attemptsRemaining: number
  createdAt: number
  expiresAt: number
}

export interface BotControlSession {
  id: string
  channel: BotControlChannel
  chatId: string
  activeRoomId: string | null
  createdAt: number
  updatedAt: number
  pendingConfirmation?: BotControlPendingConfirmation | null
  /** Recent authenticated channel delivery ids; persisted to make retries idempotent. */
  processedEventIds?: string[]
  roomHistory: BotControlSessionRoom[]
  messageHistory: BotControlSessionMessage[]
}

export interface EmailMessage {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  subject: string
  from: string
  to: string
  date: number | null
  unread: boolean
  snippet: string
  size: number
  // Normalized RFC5322 Message-ID of this message (lower-cased, angle brackets
  // stripped), captured during the lightweight scan so a reply can resolve its quoted
  // parent. Empty when the message had no Message-ID header.
  messageId?: string | null
  // Normalized Message-ID this message is a reply to (from the In-Reply-To header),
  // captured during the lightweight scan so sync can inject reply-to context. null when
  // the message is not a reply.
  inReplyTo?: string | null
}

export interface EmailReadResult extends EmailMessage {
  headers: Record<string, string>
  body: string
  // Decoded inline-MIME attachment parts, populated only on the SYNC path (withAttachments).
  // Email attachments aren't URLs the media cache can fetch — they're MIME parts whose bytes
  // the IMAP client already decoded — so sync writes these to the media cache and stamps the
  // resulting local paths onto the inbound event's mediaPaths. Absent on read paths.
  attachments?: EmailAttachment[]
}

// One decoded email attachment part (filename + content-type + raw bytes), produced by the
// IMAP fetch during sync from the message's BODYSTRUCTURE + a part download.
export interface EmailAttachment {
  // BODYSTRUCTURE part identifier (e.g. "2", "1.2") the bytes were downloaded from.
  part: string
  filename: string | null
  contentType: string | null
  // Declared size from BODYSTRUCTURE (pre-decode estimate); the cache caps on actual bytes.
  size: number | null
  data: Uint8Array
}

export interface EmailThreadResult {
  root: EmailReadResult
  messages: EmailReadResult[]
  summary: string
}

export interface EmailSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface EmailListOptions {
  query?: string
  unreadOnly?: boolean
  limit?: number
}

export interface EmailDraftInput {
  mode?: 'draft' | 'send'
  to: string
  subject: string
  body: string
  threadKey?: string
  inReplyTo?: string
}

export interface EmailResolveInput {
  id: string
  note?: string
}

export interface EmailSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: EmailSourceConfig[]
  syncedAt: number
  // Decoded MIME attachments for the emitted events, keyed by event id. The host writes these
  // bytes to the media cache and stamps the resulting local paths onto each event's mediaPaths
  // (email attachments aren't fetchable URLs, so they ride along here as raw bytes). Absent when
  // nothing in the batch had attachments. Not persisted — consumed once during email:sync.
  attachmentsByEvent?: Record<string, EmailAttachment[]>
}

export type SlackSourceKind = 'export_dir' | 'api_token'

export interface SlackSourceConfig {
  id: string
  label: string
  path: string
  kind: SlackSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
  // Incremental-sync high-water mark (newest message ts already ingested); a sync
  // only emits messages newer than this. Absent = full resync.
  cursor?: SyncCursor
  // Live API source (#api_token). `token` is a protected secret (safe:v1: at rest,
  // masked to the renderer). `channels` is the comma-separated channel IDs/names
  // TensorFold is allowed to read + reply in (blank = every channel the bot can see).
  token?: string
  channels?: string
  // Optional per-source system-prompt override (see ConnectorChannelPrompt). Injected as
  // an EPHEMERAL system message only for messages from this source; never baked into the
  // persisted system-prompt snapshot.
  channelPrompt?: string
  secretRotations?: SecretRotationMetadata
}

export interface SlackConnectorSettings {
  sources: SlackSourceConfig[]
  maxMessages: number
  lastSyncAt: number | null
}

// A reference to a Slack file attachment. `url_private` is an authenticated URL — fetching it
// requires the workspace bot token as `Authorization: Bearer <token>` — so it is downloaded via
// the auth-aware media-cache path, not the public one. Carried on the public SlackMessage so the
// host can stamp mediaPaths from it during slack:sync.
export interface SlackFileRef {
  urlPrivate: string
  name: string | null
  mimetype: string | null
}

export interface SlackMessage {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  channelId: string
  channelName: string
  senderId: string
  senderName: string
  ts: string
  threadTs?: string | null
  date: number | null
  text: string
  snippet: string
  // File attachments on this message (live API only; export JSON file refs are not wired). Each
  // needs the bot token to download (url_private). Absent/empty when the message had no files.
  files?: SlackFileRef[]
}

export interface SlackReadResult extends SlackMessage {
  raw: Record<string, unknown>
}

export interface SlackListOptions {
  query?: string
  channel?: string
  limit?: number
}

export interface SlackSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface SlackSyncResult {
  events: ConnectorEvent[]
  messages: number
  sources: SlackSourceConfig[]
  syncedAt: number
}

export type CalendarSourceKind = 'ics_file' | 'ics_dir'

export interface CalendarSourceConfig {
  id: string
  label: string
  path: string
  kind: CalendarSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
}

export interface CalendarConnectorSettings {
  sources: CalendarSourceConfig[]
  maxEvents: number
  lastSyncAt: number | null
}

export interface CalendarEvent {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  title: string
  startAt: number | null
  endAt: number | null
  allDay: boolean
  location: string
  description: string
  organizer: string
  attendees: string[]
  status: string
}

export interface CalendarReadResult extends CalendarEvent {
  uid: string
  raw: Record<string, string>
}

export interface CalendarListOptions {
  query?: string
  from?: number | string
  to?: number | string
  day?: string
  limit?: number
}

export interface CalendarSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface CalendarSyncResult {
  events: CalendarEvent[]
  sources: CalendarSourceConfig[]
  syncedAt: number
}

export interface CalendarMutationInput {
  mode?: 'create' | 'update'
  id?: string
  sourceId?: string
  title: string
  startAt?: number | string | null
  endAt?: number | string | null
  allDay?: boolean
  location?: string
  description?: string
  organizer?: string
  attendees?: string[]
}

export type DocumentSourceKind = 'file' | 'dir'

export interface DocumentSourceConfig {
  id: string
  label: string
  path: string
  kind: DocumentSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
}

export interface DocumentConnectorSettings {
  sources: DocumentSourceConfig[]
  maxDocuments: number
  maxDocumentBytes: number
  lastSyncAt: number | null
}

export interface DocumentItem {
  id: string
  sourceId: string
  sourceLabel: string
  path: string
  relativePath: string
  title: string
  excerpt: string
  size: number
  modifiedAt: number | null
  extension: string
}

export interface DocumentReadResult extends DocumentItem {
  content: string
}

export interface DocumentListOptions {
  query?: string
  path?: string
  extension?: string
  limit?: number
}

export interface DocumentSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface DocumentSyncResult {
  documents: DocumentItem[]
  sources: DocumentSourceConfig[]
  syncedAt: number
}

export interface TableColumnSummary {
  name: string
  count: number
  min: number
  max: number
  average: number
}

export interface TableItem extends DocumentItem {
  rowCount: number
  columnCount: number
  columns: string[]
  delimiter: ',' | '\t'
  numericColumns: TableColumnSummary[]
}

export interface TableReadResult extends TableItem {
  rows: string[][]
}

export interface TableListOptions extends DocumentListOptions {}

export interface TableSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export type BrowserSourceKind = 'current_tab' | 'history_export'

export interface BrowserSourceConfig {
  id: string
  label: string
  kind: BrowserSourceKind
  enabled: boolean
  createdAt: number
  updatedAt: number
  lastSyncAt: number | null
  lastError?: string | null
  url?: string
  title?: string
  excerpt?: string
  capturedAt?: number | null
  path?: string
}

export interface BrowserConnectorSettings {
  sources: BrowserSourceConfig[]
  maxItems: number
  lastSyncAt: number | null
}

export interface BrowserItem {
  id: string
  sourceId: string
  sourceLabel: string
  kind: BrowserSourceKind
  url: string
  title: string
  excerpt: string
  capturedAt: number | null
  path?: string | null
}

export interface BrowserReadResult extends BrowserItem {
  content: string
}

export interface BrowserListOptions {
  query?: string
  limit?: number
}

export interface BrowserSummaryResult {
  summary: string
  sources: ConnectorSourceRef[]
}

export interface BrowserSnapshotInput {
  url: string
  title?: string
  excerpt?: string
  capturedAt?: number | null
}

export interface BrowserSyncResult {
  events: ConnectorEvent[]
  items: BrowserItem[]
  sources: BrowserSourceConfig[]
  syncedAt: number
}
