import type { UiPreferences } from '../../../shared/ui-preferences'
import type { ApprovalMode, ApprovalRule, Provider, ThemePalette, ThemePreset } from './base'
import type { CapabilitySettings } from './capabilities'
import type {
  BotControlSettings,
  BrowserConnectorSettings,
  CalendarConnectorSettings,
  ControllerSettings,
  DocumentConnectorSettings,
  EmailConnectorSettings,
  IMessageConnectorSettings,
  RemoteClientSettings,
  RemoteControlSettings,
  RemoteOrchestratorProfile,
  SlackConnectorSettings,
  VoiceSettings,
  WhatsAppBridgeConnectorSettings,
  DiscordConnectorSettings,
  MattermostConnectorSettings,
  MatrixConnectorSettings,
  GoogleChatConnectorSettings,
} from './connectors'
import type { EngineModelProfileSet } from './engine'
import type { RoomSeatLane, SshHost, TrustedSshHostKey } from './workspace'

export type ReasoningEffortSetting =
  | 'none'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'
  | 'max'
  | 'ultrathink'
  | 'ultracode'
  | 'dynamic'
  | '512'

// A user-owned, hot-reloaded assistant voice. scope is flat for clean JSON:
//   'global' | 'assistant' | `workspace:<id>` | `connector:<id>`
export interface PersonaProfile {
  id: string
  scope: string
  name: string
  body: string
  enabled: boolean
  updatedAt: number
}

export interface Settings {
  /** Display choices persisted with the active profile settings. */
  uiPreferences?: UiPreferences
  // A completed Minimal setup keeps the local model manager in the menu bar.
  desktopMode?: 'minimal' | 'full'
  desktopOnboardingComplete?: boolean
  providers: Provider[]
  modelHelperDefaults?: import('../../../shared/model-fit').ModelHelperDefaults
  cliToolOverrides?: Record<string, { binaryPath?: string }>
  activeProviderId: string | null
  activeProviderAccountId: string | null
  activeModel: string | null
  // Optional dedicated model for the Assistant surface (summaries, reply drafting). When unset, the
  // Assistant falls back to the globally-active model. Any CLI/cloud/local provider may be chosen.
  assistantProviderId?: string | null
  assistantProviderAccountId?: string | null
  assistantModel?: string | null
  // Optional dedicated model for the coding/task surface (VST-015). When unset, new tasks fall
  // back to the globally-active model, exactly like the assistant surface above. Together these
  // are the per-surface defaults: chat → assistant*, code → code*, both defaulting to active*.
  codeProviderId?: string | null
  codeProviderAccountId?: string | null
  codeModel?: string | null
  // Editable, scoped assistant personas (voice/personality) + a profile of the user.
  // Resolved most-specific-first and injected fresh each message (hot reload).
  personas?: PersonaProfile[]
  userProfile?: string
  // Set once the built-in TensorFold provider has been added, so a user's deletion of it sticks.
  engineProviderSeeded?: boolean
  // User-chosen colour per project tag (tag name → palette colour name, e.g. 'client-a' → 'orange').
  // Optional/backward-compatible; presets (personal/work) fall back to their built-in defaults.
  tagColors?: Record<string, string>
  agentPresets: AgentPreset[]
  theme: 'light' | 'dark' | 'system'
  approvalMode: ApprovalMode
  approvalRules: ApprovalRule[]
  defaultCwd: string | null
  temperature: number
  engineModelProfiles?: EngineModelProfileSet[]
  /** Malformed legacy rows are kept for recovery and never applied to the engine. */
  engineModelProfilesRecovery?: unknown
  engineModelProfilesProblem?: string
  reasoningEffort: ReasoningEffortSetting
  serviceTier?: 'fast' | 'flex' // CLI "speed" knob (Codex service tier); fast = priority, flex = cheaper
  // long-running tasks
  autoCompact: boolean // summarize older turns when the transcript crosses contextLimit
  // when true, after a successful run an aux model reviews the skills the agent consulted
  // and may FILE an improvement suggestion (never auto-applied). Off by default: it spends
  // extra model tokens per run, so it's opt-in. See skill-self-improvement.ts.
  skillSelfImprovement?: boolean
  contextLimit: number // token threshold that triggers compaction (e.g. 128000)
  contextLimitOverride?: number // optional explicit trigger; overrides per-model scaling, still clamped to the model window (Task #41)
  localContextLimit: number // smaller token threshold used for local providers
  contextLimitExplicit?: boolean // an intentional cloud cap, including the default-valued choice
  localContextLimitExplicit?: boolean // an intentional local cap, including the default-valued choice
  // when true, sandboxed task shell commands may reach the network (default: deny,
  // so the sandbox blocks outbound connections — package-install is always allowed)
  allowTaskNetwork?: boolean
  // app updates
  autoCheckUpdates: boolean
  updateFeedUrl: string
  // power/session behavior
  power: PowerSettings
  // Assistant/email/calendar/message/document/browser connectors
  email: EmailConnectorSettings
  slack: SlackConnectorSettings
  calendar: CalendarConnectorSettings
  documents: DocumentConnectorSettings
  browser: BrowserConnectorSettings
  // iMessage local bridge (macOS, Full-Disk-Access-gated, opt-in)
  imessage: IMessageConnectorSettings
  // WhatsApp multidevice bridge (multi-account, QR pairing, experimental)
  whatsapp: WhatsAppBridgeConnectorSettings
  // Discord live connector (bot token, REST)
  discord: DiscordConnectorSettings
  // Mattermost live connector (Personal Access / bot token, REST v4)
  mattermost: MattermostConnectorSettings
  // Matrix live connector (homeserver access token, client-server API)
  matrix: MatrixConnectorSettings
  // Google Chat live connector (incoming webhook URL; optional service-account token for inbound)
  googlechat: GoogleChatConnectorSettings
  // Phone bot control for room guiding
  botControl: BotControlSettings
  // Capability/plugin runtime state
  capabilities: CapabilitySettings
  // Foreground app/window context capture
  controller: ControllerSettings
  // Local-first speech in/out
  voice: VoiceSettings
  // TensorFold Orchestrator host / client bridge
  remoteControl: RemoteControlSettings
  remoteClient: RemoteClientSettings
  remoteOrchestrators: RemoteOrchestratorProfile[]
  // Global execution target (VST-041): 'orchestrator' routes EVERYTHING this machine
  // starts (threads, chats, terminals) to the default orchestrator host — a project
  // explicitly targeting its own orchestrator keeps that host, and a `localOnly` project
  // stays a hard local pin. Optional → absent means 'local' (backward-compatible).
  defaultExecutionTarget?: 'local' | 'orchestrator'
  defaultOrchestratorId?: string | null
  // appearance
  preset: ThemePreset
  light: ThemePalette
  dark: ThemePalette
  pointerCursors: boolean
  // Bold block-glyph progress rendering for the plan/todo panel.
  // Optional: absent means on (existing stored settings get the look without a migration).
  doomTodos?: boolean
  // Native notification when a run finishes, fails or needs approval while the window is in the
  // background. Optional: absent means on.
  notifyRuns?: boolean
  // SSH remote workspaces: host library + pinned server keys (no secrets stored)
  sshHosts: SshHost[]
  trustedSshHostKeys: TrustedSshHostKey[]
  // Soft-suggest initializing Git when adding a folder-backed project (Task #37).
  // Optional + defaults to true; when false the prompt/gate is skipped and the
  // project is created even if the folder is not a Git repo.
  showGitInitPrompt?: boolean
  // Company, client and project names withheld from external (planning-only) models;
  // matched case-insensitively on word boundaries. See planning-guard.ts.
  confidentialTerms?: string[]
}

export interface BotToolScope {
  mode: 'selected'
  names: string[]
}

export interface AgentPreset {
  id: string
  name: string
  role: string
  lane: RoomSeatLane
  providerId: string
  providerAccountId?: string | null
  model: string
  instructions?: string
  approvalScope?: 'readonly' | 'review'
  executionTarget?: 'workspace' | 'vm'
  toolScope?: BotToolScope
  /** Retains damaged persisted selection data without allowing the bot to run. */
  toolScopeProblem?: { value: unknown }
  paused?: boolean
  archivedAt?: number | null
  createdAt: number
  updatedAt: number
}

/** Captured by the host at chat creation; edits only affect future chats. */
export interface BotThreadConfig {
  presetId: string
  name: string
  role: string
  instructions: string
  providerId: string
  providerAccountId: string | null
  model: string
  approvalScope: 'readonly' | 'review'
  executionTarget: 'workspace' | 'vm'
  toolScope?: BotToolScope
  toolScopeProblem?: { value: unknown }
  capturedAt: number
}

export interface PowerSettings {
  keepComputerAlive: boolean
  lastStartedAt: number | null
  lastError: string | null
}
