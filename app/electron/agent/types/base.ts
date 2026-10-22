export type Role = 'system' | 'user' | 'assistant' | 'tool'

export interface ToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

// Multimodal message content. A plain string stays the common case (back-compat);
// arrays carry text + image parts (OpenAI-style) when a message has attachments.
export type ContentBlock =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string; detail?: 'low' | 'high' | 'auto' } }

export interface ApiMessage {
  role: Role
  content: string | ContentBlock[] | null
  tool_calls?: ToolCall[]
  tool_call_id?: string
  name?: string
}

// A user-supplied attachment from the composer. Images carry a downscaled data: URL
// for vision-capable (chat-completions) transports; documents carry extracted text
// that works on every transport (incl. the CLI ones, which flatten to text).
export interface Attachment {
  id: string
  kind: 'image' | 'document'
  name: string
  mime: string
  // images → a data:<mime>;base64,... URL; absent for documents.
  dataUrl?: string
  // documents → extracted text; also the image fallback label for non-vision seats.
  text?: string
  // original size in bytes (for display).
  bytes?: number
  /** Missing status is a ready legacy attachment; unfinished/failed rows never reach a model. */
  status?: 'processing' | 'ready' | 'error' | 'needs-reselection' | 'review'
  error?: string
  warnings?: string[]
  /** Display-only session preview after accepting text extraction; never a filesystem path. */
  previewDataUrl?: string
  /** Picker metadata only. This never grants access to a path after restart. */
  source?: { lastModified: number; relativePath?: string }
  extraction?: AttachmentTextExtraction
}

export interface AttachmentTextExtraction {
  method: 'apple-vision'
  state: 'review' | 'accepted'
  text: string
  /** Vision's normalized confidence score; not a probability or accuracy guarantee. */
  confidence: number | null
  warnings: string[]
  pageCount: number
  processedPages: number
  truncated: boolean
}

export interface AttachmentTextCapability {
  available: boolean
  method: 'apple-vision'
  reason?: string
  maxBytes: number
  maxPages: number
}

export interface AttachmentTextRequest {
  name: string
  mime: string
  /** User-selected bytes only. The host never accepts a renderer-supplied filesystem path. */
  bytes: Uint8Array | number[]
}

export interface AttachmentTextResult {
  ok: boolean
  text?: string
  confidence?: number | null
  warnings?: string[]
  pageCount?: number
  processedPages?: number
  truncated?: boolean
  error?: string
}

export type ModelSpendBudgetPeriod = 'run' | 'day' | 'week' | 'month'

export interface ModelSpendBudget {
  enabled?: boolean
  period?: ModelSpendBudgetPeriod
  limitUsd?: number | null
  spentUsd?: number | null
  warnAtPercent?: number | null
  hardStop?: boolean
  updatedAt?: number | null
}

/** Non-secret audit metadata keyed by credential slot name (for example `apiKey`). */
export type SecretRotationMetadata = Record<string, number | null>

export interface Provider {
  id: string
  name: string
  transport?: 'chat-completions' | 'codex-cli' | 'claude-cli' | 'opencode' | 'kilo' | 'pi-cli' | 'cursor-cli' | 'gemini-cli'
  auth?: 'api-key' | 'codex-oauth' | 'claude-oauth'
  baseUrl: string // OpenAI-compatible, e.g. https://api.openai.com/v1 or http://localhost:11434/v1
  apiKey: string
  models: string[]
  kind?: 'cloud' | 'local'
  /** Transport may be loopback forwarding; computation still leaves this device. */
  remoteCompute?: true
  source?: 'openai' | 'openrouter' | 'ai-gateway' | 'ollama' | 'lmstudio' | 'omlx' | 'tensorfold' | 'custom'
  cliPath?: string
  cliProfile?: string
  cliSandbox?: 'read-only' | 'workspace-write' | 'danger-full-access'
  cliApprovalPolicy?: 'untrusted' | 'on-request' | 'never'
  cliPermissionMode?: 'default' | 'acceptEdits' | 'auto' | 'bypassPermissions' | 'dontAsk' | 'plan'
  cliEnv?: Record<string, string>
  // OpenCode transport (VST-033): the `opencode serve` HTTP+SSE endpoint this provider
  // drives. serverUrl defaults to http://127.0.0.1:4096; serverPassword is the optional
  // OPENCODE_SERVER_PASSWORD (HTTP Basic). Model ids are `providerID/modelID`.
  serverUrl?: string
  serverPassword?: string
  discoveredModels?: string[]
  accounts?: ProviderAccountProfile[]
  defaultAccountId?: string | null
  resolvedAccountId?: string | null // runtime account binding, populated by providerWithAccount
  spendBudget?: ModelSpendBudget | null
  secretRotations?: SecretRotationMetadata
}

export type ProviderAccountKind = 'subscription' | 'api-key' | 'profile'
export type ModelFailoverResetPolicy = 'return_after_reset' | 'stay_on_failover' | 'rotate_on_limit'

export interface ProviderAccountProfile {
  id: string
  name: string
  kind: ProviderAccountKind
  enabled: boolean
  apiKey?: string
  models?: string[]
  cliProfile?: string
  cliHome?: string
  env?: Record<string, string>
  autoFailover?: boolean
  failoverResetPolicy?: ModelFailoverResetPolicy
  spendBudget?: ModelSpendBudget | null
  notes?: string
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}

export interface OmlxDiscovery {
  found: boolean
  baseUrl: string
  apiKey: string
  models: string[]
  defaultModel: string | null
  configPath: string | null
  message: string
}

/** Probe target kinds. `custom` covers any OpenAI-compatible endpoint, including a remote TensorFold engine. */
export type ProviderDiscoveryKind = 'ollama' | 'lmstudio' | 'omlx' | 'custom'

export interface ProviderDiscovery {
  kind: ProviderDiscoveryKind
  found: boolean
  name: string
  source: Provider['source']
  baseUrl: string
  apiKey: string
  models: string[]
  defaultModel: string | null
  configPath: string | null
  message: string
}

export type ApprovalMode = 'suggest' | 'auto' | 'readonly'
export type ApprovalRuleScope = 'tool' | 'connector'
export type ApprovalRuleDecision = 'allow' | 'deny'
export type ApprovalRuleDuration = 'session' | 'always'

export interface ApprovalRule {
  id: string
  scope: ApprovalRuleScope
  key: string
  decision: ApprovalRuleDecision
  duration: ApprovalRuleDuration
  label?: string
  explanation?: string
  legacyTool?: string | null
  /** At least one scope binding is required for a rule to be active. */
  projectId?: string | null
  workspace?: string | null
  /** Persisted allow-always rules are finite; legacy rules without these fields are inactive. */
  expiresAt?: number | null
  maxUses?: number | null
  useCount?: number | null
  createdAt: number
  updatedAt: number
  lastUsedAt: number | null
}

export interface ThemePalette {
  accent: string
  /** Info/link/toggle/focus hue, separate from the primary accent. Falls back to `accent` when unset. */
  info?: string
  background: string
  foreground: string
  uiFont: string
  codeFont: string
  translucent: boolean
  /** Frosted (see-through) sidebar; defaults to true. The brand preset keeps it solid navy. */
  frosted?: boolean
  contrast: number // 0-100
}

export type ThemePreset = 'tensorfold' | 'navy' | 'glass' | 'custom'
