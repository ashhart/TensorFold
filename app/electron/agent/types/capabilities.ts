import type { RunBudgetClass } from './connectors'

// ---- MCP client / external tool servers ----
export type McpTransportKind = 'stdio'
export type McpServerStatus = 'stopped' | 'connecting' | 'connected' | 'error'
export type McpServerApprovalMode = 'ask' | 'trusted'
export type McpServerLaunchMode = 'direct' | 'docker_gateway'
export type McpDockerNetworkMode = 'none' | 'bridge' | 'host'

export interface McpDockerMount {
  source: string
  target: string
  readOnly?: boolean
}

export interface McpDockerGatewayConfig {
  image: string
  workdir?: string | null
  network?: McpDockerNetworkMode
  readOnlyRootfs?: boolean
  envAllowlist?: string[]
  mounts?: McpDockerMount[]
}

export interface McpServerConfig {
  id: string
  name: string
  transport: McpTransportKind
  launchMode?: McpServerLaunchMode
  command: string
  args: string[]
  cwd?: string | null
  env?: Record<string, string>
  dockerGateway?: McpDockerGatewayConfig | null
  manifestPath?: string | null
  manifestSignature?: string | null
  manifestPublicKey?: string | null
  manifestVerifiedAt?: number | null
  manifestDigest?: string | null
  enabled: boolean
  autoConnect: boolean
  approvalMode: McpServerApprovalMode
  createdAt: number
  updatedAt: number
}

export interface McpToolDefinition {
  serverId: string
  serverName: string
  approvalMode: McpServerApprovalMode
  name: string
  callName: string
  title?: string
  description?: string
  inputSchema: Record<string, unknown>
  readOnlyHint?: boolean
  destructiveHint?: boolean
  openWorldHint?: boolean
}

export interface McpResourceDefinition {
  serverId: string
  serverName: string
  uri: string
  name: string
  title?: string
  description?: string
  mimeType?: string
}

export interface McpPromptArgumentDefinition {
  name: string
  description?: string
  required?: boolean
}

export interface McpPromptDefinition {
  serverId: string
  serverName: string
  name: string
  title?: string
  description?: string
  arguments: McpPromptArgumentDefinition[]
}

export interface McpServerProvenance {
  commandPath: string
  commandSha256: string
  checkedAt: number
  manifestDigest?: string | null
  sandboxed?: boolean
  gateway?: {
    kind: 'docker'
    image: string
    network: McpDockerNetworkMode
    readOnlyRootfs: boolean
    envAllowlist: string[]
    mountCount: number
  } | null
}

export interface McpServerRuntime {
  id: string
  status: McpServerStatus
  pid: number | null
  provenance: McpServerProvenance | null
  serverInfo: { name?: string; version?: string } | null
  capabilities: Record<string, unknown> | null
  tools: McpToolDefinition[]
  resources: McpResourceDefinition[]
  prompts: McpPromptDefinition[]
  lastConnectedAt: number | null
  lastError: string | null
  logs: string[]
}

export interface McpServerSnapshot extends McpServerConfig {
  runtime: McpServerRuntime
}

export interface McpSnapshot {
  servers: McpServerSnapshot[]
  tools: McpToolDefinition[]
  resources: McpResourceDefinition[]
  prompts: McpPromptDefinition[]
  updatedAt: number
}

export interface McpToolCallResult {
  ok: boolean
  output: string
  title?: string
  meta?: string
}

// ---- Capability registry (policy/audit/prompt selection surface) ----
export type CapabilityBuiltInId = 'documents' | 'pdf' | 'spreadsheets' | 'presentations' | 'browser' | 'controller' | 'computer'
export type CapabilitySource = 'built_in' | 'tool' | 'connector' | 'mcp' | 'skill' | 'command' | 'local_script' | 'plugin'
export type CapabilityLane =
  | 'workspace'
  | 'shell'
  | 'network'
  | 'communication'
  | 'document'
  | 'browser'
  | 'controller'
  | 'calendar'
  | 'mcp'
  | 'meta'
export type CapabilityState = 'enabled' | 'disabled'
export type CapabilityHealthStatus = 'unknown' | 'ok' | 'warning' | 'error'
export interface CapabilityHealthState {
  status: CapabilityHealthStatus
  checkedAt: number | null
  lastError: string | null
}
export type CapabilityTrustLevel = 'unknown' | 'checksum' | 'signed'
export type CapabilitySandboxProfile =
  | 'none'
  | 'read-only'
  | 'workspace-write'
  | 'no-network'
  | 'network-allowed'
  | 'full-access'
  | 'build-test'
  | 'package-install'
  | 'browser'
  | 'network-fetch'
  | 'connector-send'
  | 'system-control'
  | 'untrusted-code'
export interface LocalScriptCapability {
  id: string
  name: string
  description?: string | null
  path: string
  command?: string | null
  args?: string[]
  enabled?: boolean
  checksumSha256?: string | null
  signature?: string | null
  signatureVerifiedAt?: number | null
  sandboxProfile?: CapabilitySandboxProfile | null
  createdAt?: number | null
  updatedAt?: number | null
}
export interface PackagedPluginCapability {
  id: string
  name: string
  description?: string | null
  manifestPath: string
  packagePath?: string | null
  enabled?: boolean
  trusted?: boolean
  checksumSha256?: string | null
  signature?: string | null
  signatureVerifiedAt?: number | null
  sandboxProfile?: CapabilitySandboxProfile | null
  toolNames?: string[]
  createdAt?: number | null
  updatedAt?: number | null
}
export interface CapabilitySettings {
  disabledCapabilityIds: string[]
  health: Record<string, CapabilityHealthState>
  localScripts: LocalScriptCapability[]
  packagedPlugins: PackagedPluginCapability[]
}
export type CapabilityPermission =
  | 'read_workspace'
  | 'write_workspace'
  | 'run_shell'
  | 'network_public'
  | 'email_read'
  | 'email_compose'
  | 'connector_read'
  | 'connector_write'
  | 'document_read'
  | 'document_write'
  | 'browser_control'
  | 'controller_capture'
  | 'computer_use'
  | 'launch_subagent'
  | 'mcp_call'
  | 'render_visualization'
  | 'plan_update'
export type CapabilityRiskFlag =
  | 'filesystem_read'
  | 'filesystem_write'
  | 'shell_exec'
  | 'network'
  | 'external_system'
  | 'local_app_context'
  | 'email_content'
  | 'outbound_message'
  | 'untrusted_content'
  | 'prompt_injection'
  | 'approval_required'
  | 'destructive'
  | 'delegation'

export interface CapabilityDefinition {
  id: string
  name: string
  description: string
  source: CapabilitySource
  lane: CapabilityLane
  state: CapabilityState
  toolNames: string[]
  permissions: CapabilityPermission[]
  riskFlags: CapabilityRiskFlag[]
  readOnly: boolean
  approvalRequired: boolean
  health?: CapabilityHealthState
  promptInjectionWarning?: string
  metadata?: Record<string, unknown>
  // Its tools reach the model only in runs that selected it (a /command or @mention).
  optIn?: boolean
}

export interface CapabilitySet {
  id: string
  name: string
  capabilityIds: string[]
  selected: boolean
}

export interface CapabilityAuditEntry {
  capabilityId: string
  name: string
  source: CapabilitySource
  lane: CapabilityLane
  state: CapabilityState
  selected: boolean
  toolNames: string[]
  permissions: CapabilityPermission[]
  riskFlags: CapabilityRiskFlag[]
  readOnly: boolean
  approvalRequired: boolean
  health?: CapabilityHealthState
  promptInjectionWarning?: string
  metadata?: Record<string, unknown>
}

export interface CapabilityRegistrySnapshot {
  generatedAt: number
  capabilities: CapabilityDefinition[]
  selectedCapabilityIds: string[]
  disabledCapabilityIds: string[]
  selectedSets: CapabilitySet[]
  visibleToolNames: string[]
  audit: CapabilityAuditEntry[]
}

export type SecurityRole = 'viewer' | 'operator' | 'owner'

export interface PersistenceFailure {
  label: string
  message: string
  at: number
}

export interface SecurityControlSnapshot {
  actor: string
  role: SecurityRole
  commandSandboxAvailable: boolean
  mcpSandboxAvailable: boolean
  signedUpdatesRequired: boolean
  updatePublicKeyConfigured: boolean
  updateTrustRootProductionReady: boolean
  signedMcpRequired: boolean
  trustedMcpExplicitlyAllowed: boolean
  localStateKeyMode: 'profile-key' | 'external-env-key'
  externalSecretKeyRequired: boolean
  copiedProfileDecryptable: boolean
  legacySafeStorageSecrets: number
  legacySafeStorageBackupPath: string | null
  quarantinedProtectedFiles: number
  quarantinedProtectedPaths: string[]
  quarantineBackupPaths: string[]
  unencryptedLocalStoreAllowed: boolean
  strictAuditEnabled: boolean
  auditHealthy: boolean
  lastAuditError: string | null
  lastPersistenceFailure: PersistenceFailure | null
  localMemoryCounts: Record<string, number>
}

export interface SecurityPurgeResult {
  ok: boolean
  purgedAt: number
  counts: Record<string, number>
  removedPaths: string[]
  preservedCredentialsAndSettings: string[]
  preservedResiduals: Array<{
    path: string
    kind: 'managed_worktree'
    reason: string
  }>
}

export interface ControlPlaneGap {
  id: string
  severity: 'info' | 'warning' | 'error'
  title: string
  detail: string
  subject: string
}

export interface RunBudgetDefinition {
  executionClass: RunBudgetClass
  label: string
  maxWallSeconds: number
  heartbeatIntervalSeconds: number
  stallGraceSeconds: number
  requiresHeartbeat: boolean
}
