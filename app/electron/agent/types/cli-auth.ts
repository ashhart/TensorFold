// Shared CLI-auth + model-readiness types (renderer + main, type-only).
// Implementation lives in electron/agent/cli-auth.ts (pure) and
// electron/agent/cli-auth-runtime.ts (spawn orchestration, main only).

export type CliAuthTransport = 'codex-cli' | 'claude-cli'

export interface CliAuthStatus {
  loggedIn: boolean
  detail?: string // e.g. "ChatGPT" (codex) or "user@example.com (max)" (claude)
}

export interface CliAuthCheck extends CliAuthStatus {
  binaryPresent: boolean
  message?: string
}

export interface CliLoginResult {
  ok: boolean
  loggedIn: boolean
  detail?: string
  message?: string
}

export type ReadinessReason =
  | 'ok'
  | 'no-provider'
  | 'needs-endpoint'
  | 'needs-key'
  | 'cli-missing'
  | 'cli-unauthed'
  | 'needs-model' // built-in engine: no downloaded model that fits this Mac

export interface Readiness {
  ready: boolean
  reason: ReadinessReason
  message: string
  detail?: string
}
