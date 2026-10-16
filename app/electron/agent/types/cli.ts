// A launchable CLI (Clyde's `Agent`, adapted). Built-ins are static; users add their own.
export interface CliDefinition {
  id: string                    // stable ('codex','shell') or a uuid for custom
  name: string                  // "Codex"
  short: string                 // "codex"
  command: string | null        // launched in the login shell; null = a plain shell
  resumeCommand?: string | null // reattach command (used in Phase 2)
  color: string                 // hex tint, e.g. '#2FB9B0'
  icon: string                  // TensorFold icon name (src/lib/icons.tsx)
  builtin: boolean
}

// CLI version advisory (VST-037). Computed host-side by `versionAdvisory()` and carried as
// plain data across the type-only renderer boundary so the readiness doctor / settings can
// render "update available" without importing runtime code.
export type VersionAdvisoryStatus = 'current' | 'update_available' | 'ahead' | 'unknown'

export interface VersionAdvisory {
  status: VersionAdvisoryStatus
  currentVersion: string | null
  latestVersion: string | null
  updateCommand: string | null
  canUpdate: boolean
  message: string
}

// 'broken': the binary is there but `--version` failed or printed no version.
export type CliToolState = 'installed' | 'missing' | 'broken'

export interface CliToolStatus extends CliDefinition {
  installed: boolean
  version?: string | null
  // Absent from older hosts: the renderer then reads `installed` alone.
  state?: CliToolState
  // For 'broken': the probe's first error line, home path shown as ~.
  problem?: string | null
  effectiveCommand: string | null
  configuredPath?: string | null
  providerIds: string[]
  // Latest published version, fetched out-of-band by the host update checker (absent → the
  // readiness doctor renders "update status unknown" rather than a false "up to date").
  latestVersion?: string | null
  // The command to run to update this CLI (e.g. `npm i -g @openai/codex`).
  updateCommand?: string | null
  // Host-computed advisory (absent until a latest-version feed exists for this CLI).
  advisory?: VersionAdvisory
}

export type CliSessionStatus = 'idle' | 'busy' | 'attention' | 'done'

// ---- external harness session import (VST-033) ----
// A session that already exists on this machine inside another harness's home
// (~/.codex/sessions rollouts, ~/.claude/projects session files) that TensorFold can adopt as a
// thread and resume via the matching engine cursor. Parsed by electron/agent/session-import.ts.
export type ImportSessionSource = 'codex' | 'claude'

export interface ImportableSession {
  sessionId: string
  source: ImportSessionSource
  title: string
  cwd: string | null
  model: string | null
  createdAt: number | null
  updatedAt: number | null
  messageCount: number
  // codex → { threadId }, claude → { sessionId } — matches each transport's cursor shape.
  resumeCursor: { threadId: string } | { sessionId: string }
  // True when the parse hit its line cap before EOF, so messageCount is a lower bound.
  truncated: boolean
}

// A running CLI session inside a project. Live in-memory; the currently-live set is also
// persisted to disk (cli-sessions.json) so terminals survive an app restart and can be
// resumed to their prior CLI session (terminal:restore + resumeString).
export interface CliSession {
  id: string
  projectId: string
  cliId: string
  title: string
  status: CliSessionStatus
  createdAt: number
  // Persistence/restore fields (optional → backward-compatible with older state files):
  cwd?: string         // the directory the PTY was spawned in (restored verbatim)
  command?: string     // resolved command for a non-builtin/custom CLI, so it can be re-spawned
  // True only for a known built-in launcher (codex/claude/pi/opencode/gemini/shell). A custom CLI's
  // `command` is an arbitrary user-typed string that gets typed character-for-character into a real
  // login shell on restore — auto-replaying that without confirmation is a command-injection risk if
  // it ever contained shell metacharacters. Missing/false is treated as "do not auto-restore".
  builtin?: boolean
  // WHERE the PTY runs: 'local' on this machine, or 'orchestrator' on the remote host (the mini).
  // Chosen per-terminal at spawn (defaults to the project's executionTarget). Missing = 'local'.
  hostLocation?: 'local' | 'orchestrator'
  // Immutable origin for an Orchestrator-hosted PTY. Project/global execution settings only choose
  // where NEW terminals start; writes, resizes, renames, and kills route through this recorded host.
  // A legacy orchestrator row without this field is deliberately non-interactive (fail closed).
  orchestratorHostId?: string
  // WHO is driving the session: 'local' when this machine owns/spawned it, 'external' when it was
  // opened + is being driven by a remote client (surfaced as an "externally managed" badge on the
  // machine that doesn't own it). Missing = 'local'.
  managedBy?: 'local' | 'external'
}
