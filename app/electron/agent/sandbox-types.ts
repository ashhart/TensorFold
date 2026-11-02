export type SandboxProfileId =
  | 'read-only'
  | 'build-test'
  | 'package-install'
  | 'browser'
  | 'network-fetch'
  | 'connector-send'
  | 'system-control'
  | 'untrusted-code'

export interface SandboxProfileDefinition {
  id: SandboxProfileId
  label: string
  allowCwdWrite: boolean
  allowNetwork: boolean
  envSecretsAllowed: boolean
  description: string
}

export const SANDBOX_ARTIFACT_SCHEMA = 'vontra.sandbox-artifacts/1'

export interface SandboxFileArtifact {
  path: string
  bytes: number
  sha256: string
  text?: string
}

export interface SandboxLogArtifact {
  kind: 'stdout' | 'stderr'
  bytes: number
  preview: string
}

export interface SandboxArtifactDiff {
  path: string
  status: 'added' | 'changed' | 'deleted'
  beforeSha256?: string
  afterSha256?: string
  bytes?: number
}

export interface SandboxArtifactBaseline {
  inputSnapshot: SandboxFileArtifact[]
  outputFiles: SandboxFileArtifact[]
  capturedAt: number

  workspaceSnapshot?: SandboxWorkspaceSnapshot
}

export interface SandboxWorkspaceSnapshot {
  root: string
  files: Array<{ path: string; fingerprint: string }>
}

export interface SandboxArtifactBundle {
  schema: typeof SANDBOX_ARTIFACT_SCHEMA
  id: string
  command: string
  cwd: string
  profile: SandboxProfileId | 'unsandboxed'
  implementation: SandboxImplementation
  inputSnapshot: SandboxFileArtifact[]
  outputFiles: SandboxFileArtifact[]
  screenshots: SandboxFileArtifact[]
  logs: SandboxLogArtifact[]
  diffs: SandboxArtifactDiff[]
  patch: string | null
  createdAt: number
}

export interface SandboxedCommandResult {
  code: number
  stdout: string
  stderr: string
  sandboxed: boolean
  profile: SandboxProfileId | 'unsandboxed'
  implementation: SandboxImplementation

  redactionCount?: number
  artifacts?: SandboxArtifactBundle

  changedPaths?: string[]
}

export type SandboxImplementation = 'macos-sandbox-exec' | 'linux-bubblewrap' | 'windows-appcontainer' | 'unsandboxed'

export interface SandboxedCommandOptions {
  cwd: string
  timeoutMs?: number
  maxBuffer?: number
  allowNetwork?: boolean
  profile?: SandboxProfileId
  signal?: AbortSignal
}
