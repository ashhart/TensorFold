// Prompt / config version registry + run linkage.
//
// Personas, skills, and subagent prompts already exist elsewhere as live editable
// data. What was missing is *provenance*: when a run used a prompt/config, which
// exact bytes did it use, and can we diff two versions or roll back to an earlier
// one? This module is that registry — pure, deterministic, dependency-injected.
//
// A "version" is content-addressed: the same normalized body always hashes to the
// same version id, so committing an unchanged body is a no-op (dedupe) and the
// label just re-points at the existing version. A run records the version id it
// resolved, so you can answer "what config produced this run" and "roll back the
// active label to the version run X used".
//
// No I/O, no clock of its own — callers inject `now` (a number). Mirrors the
// pr-publish / connector-retry injection pattern used across the codebase.

import { createHash } from 'node:crypto'

// The kinds of artifact this registry versions. Open string union so callers can
// register their own kinds without editing this file, but the common ones are named.
export type PromptArtifactKind = 'persona' | 'skill' | 'subagent' | 'system-prompt' | 'config' | (string & {})

export interface PromptVersion {
  /** Content-addressed id: `${kind}:${key}@${hash12}`. Stable across time for identical bodies. */
  id: string
  kind: PromptArtifactKind
  /** Logical name of the artifact this is a version of (e.g. a persona id or skill name). */
  key: string
  /** sha256 of the normalized body (hex). */
  hash: string
  /** The normalized body bytes. */
  body: string
  /** Monotonic sequence number within (kind,key): 1 for the first version, 2 for the next, … */
  seq: number
  /** Optional human label at creation time (e.g. "v2", "prod"). Labels also move independently — see LabelRef. */
  label: string | null
  createdAt: number
}

export interface LabelRef {
  kind: PromptArtifactKind
  key: string
  /** Movable pointer name (e.g. "active", "prod", "last-good"). */
  name: string
  /** The version id this label currently points at. */
  versionId: string
  updatedAt: number
}

export interface RunLink {
  runId: string
  versionId: string
  createdAt: number
}

export interface PromptRegistry {
  versions: PromptVersion[]
  labels: LabelRef[]
  runs: RunLink[]
}

export interface CommitInput {
  kind: PromptArtifactKind
  key: string
  body: string
  /** When set, (re)points this movable label at the committed version. */
  label?: string | null
}

export interface CommitResult {
  registry: PromptRegistry
  version: PromptVersion
  /** true when the body matched an existing version (no new version row created). */
  deduped: boolean
}

export function emptyRegistry(): PromptRegistry {
  return { versions: [], labels: [], runs: [] }
}

// ---- hashing / ids -------------------------------------------------------

/** Normalize body bytes so trailing whitespace / CRLF don't spawn spurious versions. */
export function normalizeBody(body: string | null | undefined): string {
  return String(body ?? '').replace(/\r\n?/g, '\n').replace(/[ \t]+$/gm, '').trim()
}

export function hashBody(body: string | null | undefined): string {
  return createHash('sha256').update(normalizeBody(body)).digest('hex')
}

/** Deterministic version id from kind+key+hash. Same body → same id, forever. */
export function versionIdFor(kind: PromptArtifactKind, key: string, hash: string): string {
  return `${kind}:${cleanKey(key)}@${hash.slice(0, 12)}`
}

// ---- commit --------------------------------------------------------------

/**
 * Commit a body as a version of (kind,key). Content-addressed: an identical body
 * returns the existing version (deduped=true) and only moves the label. A changed
 * body creates a new version with the next seq. Pure — returns a new registry.
 */
export function commitVersion(registry: PromptRegistry, input: CommitInput, now = 0): CommitResult {
  const kind = input.kind
  const key = cleanKey(input.key)
  const body = normalizeBody(input.body)
  const hash = createHash('sha256').update(body).digest('hex')
  const id = versionIdFor(kind, key, hash)

  const existing = registry.versions.find((v) => v.id === id)
  let versions = registry.versions
  let version: PromptVersion
  let deduped: boolean

  if (existing) {
    version = existing
    deduped = true
  } else {
    const seq = registry.versions.filter((v) => v.kind === kind && v.key === key).length + 1
    version = {
      id,
      kind,
      key,
      hash,
      body,
      seq,
      label: cleanLabel(input.label),
      createdAt: now,
    }
    versions = [...registry.versions, version]
    deduped = false
  }

  let labels = registry.labels
  if (input.label != null && cleanLabel(input.label)) {
    labels = upsertLabel(labels, kind, key, cleanLabel(input.label)!, version.id, now)
  }

  return { registry: { ...registry, versions, labels }, version, deduped }
}

// ---- run linkage ---------------------------------------------------------

/**
 * Record that `runId` used `versionId`. Idempotent per runId — a run has exactly
 * one version link; re-linking overwrites. Throws if the version is unknown, so
 * you can't dangle a run off a nonexistent version.
 */
export function linkRun(registry: PromptRegistry, runId: string, versionId: string, now = 0): PromptRegistry {
  const id = cleanKey(runId)
  if (!id) throw new Error('linkRun: runId is required')
  if (!registry.versions.some((v) => v.id === versionId)) {
    throw new Error(`linkRun: unknown versionId ${versionId}`)
  }
  const runs = [...registry.runs.filter((r) => r.runId !== id), { runId: id, versionId, createdAt: now }]
  return { ...registry, runs }
}

/** The version a run used, or null. */
export function versionForRun(registry: PromptRegistry, runId: string): PromptVersion | null {
  const link = registry.runs.find((r) => r.runId === cleanKey(runId))
  if (!link) return null
  return registry.versions.find((v) => v.id === link.versionId) ?? null
}

/** All runs that used a given version, newest first. */
export function runsForVersion(registry: PromptRegistry, versionId: string): RunLink[] {
  return registry.runs
    .filter((r) => r.versionId === versionId)
    .slice()
    .sort((a, b) => b.createdAt - a.createdAt)
}

// ---- resolution / rollback -----------------------------------------------

export function getVersion(registry: PromptRegistry, versionId: string): PromptVersion | null {
  return registry.versions.find((v) => v.id === versionId) ?? null
}

/** All versions of (kind,key), oldest→newest by seq. */
export function versionsOf(registry: PromptRegistry, kind: PromptArtifactKind, key: string): PromptVersion[] {
  const k = cleanKey(key)
  return registry.versions.filter((v) => v.kind === kind && v.key === k).slice().sort((a, b) => a.seq - b.seq)
}

/** The version a movable label currently points at, or null. */
export function resolveLabel(registry: PromptRegistry, kind: PromptArtifactKind, key: string, name: string): PromptVersion | null {
  const ref = registry.labels.find(
    (l) => l.kind === kind && l.key === cleanKey(key) && l.name === cleanLabel(name),
  )
  if (!ref) return null
  return getVersion(registry, ref.versionId)
}

/**
 * Roll a movable label back to a specific version (e.g. point "active" at the
 * version an earlier good run used). Throws if the version is unknown or does not
 * belong to (kind,key) — you can't roll a label onto a foreign artifact's version.
 */
export function rollbackLabel(
  registry: PromptRegistry,
  kind: PromptArtifactKind,
  key: string,
  name: string,
  versionId: string,
  now = 0,
): PromptRegistry {
  const target = getVersion(registry, versionId)
  if (!target) throw new Error(`rollbackLabel: unknown versionId ${versionId}`)
  if (target.kind !== kind || target.key !== cleanKey(key)) {
    throw new Error(`rollbackLabel: version ${versionId} does not belong to ${kind}:${cleanKey(key)}`)
  }
  const label = cleanLabel(name)
  if (!label) throw new Error('rollbackLabel: label name is required')
  return { ...registry, labels: upsertLabel(registry.labels, kind, key, label, versionId, now) }
}

// ---- diff ----------------------------------------------------------------

export type LineChangeKind = 'add' | 'remove' | 'context'

export interface DiffLine {
  kind: LineChangeKind
  text: string
}

export interface VersionDiff {
  fromId: string | null
  toId: string
  identical: boolean
  added: number
  removed: number
  lines: DiffLine[]
}

/**
 * Line-level diff between two versions (by id) or raw bodies. Pure LCS — no deps.
 * `fromId` may be null to diff against an empty baseline (i.e. "what did v1 add").
 */
export function diffVersions(registry: PromptRegistry, fromId: string | null, toId: string): VersionDiff {
  const from = fromId == null ? '' : requireBody(registry, fromId)
  const to = requireBody(registry, toId)
  return diffBodies(from, to, fromId, toId)
}

/** Diff two raw bodies directly (no registry lookup). Ids are cosmetic labels for the result. */
export function diffBodies(fromBody: string, toBody: string, fromId: string | null = null, toId = 'to'): VersionDiff {
  const a = normalizeBody(fromBody).length ? normalizeBody(fromBody).split('\n') : []
  const b = normalizeBody(toBody).length ? normalizeBody(toBody).split('\n') : []
  const lines = lcsDiff(a, b)
  let added = 0
  let removed = 0
  for (const l of lines) {
    if (l.kind === 'add') added++
    else if (l.kind === 'remove') removed++
  }
  return { fromId, toId, identical: added === 0 && removed === 0, added, removed, lines }
}

// ---- internals -----------------------------------------------------------

function requireBody(registry: PromptRegistry, versionId: string): string {
  const v = getVersion(registry, versionId)
  if (!v) throw new Error(`diffVersions: unknown versionId ${versionId}`)
  return v.body
}

function upsertLabel(
  labels: LabelRef[],
  kind: PromptArtifactKind,
  key: string,
  name: string,
  versionId: string,
  now: number,
): LabelRef[] {
  const k = cleanKey(key)
  const rest = labels.filter((l) => !(l.kind === kind && l.key === k && l.name === name))
  return [...rest, { kind, key: k, name, versionId, updatedAt: now }]
}

// Classic LCS-based line diff. Small O(n*m) table; prompt bodies are tiny.
function lcsDiff(a: string[], b: string[]): DiffLine[] {
  const n = a.length
  const m = b.length
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }
  const out: DiffLine[] = []
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push({ kind: 'context', text: a[i] })
      i++
      j++
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ kind: 'remove', text: a[i] })
      i++
    } else {
      out.push({ kind: 'add', text: b[j] })
      j++
    }
  }
  while (i < n) out.push({ kind: 'remove', text: a[i++] })
  while (j < m) out.push({ kind: 'add', text: b[j++] })
  return out
}

function cleanKey(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function cleanLabel(value: unknown): string | null {
  const s = typeof value === 'string' ? value.trim() : ''
  return s ? s : null
}
