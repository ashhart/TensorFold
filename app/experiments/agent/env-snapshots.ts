// Reusable environment snapshots — the PURE PLANNING CORE.
//
// A "snapshot" is a baked, reusable environment (deps + toolchain) that a fresh
// worktree / sandbox / SSH workspace can restore from instead of re-running the
// whole setup from scratch every time. This module answers the three pure
// questions that surround that idea, with NO side effects:
//
//   1. DESCRIPTOR  — given a setup script + the workspace's lock/manifest inputs,
//                    what should we bake? (toolchains + package layers to install,
//                    normalized + deduped, in a deterministic order.)
//   2. CACHE KEY   — a content hash over exactly the inputs that affect the baked
//                    result, so an unchanged workspace reuses an existing snapshot
//                    and any change to a relevant input invalidates it.
//   3. RESTORE PLAN— given a target snapshot and the environment already present in
//                    a fresh workspace, the ordered list of steps to reach it, plus
//                    an explicit invalidation verdict (reuse / rebuild / partial).
//
// Materialization (actual docker build / worktree copy / SSH rsync of the baked
// layers) is DEFERRED and lives outside this module — it is injected by the caller.
// This core is deterministic and unit-tested with plain objects, mirroring the
// dependency-injected style of connector-retry / workspace-checkpoints.

import { createHash } from 'node:crypto'

// --- inputs -----------------------------------------------------------------

export type PackageManager =
  | 'npm'
  | 'pnpm'
  | 'yarn'
  | 'bun'
  | 'pip'
  | 'poetry'
  | 'uv'
  | 'cargo'
  | 'go'
  | 'bundler'

export type ToolchainKind = 'node' | 'python' | 'rust' | 'go' | 'ruby' | 'java' | 'deno'

/** A file whose CONTENT feeds the snapshot (lockfiles, manifests, version pins). */
export interface SnapshotInputFile {
  /** Workspace-relative POSIX path, e.g. "package-lock.json". */
  path: string
  /** Full file content (or a stable digest of it). Undefined => file absent. */
  content?: string
}

/** A single install layer to bake: a package manager applied to a set of manifests. */
export interface InstallLayer {
  manager: PackageManager
  /** The input files this layer's install is derived from (lock + manifest). */
  inputs: string[]
  /** The install command to bake, e.g. "npm ci". */
  command: string
}

/** A toolchain pin to bake, e.g. node@20.11.0 (kind + version, version optional). */
export interface ToolchainPin {
  kind: ToolchainKind
  version?: string
}

/** The setup script the user/agent runs today, decomposed for baking. */
export interface SetupScript {
  /** Raw shell steps (one command per entry) the current setup runs. */
  steps: string[]
}

// --- descriptor -------------------------------------------------------------

/** What to bake into a reusable snapshot: toolchains + ordered install layers. */
export interface SnapshotDescriptor {
  toolchains: ToolchainPin[]
  layers: InstallLayer[]
  /** Steps from the setup script that are NOT baked (run at restore time). */
  runtimeSteps: string[]
}

// Manifest/lock signatures per package manager, most-specific lock first. The lock
// file (when present) is the authoritative install input; the manifest is the
// fallback signal that the manager is in use even before a lock exists.
interface ManagerSignature {
  manager: PackageManager
  toolchain: ToolchainKind
  locks: string[]
  manifests: string[]
  /** install command when a lock IS present (reproducible), and when it is NOT. */
  lockedCommand: string
  unlockedCommand: string
}

// Ordering within a toolchain sets the no-lock fallback priority: the FIRST entry
// for a toolchain wins when no manager for it has a lock present. npm is first for
// node (the default when only package.json exists); a real pnpm/yarn/bun LOCK still
// wins over npm via the lockedRival check in planSnapshotDescriptor.
const MANAGER_SIGNATURES: ManagerSignature[] = [
  { manager: 'npm', toolchain: 'node', locks: ['package-lock.json', 'npm-shrinkwrap.json'], manifests: ['package.json'], lockedCommand: 'npm ci', unlockedCommand: 'npm install' },
  { manager: 'pnpm', toolchain: 'node', locks: ['pnpm-lock.yaml'], manifests: ['package.json'], lockedCommand: 'pnpm install --frozen-lockfile', unlockedCommand: 'pnpm install' },
  { manager: 'yarn', toolchain: 'node', locks: ['yarn.lock'], manifests: ['package.json'], lockedCommand: 'yarn install --frozen-lockfile', unlockedCommand: 'yarn install' },
  { manager: 'bun', toolchain: 'node', locks: ['bun.lockb', 'bun.lock'], manifests: ['package.json'], lockedCommand: 'bun install --frozen-lockfile', unlockedCommand: 'bun install' },
  { manager: 'poetry', toolchain: 'python', locks: ['poetry.lock'], manifests: ['pyproject.toml'], lockedCommand: 'poetry install', unlockedCommand: 'poetry install' },
  { manager: 'uv', toolchain: 'python', locks: ['uv.lock'], manifests: ['pyproject.toml'], lockedCommand: 'uv sync --frozen', unlockedCommand: 'uv sync' },
  { manager: 'pip', toolchain: 'python', locks: ['requirements.txt', 'requirements-dev.txt', 'Pipfile.lock'], manifests: ['Pipfile'], lockedCommand: 'pip install -r requirements.txt', unlockedCommand: 'pip install -r requirements.txt' },
  { manager: 'cargo', toolchain: 'rust', locks: ['Cargo.lock'], manifests: ['Cargo.toml'], lockedCommand: 'cargo fetch --locked', unlockedCommand: 'cargo fetch' },
  { manager: 'go', toolchain: 'go', locks: ['go.sum'], manifests: ['go.mod'], lockedCommand: 'go mod download', unlockedCommand: 'go mod download' },
  { manager: 'bundler', toolchain: 'ruby', locks: ['Gemfile.lock'], manifests: ['Gemfile'], lockedCommand: 'bundle install --deployment', unlockedCommand: 'bundle install' },
]

// Version-pin files: kind -> parser over the file content. Keeps toolchain
// detection independent of the package layers above.
const TOOLCHAIN_VERSION_FILES: Array<{ kind: ToolchainKind; path: string; parse: (c: string) => string | undefined }> = [
  { kind: 'node', path: '.nvmrc', parse: (c) => firstNonEmptyLine(c)?.replace(/^v/, '') },
  { kind: 'node', path: '.node-version', parse: (c) => firstNonEmptyLine(c)?.replace(/^v/, '') },
  { kind: 'python', path: '.python-version', parse: (c) => firstNonEmptyLine(c) },
  { kind: 'ruby', path: '.ruby-version', parse: (c) => firstNonEmptyLine(c) },
  { kind: 'go', path: '.go-version', parse: (c) => firstNonEmptyLine(c) },
]

function firstNonEmptyLine(content: string): string | undefined {
  for (const raw of content.split('\n')) {
    const line = raw.trim()
    if (line && !line.startsWith('#')) return line
  }
  return undefined
}

function fileMap(files: SnapshotInputFile[]): Map<string, SnapshotInputFile> {
  const map = new Map<string, SnapshotInputFile>()
  // Last write wins for duplicate paths, but only when it carries content.
  for (const f of files) {
    const existing = map.get(f.path)
    if (!existing || f.content !== undefined) map.set(f.path, f)
  }
  return map
}

function isPresent(file: SnapshotInputFile | undefined): boolean {
  return !!file && file.content !== undefined
}

/**
 * Decide what to bake. Detects toolchains (from version-pin files) and install
 * layers (from lock/manifest signatures), dedupes to ONE layer per manager
 * (locked install preferred), and separates setup-script steps into bake-time
 * (install/toolchain) vs runtime (everything else). Pure: no fs, no clock.
 */
export function planSnapshotDescriptor(input: {
  files: SnapshotInputFile[]
  setup?: SetupScript
}): SnapshotDescriptor {
  const map = fileMap(input.files)

  // Toolchains: explicit version pins first, then any toolchain a package layer implies.
  const toolchainVersions = new Map<ToolchainKind, string | undefined>()
  for (const vf of TOOLCHAIN_VERSION_FILES) {
    const f = map.get(vf.path)
    if (!isPresent(f)) continue
    const version = vf.parse(f!.content!)
    // Explicit pins are authoritative; do not overwrite a found version with undefined.
    if (!toolchainVersions.has(vf.kind) || version !== undefined) toolchainVersions.set(vf.kind, version)
  }

  // Install layers: at most ONE per toolchain (node/python/... can only have one
  // package manager driving its install). Several managers share the same manifest
  // (package.json => npm/pnpm/yarn/bun; pyproject.toml => poetry/uv), so a naive
  // per-manager loop would emit a redundant install for each. Resolve the winner
  // per toolchain: the manager whose LOCK is present wins (that's the one actually
  // in use); if none has a lock, the FIRST signature listed for that toolchain
  // (highest priority, e.g. pnpm before npm) wins on its manifest alone.
  const layers: InstallLayer[] = []
  const claimedToolchains = new Set<ToolchainKind>()
  const candidates = MANAGER_SIGNATURES.map((sig) => {
    const presentLocks = sig.locks.filter((p) => isPresent(map.get(p)))
    const presentManifests = sig.manifests.filter((p) => isPresent(map.get(p)))
    return { sig, presentLocks, presentManifests }
  }).filter((c) => c.presentLocks.length > 0 || c.presentManifests.length > 0)

  for (const { sig, presentLocks, presentManifests } of candidates) {
    if (claimedToolchains.has(sig.toolchain)) continue
    // If a LATER candidate for this same toolchain has a lock but this one does not,
    // yield to it (lock == authoritative signal of the manager actually in use).
    if (presentLocks.length === 0) {
      const lockedRival = candidates.some(
        (c) => c.sig.toolchain === sig.toolchain && c.sig.manager !== sig.manager && c.presentLocks.length > 0,
      )
      if (lockedRival) continue
    }
    claimedToolchains.add(sig.toolchain)
    const locked = presentLocks.length > 0
    layers.push({
      manager: sig.manager,
      // Locks first (authoritative), then manifests. Deterministic order.
      inputs: [...presentLocks, ...presentManifests],
      command: locked ? sig.lockedCommand : sig.unlockedCommand,
    })
    // Ensure the manager's toolchain exists even without an explicit pin.
    if (!toolchainVersions.has(sig.toolchain)) toolchainVersions.set(sig.toolchain, undefined)
  }

  const toolchains: ToolchainPin[] = [...toolchainVersions.entries()]
    .map(([kind, version]) => (version === undefined ? { kind } : { kind, version }))
    .sort((a, b) => a.kind.localeCompare(b.kind))

  // Setup-script steps: a step is BAKED if it looks like a known install/toolchain
  // command (so it becomes a layer we cache); otherwise it is a runtimeStep that
  // must re-run on every restore (migrations, codegen, env-specific bootstrapping).
  const runtimeSteps: string[] = []
  for (const step of input.setup?.steps ?? []) {
    if (!isBakeableStep(step)) runtimeSteps.push(step)
  }

  return { toolchains, layers, runtimeSteps }
}

const BAKEABLE_STEP_RE =
  /\b(npm (ci|install|i)|pnpm (install|i)|yarn( install)?|bun install|pip install|poetry install|uv sync|cargo (fetch|build)|go mod download|bundle install|nvm install|rustup|asdf install)\b/

/** True when a setup step installs deps/toolchain and can be baked into a layer. */
export function isBakeableStep(step: string): boolean {
  return BAKEABLE_STEP_RE.test(step.trim())
}

// --- cache key --------------------------------------------------------------

export interface CacheKeyInput {
  descriptor: SnapshotDescriptor
  /** Contents of every input file the descriptor references (for hashing). */
  files: SnapshotInputFile[]
  /** Optional extra salt (base image id, arch, platform) that affects the bake. */
  salt?: string
  /** Schema version so a change to the baking algorithm invalidates old caches. */
  schemaVersion?: number
}

export const CACHE_KEY_SCHEMA_VERSION = 1

/**
 * Deterministic content hash over EXACTLY the inputs that change the baked result:
 * toolchain pins, each layer's command, and the content of each referenced input
 * file. Two workspaces with identical relevant inputs get the same key (snapshot
 * reuse); any change to a referenced lock/manifest/toolchain/command flips it
 * (invalidation). Files not referenced by any layer/toolchain do NOT affect the key.
 */
export function computeCacheKey(input: CacheKeyInput): string {
  const map = fileMap(input.files)
  const schema = input.schemaVersion ?? CACHE_KEY_SCHEMA_VERSION
  const hash = createHash('sha256')
  hash.update(`vontra.env-snapshot/${schema}\n`)
  if (input.salt) hash.update(`salt:${input.salt}\n`)

  for (const tc of input.descriptor.toolchains) {
    hash.update(`toolchain:${tc.kind}@${tc.version ?? '*'}\n`)
  }
  for (const layer of input.descriptor.layers) {
    hash.update(`layer:${layer.manager}:${layer.command}\n`)
    // Reference each input file's content (or an explicit ABSENT marker) so a
    // lockfile edit — or its deletion — changes the key.
    for (const p of layer.inputs) {
      const f = map.get(p)
      hash.update(`input:${p}:`)
      hash.update(isPresent(f) ? sha256(f!.content!) : 'ABSENT')
      hash.update('\n')
    }
  }
  // runtimeSteps do not change what is baked, but DO change the environment a
  // restore reaches, so fold them in (order-sensitive).
  for (const step of input.descriptor.runtimeSteps) hash.update(`runtime:${step}\n`)

  return `snap_${hash.digest('hex').slice(0, 32)}`
}

function sha256(value: string): string {
  return createHash('sha256').update(value).digest('hex')
}

// --- restore plan / invalidation --------------------------------------------

export type RestoreVerdict = 'reuse' | 'rebuild' | 'partial'

export interface RestoreStep {
  kind: 'toolchain' | 'layer' | 'runtime'
  command: string
  /** Human label for UI. */
  label: string
}

export interface RestorePlan {
  verdict: RestoreVerdict
  /** True when a cached snapshot with the target key can be mounted as-is. */
  cacheHit: boolean
  /** Ordered steps to run to reach the target descriptor from the present env. */
  steps: RestoreStep[]
  /** Why we rebuilt / partial-restored (empty on a clean reuse). */
  reasons: string[]
}

/** What the fresh workspace already has, so the plan can skip satisfied steps. */
export interface PresentEnvironment {
  /** Cache keys already materialized and mountable in this environment. */
  cachedKeys?: string[]
  /** Toolchains already installed (kind[@version]). */
  toolchains?: ToolchainPin[]
  /** Managers whose install layer is already present (node_modules etc. exist). */
  installedManagers?: PackageManager[]
}

function toolchainSatisfied(target: ToolchainPin, present: ToolchainPin[] | undefined): boolean {
  if (!present) return false
  return present.some((p) => {
    if (p.kind !== target.kind) return false
    // Target with no pinned version is satisfied by any install of that kind.
    if (target.version === undefined) return true
    return p.version === target.version
  })
}

/**
 * Compute the restore plan + invalidation verdict. Pure decision function:
 *
 *   - cacheHit  => the exact snapshot key is already materialized here: REUSE,
 *                  restore = just the runtimeSteps (idempotent re-run).
 *   - otherwise => build the ordered steps to reach the descriptor, skipping any
 *                  toolchain/layer the PresentEnvironment already satisfies.
 *                  verdict is 'rebuild' when nothing was reusable, 'partial' when
 *                  some toolchains/layers were skipped (a warm-ish environment).
 */
export function planRestore(input: {
  descriptor: SnapshotDescriptor
  targetKey: string
  present?: PresentEnvironment
}): RestorePlan {
  const present = input.present ?? {}
  const reasons: string[] = []

  const cacheHit = (present.cachedKeys ?? []).includes(input.targetKey)
  if (cacheHit) {
    // Mount the baked snapshot; only the non-bakeable steps still run.
    const steps: RestoreStep[] = input.descriptor.runtimeSteps.map((command) => ({
      kind: 'runtime',
      command,
      label: `run ${command}`,
    }))
    return { verdict: 'reuse', cacheHit: true, steps, reasons: [] }
  }

  reasons.push(`no materialized snapshot for key ${input.targetKey}`)

  const steps: RestoreStep[] = []
  let skipped = 0
  let required = 0

  for (const tc of input.descriptor.toolchains) {
    required++
    if (toolchainSatisfied(tc, present.toolchains)) { skipped++; continue }
    const ver = tc.version ? `@${tc.version}` : ''
    steps.push({ kind: 'toolchain', command: `install-toolchain ${tc.kind}${ver}`, label: `toolchain ${tc.kind}${ver}` })
  }

  const installed = new Set(present.installedManagers ?? [])
  for (const layer of input.descriptor.layers) {
    required++
    if (installed.has(layer.manager)) { skipped++; continue }
    steps.push({ kind: 'layer', command: layer.command, label: `${layer.manager}: ${layer.command}` })
  }

  // runtimeSteps always run (they are, by definition, not cacheable).
  for (const command of input.descriptor.runtimeSteps) {
    steps.push({ kind: 'runtime', command, label: `run ${command}` })
  }

  let verdict: RestoreVerdict
  if (required === 0) {
    // Descriptor bakes nothing (only runtime steps) — treat as a rebuild of runtime env.
    verdict = 'rebuild'
  } else if (skipped === 0) {
    verdict = 'rebuild'
  } else if (skipped === required) {
    // Everything bakeable was already present but there was no cache hit for the
    // exact key (e.g. runtimeSteps changed) — still a partial restore, not a reuse.
    verdict = 'partial'
    reasons.push('all baked layers present but snapshot key differs')
  } else {
    verdict = 'partial'
    reasons.push(`${skipped}/${required} baked layers reused from present environment`)
  }

  return { verdict, cacheHit: false, steps, reasons }
}

/**
 * Invalidation check: has the cached snapshot gone stale relative to the current
 * workspace inputs? Returns the fresh key and whether it differs from the cached
 * one. This is the pure predicate a materializer consults before reusing a cache.
 */
export function isSnapshotStale(input: {
  cachedKey: string | undefined
  descriptor: SnapshotDescriptor
  files: SnapshotInputFile[]
  salt?: string
  schemaVersion?: number
}): { stale: boolean; freshKey: string } {
  const freshKey = computeCacheKey({
    descriptor: input.descriptor,
    files: input.files,
    salt: input.salt,
    schemaVersion: input.schemaVersion,
  })
  return { stale: input.cachedKey !== freshKey, freshKey }
}
