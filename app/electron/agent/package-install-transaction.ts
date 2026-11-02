import { promises as fs } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { assertLocalPackageBins, publishLocalBinPermissions, rollbackLocalBinPermissions, type LocalBinPermissionChange } from './package-install-bins'
import {
  installEntryIdentity, isInsideInstallRoot, parseInstallJson, readInstallSnapshot,
  sameInstallEntry, validateInstallLockfile,
  type InstallEntryIdentity, type PackageInstallLayout,
} from './package-install-layout'

interface PublicationEntry {
  relative: string
  target: string
  incoming: string | null
  incomingIdentity: InstallEntryIdentity | null
  previous: InstallEntryIdentity | null
  publishedSha256?: string
  backup: string
  movedPrevious: boolean
  published: boolean
}

export interface PackagePublicationHooks {
  signal?: AbortSignal
  beforePublish?(relative: string): Promise<void> | void
}

export interface PackagePublicationResult {
  changedPaths: string[]
  warnings: string[]
}

export class PackagePublicationError extends Error {
  constructor(message: string, readonly recoveryPaths: string[] = []) { super(message) }
}

function assertNotCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) throw new Error('Dependency installation was cancelled before publication.')
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
      .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function unchangedManifestFields(before: Record<string, any>, after: Record<string, any>): boolean {
  const allowed = new Set(['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies', 'peerDependenciesMeta'])
  const retained = (value: Record<string, any>) => Object.fromEntries(Object.entries(value).filter(([key]) => !allowed.has(key)))
  return stableJson(retained(before)) === stableJson(retained(after))
}

async function assertParent(layout: PackageInstallLayout, relative: string): Promise<void> {
  const directory = path.dirname(relative) === '.' ? '' : path.dirname(relative).split(path.sep).join('/')
  const parent = path.join(layout.root, directory)
  const expected = layout.parents.get(directory)
  if (!expected || await fs.realpath(parent) !== parent
    || !sameInstallEntry(expected, await installEntryIdentity(parent))) {
    throw new Error(`Package directory changed while dependencies were being prepared: ${directory || '.'}`)
  }
}

async function assertOriginal(layout: PackageInstallLayout, entry: PublicationEntry): Promise<void> {
  await assertParent(layout, entry.relative)
  const actual = await installEntryIdentity(entry.target)
  if (!sameInstallEntry(entry.previous, actual)
    || actual && entry.previous && (actual.mtimeMs !== entry.previous.mtimeMs || actual.ctimeMs !== entry.previous.ctimeMs)) {
    throw new Error(`Dependency destination changed during installation: ${entry.relative}`)
  }
  if (layout.files.has(entry.relative)) {
    const previous = layout.files.get(entry.relative)
    const current = await readInstallSnapshot(entry.target)
    if (previous?.sha256 !== current?.sha256) {
      throw new Error(`Manifest changed during installation; no newer edits will be overwritten: ${entry.relative}`)
    }
  }
}

async function assertCanonicalManifests(layout: PackageInstallLayout, entries: PublicationEntry[]): Promise<void> {
  for (const [relative, original] of layout.files) {
    await assertParent(layout, relative)
    const published = entries.find((entry) => entry.relative === relative && entry.published)
    const expected = published?.incomingIdentity ?? original
    const expectedHash = published ? published.publishedSha256 : original?.sha256
    const actual = await readInstallSnapshot(path.join(layout.root, relative))
    if (actual?.sha256 !== expectedHash || actual?.ino !== expected?.ino || actual?.dev !== expected?.dev) {
      throw new Error(`Manifest changed during installation: ${relative}`)
    }
  }
}

async function resolvedStagedTarget(stage: string, target: string): Promise<string> {
  const missing: string[] = []
  let current = target
  while (true) {
    try { await fs.lstat(current); break } catch (error: any) {
      if (error?.code !== 'ENOENT') throw error
      if (current === stage) throw error
      missing.unshift(path.basename(current))
      current = path.dirname(current)
    }
  }
  return path.join(await fs.realpath(current), ...missing)
}

async function validateModuleTree(layout: PackageInstallLayout, directory: string): Promise<void> {
  let visited = 0
  const queue: Array<{ directory: string; depth: number }> = [{ directory, depth: 0 }]
  while (queue.length) {
    const current = queue.pop()!
    if (current.depth > 80) throw new Error('Installed dependency tree exceeds the depth limit.')
    for (const entry of await fs.readdir(current.directory, { withFileTypes: true })) {
      if (++visited > 300_000) throw new Error('Installed dependency tree exceeds the file limit.')
      const full = path.join(current.directory, entry.name)
      if (entry.isDirectory()) { queue.push({ directory: full, depth: current.depth + 1 }); continue }
      if (entry.isFile()) continue
      if (!entry.isSymbolicLink()) throw new Error('Installed dependency tree contains a special file.')
      const link = await fs.readlink(full)
      const target = path.resolve(path.dirname(full), link)
      if (path.isAbsolute(link) || !isInsideInstallRoot(layout.stage, target)) {
        throw new Error('Installed dependencies contain a symlink leaving the isolated package stage.')
      }
      const resolved = await resolvedStagedTarget(layout.stage, target)
      if (!isInsideInstallRoot(layout.stage, resolved)) throw new Error('Installed dependency symlink resolves outside its stage.')
      const relative = path.relative(layout.stage, resolved)
      const inModules = [...layout.modules.keys()].some((root) => isInsideInstallRoot(root, relative))
      const inLocalPackage = layout.packages.filter(Boolean).some((root) => isInsideInstallRoot(root, relative))
      if (!inModules && !inLocalPackage) throw new Error('Dependency symlink points at data that will not be published.')
      if (inLocalPackage && !inModules) {
        const hostTarget = path.join(layout.root, relative)
        const canonical = await fs.realpath(hostTarget)
        if (!isInsideInstallRoot(layout.root, canonical)) throw new Error('Local package symlink leaves the selected workspace.')
      }
    }
  }
}

async function prepareEntries(layout: PackageInstallLayout): Promise<PublicationEntry[]> {
  const entries: PublicationEntry[] = []
  const prepare = async (relative: string, source: string | null, previous: InstallEntryIdentity | null) => {
    await assertParent(layout, relative)
    const target = path.join(layout.root, relative)
    const token = randomUUID()
    const incoming = source ? path.join(path.dirname(target), `.tensorfold-dependency-next-${token}`) : null
    const entry: PublicationEntry = {
      relative, target, incoming, incomingIdentity: null, previous,
      backup: path.join(path.dirname(target), `.tensorfold-dependency-backup-${token}`),
      movedPrevious: false, published: false,
    }
    entries.push(entry)
    if (incoming && source) {
      await fs.rename(source, incoming)
      entry.incomingIdentity = await installEntryIdentity(incoming)
    }
  }
  try {
    for (const [relative, original] of layout.files) {
      const source = path.join(layout.stage, relative)
      const proposed = await readInstallSnapshot(source)
      if (original && !proposed) throw new Error(`Installer removed a required manifest: ${relative}`)
      if (proposed && relative.endsWith('package.json')) {
        if (!original || !unchangedManifestFields(parseInstallJson(original, relative), parseInstallJson(proposed, relative))) {
          throw new Error(`Installer changed non-dependency package metadata: ${relative}`)
        }
      } else if (proposed) validateInstallLockfile(proposed, layout.packages)
      if (original?.sha256 === proposed?.sha256) continue
      if (proposed && original) await fs.chmod(source, original.mode)
      const previous: InstallEntryIdentity | null = original ? {
        ...original, size: original.content.length, kind: 'file',
      } : null
      await prepare(relative, proposed ? source : null, previous)
    }
    for (const [relative, previous] of layout.modules) {
      const source = path.join(layout.stage, relative)
      const proposed = await installEntryIdentity(source)
      if (proposed && proposed.kind !== 'directory') throw new Error(`${relative} is not a dependency directory.`)
      if (proposed) await validateModuleTree(layout, source)
      if (proposed || previous) await prepare(relative, proposed ? source : null, previous)
    }
    return entries
  } catch (error) {
    for (const entry of entries) if (entry.incoming) await removeOwnedEntry(entry.incoming, entry.incomingIdentity)
    throw error
  }
}

async function removeOwnedEntry(target: string, identity: InstallEntryIdentity | null): Promise<void> {
  if (identity && sameInstallEntry(identity, await installEntryIdentity(target))) {
    await fs.rm(target, { recursive: identity.kind === 'directory', force: true })
  }
}

async function rollback(layout: PackageInstallLayout, entries: PublicationEntry[]): Promise<string[]> {
  const recovery: string[] = []
  for (const entry of [...entries].reverse()) {
    try {
      await assertParent(layout, entry.relative)
      if (entry.published && entry.incoming) {
        if (!sameInstallEntry(entry.incomingIdentity, await installEntryIdentity(entry.target))) {
          throw new Error('Published dependency destination changed before rollback.')
        }
        if (entry.publishedSha256 && (await readInstallSnapshot(entry.target))?.sha256 !== entry.publishedSha256) {
          throw new Error('A newer edit to a published manifest must be preserved.')
        }
        if (sameInstallEntry(entry.incomingIdentity, await installEntryIdentity(entry.incoming))) {
          await fs.unlink(entry.incoming)
        }
        await fs.rename(entry.target, entry.incoming)
        entry.published = false
      }
      if (entry.movedPrevious) {
        if (await installEntryIdentity(entry.target)) throw new Error('A concurrent destination prevents rollback.')
        if (!sameInstallEntry(entry.previous, await installEntryIdentity(entry.backup))) throw new Error('Backup changed.')
        await fs.rename(entry.backup, entry.target)
        entry.movedPrevious = false
      }
      if (entry.incoming) await removeOwnedEntry(entry.incoming, entry.incomingIdentity)
    } catch {
      recovery.push(entry.backup, ...(entry.incoming ? [entry.incoming] : []))
    }
  }
  return recovery
}

export async function publishPackageInstallation(
  layout: PackageInstallLayout,
  hooks: PackagePublicationHooks = {},
): Promise<PackagePublicationResult> {
  assertNotCancelled(hooks.signal)
  const entries = await prepareEntries(layout)
  const binPermissions: LocalBinPermissionChange[] = []
  try {
    await assertCanonicalManifests(layout, entries)
    await assertLocalPackageBins(layout.root, layout.bins)
    for (const entry of entries) await assertOriginal(layout, entry)
    for (const entry of entries) {
      assertNotCancelled(hooks.signal)
      await hooks.beforePublish?.(entry.relative)
      assertNotCancelled(hooks.signal)
      await assertCanonicalManifests(layout, entries)
      await assertLocalPackageBins(layout.root, layout.bins)
      await assertOriginal(layout, entry)
      if (entry.incoming && !sameInstallEntry(entry.incomingIdentity, await installEntryIdentity(entry.incoming))) {
        throw new Error('Prepared dependency contents changed before publication.')
      }
      if (entry.previous) {
        await fs.rename(entry.target, entry.backup)
        entry.movedPrevious = true
      }
      if (entry.incoming) {
        if (entry.incomingIdentity?.kind === 'file') {
          entry.publishedSha256 = (await readInstallSnapshot(entry.incoming))?.sha256
          await fs.link(entry.incoming, entry.target)
          entry.published = true
          await fs.unlink(entry.incoming)
        } else await fs.rename(entry.incoming, entry.target)
        entry.published = true
      }
    }
    await publishLocalBinPermissions(layout.root, layout.bins, binPermissions)
    assertNotCancelled(hooks.signal)
    await assertCanonicalManifests(layout, entries)
    await assertLocalPackageBins(layout.root, layout.bins, binPermissions)
  } catch (error: any) {
    const permissionRecovery = await rollbackLocalBinPermissions(layout.root, binPermissions)
    const recovery = await rollback(layout, entries)
    throw new PackagePublicationError(
      `${error?.message ?? error}${recovery.length ? ` Recovery copies were retained: ${recovery.join(', ')}` : ' Previous dependencies were restored.'}`
      + (permissionRecovery.length ? ` Newer executable changes were preserved; review permissions for: ${permissionRecovery.join(', ')}.` : ''),
      recovery,
    )
  }
  const warnings: string[] = []
  for (const entry of entries) {
    if (!entry.movedPrevious) continue
    try { await removeOwnedEntry(entry.backup, entry.previous) } catch {
      warnings.push(`Dependencies are installed, but an owned recovery copy remains at ${entry.backup}.`)
    }
  }
  return { changedPaths: [...entries.map((entry) => entry.relative), ...binPermissions.filter((entry) => entry.applied).map((entry) => entry.before.relative)], warnings }
}
