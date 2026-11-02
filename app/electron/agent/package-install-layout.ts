import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { isRegistryPackageName, isRegistryVersion } from './package-install-command'
import { localPackageBins, stageLocalPackageBins, type LocalPackageBin } from './package-install-bins'

export interface InstallFileSnapshot {
  content: Buffer
  sha256: string
  dev: number
  ino: number
  mode: number
  mtimeMs: number
  ctimeMs: number
}

export interface InstallEntryIdentity {
  dev: number
  ino: number
  size: number
  mtimeMs: number
  ctimeMs: number
  kind: 'file' | 'directory'
}

export interface PackageInstallLayout {
  root: string
  stage: string
  packages: string[]
  files: Map<string, InstallFileSnapshot | null>
  parents: Map<string, InstallEntryIdentity>
  modules: Map<string, InstallEntryIdentity | null>
  bins: LocalPackageBin[]
  stageIdentity: InstallEntryIdentity
}

const MAX_PACKAGE_FILES = 256
const MAX_MANIFEST_BYTES = 1024 * 1024
const MAX_LOCK_BYTES = 32 * 1024 * 1024
const DEPENDENCY_FIELDS = ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies']
const LOCK_FILES = ['package-lock.json', 'npm-shrinkwrap.json']

export function isInsideInstallRoot(root: string, target: string): boolean {
  return target === root || target.startsWith(root + path.sep)
}

export async function installEntryIdentity(target: string): Promise<InstallEntryIdentity | null> {
  const stat = await fs.lstat(target).catch((error: NodeJS.ErrnoException) => {
    if (error.code === 'ENOENT') return null
    throw error
  })
  if (!stat) return null
  if (stat.isSymbolicLink() || (!stat.isDirectory() && !stat.isFile())) {
    throw new Error(`Dependency setup refuses a symlink or special-file destination: ${target}`)
  }
  return {
    dev: stat.dev, ino: stat.ino, size: stat.size,
    mtimeMs: stat.mtimeMs, ctimeMs: stat.ctimeMs,
    kind: stat.isDirectory() ? 'directory' : 'file',
  }
}

export function sameInstallEntry(left: InstallEntryIdentity | null, right: InstallEntryIdentity | null): boolean {
  if (!left || !right) return left === right
  return left.dev === right.dev && left.ino === right.ino && left.kind === right.kind
}

async function safeDirectory(root: string, directory: string): Promise<string> {
  const target = path.resolve(root, directory)
  if (!isInsideInstallRoot(root, target)) throw new Error('A local package path leaves the selected workspace.')
  const segments = path.relative(root, target).split(path.sep).filter(Boolean)
  if (segments.some((segment) => segment.startsWith('.') || segment === 'node_modules')) {
    throw new Error('Local package paths cannot enter hidden directories or node_modules.')
  }
  let current = root
  for (const segment of segments) {
    current = path.join(current, segment)
    const stat = await fs.lstat(current)
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Local package is not an owned directory: ${current}`)
  }
  if (await fs.realpath(target) !== target) throw new Error(`Local package path changed: ${target}`)
  return path.relative(root, target).split(path.sep).join('/')
}

export async function readInstallSnapshot(target: string, limit = MAX_LOCK_BYTES): Promise<InstallFileSnapshot | null> {
  let handle: Awaited<ReturnType<typeof fs.open>>
  try {
    handle = await fs.open(target, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  } catch (error: any) {
    if (error?.code === 'ENOENT') return null
    throw error
  }
  try {
    const before = await handle.stat()
    if (!before.isFile() || before.size > limit) throw new Error(`Dependency manifest is not a bounded file: ${target}`)
    const content = Buffer.alloc(before.size + 1)
    let offset = 0
    while (offset < content.length) {
      const read = await handle.read(content, offset, content.length - offset, offset)
      if (!read.bytesRead) break
      offset += read.bytesRead
    }
    const after = await fs.lstat(target)
    if (after.isSymbolicLink() || offset !== before.size || before.dev !== after.dev || before.ino !== after.ino
      || before.mtimeMs !== after.mtimeMs || before.ctimeMs !== after.ctimeMs) {
      throw new Error(`Dependency manifest changed while reading: ${target}`)
    }
    const bytes = content.subarray(0, offset)
    return {
      content: bytes, sha256: createHash('sha256').update(bytes).digest('hex'),
      dev: before.dev, ino: before.ino, mode: before.mode & 0o777,
      mtimeMs: before.mtimeMs, ctimeMs: before.ctimeMs,
    }
  } finally { await handle.close() }
}

export function parseInstallJson(snapshot: InstallFileSnapshot, label: string): Record<string, any> {
  const parsed = JSON.parse(snapshot.content.toString('utf8'))
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must contain a JSON object.`)
  }
  return parsed
}

async function expandWorkspaces(root: string, manifest: Record<string, any>): Promise<string[]> {
  if (manifest.workspaces === undefined) return []
  const patterns = Array.isArray(manifest.workspaces) ? manifest.workspaces : manifest.workspaces?.packages
  if (!Array.isArray(patterns) || patterns.length > MAX_PACKAGE_FILES) throw new Error('Unsupported npm workspaces declaration.')
  const found = new Set<string>()
  let visits = 0
  for (const pattern of patterns) {
    if (typeof pattern !== 'string' || !pattern || pattern.startsWith('/') || /[\\!{}?\[\]]/.test(pattern)) {
      throw new Error('Workspace globs support ordinary relative paths and one * per path segment.')
    }
    const segments = pattern.replace(/\/$/, '').split('/')
    if (segments.length > 12 || segments.some((part) => !part || part.startsWith('.') || part.includes('**'))) {
      throw new Error('Workspace globs cannot escape the project, use hidden directories or recursive ** patterns.')
    }
    let candidates = ['']
    for (const segment of segments) {
      const next: string[] = []
      for (const base of candidates) {
        if (++visits > 4096) throw new Error('Workspace discovery exceeds the bounded directory limit.')
        const names = segment.includes('*')
          ? (await fs.readdir(path.join(root, base), { withFileTypes: true }))
            .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules')
            .map((entry) => entry.name)
          : [segment]
        const expression = new RegExp(`^${segment.split('*').map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('[^/]*')}$`)
        for (const name of names.filter((value) => expression.test(value))) {
          next.push(await safeDirectory(root, path.join(base, name)))
          if (next.length > MAX_PACKAGE_FILES) throw new Error('Too many npm workspace packages.')
        }
      }
      candidates = next
    }
    for (const candidate of candidates) found.add(candidate)
  }
  return [...found].sort()
}

async function localDependencies(root: string, directory: string, manifest: Record<string, any>): Promise<string[]> {
  const local: string[] = []
  for (const field of DEPENDENCY_FIELDS) {
    if (manifest[field] === undefined) continue
    const entries = manifest[field]
    if (!entries || typeof entries !== 'object' || Array.isArray(entries)) throw new Error(`Invalid ${field} declaration.`)
    for (const [name, spec] of Object.entries(entries)) {
      if (!isRegistryPackageName(name) || typeof spec !== 'string') throw new Error(`Invalid dependency entry: ${name}`)
      if (spec.startsWith('file:')) {
        const value = spec.slice(5)
        if (!value || /[\\\x00]/.test(value) || path.isAbsolute(value)) throw new Error('Use a relative internal file: directory.')
        const dependency = await safeDirectory(root, path.join(directory, value))
        if (!dependency) throw new Error('An internal file: dependency must name a package beneath the workspace root.')
        local.push(dependency)
      } else if (!isRegistryVersion(spec)) {
        throw new Error(`Dependency ${name} needs a registry version or an internal file: directory; external sources are refused.`)
      }
    }
  }
  const validateOverrides = (value: unknown): void => {
    if (typeof value === 'string') {
      if (!value.startsWith('$') && !isRegistryVersion(value)) throw new Error('Only registry dependency overrides are supported.')
    } else if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const child of Object.values(value)) validateOverrides(child)
    } else throw new Error('Invalid npm dependency override.')
  }
  if (manifest.overrides !== undefined) validateOverrides(manifest.overrides)
  return local
}

export function validateInstallLockfile(snapshot: InstallFileSnapshot, packages: readonly string[]): void {
  const lock = parseInstallJson(snapshot, 'npm lockfile')
  if (![2, 3].includes(lock.lockfileVersion) || !lock.packages || typeof lock.packages !== 'object') {
    throw new Error('Dependency setup requires an npm lockfile with lockfileVersion 2 or 3.')
  }
  for (const [location, raw] of Object.entries(lock.packages)) {
    if (location.startsWith('/') || location.split('/').some((part) => part === '..') || location.includes('\\')) {
      throw new Error('npm lockfile contains an unsafe package path.')
    }
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid npm lockfile package.')
    const entry = raw as Record<string, any>
    if (entry.link) {
      if (typeof entry.resolved !== 'string' || !packages.includes(entry.resolved)) {
        throw new Error('npm lockfile links must target declared internal packages.')
      }
    } else if (entry.resolved !== undefined) {
      const url = new URL(String(entry.resolved))
      if (url.protocol !== 'https:' || url.hostname !== 'registry.npmjs.org' || url.username || url.password
        || url.port || url.search || url.hash) {
        throw new Error('Dependency setup accepts lockfile downloads from https://registry.npmjs.org only.')
      }
    }
  }
}

export async function preparePackageInstallLayout(rootPath: string): Promise<PackageInstallLayout> {
  const root = await fs.realpath(rootPath)
  const files = new Map<string, InstallFileSnapshot | null>()
  const parents = new Map<string, InstallEntryIdentity>()
  const modules = new Map<string, InstallEntryIdentity | null>()
  const packages: string[] = []
  const bins = new Map<string, LocalPackageBin>()
  const queue = ['']
  while (queue.length) {
    const directory = queue.shift()!
    if (packages.includes(directory)) continue
    if (packages.length >= MAX_PACKAGE_FILES) throw new Error('Too many local dependency manifests.')
    const parent = path.join(root, directory)
    const identity = await installEntryIdentity(parent)
    if (!identity || identity.kind !== 'directory') throw new Error(`Package directory is missing: ${parent}`)
    parents.set(directory, identity)
    const relative = path.posix.join(directory, 'package.json')
    const snapshot = await readInstallSnapshot(path.join(root, relative), MAX_MANIFEST_BYTES)
    if (!snapshot) throw new Error(`Select a package folder with package.json; missing ${relative}.`)
    files.set(relative, snapshot)
    const manifest = parseInstallJson(snapshot, relative)
    for (const bin of await localPackageBins(root, directory, manifest)) bins.set(bin.relative, bin)
    if (bins.size > 1024) throw new Error('Too many declared local package executables.')
    if (!directory) queue.push(...await expandWorkspaces(root, manifest))
    queue.push(...await localDependencies(root, directory, manifest))
    packages.push(directory)
    const modulesPath = path.posix.join(directory, 'node_modules')
    const existing = await installEntryIdentity(path.join(root, modulesPath))
    if (existing && existing.kind !== 'directory') throw new Error(`${modulesPath} must be a directory.`)
    modules.set(modulesPath, existing)
  }
  for (const name of LOCK_FILES) {
    const snapshot = await readInstallSnapshot(path.join(root, name))
    if (snapshot) validateInstallLockfile(snapshot, packages)
    files.set(name, snapshot)
  }
  const stage = await fs.mkdtemp(path.join(path.dirname(root), '.tensorfold-install-'))
  await fs.chmod(stage, 0o700)
  try {
    await fs.mkdir(path.join(stage, '.npm-tmp'), { mode: 0o700 })
    await fs.writeFile(path.join(stage, '.npm-userconfig'), '', { flag: 'wx', mode: 0o600 })
    await fs.writeFile(path.join(stage, '.npm-globalconfig'), '', { flag: 'wx', mode: 0o600 })
    for (const [relative, snapshot] of files) {
      if (!snapshot) continue
      const destination = path.join(stage, relative)
      await fs.mkdir(path.dirname(destination), { recursive: true, mode: 0o700 })
      await fs.writeFile(destination, snapshot.content, { flag: 'wx', mode: 0o600 })
    }
    await stageLocalPackageBins(stage, [...bins.values()])
    return { root, stage, packages, files, parents, modules, bins: [...bins.values()], stageIdentity: (await installEntryIdentity(stage))! }
  } catch (error) {
    await fs.rm(stage, { recursive: true, force: true })
    throw error
  }
}
