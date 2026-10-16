import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Where the bundled engine lives, where the shared Hugging Face cache is, and the repo-id rules
// every engine command and cache path goes through.

export interface EngineBinary {
  command: string
  args: string[]
}

export const ENGINE_BINARY_OVERRIDE_ENV = 'TENSORFOLD_STUDIO_ENGINE_BIN'

export interface ResolveBinaryInput {
  packaged: boolean
  resourcesPath: string
  appRoot: string
  env: NodeJS.ProcessEnv
}

// Packaged: Contents/Resources/engine/bin. Dev: build/engine/bin from fetch-engine. The override env
// is honoured only in dev, so a packaged app always runs the binary it shipped with.
export function resolveEngineBinaryPath(input: ResolveBinaryInput): string {
  if (input.packaged) return path.join(input.resourcesPath, 'engine', 'bin', 'tensorfold-native')
  const override = input.env[ENGINE_BINARY_OVERRIDE_ENV]?.trim()
  if (override) return path.resolve(override)
  return path.join(input.appRoot, 'build', 'engine', 'bin', 'tensorfold-native')
}

// The engine's own rule (cli/hub.zig cacheDir): HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub.
export function hubCacheDir(env: NodeJS.ProcessEnv = process.env): string {
  const explicit = env.HF_HUB_CACHE?.trim()
  if (explicit) return path.resolve(explicit)
  const hfHome = env.HF_HOME?.trim()
  if (hfHome) return path.join(path.resolve(hfHome), 'hub')
  return path.join(env.HOME?.trim() || os.homedir(), '.cache', 'huggingface', 'hub')
}

const REPO_PART = /^[A-Za-z0-9][A-Za-z0-9._-]{0,94}[A-Za-z0-9]$|^[A-Za-z0-9]$/

// owner/name with Hugging Face's own limits (no '--' or '..', no leading or trailing '.' or '-').
export function isRepoId(value: unknown): value is string {
  if (typeof value !== 'string') return false
  const parts = value.split('/')
  if (parts.length !== 2) return false
  return parts.every((part) => REPO_PART.test(part) && !part.includes('--') && !part.includes('..'))
}

export function assertRepoId(value: unknown): string {
  if (!isRepoId(value)) throw new Error('Expected a Hugging Face repo id like owner/name.')
  return value
}

// A branch, tag or 40-hex commit: one URL path segment, as pull.zig's revisionOk accepts.
export function isRevision(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 128 && /^[A-Za-z0-9._-]+$/.test(value) && !value.includes('..')
}

export function repoCacheDir(hub: string, repo: string): string {
  const [owner, name] = assertRepoId(repo).split('/')
  return path.join(hub, `models--${owner}--${name}`)
}

// Bytes actually on disk under a folder. pull.zig sizes big files to their full length before the
// pieces arrive, so allocated blocks (not st_size) measure progress; a finished file counts as its size.
export async function allocatedBytes(dir: string): Promise<number> {
  let total = 0
  let entries: import('node:fs').Dirent[]
  try {
    entries = await fs.readdir(dir, { withFileTypes: true })
  } catch {
    return 0
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      total += await allocatedBytes(full)
    } else if (entry.isFile()) {
      try {
        const info = await fs.lstat(full)
        total += Math.min(info.size, info.blocks * 512)
      } catch { /* removed mid-walk */ }
    }
  }
  return total
}

export async function repoCachedBytes(hub: string, repo: string): Promise<number> {
  return allocatedBytes(path.join(repoCacheDir(hub, repo), 'blobs'))
}

// Free space on the volume that holds (or will hold) the cache: the nearest existing ancestor.
export async function freeDiskBytes(target: string): Promise<number | null> {
  let dir = path.resolve(target)
  for (;;) {
    try {
      const stats = await fs.statfs(dir)
      return Number(stats.bavail) * Number(stats.bsize)
    } catch {
      const parent = path.dirname(dir)
      if (parent === dir) return null
      dir = parent
    }
  }
}

// A cached snapshot exists when refs/main names a 40-hex sha whose snapshot folder is present.
export async function hasCachedSnapshot(hub: string, repo: string): Promise<boolean> {
  const root = repoCacheDir(hub, repo)
  try {
    const ref = (await fs.readFile(path.join(root, 'refs', 'main'), 'utf8')).trim()
    if (!/^[0-9a-fA-F]{40}$/.test(ref)) return false
    return (await fs.stat(path.join(root, 'snapshots', ref))).isDirectory()
  } catch {
    return false
  }
}
