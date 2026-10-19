import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { assertRepoId, isRepoId, repoCacheDir } from './paths'

const SHA = /^[a-f0-9]{40}$/
const MAX_BYTES = 128 * 1024
const INVALID = 'The saved model revisions are invalid. Preserve the registry and repair it before loading or downloading models.'

// App-owned selections keep a reviewed checkpoint independent of another HF client's refs/main.
// Missing registries retain discovery of existing legacy caches; malformed or stale selections
// never silently fall back to another checkpoint. The shared cache's refs are not edited here.
export class ModelRevisions {
  private writes: Promise<unknown> = Promise.resolve()
  readonly file: string

  constructor(appHome: string, private readonly hub: () => string) {
    this.file = path.join(appHome, 'engine', 'model-revisions.json')
  }

  async read(): Promise<Record<string, string>> {
    let stat
    try { stat = await fs.lstat(this.file) } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {}
      throw error
    }
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > MAX_BYTES) throw new Error(INVALID)
    let doc: unknown
    try { doc = JSON.parse(await fs.readFile(this.file, 'utf8')) } catch { throw new Error(INVALID) }
    const value = doc as { version?: unknown; revisions?: unknown }
    if (!value || value.version !== 1 || Object.keys(value).some((key) => !['version', 'revisions'].includes(key)) || !value.revisions || typeof value.revisions !== 'object' || Array.isArray(value.revisions)) throw new Error(INVALID)
    const revisions = Object.entries(value.revisions)
    if (revisions.length > 512 || revisions.some(([repo, sha]) => !isRepoId(repo) || typeof sha !== 'string' || !SHA.test(sha))) throw new Error(INVALID)
    return Object.fromEntries(revisions) as Record<string, string>
  }

  async selected(repo: string): Promise<string | null> {
    const sha = (await this.read())[assertRepoId(repo)]
    return sha ? this.snapshot(repo, sha) : null
  }

  async snapshot(repo: string, sha: string, boundHub?: string): Promise<string> {
    assertRepoId(repo)
    if (!SHA.test(sha)) throw new Error(INVALID)
    const missing = () => new Error(`The selected checkpoint for ${repo} is missing or unreadable. Download it again in Models.`)
    try {
      const hub = await fs.realpath(boundHub ?? this.hub())
      const directory = path.join(repoCacheDir(hub, repo), 'snapshots', sha)
      // Canonical directory authority: no repo/snapshots/checkpoint symlink may redirect the CLI.
      if (await fs.realpath(directory) !== directory || !(await fs.lstat(directory)).isDirectory()) throw missing()
      if (!(await fs.stat(path.join(directory, 'config.json'))).isFile()) throw missing()
      const files = await fs.readdir(directory)
      let weights = false
      for (const name of files.filter((name) => name.endsWith('.safetensors'))) {
        const stat = await fs.stat(path.join(directory, name)) // HF's normal blob symlinks are allowed.
        if (!stat.isFile() || stat.size <= 0) throw missing()
        weights = true
      }
      if (!weights) throw missing()
      return directory
    } catch { throw missing() }
  }

  async snapshotBytes(directory: string): Promise<number> {
    let bytes = 0
    for (const name of await fs.readdir(directory)) {
      if (!name.endsWith('.safetensors')) continue
      bytes += (await fs.stat(path.join(directory, name))).size
    }
    return bytes
  }

  // boundHub pins verification to the cache root an operation started on, so a later cache-location change cannot verify against another root.
  commit(selections: { repo: string; revision: string }[], boundHub?: string): Promise<void> {
    return this.mutate(async (revisions) => {
      for (const selection of selections) {
        await this.snapshot(selection.repo, selection.revision, boundHub)
        revisions[selection.repo] = selection.revision
      }
    })
  }

  remove(repos: string[]): Promise<void> {
    return this.mutate(async (revisions) => { for (const repo of repos) delete revisions[assertRepoId(repo)] })
  }

  private mutate(change: (revisions: Record<string, string>) => Promise<void>): Promise<void> {
    const next = this.writes.then(async () => {
      const revisions = await this.read()
      await change(revisions)
      const directory = path.dirname(this.file)
      await fs.mkdir(directory, { recursive: true, mode: 0o700 })
      const stat = await fs.lstat(directory)
      if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(INVALID)
      const temporary = path.join(directory, `.model-revisions-${randomUUID()}.tmp`)
      try {
        const file = await fs.open(temporary, 'wx', 0o600)
        try { await file.writeFile(JSON.stringify({ version: 1, revisions })); await file.sync() } finally { await file.close() }
        await fs.rename(temporary, this.file)
        const dir = await fs.open(directory, 'r')
        try { await dir.sync() } finally { await dir.close() }
      } finally { await fs.rm(temporary, { force: true }) }
    })
    this.writes = next.catch(() => {})
    return next
  }
}
