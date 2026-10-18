import { createHash, randomUUID } from 'node:crypto'
import { constants, promises as fs, realpathSync } from 'node:fs'
import path from 'node:path'
import { freeDiskBytes } from './paths'

export const LEARNING_MODEL = 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
const MANIFEST = 'profile.json'
const RESERVE = 1024 ** 3
const UID = process.getuid?.()
const NATIVE_BACKUPS = new Set(['.tensorfold-original', '.tensorfold-original.partial'])

function within(root: string, file: string): boolean {
  const relative = path.relative(root, file)
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)
}

async function ownedDirectory(directory: string, create = false): Promise<void> {
  if (create) await fs.mkdir(directory, { mode: 0o700 })
  const stat = await fs.lstat(directory)
  if (!stat.isDirectory() || stat.isSymbolicLink() || (UID != null && stat.uid !== UID) || (stat.mode & 0o077) !== 0) throw new Error('Living Weights needs a private app-owned folder with no links.')
  if (await fs.realpath(directory) !== directory) throw new Error('Living Weights cannot use a redirected folder.')
}

export class LearningCopies {
  readonly root: string
  private readonly canonicalHome: string

  constructor(private readonly appHome: string) {
    let ancestor = path.resolve(appHome)
    const rest: string[] = []
    for (;;) {
      try { this.canonicalHome = path.join(realpathSync(ancestor), ...rest); break } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
        rest.unshift(path.basename(ancestor)); ancestor = path.dirname(ancestor)
      }
    }
    this.root = path.join(this.canonicalHome, 'engine', 'learning')
  }

  private async directories(): Promise<void> {
    await fs.mkdir(this.appHome, { recursive: true, mode: 0o700 })
    const home = path.resolve(this.appHome)
    const homeStat = await fs.lstat(home)
    if (homeStat.isSymbolicLink() || await fs.realpath(home) !== this.canonicalHome || !homeStat.isDirectory() || (UID != null && homeStat.uid !== UID)) throw new Error('The app data folder is redirected; Living Weights cannot prepare a model safely.')
    const engine = path.dirname(this.root)
    await fs.mkdir(engine, { recursive: true, mode: 0o700 })
    const stat = await fs.lstat(engine)
    if (stat.isSymbolicLink() || !stat.isDirectory() || (UID != null && stat.uid !== UID)) throw new Error('The engine data folder is not app-owned.')
    await fs.mkdir(this.root, { recursive: true, mode: 0o700 })
    await ownedDirectory(this.root)
  }

  folder(repo: string): string {
    if (repo !== LEARNING_MODEL) throw new Error('Living Weights currently supports only Nemotron 3.5 Lightning MLX 4-bit.')
    return path.join(this.root, createHash('sha256').update(repo).digest('hex').slice(0, 24))
  }

  graphFile(repo: string): string { return path.join(this.folder(repo), 'facts.json') }

  async purge(): Promise<{ removedPaths: string[]; files: number }> {
    // Also checks the journal's engine parent authority when no learned model exists yet.
    await this.directories()
    await validatePrivateTree(this.root, path.join(this.folder(LEARNING_MODEL), 'model'))
    const count = async (directory: string): Promise<number> => {
      let files = 0
      for (const entry of await fs.readdir(directory, { withFileTypes: true })) files += entry.isDirectory() ? await count(path.join(directory, entry.name)) : 1
      return files
    }
    const files = await count(this.root)
    await ownedDirectory(this.root)
    await fs.rm(this.root, { recursive: true })
    return { removedPaths: [this.root], files }
  }

  async existing(repo: string): Promise<string | null> {
    const directory = this.folder(repo)
    try { await fs.lstat(directory) } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
      throw error
    }
    await this.directories()
    await ownedDirectory(directory)
    const manifest = path.join(directory, MANIFEST)
    await privateFile(manifest)
    if ((await fs.stat(manifest)).size > 64 * 1024) throw new Error('The Living Weights profile is invalid.')
    const doc = JSON.parse(await fs.readFile(manifest, 'utf8'))
    if (doc.version !== 1 || doc.repo !== repo || typeof doc.sourceRevision !== 'string' || !/^[a-f0-9]{40}$/i.test(doc.sourceRevision)) throw new Error('The Living Weights profile is invalid. Preserve it before repairing the app data.')
    const model = path.join(directory, 'model')
    await ownedDirectory(model)
    await validatePrivateTree(model, model)
    const configFile = path.join(model, 'config.json')
    if ((await fs.stat(configFile)).size > 1024 * 1024) throw new Error('The private model configuration is too large.')
    const config = JSON.parse(await fs.readFile(configFile, 'utf8'))
    assertLearningConfig(config)
    // Native graph writes use a renamed file. Links or shared files must never be passed to it.
    try { await privateFile(this.graphFile(repo), false) } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
    return model
  }

  async prepare(repo: string, source: string, cacheRepo: string, onProgress: (copiedBytes: number, totalBytes: number) => void, signal?: AbortSignal): Promise<string> {
    await this.directories()
    const existing = await this.existing(repo)
    if (existing) return existing
    const realRepo = await fs.realpath(cacheRepo)
    const realSource = await fs.realpath(source)
    if (realSource !== source || !within(realRepo, realSource) || !(await fs.lstat(source)).isDirectory()) throw new Error('The original model must be an installed immutable Hugging Face checkpoint.')
    const revision = path.basename(source)
    if (!/^[a-f0-9]{40}$/i.test(revision)) throw new Error('The original model checkpoint has no immutable revision.')
    const configPath = path.join(source, 'config.json')
    if ((await fs.stat(configPath)).size > 1024 * 1024) throw new Error('The model configuration is too large.')
    const config = JSON.parse(await fs.readFile(configPath, 'utf8'))
    assertLearningConfig(config)
    if (config.tensorfold_learned_neurons != null) throw new Error('Prepare Living Weights from the untouched original checkpoint, not an already taught download.')
    const files: { relative: string; real: string; size: number; dev: number; ino: number; mtimeMs: number }[] = []
    const walk = async (directory: string, depth: number): Promise<void> => {
      if (depth > 8) throw new Error('The original model has too many nested folders.')
      for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
        signal?.throwIfAborted()
        const full = path.join(directory, entry.name)
        const real = await fs.realpath(full)
        if (!within(realRepo, real)) throw new Error('A model file points outside its Hugging Face repository; refusing to copy it.')
        const stat = await fs.stat(real)
        if (entry.isDirectory()) { await walk(full, depth + 1); continue }
        if (!stat.isFile() || (entry.isSymbolicLink() && stat.isDirectory())) throw new Error('The original model contains an unsupported file or linked directory.')
        if (files.length >= 20_000) throw new Error('The original model contains too many files.')
        files.push({ relative: path.relative(source, full), real, size: stat.size, dev: stat.dev, ino: stat.ino, mtimeMs: stat.mtimeMs })
      }
    }
    await walk(source, 0)
    const total = files.reduce((sum, file) => sum + file.size, 0)
    if (!Number.isSafeInteger(total) || !files.some((file) => file.relative.endsWith('.safetensors') && file.size > 0)) throw new Error('The original checkpoint has no readable model weights.')
    const free = await freeDiskBytes(this.root)
    // Clones save space initially, but a lesson may rewrite every shard. Reserve the full copy size.
    if (free == null || free < total + RESERVE) throw new Error('Living Weights needs free disk space for a full independent model copy plus 1 GB for saving changes.')
    const temporary = path.join(this.root, `.prepare-${randomUUID()}`)
    await ownedDirectory(temporary, true)
    const model = path.join(temporary, 'model')
    await ownedDirectory(model, true)
    let copied = 0
    onProgress(0, total)
    try {
      for (const file of files) {
        signal?.throwIfAborted()
        const target = path.join(model, file.relative)
        await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 })
        await fs.copyFile(file.real, target, constants.COPYFILE_EXCL | constants.COPYFILE_FICLONE)
        await fs.chmod(target, 0o600)
        const durable = await fs.open(target, 'r+')
        try { await durable.sync() } finally { await durable.close() }
        const before = await fs.stat(file.real)
        if (before.dev !== file.dev || before.ino !== file.ino || before.size !== file.size || before.mtimeMs !== file.mtimeMs) throw new Error('The downloaded model changed while making the private copy; try again after other model tools finish.')
        const after = await privateFile(target)
        if (after.dev === file.dev && after.ino === file.ino) throw new Error('The model copy shares a file with the original; refusing to enable learning.')
        copied += file.size
        onProgress(copied, total)
      }
      await validatePrivateTree(model)
      const manifest = await fs.open(path.join(temporary, MANIFEST), 'wx', 0o600)
      try { await manifest.writeFile(JSON.stringify({ version: 1, repo, sourceRevision: revision, preparedAt: Date.now(), bytes: total })); await manifest.sync() } finally { await manifest.close() }
      signal?.throwIfAborted()
      await fs.rename(temporary, this.folder(repo))
      const directory = await fs.open(this.root, 'r')
      try { await directory.sync() } finally { await directory.close() }
      return path.join(this.folder(repo), 'model')
    } finally { await fs.rm(temporary, { recursive: true, force: true }) }
  }
}

export function assertLearningConfig(config: any): void {
  if (config?.model_type !== 'nemotron_h' || config?.quantization?.bits !== 4 || config?.quantization?.group_size !== 64) throw new Error('Living Weights requires the Nemotron 3.5 Lightning MLX 4-bit checkpoint with group size 64.')
}

async function privateFile(file: string, strictMode = true) {
  const stat = await fs.lstat(file)
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || (UID != null && stat.uid !== UID) || (strictMode && (stat.mode & 0o077) !== 0)) throw new Error('A Living Weights file is linked, shared or not private; refusing to teach this folder.')
  return stat
}

async function nativeBackupDirectory(directory: string, model: string): Promise<void> {
  // v1.0.12 snapshots use mkdir(0755), then rename .partial to the completed name.
  // Their only privacy boundary is this verified 0700 private model, never a cache folder.
  await ownedDirectory(model)
  const stat = await fs.lstat(directory)
  const mode = stat.mode & 0o7777
  if (!stat.isDirectory() || stat.isSymbolicLink() || (UID != null && stat.uid !== UID) || (mode & 0o700) !== 0o700 || (mode & 0o7022) !== 0) throw new Error('A Living Weights native backup is linked, shared or writable by another user.')
  if (await fs.realpath(directory) !== directory) throw new Error('Living Weights cannot use a redirected backup folder.')
}

async function validatePrivateTree(directory: string, nativeModel?: string, depth = 0, budget = { files: 0 }): Promise<void> {
  if (depth > 8) throw new Error('The private model has too many nested folders.')
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    if (++budget.files > 20_000) throw new Error('The private model contains too many files.')
    const full = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      if (directory === nativeModel && NATIVE_BACKUPS.has(entry.name)) await nativeBackupDirectory(full, nativeModel)
      else await ownedDirectory(full)
      await validatePrivateTree(full, nativeModel, depth + 1, budget)
    }
    else await privateFile(full, false) // native checkpoint rewrites may use 0644 inside this private 0700 tree.
  }
}
