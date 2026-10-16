import { createHash, randomUUID } from 'node:crypto'
import { constants, promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { EngineCapabilities, EngineLibraryFolder, EngineLibraryModel, EngineLibraryScanOptions, EngineModelLibrarySnapshot } from '../types'
import { formatName } from './cli'
import { isRepoId } from './paths'

const STATE_LIMIT = 2 * 1024 * 1024
// The native engine reads checkpoint metadata well above this cap; the limit only guards the scan's bounded read loop.
const JSON_LIMIT = 64 * 1024 * 1024
const DEFAULT_LIMITS = { directories: 512, entries: 12_000, models: 256, depth: 4, metadataBytes: 256 * 1024 * 1024, durationMs: 15_000 }
const INVALID = 'The local model library is invalid. Preserve its settings and repair them before scanning or loading models.'
const LOCAL_ID = /^local\/[a-f0-9]{64}$/
const SHA = /^[a-f0-9]{40}$/
const GIB = 1024 ** 3

export function isLocalModelId(value: unknown): boolean { return typeof value === 'string' && LOCAL_ID.test(value) }
const digest = (value: string) => createHash('sha256').update(value).digest('hex')
const contains = (root: string, target: string) => target === root || target.startsWith(`${root}${path.sep}`)
const plain = (value: unknown): value is Record<string, any> => !!value && typeof value === 'object' && !Array.isArray(value)

interface Root { id: string; path: string; source: 'huggingface' | 'folder' }
interface State { version: 1; includeHuggingFace: boolean; folders: EngineLibraryFolder[]; models: EngineLibraryModel[]; scannedAt: number | null; warnings: string[]; truncated: boolean }
interface ScanBudget { directories: number; entries: number; metadataBytes: number; deadline: number; signal?: AbortSignal }
class Limit extends Error {}
class Cancelled extends Error {}

export interface ModelLibraryOptions {
  appHome: string
  env: NodeJS.ProcessEnv
  capabilities: () => Promise<EngineCapabilities>
  memoryLimitBytes: () => number
  limits?: Partial<typeof DEFAULT_LIMITS>
}

// A local path is admitted only by a native folder chooser or an explicit HF-cache scan.
// No checkpoint files are written, followed outside the admitted roots, or removed here.
export class ModelLibrary {
  readonly file: string
  private writes: Promise<unknown> = Promise.resolve()
  private running: { controller: AbortController; promise: Promise<EngineModelLibrarySnapshot> } | null = null
  private cancelled = false
  private readonly limits: typeof DEFAULT_LIMITS

  constructor(private readonly options: ModelLibraryOptions) {
    this.file = path.join(options.appHome, 'engine', 'model-library.json')
    this.limits = { ...DEFAULT_LIMITS, ...options.limits }
  }

  private empty(): State { return { version: 1, includeHuggingFace: false, folders: [], models: [], scannedAt: null, warnings: [], truncated: false } }

  private async read(): Promise<State> {
    let stat
    try { stat = await fs.lstat(this.file) } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return this.empty(); throw error }
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > STATE_LIMIT) throw new Error(INVALID)
    let doc: any
    try { doc = JSON.parse(await fs.readFile(this.file, 'utf8')) } catch { throw new Error(INVALID) }
    if (!plain(doc) || doc.version !== 1 || typeof doc.includeHuggingFace !== 'boolean' || !Array.isArray(doc.folders) || doc.folders.length > 16 || !Array.isArray(doc.models) || doc.models.length > 256 || !Array.isArray(doc.warnings) || doc.warnings.length > 32 || doc.warnings.some((v: unknown) => typeof v !== 'string') || typeof doc.truncated !== 'boolean' || (doc.scannedAt !== null && (!Number.isSafeInteger(doc.scannedAt) || doc.scannedAt <= 0))) throw new Error(INVALID)
    const folderIds = new Set<string>()
    for (const folder of doc.folders) {
      if (!plain(folder) || typeof folder.path !== 'string' || !path.isAbsolute(folder.path) || folder.path.length > 4096 || folder.id !== digest(folder.path) || typeof folder.label !== 'string' || folder.label !== path.basename(folder.path) || folderIds.has(folder.id)) throw new Error(INVALID)
      folderIds.add(folder.id)
    }
    const ids = new Set<string>()
    for (const model of doc.models) {
      if (!plain(model) || !isLocalModelId(model.id) || typeof model.path !== 'string' || !path.isAbsolute(model.path) || model.path.length > 4096 || model.id !== `local/${digest(model.path)}` || typeof model.folderId !== 'string' || !['folder', 'huggingface'].includes(model.source) || typeof model.title !== 'string' || model.title.length > 512 || (model.repo !== null && !isRepoId(model.repo)) || (model.revision !== null && (typeof model.revision !== 'string' || !SHA.test(model.revision))) || !['complete', 'compatible', 'loadable'].every((key) => typeof model[key] === 'boolean') || ids.has(model.id)) throw new Error(INVALID)
      if (model.source === 'folder' && !folderIds.has(model.folderId)) throw new Error(INVALID)
      ids.add(model.id)
    }
    return doc as unknown as State
  }

  private mutate<T>(change: (state: State) => Promise<T>): Promise<T> {
    const next = this.writes.then(async () => {
      const state = await this.read()
      const result = await change(state)
      const dir = path.dirname(this.file)
      await fs.mkdir(dir, { recursive: true, mode: 0o700 })
      const info = await fs.lstat(dir)
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(INVALID)
      const temporary = path.join(dir, `.model-library-${randomUUID()}.tmp`)
      try {
        const file = await fs.open(temporary, 'wx', 0o600)
        try { await file.writeFile(JSON.stringify(state)); await file.sync() } finally { await file.close() }
        await fs.rename(temporary, this.file)
      } finally { await fs.rm(temporary, { force: true }) }
      return result
    })
    this.writes = next.catch(() => {})
    return next
  }

  async snapshot(): Promise<EngineModelLibrarySnapshot> {
    const { version: _version, ...state } = await this.read()
    return { ...state, scanning: this.running !== null, cancelled: this.cancelled }
  }

  async addFolder(selected: string): Promise<EngineModelLibrarySnapshot> {
    await this.cancel()
    if (typeof selected !== 'string' || !path.isAbsolute(selected) || selected.length > 4096) throw new Error('Choose a local model folder.')
    const directory = await fs.realpath(selected)
    if (!(await fs.lstat(directory)).isDirectory()) throw new Error('Choose a folder, rather than a file.')
    if (directory === path.parse(directory).root || directory === path.resolve(this.options.env.HOME || os.homedir())) throw new Error('Choose the folder containing your models, rather than your home folder or an entire disk.')
    await this.mutate(async (state) => {
      if (state.folders.some((folder) => folder.path === directory)) return
      if (state.folders.length >= 16) throw new Error('You can add up to 16 model folders.')
      state.folders.push({ id: digest(directory), path: directory, label: path.basename(directory) })
    })
    return this.scan()
  }

  async removeFolder(id: string): Promise<EngineModelLibrarySnapshot> {
    if (typeof id !== 'string' || !/^[a-f0-9]{64}$/.test(id)) throw new Error('Choose a saved model folder.')
    await this.cancel()
    await this.mutate(async (state) => {
      state.folders = state.folders.filter((folder) => folder.id !== id)
      state.models = state.models.filter((model) => model.folderId !== id)
    })
    return this.snapshot()
  }

  async cancel(): Promise<EngineModelLibrarySnapshot> {
    const running = this.running
    if (running) { running.controller.abort(); await running.promise }
    return this.snapshot()
  }

  scan(options: EngineLibraryScanOptions = {}): Promise<EngineModelLibrarySnapshot> {
    if (this.running) {
      if (options.includeHuggingFace !== undefined) return Promise.reject(new Error('Stop the current scan before changing Hugging Face cache discovery.'))
      return this.running.promise.then(() => this.snapshot())
    }
    const controller = new AbortController()
    this.cancelled = false
    const promise = this.scanNow(options, controller.signal).finally(() => { this.running = null })
    this.running = { controller, promise }
    return promise.then(() => this.snapshot())
  }

  private async roots(state: State): Promise<Root[]> {
    const roots: Root[] = state.folders.map((folder) => ({ ...folder, source: 'folder' }))
    if (state.includeHuggingFace) {
      const env = this.options.env
      const candidates = [env.HF_HUB_CACHE, env.HF_HOME ? path.join(env.HF_HOME, 'hub') : null, path.join(env.HOME || os.homedir(), '.cache', 'huggingface', 'hub')]
      for (const candidate of candidates) {
        if (!candidate?.trim()) continue
        const canonical = await fs.realpath(path.resolve(candidate)).catch(() => path.resolve(candidate))
        if (!roots.some((root) => root.path === canonical)) roots.push({ id: `hf-${digest(canonical)}`, path: canonical, source: 'huggingface' })
      }
    }
    return roots
  }

  private check(budget: ScanBudget): void {
    if (budget.signal?.aborted) throw new Cancelled()
    if (budget.directories > this.limits.directories || budget.entries > this.limits.entries || budget.metadataBytes > this.limits.metadataBytes || Date.now() > budget.deadline) throw new Limit('The scan reached its safety limit. Choose a more specific model folder to find the remaining models.')
  }

  private async list(directory: string, budget: ScanBudget): Promise<import('node:fs').Dirent[]> {
    this.check(budget)
    budget.directories++
    const entries = []
    const opened = await fs.opendir(directory)
    try {
      for await (const entry of opened) { budget.entries++; this.check(budget); entries.push(entry) }
    } finally { await opened.close().catch(() => {}) }
    return entries.sort((a, b) => a.name.localeCompare(b.name))
  }

  private async safeFile(directory: string, name: string, blobRoot: string | null): Promise<{ path: string; stat: import('node:fs').Stats }> {
    if (!name || path.basename(name) !== name || name === '.' || name === '..') throw new Error('The checkpoint references an invalid filename.')
    const file = path.join(directory, name)
    const stat = await fs.lstat(file)
    let actual = file
    if (stat.isSymbolicLink()) {
      if (!blobRoot || await fs.realpath(blobRoot) !== blobRoot) throw new Error('The checkpoint contains a link outside its model folder.')
      actual = await fs.realpath(file)
      if (!contains(blobRoot, actual) || path.dirname(actual) !== blobRoot || actual.endsWith('.incomplete') || actual.endsWith('.ranges')) throw new Error('A Hugging Face snapshot link leaves its own blob cache or is still downloading.')
    } else if (!stat.isFile()) throw new Error('The checkpoint contains a non-file model asset.')
    const info = await fs.lstat(actual)
    if (!info.isFile() || info.isSymbolicLink() || info.size <= 0) throw new Error('A model file is missing, empty, or still downloading.')
    return { path: actual, stat: info }
  }

  private async json(directory: string, name: string, blobRoot: string | null, budget: ScanBudget): Promise<any> {
    const file = await this.safeFile(directory, name, blobRoot)
    if (file.stat.size > JSON_LIMIT) throw new Error('The checkpoint metadata is too large to inspect safely.')
    budget.metadataBytes += file.stat.size; this.check(budget)
    const handle = await fs.open(file.path, constants.O_RDONLY | constants.O_NOFOLLOW)
    try {
      const stat = await handle.stat()
      if (!stat.isFile() || stat.size !== file.stat.size) throw new Error('The checkpoint changed during inspection; scan again when its download has finished.')
      const data = Buffer.alloc(stat.size)
      const read = await handle.read(data, 0, data.length, 0)
      if (read.bytesRead !== data.length) throw new Error('The checkpoint metadata is incomplete.')
      return JSON.parse(data.toString('utf8'))
    } finally { await handle.close() }
  }

  private async weight(directory: string, name: string, blobRoot: string | null, budget: ScanBudget): Promise<{ bytes: number; identity: string }> {
    const file = await this.safeFile(directory, name, blobRoot)
    const handle = await fs.open(file.path, constants.O_RDONLY | constants.O_NOFOLLOW)
    try {
      const stat = await handle.stat()
      const prefix = Buffer.alloc(8)
      if (stat.size < 10 || (await handle.read(prefix, 0, 8, 0)).bytesRead !== 8) throw new Error('The safetensors file is incomplete.')
      const length = Number(prefix.readBigUInt64LE())
      if (!Number.isSafeInteger(length) || length < 2 || length > JSON_LIMIT || length + 8 >= stat.size) throw new Error('The safetensors header is invalid or incomplete.')
      budget.metadataBytes += length; this.check(budget)
      const header = Buffer.alloc(length)
      if ((await handle.read(header, 0, length, 8)).bytesRead !== length) throw new Error('The safetensors header is incomplete.')
      const parsed = JSON.parse(header.toString('utf8'))
      if (!plain(parsed)) throw new Error('The safetensors header is invalid.')
      const tensors = Object.entries(parsed).filter(([key]) => key !== '__metadata__')
      if (!tensors.length || tensors.some(([, tensor]) => !plain(tensor) || !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2 || !tensor.data_offsets.every((v: any) => Number.isSafeInteger(v) && v >= 0) || tensor.data_offsets[1] <= tensor.data_offsets[0] || tensor.data_offsets[1] > stat.size - 8 - length)) throw new Error('The safetensors payload is incomplete or invalid.')
      return { bytes: stat.size, identity: `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}` }
    } finally { await handle.close() }
  }

  private async inspect(root: Root, directory: string, rawNames: string[], caps: EngineCapabilities, budget: ScanBudget, repo: string | null = null, revision: string | null = null, blobRoot: string | null = null): Promise<{ model: EngineLibraryModel; fingerprint: string }> {
    // AppleDouble companions (._file) hold resource forks, never checkpoint assets, so drop them before inspection.
    const names = rawNames.filter((name) => !name.startsWith('._'))
    const model: EngineLibraryModel = { id: `local/${digest(directory)}`, repo, title: repo ?? path.basename(directory), path: directory, source: blobRoot ? 'huggingface' : 'folder', folderId: root.id, revision, modelType: null, format: null, complete: false, compatible: false, loadable: false, reason: null, weightsBytes: null, memoryFloorBytes: null, fitsMemory: null }
    let fingerprint = directory
    try {
      if (await fs.realpath(directory) !== directory) throw new Error('The model folder was redirected by a symbolic link; select its real folder instead.')
      if (names.some((name) => /\.(gguf|bin|pt|pth)$/i.test(name)) && !names.some((name) => name.endsWith('.safetensors'))) { model.format = names.some((name) => name.endsWith('.gguf')) ? 'GGUF' : 'PyTorch'; throw new Error('The bundled engine loads supported MLX safetensors checkpoints; this model uses another format.') }
      const config = await this.json(directory, 'config.json', blobRoot, budget)
      if (!plain(config) || typeof config.model_type !== 'string') throw new Error('The checkpoint has no readable model_type in config.json.')
      model.modelType = config.model_type
      model.drafterOnly = !!config.dflash_config
      model.format = formatName(config) ?? 'full precision'
      const formats = caps.families[model.modelType]
      model.compatible = !!formats && !config.dflash_config && (formats.length === 0 || formats.includes(formatName(config) ?? ''))
      const weights = names.filter((name) => name.endsWith('.safetensors'))
      if (!weights.length) throw new Error('The checkpoint has no safetensors weights yet.')
      if (names.some((name) => /\.(incomplete|ranges)$/.test(name))) throw new Error('This model is still downloading; finish the download and scan again.')
      if (names.includes('model.safetensors.index.json')) {
        const index = await this.json(directory, 'model.safetensors.index.json', blobRoot, budget)
        if (!plain(index?.weight_map) || !Object.keys(index.weight_map).length) throw new Error('The checkpoint shard index is invalid.')
        const shards = [...new Set(Object.values(index.weight_map))]
        if (shards.some((name) => typeof name !== 'string' || !name.endsWith('.safetensors') || path.basename(name) !== name || !weights.includes(name))) throw new Error('The checkpoint is missing one or more weight shards.')
      } else {
        const shards = weights.map((name) => /-(\d{5})-of-(\d{5})\.safetensors$/.exec(name)).filter((v) => v !== null)
        if (shards.length && (shards.some((match) => Number(match![2]) !== shards.length) || new Set(shards.map((match) => Number(match![1]))).size !== shards.length || shards.some((match) => Number(match![1]) < 1 || Number(match![1]) > shards.length))) throw new Error('The checkpoint is missing one or more weight shards.')
      }
      if (!model.drafterOnly && !names.includes('tokenizer.json') && !(names.includes('tokenizer.model') && names.includes('tokenizer_config.json'))) throw new Error('The checkpoint tokenizer is missing; finish downloading the model and scan again.')
      // Tokenizer JSON can exceed 20 MB. Keep scans bounded and leave vocabulary parsing to the
      // native loader, while checking that all assets it may open remain inside this checkpoint.
      for (const name of names.filter((name) => /\.(json|txt|model)$/.test(name))) await this.safeFile(directory, name, blobRoot)
      if (names.includes('tokenizer.json')) await this.safeFile(directory, 'tokenizer.json', blobRoot)
      else if (!model.drafterOnly) { await this.safeFile(directory, 'tokenizer.model', blobRoot); await this.json(directory, 'tokenizer_config.json', blobRoot, budget) }
      const identities = []
      let bytes = 0
      for (const name of weights) { this.check(budget); const weight = await this.weight(directory, name, blobRoot, budget); bytes += weight.bytes; identities.push(weight.identity) }
      model.complete = true
      model.weightsBytes = bytes
      model.memoryFloorBytes = bytes + 2 * GIB
      model.fitsMemory = model.memoryFloorBytes <= this.options.memoryLimitBytes()
      fingerprint = `${repo || ''}:${digest(JSON.stringify(config))}:${identities.sort().join('|')}`
      if (!model.compatible) model.reason = config.dflash_config ? 'This is a drafter checkpoint, not a standalone chat model.' : !formats ? `The bundled engine does not support ${model.modelType}.` : `The bundled engine does not support this ${model.format} checkpoint format.`
      else if (!model.fitsMemory) model.reason = 'This checkpoint needs more model memory than the current hardware allows.'
      model.loadable = model.complete && model.compatible && model.fitsMemory === true
    } catch (error) {
      if (error instanceof Limit || error instanceof Cancelled) throw error
      const code = (error as NodeJS.ErrnoException).code
      model.reason = code === 'ENOENT' ? 'A checkpoint file is missing or its download is incomplete.' : code === 'EACCES' || code === 'EPERM' ? 'TensorFold cannot read this checkpoint; check folder permissions.' : error instanceof SyntaxError ? 'The checkpoint metadata is incomplete or invalid JSON.' : (error as Error).message
    }
    return { model, fingerprint }
  }

  private async scanNow(options: EngineLibraryScanOptions, signal: AbortSignal): Promise<EngineModelLibrarySnapshot> {
    if (options.includeHuggingFace !== undefined && typeof options.includeHuggingFace !== 'boolean') throw new Error('Hugging Face cache scanning must be on or off.')
    if (options.includeHuggingFace !== undefined) await this.mutate(async (state) => { state.includeHuggingFace = options.includeHuggingFace!; if (!state.includeHuggingFace) state.models = state.models.filter((model) => !model.folderId.startsWith('hf-')) })
    const state = await this.read()
    const caps = await this.options.capabilities()
    const roots = await this.roots(state)
    const budget: ScanBudget = { directories: 0, entries: 0, metadataBytes: 0, deadline: Date.now() + this.limits.durationMs, signal }
    const models: EngineLibraryModel[] = [], warnings: string[] = []
    const visited = new Set<string>(), fingerprints = new Set<string>()
    let truncated = false
    const collect = async (root: Root, directory: string, entries: import('node:fs').Dirent[], repo: string | null = null, revision: string | null = null, blobs: string | null = null) => {
      // Users often choose the snapshot itself, its snapshots parent, or a models-- repo folder.
      // Recognize the exact HF layout and admit only that repo's own sibling blob directory.
      if (!repo && SHA.test(path.basename(directory)) && path.basename(path.dirname(directory)) === 'snapshots') {
        const repoRoot = path.dirname(path.dirname(directory))
        const match = /^models--(.+)--(.+)$/.exec(path.basename(repoRoot))
        if (match && isRepoId(`${match[1]}/${match[2]}`) && await fs.realpath(repoRoot) === repoRoot && await fs.realpath(path.dirname(directory)) === path.dirname(directory)) {
          repo = `${match[1]}/${match[2]}`; revision = path.basename(directory); blobs = path.join(repoRoot, 'blobs')
        }
      }
      const { model, fingerprint } = await this.inspect(root, directory, entries.map((entry) => entry.name), caps, budget, repo, revision, blobs)
      if (fingerprints.has(fingerprint)) return
      if (models.length >= this.limits.models) throw new Limit('The scan reached its model limit. Choose a more specific folder to find the remaining models.')
      fingerprints.add(fingerprint); models.push(model)
    }
    const walk = async (root: Root, directory: string, depth: number): Promise<void> => {
      this.check(budget)
      if (visited.has(directory)) return
      visited.add(directory)
      if (await fs.realpath(directory) !== directory || !(await fs.lstat(directory)).isDirectory()) throw new Error('A model search folder is missing or has been redirected; select it again.')
      const entries = await this.list(directory, budget)
      const assets = entries.filter((entry) => !entry.name.startsWith('._'))
      if (assets.some((entry) => entry.name === 'config.json' || /\.(safetensors|gguf|bin|pt|pth)$/i.test(entry.name))) { await collect(root, directory, assets); return }
      for (const entry of entries) {
        this.check(budget)
        if (!entry.isDirectory() || entry.isSymbolicLink() || entry.name.startsWith('.') || ['blobs', 'node_modules', 'venv', '__pycache__'].includes(entry.name)) continue
        const child = path.join(directory, entry.name)
        const hf = /^models--(.+)--(.+)$/.exec(entry.name)
        if (hf && isRepoId(`${hf[1]}/${hf[2]}`)) {
          const repo = `${hf[1]}/${hf[2]}`
          const snapshots = path.join(child, 'snapshots')
          if (await fs.realpath(snapshots).catch(() => null) !== snapshots) continue
          for (const checkpoint of await this.list(snapshots, budget)) {
            if (!checkpoint.isDirectory() || !SHA.test(checkpoint.name)) continue
            const snapshot = path.join(snapshots, checkpoint.name)
            if (visited.has(snapshot)) continue
            visited.add(snapshot)
            await collect(root, snapshot, await this.list(snapshot, budget), repo, checkpoint.name, path.join(child, 'blobs'))
          }
        } else if (depth < this.limits.depth) await walk(root, child, depth + 1)
        else if (!truncated) { truncated = true; warnings.push('Some folders are deeper than the scan limit; add those model folders directly.') }
      }
    }
    try {
      for (const root of roots) {
        try { await walk(root, root.path, 0) } catch (error) {
          if (error instanceof Limit || error instanceof Cancelled) throw error
          if (warnings.length < 32) warnings.push((error as NodeJS.ErrnoException).code === 'ENOENT' ? `${path.basename(root.path)} was not found; choose it again if it moved.` : `${path.basename(root.path)} could not be scanned; check folder permissions or select its real location.`)
        }
      }
      this.check(budget)
    } catch (error) {
      if (error instanceof Cancelled) { this.cancelled = true; return this.snapshot() }
      if (!(error instanceof Limit)) throw error
      truncated = true
      if (warnings.length < 32) warnings.push(error.message)
    }
    for (const model of models) if (model.repo && models.filter((other) => other.repo === model.repo).length > 1) model.title += ` @ ${model.revision!.slice(0, 8)}`
    await this.mutate(async (current) => { current.models = models; current.scannedAt = Date.now(); current.warnings = warnings; current.truncated = truncated })
    return this.snapshot()
  }

  async resolve(id: string, allowDrafter = false): Promise<EngineLibraryModel> {
    if (!isLocalModelId(id)) throw new Error('Choose a model from the local library.')
    const state = await this.read()
    const saved = state.models.find((model) => model.id === id)
    if (!saved) throw new Error('This local model is no longer in the library. Scan its folder again.')
    const roots = await this.roots(state)
    const root = roots.find((entry) => entry.id === saved.folderId)
    if (!root || !contains(root.path, saved.path) || await fs.realpath(root.path) !== root.path) throw new Error('The model folder is no longer authorized. Select it again.')
    const budget: ScanBudget = { directories: 0, entries: 0, metadataBytes: 0, deadline: Date.now() + this.limits.durationMs }
    let blobs: string | null = null
    if (saved.source === 'huggingface') {
      if (!saved.repo || !saved.revision) throw new Error(INVALID)
      // HF caches may also be nested under a user-selected folder.
      const actualRepo = path.dirname(path.dirname(saved.path))
      const scopedSnapshotRoot = root.path === saved.path || root.path === path.join(actualRepo, 'snapshots')
      if (path.basename(actualRepo) !== `models--${saved.repo.replace('/', '--')}` || !contains(root.path, actualRepo) && !scopedSnapshotRoot || saved.path !== path.join(actualRepo, 'snapshots', saved.revision) || await fs.realpath(actualRepo) !== actualRepo) throw new Error(INVALID)
      blobs = path.join(actualRepo, 'blobs')
    }
    const { model } = await this.inspect(root, saved.path, (await this.list(saved.path, budget)).map((entry) => entry.name), await this.options.capabilities(), budget, saved.repo, saved.revision, blobs)
    if (!model.loadable && !(allowDrafter && model.drafterOnly && model.complete && model.fitsMemory)) throw new Error(model.reason || 'This local checkpoint cannot be loaded.')
    return model
  }
}
