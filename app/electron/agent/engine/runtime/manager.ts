import path from 'node:path'
import { promises as fs } from 'node:fs'
import { downloadVerifiedUpdate, reverifyDownloadedUpdate, type UpdateDownloadDependencies } from '../../update-download'
import type { EngineRuntimeSnapshot } from '../../types/engine-runtime'
import { compatibleRuntime, verifyRuntimeCatalog, publicRuntime, runtimeId, PACKAGE_LIMIT, type RuntimePackage } from './catalog'
import { fetchRuntimeCatalog } from './network'
import { atomicJson, privateDirectory, readPrivateJson, extractRuntime, verifyRuntimeTree } from './storage'
import { probeRuntime } from './probe'

interface RuntimeManagerOptions {
  root: string
  bundledVersion: string
  appVersion: string
  platform?: string
  arch?: string
  transition: (apply: () => Promise<void>, rollback: () => Promise<void>) => Promise<void>
  // Dependency injection is for isolated tests; the desktop host never accepts a renderer key or URL.
  publicKey?: string
  network?: UpdateDownloadDependencies
  probe?: typeof probeRuntime
}
export class EngineRuntimeManager {
  private catalog = new Map<string, RuntimePackage>()
  private catalogRaw = ''
  private installed = new Map<string, RuntimePackage>()
  private active: RuntimePackage | null = null
  private controller: AbortController | null = null
  private state: EngineRuntimeSnapshot['state'] = 'idle'
  private progress: EngineRuntimeSnapshot['progress'] = null
  private checkedAt: number | null = null
  private error: string | null = null
  private publishedAt = 0
  readonly ready: Promise<void>
  constructor(private readonly options: RuntimeManagerOptions) { this.ready = this.restore() }
  binaryPath(): string | null { return this.active ? path.join(this.options.root, this.active.id, this.active.entrypoint) : null }
  private compatible(pkg: RuntimePackage): boolean {
    return compatibleRuntime(pkg, this.options.appVersion, this.options.platform ?? process.platform, this.options.arch ?? process.arch)
  }
  private parse(raw: string, installed = false) { return verifyRuntimeCatalog(raw, { publicKey: this.options.publicKey, installed }) }
  private async restore(): Promise<void> {
    try {
      await privateDirectory(this.options.root)
      for (const id of await fs.readdir(this.options.root)) {
        if (id.startsWith('.')) continue
        try {
          runtimeId(id)
          const dir = path.join(this.options.root, id)
          const raw = await readPrivateJson(path.join(dir, '.catalog.json'))
          const pkg = this.parse(raw, true).packages.find(p => p.id === id && this.compatible(p))
          if (!pkg) continue
          await verifyRuntimeTree(dir, pkg)
          this.installed.set(id, pkg)
        } catch {  }
      }
      try {
        const saved = JSON.parse(await readPrivateJson(path.join(this.options.root, '.selection.json'), 4096))
        if (Number.isSafeInteger(saved.publishedAt)) this.publishedAt = Math.max(0, saved.publishedAt)
        const pkg = this.installed.get(saved.pending ? saved.previous : saved.active)
        if (pkg) { await (this.options.probe ?? probeRuntime)(path.join(this.options.root, pkg.id), pkg); this.active = pkg }
        else if (saved.active != null) this.error = 'The selected runtime failed verification. Using the bundled engine.'
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') this.error = 'The selected runtime could not start. Using the bundled engine.'
      }
    } catch { this.error = 'Private runtime storage is unavailable. Using the bundled engine.' }
  }
  async snapshot(): Promise<EngineRuntimeSnapshot> {
    await this.ready
    return { bundledVersion: this.options.bundledVersion, active: this.active ? publicRuntime(this.active) : null,
      installed: [...this.installed.values()].map(publicRuntime), available: [...this.catalog.values()].map(publicRuntime),
      state: this.state, progress: this.progress, checkedAt: this.checkedAt, error: this.error }
  }
  private async run(state: EngineRuntimeSnapshot['state'], task: (signal: AbortSignal) => Promise<void>): Promise<EngineRuntimeSnapshot> {
    await this.ready
    await privateDirectory(this.options.root)
    if (this.controller) throw new Error('Wait for the current runtime operation to finish.')
    this.controller = new AbortController(); this.state = state; this.error = null
    const timeout = setTimeout(() => this.controller?.abort(new Error('Runtime operation timed out.')), 15 * 60_000)
    try { await task(this.controller.signal) }
    catch (e) { this.error = (e as Error).message; throw e }
    finally { clearTimeout(timeout); this.controller = null; this.state = 'idle'; this.progress = null }
    return this.snapshot()
  }
  cancel(): boolean {
    if (!this.controller || this.state === 'switching') return false
    this.controller.abort(new Error('Runtime download cancelled.')); return true
  }
  check(): Promise<EngineRuntimeSnapshot> {
    return this.run('checking', async signal => {
      const raw = await fetchRuntimeCatalog(signal, this.options.network)
      const catalog = this.parse(raw)
      if (catalog.publishedAt < this.publishedAt) throw new Error('Refusing an older runtime catalog.')
      this.catalog = new Map(catalog.packages.filter(p => this.compatible(p)).map(p => [p.id, p]))
      this.catalogRaw = raw; this.checkedAt = Date.now(); this.publishedAt = catalog.publishedAt
      await this.saveSelection()
    })
  }
  install(id: string): Promise<EngineRuntimeSnapshot> {
    runtimeId(id)
    return this.run('downloading', async signal => {
      const pkg = this.catalog.get(id)
      if (!pkg) throw new Error('Check for a signed runtime before downloading.')
      // Expiration is checked again in case the UI has been open for days.
      this.parse(this.catalogRaw)
      if (this.installed.has(id)) return
      const downloads = path.join(this.options.root, '.downloads')
      await privateDirectory(downloads)
      let archive: string | null = null, staging: string | null = null
      try {
        const download = await downloadVerifiedUpdate({ downloadUrl: pkg.url, checksum: { algorithm: 'sha256', value: pkg.sha256 },
          manifestSignatureVerified: true, downloadRoot: downloads, signal, onProgress: progress => { this.progress = progress } },
          { ...this.options.network, maxBytes: Math.min(PACKAGE_LIMIT, pkg.bytes) })
        archive = download.path
        if (download.bytes !== pkg.bytes) throw new Error('Runtime archive size mismatch.')
        await reverifyDownloadedUpdate(download, downloads)
        signal.throwIfAborted(); this.state = 'installing'
        staging = await fs.mkdtemp(path.join(this.options.root, '.stage-'))
        await extractRuntime(archive, staging, pkg, signal)
        await (this.options.probe ?? probeRuntime)(staging, pkg)
        signal.throwIfAborted()
        await atomicJson(path.join(staging, '.catalog.json'), JSON.parse(this.catalogRaw))
        const destination = path.join(this.options.root, id)
        // Immutable IDs cannot replace an existing installation, including a rejected partial one.
        try { await fs.lstat(destination); throw new Error('Runtime ID already exists. Choose a new signed package ID.') }
        catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e }
        await fs.rename(staging, destination); staging = null
        this.installed.set(id, pkg)
      } finally {
        if (archive) await fs.rm(path.dirname(archive), { recursive: true, force: true })
        if (staging) await fs.rm(staging, { recursive: true, force: true })
      }
    })
  }
  use(id: string | null): Promise<EngineRuntimeSnapshot> {
    if (id !== null) runtimeId(id)
    return this.run('switching', async () => {
      const next = id === null ? null : this.installed.get(id)
      if (next === undefined) throw new Error('Download and verify the runtime first.')
      const previous = this.active
      if (next) {
        const dir = path.join(this.options.root, next.id)
        await verifyRuntimeTree(dir, next)
        await (this.options.probe ?? probeRuntime)(dir, next)
      }
      await this.options.transition(async () => { this.active = next; await this.saveSelection({ previous: previous?.id ?? null }) },
        async () => {
          let restored = previous
          if (restored) {
            try { await verifyRuntimeTree(path.join(this.options.root, restored.id), restored) }
            catch { restored = null }
          }
          this.active = restored; await this.saveSelection()
        })
      await this.saveSelection()
    })
  }
  private saveSelection(pending?: { previous: string | null }): Promise<void> {
    return atomicJson(path.join(this.options.root, '.selection.json'), { active: this.active?.id ?? null, publishedAt: this.publishedAt, ...(pending ? { pending: true, previous: pending.previous } : {}) })
  }
}
