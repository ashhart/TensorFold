import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { EngineEvent, EngineInstalledModel, PullProgress } from '../types'
import { drafterFor } from './catalog'
import type { InstalledModels } from './installed-models'
import { isLocalModelId } from './library'
import type { ModelHub } from './model-hub'
import type { ModelRevisions } from './model-revisions'
import { assertRepoId, repoCacheDir, type EngineBinary } from './paths'
import { PullQueue, type PullHandle } from './pull'
import type { EngineSupervisor } from './supervisor'

// Downloads into the shared Hugging Face cache and the cleanup of the checkpoints the app
// downloaded: the queue, per-repo progress, and the guarded removal of cache folders.

export interface ModelDownloadsDeps {
  models: ModelHub
  revisions: ModelRevisions
  installed: InstalledModels
  supervisor: Pick<EngineSupervisor, 'status' | 'mutateCache'>
  requireBinary: () => EngineBinary
  env: () => NodeJS.ProcessEnv
  hub: () => string
  pollMs?: number
  emit: (event: EngineEvent) => void
  // Refuses while the service is quitting, erasing learned data or benchmarking.
  assertAdmission: () => void
}

export class ModelDownloads {
  private readonly pulls = new Map<string, { handle: PullHandle; last: PullProgress }>()
  private readonly finished = new Map<string, PullProgress>()
  private readonly queue = new PullQueue()
  private readonly lastEmit = new Map<string, number>()
  private cleaning = false

  constructor(private readonly deps: ModelDownloadsDeps) {}

  // Model cache cleanup is in progress.
  get deleting(): boolean { return this.cleaning }

  get queueBusy(): boolean { return this.queue.busy }

  get activeCount(): number { return this.pulls.size }

  hasRepo(repo: string): boolean { return this.queue.hasRepo(repo) }

  list(): PullProgress[] {
    const running = [...this.pulls.values()].map((p) => p.last)
    return [...running, ...[...this.finished.values()].filter((p) => !this.pulls.has(p.repo))]
  }

  async pull(repo: string, revision: string | null = null): Promise<PullProgress> {
    this.deps.assertAdmission()
    if (this.cleaning) throw new Error('Wait for model cache cleanup before starting a download.')
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place and cannot be downloaded by their library ID.')
    if (this.pulls.has(assertRepoId(repo))) return this.pulls.get(repo)!.last
    await this.deps.revisions.read() // Preserve invalid registry bytes and refuse before starting a child.
    this.assertIdle(repo)
    const pre = this.deps.models.cached(JSON.stringify([repo, revision])) ?? await this.deps.models.preflight(repo, revision)
    this.deps.assertAdmission()
    if (this.cleaning) throw new Error('Wait for model cache cleanup before starting a download.')
    this.assertIdle(repo)
    if (this.pulls.has(repo)) return this.pulls.get(repo)!.last
    if (!['ok', 'installed'].includes(pre.verdict)) throw new Error(pre.reasons[0] ?? `${repo} cannot be downloaded for the built-in engine.`)
    if (!pre.resolvedRevision || (pre.drafter && !pre.drafter.resolvedRevision)) throw new Error('Retry the model checks to resolve an immutable checkpoint before downloading.')
    const steps = [{ repo, revision: pre.resolvedRevision }, ...(pre.drafter ? [{ repo: pre.drafter.repo, revision: pre.drafter.resolvedRevision! }] : [])]
    for (const step of steps) this.assertIdle(step.repo)
    this.finished.delete(repo)
    const startedHub = this.deps.hub()
    const handle = this.queue.enqueue({
      binary: this.deps.requireBinary(),
      env: this.deps.env(),
      hub: this.deps.hub(),
      steps,
      totalBytes: pre.totalBytes,
      pollMs: this.deps.pollMs ? Math.max(50, this.deps.pollMs) : 1000,
      onUpdate: (progress) => this.onUpdate(progress),
    }, async (final) => { await this.deps.revisions.commit(steps, startedHub); return final })
    this.pulls.set(repo, { handle, last: handle.progress() })
    this.deps.emit({ kind: 'pull', pull: handle.progress() })
    void handle.done.then(async (final) => {
      this.pulls.delete(repo)
      this.finished.set(repo, final)
      this.deps.models.invalidate(repo)
      this.deps.installed.invalidate()
      this.deps.emit({ kind: 'pull', pull: final })
      if (final.state === 'done') await this.deps.installed.list(true).catch(() => {})
    })
    return handle.progress()
  }

  private onUpdate(progress: PullProgress): void {
    if (!['queued', 'running', 'verifying'].includes(progress.state)) return // Completion follows durable selection.
    const job = this.pulls.get(progress.repo)
    if (job) job.last = progress
    const now = Date.now()
    if (progress.state === 'running' && now - (this.lastEmit.get(progress.repo) ?? 0) < 250) return
    this.lastEmit.set(progress.repo, now)
    this.deps.emit({ kind: 'pull', pull: progress })
  }

  private assertIdle(repo: string): void {
    const status = this.deps.supervisor.status()
    if ((status.model === repo || status.drafter === repo) && status.state !== 'stopped' && status.state !== 'failed') throw new Error('Unload the model before downloading another checkpoint for it.')
  }

  cancel(repo: string): boolean {
    const job = this.pulls.get(assertRepoId(repo))
    if (!job) return false
    job.handle.cancel()
    return true
  }

  // Quit path: stop admission before cancelling scans and running downloads.
  cancelQueued(): void { this.queue.cancelAll() }

  cancelActive(): void {
    for (const { handle } of this.pulls.values()) handle.cancel()
  }

  // Removes the repo's folder from the shared cache, and its drafter when no other installed model uses it.
  async remove(repo: string): Promise<EngineInstalledModel[]> {
    assertRepoId(repo)
    this.deps.assertAdmission()
    if (this.cleaning || this.queue.busy) throw new Error('Finish or cancel model downloads before cleaning the shared cache.')
    this.cleaning = true
    try { return await this.deps.supervisor.mutateCache(() => this.removeNow(repo)) } finally { this.cleaning = false }
  }

  private async removeNow(repo: string): Promise<EngineInstalledModel[]> {
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints are read-only. Remove their search folder to unlink them; their files stay untouched.')
    const selected = await this.deps.revisions.read()
    if (!selected[repo]) throw new Error('This checkpoint belongs to the shared cache, rather than an app-managed download. TensorFold will not delete it.')
    const status = this.deps.supervisor.status()
    if ((status.model === repo || this.deps.installed.canonical(status.model ?? '') === repo || status.drafter === repo) && status.state !== 'stopped' && status.state !== 'failed') throw new Error(`Unload ${repo} before deleting it.`)
    if (this.pulls.has(repo)) throw new Error(`Cancel the download of ${repo} before deleting it.`)
    const installed = await this.deps.installed.list()
    const drafter = drafterFor(repo)
    const targets = [repo]
    if (drafter && selected[drafter] && !installed.some((m) => m.repo !== repo && drafterFor(this.deps.installed.canonical(m.repo) ?? m.repo) === drafter)) targets.push(drafter)
    for (const target of targets) await this.removeRepoFolder(target, selected[target])
    await this.deps.revisions.remove(targets)
    this.deps.models.invalidate(repo)
    this.finished.delete(repo)
    return this.deps.installed.list(true)
  }

  private async removeRepoFolder(repo: string, revision: string): Promise<void> {
    const hub = await fs.realpath(this.deps.hub()).catch(() => null)
    if (!hub) return
    const folder = repoCacheDir(hub, repo)
    let info
    try { info = await fs.lstat(folder) } catch { return }
    if (info.isSymbolicLink() || !info.isDirectory()) throw new Error(`${folder} is not a cache folder; refusing to delete it.`)
    if (path.dirname(folder) !== hub || !path.basename(folder).startsWith('models--')) throw new Error('Refusing to delete outside the Hugging Face cache.')
    const snapshots = path.join(folder, 'snapshots')
    if (await fs.realpath(snapshots).catch(() => null) !== snapshots) throw new Error('The model snapshots folder was redirected; refusing to delete it.')
    const entries = await fs.readdir(snapshots)
    if (entries.some((name) => name !== revision)) {
      // Other HF clients may keep different revisions here. Preserve those revisions, shared
      // blobs and refs; only unlink our exact selected checkpoint, without following its links.
      const selected = path.join(snapshots, revision)
      const stat = await fs.lstat(selected).catch(() => null)
      if (stat?.isSymbolicLink() || stat && !stat.isDirectory()) throw new Error('The selected checkpoint was redirected; refusing to delete it.')
      if (stat) await fs.rm(selected, { recursive: true })
      return
    }
    await fs.rm(folder, { recursive: true, force: true })
  }
}
