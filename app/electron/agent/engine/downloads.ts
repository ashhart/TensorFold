import type { EngineEvent, EngineInstalledModel, ModelPreflight, PullProgress } from '../types'
import { assertRepoId, type EngineBinary } from './paths'
import { PullQueue, type PullHandle } from './pull'
import { EngineSupervisor } from './supervisor'
import { ModelRevisions } from './model-revisions'
import { isLocalModelId } from './library'

interface EngineDownloadsOptions {
  binary: () => EngineBinary
  env: () => NodeJS.ProcessEnv
  hub: () => string
  pollMs?: number
  assertAllowed: () => void
  deletingModel: () => boolean
  cachedPreflight: (key: string) => ModelPreflight | undefined
  preflight: (repo: string, revision: string | null) => Promise<ModelPreflight>
  revisions: ModelRevisions
  status: () => ReturnType<EngineSupervisor['status']>
  invalidate: (repo: string) => void
  invalidateInstalled: () => void
  refreshInstalled: () => Promise<EngineInstalledModel[]>
  emit: (event: EngineEvent) => void
}

export class EngineDownloads {
  private readonly pulls = new Map<string, { handle: PullHandle; last: PullProgress }>()
  private readonly finished = new Map<string, PullProgress>()
  private readonly pullQueue = new PullQueue()
  private readonly lastPullEmit = new Map<string, number>()
  constructor(private options: EngineDownloadsOptions) { }
  get busy(): boolean { return this.pullQueue.busy || this.pulls.size > 0 }
  get queueBusy(): boolean { return this.pullQueue.busy }
  hasRepo(repo: string): boolean { return this.pullQueue.hasRepo(repo) }
  hasJob(repo: string): boolean { return this.pulls.has(repo) }
  forget(repo: string): void { this.finished.delete(repo) }
  cancelAll(): void { this.pullQueue.cancelAll(); for (const job of this.pulls.values()) job.handle.cancel() }
  pullList(): PullProgress[] {
    const running = [...this.pulls.values()].map((p) => p.last)
    return [...running, ...[...this.finished.values()].filter((p) => !this.pulls.has(p.repo))]
  }

  async pull(repo: string, revision: string | null = null): Promise<PullProgress> {
    this.options.assertAllowed()
    if (this.options.deletingModel()) throw new Error('Wait for model cache cleanup before starting a download.')
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place and cannot be downloaded by their library ID.')
    if (this.pulls.has(assertRepoId(repo))) return this.pulls.get(repo)!.last
    await this.options.revisions.read() // Preserve invalid registry bytes and refuse before starting a child.
    this.assertPullIdle(repo)
    const pre = this.options.cachedPreflight(JSON.stringify([repo, revision])) ?? await this.options.preflight(repo, revision)
    this.options.assertAllowed()
    if (this.options.deletingModel()) throw new Error('Wait for model cache cleanup before starting a download.')
    this.assertPullIdle(repo)
    if (this.pulls.has(repo)) return this.pulls.get(repo)!.last
    if (!['ok', 'installed'].includes(pre.verdict)) throw new Error(pre.reasons[0] ?? `${repo} cannot be downloaded for the built-in engine.`)
    if (!pre.resolvedRevision || (pre.drafter && !pre.drafter.resolvedRevision)) throw new Error('Retry the model checks to resolve an immutable checkpoint before downloading.')
    const steps = [{ repo, revision: pre.resolvedRevision }, ...(pre.drafter ? [{ repo: pre.drafter.repo, revision: pre.drafter.resolvedRevision! }] : [])]
    for (const step of steps) this.assertPullIdle(step.repo)
    this.finished.delete(repo)
    const startedHub = this.options.hub()
    const handle = this.pullQueue.enqueue({
      binary: this.options.binary(),
      env: this.options.env(),
      hub: this.options.hub(),
      steps,
      totalBytes: pre.totalBytes,
      pollMs: this.options.pollMs ? Math.max(50, this.options.pollMs) : 1000,
      onUpdate: (progress) => this.onPullUpdate(progress),
    }, async (final) => { await this.options.revisions.commit(steps, startedHub); return final })
    this.pulls.set(repo, { handle, last: handle.progress() })
    this.options.emit({ kind: 'pull', pull: handle.progress() })
    void handle.done.then(async (final) => {
      this.pulls.delete(repo)
      this.finished.set(repo, final)
      this.options.invalidate(repo)
      this.options.invalidateInstalled()
      this.options.emit({ kind: 'pull', pull: final })
      if (final.state === 'done') await this.options.refreshInstalled().catch(() => { })
    })
    return handle.progress()
  }

  private onPullUpdate(progress: PullProgress): void {
    if (!['queued', 'running', 'verifying'].includes(progress.state)) return // Completion follows durable selection.
    const job = this.pulls.get(progress.repo)
    if (job) job.last = progress
    const now = Date.now()
    if (progress.state === 'running' && now - (this.lastPullEmit.get(progress.repo) ?? 0) < 250) return
    this.lastPullEmit.set(progress.repo, now)
    this.options.emit({ kind: 'pull', pull: progress })
  }

  private assertPullIdle(repo: string): void {
    const status = this.options.status()
    if ((status.model === repo || status.drafter === repo) && status.state !== 'stopped' && status.state !== 'failed') throw new Error('Unload the model before downloading another checkpoint for it.')
  }

  cancelPull(repo: string): boolean {
    const job = this.pulls.get(assertRepoId(repo))
    if (!job) return false
    job.handle.cancel()
    return true
  }
}
