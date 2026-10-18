import type { EngineEvent, ModelPreflight, PullProgress } from '../types'
import { assertRepoId, type EngineBinary } from './paths'
import { isLocalModelId } from './library'
import { startPull, type PullHandle } from './pull'
import type { ModelRevisions } from './model-revisions'

export type RunningPull = { handle: PullHandle; last: PullProgress }
interface DownloadOptions {
  running: Map<string, RunningPull>
  revisions: ModelRevisions
  env: NodeJS.ProcessEnv
  hub: () => string
  binary: () => EngineBinary
  assertIdle: (repo: string) => void
  cached: (key: string) => ModelPreflight | undefined
  preflight: (repo: string, revision: string | null) => Promise<ModelPreflight>
  invalidate: (repo: string) => void
  invalidateInstalled: () => void
  refreshInstalled: () => Promise<unknown>
  emit: (event: EngineEvent) => void
  pollMs?: number
}

// Completion becomes visible only after the immutable revision selection is durable.
export class EngineDownloads {
  private readonly finished = new Map<string, PullProgress>()
  private readonly lastPullEmit = new Map<string, number>()
  constructor(private readonly options: DownloadOptions) {}
  forget(repo: string): void { this.finished.delete(repo) }

  list(): PullProgress[] {
    const running = [...this.options.running.values()].map((p) => p.last)
    return [...running, ...[...this.finished.values()].filter((p) => !this.options.running.has(p.repo))]
  }

  async pull(repo: string, revision: string | null = null): Promise<PullProgress> {
    if (isLocalModelId(repo)) throw new Error('Local library checkpoints load in place and cannot be downloaded by their library ID.')
    if (this.options.running.has(assertRepoId(repo))) return this.options.running.get(repo)!.last
    await this.options.revisions.read() // Preserve invalid registry bytes and refuse before starting a child.
    this.options.assertIdle(repo)
    const pre = this.options.cached(JSON.stringify([repo, revision])) ?? await this.options.preflight(repo, revision)
    this.options.assertIdle(repo)
    if (this.options.running.has(repo)) return this.options.running.get(repo)!.last
    if (!['ok', 'installed'].includes(pre.verdict)) throw new Error(pre.reasons[0] ?? `${repo} cannot be downloaded for the built-in engine.`)
    if (!pre.resolvedRevision || (pre.drafter && !pre.drafter.resolvedRevision)) throw new Error('Retry the model checks to resolve an immutable checkpoint before downloading.')
    const steps = [{ repo, revision: pre.resolvedRevision }, ...(pre.drafter ? [{ repo: pre.drafter.repo, revision: pre.drafter.resolvedRevision! }] : [])]
    this.finished.delete(repo)
    const handle = startPull({
      binary: this.options.binary(),
      env: this.options.env,
      hub: this.options.hub(),
      steps,
      totalBytes: pre.totalBytes,
      pollMs: this.options.pollMs ? Math.max(50, this.options.pollMs) : 1000,
      onUpdate: (progress) => this.onPullUpdate(progress),
    })
    this.options.running.set(repo, { handle, last: handle.progress() })
    this.options.emit({ kind: 'pull', pull: handle.progress() })
    void handle.done.then(async (final) => {
      if (final.state === 'done') {
        try { await this.options.revisions.commit(steps) } catch (error) {
          final = { ...final, state: 'failed', phase: 'Failed', message: (error as Error).message }
        }
      }
      this.options.running.delete(repo)
      this.finished.set(repo, final)
      this.options.invalidate(repo)
      this.options.invalidateInstalled()
      this.options.emit({ kind: 'pull', pull: final })
      if (final.state === 'done') await this.options.refreshInstalled().catch(() => {})
    })
    return handle.progress()
  }

  private onPullUpdate(progress: PullProgress): void {
    if (progress.state !== 'running') return // Completion is emitted only after the selection is durable.
    const job = this.options.running.get(progress.repo)
    if (job) job.last = progress
    const now = Date.now()
    if (progress.state === 'running' && now - (this.lastPullEmit.get(progress.repo) ?? 0) < 250) return
    this.lastPullEmit.set(progress.repo, now)
    this.options.emit({ kind: 'pull', pull: progress })
  }

}
