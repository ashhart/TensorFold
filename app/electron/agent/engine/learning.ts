import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { EngineLearningSnapshot } from '../types'
import { LearningCopies } from './learning-copy'
import { parseLearningGraph, readLearningGraph, streamLesson, type LessonEvent } from './learning-client'
import type { EngineSupervisor } from './supervisor'

type Job = NonNullable<EngineLearningSnapshot['job']>
interface Dependencies {
  appHome: string
  supervisor: EngineSupervisor
  copies: LearningCopies
  settings(): { supported: boolean; enabled: boolean; model: string | null }
  emit(snapshot: EngineLearningSnapshot): void
  pollMs?: number
}

// Only explicit user lessons reach this manager. No chat history or file contents are ingested.
export class EngineLearning {
  private graph: EngineLearningSnapshot['graph'] = null
  private job: Job | null = null
  private hash: string | null = null
  private readonly submitted = new Set<string>()
  private initialized: Promise<void> | null = null
  private controller: AbortController | null = null
  private shutdownController = new AbortController()
  private pending = false
  private cancelRequested = false
  private sawFact = false
  private preparing = false
  private progress: EngineLearningSnapshot['progress'] = null
  private message: string | null = null
  private copyReady = false
  private graphRequest: Promise<void> | null = null
  private running: Promise<void> | null = null
  private completion: Promise<void> | null = null
  private completed: (() => void) | null = null
  private paused = false

  constructor(private readonly deps: Dependencies) {}

  get busy(): boolean { return this.pending || this.preparing }

  private journalFile(): string { return path.join(this.deps.appHome, 'engine', 'learning-job.json') }

  private init(): Promise<void> {
    return this.initialized ??= (async () => {
      try {
        const file = this.journalFile()
        const stat = await fs.lstat(file)
        if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > 512 * 1024 || (process.getuid && stat.uid !== process.getuid())) throw new Error('The saved Living Weights job is invalid.')
        const doc = JSON.parse(await fs.readFile(file, 'utf8'))
        const j = doc.job
        if (doc.version !== 1 || typeof doc.hash !== 'string' || !/^[a-f0-9]{64}$/.test(doc.hash) || typeof j?.id !== 'string' || j.id.length > 128 || !Number.isSafeInteger(j.startedAt) || j.startedAt < 0 || !['running', 'cancelling', 'done', 'cancelled', 'failed', 'interrupted'].includes(j.state) || !Number.isSafeInteger(j.learned) || j.learned < 0 || !Number.isSafeInteger(j.missed) || j.missed < 0 || (j.finishedAt != null && (!Number.isSafeInteger(j.finishedAt) || j.finishedAt < j.startedAt)) || (j.message != null && (typeof j.message !== 'string' || j.message.length > 8192))) throw new Error('The saved Living Weights job is invalid.')
        this.hash = doc.hash
        this.job = { id: doc.job.id, state: ['done', 'failed', 'cancelled'].includes(doc.job.state) ? doc.job.state : 'interrupted', startedAt: doc.job.startedAt, finishedAt: doc.job.finishedAt ?? Date.now(), learned: doc.job.learned ?? 0, missed: doc.job.missed ?? 0, message: doc.job.message ?? null }
        if (this.job.state === 'interrupted') this.job.message = 'The app closed during the last lesson; inspect its facts and saved copy before retrying.'
        if (doc.submitted != null && (!Array.isArray(doc.submitted) || doc.submitted.length > 4096 || doc.submitted.some((hash: unknown) => typeof hash !== 'string' || !/^[a-f0-9]{64}$/.test(hash)))) throw new Error('The saved Living Weights lesson history is invalid.')
        for (const hash of doc.submitted ?? []) this.submitted.add(hash)
        if (this.job.state !== 'failed' || this.job.learned > 0) this.submitted.add(doc.hash)
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
      }
    })()
  }

  private async saveJob(): Promise<void> {
    if (!this.job || !this.hash) return
    const requested = path.dirname(this.journalFile())
    await fs.mkdir(requested, { recursive: true, mode: 0o700 })
    if ((await fs.lstat(requested)).isSymbolicLink()) throw new Error('The engine data folder is redirected.')
    const directory = await fs.realpath(requested)
    const temporary = path.join(directory, `.lesson-${randomUUID()}.tmp`)
    try {
      const file = await fs.open(temporary, 'wx', 0o600)
      try { await file.writeFile(JSON.stringify({ version: 1, hash: this.hash, job: this.job, submitted: [...this.submitted] })); await file.sync() } finally { await file.close() }
      await fs.rename(temporary, this.journalFile())
    } finally { await fs.rm(temporary, { force: true }) }
  }

  now(): EngineLearningSnapshot {
    const settings = this.deps.settings()
    const live = this.deps.supervisor.status()
    return {
      ...settings, experimental: true, active: settings.enabled && live.model === settings.model && live.state === 'ready' && live.learningActive,
      state: this.preparing ? 'preparing' : this.pending ? this.cancelRequested ? 'cancelling' : 'teaching' : this.job?.state === 'failed' || this.job?.state === 'interrupted' ? 'failed' : settings.enabled ? 'ready' : 'off',
      message: this.message ?? this.job?.message ?? null, privateCopy: this.copyReady, progress: this.progress ? { ...this.progress } : null,
      job: this.job ? { ...this.job, ...(this.pending && ['done', 'cancelled'].includes(this.job.state) ? { state: this.cancelRequested ? 'cancelling' as const : 'running' as const } : {}) } : null, graph: this.graph,
    }
  }

  changed(): void { this.deps.emit(this.now()) }

  preparation(copiedBytes: number | null, totalBytes = 0, message: string | null = null): void {
    this.preparing = copiedBytes != null
    this.progress = copiedBytes == null ? null : { copiedBytes, totalBytes }
    this.message = message
    this.changed()
  }

  async snapshot(refresh = true): Promise<EngineLearningSnapshot> {
    if (this.paused) return this.now()
    await this.init()
    if (this.paused) return this.now()
    const { model } = this.deps.settings()
    if (model) {
      this.copyReady = !!await this.deps.copies.existing(model)
      if (this.paused) return this.now()
      if (refresh && this.copyReady) {
        const live = this.deps.supervisor.live(model)
        if (live && this.deps.supervisor.status().learningActive) await this.refresh()
        else {
          const file = this.deps.copies.graphFile(model)
          try {
            const stat = await fs.lstat(file)
            if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size > 8 * 1024 ** 2) throw new Error('The saved learning graph is invalid or oversized.')
            this.graph = parseLearningGraph(JSON.parse(await fs.readFile(file, 'utf8')))
          } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
        }
      }
    }
    return this.now()
  }

  private refresh(): Promise<void> {
    if (this.graphRequest) return this.graphRequest
    const model = this.deps.settings().model
    const live = model ? this.deps.supervisor.live(model) : null
    if (!live) return Promise.resolve()
    const request = readLearningGraph(live, this.shutdownController.signal).then((graph) => { this.graph = graph; this.changed() })
    this.graphRequest = request
    return request.finally(() => { if (this.graphRequest === request) this.graphRequest = null })
  }

  async teach(text: string): Promise<EngineLearningSnapshot> {
    await this.init()
    if (this.paused) throw new Error('Living Weights is paused while learned data is erased.')
    if (typeof text !== 'string' || !text.trim() || Buffer.byteLength(text, 'utf8') > 8192) throw new Error('Teach a short fact or a few sentences, up to 8 KB.')
    if (this.busy) throw new Error('Wait for the current lesson or model preparation to finish.')
    const settings = this.deps.settings()
    if (!settings.supported || !settings.enabled || !settings.model || !this.deps.supervisor.status().learningActive) throw new Error('Enable Living Weights on the loaded Nemotron model before teaching it.')
    const hash = createHash('sha256').update(text.trim()).digest('hex')
    if (this.submitted.has(hash)) throw new Error('This lesson was already submitted. Inspect its graph instead of submitting it again; an interrupted lesson may already have changed the private copy.')
    if (this.submitted.size >= 4096) throw new Error('This private model has reached the 4096 lesson history limit; preserve it before preparing another learning profile.')
    this.pending = true
    this.completion = new Promise<void>((resolve) => { this.completed = resolve })
    this.cancelRequested = false
    this.sawFact = false
    this.message = null
    this.controller = new AbortController()
    let lease: Awaited<ReturnType<EngineSupervisor['acquire']>>
    try {
      lease = await this.deps.supervisor.acquire(settings.model, { start: false, signal: this.shutdownController.signal })
      if (!await this.deps.copies.existing(settings.model)) throw new Error('The private learning model is missing.')
      this.hash = hash
      this.submitted.add(hash)
      this.job = { id: randomUUID(), state: 'running', startedAt: Date.now(), finishedAt: null, learned: 0, missed: 0, message: null }
      await this.saveJob() // Persist before sending; a crash must never trigger an automatic replay.
    } catch (error) {
      this.pending = false
      this.controller = null
      this.submitted.delete(hash) // No POST was issued: retrying cannot duplicate a lesson.
      if (typeof lease! !== 'undefined') lease.release()
      this.completed?.()
      throw error
    }
    this.changed()
    this.running = this.run(lease, text.trim())
    return this.now()
  }

  async cancel(): Promise<EngineLearningSnapshot> {
    if (!this.pending || !this.controller) return this.snapshot(false)
    this.cancelRequested = true
    if (this.job) this.job.state = 'cancelling'
    this.message = 'Stopping at a safe learning boundary; facts already kept remain in the private copy.'
    // Waiting for the first fact makes the native save observable. Before that, let fact extraction end.
    if (this.sawFact) this.controller.abort()
    await this.saveJob()
    this.changed()
    return this.now()
  }

  private event(event: LessonEvent): void {
    if (!this.job) return
    if (event.kind === 'fact') { this.sawFact = true; if (this.cancelRequested) this.controller?.abort() }
    if (event.kind === 'learned') { if (event.recalled) this.job.learned++; else this.job.missed++; if (event.message) this.job.message = event.message }
    if (event.kind === 'failed') { this.job.state = 'failed'; this.job.message = event.message }
    this.changed()
  }

  private async run(lease: Awaited<ReturnType<EngineSupervisor['acquire']>>, text: string): Promise<void> {
    const controller = this.controller!
    let streamError: unknown = null
    const interval = setInterval(() => { void this.refresh().catch(() => {}) }, this.deps.pollMs ?? 1500)
    interval.unref()
    try {
      try { await streamLesson(lease, text, controller.signal, (event) => this.event(event)) } catch (error) { streamError = error }
      if ((streamError as any)?.accepted === false) {
        this.submitted.delete(this.hash!) // The engine refused before accepting any lesson.
        this.job!.state = 'failed'
        this.job!.message = (streamError as Error).message
        return
      }
      // The native [DONE] marker precedes its deferred checkpoint write. Keep the request lease
      // until the graph confirms that write ended; never kill/switch the engine during saving.
      const deadline = Date.now() + 30 * 60_000
      let reportedUncertain = false
      for (;;) {
        this.shutdownController.signal.throwIfAborted()
        if (!this.deps.supervisor.live(lease.model)) throw new Error('The engine exited during the lesson; inspect the graph before retrying.')
        await this.refresh()
        const save = this.graph?.model?.save
        const ours = this.graph?.facts.filter((fact) => fact.at >= this.job!.startedAt) ?? []
        const saved = save && save.at >= this.job!.startedAt && save.state !== 'saving'
        const idle = ours.every((fact) => fact.state !== 'learning')
        if (saved && idle) {
          if (save.state === 'failed') throw new Error('The lesson finished but its checkpoint could not be saved. Preserve the private copy and inspect its graph.')
          this.job!.learned = ours.filter((fact) => fact.state === 'learned').length
          this.job!.missed = ours.filter((fact) => fact.state === 'missed').length
          break
        }
        if (!streamError && !this.sawFact && ours.length === 0) break // no facts found; no checkpoint write occurs.
        if (Date.now() >= deadline && !reportedUncertain) {
          // No automatic replay or process restart is safe when the last save is unknown.
          this.job!.state = 'interrupted'
          this.job!.message = 'The engine has not confirmed the checkpoint save; model changes remain blocked until it finishes or the app exits.'
          reportedUncertain = true
          await this.saveJob(); this.changed()
        }
        await new Promise((resolve) => setTimeout(resolve, this.deps.pollMs ?? 1500))
      }
      if (this.job!.state !== 'failed') this.job!.state = this.cancelRequested ? 'cancelled' : streamError ? 'interrupted' : 'done'
      if (!streamError && !this.sawFact && this.job!.state === 'failed') this.submitted.delete(this.hash!)
      if (streamError && !this.cancelRequested) this.job!.message = 'The progress connection was interrupted; the graph shows the final saved facts. This lesson will not be retried automatically.'
    } catch (error) {
      this.job!.state = 'interrupted'
      this.job!.message = this.shutdownController.signal.aborted ? 'The app closed during the lesson; inspect its graph before retrying.' : (error as Error).message
    } finally {
      clearInterval(interval)
      this.job!.finishedAt = Date.now()
      this.message = null
      await this.saveJob().catch(() => { this.job!.state = 'interrupted'; this.job!.message = 'The final lesson status could not be saved. Inspect its graph before retrying.' })
      lease.release()
      this.pending = false
      this.controller = null
      this.completed?.()
      this.changed()
    }
  }

  shutdown(): void { this.shutdownController.abort(); this.controller?.abort() }

  async settle(): Promise<void> {
    if (this.pending) await this.cancel()
    await this.completion
  }

  setPaused(paused: boolean): void { this.paused = paused }

  async reset(): Promise<{ removedPaths: string[]; files: number }> {
    if (!this.paused || this.busy) throw new Error('Pause Living Weights and wait for its checkpoint save before erasing it.')
    await this.graphRequest?.catch(() => {})
    const file = this.journalFile()
    let removed = false
    try {
      const stat = await fs.lstat(file)
      if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || (process.getuid && stat.uid !== process.getuid())) throw new Error('The Living Weights job journal is not an app-owned independent file; refusing to erase it.')
      await fs.unlink(file)
      removed = true
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
    this.graph = null
    this.job = null
    this.hash = null
    this.submitted.clear()
    this.initialized = Promise.resolve()
    this.copyReady = false
    this.progress = null
    this.message = null
    this.running = null
    this.completion = null
    this.completed = null
    this.changed()
    return { removedPaths: removed ? [file] : [], files: removed ? 1 : 0 }
  }
}
