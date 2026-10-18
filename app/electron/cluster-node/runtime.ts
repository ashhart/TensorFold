import os from 'node:os'
import { randomUUID } from 'node:crypto'
import { validateClusterNodeAdmission, validateClusterNodeSnapshot } from '../../shared/cluster-validation'
import type { ClusterNodeAdmission, ClusterNodeJob, ClusterNodeLaunch, ClusterNodeSnapshot } from '../../shared/cluster-workspace'
import { NodeHttpError } from './config'
import { commandDigest } from './journal'
import type { EngineAdapter, EngineInspection, EngineSession, EngineStream, NodeAdmissionInput, NodeJournal, StoredNodeState } from './types'

type Work = { kind: ClusterNodeJob['kind']; input: any }
const pending = (job: ClusterNodeJob) => ['queued', 'running', 'reconciling'].includes(job.phase)
export class ClusterNodeRuntime {
  private queue = new Map<string, Work>()
  private running: Promise<void> | null = null
  private current: { id: string; kind: ClusterNodeJob['kind']; deploymentId?: string; controller: AbortController } | null = null
  private session: EngineSession | null = null
  private inference: AbortController | null = null
  private closing = false
  private durabilityLost = false
  private inspection: { at: number; value: EngineInspection } | null = null
  private inspecting: Promise<EngineInspection> | null = null
  private inspectionEpoch = 0
  constructor(readonly journal: NodeJournal, private engine: EngineAdapter, private now = Date.now) {}
  async initialize(): Promise<void> { await this.journal.read() }
  private async write<T>(edit: (state: StoredNodeState) => T): Promise<T> {
    if (this.durabilityLost) throw new NodeHttpError(503, 'Node receipt durability needs review before any further effects.')
    try { return await this.journal.update(edit) }
    catch (error) {
      if (error instanceof NodeHttpError) throw error
      this.durabilityLost = true; this.current?.controller.abort(); this.inference?.abort()
      if (this.session) await this.session.stop().catch(() => {})
      throw new NodeHttpError(503, 'The node receipt could not be saved; no further model effects are allowed until review.')
    }
  }
  async state(): Promise<StoredNodeState> {
    const state = await this.journal.read()
    if (this.durabilityLost) {
      for (const job of state.jobs) if (pending(job)) { job.phase = 'needs-review'; job.step = 'storage-review'; job.message = 'A receipt write failed; physical outcomes must be reconciled before retry.' }
      if (state.active) { state.active.state = 'needs-review'; state.active.endpointReady = false; state.active.canaryPassed = false; state.active.error = 'Node receipt durability was lost.' }
    }
    return state
  }
  async assertNode(nodeId: string): Promise<void> { if ((await this.state()).nodeId !== nodeId) throw new NodeHttpError(409, 'The node identity changed; enroll and inspect it again.') }
  private async inspect(): Promise<EngineInspection> {
    if (this.inspection && this.now() - this.inspection.at < 2_000) return this.inspection.value
    if (this.inspecting) return this.inspecting
    const epoch = this.inspectionEpoch
    const pending = this.engine.inspect(AbortSignal.timeout(30_000)).then(value => { if (epoch === this.inspectionEpoch) this.inspection = { at: this.now(), value }; return value })
    this.inspecting = pending
    try { return await pending } finally { if (this.inspecting === pending) this.inspecting = null }
  }
  async snapshot(loopback = true): Promise<ClusterNodeSnapshot> {
    const engine = await this.inspect(), state = await this.state()
    const active = state.active ? { ...state.active } : null
    if (active?.state === 'ready' && !this.session?.alive()) { active.state = 'failed'; active.endpointReady = false; active.canaryPassed = false; active.error = 'The owned native engine is no longer running.' }
    return validateClusterNodeSnapshot({ protocol: 1, nodeId: state.nodeId, version: '1.0.0', platform: process.platform, arch: process.arch, observedAt: this.now(), totalMemoryBytes: os.totalmem(), freeMemoryBytes: os.freemem(), engine: { available: engine.available && !this.durabilityLost, version: engine.version, profiles: engine.profiles, reason: this.durabilityLost ? 'Node receipt durability needs review.' : engine.reason }, models: engine.models, links: engine.links, active, ...(active ? { endpoint: { model: active.model.repo, ready: active.state === 'ready' && active.endpointReady && active.canaryPassed, loopback } } : {}) })
  }
  async admission(input: NodeAdmissionInput, signal = AbortSignal.timeout(30_000)): Promise<ClusterNodeAdmission> {
    await this.assertNode(input.expectedNodeId)
    const result = validateClusterNodeAdmission({ ...await this.engine.admission(input, signal), nodeId: input.expectedNodeId })
    const state = await this.state()
    if (state.jobs.some(job => job.phase === 'needs-review') || state.active && !['stopped', 'failed'].includes(state.active.state)) return { ...result, allowed: false, reason: 'Stop or reconcile the earlier owned deployment and uncertain receipts before another activation.' }
    const inspection = await this.inspect(), profile = inspection.profiles.find(item => item.id === input.profileId)
    if (result.allowed && (!inspection.available || !profile || result.profile?.id !== profile.id || commandDigest(result.profile) !== commandDigest(profile) || profile.engineVersion !== inspection.version || profile.ranks !== input.worldSize || !profile.transports.includes(input.transport) || profile.platform !== process.platform || profile.arch !== process.arch || result.requiredBytes === null || result.requiredBytes <= 0 || result.requiredBytes > result.budgetBytes || !result.cached)) return { ...result, allowed: false, reason: 'The node cannot confirm a matching engine profile, immutable cache and per-machine memory admission.' }
    if (result.allowed && profile && profile.mode !== 'single') {
      const link = inspection.links.find(link => link.transport === input.transport && link.available && link.verified && link.proof?.matched && link.proof.bytes > 0)
      if (!profile.settingsSchema || !profile.admission || !profile.checkpoint || !profile.linkDiagnostics || !profile.health || !link) return { ...result, allowed: false, reason: 'The distributed profile lacks a current native memory, checkpoint, byte-link or session contract.' }
    }
    return result
  }
  async accept(kind: ClusterNodeJob['kind'], input: any): Promise<ClusterNodeJob> {
    if (this.closing) throw new NodeHttpError(503, 'The node is stopping.')
    await this.assertNode(input.expectedNodeId)
    const digest = commandDigest({ kind, input })
    let accepted = false
    const job = await this.write(state => {
      const previous = state.receipts.find(receipt => receipt.commandId === input.commandId)
      if (previous) {
        if (previous.digest !== digest) throw new NodeHttpError(409, 'This command identity already belongs to a different request.')
        return state.jobs.find(job => job.id === previous.jobId)!
      }
      if (state.jobs.length >= 2048 || kind !== 'stop' && state.jobs.length >= 2000 || state.receipts.length >= 6144 || kind !== 'stop' && state.receipts.length >= 6080) throw new NodeHttpError(507, 'The bounded node receipt journal is full; review and archive it before submitting more operations.')
      if ((kind === 'stop' || kind === 'canary') && state.active && state.active.id !== input.deploymentId) throw new NodeHttpError(409, 'The command belongs to a different deployment.')
      const job: ClusterNodeJob = { id: `job-${randomUUID()}`, commandId: input.commandId, kind, phase: 'queued', step: 'accepted', message: 'Durable command accepted; no model effect has started.', createdAt: this.now(), updatedAt: this.now(), ...(input.model ? { model: input.model } : {}) }
      state.jobs.push(job); state.receipts.push({ commandId: input.commandId, digest, jobId: job.id }); accepted = true
      return job
    })
    if (accepted) {
      this.queue.set(job.id, { kind, input: structuredClone(input) })
      if (kind === 'stop') { if (this.current?.kind === 'start' && this.current.deploymentId === input.deploymentId) this.current.controller.abort(); const active = (await this.state()).active; if (active?.id === input.deploymentId) this.inference?.abort() }
      this.pump()
    }
    return job
  }
  async cancel(jobId: string, input: { commandId: string; expectedNodeId: string }): Promise<ClusterNodeJob> {
    await this.assertNode(input.expectedNodeId)
    const digest = commandDigest({ kind: 'cancel', jobId, input })
    const job = await this.write(state => {
      const target = state.jobs.find(job => job.id === jobId)
      if (!target) throw new NodeHttpError(404, 'This node job was not found.')
      const previous = state.receipts.find(receipt => receipt.commandId === input.commandId)
      if (previous && previous.digest !== digest) throw new NodeHttpError(409, 'This command identity already belongs to a different request.')
      if (!previous) { if (state.receipts.length >= 6144) throw new NodeHttpError(507, 'The bounded command journal is full.'); state.receipts.push({ commandId: input.commandId, digest, jobId }) }
      if (pending(target) && !state.cancelRequested.includes(jobId)) { state.cancelRequested.push(jobId); target.step = 'cancel-requested'; target.message = 'Cancellation saved; waiting for the owned physical work to drain.'; target.updatedAt = this.now() }
      return target
    })
    if (this.current?.id === jobId) this.current.controller.abort()
    this.pump()
    return job
  }
  private pump(): void {
    if (this.running || this.closing || this.durabilityLost || !this.queue.size) return
    this.running = this.run().finally(() => { this.running = null; this.pump() })
    void this.running.catch(() => {})
  }
  private async progress(jobId: string, message: string): Promise<void> {
    await this.write(state => { const job = state.jobs.find(job => job.id === jobId)!; if (state.cancelRequested.includes(jobId) || this.current?.controller.signal.aborted) throw new NodeHttpError(409, 'The owned work was cancelled.'); job.step = 'working'; job.message = message.slice(0, 1000); job.updatedAt = this.now() })
  }
  private async run(): Promise<void> {
    while (this.queue.size && !this.closing && !this.durabilityLost) {
      const [id, work] = this.queue.entries().next().value!
      this.queue.delete(id)
      const controller = new AbortController(); this.current = { id, kind: work.kind, deploymentId: work.input.deploymentId, controller }
      let effect = false, cleanupFailed = false
      try {
        await this.write(state => { const job = state.jobs.find(job => job.id === id)!; if (state.cancelRequested.includes(id)) controller.abort(); job.phase = 'running'; job.step = 'admission'; job.message = 'Rechecking ownership and native admission before effects.'; job.updatedAt = this.now() })
        if (controller.signal.aborted) throw new NodeHttpError(409, 'The queued operation was cancelled before effects.')
        const state = await this.state()
        if (state.active?.state === 'needs-review' || state.jobs.some(job => job.id !== id && job.phase === 'needs-review')) throw new NodeHttpError(409, 'The earlier physical deployment needs review; no process ownership was assumed after restart.')
        if (work.kind === 'ensure-model') {
          effect = true; await this.engine.ensureModel(work.input.model, controller.signal, message => this.progress(id, message))
        } else if (work.kind === 'start') {
          if (this.session || state.active && !['failed', 'stopped'].includes(state.active.state)) throw new NodeHttpError(409, 'Stop the owned deployment before starting another model.')
          const admission = await this.admission(work.input, controller.signal)
          if (!admission.allowed) throw new NodeHttpError(409, admission.reason)
          await this.write(state => { state.active = { id: work.input.deploymentId, commandId: work.input.commandId, state: 'starting', model: work.input.model, rank: work.input.rank, profileId: work.input.profileId, startedAt: null, error: null, endpointReady: false, canaryPassed: false } })
          effect = true
          this.session = await this.engine.start(work.input as ClusterNodeLaunch, controller.signal, message => this.progress(id, message))
          if (controller.signal.aborted || !this.session.alive()) throw new NodeHttpError(409, 'The model load was cancelled or its process exited.')
          await this.write(state => { state.active!.state = 'ready'; state.active!.startedAt = this.now(); state.active!.endpointReady = true })
        } else if (work.kind === 'stop') {
          if (state.active && state.active.id !== work.input.deploymentId) throw new NodeHttpError(409, 'The stop command belongs to a different deployment.')
          this.inference?.abort()
          await this.write(state => { if (state.active) { state.active.state = 'stopping'; state.active.endpointReady = false; state.active.canaryPassed = false } })
          if (this.session) { effect = true; await this.session.stop(); this.session = null }
          await this.write(state => { if (state.active) state.active.state = 'stopped' })
        } else {
          const active = state.active
          if (!active || active.id !== work.input.deploymentId || active.state !== 'ready' || !this.session?.alive()) throw new NodeHttpError(409, 'A ready owned deployment is required for its canary.')
          if (this.inference) throw new NodeHttpError(409, 'One inference request is already running on this node.')
          await this.write(state => { if (state.active?.id === active.id) state.active.canaryPassed = false })
          effect = true
          await this.runCanary(controller.signal, active.model.repo)
          await this.write(state => { if (state.active?.id !== active.id) throw new NodeHttpError(409, 'The canary deployment changed.'); state.active.canaryPassed = true })
        }
        if (controller.signal.aborted) throw new NodeHttpError(409, 'The owned operation was cancelled.')
        this.inspectionEpoch++; this.inspection = null; this.inspecting = null
        await this.write(state => { const job = state.jobs.find(job => job.id === id)!; job.phase = 'completed'; job.step = 'finished'; job.message = work.kind === 'start' ? 'Model readiness confirmed; a separate canary must pass before inference is exposed.' : 'Owned operation completed.'; job.updatedAt = this.now() })
      } catch (error) {
        if (work.kind === 'start' && this.session) { try { await this.session.stop(); this.session = null } catch { cleanupFailed = true } }
        if (work.kind === 'stop' && effect) cleanupFailed = true
        const phase = cleanupFailed || this.durabilityLost ? 'needs-review' : controller.signal.aborted ? 'cancelled' : 'failed'
        if (!this.durabilityLost) await this.write(state => {
          const job = state.jobs.find(job => job.id === id)!
          job.phase = phase; job.step = phase; job.message = error instanceof NodeHttpError ? error.message : effect ? 'Owned native work did not complete; any partial staging remains separate from a finished deployment.' : 'The node could not finish native inspection or admission.'; job.updatedAt = this.now()
          if (work.kind === 'start' && state.active && state.active.id === work.input.deploymentId) { state.active.state = cleanupFailed ? 'needs-review' : controller.signal.aborted ? 'stopped' : 'failed'; state.active.endpointReady = false; state.active.canaryPassed = false; state.active.error = job.message }
          if (work.kind === 'canary' && state.active && state.active.id === work.input.deploymentId) state.active.canaryPassed = false
          if (work.kind === 'stop' && state.active && cleanupFailed) state.active.state = 'needs-review'
        }).catch(() => {})
      } finally { this.current = null; this.inspectionEpoch++; this.inspection = null; this.inspecting = null }
    }
  }
  private async runCanary(signal: AbortSignal, model: string): Promise<void> {
    const lease = await this.requestLease({ model, messages: [{ role: 'user', content: 'Reply with the single word ready.' }], max_tokens: 8, temperature: 0, stream: false }, signal, true)
    try {
      const stream = await lease.request('/v1/chat/completions')
      if (stream.status !== 200 || !/^application\/json/i.test(stream.contentType)) throw new Error('The native canary did not return a complete JSON response.')
      const chunks: Buffer[] = []; let bytes = 0
      for await (const part of stream.body) { bytes += part.byteLength; if (bytes > 64 * 1024) throw new Error('The native canary exceeded its response limit.'); chunks.push(Buffer.from(part)) }
      const body = JSON.parse(Buffer.concat(chunks, bytes).toString('utf8')), choice = body?.choices?.[0]
      if (!choice || choice.finish_reason !== 'stop' || typeof choice.message?.content !== 'string' || !choice.message.content.trim() || choice.message.tool_calls?.length) throw new Error('The native canary was partial, empty or requested an unsupported tool action.')
      await lease.finish('completed')
    } catch (error) { await lease.finish(signal.aborted ? 'cancelled' : 'failed'); throw error }
  }
  async requestLease(body: Record<string, unknown>, signal: AbortSignal, canary = false, binding?: { deploymentId: string; commandId: string }): Promise<{ request(route: '/v1/chat/completions' | '/v1/responses'): Promise<EngineStream>; finish(phase: 'completed' | 'failed' | 'cancelled'): Promise<void> }> {
    if (this.closing || this.durabilityLost || this.inference || this.current?.kind === 'stop' || [...this.queue.values()].some(work => work.kind === 'stop')) throw new NodeHttpError(409, 'One owned inference request is already running or the node is stopping.')
    const controller = new AbortController(); this.inference = controller
    const abort = () => controller.abort(); signal.addEventListener('abort', abort, { once: true }); if (signal.aborted) abort()
    const id = `request-${randomUUID()}`
    try {
      await this.write(state => {
        if (binding && (state.active?.id !== binding.deploymentId || state.active.commandId !== binding.commandId)) throw new NodeHttpError(409, 'The authenticated deployment changed before inference dispatch.')
        if (!state.active || state.active.state !== 'ready' || !state.active.endpointReady || !canary && !state.active.canaryPassed || !this.session?.alive() || body.model !== state.active.model.repo) throw new NodeHttpError(409, 'The requested deployment is not ready and canary-qualified.')
        state.request = { id, phase: 'running', digest: commandDigest(body), updatedAt: this.now() }
      })
    } catch (error) { signal.removeEventListener('abort', abort); if (this.inference === controller) this.inference = null; throw error }
    let finished = false
    const owned = this.session!
    return {
      request: async route => { if (controller.signal.aborted) throw new NodeHttpError(409, 'Inference cancelled before dispatch.'); return owned.request(route, { ...body, model: owned.model }, controller.signal) },
      finish: async phase => {
        if (finished) return; finished = true
        if (phase !== 'completed') controller.abort()
        try { await this.write(state => { if (state.request?.id === id) { state.request.phase = phase; state.request.updatedAt = this.now() } }) }
        finally { signal.removeEventListener('abort', abort); if (this.inference === controller) this.inference = null }
      },
    }
  }
  async close(): Promise<void> {
    this.closing = true; this.current?.controller.abort(); this.inference?.abort()
    await this.running?.catch(() => {})
    if (this.session) { await this.session.stop(); this.session = null }
    if (!this.durabilityLost) await this.write(state => {
      for (const job of state.jobs) if (pending(job)) { job.phase = 'cancelled'; job.step = 'node-stopped'; job.message = 'Node shutdown drained its owned work; nothing will replay on restart.'; job.updatedAt = this.now() }
      if (state.active && state.active.state !== 'needs-review') { state.active.state = 'stopped'; state.active.endpointReady = false; state.active.canaryPassed = false }
      if (state.request?.phase === 'running') state.request.phase = 'cancelled'
    })
    await this.journal.drain()
  }
}
