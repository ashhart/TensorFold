import type { ClusterModel, ClusterNodeLaunch, ClusterProfile } from '../../../shared/cluster-workspace'
import type { EngineAdapter, EngineSession, EngineStream, NodeAdmissionInput } from '../types'

export const model: ClusterModel = { repo: 'Example/Small', revision: 'a'.repeat(40) }
export function deferred() { let resolve!: () => void; const promise = new Promise<void>(done => { resolve = done }); return { promise, resolve } }
export const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))
export async function until(check: () => boolean | Promise<boolean>, message = 'synthetic condition'): Promise<void> { const deadline = Date.now() + 5_000; while (!await check()) { if (Date.now() > deadline) throw new Error(`Timed out waiting for ${message}`); await sleep(5) } }
async function wait(signal: AbortSignal, gate: ReturnType<typeof deferred> | null) {
  if (signal.aborted) throw new Error('Synthetic cancellation')
  if (!gate) return
  let reject!: () => void
  const aborted = new Promise<never>((_resolve, fail) => { reject = () => fail(new Error('Synthetic cancellation')); signal.addEventListener('abort', reject, { once: true }) })
  try { await Promise.race([gate.promise, aborted]) } finally { signal.removeEventListener('abort', reject) }
}
export class FakeEngine implements EngineAdapter {
  readonly profile: ClusterProfile = { id: 'single-native', title: 'Synthetic native profile', mode: 'single', platform: process.platform as 'darwin' | 'linux', arch: process.arch as 'arm64' | 'x64', ranks: 1, transports: ['tcp'], modelTypes: ['qwen3_5'], poolsWeights: false, requiresFabric: false, engineVersion: '1.0.12' }
  cached = new Set<string>()
  ensureCalls = 0; startCalls = 0; stopCalls = 0; requestCalls = 0; physicalParsers = 0; maximumParsers = 0; physicalRequests = 0; maximumRequests = 0; cancelled = false; refuse = false
  requests: Array<Record<string, unknown>> = []
  ensureGate: ReturnType<typeof deferred> | null = null; ensureDrain: ReturnType<typeof deferred> | null = null
  startGate: ReturnType<typeof deferred> | null = null; requestDrain: ReturnType<typeof deferred> | null = null; requestGate: ReturnType<typeof deferred> | null = null
  afterEnsure: (() => void) | null = null
  mode: 'normal' | 'sse' | 'partial-sse' | 'partial-json' | 'slow' | 'large' = 'normal'
  key(value: ClusterModel) { return `${value.repo}@${value.revision}` }
  async inspect() { return { available: true, version: '1.0.12', profiles: [this.profile], models: [...this.cached].map(key => { const [repo, revision] = key.split('@'); return { repo, revision, modelType: 'qwen3_5', weightsBytes: 1024, memoryFloorBytes: 2048 } }), links: [], reason: null } }
  async admission(input: NodeAdmissionInput) { const cached = this.cached.has(this.key(input.model)); return { allowed: cached && !this.refuse && input.worldSize === 1 && input.transport === 'tcp', requiredBytes: cached ? 2048 : null, budgetBytes: 4096, cached, profile: this.profile, reason: this.refuse ? 'Synthetic native memory refusal' : cached ? 'Synthetic native admission' : 'Stage the exact revision first' } }
  async ensureModel(value: ClusterModel, signal: AbortSignal, progress: (text: string) => Promise<void>) {
    this.ensureCalls++; this.physicalParsers++; this.maximumParsers = Math.max(this.maximumParsers, this.physicalParsers)
    try { await progress('Synthetic immutable staging'); await wait(signal, this.ensureGate); this.cached.add(this.key(value)); this.afterEnsure?.() }
    catch (error) { this.cancelled = true; throw error }
    finally { if (this.ensureDrain) await this.ensureDrain.promise; this.physicalParsers-- }
  }
  async start(input: ClusterNodeLaunch, signal: AbortSignal, progress: (text: string) => Promise<void>): Promise<EngineSession> {
    this.startCalls++; await progress('Synthetic owned startup'); await wait(signal, this.startGate)
    let live = true
    const engine = this
    return { model: input.deploymentId, alive: () => live, stop: async () => { if (live) { engine.stopCalls++; live = false; if (engine.requestDrain) await engine.requestDrain.promise } },
      request: async (route, body, signal): Promise<EngineStream> => {
        if (!live || signal.aborted) throw new Error('Synthetic stopped request')
        engine.requestCalls++; engine.physicalRequests++; engine.maximumRequests = Math.max(engine.maximumRequests, engine.physicalRequests); engine.requests.push(body)
        const mode = engine.mode
        async function* output() {
          try {
            if (mode === 'normal' || mode === 'partial-json') { yield Buffer.from(JSON.stringify(route === '/v1/responses' ? { object: 'response', status: mode === 'normal' ? 'completed' : 'in_progress', output: [] } : { object: 'chat.completion', choices: [{ finish_reason: mode === 'normal' ? 'stop' : null, message: { content: 'ready' } }] })); return }
            if (mode === 'large') { yield Buffer.alloc(2 * 1024 * 1024, 120); return }
            yield Buffer.from('data: {"choices":[{"delta":{"content":"Synthetic text"},"finish_reason":null}]}\r')
            yield Buffer.from('\n\r\n')
            if (mode === 'slow') { try { await wait(signal, engine.requestGate ??= deferred()) } catch (error) { engine.cancelled = true; throw error } }
            if (mode !== 'partial-sse') yield Buffer.from('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
          } finally { if (engine.requestDrain) await engine.requestDrain.promise; engine.physicalRequests-- }
        }
        return { status: 200, contentType: mode === 'normal' || mode === 'partial-json' || mode === 'large' ? 'application/json' : 'text/event-stream', body: output() }
      },
    }
  }
}
