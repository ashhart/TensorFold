import { randomBytes } from 'node:crypto'
import type { ClusterNodeAdmission, ClusterNodeJob, ClusterNodeLaunch, ClusterNodeSnapshot } from '../../../../shared/cluster-workspace'
import { clusterId, clusterText, clusterList, clusterObject, validateClusterNodeAdmission, validateClusterNodeJob, validateClusterNodeSnapshot, validateClusterToken } from '../../../../shared/cluster-validation'
import type { NodeAdmissionInput } from '../../../cluster-node/types'
import { resolveRemoteOrchestratorRequest } from '../../../agent/security'
import { createPinnedFetch } from '../../../agent/pinned-fetch'

export class ClusterRemoteError extends Error {
  constructor(message: string, readonly uncertain: boolean) { super(message); this.name = 'ClusterRemoteError' }
}
export interface ClusterNodeClientOptions {
  baseUrl: string; token: string; nodeId: string | null
  assertCurrent(): void | Promise<void>; fetch?: typeof fetch; timeoutMs?: number
  /** A verified SSH byte channel or fixture may replace DNS/socket pinning; URL endpoints always resolve their configured origin. */
  request?: (url: string, init: RequestInit) => Promise<Response>
}
/** No redirects, no credential URL fields, and no automatic retry of a mutation. */
export class ClusterNodeClient {
  constructor(readonly options: ClusterNodeClientOptions) { validateClusterToken(options.token) }
  get baseUrl() { return this.options.baseUrl.replace(/\/$/, '') }
  async request(route: string, body?: unknown, signal?: AbortSignal): Promise<unknown> {
    if (!/^\/api\/cluster\/v1\/[A-Za-z0-9_/?=.-]+$/.test(route)) throw new Error('Invalid cluster protocol route.')
    await this.options.assertCurrent()
    const mutating = body !== undefined && route !== '/api/cluster/v1/admission'
    const combined = signal ? AbortSignal.any([signal, AbortSignal.timeout(this.options.timeoutMs ?? 15_000)]) : AbortSignal.timeout(this.options.timeoutMs ?? 15_000)
    const url = `${this.baseUrl}${route}`
    const init: RequestInit = { method: body === undefined ? 'GET' : 'POST', headers: { Authorization: `Bearer ${this.options.token}`, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) }, ...(body === undefined ? {} : { body: JSON.stringify(body) }), redirect: 'manual', signal: combined }
    let response: Response | null = null
    try {
      if (this.options.request) response = await this.options.request(url, init)
      else {
        const { target } = await resolveRemoteOrchestratorRequest(this.baseUrl, url)
        await this.options.assertCurrent()
        response = await createPinnedFetch(target, this.options.fetch ?? fetch)(url, init)
      }
      await this.options.assertCurrent()
      if (response.status >= 300 && response.status < 400) throw new ClusterRemoteError('The node redirected a credential-bearing request; no redirect was followed.', mutating)
      if (!response.ok) throw new ClusterRemoteError(`The node refused this request (${response.status}); inspect its status before retrying.`, mutating && response.status >= 500)
      const length = Number(response.headers.get('content-length'))
      if (Number.isFinite(length) && length > 2_000_000) throw new ClusterRemoteError('The node response exceeds its size limit.', mutating)
      const reader = response.body?.getReader(); if (!reader) throw new ClusterRemoteError('The node returned no response body.', mutating)
      const chunks: Uint8Array[] = []; let bytes = 0
      try {
        while (true) {
          const next = await reader.read(); if (next.done) break
          bytes += next.value.byteLength
          if (bytes > 2_000_000) throw new ClusterRemoteError('The node response exceeds its size limit.', mutating)
          chunks.push(next.value)
        }
      } catch (error) { await reader.cancel().catch(() => {}); throw error }
      await this.options.assertCurrent()
      const raw = Buffer.concat(chunks).toString('utf8').split(this.options.token).join('[redacted]')
      try { return JSON.parse(raw) } catch { throw new ClusterRemoteError('The node returned an incomplete or malformed response.', mutating) }
    } catch (error) {
      await response?.body?.cancel().catch(() => {})
      if (error instanceof ClusterRemoteError) throw error
      throw new ClusterRemoteError(mutating ? 'The node did not confirm this operation; reconcile its durable receipt before retrying.' : 'The node could not be verified; check its connection and inspect again.', mutating)
    }
  }
  async status(signal?: AbortSignal): Promise<ClusterNodeSnapshot> {
    const value = validateClusterNodeSnapshot(await this.request('/api/cluster/v1/status', undefined, signal))
    if (this.options.nodeId && value.nodeId !== this.options.nodeId) throw new ClusterRemoteError('The node identity changed; re-enroll it before sending more work.', false)
    return value
  }
  async identity(signal?: AbortSignal): Promise<string> {
    const nonce = randomBytes(24).toString('base64url')
    const value = clusterObject(await this.request(`/api/cluster/v1/identity?nonce=${nonce}`, undefined, signal), ['protocol', 'nodeId', 'nonce', 'authenticated'])
    if (value.protocol !== 1 || value.authenticated !== true || value.nonce !== nonce || this.options.nodeId && value.nodeId !== this.options.nodeId) throw new Error('The authenticated node challenge did not match.')
    return clusterId(value.nodeId)
  }
  async admission(input: NodeAdmissionInput, signal?: AbortSignal): Promise<ClusterNodeAdmission> { return validateClusterNodeAdmission(await this.request('/api/cluster/v1/admission', input, signal)) }
  async ensure(input: { commandId: string; expectedNodeId: string; model: ClusterNodeLaunch['model'] }, signal?: AbortSignal) { return validateClusterNodeJob(await this.request('/api/cluster/v1/models/ensure', input, signal)) }
  async start(input: ClusterNodeLaunch, signal?: AbortSignal) { return validateClusterNodeJob(await this.request('/api/cluster/v1/deployments/start', input, signal)) }
  async stop(input: { commandId: string; expectedNodeId: string; deploymentId: string }, signal?: AbortSignal) { return validateClusterNodeJob(await this.request('/api/cluster/v1/deployments/stop', input, signal)) }
  async canary(input: { commandId: string; expectedNodeId: string; deploymentId: string }, signal?: AbortSignal) { return validateClusterNodeJob(await this.request('/api/cluster/v1/canary', input, signal)) }
  async job(id: string, signal?: AbortSignal) { return validateClusterNodeJob(await this.request(`/api/cluster/v1/jobs/${clusterId(id)}`, undefined, signal)) }
  async jobs(signal?: AbortSignal) { const value = clusterObject(await this.request('/api/cluster/v1/jobs', undefined, signal), ['jobs']); return clusterList(value.jobs, 256).map(validateClusterNodeJob) }
  async cancel(id: string, input: { commandId: string; expectedNodeId: string }, signal?: AbortSignal) { return validateClusterNodeJob(await this.request(`/api/cluster/v1/jobs/${clusterId(id)}/cancel`, input, signal)) }
  async credentials(signal?: AbortSignal): Promise<{ apiKey: string; model: string }> {
    const value = clusterObject(await this.request('/api/cluster/v1/endpoint-credentials', undefined, signal), ['apiKey', 'model'])
    return { apiKey: validateClusterToken(value.apiKey), model: clusterText(value.model, 200) }
  }
}
