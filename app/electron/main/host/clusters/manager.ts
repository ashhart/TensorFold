import { randomUUID } from 'node:crypto'
import type { ClusterDeployment, ClusterHost, ClusterJob, ClusterNodeSnapshot, ClusterPlan, ClusterPlanInput, SaveClusterHost } from '../../../../shared/cluster-workspace'
import { clusterId, validateClusterCommandId, validateSaveClusterHost, validateClusterPlanInput } from '../../../../shared/cluster-validation'
import { ClusterWorkspaceStore, type StoredClusterHost } from './store'
import type { ClusterNodeClient } from './client'
import { buildClusterPlan, clusterDigest } from './planner'
import { runClusterOperation, reconcileClusterOperation, cancelClusterOperation } from './operations'

export interface ClusterManagerOptions {
  store: ClusterWorkspaceStore
  assertCurrent(): void
  client(host: StoredClusterHost): Promise<ClusterNodeClient>
  sshStamp(id: string): Promise<string>
  invalidate(id: string): Promise<void>
  reserve(): { release(): void }
  useEndpoint(value: { id: string; label: string; baseUrl: string; apiKey: string; model: string }): Promise<{ providerId: string }>
  delay?: (signal: AbortSignal) => Promise<void>
}
export function planEvidence(hostId: string, snapshot: ClusterNodeSnapshot, input: ClusterPlanInput) {
  return { hostId, digest: clusterDigest({ nodeId: snapshot.nodeId, platform: snapshot.platform, arch: snapshot.arch, engine: snapshot.engine, models: snapshot.models.filter(model => model.repo === input.model.repo && model.revision === input.model.revision), links: input.transport === 'tcp' ? [] : snapshot.links }) }
}
export class ClusterManager {
  readonly running = new Map<string, { controller: AbortController; promise: Promise<void> }>()
  private closed = false
  constructor(readonly options: ClusterManagerOptions) {}
  current = () => { this.options.assertCurrent(); if (this.closed) throw new Error('The cluster workspace closed.') }
  async snapshot() { this.current(); return this.options.store.publicSnapshot() }
  async hostCurrent(host: StoredClusterHost) {
    this.current(); const current = (await this.options.store.get()).hosts.find(item => item.id === host.id); this.current()
    if (!current || current.revision !== host.revision || current.token !== host.token || current.sshStamp !== host.sshStamp || JSON.stringify(current.connection) !== JSON.stringify(host.connection)) throw new Error('The machine configuration changed; inspect and review it again.')
  }
  private assertEditable(state: Awaited<ReturnType<ClusterWorkspaceStore['get']>>, id: string) {
    if (state.jobs.some(job => job.hostIds.includes(id) && ['queued', 'running', 'reconciling', 'needs-review'].includes(job.phase)) || state.deployments.some(item => item.plan.members.some(member => member.hostId === id) && !['stopped', 'failed'].includes(item.state))) throw new Error('Reconcile or stop work on this machine before changing or removing it.')
  }
  async save(raw: SaveClusterHost, expectedSshStamp?: string) {
    this.current(); const input = validateSaveClusterHost(raw)
    const state = await this.options.store.get(), old = state.hosts.find(host => host.id === input.id)
    if (old) this.assertEditable(state, old.id)
    if (input.id && !old || old && old.revision !== input.expectedRevision || !old && input.expectedRevision !== 0) throw new Error('This machine changed; refresh before saving.')
    const same = !!old && JSON.stringify(old.connection) === JSON.stringify(input.connection)
    let token = input.token || (same ? old?.token : null) || null
    let sshStamp = input.connection.kind === 'ssh' && token ? await this.options.sshStamp(input.connection.sshHostId) : null
    this.current()
    if (input.connection.kind === 'ssh' && !input.token && old?.sshStamp !== sshStamp) { token = null; sshStamp = null }
    if (expectedSshStamp && sshStamp !== expectedSshStamp) throw new Error('The verified SSH enrollment target changed before saving.')
    const id = old?.id ?? randomUUID(), now = Date.now()
    await this.options.store.transaction(next => {
      const live = next.hosts.find(host => host.id === id)
      if (live?.revision !== old?.revision) throw new Error('This machine changed while saving.')
      this.assertEditable(next, id)
      if (!live && next.hosts.length >= 64) throw new Error('The machine limit is reached.')
      const host: StoredClusterHost = { id, revision: (old?.revision ?? 0) + 1, label: input.label, role: input.role, connection: input.connection, token, sshStamp, hasCredential: !!token, nodeId: same && old?.sshStamp === sshStamp && old.token === token ? old.nodeId : null, snapshot: null, state: 'unverified', lastError: null, createdAt: old?.createdAt ?? now, updatedAt: now }
      next.hosts = next.hosts.filter(row => row.id !== id).concat(host)
    })
    await this.options.invalidate(id); return this.snapshot()
  }
  async remove(id: string, revision: number) {
    this.current(); clusterId(id)
    await this.options.store.transaction(state => {
      const host = state.hosts.find(row => row.id === id)
      if (!host || host.revision !== revision) throw new Error('Refresh this machine before removing it.')
      this.assertEditable(state, id); state.hosts = state.hosts.filter(row => row.id !== id)
    })
    await this.options.invalidate(id); return this.snapshot()
  }
  async inspect(id: string, revision: number, commandId: string) {
    this.current(); validateClusterCommandId(commandId)
    const host = (await this.options.store.get()).hosts.find(row => row.id === id)
    if (!host || host.revision !== revision) throw new Error('Refresh this machine before inspection.')
    try {
      const client = await this.options.client(host), snapshot = await client.status()
      await this.options.store.transaction(state => {
        const live = state.hosts.find(row => row.id === id)
        if (!live || live.revision !== revision) throw new Error('This machine changed during inspection.')
        live.nodeId = snapshot.nodeId; live.snapshot = snapshot; live.state = 'ready'; live.lastError = null; live.updatedAt = Date.now()
      })
    } catch {
      await this.options.store.transaction(state => { const live = state.hosts.find(row => row.id === id); if (live?.revision === revision) { live.state = 'offline'; live.snapshot = null; live.lastError = 'Inspection failed; verify the saved address, SSH target, node identity and management credential.' } })
    }
    return this.snapshot()
  }
  async plan(raw: ClusterPlanInput) {
    this.current(); const input = validateClusterPlanInput(raw), state = await this.options.store.get()
    const evidence: Array<{ hostId: string; digest: string }> = []
    const plan = await buildClusterPlan(input, { hosts: state.hosts, client: async host => {
      const client = await this.options.client(host)
      return new Proxy(client, { get(target, key, receiver) { if (key === 'status') return async () => { const snapshot = await target.status(); evidence.push(planEvidence(host.id, snapshot, input)); return snapshot }; return Reflect.get(target, key, receiver) } })
    } })
    await this.options.store.transaction(next => {
      for (const member of input.members) if (next.hosts.find(host => host.id === member.hostId)?.revision !== member.hostRevision) throw new Error('A machine changed while planning.')
      const retained = next.plans.filter(row => row.expiresAt > Date.now() || next.deployments.some(item => item.plan.id === row.id) || next.jobs.some(job => job.planId === row.id && ['queued', 'running', 'reconciling', 'needs-review'].includes(job.phase)))
      if (retained.length >= 256) throw new Error('The saved plan limit is reached; reconcile older work first.')
      next.plans = retained.concat(plan); next.evidence = next.evidence.filter(row => retained.some(item => item.id === row.planId)).concat({ planId: plan.id, members: evidence })
    })
    return plan
  }
  async begin(kind: 'ensure-model' | 'activate' | 'stop', input: { commandId: string; planId?: string; digest?: string; deploymentId?: string }) {
    this.current(); validateClusterCommandId(input.commandId)
    const payloadDigest = clusterDigest({ kind, ...input }), lease = this.options.reserve()
    let job: ClusterJob | undefined, duplicate = false, launched = false
    try {
      await this.options.store.transaction(state => {
        const operation = state.operations.find(row => row.commandId === input.commandId)
        if (operation) { if (operation.digest !== payloadDigest) throw new Error('This command identity already belongs to a different operation.'); job = state.jobs.find(row => row.id === operation.jobId); duplicate = true; return }
        if (state.operations.length >= 512 || state.jobs.length >= 256) throw new Error('The durable receipt limit is reached; preserve and reconcile older work first.')
        let plan: ClusterPlan
        if (kind === 'stop') {
          const deployment = state.deployments.find(row => row.id === input.deploymentId)
          if (!deployment || deployment.state === 'stopped') throw new Error('Choose an existing deployment to stop.')
          if (state.jobs.some(row => row.deploymentId === deployment.id && this.running.has(row.id))) throw new Error('Cancel or settle this deployment’s current operation first.')
          plan = deployment.plan
        } else {
          const saved = state.plans.find(row => row.id === input.planId)
          if (!saved || saved.digest !== input.digest || saved.expiresAt <= Date.now()) throw new Error('Review a fresh plan before sending work.')
          if (kind === 'activate' && saved.blockers.length || kind === 'ensure-model' && !saved.stageAllowed) throw new Error('This plan has unresolved engine, hardware or memory blockers.')
          plan = saved
          for (const member of plan.members) this.assertEditable(state, member.hostId)
        }
        for (const member of plan.members) { const host = state.hosts.find(row => row.id === member.hostId); if (!host || host.revision !== member.hostRevision || host.nodeId !== member.nodeId) throw new Error('A reviewed machine changed; inspect and re-plan.') }
        const now = Date.now(), deploymentId = kind === 'activate' ? randomUUID() : input.deploymentId
        job = { id: randomUUID(), commandId: input.commandId, kind, phase: 'queued', hostIds: plan.members.map(member => member.hostId), planId: plan.id, ...(deploymentId ? { deploymentId } : {}), step: 'admission', message: 'Waiting for fresh per-machine admission.', createdAt: now, updatedAt: now, cancelRequested: false }
        state.jobs.push(job); state.operations.push({ commandId: input.commandId, digest: payloadDigest, jobId: job.id })
        if (kind === 'activate') state.deployments.push({ id: deploymentId!, name: plan.name, plan, state: 'preparing', endpoint: null, createdAt: now, updatedAt: now, error: null })
      })
      if (!job || duplicate) { lease.release(); return this.snapshot() }
      const controller = new AbortController(), owned = job
      const promise = runClusterOperation(this, owned, controller.signal).finally(() => { this.running.delete(owned.id); lease.release() })
      this.running.set(job.id, { controller, promise }); launched = true; void promise.catch(() => {})
      return this.snapshot()
    } catch (error) { if (!launched) lease.release(); throw error }
  }
  async cancel(jobId: string) {
    this.current(); clusterId(jobId)
    await this.options.store.transaction(state => { const job = state.jobs.find(row => row.id === jobId); if (!job || !['queued', 'running', 'reconciling', 'needs-review'].includes(job.phase)) throw new Error('Choose unfinished work to cancel.'); job.cancelRequested = true; job.message = 'Cancellation requested; remote receipts must settle before this machine is reusable.' })
    const running = this.running.get(jobId)
    running?.controller.abort(); await running?.promise
    await cancelClusterOperation(this, jobId)
    return this.snapshot()
  }
  async reconcile(input: { deploymentId: string } | { jobId: string }) { this.current(); await reconcileClusterOperation(this, input); return this.snapshot() }
  async credentials(id: string) {
    this.current(); const state = await this.options.store.get(), deployment = state.deployments.find(row => row.id === id)
    if (!deployment || deployment.state !== 'ready' || !deployment.endpoint?.ready) throw new Error('Reconcile and pass its inference canary before using this endpoint.')
    const member = deployment.plan.members.find(row => row.rank === 0)!, host = state.hosts.find(row => row.id === member.hostId)
    if (!host) throw new Error('The endpoint machine is missing.')
    const client = await this.options.client(host), snapshot = await client.status()
    if (snapshot.active?.id !== deployment.id || !snapshot.active.canaryPassed || snapshot.active.state !== 'ready' || snapshot.active.model.repo !== deployment.plan.model.repo || snapshot.active.model.revision !== deployment.plan.model.revision || snapshot.active.rank !== 0 || snapshot.active.profileId !== deployment.plan.profileId || !snapshot.active.endpointReady) throw new Error('The remote deployment changed; reconcile before using it.')
    const credentials = await client.credentials(); this.current()
    if (credentials.model !== deployment.plan.model.repo) throw new Error('The inference alias changed; inspect the remote deployment.')
    return { baseUrl: `${client.baseUrl}/v1`, ...credentials }
  }
  async use(id: string) {
    const value = await this.credentials(id), deployment = (await this.options.store.get()).deployments.find(row => row.id === id)!
    this.current(); return this.options.useEndpoint({ id, label: deployment.name, ...value })
  }
  async idle() { const state = await this.options.store.get(); return this.running.size === 0 && !state.jobs.some(job => ['queued', 'running', 'reconciling', 'needs-review'].includes(job.phase)) && !state.deployments.some(item => !['stopped', 'failed'].includes(item.state)) }
  async settle() { await Promise.allSettled([...this.running.values()].map(row => row.promise)) }
  async close() { this.closed = true; for (const row of this.running.values()) row.controller.abort(); await this.settle() }
}
