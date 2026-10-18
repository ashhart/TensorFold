import { setTimeout as wait } from 'node:timers/promises'
import type { ClusterJob, ClusterNodeJob, ClusterPlan } from '../../../../shared/cluster-workspace'
import type { ClusterManager } from './manager'
import { planEvidence } from './manager'
import { buildClusterPlan, clusterDigest, memberAdmissionInput } from './planner'
import { ClusterRemoteError } from './client'

const terminal = (job: ClusterNodeJob) => ['completed', 'failed', 'cancelled'].includes(job.phase)
const command = (job: ClusterJob, host: string, kind: string) => clusterDigest([job.commandId, host, kind])
async function update(manager: ClusterManager, jobId: string, phase: ClusterJob['phase'], message: string) {
  await manager.options.store.transaction(state => { const job = state.jobs.find(row => row.id === jobId)!; job.phase = phase; job.message = message; job.updatedAt = Date.now() })
}
async function remoteStep(manager: ClusterManager, job: ClusterJob, hostId: string, kind: string, submit: (id: string) => Promise<ClusterNodeJob>, signal: AbortSignal) {
  manager.current(); signal.throwIfAborted()
  const commandId = command(job, hostId, kind)
  await manager.options.store.transaction(state => {
    if (state.remote.some(row => row.commandId === commandId)) throw new Error('This step already has a receipt; reconcile it instead of replaying it.')
    if (state.remote.length >= 2048) throw new Error('The remote receipt limit is reached.')
    state.remote.push({ jobId: job.id, hostId, nodeJobId: null, commandId, kind })
    const live = state.jobs.find(row => row.id === job.id)!
    live.phase = 'running'; live.step = kind; live.message = 'Awaiting the machine’s durable receipt.'; live.updatedAt = Date.now()
  })
  manager.current(); signal.throwIfAborted()
  let remote = await submit(commandId)
  if (remote.commandId !== commandId || remote.kind !== kind) throw new ClusterRemoteError('The machine returned a mismatched operation receipt.', true)
  await manager.options.store.transaction(state => { state.remote.find(row => row.commandId === commandId)!.nodeJobId = remote.id })
  const host = (await manager.options.store.get()).hosts.find(row => row.id === hostId)!, client = await manager.options.client(host)
  const deadline = Date.now() + 10 * 60_000
  while (!terminal(remote)) {
    manager.current(); signal.throwIfAborted()
    if (remote.phase === 'needs-review' || Date.now() >= deadline) throw new ClusterRemoteError('Remote work needs reconciliation before this machine can be reused.', true)
    if (manager.options.delay) await manager.options.delay(signal)
    else await wait(1000, undefined, { signal })
    remote = await client.job(remote.id, signal)
    if (remote.commandId !== commandId || remote.kind !== kind) throw new ClusterRemoteError('The remote receipt changed identity.', true)
  }
  if (remote.phase !== 'completed') throw new ClusterRemoteError('The remote operation did not complete; inspect its machine before retrying.', false)
}
async function freshAdmission(manager: ClusterManager, plan: ClusterPlan, activation: boolean) {
  const state = await manager.options.store.get(), evidence: Array<{ hostId: string; digest: string }> = []
  const { name, model, profileId, generation, members, transport } = plan
  const checked = await buildClusterPlan({ name, model, profileId, generation, members, transport }, { hosts: state.hosts, client: async host => {
    const client = await manager.options.client(host)
    return new Proxy(client, { get(target, key, receiver) { if (key === 'status') return async () => { const snapshot = await target.status(); evidence.push(planEvidence(host.id, snapshot, plan)); return snapshot }; return Reflect.get(target, key, receiver) } })
  } })
  if (activation ? checked.blockers.length > 0 : !checked.stageAllowed) throw new Error('Fresh engine, memory or link admission refused this plan; review it again.')
  const saved = state.evidence.find(row => row.planId === plan.id)
  if (!saved || clusterDigest([...saved.members].sort((a, b) => a.hostId.localeCompare(b.hostId))) !== clusterDigest(evidence.sort((a, b) => a.hostId.localeCompare(b.hostId)))) throw new Error('The checkpoint, engine profile or data-link proof changed after review; create a new plan.')
  if (activation && plan.admission.some(previous => { const current = checked.admission.find(row => row.hostId === previous.hostId); return !current || current.requiredBytes !== previous.requiredBytes })) throw new Error('The engine’s per-rank memory requirement changed; review a new plan.')
}
async function finishDeployment(manager: ClusterManager, job: ClusterJob, plan: ClusterPlan) {
  const state = await manager.options.store.get()
  let baseUrl = ''
  for (const member of plan.members) {
    const host = state.hosts.find(row => row.id === member.hostId)!, client = await manager.options.client(host), node = await client.status()
    if (job.kind === 'stop') {
      if (node.active && node.active.state !== 'stopped') throw new ClusterRemoteError('A machine still has active work; reconcile the stop before proceeding.', true)
    } else {
      const active = node.active
      if (!active || active.id !== job.deploymentId || active.state !== 'ready' || active.model.repo !== plan.model.repo || active.model.revision !== plan.model.revision || active.rank !== member.rank || active.profileId !== plan.profileId || member.rank === 0 && (!active.canaryPassed || !active.endpointReady)) throw new ClusterRemoteError('Every rank must confirm the exact pinned deployment before its endpoint is usable.', true)
      if (member.rank === 0) { const credential = await client.credentials(); if (credential.model !== plan.model.repo) throw new ClusterRemoteError('The inference alias does not match this deployment.', true); baseUrl = `${client.baseUrl}/v1` }
    }
  }
  await manager.options.store.transaction(next => {
    const deployment = next.deployments.find(row => row.id === job.deploymentId)!
    deployment.state = job.kind === 'stop' ? 'stopped' : 'ready'; deployment.error = null; deployment.updatedAt = Date.now()
    deployment.endpoint = job.kind === 'stop' ? null : { baseUrl, model: plan.model.repo, hasKey: true, location: 'remote', ready: true }
    if (job.kind === 'stop') for (const previous of next.jobs) if (previous.deploymentId === job.deploymentId && previous.id !== job.id && previous.phase === 'needs-review') { previous.phase = 'cancelled'; previous.message = 'The deployment was explicitly stopped on every machine.' }
  })
}
export async function runClusterOperation(manager: ClusterManager, job: ClusterJob, signal: AbortSignal) {
  let sent = false
  try {
    const state = await manager.options.store.get(), plan = state.plans.find(row => row.id === job.planId) ?? state.deployments.find(row => row.id === job.deploymentId)?.plan
    if (!plan) throw new Error('The reviewed plan is unavailable.')
    if (job.kind !== 'stop') await freshAdmission(manager, plan, job.kind === 'activate')
    for (const member of [...plan.members].sort((a, b) => b.rank - a.rank)) {
      manager.current(); signal.throwIfAborted()
      const host = (await manager.options.store.get()).hosts.find(row => row.id === member.hostId)!, client = await manager.options.client(host)
      if (job.kind === 'ensure-model') {
        sent = true
        await remoteStep(manager, job, host.id, 'ensure-model', id => client.ensure({ commandId: id, expectedNodeId: member.nodeId, model: plan.model }, signal), signal)
      } else if (job.kind === 'activate') {
        // Admission is repeated directly before loading each rank, not just when queued.
        const admitted = await client.admission(memberAdmissionInput(plan, member, host), signal)
        if (!admitted.allowed || !admitted.cached || admitted.requiredBytes === null || admitted.nodeId !== member.nodeId) throw new Error('The machine no longer admits this pinned model and context.')
        await manager.options.store.transaction(next => { const deployment = next.deployments.find(row => row.id === job.deploymentId)!; deployment.state = 'starting' })
        sent = true
        await remoteStep(manager, job, host.id, 'start', id => client.start({ deploymentId: job.deploymentId!, commandId: id, expectedNodeId: member.nodeId, model: plan.model, profileId: plan.profileId, role: host.role, generation: plan.generation, transport: plan.transport, rank: member.rank, worldSize: plan.members.length, links: member.link ? [member.link] : [] }, signal), signal)
      } else if (job.kind === 'stop') {
        sent = true
        await remoteStep(manager, job, host.id, 'stop', id => client.stop({ commandId: id, expectedNodeId: member.nodeId, deploymentId: job.deploymentId! }, signal), signal)
      }
    }
    if (job.kind === 'activate') {
      const member = plan.members.find(row => row.rank === 0)!, host = (await manager.options.store.get()).hosts.find(row => row.id === member.hostId)!, client = await manager.options.client(host)
      await remoteStep(manager, job, host.id, 'canary', id => client.canary({ commandId: id, expectedNodeId: member.nodeId, deploymentId: job.deploymentId! }, signal), signal)
    }
    if (job.kind !== 'ensure-model') await finishDeployment(manager, job, plan)
    await update(manager, job.id, 'completed', job.kind === 'ensure-model' ? 'The pinned checkpoint is staged; inspect and review a new plan before loading.' : job.kind === 'stop' ? 'Every machine confirmed the deployment stopped.' : 'Every rank matched and the inference canary passed; the endpoint is ready.')
  } catch (error) {
    // An interrupted request may have reached its target. Never replay or silently roll it back.
    const uncertain = sent || error instanceof ClusterRemoteError && error.uncertain
    await manager.options.store.transaction(state => {
      const live = state.jobs.find(row => row.id === job.id)!
      live.phase = uncertain ? 'needs-review' : signal.aborted ? 'cancelled' : 'failed'; live.updatedAt = Date.now()
      live.message = uncertain ? 'This operation may have affected remote machines; reconcile its durable receipts before retrying or changing them.' : signal.aborted ? 'Cancelled before remote work was sent.' : 'Fresh validation refused this operation; inspect the machines and review a new plan.'
      if (job.deploymentId) { const deployment = state.deployments.find(row => row.id === job.deploymentId)!; deployment.state = uncertain ? 'needs-review' : 'failed'; deployment.error = live.message; if (deployment.endpoint) deployment.endpoint.ready = false }
    }).catch(() => {})
  }
}
export async function reconcileClusterOperation(manager: ClusterManager, input: { deploymentId: string } | { jobId: string }) {
  const state = await manager.options.store.get()
  const deployment = 'deploymentId' in input ? state.deployments.find(row => row.id === input.deploymentId) : null
  const jobs = 'jobId' in input ? state.jobs.filter(row => row.id === input.jobId) : state.jobs.filter(row => row.deploymentId === input.deploymentId && (['needs-review', 'running', 'reconciling'].includes(row.phase) || deployment?.state === 'needs-review' && row.kind === 'activate' && row.phase === 'completed'))
  if (!jobs.length) throw new Error('Choose an operation with a durable receipt to reconcile.')
  for (const job of jobs) {
    if (manager.running.has(job.id)) throw new Error('Wait for this operation to settle before reconciliation.')
    const rows = state.remote.filter(row => row.jobId === job.id)
    if (!rows.length) { await update(manager, job.id, 'failed', 'No remote step was recorded; inspect and review a fresh plan.'); if (job.deploymentId) await manager.options.store.transaction(next => { const deployment = next.deployments.find(row => row.id === job.deploymentId)!; deployment.state = 'failed'; deployment.endpoint = null }); continue }
    let complete = true, failed = false
    for (const row of rows) {
      const host = state.hosts.find(item => item.id === row.hostId)!, client = await manager.options.client(host)
      const remote = row.nodeJobId ? await client.job(row.nodeJobId) : (await client.jobs()).find(item => item.commandId === row.commandId)
      if (!remote || remote.commandId !== row.commandId || remote.kind !== row.kind) { complete = false; continue }
      if (!row.nodeJobId) await manager.options.store.transaction(next => { next.remote.find(item => item.commandId === row.commandId)!.nodeJobId = remote.id })
      if (job.cancelRequested && !terminal(remote)) {
        // The cancel intent has its own stable identity on the remote journal.
        await client.cancel(remote.id, { commandId: command(job, host.id, `cancel-${row.kind}`), expectedNodeId: host.nodeId! })
        complete = false; continue
      }
      complete &&= terminal(remote); failed ||= remote.phase === 'failed' || remote.phase === 'cancelled'
    }
    const plan = state.plans.find(row => row.id === job.planId) ?? state.deployments.find(row => row.id === job.deploymentId)?.plan
    if (!complete || !plan) { await update(manager, job.id, 'needs-review', 'Remote receipts are missing or unfinished; no command was replayed.'); continue }
    if (failed) { await update(manager, job.id, job.kind === 'ensure-model' ? job.cancelRequested ? 'cancelled' : 'failed' : 'needs-review', job.kind === 'ensure-model' ? 'Remote staging drained; inspect the partial immutable cache and review a new plan before retrying.' : 'Remote work failed or was cancelled; explicitly stop any affected deployment before reusing its machines.'); continue }
    if (job.kind === 'ensure-model' && rows.filter(row => row.kind === 'ensure-model').length !== plan.members.length) { await update(manager, job.id, 'failed', 'The recorded staging steps settled; only part of this plan was staged, so inspect and review a fresh plan.'); continue }
    if (job.kind !== 'ensure-model') {
      try { await finishDeployment(manager, job, plan) } catch { await update(manager, job.id, 'needs-review', 'The machines have not confirmed the complete reviewed deployment.'); continue }
    }
    await update(manager, job.id, 'completed', 'The remote receipts and current machine state were reconciled without replaying work.')
  }
}

/** cancelRequested is committed before this stable remote cancellation identity is sent. */
export async function cancelClusterOperation(manager: ClusterManager, jobId: string) {
  const state = await manager.options.store.get(), job = state.jobs.find(row => row.id === jobId)!
  for (const row of state.remote.filter(row => row.jobId === jobId)) {
    const host = state.hosts.find(item => item.id === row.hostId)!
    try {
      const client = await manager.options.client(host)
      let remote = row.nodeJobId ? await client.job(row.nodeJobId) : (await client.jobs()).find(item => item.commandId === row.commandId)
      if (!remote || remote.commandId !== row.commandId || remote.kind !== row.kind || terminal(remote)) continue
      if (!row.nodeJobId) await manager.options.store.transaction(next => { next.remote.find(item => item.commandId === row.commandId)!.nodeJobId = remote!.id })
      await client.cancel(remote.id, { commandId: command(job, host.id, `cancel-${row.kind}`), expectedNodeId: host.nodeId! })
      const deadline = Date.now() + 15_000
      while (!terminal(remote) && Date.now() < deadline) {
        const signal = AbortSignal.timeout(2000)
        if (manager.options.delay) await manager.options.delay(signal); else await wait(250, undefined, { signal })
        remote = await client.job(remote.id)
        if (remote.commandId !== row.commandId) break
      }
    } catch { /* The saved cancel intent remains visible for explicit reconciliation. */ }
  }
  await reconcileClusterOperation(manager, { jobId }).catch(() => {})
}
