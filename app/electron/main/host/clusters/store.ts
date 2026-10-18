import { randomUUID } from 'node:crypto'
import path from 'node:path'
import type { ClusterBootstrapPreview, ClusterDeployment, ClusterHost, ClusterJob, ClusterPlan, ClusterWorkspaceSnapshot } from '../../../../shared/cluster-workspace'
import { clusterList, clusterNumber, clusterObject, clusterText, validateClusterNodeSnapshot, validateClusterPlanInput, validateClusterToken, validateSaveClusterHost, validateStoredClusterPlan, validateClusterJob, validateClusterDeployment, validateClusterCommandId, clusterId } from '../../../../shared/cluster-validation'
import { readProtectedJson, writeProtectedJson } from '../../../store/write-infra'

export interface StoredClusterHost extends ClusterHost { token: string | null; sshStamp: string | null }
export interface StoredClusterState {
  schema: 1; revision: number; hosts: StoredClusterHost[]; plans: ClusterPlan[]; jobs: ClusterJob[]; deployments: ClusterDeployment[]
  evidence: Array<{ planId: string; members: Array<{ hostId: string; digest: string }> }>
  operations: Array<{ commandId: string; digest: string; jobId: string }>
  remote: Array<{ jobId: string; hostId: string; nodeJobId: string | null; commandId: string; kind: string }>
  previews: Array<{ public: ClusterBootstrapPreview; internal: Record<string, unknown>; appliedCommand: string | null; state: 'preview' | 'applying' | 'needs-review' | 'verified' }>
}
export const emptyClusterState = (): StoredClusterState => ({ schema: 1, revision: 0, hosts: [], plans: [], jobs: [], deployments: [], remote: [], previews: [], operations: [], evidence: [] })
function clone<T>(value: T): T { return structuredClone(value) }
function validateState(value: unknown): StoredClusterState {
  const raw = clusterObject(value, ['schema', 'revision', 'hosts', 'plans', 'jobs', 'deployments', 'previews', 'remote', 'operations', 'evidence'], 'saved cluster workspace')
  if (raw.schema !== 1) throw new Error('Unsupported saved cluster schema; preserve the workspace before repairing it.')
  clusterNumber(raw.revision)
  const hosts = clusterList(raw.hosts, 64) as StoredClusterHost[]
  for (const host of hosts) {
    clusterObject(host, ['id', 'revision', 'label', 'role', 'connection', 'nodeId', 'hasCredential', 'snapshot', 'state', 'lastError', 'createdAt', 'updatedAt', 'token', 'sshStamp'])
    const parsed = validateSaveClusterHost({ id: host.id, expectedRevision: host.revision, label: host.label, role: host.role, connection: host.connection, ...(host.token ? { token: host.token } : {}) })
    if (host.token !== null) validateClusterToken(host.token)
    if (host.sshStamp !== null && !/^[a-f0-9]{64}$/.test(host.sshStamp)) throw new Error('Invalid SSH target binding.')
    if (host.snapshot !== null) validateClusterNodeSnapshot(host.snapshot)
    if (host.nodeId !== null && typeof host.nodeId !== 'string' || !['unverified', 'ready', 'offline', 'identity-changed'].includes(host.state)) throw new Error('Invalid saved cluster host.')
    if (parsed.id !== host.id || host.hasCredential !== !!host.token) throw new Error('Invalid cluster credential state.')
  }
  for (const field of ['hosts', 'plans', 'jobs', 'deployments'] as const) {
    const rows = clusterList(raw[field], field === 'hosts' ? 64 : 256) as Array<{ id: string }>
    if (rows.some(row => !row || typeof row.id !== 'string') || new Set(rows.map(row => row.id)).size !== rows.length) throw new Error('Invalid or duplicate saved cluster identity.')
  }
  for (const plan of raw.plans) validateStoredClusterPlan(plan)
  for (const job of raw.jobs) validateClusterJob(job)
  for (const deployment of raw.deployments) validateClusterDeployment(deployment)
  for (const item of clusterList(raw.remote, 2048)) { const row = clusterObject(item, ['jobId', 'hostId', 'nodeJobId', 'commandId', 'kind']); for (const key of ['jobId', 'hostId', 'commandId', 'kind']) clusterId(row[key]); if (row.nodeJobId !== null) clusterId(row.nodeJobId) }
  const evidence = clusterList(raw.evidence, 256)
  const evidenceIds = new Set<string>()
  for (const item of evidence) {
    const row = clusterObject(item, ['planId', 'members'], 'saved evidence'), planId = clusterId(row.planId)
    const plan = (raw.plans as ClusterPlan[]).find(plan => plan.id === planId)
    if (!plan || evidenceIds.has(planId)) throw new Error('Invalid or duplicate saved plan evidence.')
    evidenceIds.add(planId)
    const members = clusterList(row.members, 8), seen = new Set<string>()
    for (const member of members) {
      const node = clusterObject(member, ['hostId', 'digest'], 'saved evidence member'), hostId = clusterId(node.hostId)
      if (seen.has(hostId) || !plan.members.some(member => member.hostId === hostId) || typeof node.digest !== 'string' || !/^[a-f0-9]{64}$/.test(node.digest)) throw new Error('Invalid saved evidence digest or member.')
      seen.add(hostId)
    }
  }
  for (const item of clusterList(raw.operations, 512)) { const row = clusterObject(item, ['commandId', 'digest', 'jobId']); clusterId(row.commandId); clusterId(row.jobId); if (!/^[a-f0-9]{64}$/.test(row.digest as string)) throw new Error('Invalid operation receipt.') }
  const previewIds = new Set<string>()
  for (const item of clusterList(raw.previews, 128)) {
    const row = clusterObject(item, ['public', 'internal', 'appliedCommand', 'state'], 'saved preview')
    const preview = clusterObject(row.public, ['id', 'digest', 'sshHostId', 'platform', 'nodeAvailable', 'engineAvailable', 'steps', 'blockers', 'expiresAt'], 'saved preview')
    const id = clusterId(preview.id)
    if (previewIds.has(id) || typeof preview.digest !== 'string' || !/^[a-f0-9]{64}$/.test(preview.digest) || !['darwin', 'linux'].includes(preview.platform as string) || typeof preview.nodeAvailable !== 'boolean' || typeof preview.engineAvailable !== 'boolean') throw new Error('Invalid saved setup preview.')
    previewIds.add(id); clusterText(preview.sshHostId, 320); clusterNumber(preview.expiresAt, 1)
    for (const field of ['steps', 'blockers']) for (const message of clusterList(preview[field], 64)) clusterText(message, 2000)
    if (!row.internal || typeof row.internal !== 'object' || Array.isArray(row.internal) || !['preview', 'applying', 'needs-review', 'verified'].includes(row.state as string)) throw new Error('Invalid saved setup receipt.')
    if (row.appliedCommand !== null) validateClusterCommandId(row.appliedCommand)
    if ((row.state === 'preview') !== (row.appliedCommand === null)) throw new Error('Invalid setup command ownership.')
  }
  if (JSON.stringify(value).length > 12 * 1024 * 1024) throw new Error('The cluster workspace exceeds its storage bound.')
  return clone(raw as unknown as StoredClusterState)
}
export interface ClusterStoreOptions {
  profile: string; assertCurrent(): void; observe?(state: StoredClusterState): void
  read?: () => Promise<unknown>; write?: (value: StoredClusterState, current: () => void) => Promise<void>
}
/** No live side effect can be admitted until its protected transaction has committed. */
export class ClusterWorkspaceStore {
  private chain: Promise<unknown> = Promise.resolve()
  constructor(readonly options: ClusterStoreOptions) {}
  private async read(): Promise<StoredClusterState> {
    this.options.assertCurrent()
    const data = this.options.read ? await this.options.read() : await readProtectedJson(path.join(this.options.profile, 'state', 'cluster-workspace.json'), emptyClusterState(), false, true)
    this.options.assertCurrent(); const state = validateState(data); this.options.observe?.(state); return state
  }
  async get(): Promise<StoredClusterState> { await this.chain.catch(() => {}); return this.read() }
  transaction<T>(apply: (state: StoredClusterState) => T): Promise<T> {
    const next = this.chain.catch(() => {}).then(async () => {
      const state = await this.read(), result = apply(state)
      if (state.revision >= Number.MAX_SAFE_INTEGER) throw new Error('Cluster revision limit reached.')
      state.revision++
      const validated = validateState(state); this.options.assertCurrent()
      if (this.options.write) await this.options.write(validated, this.options.assertCurrent)
      else await writeProtectedJson(path.join(this.options.profile, 'state', 'cluster-workspace.json'), validated, { isEmpty: false, prepareData: () => { this.options.assertCurrent(); return validated }, validateCommit: this.options.assertCurrent })
      this.options.assertCurrent(); this.options.observe?.(validated); return clone(result)
    })
    this.chain = next; return next
  }
  async publicSnapshot(): Promise<ClusterWorkspaceSnapshot> {
    const state = await this.get()
    return { revision: state.revision, hosts: state.hosts.map(({ token: _token, sshStamp: _stamp, ...host }) => host), plans: state.plans, jobs: state.jobs, deployments: state.deployments }
  }
  async recover(): Promise<void> {
    const state = await this.get()
    if (!state.hosts.some(host => host.state === 'ready') && !state.jobs.some(job => ['queued', 'running', 'reconciling'].includes(job.phase)) && !state.deployments.some(item => ['preparing', 'starting', 'ready', 'stopping'].includes(item.state))) return
    await this.transaction(next => {
      for (const host of next.hosts) if (host.state === 'ready') { host.state = 'unverified'; host.lastError = 'Inspect this machine after restarting the app.' }
      for (const job of next.jobs) if (['queued', 'running', 'reconciling'].includes(job.phase)) { job.phase = 'needs-review'; job.step = 'reconcile'; job.message = 'The app restarted before this operation was confirmed; inspect its remote receipt before continuing.'; job.updatedAt = Date.now() }
      for (const deployment of next.deployments) if (['preparing', 'starting', 'ready', 'stopping'].includes(deployment.state)) { deployment.state = 'needs-review'; if (deployment.endpoint) deployment.endpoint.ready = false; deployment.updatedAt = Date.now() }
    })
  }
  async savePreview(preview: ClusterBootstrapPreview, internal: Record<string, unknown>): Promise<void> {
    await this.transaction(state => { state.previews = state.previews.filter(row => row.public.expiresAt > Date.now()); if (state.previews.length >= 128) throw new Error('Too many cluster setup previews.'); state.previews.push({ public: preview, internal, appliedCommand: null, state: 'preview' }) })
  }
  async getPreview(id: string) { return (await this.get()).previews.find(row => row.public.id === id) ?? null }
  async markPreview(id: string, commandId: string, result: { state: 'applying' | 'needs-review' | 'verified' }): Promise<void> {
    await this.transaction(state => {
      const row = state.previews.find(item => item.public.id === id)
      if (!row) throw new Error('Setup preview is no longer available.')
      if (result.state === 'applying' && row.state !== 'preview') throw new Error('Setup already has a receipt; reconcile the remote service before retrying.')
      if (row.appliedCommand && row.appliedCommand !== commandId) throw new Error('This setup preview belongs to another operation.')
      row.appliedCommand = commandId; row.state = result.state
    })
  }
  newId(): string { return randomUUID() }
}
