import { configureManagedComputeTransport, holdManagedComputeResponse } from '../../agent/managed-compute-transport'
import { createPinnedFetch } from '../../agent/pinned-fetch'
import { resolveRemoteOrchestratorRequest } from '../../agent/security'
import { isManagedComputeProvider } from '../../../shared/provider-trust'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { existsSync, promises as fs } from 'node:fs'
import type { SshManager } from './ssh'
import { store } from '../state'
import { saveSettings, vontraHomeDir } from '../../store'
import { requireSecurityRole } from './audit'
import { handle } from './ipc'
import { sendToRenderer } from './renderer-events'
import { redactSettings } from '../../agent/security'
import type { WorkspaceMutationGuard } from './workspace-mutation-guard'
import type { StoredClusterState } from './clusters/store'
import { ClusterWorkspaceStore } from './clusters/store'
import { ClusterManager } from './clusters/manager'
import { createClusterNetwork } from './clusters/network'
import { createClusterDiscovery } from './clusters/discovery'
import { createClusterSshSetup, type ClusterBootstrapRecord } from './clusters/ssh-setup'
import { clusterId, clusterObject, clusterNumber, clusterText, validateClusterCommandId } from '../../../shared/cluster-validation'

interface RuntimeOptions {
  sshManager: Pick<SshManager, 'resolveHost' | 'sessionForHost'>
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  serializeSettings<T>(operation: () => Promise<T>): Promise<T>
  bundlePath(): string
}
let configured: RuntimeOptions | null = null
let active: { root: string; manager: ClusterManager; network: ReturnType<typeof createClusterNetwork>; setup: ReturnType<typeof createClusterSshSetup>; initialized: Promise<void> } | null = null
let accepting = true, unresolved = false, checkedRoot: string | null = null
const discovery = createClusterDiscovery()
function observe(state: StoredClusterState) {
  checkedRoot = vontraHomeDir()
  unresolved = state.jobs.some(job => ['queued', 'running', 'reconciling', 'needs-review'].includes(job.phase)) || state.deployments.some(item => !['stopped', 'failed'].includes(item.state))
}
async function runtime() {
  requireSecurityRole('owner', 'manage compute machines')
  if (!configured || !accepting) throw new Error('Compute management is paused while this workspace stops.')
  const options = configured, root = vontraHomeDir()
  if (active?.root !== root) {
    if (active) { await active.manager.close(); await active.network.close() }
    const current = () => { requireSecurityRole('owner', 'manage compute machines'); if (vontraHomeDir() !== root || !accepting) throw new Error('The compute profile changed or paused.') }
    const protectedStore = new ClusterWorkspaceStore({ profile: root, assertCurrent: current, observe })
    let manager: ClusterManager
    const network = createClusterNetwork({ assertCurrent: current, hostCurrent: host => manager.hostCurrent(host), resolveSshHost: id => options.sshManager.resolveHost(id), sessionForHost: id => options.sshManager.sessionForHost(id) })
    manager = new ClusterManager({ store: protectedStore, assertCurrent: current, client: host => network.client(host), sshStamp: id => network.stamp(id), invalidate: id => network.invalidate(id), reserve: () => options.workspaceMutationGuard.reserveStart({ workspaceKeys: ['clusters:management'] }), useEndpoint: value => options.serializeSettings(async () => {
      current(); const previous = store.settings, captured = JSON.stringify(previous), id = `cluster:${value.id}`
      const next = { ...previous, providers: previous.providers.filter(row => row.id !== id).concat({ id, name: value.label, transport: 'chat-completions' as const, auth: 'api-key' as const, kind: 'cloud' as const, remoteCompute: true as const, source: 'custom' as const, baseUrl: value.baseUrl, apiKey: value.apiKey, models: [value.model] }) }
      const validateCommit = () => { current(); if (store.settings !== previous || JSON.stringify(previous) !== captured) throw new Error('Provider settings changed while saving the endpoint.') }
      validateCommit(); await saveSettings(next, { validateCommit }); validateCommit(); store.settings = next
      sendToRenderer('desktop:modeChanged', redactSettings(next)); return { providerId: id }
    }) })
    const setup = createClusterSshSetup({ profileRoot: () => { current(); return root }, getSshHost: id => options.sshManager.resolveHost(id), sessionForHost: id => options.sshManager.sessionForHost(id), agentBundle: async () => {
      current(); const bytes = await fs.readFile(options.bundlePath()); current()
      return { bytes, sha256: createHash('sha256').update(bytes).digest('hex'), version: '1', entry: 'cluster-node.js' }
    }, persistPreview: record => protectedStore.savePreview(record.preview, record as unknown as Record<string, unknown>), loadPreview: async id => (await protectedStore.getPreview(id))?.internal as unknown as ClusterBootstrapRecord ?? null, markApplied: (id, commandId, result) => protectedStore.markPreview(id, commandId, result), verifyAgent: input => network.verify({ ...input, session: input.session as Awaited<ReturnType<SshManager['sessionForHost']>> }) })
    active = { root, manager, network, setup, initialized: protectedStore.recover() }
  }
  const captured = active!; await captured.initialized
  if (active !== captured || root !== vontraHomeDir()) throw new Error('The compute workspace changed during initialization.')
  return captured
}
/** Startup reads receipts without opening sockets or assuming native process ownership. */
export async function loadClusterWorkspaceIdleState(): Promise<void> {
  const root = vontraHomeDir()
  const protectedStore = new ClusterWorkspaceStore({ profile: root, assertCurrent: () => { if (vontraHomeDir() !== root) throw new Error('The compute profile changed during startup.') } })
  try { const state = await protectedStore.get(); if (vontraHomeDir() === root) observe(state) }
  catch { if (vontraHomeDir() === root) { checkedRoot = root; unresolved = true } }
}
export function assertClusterWorkspaceIdle() {
  if (unresolved || active?.manager.running.size || checkedRoot !== vontraHomeDir() && existsSync(path.join(vontraHomeDir(), 'state', 'cluster-workspace.json'))) throw new Error('Open Compute and clusters to reconcile or stop remote work before replacing or clearing this workspace.')
}
export async function quiesceClusterWorkspace() {
  // Remote services are independent; closing the GUI only closes its local connections.
  discovery.cancel()
  if (active) { await active.manager.close(); await active.network.close(); active = null }
  await discovery.settle()
}
export function registerClusterRuntime(options: RuntimeOptions) {
  configured = options
  configureManagedComputeTransport(async (provider, rawUrl, init, assertRequestAllowed) => {
    const binding = (value: typeof provider | undefined) => value && JSON.stringify([value.id, value.transport ?? 'chat-completions', value.auth ?? 'api-key', value.kind, value.source, value.remoteCompute, value.baseUrl, value.apiKey, value.models])
    const root = vontraHomeDir(), captured = binding(provider)
    const assertCurrent = () => {
      requireSecurityRole('owner', 'use the verified compute endpoint')
      assertRequestAllowed?.()
      if (root !== vontraHomeDir() || binding(store.settings.providers.find(row => row.id === provider.id)) !== captured || !isManagedComputeProvider(provider)) throw new Error('The managed compute route changed; choose its current endpoint again.')
    }
    assertCurrent()
    const requested = new URL(rawUrl), configuredUrl = new URL(provider.baseUrl)
    if (requested.origin !== configuredUrl.origin || !['/v1/models', '/v1/chat/completions', '/v1/responses'].includes(requested.pathname) || requested.search || requested.hash || requested.username || requested.password || (init.method ?? 'GET') !== (requested.pathname === '/v1/models' ? 'GET' : 'POST')) throw new Error('Unsupported managed compute inference route.')
    const lease = options.workspaceMutationGuard.reserveStart({ workspaceKeys: ['clusters:inference'] })
    let response: Response | null = null, retained = false
    try {
      const value = await runtime(), deploymentId = provider.id.slice('cluster:'.length)
      const live = await value.manager.credentials(deploymentId); assertCurrent()
      if (provider.apiKey !== live.apiKey || !provider.models.includes(live.model)) throw new Error('The compute deployment credential changed; verify and use its endpoint again.')
      const state = await value.manager.options.store.get(), deployment = state.deployments.find(row => row.id === deploymentId)!, member = deployment.plan.members.find(row => row.rank === 0)!, host = state.hosts.find(row => row.id === member.hostId)!
      const targetUrl = new URL(requested.pathname, live.baseUrl).toString()
      const headers = new Headers(init.headers); headers.set('Authorization', `Bearer ${live.apiKey}`)
      const request = { ...init, headers, redirect: 'manual' as const, signal: init.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(10 * 60_000)]) : AbortSignal.timeout(10 * 60_000) }
      assertCurrent()
      if (host.connection.kind === 'ssh') response = await value.network.consumer(host, targetUrl, request)
      else {
        if (new URL(live.baseUrl).origin !== configuredUrl.origin) throw new Error('The configured endpoint origin changed; verify it before use.')
        const { target } = await resolveRemoteOrchestratorRequest(host.connection.url, targetUrl)
        assertCurrent(); response = await createPinnedFetch(target)(targetUrl, request)
      }
      assertCurrent()
      if (response.status >= 300 && response.status < 400) throw new Error('The managed endpoint redirected; no redirected request was sent.')
      const held = holdManagedComputeResponse(response, () => lease.release(), 32 * 1024 * 1024, assertCurrent); retained = true; return held
    } catch (error) { await response?.body?.cancel().catch(() => {}); throw error }
    finally { if (!retained) lease.release() }
  })
  const guarded = <T>(action: (value: Awaited<ReturnType<typeof runtime>>, input: unknown) => Promise<T>) => async (_event: unknown, input?: unknown) => {
    requireSecurityRole('owner', 'manage compute machines')
    const lease = options.workspaceMutationGuard.reserveStart({ workspaceKeys: ['clusters:management'] })
    try { const root = vontraHomeDir(), value = await runtime(), result = await action(value, input); if (root !== vontraHomeDir()) throw new Error('The compute profile changed before its result was returned.'); requireSecurityRole('owner', 'manage compute machines'); return result } finally { lease.release() }
  }
  const call = (name: string, action: (value: Awaited<ReturnType<typeof runtime>>, input: any) => Promise<any>) => handle(`cluster:${name}`, guarded((value, input) => action(value, input)))
  call('workspace', value => value.manager.snapshot())
  call('saveHost', (value, input) => value.manager.save(input))
  call('removeHost', (value, input) => { const raw = clusterObject(input, ['id', 'expectedRevision']); return value.manager.remove(clusterId(raw.id), clusterNumber(raw.expectedRevision)) })
  call('discover', async () => discovery.scan())
  call('inspectHost', (value, input) => { const raw = clusterObject(input, ['id', 'expectedRevision', 'commandId']); return value.manager.inspect(clusterId(raw.id), clusterNumber(raw.expectedRevision), validateClusterCommandId(raw.commandId)) })
  call('plan', (value, input) => value.manager.plan(input))
  for (const [name, kind] of [['ensureModel', 'ensure-model'], ['activate', 'activate'], ['stop', 'stop']] as const) call(name, (value, input) => {
    const raw = clusterObject(input, kind === 'stop' ? ['deploymentId', 'commandId'] : ['planId', 'digest', 'commandId'])
    return value.manager.begin(kind, { commandId: validateClusterCommandId(raw.commandId), ...(kind === 'stop' ? { deploymentId: clusterId(raw.deploymentId) } : { planId: clusterId(raw.planId), digest: clusterText(raw.digest, 64) }) })
  })
  call('cancelJob', (value, input) => { const raw = clusterObject(input, ['jobId']); return value.manager.cancel(clusterId(raw.jobId)) })
  call('reconcile', (value, input) => { const raw = clusterObject(input, ['deploymentId', 'jobId']); if (!!raw.deploymentId === !!raw.jobId) throw new Error('Choose one deployment or operation to reconcile.'); return value.manager.reconcile(raw.jobId ? { jobId: clusterId(raw.jobId) } : { deploymentId: clusterId(raw.deploymentId) }) })
  call('endpointCredentials', (value, input) => { const raw = clusterObject(input, ['deploymentId']); return value.manager.credentials(clusterId(raw.deploymentId)) })
  call('useEndpoint', (value, input) => { const raw = clusterObject(input, ['deploymentId']); return value.manager.use(clusterId(raw.deploymentId)) })
  call('sshKey', (value, input) => { const raw = clusterObject(input, ['action']); return value.setup.key({ action: raw.action as 'inspect' | 'create' }) })
  call('bootstrapPreview', (value, input) => { const raw = clusterObject(input, ['sshHostId', 'port']); return value.setup.preview({ sshHostId: clusterText(raw.sshHostId, 320), port: clusterNumber(raw.port, 1, 65535) }) })
  call('bootstrapApply', async (value, input) => {
    const raw = clusterObject(input, ['previewId', 'digest', 'commandId'])
    const result = await value.setup.apply({ previewId: clusterId(raw.previewId), digest: clusterText(raw.digest, 64), commandId: validateClusterCommandId(raw.commandId) })
    // Enrollment is internal; the renderer never receives management credentials.
    const previous = (await value.manager.options.store.get()).hosts.find(host => host.connection.kind === 'ssh' && host.connection.sshHostId === result.sshHostId && host.connection.port === result.port)
    await value.manager.save({ ...(previous ? { id: previous.id } : {}), expectedRevision: previous?.revision ?? 0, label: previous?.label ?? 'Compute machine', role: previous?.role ?? 'headless', connection: { kind: 'ssh', sshHostId: result.sshHostId, port: result.port }, token: result.token }, result.hostStamp)
    const enrolled = (await value.manager.options.store.get()).hosts.find(host => host.connection.kind === 'ssh' && host.connection.sshHostId === result.sshHostId && host.connection.port === result.port)!
    if (enrolled.sshStamp !== result.hostStamp) throw new Error('The SSH target changed after enrollment; its credential was not verified for use.')
    return value.manager.inspect(enrolled.id, enrolled.revision, validateClusterCommandId(raw.commandId))
  })
}
