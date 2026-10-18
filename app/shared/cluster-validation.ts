import type { ClusterGeneration, ClusterLinkSettings, ClusterMember, ClusterModel, ClusterNodeAdmission, ClusterNodeDeployment, ClusterNodeJob, ClusterNodeLaunch, ClusterNodeSnapshot, ClusterPlanInput, ClusterProfile, SaveClusterHost } from './cluster-workspace'

export function clusterObject(value: unknown, keys: string[], label = 'cluster request'): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !keys.includes(key))) throw new Error(`Invalid ${label}.`)
  return value as Record<string, any>
}
export function clusterText(value: unknown, limit = 160, empty = false): string {
  if (typeof value !== 'string' || value.length > limit || !empty && !value.trim() || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value)) throw new Error('Invalid cluster text.')
  return value
}
export function clusterId(value: unknown): string {
  const text = clusterText(value, 160)
  if (!/^[A-Za-z0-9][A-Za-z0-9_.:-]*$/.test(text)) throw new Error('Invalid cluster identity.')
  return text
}
export function clusterNumber(value: unknown, min = 0, max = Number.MAX_SAFE_INTEGER, integer = true): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || integer && !Number.isSafeInteger(value)) throw new Error('Invalid cluster number.')
  return value
}
export function clusterList(value: unknown, max: number): unknown[] {
  if (!Array.isArray(value) || value.length > max) throw new Error('Cluster list is too large or invalid.')
  return value
}
const boolean = (value: unknown): boolean => { if (typeof value !== 'boolean') throw new Error('Invalid cluster flag.'); return value }
const choice = <T extends string>(value: unknown, choices: readonly T[]): T => { if (!choices.includes(value as T)) throw new Error('Unsupported cluster option.'); return value as T }
export function validateClusterCommandId(value: unknown): string {
  const id = clusterId(value)
  if (id.length < 8) throw new Error('Use a durable command identity of at least eight characters.')
  return id
}
export function validateClusterToken(value: unknown): string {
  const token = clusterText(value, 256)
  if (!/^[A-Za-z0-9_-]{43,256}$/.test(token)) throw new Error('Use a cluster credential with at least 256 bits of entropy.')
  return token
}
export function validateClusterModel(value: unknown): ClusterModel {
  const raw = clusterObject(value, ['repo', 'revision'], 'model')
  const repo = clusterText(raw.repo, 192), revision = clusterText(raw.revision, 40)
  const parts = repo.split('/')
  if (parts.length !== 2 || parts.some(part => !/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/.test(part) || part.includes('..') || part.includes('--'))) throw new Error('Choose a valid model repository.')
  if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error('Pin an immutable 40-character model revision.')
  return { repo, revision }
}
export function validateClusterGeneration(value: unknown): ClusterGeneration {
  const raw = clusterObject(value, ['contextTokens', 'maxOutputTokens', 'temperature'], 'generation settings')
  const result = { contextTokens: clusterNumber(raw.contextTokens, 128, 1_048_576), maxOutputTokens: clusterNumber(raw.maxOutputTokens, 1, 65_536), temperature: clusterNumber(raw.temperature, 0, 2, false) }
  if (result.maxOutputTokens > result.contextTokens) throw new Error('The reply reserve must fit the context window.')
  return result
}
export function validateClusterLink(value: unknown): ClusterLinkSettings {
  const raw = clusterObject(value, ['peer', 'device', 'via', 'port', 'peerPort', 'name', 'gid'], 'rank link')
  const device = clusterText(raw.device, 80), via = clusterText(raw.via, 253), name = clusterText(raw.name, 20)
  if (!/^[A-Za-z0-9_.-]+$/.test(device) || !/^[A-Za-z0-9_.:%-]+(?:\/[A-Za-z0-9_.:%-]+)?$/.test(via) || !/^[A-Za-z0-9_.-]+$/.test(name)) throw new Error('Use a valid network device, peer address and link name.')
  return { peer: clusterNumber(raw.peer, 0, 7), device, via, port: clusterNumber(raw.port, 1, 65_535), name,
    ...(raw.peerPort === undefined ? {} : { peerPort: clusterNumber(raw.peerPort, 1, 65_535) }), ...(raw.gid === undefined ? {} : { gid: clusterNumber(raw.gid, -1, 65_535) }) }
}
export function validateClusterConnection(value: unknown): SaveClusterHost['connection'] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Choose a cluster connection.')
  const kind = (value as any).kind
  if (kind === 'ssh') { const raw = clusterObject(value, ['kind', 'sshHostId', 'port']); return { kind, sshHostId: clusterId(raw.sshHostId), port: clusterNumber(raw.port, 1, 65_535) } }
  const raw = clusterObject(value, ['kind', 'url']); if (raw.kind !== 'endpoint') throw new Error('Choose an HTTPS endpoint or a saved SSH host.')
  const url = new URL(clusterText(raw.url, 2048)), host = url.hostname.toLowerCase().replace(/^\[|\]$/g, '')
  const loopback = host === 'localhost' || host === '::1' || /^127(?:\.\d{1,3}){3}$/.test(host)
  if (url.username || url.password || url.search || url.hash || url.pathname !== '/' || !['https:', 'http:'].includes(url.protocol) || url.protocol === 'http:' && !loopback) throw new Error('Cluster endpoints require HTTPS, except explicit loopback HTTP; omit credentials, paths, queries and fragments.')
  return { kind: 'endpoint', url: url.origin }
}
export function validateSaveClusterHost(value: unknown): SaveClusterHost {
  const raw = clusterObject(value, ['id', 'expectedRevision', 'label', 'role', 'connection', 'token'])
  return { ...(raw.id === undefined ? {} : { id: clusterId(raw.id) }), expectedRevision: clusterNumber(raw.expectedRevision), label: clusterText(raw.label, 120), role: choice(raw.role, ['workstation', 'headless']), connection: validateClusterConnection(raw.connection), ...(raw.token === undefined ? {} : { token: validateClusterToken(raw.token) }) }
}
export function validateClusterPlanInput(value: unknown): ClusterPlanInput {
  const raw = clusterObject(value, ['name', 'model', 'profileId', 'generation', 'members', 'transport'])
  const members: ClusterMember[] = clusterList(raw.members, 8).map(value => {
    const row = clusterObject(value, ['hostId', 'hostRevision', 'nodeId', 'rank', 'link'])
    return { hostId: clusterId(row.hostId), hostRevision: clusterNumber(row.hostRevision), nodeId: clusterId(row.nodeId), rank: clusterNumber(row.rank, 0, 7), ...(row.link === undefined ? {} : { link: validateClusterLink(row.link) }) }
  })
  if (!members.length || new Set(members.map(row => row.hostId)).size !== members.length || new Set(members.map(row => row.nodeId)).size !== members.length || new Set(members.map(row => row.rank)).size !== members.length || members.some(row => row.rank >= members.length)) throw new Error('Choose distinct nodes with consecutive ranks starting at zero.')
  return { name: clusterText(raw.name, 120), model: validateClusterModel(raw.model), profileId: clusterId(raw.profileId), generation: validateClusterGeneration(raw.generation), members, transport: choice(raw.transport, ['tcp', 'thunderbolt-rdma', 'mcdma-rdma']) }
}
export function validateClusterNodeLaunch(value: unknown): ClusterNodeLaunch {
  const raw = clusterObject(value, ['deploymentId', 'commandId', 'expectedNodeId', 'model', 'profileId', 'role', 'generation', 'transport', 'rank', 'worldSize', 'links'])
  const rank = clusterNumber(raw.rank, 0, 7), worldSize = clusterNumber(raw.worldSize, 1, 8)
  if (rank >= worldSize) throw new Error('Rank must be within the cluster.')
  const links = clusterList(raw.links, 7).map(validateClusterLink)
  if (links.some(link => link.peer === rank || link.peer >= worldSize) || new Set(links.map(link => link.peer)).size !== links.length) throw new Error('Invalid peer link geometry.')
  return { deploymentId: clusterId(raw.deploymentId), commandId: validateClusterCommandId(raw.commandId), expectedNodeId: clusterId(raw.expectedNodeId), model: validateClusterModel(raw.model), profileId: clusterId(raw.profileId), role: choice(raw.role, ['workstation', 'headless']), generation: validateClusterGeneration(raw.generation), transport: choice(raw.transport, ['tcp', 'thunderbolt-rdma', 'mcdma-rdma']), rank, worldSize, links }
}
export function validateClusterProfile(value: unknown): ClusterProfile {
  const raw = clusterObject(value, ['id', 'title', 'mode', 'platform', 'arch', 'ranks', 'transports', 'modelTypes', 'poolsWeights', 'requiresFabric', 'engineVersion', 'formats', 'settingsSchema', 'admission', 'checkpoint', 'linkDiagnostics', 'health'], 'engine execution profile')
  const result: ClusterProfile = { id: clusterId(raw.id), title: clusterText(raw.title, 120), mode: choice(raw.mode, ['single', 'replicated-speed-up', 'tensor', 'pipeline']), platform: choice(raw.platform, ['darwin', 'linux']), arch: choice(raw.arch, ['arm64', 'x64']), ranks: clusterNumber(raw.ranks, 1, 8), transports: clusterList(raw.transports, 3).map(value => choice(value, ['tcp', 'thunderbolt-rdma', 'mcdma-rdma'])), modelTypes: clusterList(raw.modelTypes, 128).map(clusterId), poolsWeights: boolean(raw.poolsWeights), requiresFabric: boolean(raw.requiresFabric), engineVersion: clusterText(raw.engineVersion, 48) }
  if (!result.transports.length || !result.modelTypes.length || result.mode === 'single' && (result.ranks !== 1 || result.poolsWeights) || result.mode === 'replicated-speed-up' && result.poolsWeights) throw new Error('Inconsistent engine execution profile.')
  if (raw.formats !== undefined) result.formats = clusterList(raw.formats, 64).map(clusterId)
  if (result.mode !== 'single') {
    if (raw.settingsSchema !== 1 || raw.admission !== 'native-json-v1' || raw.checkpoint !== 'immutable-manifest-v1' || raw.linkDiagnostics !== 'native-json-v1' || raw.health !== 'native-json-v1' || !result.formats?.length) throw new Error('This engine does not report the complete distributed admission, checkpoint, link and health contract.')
    Object.assign(result, { settingsSchema: 1, admission: raw.admission, checkpoint: raw.checkpoint, linkDiagnostics: raw.linkDiagnostics, health: raw.health })
  }
  return result
}
export function validateDistributedProfiles(value: unknown): ClusterProfile[] {
  const raw = clusterObject(value, ['schema', 'profiles'], 'distributed capabilities')
  if (raw.schema !== 1) throw new Error('Unsupported distributed capability schema.')
  return clusterList(raw.profiles, 32).map(validateClusterProfile)
}
export function validateClusterNodeDeployment(value: unknown): ClusterNodeDeployment {
  const raw = clusterObject(value, ['id', 'commandId', 'state', 'model', 'rank', 'profileId', 'startedAt', 'error', 'endpointReady', 'canaryPassed'])
  return { id: clusterId(raw.id), commandId: validateClusterCommandId(raw.commandId), state: choice(raw.state, ['preparing', 'starting', 'ready', 'stopping', 'stopped', 'needs-review', 'failed']), model: validateClusterModel(raw.model), rank: clusterNumber(raw.rank, 0, 7), profileId: clusterId(raw.profileId), startedAt: raw.startedAt === null ? null : clusterNumber(raw.startedAt), error: raw.error === null ? null : clusterText(raw.error, 1000), endpointReady: boolean(raw.endpointReady), canaryPassed: boolean(raw.canaryPassed) }
}
export function validateClusterNodeJob(value: unknown): ClusterNodeJob {
  const raw = clusterObject(value, ['id', 'commandId', 'kind', 'phase', 'step', 'message', 'createdAt', 'updatedAt', 'model'])
  return { id: clusterId(raw.id), commandId: validateClusterCommandId(raw.commandId), kind: choice(raw.kind, ['ensure-model', 'start', 'stop', 'canary']), phase: choice(raw.phase, ['queued', 'running', 'reconciling', 'needs-review', 'completed', 'failed', 'cancelled']), step: clusterText(raw.step, 120), message: clusterText(raw.message, 1000, true), createdAt: clusterNumber(raw.createdAt), updatedAt: clusterNumber(raw.updatedAt), ...(raw.model === undefined ? {} : { model: validateClusterModel(raw.model) }) }
}
export function validateClusterNodeAdmission(value: unknown): ClusterNodeAdmission {
  const raw = clusterObject(value, ['allowed', 'nodeId', 'requiredBytes', 'budgetBytes', 'reason', 'cached', 'profile'])
  return { allowed: boolean(raw.allowed), nodeId: clusterId(raw.nodeId), requiredBytes: raw.requiredBytes === null ? null : clusterNumber(raw.requiredBytes), budgetBytes: clusterNumber(raw.budgetBytes), reason: clusterText(raw.reason, 1000, true), cached: boolean(raw.cached), profile: raw.profile === null ? null : validateClusterProfile(raw.profile) }
}
export function validateClusterNodeSnapshot(value: unknown): ClusterNodeSnapshot {
  const raw = clusterObject(value, ['protocol', 'nodeId', 'version', 'platform', 'arch', 'observedAt', 'totalMemoryBytes', 'freeMemoryBytes', 'engine', 'models', 'links', 'active', 'endpoint'])
  if (raw.protocol !== 1) throw new Error('This node uses an unsupported cluster protocol.')
  const engine = clusterObject(raw.engine, ['available', 'version', 'profiles', 'reason'])
  const result: ClusterNodeSnapshot = { protocol: 1, nodeId: clusterId(raw.nodeId), version: clusterText(raw.version, 48), platform: clusterText(raw.platform, 24), arch: clusterText(raw.arch, 24), observedAt: clusterNumber(raw.observedAt), totalMemoryBytes: clusterNumber(raw.totalMemoryBytes), freeMemoryBytes: clusterNumber(raw.freeMemoryBytes), engine: { available: boolean(engine.available), version: engine.version === null ? null : clusterText(engine.version, 48), profiles: clusterList(engine.profiles, 32).map(validateClusterProfile), reason: engine.reason === null ? null : clusterText(engine.reason, 1000) }, models: clusterList(raw.models, 1024).map(value => {
    const row = clusterObject(value, ['repo', 'revision', 'modelType', 'weightsBytes', 'memoryFloorBytes'])
    return { ...validateClusterModel({ repo: row.repo, revision: row.revision }), modelType: clusterId(row.modelType), weightsBytes: row.weightsBytes === null ? null : clusterNumber(row.weightsBytes), memoryFloorBytes: row.memoryFloorBytes === null ? null : clusterNumber(row.memoryFloorBytes) }
  }), links: clusterList(raw.links, 32).map(value => {
    const row = clusterObject(value, ['transport', 'available', 'verified', 'checkedAt', 'devices', 'message', 'proof'])
    const proof = row.proof === undefined ? null : clusterObject(row.proof, ['id', 'peerNodeId', 'bytes', 'matched', 'generation'])
    const link = { transport: choice(row.transport, ['tcp', 'thunderbolt-rdma', 'mcdma-rdma']), available: boolean(row.available), verified: boolean(row.verified), checkedAt: clusterNumber(row.checkedAt), devices: clusterList(row.devices, 32).map(clusterId), message: clusterText(row.message, 1000, true), ...(proof ? { proof: { id: clusterId(proof.id), peerNodeId: clusterId(proof.peerNodeId), bytes: clusterNumber(proof.bytes, 1), matched: boolean(proof.matched), generation: clusterId(proof.generation) } } : {}) }
    if (link.verified && (!link.available || !link.proof?.matched)) throw new Error('A verified data link needs a successful byte-check receipt.')
    return link
  }), active: raw.active === null ? null : validateClusterNodeDeployment(raw.active) }
  if (raw.endpoint !== undefined) { const endpoint = clusterObject(raw.endpoint, ['model', 'ready', 'loopback']); result.endpoint = { model: clusterText(endpoint.model, 192), ready: boolean(endpoint.ready), loopback: boolean(endpoint.loopback) } }
  return result
}

export function validateStoredClusterPlan(value: unknown): import('./cluster-workspace').ClusterPlan {
  const raw = clusterObject(value, ['id', 'name', 'model', 'profileId', 'generation', 'members', 'transport', 'digest', 'createdAt', 'expiresAt', 'blockers', 'warnings', 'poolsWeights', 'stageAllowed', 'admission'], 'saved plan')
  const input = validateClusterPlanInput({ name: raw.name, model: raw.model, profileId: raw.profileId, generation: raw.generation, members: raw.members, transport: raw.transport })
  const digest = clusterText(raw.digest, 64); if (!/^[a-f0-9]{64}$/.test(digest)) throw new Error('Invalid plan digest.')
  const admission = clusterList(raw.admission, 8).map(value => {
    const row = clusterObject(value, ['hostId', 'allowed', 'requiredBytes', 'budgetBytes', 'reason'])
    return { hostId: clusterId(row.hostId), allowed: boolean(row.allowed), requiredBytes: row.requiredBytes === null ? null : clusterNumber(row.requiredBytes), budgetBytes: clusterNumber(row.budgetBytes), reason: clusterText(row.reason, 1000, true) }
  })
  if (admission.some(row => !input.members.some(member => member.hostId === row.hostId))) throw new Error('Plan admission names an unselected machine.')
  return { ...input, id: clusterId(raw.id), digest, createdAt: clusterNumber(raw.createdAt), expiresAt: clusterNumber(raw.expiresAt), blockers: clusterList(raw.blockers, 64).map(value => clusterText(value, 1000)), warnings: clusterList(raw.warnings, 64).map(value => clusterText(value, 1000)), poolsWeights: boolean(raw.poolsWeights), stageAllowed: boolean(raw.stageAllowed), admission }
}
export function validateClusterJob(value: unknown): import('./cluster-workspace').ClusterJob {
  const raw = clusterObject(value, ['id', 'commandId', 'kind', 'phase', 'hostIds', 'planId', 'deploymentId', 'step', 'message', 'createdAt', 'updatedAt', 'cancelRequested'], 'saved operation')
  return { id: clusterId(raw.id), commandId: validateClusterCommandId(raw.commandId), kind: choice(raw.kind, ['inspect', 'ensure-model', 'activate', 'stop', 'bootstrap']), phase: choice(raw.phase, ['queued', 'running', 'reconciling', 'needs-review', 'completed', 'failed', 'cancelled']), hostIds: clusterList(raw.hostIds, 8).map(clusterId), ...(raw.planId === undefined ? {} : { planId: clusterId(raw.planId) }), ...(raw.deploymentId === undefined ? {} : { deploymentId: clusterId(raw.deploymentId) }), step: clusterText(raw.step, 120), message: clusterText(raw.message, 1000, true), createdAt: clusterNumber(raw.createdAt), updatedAt: clusterNumber(raw.updatedAt), cancelRequested: boolean(raw.cancelRequested) }
}
export function validateClusterDeployment(value: unknown): import('./cluster-workspace').ClusterDeployment {
  const raw = clusterObject(value, ['id', 'name', 'plan', 'state', 'endpoint', 'createdAt', 'updatedAt', 'error'])
  const state = choice(raw.state, ['preparing', 'starting', 'ready', 'stopping', 'stopped', 'needs-review', 'failed'])
  const endpoint = raw.endpoint === null ? null : clusterObject(raw.endpoint, ['baseUrl', 'model', 'hasKey', 'location', 'ready'])
  if (endpoint) {
    const url = new URL(clusterText(endpoint.baseUrl, 2048))
    validateClusterConnection({ kind: 'endpoint', url: url.origin })
    if (url.pathname !== '/v1' || url.search || url.hash || url.username || url.password || endpoint.location !== 'remote' || endpoint.ready && state !== 'ready') throw new Error('Invalid managed cluster endpoint.')
  }
  return { id: clusterId(raw.id), name: clusterText(raw.name, 120), plan: validateStoredClusterPlan(raw.plan), state, endpoint: endpoint ? { baseUrl: endpoint.baseUrl, model: clusterText(endpoint.model, 200), hasKey: boolean(endpoint.hasKey), location: 'remote', ready: boolean(endpoint.ready) } : null, createdAt: clusterNumber(raw.createdAt), updatedAt: clusterNumber(raw.updatedAt), error: raw.error === null ? null : clusterText(raw.error, 1000) }
}
