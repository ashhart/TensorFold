/** Serializable cluster contracts; no runtime imports from the main process. */
export type ClusterRole = 'workstation' | 'headless'
export type ClusterMode = 'single' | 'replicated-speed-up' | 'tensor' | 'pipeline'
export type ClusterTransport = 'tcp' | 'thunderbolt-rdma' | 'mcdma-rdma'
export interface ClusterModel { repo: string; revision: string }
export interface ClusterGeneration { contextTokens: number; maxOutputTokens: number; temperature: number }
export interface ClusterProfile {
  id: string; title: string; mode: ClusterMode; platform: 'darwin' | 'linux'; arch: 'arm64' | 'x64'
  ranks: number; transports: ClusterTransport[]; modelTypes: string[]
  poolsWeights: boolean; requiresFabric: boolean; engineVersion: string
  formats?: string[]; settingsSchema?: 1; admission?: 'native-json-v1'; checkpoint?: 'immutable-manifest-v1'
  linkDiagnostics?: 'native-json-v1'; health?: 'native-json-v1'
}
export interface ClusterLinkReport {
  transport: ClusterTransport; available: boolean; verified: boolean; checkedAt: number
  devices: string[]; message: string
  /** A successful byte check is separate from model admission and a model canary. */
  proof?: { id: string; peerNodeId: string; bytes: number; matched: boolean; generation: string }
}
export interface ClusterNodeSnapshot {
  protocol: 1; nodeId: string; version: string; platform: string; arch: string; observedAt: number
  totalMemoryBytes: number; freeMemoryBytes: number
  engine: { available: boolean; version: string | null; profiles: ClusterProfile[]; reason: string | null }
  models: Array<ClusterModel & { modelType: string; weightsBytes: number | null; memoryFloorBytes: number | null }>
  links: ClusterLinkReport[]
  active: ClusterNodeDeployment | null
  endpoint?: { model: string; ready: boolean; loopback: boolean }
}
export interface ClusterHost {
  id: string; revision: number; label: string; role: ClusterRole
  connection: { kind: 'endpoint'; url: string } | { kind: 'ssh'; sshHostId: string; port: number }
  nodeId: string | null; hasCredential: boolean; snapshot: ClusterNodeSnapshot | null
  state: 'unverified' | 'ready' | 'offline' | 'identity-changed'; lastError: string | null
  createdAt: number; updatedAt: number
}
export interface SaveClusterHost {
  id?: string; expectedRevision: number; label: string; role: ClusterRole
  connection: ClusterHost['connection']; token?: string
}
export interface ClusterDiscoveryCandidate { id: string; label: string; host: string; port: number; protocol: 1; expiresAt: number; trusted: false; via?: 'ssh' | 'https' }
export interface ClusterLinkSettings {
  peer: number; device: string; via: string; port: number; peerPort?: number; name: string; gid?: number
}
export interface ClusterMember { hostId: string; hostRevision: number; nodeId: string; rank: number; link?: ClusterLinkSettings }
export interface ClusterPlanInput {
  name: string; model: ClusterModel; profileId: string; generation: ClusterGeneration
  members: ClusterMember[]; transport: ClusterTransport
}
export interface ClusterPlan extends ClusterPlanInput {
  id: string; digest: string; createdAt: number; expiresAt: number
  blockers: string[]; warnings: string[]; poolsWeights: boolean; stageAllowed: boolean
  admission: Array<{ hostId: string; allowed: boolean; requiredBytes: number | null; budgetBytes: number; reason: string }>
}
export type ClusterJobPhase = 'queued' | 'running' | 'reconciling' | 'needs-review' | 'completed' | 'failed' | 'cancelled'
export interface ClusterJob {
  id: string; commandId: string; kind: 'inspect' | 'ensure-model' | 'activate' | 'stop' | 'bootstrap'
  phase: ClusterJobPhase; hostIds: string[]; planId?: string; deploymentId?: string
  step: string; message: string; createdAt: number; updatedAt: number; cancelRequested: boolean
}
export interface ClusterEndpoint {
  baseUrl: string; model: string; hasKey: boolean; location: 'remote'; ready: boolean
}
export interface ClusterDeployment {
  id: string; name: string; plan: ClusterPlan; state: 'preparing' | 'starting' | 'ready' | 'stopping' | 'stopped' | 'needs-review' | 'failed'
  endpoint: ClusterEndpoint | null; createdAt: number; updatedAt: number; error: string | null
}
export interface ClusterWorkspaceSnapshot { revision: number; hosts: ClusterHost[]; plans: ClusterPlan[]; jobs: ClusterJob[]; deployments: ClusterDeployment[] }
export interface ClusterNodeLaunch {
  deploymentId: string; commandId: string; expectedNodeId: string; model: ClusterModel
  profileId: string; role: ClusterRole; generation: ClusterGeneration; transport: ClusterTransport
  rank: number; worldSize: number; links: ClusterLinkSettings[]
}
export interface ClusterNodeDeployment {
  id: string; commandId: string; state: 'preparing' | 'starting' | 'ready' | 'stopping' | 'stopped' | 'needs-review' | 'failed'
  model: ClusterModel; rank: number; profileId: string; startedAt: number | null; error: string | null
  endpointReady: boolean; canaryPassed: boolean
}
export interface ClusterNodeJob {
  id: string; commandId: string; kind: 'ensure-model' | 'start' | 'stop' | 'canary'; phase: ClusterJobPhase
  step: string; message: string; createdAt: number; updatedAt: number; model?: ClusterModel
}
export interface ClusterNodeAdmission {
  allowed: boolean; nodeId: string; requiredBytes: number | null; budgetBytes: number; reason: string
  cached: boolean; profile: ClusterProfile | null
}
export interface ClusterSshKey { available: boolean; publicKey: string | null; fingerprint: string | null; message: string }
export interface ClusterBootstrapPreview {
  id: string; digest: string; sshHostId: string; platform: string; nodeAvailable: boolean; engineAvailable: boolean
  steps: string[]; blockers: string[]; expiresAt: number
}
export interface ClusterWorkspaceApi {
  clusterWorkspace(): Promise<ClusterWorkspaceSnapshot>
  clusterSaveHost(input: SaveClusterHost): Promise<ClusterWorkspaceSnapshot>
  clusterRemoveHost(input: { id: string; expectedRevision: number }): Promise<ClusterWorkspaceSnapshot>
  clusterDiscover(): Promise<ClusterDiscoveryCandidate[]>
  clusterInspectHost(input: { id: string; expectedRevision: number; commandId: string }): Promise<ClusterWorkspaceSnapshot>
  clusterPlan(input: ClusterPlanInput): Promise<ClusterPlan>
  clusterEnsureModel(input: { planId: string; digest: string; commandId: string }): Promise<ClusterWorkspaceSnapshot>
  clusterActivate(input: { planId: string; digest: string; commandId: string }): Promise<ClusterWorkspaceSnapshot>
  clusterStop(input: { deploymentId: string; commandId: string }): Promise<ClusterWorkspaceSnapshot>
  clusterCancelJob(input: { jobId: string }): Promise<ClusterWorkspaceSnapshot>
  clusterReconcile(input: { deploymentId: string } | { jobId: string }): Promise<ClusterWorkspaceSnapshot>
  clusterEndpointCredentials(input: { deploymentId: string }): Promise<{ baseUrl: string; apiKey: string; model: string }>
  clusterUseEndpoint(input: { deploymentId: string }): Promise<{ providerId: string }>
  clusterSshKey(input: { action: 'inspect' | 'create' }): Promise<ClusterSshKey>
  clusterBootstrapPreview(input: { sshHostId: string; port: number }): Promise<ClusterBootstrapPreview>
  clusterBootstrapApply(input: { previewId: string; digest: string; commandId: string }): Promise<ClusterWorkspaceSnapshot>
}
