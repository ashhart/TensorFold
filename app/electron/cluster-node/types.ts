import type { NodePublisher } from './advertise'
import type { Server } from 'node:http'
import type { ClusterGeneration, ClusterLinkReport, ClusterLinkSettings, ClusterModel, ClusterNodeAdmission, ClusterNodeDeployment, ClusterNodeJob, ClusterNodeLaunch, ClusterNodeSnapshot, ClusterProfile, ClusterRole, ClusterTransport } from '../../shared/cluster-workspace'

export interface ClusterNodeConfig {
  schema: 1
  dataDir: string
  enginePath?: string
  cacheDir?: string
  host: string
  port: number
  tls?: { certFile: string; keyFile: string }
  fabricLibraryPath?: string
  advertise?: boolean
}
export interface NodeAdmissionInput {
  expectedNodeId: string
  role: ClusterRole
  model: ClusterModel
  profileId: string
  generation: ClusterGeneration
  transport: ClusterTransport
  rank: number
  worldSize: number
  links: ClusterLinkSettings[]
}
export interface NodeModelInfo extends ClusterModel { modelType: string; weightsBytes: number | null; memoryFloorBytes: number | null }
export interface EngineInspection {
  available: boolean
  version: string | null
  profiles: ClusterProfile[]
  models: NodeModelInfo[]
  links: ClusterLinkReport[]
  reason: string | null
}
export interface EngineStream { status: number; contentType: string; body: AsyncIterable<Uint8Array> }
export interface EngineSession {
  readonly model: string
  alive(): boolean
  request(route: '/v1/chat/completions' | '/v1/responses', body: Record<string, unknown>, signal: AbortSignal): Promise<EngineStream>
  stop(): Promise<void>
}
export interface EngineAdapter {
  inspect(signal: AbortSignal): Promise<EngineInspection>
  admission(input: NodeAdmissionInput, signal: AbortSignal): Promise<Omit<ClusterNodeAdmission, 'nodeId'>>
  ensureModel(model: ClusterModel, signal: AbortSignal, progress: (message: string) => Promise<void>): Promise<void>
  start(input: ClusterNodeLaunch, signal: AbortSignal, progress: (message: string) => Promise<void>): Promise<EngineSession>
}
export interface NodeCommandReceipt { commandId: string; digest: string; jobId: string }
export interface StoredNodeState {
  schema: 1
  nodeId: string
  revision: number
  jobs: ClusterNodeJob[]
  receipts: NodeCommandReceipt[]
  cancelRequested: string[]
  active: ClusterNodeDeployment | null
  request: { id: string; phase: 'running' | 'completed' | 'failed' | 'cancelled' | 'needs-review'; digest: string; updatedAt: number } | null
}
export interface NodeJournal {
  read(): Promise<StoredNodeState>
  update<T>(edit: (state: StoredNodeState) => T): Promise<T>
  drain(): Promise<void>
}
export interface NodeServerOptions {
  config: ClusterNodeConfig
  engine?: EngineAdapter
  journal?: NodeJournal
  tokenFiles?: { management: string; inference: string }
  publisher?: NodePublisher
  diagnostic?: (message: string) => void
  now?: () => number
  maxBodyBytes?: number
  maxResponseBytes?: number
  requestTimeoutMs?: number
}
export interface ClusterNodeServer {
  server: Server
  listen(): Promise<{ host: string; port: number; url: string }>
  close(): Promise<void>
  address(): { host: string; port: number; url: string } | null
  snapshot(): Promise<ClusterNodeSnapshot>
}
