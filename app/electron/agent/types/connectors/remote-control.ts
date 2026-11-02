import type { Job, Mission } from '../jobs'
import type { Thread } from '../timeline'
import type { SecretRotationMetadata } from '../base'
import type { Automation, Project, Room, Ticket } from '../workspace'
import { ConnectorEvent } from './core'

// ---- TensorFold Orchestrator host / remote client bridge ----
export type RemoteControlBindHost = '127.0.0.1' | '0.0.0.0'

export type RemoteControlEventKind = 'agent' | 'project' | 'thread' | 'room' | 'mission' | 'ticket' | 'job' | 'automation' | 'connector' | 'snapshot' | 'terminal' | 'stream_gap'

export interface RemoteControlSettings {
  enabled: boolean
  serverName: string
  bindHost: RemoteControlBindHost
  port: number
  authToken: string
  allowLan: boolean

  tunnelEnabled?: boolean
  tunnelProvider?: 'cloudflared' | 'ngrok'
  tunnelUrl?: string | null

  headlessBootstrappedAt?: number | null
  lastStartedAt: number | null
  lastClientAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface RemoteClientSettings {
  enabled: boolean
  serverUrl: string
  authToken: string
  lastConnectedAt: number | null
  lastError: string | null
  secretRotations?: SecretRotationMetadata
}

export interface RemoteOrchestratorProfile {
  id: string
  name: string
  serverUrl: string
  authToken: string
  enabled: boolean
  lastConnectedAt: number | null
  lastError: string | null
  createdAt: number
  updatedAt: number
  secretRotations?: SecretRotationMetadata
}

export interface RemoteControlStatus {
  enabled: boolean
  listening: boolean
  serverName: string
  bindHost: RemoteControlBindHost
  port: number | null
  url: string | null
  clients: number
  lastStartedAt: number | null
  lastClientAt: number | null
  lastError: string | null
}

export interface RemoteControlWorkSnapshot {
  jobs: {
    total: number
    visible: number
    queued: number
    running: number
    waitingApproval: number
    paused: number
    stalled: number
    failed: number
    deadLetter: number
    retryable: number
  }
  leases: {
    active: number
    expired: number
    expiringSoon: number
    missingRunning: number
  }
  approvals: {
    pending: number
    tool: number
    connector: number
    mission: number
  }
  nextWakeAt: number | null
  nextWakeJobId: string | null
  generatedAt: number

  terminals?: {
    total: number
    running: number
    attention: number
  }
}

export type RemoteControlCommandEffect = 'read' | 'write'

export interface RemoteControlCommandDescriptor {
  id: string
  title: string
  method: 'GET' | 'POST'
  path: string
  effect: RemoteControlCommandEffect
  description: string
}

export interface RemoteControlDoctorSnapshot {
  generatedAt: number
  serverName: string
  appVersion: string
  headless: boolean
  counts: {
    threads: number
    projects: number
    rooms: number
    tickets: number
    automations: number
    missions: number
    jobs: number
    connectorEvents: number
    activeThreads: number
    activeRooms: number
    activeMissions: number
  }
  work: RemoteControlWorkSnapshot
}

export interface RedactedProvider {
  id: string
  name: string
  kind: string
  transport?: string | null
  models: string[]
  defaultModel?: string | null
}

export interface RemoteControlSnapshot {
  generatedAt: number
  serverName: string
  appVersion: string
  headless: boolean
  threads: Thread[]
  projects: Project[]
  rooms: Room[]
  tickets: Ticket[]
  automations: Automation[]
  missions: Mission[]
  jobs: Job[]
  connectorEvents: ConnectorEvent[]
  work: RemoteControlWorkSnapshot
  activeThreadIds: string[]
  activeRoomIds: string[]
  activeMissionIds: string[]

  providers?: RedactedProvider[]
}

export interface RemoteControlEvent {
  id: string
  kind: RemoteControlEventKind
  createdAt: number
  data: unknown
}

export interface RemoteControlActionResult {
  ok: boolean
  error?: string
}

export interface RemoteConnectionTestResult extends RemoteControlActionResult {
  status?: RemoteControlStatus
  snapshot?: Pick<RemoteControlSnapshot, 'serverName' | 'generatedAt' | 'activeThreadIds' | 'activeRoomIds' | 'activeMissionIds' | 'work'>
}

export interface RemoteDirectoryEntry {
  name: string
  path: string
  isDirectory: boolean
}

export interface RemoteDirectoryListResult extends RemoteControlActionResult {
  path: string
  parent: string | null
  entries: RemoteDirectoryEntry[]
}

export interface RemoteControlPairingInfo {
  url: string | null
  token: string
  status: RemoteControlStatus
}

export interface RemoteOrchestratorSnapshotResult extends RemoteControlActionResult {
  hostId: string | null
  name: string
  serverUrl: string
  generatedAt?: number
  activeThreadIds?: string[]
  activeRoomIds?: string[]
  activeMissionIds?: string[]
  work?: RemoteControlWorkSnapshot
  totalProjects?: number
  totalThreads?: number
  totalRooms?: number
  totalMissions?: number
  projects?: Project[]
  threads?: Thread[]
  rooms?: Room[]
  tickets?: Ticket[]
  jobs?: Job[]
}
