import type { RemoteOrchestratorProfile, Project, Ticket } from '../../../agent/types'

export interface RemoteIpcContext {
  remoteHostForId(hostId?: string | null): RemoteOrchestratorProfile | null
  ensureRemoteEventStream(host: RemoteOrchestratorProfile): void
  fetchRemoteOrchestratorJson<T>(host: RemoteOrchestratorProfile, path: string, init?: RequestInit): Promise<{ base: string; data: T }>
  remoteOrchestratorConnectionError(error: unknown, base: string, rawBase: string): string
  withRemoteProjectHost(host: RemoteOrchestratorProfile, project: Project): Project
  withRemoteTicketHost(host: RemoteOrchestratorProfile, ticket: Ticket): Ticket
}
