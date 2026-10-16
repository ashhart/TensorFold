import type { Project, Thread, Room, Ticket } from '../../types'
import { resolveExecutionTarget } from '../../execution-target'
import type { AppState } from '../types'

export function remoteHostIdForProject(state: AppState, project: Project | null | undefined): string | null {
  // Creation/execution default only. Existing records use immutable origin helpers below.
  return resolveExecutionTarget(state.settings, project).hostId
}

export function projectRecordHostId(project: Project | null | undefined): string | null {
  return project?.executionTarget === 'orchestrator' && project.orchestratorId ? project.orchestratorId : null
}

export function remoteHostIdForProjectId(state: AppState, projectId: string): string | null {
  const project = state.projects.find((candidate) => candidate.id === projectId) ?? null
  return remoteHostIdForProject(state, project)
}

export function remoteHostIdForThread(state: AppState, threadId: string): string | null {
  const thread = state.threads.find((candidate) => candidate.id === threadId)
  return thread?.orchestratorHostId ?? null
}

export function remoteHostIdForRoom(state: AppState, roomId: string): string | null {
  const room = state.rooms.find((candidate) => candidate.id === roomId)
  return room?.orchestratorHostId ?? null
}

export function remoteHostIdForTicket(state: AppState, ticket: Pick<Ticket, 'projectId' | 'orchestratorHostId'> | null | undefined): string | null {
  return ticket?.orchestratorHostId ?? null
}

export function remoteHostIdForNewTicket(state: AppState, ticket: Pick<Ticket, 'projectId'>): string | null {
  const project = state.projects.find((candidate) => candidate.id === ticket.projectId) ?? null
  return remoteHostIdForProject(state, project)
}

// Once a ticket has an execution record, that record is its authority. Project routing can
// be edited while a run is in flight; lifecycle/cancel/update writes must still settle on
// the host that owns the ticket and thread rather than leaking into the local store or a
// newly-selected Orchestrator.
export function owningRemoteHostIdForTicket(state: AppState, ticket: Ticket | null | undefined): string | null {
  // The record origin is stamped by this controller's transport. A persisted execution.hostId
  // came from whichever controller began the run and is not portable across controllers.
  return remoteHostIdForTicket(state, ticket) ?? (ticket?.execution?.target === 'remote' ? ticket.execution.hostId ?? null : null)
}
