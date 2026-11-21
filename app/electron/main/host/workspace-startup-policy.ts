import type { Project, Room } from '../../agent/types'
import type { WorkspaceMutationGuard, WorkspaceMutationLease } from './workspace-mutation-guard'

export function reserveRoomStartup(
  roomId: string,
  roomAborters: Map<string, AbortController>,
  guard: Pick<WorkspaceMutationGuard, 'reserveStart'>,
  workspaceKeys: Iterable<string | null | undefined>,
): { controller: AbortController; runLease: WorkspaceMutationLease } | null {
  if (roomAborters.has(roomId)) return null
  const runLease = guard.reserveStart({ workspaceKeys })
  const controller = new AbortController()
  roomAborters.set(roomId, controller)
  return { controller, runLease }
}

export function assertLocalTicketGenerationProject(project: Pick<Project, 'workspace'>): void {
  if (project.workspace?.kind === 'ssh') {
    throw new Error('Ticket generation for SSH projects is unavailable until the planner can run on the remote workspace.')
  }
}

export function requireActiveRoomForCancel(rooms: readonly Room[], roomId: string): Room {
  const room = rooms.find((candidate) => candidate.id === roomId)
  if (!room) throw new Error('Room not found')
  if (room.archivedAt) throw new Error('Restore this archived room before cancelling it.')
  return room
}

export function assertRecordMetadataNotPatched(patch: object, additionalHostFields: readonly string[] = []): void {
  for (const field of ['id', 'createdAt', 'updatedAt', ...additionalHostFields]) {
    if (Object.prototype.hasOwnProperty.call(patch, field)) {
      throw new Error(`${field} is host-owned and cannot be changed through update endpoints.`)
    }
  }
}
