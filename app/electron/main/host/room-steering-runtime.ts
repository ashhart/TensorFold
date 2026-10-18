import type { Room, RoomEventItem, RoomStreamEvent } from '../../agent/types'
import { appendRoomSteeringEvent } from './room-ticket-patch-policy'

export interface RoomSteeringDeps {
  rooms(): readonly Room[]
  requireOperator(): void
  createId(): string
  persistRooms(): void
  emitRoom(event: RoomStreamEvent): void
  hasActiveRoom?(id: string): boolean
  now?(): number
}

/** Shared by native IPC and authenticated remote steering, never starts a room. */
export function steerExistingRoom(deps: RoomSteeringDeps, roomId: string, text: string): RoomEventItem {
  deps.requireOperator()
  const room = deps.rooms().find((candidate) => candidate.id === roomId)
  if (!room) throw new Error('Room not found')
  if (room.archivedAt) throw new Error('Restore this archived room before steering it.')
  if (deps.hasActiveRoom && !deps.hasActiveRoom(roomId)) throw new Error('Start the room before sending steering guidance.')
  const item = appendRoomSteeringEvent(room, text, deps.createId, deps.now?.() ?? Date.now())
  room.updatedAt = deps.now?.() ?? Date.now()
  deps.persistRooms()
  deps.emitRoom({ roomId, type: 'event', item })
  return item
}
