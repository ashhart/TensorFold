import { api } from '../../ipc'
import { enqueueRoomMutation, enqueueWorkspaceMutation } from '../shared/queue'
import { remoteHostIdForRoom, owningRemoteHostIdForTicket, remoteHostIdForProject } from '../shared/remote-host'
import { defaultRoomSeats, roomProblemFromTicket, runErrorMessage } from '../shared/model-policy'
import { mergeRoomSnapshot } from '../shared/merge'
import { projectedHostId, projectionOriginsMatch, quarantineProjectionCollision } from '../shared/proxy-state'
import type { Slice } from '../types'
import type { Attachment, Room, RoomManualMemoryCommand, TicketStatus } from '../../types'

export const roomsSlice: Slice = (set, get) => ({
  rooms: [] as Room[],
  roomStatus: {} as Record<string, Room['state']>,
  roomLive: {} as Record<string, { seat: string; text: string } | undefined>,

  async createRoom(projectId, name) {
    const hostId = remoteHostIdForProject(get(), get().projects.find((candidate) => candidate.id === projectId) ?? null)
    const r = hostId
      ? await api.createRemoteOrchestratorRoom(hostId, projectId, name || 'New room')
      : await api.createRoom(projectId, name || 'New room')
    set((st) => ({ rooms: [r, ...st.rooms], view: { name: 'room', id: r.id } }))
    return r
  },
  async deleteRoom(id) {
    await get().archiveItem('room', id)
  },
  async updateRoom(id, patch) {
    const baseBudget = get().rooms.find((room) => room.id === id)?.workerBudget
    const budgetDelta: Partial<NonNullable<Room['workerBudget']>> = {}
    if (patch.workerBudget) {
      for (const field of ['maxActive', 'maxStarts'] as const) {
        if (patch.workerBudget[field] !== baseBudget?.[field]) budgetDelta[field] = patch.workerBudget[field]
      }
    }
    await enqueueRoomMutation(id, async () => {
      const hostId = remoteHostIdForRoom(get(), id)
      const current = get().rooms.find((room) => room.id === id)
      const effectivePatch: Partial<Room> = patch.workerBudget
        ? { ...patch, workerBudget: { ...(current?.workerBudget ?? patch.workerBudget), ...budgetDelta } }
        : patch
      try {
        const updated = hostId ? await api.updateRemoteOrchestratorRoom(hostId, id, effectivePatch) : await api.updateRoom(id, effectivePatch)
        set((st) => ({ rooms: st.rooms.map((r) => (r.id === id ? mergeRoomSnapshot(r, updated, effectivePatch) : r)) }))
      } catch (error) {
        set({ notice: { kind: 'error', message: runErrorMessage(error) } })
        throw error
      }
    })
  },
  applyRoomUpdate(r) {
    const existingRoom = get().rooms.find((room) => room.id === r.id)
    if (existingRoom && !projectionOriginsMatch(projectedHostId(existingRoom), projectedHostId(r))) {
      const message = quarantineProjectionCollision('room', r.id, projectedHostId(existingRoom), projectedHostId(r))
      if (message) set({ notice: { kind: 'error', message } })
      return
    }
    if (r.archivedAt) {
      set((st) => {
        const roomStatus = { ...st.roomStatus }
        const roomLive = { ...st.roomLive }
        delete roomStatus[r.id]
        delete roomLive[r.id]
        const projectId = st.rooms.find((room) => room.id === r.id)?.projectId ?? r.projectId
        return {
          rooms: st.rooms.filter((room) => room.id !== r.id),
          tickets: st.tickets.filter((ticket) => ticket.roomId !== r.id || !projectionOriginsMatch(projectedHostId(ticket), projectedHostId(r))),
          roomStatus,
          roomLive,
          view: st.view.name === 'room' && st.view.id === r.id ? { name: 'project', id: projectId } : st.view,
        }
      })
      return
    }
    set((st) => ({
      rooms: [r, ...st.rooms.filter((room) => room.id !== r.id)].sort((a, b) => b.updatedAt - a.updatedAt),
      roomStatus: { ...st.roomStatus, [r.id]: r.state },
    }))
  },
  async renameRoom(id, name) {
    const finalName = name.trim() || 'New room'
    await enqueueRoomMutation(id, async () => {
      const hostId = remoteHostIdForRoom(get(), id)
      if (hostId) {
        const updated = await api.renameRemoteOrchestratorRoom(hostId, id, finalName)
        set((st) => ({ rooms: st.rooms.map((r) => (r.id === id ? mergeRoomSnapshot(r, updated, { name: finalName }) : r)) }))
      } else {
        await api.renameRoom(id, finalName)
        set((st) => ({ rooms: st.rooms.map((room) => room.id === id ? { ...room, name: finalName } : room) }))
      }
    })
  },
  async setRoomSeats(roomId, seats) {
    await get().updateRoom(roomId, { seats })
  },
  async steerRoom(roomId, text) {
    const body = text.trim()
    if (!body) return
    await enqueueRoomMutation(roomId, async () => {
      try {
        const hostId = remoteHostIdForRoom(get(), roomId)
        const item = hostId ? await api.steerRemoteOrchestratorRoom(hostId, roomId, body) : await api.steerRoom(roomId, body)
        set((st) => ({ rooms: st.rooms.map((room) => room.id === roomId && !room.events.some((event) => event.id === item.id)
          ? { ...room, events: [...room.events, item], updatedAt: Math.max(room.updatedAt, item.createdAt) }
          : room) }))
      } catch (e: any) {
        set({ notice: { kind: 'error', message: String(e?.message ?? e) } })
      }
    })
  },
  async mutateRoomMemory(roomId, command: RoomManualMemoryCommand) {
    await enqueueRoomMutation(roomId, async () => {
      try {
        const hostId = remoteHostIdForRoom(get(), roomId)
        const updated = hostId
          ? await api.mutateRemoteOrchestratorRoomMemory(hostId, roomId, command)
          : await api.mutateRoomMemory(roomId, command)
        set((st) => ({ rooms: st.rooms.map((room) => room.id === roomId ? mergeRoomSnapshot(room, updated, {}) : room) }))
      } catch (error) {
        set({ notice: { kind: 'error', message: runErrorMessage(error) } })
        throw error
      }
    })
  },
  async startRoom(roomId, problem, attachments?: Attachment[]) {
    await enqueueRoomMutation(roomId, async () => {
      const hostId = remoteHostIdForRoom(get(), roomId)
      set((st) => ({
        rooms: st.rooms.map((r) => (r.id === roomId ? { ...r, problem, events: [], finalText: '', state: 'queued' } : r)),
        roomStatus: { ...st.roomStatus, [roomId]: 'queued' },
        roomLive: { ...st.roomLive, [roomId]: undefined },
      }))
      try {
        const result = hostId ? await api.startRemoteOrchestratorRoom(hostId, roomId, problem, attachments) : await api.startRoom(roomId, problem, attachments)
        if (result.room) set((st) => ({ rooms: st.rooms.map((r) => (r.id === roomId ? result.room! : r)), roomStatus: { ...st.roomStatus, [roomId]: result.room!.state } }))
      } catch (e: any) { get().applyRoomEvent({ roomId, type: 'error', message: String(e?.message ?? e), fatal: true }) }
    })
  },
  async startRoomFromTicket(ticketId, roomId) {
    // Multi-entity operations acquire their sorted lock set atomically. The broad room/ticket
    // gates make a room chosen from the post-queue ticket state safe without nested locks.
    await enqueueWorkspaceMutation(['rooms:*', 'tickets:*', `ticket:${ticketId}`], async () => {
      const ticket = get().tickets.find((candidate) => candidate.id === ticketId)
      if (!ticket) return
      const project = get().projects.find((candidate) => candidate.id === ticket.projectId) ?? null
      const ticketHostId = owningRemoteHostIdForTicket(get(), ticket)
      const requestedRoomId = roomId === undefined ? (ticket.roomId ?? null) : roomId

      let selectedRoom: Room
      let createdNewRoom = false
      if (requestedRoomId) {
        const existing = get().rooms.find((candidate) => candidate.id === requestedRoomId && candidate.projectId === ticket.projectId)
        if (!existing) {
          set({ notice: { kind: 'error', message: 'That room is not available for this project.' } })
          return
        }
        if (remoteHostIdForRoom(get(), existing.id) !== ticketHostId) {
          set({ notice: { kind: 'error', message: 'That room and ticket belong to different hosts. Choose a room on the ticket owner.' } })
          return
        }
        selectedRoom = existing
      } else {
        selectedRoom = ticketHostId
          ? await api.createRemoteOrchestratorRoom(ticketHostId, ticket.projectId, ticket.title ? `Room: ${ticket.title.slice(0, 54)}` : 'Ticket room')
          : await api.createRoom(ticket.projectId, ticket.title ? `Room: ${ticket.title.slice(0, 54)}` : 'Ticket room')
        createdNewRoom = true
      }

        const currentTicket = get().tickets.find((candidate) => candidate.id === ticketId)
        const currentRoom = get().rooms.find((candidate) => candidate.id === selectedRoom.id) ?? selectedRoom
        if (!currentTicket) return
        const problem = roomProblemFromTicket(currentTicket)
        const roomHostId = remoteHostIdForRoom(get(), currentRoom.id) ?? (createdNewRoom ? ticketHostId : null)
        if (roomHostId !== ticketHostId) {
          set({ notice: { kind: 'error', message: 'Room ownership changed before dispatch. Refresh the project and try again.' } })
          return
        }

        const seats = currentRoom.seats.length ? currentRoom.seats : defaultRoomSeats(get().settings, project)
        const configured = seats.length && !currentRoom.seats.length
          ? roomHostId
            ? await api.updateRemoteOrchestratorRoom(roomHostId, currentRoom.id, { seats })
            : await api.updateRoom(currentRoom.id, { seats })
          : currentRoom
        const nextStatus: TicketStatus = currentTicket.status === 'todo' ? 'ready' : currentTicket.status
        const assigned = ticketHostId
          ? await api.updateRemoteOrchestratorTicket(ticketHostId, currentTicket.id, { roomId: configured.id, executionMode: 'room', status: nextStatus })
          : await api.updateTicket(currentTicket.id, { roomId: configured.id, executionMode: 'room', status: nextStatus })

        set((st) => ({
          rooms: createdNewRoom
            ? [configured, ...st.rooms.filter((candidate) => candidate.id !== configured.id)]
            : st.rooms.map((candidate) => candidate.id === configured.id ? mergeRoomSnapshot(candidate, configured, { seats: configured.seats }) : candidate),
          tickets: st.tickets.map((candidate) => (candidate.id === assigned.id ? assigned : candidate)),
          view: { name: 'room', id: configured.id },
        }))

        const liveRoom = get().rooms.find((candidate) => candidate.id === configured.id) ?? configured
        const activeRoomState = get().roomStatus[configured.id] ?? liveRoom.state
        if (!createdNewRoom && (activeRoomState === 'queued' || activeRoomState === 'running')) {
          const body = `New Kanban ticket sent to this room:\n\n${problem}`
          try {
            const item = roomHostId ? await api.steerRemoteOrchestratorRoom(roomHostId, configured.id, body) : await api.steerRoom(configured.id, body)
            set((st) => ({ rooms: st.rooms.map((room) => room.id === configured.id && !room.events.some((event) => event.id === item.id)
              ? { ...room, events: [...room.events, item], updatedAt: Math.max(room.updatedAt, item.createdAt) }
              : room) }))
          } catch (error) {
            set({ notice: { kind: 'error', message: runErrorMessage(error) } })
          }
          return
        }

        set((st) => ({
          rooms: st.rooms.map((room) => room.id === configured.id ? { ...room, problem, events: [], finalText: '', state: 'queued' } : room),
          roomStatus: { ...st.roomStatus, [configured.id]: 'queued' },
          roomLive: { ...st.roomLive, [configured.id]: undefined },
        }))
        try {
          const result = roomHostId ? await api.startRemoteOrchestratorRoom(roomHostId, configured.id, problem) : await api.startRoom(configured.id, problem)
          if (result.room) set((st) => ({ rooms: st.rooms.map((room) => room.id === configured.id ? result.room! : room), roomStatus: { ...st.roomStatus, [configured.id]: result.room!.state } }))
        } catch (error) {
          get().applyRoomEvent({ roomId: configured.id, type: 'error', message: runErrorMessage(error), fatal: true })
        }
    })
  },
  async cancelRoom(roomId) {
    await enqueueRoomMutation(roomId, async () => {
      const hostId = remoteHostIdForRoom(get(), roomId)
      await (hostId ? api.cancelRemoteOrchestratorRoom(hostId, roomId) : api.cancelRoom(roomId))
      set((st) => ({ rooms: st.rooms.map((r) => (r.id === roomId ? { ...r, state: 'stopped' } : r)), roomStatus: { ...st.roomStatus, [roomId]: 'stopped' } }))
    })
  },
  applyRoomEvent(e) {
    const targetRoom = get().rooms.find((room) => room.id === e.roomId)
    if (!targetRoom) return
    if (!projectionOriginsMatch(projectedHostId(targetRoom), e.orchestratorHostId)) {
      const message = quarantineProjectionCollision('room event', e.roomId, projectedHostId(targetRoom), e.orchestratorHostId ?? null)
      if (message) set({ notice: { kind: 'error', message } })
      return
    }
    set((st) => {
      const rooms = [...st.rooms]
      const idx = rooms.findIndex((r) => r.id === e.roomId)
      const roomStatus = { ...st.roomStatus }
      const roomLive = { ...st.roomLive }
      if (e.type === 'status') { roomStatus[e.roomId] = e.state; if (idx >= 0) rooms[idx] = { ...rooms[idx], state: e.state } }
      else if (e.type === 'turn') roomLive[e.roomId] = { seat: e.seat, text: '' }
      else if (e.type === 'delta') { const cur = roomLive[e.roomId] || { seat: e.seat, text: '' }; roomLive[e.roomId] = { seat: e.seat, text: cur.text + e.chunk } }
      else if (e.type === 'event') {
        if (idx >= 0 && !rooms[idx].events.some((item) => item.id === e.item.id)) rooms[idx] = { ...rooms[idx], events: [...rooms[idx].events, e.item] }
        roomLive[e.roomId] = undefined
      }
      else if (e.type === 'memory') { if (idx >= 0) rooms[idx] = { ...rooms[idx], memory: e.memory, memorySummary: e.memorySummary, updatedAt: Date.now() } }
      else if (e.type === 'final') { if (idx >= 0) rooms[idx] = { ...rooms[idx], finalText: e.text } }
      else if (e.type === 'done') { roomLive[e.roomId] = undefined }
      else if (e.type === 'error') {
        const item = { id: 'err-' + Math.random().toString(36).slice(2), tick: idx >= 0 ? rooms[idx].events.length + 1 : 1, kind: 'error' as const, seat: 'system', target: 'all', text: e.message, createdAt: Date.now() }
        if (e.fatal) {
          roomStatus[e.roomId] = 'failed'
          if (idx >= 0) rooms[idx] = { ...rooms[idx], state: 'failed', events: [...rooms[idx].events, item] }
        } else if (idx >= 0) {
          rooms[idx] = { ...rooms[idx], events: [...rooms[idx].events, item] }
        }
      }
      return { rooms, roomStatus, roomLive }
    })
  },
})
