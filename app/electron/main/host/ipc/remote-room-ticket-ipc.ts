import type { Room, RoomEventItem, RoomManualMemoryCommand, Thread, Ticket, TicketRemoteLifecycleCommand } from '../../../agent/types'
import { store } from '../../state'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import type { RemoteIpcContext } from './remote-ipc-context'

export function registerRemoteRoomTicketIpc(context: RemoteIpcContext): void {
  const { remoteHostForId, ensureRemoteEventStream, fetchRemoteOrchestratorJson, remoteOrchestratorConnectionError, withRemoteTicketHost } = context
  handle('remote:createRoom', async (_e, input: { hostId?: string | null; projectId: string; name?: string }): Promise<Room> => {
    requireSecurityRole('operator', 'create Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Room>(host, '/v1/rooms', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ projectId: input.projectId, name: input.name }),
        signal: AbortSignal.timeout(12000),
      })
      const room = { ...data, orchestratorHostId: host.id }
      store.win?.webContents.send('room:update', room)
      return room
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:updateRoom', async (_e, input: { hostId?: string | null; id: string; patch: Record<string, unknown> }): Promise<any> => {
    requireSecurityRole('operator', 'update Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Room>(host, `/v1/rooms/${encodeURIComponent(input.id)}/update`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(input.patch ?? {}),
        signal: AbortSignal.timeout(12000),
      })
      const room = { ...data, orchestratorHostId: host.id }
      store.win?.webContents.send('room:update', room)
      return room
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:renameRoom', async (_e, input: { hostId?: string | null; id: string; name: string }): Promise<any> => {
    requireSecurityRole('operator', 'rename Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Room>(host, `/v1/rooms/${encodeURIComponent(input.id)}/rename`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: input.name }),
        signal: AbortSignal.timeout(12000),
      })
      const room = { ...data, orchestratorHostId: host.id }
      store.win?.webContents.send('room:update', room)
      return room
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:steerRoom', async (_e, input: { hostId?: string | null; roomId: string; text: string }): Promise<any> => {
    requireSecurityRole('operator', 'steer Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<RoomEventItem>(host, `/v1/rooms/${encodeURIComponent(input.roomId)}/steer`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text: input.text }),
        signal: AbortSignal.timeout(12000),
      })
      return data
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:mutateRoomMemory', async (_e, input: { hostId?: string | null; roomId: string; command: RoomManualMemoryCommand }): Promise<any> => {
    requireSecurityRole('operator', 'change Orchestrator room memory')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Room>(host, `/v1/rooms/${encodeURIComponent(input.roomId)}/memory`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(input.command),
        signal: AbortSignal.timeout(12000),
      })
      const room = { ...data, orchestratorHostId: host.id }
      store.win?.webContents.send('room:update', room)
      return room
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:deleteRoom', async (_e, input: { hostId?: string | null; id: string }): Promise<any> => {
    requireSecurityRole('operator', 'archive Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Room | null>(host, `/v1/rooms/${encodeURIComponent(input.id)}/delete`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{}',
        signal: AbortSignal.timeout(12000),
      })
      const room = data ? { ...data, orchestratorHostId: host.id } : null
      if (room) store.win?.webContents.send('room:update', room)
      return room
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:restoreRoom', async (_e, input: { hostId?: string | null; id: string }): Promise<any> => {
    requireSecurityRole('operator', 'restore Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Room | null>(host, `/v1/rooms/${encodeURIComponent(input.id)}/restore`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(12000),
      })
      const room = data ? { ...data, orchestratorHostId: host.id } : null
      if (room) store.win?.webContents.send('room:update', room)
      return room
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:startRoom', async (_e, input: { hostId?: string | null; roomId: string; problem: string; attachments?: unknown[] }): Promise<{ ok: boolean; room?: unknown }> => {
    requireSecurityRole('operator', 'start Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<{ ok: boolean; room?: unknown }>(host, `/v1/rooms/${encodeURIComponent(input.roomId)}/start`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ problem: input.problem, attachments: input.attachments }),
        signal: AbortSignal.timeout(12000),
      })
      return data
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:cancelRoom', async (_e, input: { hostId?: string | null; roomId: string }): Promise<void> => {
    requireSecurityRole('operator', 'cancel Orchestrator room')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      await fetchRemoteOrchestratorJson(host, `/v1/rooms/${encodeURIComponent(input.roomId)}/cancel`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{}',
        signal: AbortSignal.timeout(12000),
      })
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:createTicket', async (_e, input: { hostId?: string | null; ticket: Record<string, unknown> & { projectId: string } }): Promise<any> => {
    requireSecurityRole('operator', 'create Orchestrator ticket')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Ticket>(host, '/v1/tickets', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(input.ticket ?? {}),
        signal: AbortSignal.timeout(12000),
      })
      const ticket = withRemoteTicketHost(host, data)
      store.win?.webContents.send('ticket:update', ticket)
      return ticket
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:startTicketExecution', async (_e, input: { hostId?: string | null; id: string; userText: string }): Promise<any> => {
    requireSecurityRole('operator', 'start Orchestrator ticket execution')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<{ ticket: Ticket; thread: Thread; runId: string }>(host, `/v1/tickets/${encodeURIComponent(input.id)}/start`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ hostId: host.id, userText: input.userText }),
        signal: AbortSignal.timeout(12000),
      })
      const ticket = withRemoteTicketHost(host, data.ticket)
      const thread = { ...data.thread, orchestratorHostId: host.id }
      store.win?.webContents.send('ticket:update', ticket)
      store.win?.webContents.send('thread:update', thread)
      return { ticket, thread, runId: data.runId }
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:updateTicket', async (_e, input: { hostId?: string | null; id: string; patch: Record<string, unknown> }): Promise<any> => {
    requireSecurityRole('operator', 'update Orchestrator ticket')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Ticket>(host, `/v1/tickets/${encodeURIComponent(input.id)}/update`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(input.patch ?? {}),
        signal: AbortSignal.timeout(12000),
      })
      const ticket = withRemoteTicketHost(host, data)
      store.win?.webContents.send('ticket:update', ticket)
      return ticket
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:transitionTicket', async (_e, input: { hostId?: string | null; id: string; command: TicketRemoteLifecycleCommand }): Promise<any> => {
    requireSecurityRole('operator', 'transition Orchestrator ticket lifecycle')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const command = input.command?.action === 'begin' ? { ...input.command, hostId: host.id } : input.command
      const { data } = await fetchRemoteOrchestratorJson<Ticket>(host, `/v1/tickets/${encodeURIComponent(input.id)}/lifecycle`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(command),
        signal: AbortSignal.timeout(12000),
      })
      const ticket = withRemoteTicketHost(host, data)
      store.win?.webContents.send('ticket:update', ticket)
      return ticket
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:deleteTicket', async (_e, input: { hostId?: string | null; id: string }): Promise<any> => {
    requireSecurityRole('operator', 'archive Orchestrator ticket')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Ticket | null>(host, `/v1/tickets/${encodeURIComponent(input.id)}/delete`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{}',
        signal: AbortSignal.timeout(12000),
      })
      const ticket = data ? withRemoteTicketHost(host, data) : null
      if (ticket) store.win?.webContents.send('ticket:update', ticket)
      return ticket
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })

  handle('remote:restoreTicket', async (_e, input: { hostId?: string | null; id: string }): Promise<any> => {
    requireSecurityRole('operator', 'restore Orchestrator ticket')
    const host = remoteHostForId(input?.hostId)
    if (!host) throw new Error('No Orchestrator selected.')
    try {
      ensureRemoteEventStream(host)
      const { data } = await fetchRemoteOrchestratorJson<Ticket | null>(host, `/v1/tickets/${encodeURIComponent(input.id)}/restore`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(12000),
      })
      const ticket = data ? withRemoteTicketHost(host, data) : null
      if (ticket) store.win?.webContents.send('ticket:update', ticket)
      return ticket
    } catch (e) {
      throw new Error(remoteOrchestratorConnectionError(e, host.serverUrl, host.serverUrl))
    }
  })
}
