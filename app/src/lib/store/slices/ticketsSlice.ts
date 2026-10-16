import { api } from '../../ipc'
import { buildRemoteTicketPrompt } from '../../ticket-remote'
import { enqueueTicketMutation } from '../shared/queue'
import { owningRemoteHostIdForTicket, remoteHostIdForNewTicket } from '../shared/remote-host'
import { runErrorMessage } from '../shared/model-policy'
import { projectedHostId, projectionOriginsMatch, quarantineProjectionCollision } from '../shared/proxy-state'
import type { Slice } from '../types'
import type { Job, Ticket, TicketStatus, TicketWorkOptions } from '../../types'

export const ticketsSlice: Slice = (set, get) => ({
  tickets: [] as Ticket[],
  jobs: [] as Job[],

  async createTicket(t) {
    const hostId = remoteHostIdForNewTicket(get(), t)
    const ticket = hostId ? await api.createRemoteOrchestratorTicket(hostId, t) : await api.createTicket(t)
    set((st) => ({ tickets: [ticket, ...st.tickets] }))
  },
  async updateTicket(id, patch) {
    await enqueueTicketMutation(id, async () => {
      const ticket = get().tickets.find((candidate) => candidate.id === id)
      const hostId = owningRemoteHostIdForTicket(get(), ticket)
      try {
        const updated = await (hostId ? api.updateRemoteOrchestratorTicket(hostId, id, patch) : api.updateTicket(id, patch))
        get().applyTicketUpdate(updated)
      } catch (error) {
        set({ notice: { kind: 'error', message: runErrorMessage(error) } })
        throw error
      }
    })
  },
  moveTicket(id, status: TicketStatus) { get().updateTicket(id, { status }) },
  async deleteTicket(id) {
    await get().archiveItem('ticket', id)
  },
  async generateTickets(projectId, idea, opts?: TicketWorkOptions) {
    const created = await api.generateTickets(projectId, idea, opts)
    set((st) => ({ tickets: [...created, ...st.tickets] }))
  },
  async workTicket(id, opts?: TicketWorkOptions) {
    await enqueueTicketMutation(id, async () => {
      const thread = await api.workTicket(id, opts)
      set((st) => ({ threads: [thread, ...st.threads.filter((x) => x.id !== thread.id)], view: { name: 'thread', id: thread.id } }))
    })
  },
  // Remote ticket execution (VST-018): run the ticket as a thread on the project's remote
  // Orchestrator and track its lifecycle on the ticket via remoteTicketLifecycle (which
  // applyThreadUpdate consults on every mirrored remote-thread update).
  async workTicketRemote(id) {
    await enqueueTicketMutation(id, async () => {
      const st = get()
      const ticket = st.tickets.find((candidate) => candidate.id === id)
      if (!ticket) return
      const remoteHostId = owningRemoteHostIdForTicket(st, ticket)
      if (!remoteHostId) {
        set({ notice: { kind: 'error', message: 'This ticket is locally owned. Remote execution cannot move an existing ticket implicitly; create it on an Orchestrator or pick another run mode.' } })
        return
      }
      try {
        const result = await api.startRemoteOrchestratorTicketExecution(remoteHostId, id, buildRemoteTicketPrompt(ticket, 'build'))
        get().applyTicketUpdate(result.ticket)
        set((state) => ({ threads: [result.thread, ...state.threads.filter((thread) => thread.id !== result.thread.id)] }))
      } catch (e) {
        set({ notice: { kind: 'error', message: runErrorMessage(e) } })
      }
    })
  },
  applyTicketUpdate(t) {
    const existingTicket = get().tickets.find((ticket) => ticket.id === t.id)
    if (existingTicket && !projectionOriginsMatch(projectedHostId(existingTicket), projectedHostId(t))) {
      const message = quarantineProjectionCollision('ticket', t.id, projectedHostId(existingTicket), projectedHostId(t))
      if (message) set({ notice: { kind: 'error', message } })
      return
    }
    if (t.archivedAt) {
      set((st) => ({
        // Archive is recoverable: retain descendant ownership so restoring the parent ticket
        // does not leave its subtickets permanently orphaned in the renderer projection.
        tickets: st.tickets.filter((ticket) => ticket.id !== t.id),
      }))
      return
    }
    set((st) => ({ tickets: [t, ...st.tickets.filter((x) => x.id !== t.id)].sort((a, b) => b.updatedAt - a.updatedAt) }))
  },
  applyJobUpdate(j) {
    if (j.archivedAt) {
      set((st) => ({ jobs: st.jobs.filter((job) => job.id !== j.id) }))
      return
    }
    set((st) => ({ jobs: [j, ...st.jobs.filter((x) => x.id !== j.id)] }))
  },
  async retryJob(id, reason, nextWakeAt, resumePlan) {
    const job = await api.retryJob(id, reason, nextWakeAt, resumePlan)
    get().applyJobUpdate(job)
  },
  async resumeJob(id, resumePlan) {
    const job = await api.resumeJob(id, resumePlan)
    get().applyJobUpdate(job)
  },
  async deadLetterJob(id, reason) {
    const job = await api.deadLetterJob(id, reason)
    get().applyJobUpdate(job)
  },
})
