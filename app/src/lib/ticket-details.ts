import type { Ticket } from './types'

export type TicketDetails = Pick<Ticket, 'title' | 'body' | 'type' | 'priority' | 'effort'>
const fields = ['title', 'body', 'type', 'priority', 'effort'] as const

export function ticketDetails(ticket: Ticket): TicketDetails {
  return { title: ticket.title, body: ticket.body, type: ticket.type, priority: ticket.priority, effort: ticket.effort }
}

export function ticketDetailsPatch(original: Ticket, current: Ticket | undefined, draft: TicketDetails): Partial<TicketDetails> {
  if (!current || current.archivedAt) throw new Error('This ticket is no longer active. Close Details and refresh the board.')
  if (current.id !== original.id || current.projectId !== original.projectId || (current.orchestratorHostId ?? null) !== (original.orchestratorHostId ?? null)) throw new Error('Ticket ownership changed. Close Details and refresh the board.')
  if (current.runState === 'queued' || current.runState === 'running') throw new Error('Stop the active ticket run before editing its details.')
  if (fields.some((field) => current[field] !== original[field])) throw new Error('Ticket details changed in another window. Close and reopen Details before saving.')
  const next = { ...draft, title: draft.title.trim() }
  if (!next.title || next.title.length > 500) throw new Error('Ticket title must contain 1 to 500 characters.')
  if (next.body.length > 64_000) throw new Error('Ticket description must contain at most 64000 characters.')
  const patch: Partial<TicketDetails> = {}
  for (const field of fields) if (next[field] !== original[field]) Object.assign(patch, { [field]: next[field] })
  return patch
}
