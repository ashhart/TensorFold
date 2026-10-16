import type { Ticket } from '../electron/agent/types'

export type TicketDependencyState = 'ready' | 'waiting' | 'blocked'
export type TicketDependencyFailurePolicy = 'block' | 'continue'

export interface TicketDependencyResolution {
  state: TicketDependencyState
  blockers: string[]
  waitingOn: string[]
}

export function normalizeTicketDependencyIds(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const ids: string[] = []
  for (const item of value) {
    if (typeof item !== 'string') continue
    const id = item.trim()
    if (!id || ids.includes(id)) continue
    ids.push(id)
    if (ids.length >= 64) break
  }
  return ids
}

export function resolveTicketDependencies(
  ticket: Pick<Ticket, 'id' | 'dependsOn' | 'dependencyFailurePolicy'>,
  tickets: readonly Ticket[],
): TicketDependencyResolution {
  const ids = normalizeTicketDependencyIds(ticket.dependsOn)
  if (!ids.length) return { state: 'ready', blockers: [], waitingOn: [] }

  const byId = new Map(tickets.map((candidate) => [candidate.id, candidate]))
  const blockers: string[] = []
  const waitingOn: string[] = []
  const continueOnFailure = ticket.dependencyFailurePolicy === 'continue'

  for (const id of ids) {
    const dependency = byId.get(id)
    if (!dependency || dependency.archivedAt) {
      blockers.push(`Missing dependency ${id}.`)
      continue
    }
    if (dependency.status === 'completed') continue
    if (dependency.status === 'needs_clarification') {
      if (!continueOnFailure) blockers.push(`Dependency "${dependency.title}" needs clarification.`)
      continue
    }
    waitingOn.push(dependency.title || dependency.id)
  }

  if (blockers.length) return { state: 'blocked', blockers, waitingOn }
  if (waitingOn.length) return { state: 'waiting', blockers: [], waitingOn }
  return { state: 'ready', blockers: [], waitingOn: [] }
}

export function assertTicketDependencyGraph(
  tickets: readonly Ticket[],
  candidate: Pick<Ticket, 'id' | 'projectId' | 'dependsOn'>,
): void {
  const ids = normalizeTicketDependencyIds(candidate.dependsOn)
  if (ids.length !== (candidate.dependsOn?.length ?? 0)) {
    throw new Error('Ticket dependencies must contain unique non-empty ticket ids.')
  }
  if (ids.includes(candidate.id)) throw new Error('A ticket cannot depend on itself.')

  const byId = new Map(tickets.map((ticket) => [ticket.id, ticket]))
  for (const id of ids) {
    const dependency = byId.get(id)
    if (!dependency) throw new Error(`Ticket dependency ${id} was not found.`)
    if (dependency.archivedAt) throw new Error(`Restore dependency "${dependency.title}" before linking it.`)
    if (dependency.projectId !== candidate.projectId) throw new Error('Ticket dependencies must belong to the same project.')
  }

  const visiting = new Set<string>()
  const visited = new Set<string>()
  const dependenciesFor = (id: string): string[] => {
    if (id === candidate.id) return ids
    return normalizeTicketDependencyIds(byId.get(id)?.dependsOn)
  }
  const visit = (id: string): void => {
    if (visiting.has(id)) throw new Error('Ticket dependencies cannot contain a cycle.')
    if (visited.has(id)) return
    visiting.add(id)
    for (const dependencyId of dependenciesFor(id)) visit(dependencyId)
    visiting.delete(id)
    visited.add(id)
  }
  visit(candidate.id)
}

export function requireTicketDependenciesReady(ticket: Ticket, tickets: readonly Ticket[]): void {
  const resolution = resolveTicketDependencies(ticket, tickets)
  if (resolution.state === 'ready') return
  if (resolution.state === 'waiting') {
    throw new Error(`Ticket is waiting on: ${resolution.waitingOn.join(', ')}.`)
  }
  throw new Error(`Ticket dependencies are blocked: ${resolution.blockers.join(' ')}`)
}
