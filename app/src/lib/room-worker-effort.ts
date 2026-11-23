import type { ReasoningEffortSetting, Ticket } from './types'

export function activeRoomWorkerEffort(tickets: readonly Ticket[], threadId: string): ReasoningEffortSetting | undefined {
  return tickets.find((ticket) => ticket.runState === 'running' && ticket.threadId === threadId
    && ticket.execution?.target === 'room' && ticket.execution.threadId === threadId
    && ticket.execution.finishedAt === null)?.execution?.reasoningEffort
}
