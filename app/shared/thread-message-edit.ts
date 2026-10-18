import type { ApiMessage, TimelineItem } from '../electron/agent/types'

export interface ThreadMessageEditInput {
  threadId: string
  itemId: string
  text: string
  expectedConversationRevision: number
  expectedHistoryToken: string
  expectedMessageText: string
}

export interface ThreadMessageEditPreparation {
  itemId: string
  expectedConversationRevision: number
  expectedHistoryToken: string
  expectedMessageText: string
  affectedItems: number
  mode: 'replace-chat' | 'branch-task'
}

/** Protected transcript evidence only; never supplied as active model history. */
export interface ThreadMessageEditRevision {
  id: string
  itemId: string
  editedAt: number
  conversationRevision: number
  timeline: TimelineItem[]
  apiMessages: ApiMessage[]
}
