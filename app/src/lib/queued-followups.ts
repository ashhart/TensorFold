import type { Steer } from '../components/composer/SteerQueueList'
import type { ConversationDraftSendToken } from './conversation-drafts'
import { MAX_DRAFT_TEXT_CHARS } from '../../shared/composer-drafts'

export interface QueuedFollowUpEdit { id: string; expectedText: string; text: string }
export type QueuedFollowUpEditResult = { ok: true; queued: Steer[] } | { ok: false; reason: 'missing' | 'started' | 'changed' | 'empty' | 'too-long'; queued: Steer[] }

/** Queue edits are replacements in place; they never dispatch, re-order or revive a row. */
export function editQueuedFollowUp(queued: Steer[], input: QueuedFollowUpEdit): QueuedFollowUpEditResult {
  const index = queued.findIndex(row => row.id === input.id)
  if (index < 0) return { ok: false, reason: 'missing', queued }
  const row = queued[index]
  if (row.status !== 'queued') return { ok: false, reason: 'started', queued }
  if (row.text !== input.expectedText) return { ok: false, reason: 'changed', queued }
  if (!input.text.trim()) return { ok: false, reason: 'empty', queued }
  if (input.text.length > MAX_DRAFT_TEXT_CHARS) return { ok: false, reason: 'too-long', queued }
  const next = queued.slice()
  next[index] = { ...row, text: input.text }
  return { ok: true, queued: next }
}

/** Only the unchanged saved owner draft may follow an edited queue row. */
export function capturedDraftMatches(original: ConversationDraftSendToken | undefined, current: ConversationDraftSendToken): boolean {
  return !!original && original.key === current.key && original.version === current.version && original.text === current.text
}
