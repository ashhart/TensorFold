import { flowTextFor } from '../../../lib/preferences/flow-catalogue'
import type { ComposerDraftScope } from '../../../../shared/composer-drafts'
import type { ConversationDrafts } from '../../../lib/conversation-drafts'
import type { Thread } from '../../../lib/types'
import { FIRST_ANSWER_STARTERS, type FirstAnswerStarterId } from './model'

/** The same coordinator used by Composer; a starter is editable text, never a run request. */
export async function openFirstAnswerDraft(input: {
  starter: FirstAnswerStarterId
  locale?: unknown
  drafts: Pick<ConversationDrafts, 'get' | 'update' | 'flush' | 'flushAll'>
  persistent: boolean
  threadIds(): string[]
  currentThread(): Thread | null
  newChat(): Promise<void>
}): Promise<{ threadId: string; seeded: boolean; preservedExistingDraft: boolean }> {
  const starter = FIRST_ANSWER_STARTERS.find((item) => item.id === input.starter)
  if (!starter) throw new Error('Choose a supported first task.')
  await input.drafts.flushAll()
  const known = new Set(input.threadIds())
  await input.newChat()
  const thread = input.currentThread()
  const created = input.threadIds().filter((id) => !known.has(id))
  if (!thread || created.length !== 1 || created[0] !== thread.id || thread.kind !== 'chat' || thread.timeline.length || thread.apiMessages.length
    || thread.orchestratorHostId || thread.incognito) throw new Error('The new local conversation changed; your existing drafts were kept.')
  if (!starter.prompt) return { threadId: thread.id, seeded: false, preservedExistingDraft: false }
  const scope: ComposerDraftScope = { surface: 'chat', conversationId: thread.id, hostId: null, incognito: false }
  input.drafts.get(scope)
  await input.drafts.flush(scope)
  if (input.currentThread()?.id !== thread.id) throw new Error('The selected conversation changed; the starter was not inserted.')
  const draft = input.drafts.get(scope)
  if (input.persistent && (!draft.ready || draft.error)) throw new Error('The new draft could not be loaded; your unfinished messages were kept.')
  if (draft.text || draft.attachments.length || draft.version > 0) return { threadId: thread.id, seeded: false, preservedExistingDraft: true }
  input.drafts.update(scope, { text: flowTextFor(input.locale).authored(starter.prompt) })
  await input.drafts.flush(scope)
  return { threadId: thread.id, seeded: true, preservedExistingDraft: false }
}
