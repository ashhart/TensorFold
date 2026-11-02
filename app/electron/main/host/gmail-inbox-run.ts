import type { Provider, StartRunPayload, Thread } from '../../agent/types'
import { parseGmailInboxKey } from '../../../shared/gmail-inbox'
import { assertGoogleContextTransport } from '../../../shared/google-workspace-privacy'

export async function bindGmailInboxRun(payload: StartRunPayload, deps: {
  authorize(): void
  selected(accountId: string, revision: number): Promise<void>
  thread(id: string): Thread | undefined
  provider(thread: Thread): Provider | undefined
  externalResearch(): boolean
  persist(): Promise<void>
}): Promise<void> {
  const target = parseGmailInboxKey(payload.sourceThreadKey)
  if (!target) return
  deps.authorize()
  await deps.selected(target.accountId, target.revision)
  const thread = deps.thread(payload.threadId)
  if (!thread || thread.archivedAt || thread.bot || thread.phoneLocalOnly || deps.externalResearch()) throw new Error('Gmail Inbox tasks require an ordinary local model conversation with external research off.')
  assertGoogleContextTransport({ ...thread, googleLocalOnly: true }, deps.provider(thread))
  thread.googleLocalOnly = true
  await deps.persist()
  if (deps.thread(thread.id) !== thread) throw new Error('The Gmail conversation changed before starting the task.')
  await deps.selected(target.accountId, target.revision)
  assertGoogleContextTransport(thread, deps.provider(thread))
}
