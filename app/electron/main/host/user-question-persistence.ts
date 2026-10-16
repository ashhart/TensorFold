import type { Thread, TimelineItem, UserQuestion, UserQuestionRun } from '../../agent/types'

type Run = Pick<UserQuestionRun, 'threadId' | 'runId' | 'itemId'>
export function createUserQuestionPersistence(options: {
  getThread(id: string): Thread | null
  save(run: Run, patch: Pick<TimelineItem, 'question' | 'state' | 'output'>): Promise<void>
  emit(run: Run, patch: Partial<TimelineItem>): void
}) {
  let tail = Promise.resolve()
  const commit = (run: Run, question: UserQuestion): Promise<void> => {
    const next = tail.then(async () => {
      const thread = options.getThread(run.threadId)
      const item = thread?.timeline.find((entry) => entry.id === run.itemId && entry.tool === 'ask_user')
      if (!thread || !item || question.threadId !== run.threadId || question.runId !== run.runId || question.itemId !== run.itemId) throw new Error('This question no longer belongs to the saved chat turn.')
      const snapshot: UserQuestion = { ...question, choices: [...question.choices] }
      const state = snapshot.state === 'pending' ? 'awaiting' : snapshot.state === 'interrupted' ? 'error' : 'ok'
      const output = snapshot.state === 'pending' ? 'Waiting for your answer.' : snapshot.state === 'interrupted' ? 'The run ended before this question was answered.' : snapshot.state === 'skipped' ? 'Question skipped.' : 'Your answer was saved.'
      await options.save(run, { question: snapshot, state, output })
      // A disposed renderer must not turn a durable answer into an unresumable run.
      try { options.emit(run, { question: snapshot, state, output }) } catch { /* reconnect reads the saved snapshot */ }
    })
    tail = next.catch(() => {})
    return next
  }
  return { commit, drain: () => tail }
}
