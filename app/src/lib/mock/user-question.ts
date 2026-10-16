import type { UserQuestion, UserQuestionAnswer } from '../types'
import { mockState } from './state'
export function createMockUserQuestionApi() {
  const find = (threadId: string, questionId: string) => mockState.threads.find((thread) => thread.id === threadId)?.timeline.find((item) => item.question?.id === questionId)?.question ?? null
  return {
    userQuestionState: async (input: { threadId: string; questionId: string }): Promise<UserQuestion | null> => find(input.threadId, input.questionId),
    userQuestionAnswer: async (input: UserQuestionAnswer): Promise<UserQuestion> => {
      const current = find(input.threadId, input.questionId)
      if (!current || current.runId !== input.runId || current.state !== 'pending') throw new Error('This demo question is no longer waiting.')
      const answer = input.action === 'answer' ? input.answer.trim() : null
      if (input.action === 'answer' && (!answer || answer.length > 2000)) throw new Error('Enter an answer or choose Skip.')
      const result: UserQuestion = { ...current, answer, state: input.action === 'skip' ? 'skipped' : 'answered', resolvedAt: Date.now() }
      Object.assign(current, result); return result
    },
  }
}
