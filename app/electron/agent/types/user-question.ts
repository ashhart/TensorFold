export interface UserQuestionInput { question: string; choices?: string[] }
export interface UserQuestion {
  id: string
  threadId: string
  runId: string
  itemId: string
  question: string
  choices: string[]
  state: 'pending' | 'answered' | 'skipped' | 'interrupted'
  answer: string | null
  createdAt: number
  resolvedAt: number | null
}
export interface UserQuestionRun { threadId: string; runId: string; itemId: string; signal: AbortSignal }
export type UserQuestionOutcome = { state: 'answered'; answer: string } | { state: 'skipped'; answer: null }
export type UserQuestionAnswer = { questionId: string; threadId: string; runId: string } & ({ action: 'answer'; answer: string } | { action: 'skip' })
