import type { UserQuestionAnswer, UserQuestionInput } from '../types/user-question'
function text(value: unknown, max: number, label: string): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max || /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(value)) throw new Error(`${label} must contain 1–${max} plain-text characters.`)
  return value.trim()
}
export function normalizeUserQuestion(value: unknown): Required<UserQuestionInput> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Ask one concise question.')
  const raw = value as Record<string, unknown>, question = text(raw.question, 400, 'Question')
  if (raw.choices != null && (!Array.isArray(raw.choices) || raw.choices.length > 3)) throw new Error('Offer at most three short choices; a typed answer is always available.')
  const choices = ((raw.choices as unknown[] | undefined) ?? []).map((choice) => text(choice, 60, 'Choice'))
  if (new Set(choices.map((choice) => choice.toLocaleLowerCase())).size !== choices.length) throw new Error('Question choices must be distinct.')
  return { question, choices }
}
export function normalizeQuestionAnswer(value: unknown): UserQuestionAnswer {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Answer or skip this question explicitly.')
  const raw = value as Record<string, unknown>
  const ids = { questionId: text(raw.questionId, 120, 'Question identity'), threadId: text(raw.threadId, 120, 'Chat identity'), runId: text(raw.runId, 120, 'Run identity') }
  if (raw.action === 'skip') return { ...ids, action: 'skip' }
  if (raw.action !== 'answer') throw new Error('Answer or skip this question explicitly.')
  return { ...ids, action: 'answer', answer: text(raw.answer, 2000, 'Answer') }
}
