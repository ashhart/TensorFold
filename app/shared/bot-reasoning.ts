import type { ReasoningEffortSetting, Thread } from '../electron/agent/types'
import { isReasoningEffortSetting } from './reasoning-effort'

export function normalizeBotReasoningEffort(value: unknown): ReasoningEffortSetting | undefined {
  if (value === undefined) return undefined
  if (!isReasoningEffortSetting(value)) {
    throw new Error('Invalid bot reasoning effort; choose an available effort level.')
  }
  return value as ReasoningEffortSetting
}

export function botRunReasoningEffort(thread: Pick<Thread, 'bot'>, fallback: ReasoningEffortSetting): ReasoningEffortSetting {
  return normalizeBotReasoningEffort(thread.bot?.reasoningEffort) ?? fallback
}
