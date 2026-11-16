import type { ReasoningEffortSetting, Thread } from '../electron/agent/types'

const efforts = new Set<ReasoningEffortSetting>([
  'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultrathink', 'ultracode', 'dynamic', '512',
])

export function normalizeBotReasoningEffort(value: unknown): ReasoningEffortSetting | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || !efforts.has(value as ReasoningEffortSetting)) {
    throw new Error('Invalid bot reasoning effort; choose an available effort level.')
  }
  return value as ReasoningEffortSetting
}

export function botRunReasoningEffort(thread: Pick<Thread, 'bot'>, fallback: ReasoningEffortSetting): ReasoningEffortSetting {
  return normalizeBotReasoningEffort(thread.bot?.reasoningEffort) ?? fallback
}
