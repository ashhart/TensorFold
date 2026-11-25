import type { ReasoningEffortSetting } from '../electron/agent/types'

const efforts = new Set<ReasoningEffortSetting>([
  'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultrathink', 'ultracode', 'dynamic', '512',
])

export function isReasoningEffortSetting(value: unknown): value is ReasoningEffortSetting {
  return typeof value === 'string' && efforts.has(value as ReasoningEffortSetting)
}

export function captureAgentReasoningEffort<T extends string>(
  agent: { reasoningEffort?: ReasoningEffortSetting } | undefined,
  fallback: T,
): ReasoningEffortSetting | T {
  const value = agent?.reasoningEffort
  if (value === undefined) return fallback
  if (!isReasoningEffortSetting(value)) throw new Error('Invalid agent reasoning effort; choose an available effort level.')
  return value
}
