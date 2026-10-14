import { captureBotConfig } from '../../shared/bot-policy'
import type { AgentPreset, BotThreadConfig, Provider, Thread } from './types'

const configKeys = ['presetId', 'name', 'role', 'instructions', 'providerId', 'providerAccountId', 'model', 'approvalScope', 'executionTarget'] as const satisfies readonly (keyof BotThreadConfig)[]

/** Open the newest conversation only when its captured behavior still matches the bot. */
export function mostRecentCompatibleBotChat(preset: AgentPreset, providers: Provider[], threads: Thread[]): Thread | null {
  let expected: BotThreadConfig
  try { expected = captureBotConfig({ ...preset, paused: false, archivedAt: null }, providers) }
  catch { return null }
  return threads.filter((thread) => {
    if (thread.archivedAt || thread.orchestratorHostId || thread.kind !== 'chat' || thread.projectId || thread.workspace || thread.engine && thread.engine !== 'native') return false
    if (!thread.bot || !configKeys.every((key) => thread.bot![key] === expected[key])) return false
    return thread.providerId === expected.providerId && (thread.providerAccountId ?? null) === expected.providerAccountId && thread.model === expected.model && thread.cwd === (expected.executionTarget === 'vm' ? '/workspace' : '')
  }).sort((a, b) => b.updatedAt - a.updatedAt)[0] ?? null
}
