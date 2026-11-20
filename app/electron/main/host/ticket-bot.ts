import { isDeepStrictEqual } from 'node:util'
import { captureBotConfig, botRunApprovalMode } from '../../../shared/bot-policy'
import { isRestrictedBot } from '../../../shared/bot-tool-policy'
import type { BotThreadConfig, Settings, Ticket } from '../../agent/types'
import { assertBotLifecycleIdle } from './bot-run-boundary'
import { engineService } from './engine-runtime'
import { isEngineProvider } from '../../agent/engine/provider'

export function captureTicketBot(ticket: Ticket, phase: 'build' | 'review', settings: Settings): BotThreadConfig | undefined {
  if (phase !== 'build' || (ticket.executionMode ?? 'solo') !== 'solo' || !ticket.botId) return undefined
  const preset = settings.agentPresets.find((candidate) => candidate.id === ticket.botId)
  if (!preset) throw new Error('The assigned bot is unavailable; choose another Bot or Agent on this ticket.')
  const bot = captureBotConfig(preset, settings.providers)
  if (bot.executionTarget !== 'workspace') throw new Error('Solo tickets require a bot that runs in This Mac workspace; VM ticket execution is unavailable.')
  return bot
}

export function assertTicketBotReady(bot: BotThreadConfig, settings: Settings): void {
  assertBotLifecycleIdle(bot.presetId)
  botRunApprovalMode({ bot }, settings.agentPresets, settings.providers, settings.approvalMode)
  if (isRestrictedBot(bot)) {
    const engine = engineService()
    const provider = settings.providers.find((candidate) => candidate.id === bot.providerId)
    if (engine?.hasExternalResearchEnabled() || isEngineProvider(provider) && !engine) {
      throw new Error('Turn off Cortheon in AI settings before using a restricted bot.')
    }
  }
}

export function sameTicketBot(previous: BotThreadConfig | undefined, next: BotThreadConfig | undefined): boolean {
  if (!previous || !next) return previous === next
  const { capturedAt: previousTime, ...previousConfig } = previous
  const { capturedAt: nextTime, ...nextConfig } = next
  return isDeepStrictEqual(previousConfig, nextConfig)
}
