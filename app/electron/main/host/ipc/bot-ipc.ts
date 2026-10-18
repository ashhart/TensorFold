import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { registerBotRecipeIpc, type BotRecipeIpcDeps } from './bot-recipe-ipc'
import { getTelegramRemoteRuntime } from '../telegram-remote-runtime'
import type { TelegramRemoteConfigInput } from '../../../../shared/telegram-remote'

export interface BotIpcDeps {
  pollTelegramUpdates(): Promise<unknown>
  startBotControlPolling?(): void
  recipes?: BotRecipeIpcDeps
}

export function registerBotIpc(deps: BotIpcDeps) {
  const { pollTelegramUpdates } = deps
  if (deps.recipes) registerBotRecipeIpc(deps.recipes)

handle('bot:pollTelegram', async () => {
  requireSecurityRole('operator', 'poll Telegram bot control')
  return pollTelegramUpdates()
})

handle('bot:verifyTelegram', async () => {
  requireSecurityRole('owner', 'verify Telegram bot identity')
  return getTelegramRemoteRuntime().verifyTelegram()
})

handle('bot:configureTelegramRemote', async (_event, input: TelegramRemoteConfigInput) => {
  requireSecurityRole('owner', 'configure Telegram remote assistant access')
  const status = await getTelegramRemoteRuntime().configureRemote(input)
  deps.startBotControlPolling?.()
  return status
})

handle('bot:listTelegramRemoteTargets', async () => {
  requireSecurityRole('owner', 'list Telegram remote assistant targets')
  return getTelegramRemoteRuntime().listTelegramRemoteTargets()
})

}
