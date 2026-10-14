import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { registerBotRecipeIpc, type BotRecipeIpcDeps } from './bot-recipe-ipc'

export interface BotIpcDeps {
  pollTelegramUpdates(): Promise<unknown>
  recipes?: BotRecipeIpcDeps
}

export function registerBotIpc(deps: BotIpcDeps) {
  const { pollTelegramUpdates } = deps
  if (deps.recipes) registerBotRecipeIpc(deps.recipes)

handle('bot:pollTelegram', async () => {
  requireSecurityRole('operator', 'poll Telegram bot control')
  return pollTelegramUpdates()
})

}
