import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'

export interface BotIpcDeps {
  pollTelegramUpdates(): Promise<unknown>
}

export function registerBotIpc(deps: BotIpcDeps) {
  const { pollTelegramUpdates } = deps

handle('bot:pollTelegram', async () => {
  requireSecurityRole('operator', 'poll Telegram bot control')
  return pollTelegramUpdates()
})

}
