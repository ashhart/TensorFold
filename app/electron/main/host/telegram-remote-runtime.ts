import type { TelegramVerificationResult } from '../../../shared/telegram-remote'
import { createTelegramRemoteService, type TelegramRemoteServiceDeps } from '../../agent/telegram-remote/service'
import { store } from '../state'
import { persistSettingsCritical } from './persistence'
import { createTelegramApiFetch, readTelegramJson } from './bot-polling-runtime'

export async function verifyTelegramToken(token: string, fetchImpl: typeof fetch = createTelegramApiFetch()): Promise<TelegramVerificationResult> {
  const secret = token.trim()
  if (!/^\d{4,20}:[A-Za-z0-9_-]{20,160}$/.test(secret)) return { ok: false, botId: null, username: null, error: 'Enter the bot token supplied by BotFather.' }
  try {
    const response = await fetchImpl(`https://api.telegram.org/bot${secret}/getMe`, { method: 'GET', redirect: 'error', signal: AbortSignal.timeout(12_000) })
    const body = await readTelegramJson(response, 64_000)
    if (!response.ok || body.ok !== true || body.result?.is_bot !== true || !Number.isSafeInteger(body.result?.id) || body.result.id <= 0 || !/^[A-Za-z0-9_]{5,64}$/.test(body.result?.username ?? '')) throw new Error('Telegram could not verify this bot token. Check the token and connection.')
    return { ok: true, botId: String(body.result.id), username: body.result.username, error: null }
  } catch {
    return { ok: false, botId: null, username: null, error: 'Telegram verification failed. Check the bot token, network, and any Telegram service error; no polling or messages were sent.' }
  }
}

export function createTelegramRemoteRuntime(deps: Pick<TelegramRemoteServiceDeps, 'dispatchTurn' | 'now' | 'timeoutMs'> & { telegramFetch?: typeof fetch }) {
  const service = createTelegramRemoteService({
    settings: () => store.settings.botControl.telegram,
    threads: () => store.threads,
    rooms: () => store.rooms,
    persist: () => persistSettingsCritical('telegram.remote', store.settings),
    dispatchTurn: deps.dispatchTurn, now: deps.now, timeoutMs: deps.timeoutMs,
  })
  const runtime = {
    ...service,
    listTelegramRemoteTargets: service.getStatus,
    async verifyTelegram(): Promise<TelegramVerificationResult> {
      const token = store.settings.botControl.telegram.botToken
      const result = await verifyTelegramToken(token, deps.telegramFetch)
      if (store.settings.botControl.telegram.botToken !== token) return { ok: false, botId: null, username: null, error: 'The Telegram token changed during verification. Check the current token again.' }
      return result
    },
  }
  currentRuntime = runtime
  return runtime
}

type TelegramRemoteRuntime = ReturnType<typeof createTelegramRemoteRuntime>
let currentRuntime: TelegramRemoteRuntime | null = null

export function getTelegramRemoteRuntime(): TelegramRemoteRuntime {
  if (!currentRuntime) throw new Error('Telegram remote assistant is not initialized.')
  return currentRuntime
}
