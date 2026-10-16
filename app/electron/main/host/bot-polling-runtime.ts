import { parseAllowedBotChatIds, telegramUpdateToIngest } from '../../agent/bot-control'
import { parseApprovalReply } from '../../agent/channel-adapter'
import { createPinnedFetch, discardResponseBody, resolvePinnedHost, type PinnedHostResolver } from '../../agent/pinned-fetch'
import { isGloballyRoutableIp } from '../../agent/security'
import type { BotControlPollResult, ConnectorApprovalDecision } from '../../agent/types'
import { saveSettings } from '../../store'
import { store } from '../state'
import { persistSettingsCritical } from './persistence'
import { botChannelAllowed, ingestBotControl } from './bot-control'

export interface BotPollingRuntime {
  pollTelegramUpdates(): Promise<BotControlPollResult>
  startBotControlPolling(): void
  stopBotControlPolling(): void
}

interface BotPollingRuntimeDeps {
  botControlRuntime(): any
  // Resolve a parked in-channel approval from a Telegram button (token) or, failing that,
  // a plain "/approve" reply matched by conversation threadKey. Return true when one matched
  // so the update is consumed as an approval decision rather than a bot command.
  resolveChannelApproval(token: string, decision: ConnectorApprovalDecision): boolean
  resolveChannelApprovalByThread(threadKey: string, decision: ConnectorApprovalDecision): boolean
  telegramFetch?: typeof fetch
}

export const TELEGRAM_API_ORIGIN = 'https://api.telegram.org'
const TELEGRAM_TIMEOUT_MS = 12_000
const TELEGRAM_MAX_JSON_BYTES = 2_000_000

export function createTelegramApiFetch(options: { lookup?: PinnedHostResolver; fetchImpl?: typeof fetch } = {}): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const url = input instanceof URL ? new URL(input.toString()) : new URL(typeof input === 'string' ? input : input.url)
    if (url.origin !== TELEGRAM_API_ORIGIN || url.username || url.password) throw new Error('Telegram API requests must use the exact configured HTTPS origin.')
    const target = await resolvePinnedHost(url, (address) => {
      if (!isGloballyRoutableIp(address)) throw new Error('Telegram API resolved to a private or non-public address; refusing the request.')
    }, { lookup: options.lookup })
    return createPinnedFetch(target, options.fetchImpl ?? fetch)(input, { ...init, redirect: 'error' })
  }) as typeof fetch
}

export async function readTelegramJson(response: Response, maxBytes = TELEGRAM_MAX_JSON_BYTES): Promise<any> {
  const contentType = (response.headers.get('content-type') ?? '').split(';')[0].trim().toLowerCase()
  if (contentType !== 'application/json' && !contentType.endsWith('+json')) {
    await discardResponseBody(response)
    throw new Error('Telegram API returned a non-JSON response.')
  }
  const declared = Number(response.headers.get('content-length'))
  if (Number.isFinite(declared) && declared > maxBytes) {
    await discardResponseBody(response)
    throw new Error('Telegram API response exceeded the allowed size.')
  }
  const reader = response.body?.getReader()
  if (!reader) return {}
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      if (!value) continue
      size += value.byteLength
      if (size > maxBytes) {
        await reader.cancel().catch(() => {})
        throw new Error('Telegram API response exceeded the allowed size.')
      }
      chunks.push(value)
    }
  } catch (error) {
    await reader.cancel().catch(() => {})
    throw error
  }
  const raw = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8')
  try { return JSON.parse(raw || '{}') } catch { throw new Error('Telegram API returned invalid JSON.') }
}

// Telegram chat id -> the threadKey the connector adapter parks approvals under. The
// adapter uses the raw threadKey ("telegram:<chatId>"); bot-control sessions key off chatId.
function telegramThreadKey(chatId: string): string {
  return `telegram:${chatId}`
}

// Pull the approval token out of an inline-button payload ("/approve:ca_xxxx" -> "ca_xxxx").
function approvalTokenFrom(text: string): string | null {
  const match = String(text ?? '').match(/(?:^|:)(ca_[a-z0-9]+)\b/i)
  return match ? match[1] : null
}

export function createBotPollingRuntime(deps: BotPollingRuntimeDeps): BotPollingRuntime {
  let telegramPollTimer: ReturnType<typeof setInterval> | null = null
  let telegramPollInFlight: Promise<BotControlPollResult> | null = null
  const telegramFetch = deps.telegramFetch ?? createTelegramApiFetch()

  function startBotControlPolling() {
    if (telegramPollTimer) { clearInterval(telegramPollTimer); telegramPollTimer = null }
    const telegram = store.settings?.botControl?.telegram
    if (!telegram?.enabled || !telegram.polling || !telegram.botToken || !parseAllowedBotChatIds(telegram.allowedChatIds).size || !parseAllowedBotChatIds(telegram.allowedUserIds).size) return
    setTimeout(() => { void pollTelegramUpdates() }, 1500)
    telegramPollTimer = setInterval(() => { void pollTelegramUpdates() }, 10_000)
  }

  async function pollTelegramUpdates(): Promise<BotControlPollResult> {
    if (telegramPollInFlight) return telegramPollInFlight
    const telegram = store.settings.botControl.telegram
    const token = telegram.botToken.trim()
    if (!telegram.enabled || !telegram.polling || !token) return { ok: false, processed: 0, skipped: 0, error: 'Telegram polling is not configured.', lastUpdateId: telegram.lastUpdateId }
    if (!parseAllowedBotChatIds(telegram.allowedChatIds).size) return { ok: false, processed: 0, skipped: 0, error: 'Add at least one allowed Telegram chat ID before polling.', lastUpdateId: telegram.lastUpdateId }
    if (!parseAllowedBotChatIds(telegram.allowedUserIds).size) return { ok: false, processed: 0, skipped: 0, error: 'Add at least one allowed Telegram user ID before polling.', lastUpdateId: telegram.lastUpdateId }

    telegramPollInFlight = (async () => {
      let processed = 0
      let skipped = 0
      let lastUpdateId = telegram.lastUpdateId
      const deliveryErrors: string[] = []
      const rememberDeliveryError = (error: unknown) => {
        const safe = String((error as { message?: unknown } | null)?.message ?? error).split(token).join('[redacted]')
        deliveryErrors.push(safe.slice(0, 240))
      }
      try {
        const url = new URL(`https://api.telegram.org/bot${token}/getUpdates`)
        url.searchParams.set('timeout', '0')
        url.searchParams.set('allowed_updates', JSON.stringify(['message', 'callback_query']))
        if (typeof telegram.lastUpdateId === 'number') url.searchParams.set('offset', String(telegram.lastUpdateId + 1))
        const response = await telegramFetch(url, { method: 'GET', signal: AbortSignal.timeout(TELEGRAM_TIMEOUT_MS), redirect: 'error' })
        const payload = await readTelegramJson(response) as { ok?: boolean; result?: any[]; description?: string }
        if (!response.ok || payload.ok === false) throw new Error(payload.description || `Telegram getUpdates failed (${response.status})`)

        for (const update of payload.result ?? []) {
          const extracted = telegramUpdateToIngest(update)
          if (typeof extracted.updateId === 'number') lastUpdateId = Math.max(lastUpdateId ?? extracted.updateId, extracted.updateId)
          if (!extracted.input) { skipped++; continue }
          if (!botChannelAllowed(extracted.input)) { skipped++; continue }
          // Inline-approval interception: an approve/deny button (or "/approve" reply) resolves
          // the parked in-channel approval and is NOT routed through the room-control bot.
          const decision = parseApprovalReply(extracted.input.text)
          if (decision) {
            const token = approvalTokenFrom(extracted.input.text)
            const matched = token
              ? deps.resolveChannelApproval(token, decision)
              : deps.resolveChannelApprovalByThread(telegramThreadKey(extracted.input.chatId), decision)
            if (matched) {
              processed++
              await sendTelegramMessage(extracted.input.chatId, decision === 'approve' ? 'Approved.' : 'Denied.').catch(rememberDeliveryError)
              if (extracted.callbackQueryId) await answerTelegramCallback(extracted.callbackQueryId).catch(() => {})
              continue
            }
          }
          const result = ingestBotControl(extracted.input, deps.botControlRuntime())
          processed++
          // The command has already run. A transient reply failure must not abort later
          // updates (or replay this command on the next poll); record it and keep advancing.
          await sendTelegramMessage(extracted.input.chatId, result.reply).catch(rememberDeliveryError)
          if (extracted.callbackQueryId) await answerTelegramCallback(extracted.callbackQueryId).catch(() => {})
        }

        const deliveryError = deliveryErrors.length
          ? `Telegram reply delivery failed for ${deliveryErrors.length} update(s): ${deliveryErrors[0]}`
          : null
        store.settings.botControl.telegram = { ...store.settings.botControl.telegram, lastUpdateId, lastPollAt: Date.now(), lastError: deliveryError }
        await saveSettings(store.settings)
        return { ok: deliveryErrors.length === 0, processed, skipped, error: deliveryError, lastUpdateId }
      } catch (e: any) {
        const error = String(e?.message ?? e).split(token).join('[redacted]')
        store.settings.botControl.telegram = { ...store.settings.botControl.telegram, lastUpdateId, lastPollAt: Date.now(), lastError: error }
        await persistSettingsCritical('telegram.pollError', store.settings).catch(() => {})
        return { ok: false, processed, skipped, error, lastUpdateId }
      }
    })().finally(() => { telegramPollInFlight = null }) as Promise<BotControlPollResult>
    return telegramPollInFlight
  }

  async function sendTelegramMessage(chatId: string, text: string): Promise<void> {
    const token = store.settings.botControl.telegram.botToken.trim()
    if (!token) return
    const response = await telegramFetch(`${TELEGRAM_API_ORIGIN}/bot${token}/sendMessage`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, text, disable_web_page_preview: true }),
      signal: AbortSignal.timeout(TELEGRAM_TIMEOUT_MS), redirect: 'error',
    })
    try { if (!response.ok) throw new Error(`Telegram sendMessage failed (${response.status}).`) } finally { await discardResponseBody(response) }
  }

  async function answerTelegramCallback(callbackQueryId: string): Promise<void> {
    const token = store.settings.botControl.telegram.botToken.trim()
    if (!token || !callbackQueryId) return
    const response = await telegramFetch(`${TELEGRAM_API_ORIGIN}/bot${token}/answerCallbackQuery`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ callback_query_id: callbackQueryId }),
      signal: AbortSignal.timeout(TELEGRAM_TIMEOUT_MS), redirect: 'error',
    })
    try { if (!response.ok) throw new Error(`Telegram answerCallbackQuery failed (${response.status}).`) } finally { await discardResponseBody(response) }
  }

  function stopBotControlPolling(): void {
    if (telegramPollTimer) clearInterval(telegramPollTimer)
    telegramPollTimer = null
  }

  return { pollTelegramUpdates, startBotControlPolling, stopBotControlPolling }
}
