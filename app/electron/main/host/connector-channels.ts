import { requestApprovalViaChannel, type ChannelAdapter, type ChannelApprovalRequest } from '../../agent/channel-adapter'
import { createChannelApprovalRegistry, discordApprovalComponents, slackApprovalBlocks, telegramApprovalKeyboard } from '../../agent/channel-approvals'
import { blueBubblesIdentity } from '../../agent/bluebubbles-connector'
import { sendIMessage } from '../../agent/imessage-connector'
import { sendWhatsAppMessage, whatsappEventFromMessage } from '../../agent/whatsapp-bridge'
import { editSlackMessage, postSlackMessage } from '../../agent/slack-live'
import { editDiscordMessage, postDiscordMessage } from '../../agent/discord-connector'
import { splitForPlatform } from '../../agent/channel-adapter'
import { auditPreview } from '../../agent/security'
import { discardResponseBody } from '../../agent/pinned-fetch'
import { recordConnectorInteraction } from '../../agent/connector-spine'
import { enrichContacts, whatsappContactObservations } from './connector-enrich'
import { createTelegramApiFetch, readTelegramJson, TELEGRAM_API_ORIGIN } from './bot-polling-runtime'
import { persistContacts } from './persistence'
import { auditEvent } from './audit'
import { store } from '../state'
import type { ConnectorApprovalDecision } from '../../agent/types'
import type { ConnectorEvent } from '../../agent/types'
import { reconcileTelegramDraft } from './connector-runtime'
import type { ConnectorEventSpine } from './connector-event-spine'

export interface ConnectorChannels {
  requestChannelApproval: (threadKey: string, request: ChannelApprovalRequest) => Promise<import('../../agent/channel-adapter').ChannelApprovalResult>
  resolveChannelApproval: (token: string, decision: ConnectorApprovalDecision) => boolean
  resolveChannelApprovalByThread: (threadKey: string, decision: ConnectorApprovalDecision) => boolean
  rememberWhatsAppEvents: (events: ConnectorEvent[]) => void
  channelAdapters: ChannelAdapter[]
}

export function createConnectorChannels(deps: {
  spine: Pick<ConnectorEventSpine, 'rememberConnectorEvents'>
}): ConnectorChannels {
  const { rememberConnectorEvents } = deps.spine

  function rememberWhatsAppEvents(events: ConnectorEvent[]): void {
    const observations = whatsappContactObservations(events)
    if (observations.length) {
      store.contacts = enrichContacts(store.contacts, [], observations).directory
      persistContacts()
    }
    void rememberConnectorEvents(events)
  }

  function approvalNotice(request: ChannelApprovalRequest): string {
    const detail = request.command?.trim() ? ` (${request.command.trim()})` : ''
    return `Approval needed: ${request.title}${detail}`.slice(0, 280)
  }

  const approvalRegistry = createChannelApprovalRegistry()
  const telegramApiFetch = createTelegramApiFetch()

  async function sendTelegramApprovalPrompt(chatId: string, text: string, replyMarkup?: unknown): Promise<boolean> {
    const token = store.settings.botControl?.telegram?.botToken?.trim()
    if (!token) return false
    const body: Record<string, unknown> = { chat_id: chatId, text, disable_web_page_preview: true }
    if (replyMarkup) body.reply_markup = replyMarkup
    const res = await telegramApiFetch(`${TELEGRAM_API_ORIGIN}/bot${token}/sendMessage`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(12_000), redirect: 'error',
    }).catch(() => null)
    if (!res) return false
    try { return res.ok } finally { await discardResponseBody(res) }
  }

  async function sendTelegramText(chatId: string, text: string): Promise<string | null> {
    const token = store.settings.botControl?.telegram?.botToken?.trim()
    if (!token) return null
    const response = await telegramApiFetch(`${TELEGRAM_API_ORIGIN}/bot${token}/sendMessage`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, text, disable_web_page_preview: true }),
      signal: AbortSignal.timeout(12_000), redirect: 'error',
    }).catch(() => null)
    if (!response) return null
    if (!response.ok) { await discardResponseBody(response); return null }
    const res = await readTelegramJson(response).catch(() => null)
    const id = (res as { result?: { message_id?: number } } | null)?.result?.message_id
    return id == null ? null : String(id)
  }

  async function editTelegramText(chatId: string, messageId: string, text: string): Promise<boolean> {
    const token = store.settings.botControl?.telegram?.botToken?.trim()
    if (!token) return false
    const res = await telegramApiFetch(`${TELEGRAM_API_ORIGIN}/bot${token}/editMessageText`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, message_id: Number(messageId), text, disable_web_page_preview: true }),
      signal: AbortSignal.timeout(12_000), redirect: 'error',
    }).catch(() => null)
    if (!res) return false
    try { return res.ok } finally { await discardResponseBody(res) }
  }

  async function deleteTelegramText(chatId: string, messageId: string): Promise<boolean> {
    const token = store.settings.botControl?.telegram?.botToken?.trim()
    if (!token) return false
    const res = await telegramApiFetch(`${TELEGRAM_API_ORIGIN}/bot${token}/deleteMessage`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, message_id: Number(messageId) }),
      signal: AbortSignal.timeout(12_000), redirect: 'error',
    }).catch(() => null)
    if (!res) return false
    try { return res.ok } finally { await discardResponseBody(res) }
  }

  const channelAdapters: ChannelAdapter[] = [
    { platform: 'imessage', send: async (addr, reply) => {
      const settings = store.settings.imessage
      const account = settings.transport === 'bluebubbles' ? `bb-${blueBubblesIdentity(settings)}` : null
      if (addr.account !== account) return { ok: false, error: 'This iMessage connection changed; sync and review the conversation again.' }
      const r = await sendIMessage(addr.conversation, reply.text, settings, () => {
        if (!store.settings.imessage.enabled || blueBubblesIdentity(store.settings.imessage) !== blueBubblesIdentity(settings)) throw new Error('The iMessage connection changed before dispatch.')
      })
      await auditEvent('imessage.send', { to: auditPreview(addr.conversation), chars: reply.text.length })
      return { ok: r.ok }
    } },
    { platform: 'whatsapp', send: async (addr, reply) => {
      if (!addr.account) return { ok: false, error: 'WhatsApp account missing from the conversation address.' }
      const r = await sendWhatsAppMessage(addr.account, addr.conversation, reply.text)
      if (r.ok) await rememberConnectorEvents([whatsappEventFromMessage({ sourceId: addr.account, jid: addr.conversation, fromMe: true, text: reply.text, sender: 'You', ts: Date.now(), messageId: r.sentId, authenticatedAccountJid: r.authenticatedAccountJid })])
      return { ok: r.ok, sentId: r.sentId }
    } },
    { platform: 'slack', supportsInlineApproval: true,

      supportsDraftStreaming: true, send: async (addr, reply) => {
      const source = store.settings.slack.sources.find((s) => s.id === addr.account && s.kind === 'api_token')
      if (!source) return { ok: false, error: 'Live Slack source not found.' }
      const r = await postSlackMessage(source, addr.conversation, reply.text, reply.inReplyTo ?? addr.message ?? undefined)
      rememberConnectorEvents([recordConnectorInteraction({ connectorId: 'slack', pluginName: 'Slack', action: 'message_sent', direction: 'outbound', target: addr.conversation, output: { title: 'Sent a Slack message', summary: `${reply.text.slice(0, 120)} -> ${addr.conversation}`, threadKey: addr.raw } })])
      return { ok: r?.ok ?? true, sentId: r?.ts }
    }, sendDraft: async (addr, text) => {
      const source = store.settings.slack.sources.find((s) => s.id === addr.account && s.kind === 'api_token')
      if (!source) return { ok: false, error: 'Live Slack source not found.' }
      const r = await postSlackMessage(source, addr.conversation, text, addr.message ?? undefined).catch(() => null)
      return r?.ts ? { ok: true, handle: { messageId: r.ts, platform: 'slack' } } : { ok: false, error: 'Slack draft could not be posted.' }
    }, updateDraft: async (addr, handle, text) => {
      if (!handle.messageId) return { ok: true, handle }
      const source = store.settings.slack.sources.find((s) => s.id === addr.account && s.kind === 'api_token')
      if (!source) return { ok: true, handle }
      // Best-effort: a failed/no-op edit is swallowed so streaming never throws; finalize lands the text.
      await editSlackMessage(source, addr.conversation, handle.messageId, text).catch(() => {})
      return { ok: true, handle }
    }, finalizeDraft: async (addr, handle, reply) => {
      const source = store.settings.slack.sources.find((s) => s.id === addr.account && s.kind === 'api_token')
      if (!source) return { ok: false, error: 'Live Slack source not found.' }
      const chunks = splitForPlatform('slack', reply.text)
      const first = chunks[0] ?? reply.text

      const edited = handle.messageId ? await editSlackMessage(source, addr.conversation, handle.messageId, first).then(() => true).catch(() => false) : false
      const firstId = edited ? handle.messageId : (await postSlackMessage(source, addr.conversation, first, reply.inReplyTo ?? addr.message ?? undefined).catch(() => null))?.ts
      if (!firstId) return { ok: false, error: 'Slack reply could not be sent.' }
      const continuationIds: string[] = []
      for (let i = 1; i < chunks.length; i++) {
        const r = await postSlackMessage(source, addr.conversation, chunks[i], reply.inReplyTo ?? addr.message ?? undefined).catch(() => null)
        if (r?.ts) continuationIds.push(r.ts)
      }
      rememberConnectorEvents([recordConnectorInteraction({ connectorId: 'slack', pluginName: 'Slack', action: 'message_sent', direction: 'outbound', target: addr.conversation, output: { title: 'Sent a Slack message', summary: `${reply.text.slice(0, 120)} -> ${addr.conversation}`, threadKey: addr.raw } })])
      return { ok: true, sentId: firstId, continuationIds: continuationIds.length ? continuationIds : undefined, chunkCount: chunks.length }
    }, requestApproval: async (addr, request) => {
      const source = store.settings.slack.sources.find((s) => s.id === addr.account && s.kind === 'api_token')
      if (!source) throw new Error('Live Slack source not found.')
      const { token, promise } = approvalRegistry.create('slack', addr.raw)

      try {
        await postSlackMessage(source, addr.conversation, approvalNotice(request), addr.message ?? undefined, slackApprovalBlocks(request, token))
      } catch (e) { approvalRegistry.resolve(token, 'deny'); throw e }
      await auditEvent('connector.approval_prompt', { platform: 'slack', threadKey: auditPreview(addr.raw), inline: true })
      return promise
    } },
    { platform: 'discord', supportsInlineApproval: true,

      supportsDraftStreaming: true, send: async (addr, reply) => {
      const source = store.settings.discord.sources.find((s) => s.id === addr.account)
      if (!source) return { ok: false, error: 'Discord source not found.' }
      const r = await postDiscordMessage(source, addr.conversation, reply.text, reply.inReplyTo ?? undefined)
      rememberConnectorEvents([recordConnectorInteraction({ connectorId: 'discord', pluginName: 'Discord', action: 'message_sent', direction: 'outbound', target: addr.conversation, output: { title: 'Sent a Discord message', summary: `${reply.text.slice(0, 120)} -> ${addr.conversation}`, threadKey: `discord:${source.id}:${addr.conversation}` } })])
      return { ok: r?.ok ?? true, sentId: r?.id }
    }, sendDraft: async (addr, text) => {
      const source = store.settings.discord.sources.find((s) => s.id === addr.account)
      if (!source) return { ok: false, error: 'Discord source not found.' }
      const r = await postDiscordMessage(source, addr.conversation, text).catch(() => null)
      return r?.id ? { ok: true, handle: { messageId: r.id, platform: 'discord' } } : { ok: false, error: 'Discord draft could not be sent.' }
    }, updateDraft: async (addr, handle, text) => {
      if (!handle.messageId) return { ok: true, handle }
      const source = store.settings.discord.sources.find((s) => s.id === addr.account)
      if (!source) return { ok: true, handle }
      // Best-effort: a failed in-flight edit is swallowed so streaming never throws; finalize lands it.
      await editDiscordMessage(source, addr.conversation, handle.messageId, text).catch(() => {})
      return { ok: true, handle }
    }, finalizeDraft: async (addr, handle, reply) => {
      const source = store.settings.discord.sources.find((s) => s.id === addr.account)
      if (!source) return { ok: false, error: 'Discord source not found.' }
      const chunks = splitForPlatform('discord', reply.text)
      const first = chunks[0] ?? reply.text

      const edited = handle.messageId ? await editDiscordMessage(source, addr.conversation, handle.messageId, first).then(() => true).catch(() => false) : false
      const firstId = edited ? handle.messageId : (await postDiscordMessage(source, addr.conversation, first, reply.inReplyTo ?? undefined).catch(() => null))?.id
      if (!firstId) return { ok: false, error: 'Discord reply could not be sent.' }
      const continuationIds: string[] = []
      for (let i = 1; i < chunks.length; i++) {
        const r = await postDiscordMessage(source, addr.conversation, chunks[i]).catch(() => null)
        if (r?.id) continuationIds.push(r.id)
      }
      rememberConnectorEvents([recordConnectorInteraction({ connectorId: 'discord', pluginName: 'Discord', action: 'message_sent', direction: 'outbound', target: addr.conversation, output: { title: 'Sent a Discord message', summary: `${reply.text.slice(0, 120)} -> ${addr.conversation}`, threadKey: `discord:${source.id}:${addr.conversation}` } })])
      return { ok: true, sentId: firstId, continuationIds: continuationIds.length ? continuationIds : undefined, chunkCount: chunks.length }
    }, requestApproval: async (addr, request) => {
      const source = store.settings.discord.sources.find((s) => s.id === addr.account)
      if (!source) throw new Error('Discord source not found.')
      const { token, promise } = approvalRegistry.create('discord', addr.raw)
      try {
        await postDiscordMessage(source, addr.conversation, approvalNotice(request), undefined, discordApprovalComponents(token))
      } catch (e) { approvalRegistry.resolve(token, 'deny'); throw e }
      await auditEvent('connector.approval_prompt', { platform: 'discord', threadKey: auditPreview(addr.raw), inline: true })
      return promise
    } },
    { platform: 'telegram', supportsInlineApproval: true,

      supportsDraftStreaming: true, send: async (addr, reply) => {
      const id = await sendTelegramText(addr.conversation, reply.text)
      return id ? { ok: true, sentId: id } : { ok: false, error: 'Telegram bot is not configured.' }
    }, sendDraft: async (addr, text) => {
      const id = await sendTelegramText(addr.conversation, text)
      return id ? { ok: true, handle: { messageId: id, platform: 'telegram' } } : { ok: false, error: 'Telegram bot is not configured.' }
    }, updateDraft: async (addr, handle, text) => {
      if (!handle.messageId) return { ok: true, handle }
      await editTelegramText(addr.conversation, handle.messageId, text)
      return { ok: true, handle }
    }, finalizeDraft: async (addr, handle, reply) => {
      const chunks = splitForPlatform('telegram', reply.text)
      const first = chunks[0] ?? reply.text
      const reconciled = await reconcileTelegramDraft(handle.messageId, first, {
        edit: (messageId, text) => editTelegramText(addr.conversation, messageId, text),
        remove: (messageId) => deleteTelegramText(addr.conversation, messageId),
        send: (text) => sendTelegramText(addr.conversation, text),
      })
      const firstId = reconciled.sentId
      if (!firstId) return { ok: false, error: reconciled.error ?? 'Telegram bot is not configured.' }
      const continuationIds: string[] = []
      for (let i = 1; i < chunks.length; i++) {
        const id = await sendTelegramText(addr.conversation, chunks[i])
        if (id) continuationIds.push(id)
      }
      return { ok: true, sentId: firstId, continuationIds: continuationIds.length ? continuationIds : undefined, chunkCount: chunks.length }
    }, requestApproval: async (addr, request) => {
      const { token, promise } = approvalRegistry.create('telegram', addr.raw)
      const ok = await sendTelegramApprovalPrompt(addr.conversation, approvalNotice(request), telegramApprovalKeyboard(token))
      if (!ok) { approvalRegistry.resolve(token, 'deny'); throw new Error('Telegram bot is not configured.') }
      await auditEvent('connector.approval_prompt', { platform: 'telegram', threadKey: auditPreview(addr.raw), inline: true })
      return promise
    } },
  ]

  // Prompt for an approval in the channel that owns this thread, blocking until answered.
  async function requestChannelApproval(threadKey: string, request: ChannelApprovalRequest) {
    return requestApprovalViaChannel(channelAdapters, threadKey, request)
  }

  function resolveChannelApproval(token: string, decision: ConnectorApprovalDecision): boolean {
    return approvalRegistry.resolve(token, decision)
  }

  function resolveChannelApprovalByThread(threadKey: string, decision: ConnectorApprovalDecision): boolean {
    return approvalRegistry.resolveByThreadKey(threadKey, decision)
  }

  return {
    requestChannelApproval,
    resolveChannelApproval,
    resolveChannelApprovalByThread,
    rememberWhatsAppEvents,
    channelAdapters,
  }
}
