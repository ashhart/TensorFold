import { createHash, randomUUID } from 'node:crypto'
import type { IMessageConnectorSettings, IMessageItem } from './types'
import { fetchConnectorJson, type ConnectorHttpDeps } from './connector-http'

export interface BlueBubblesDeps extends ConnectorHttpDeps { assertCurrent?: () => void }

export function blueBubblesIdentity(settings: IMessageConnectorSettings): string {
  return createHash('sha256').update(JSON.stringify({ transport: settings.transport ?? 'local', url: settings.blueBubblesUrl?.trim().replace(/\/+$/, ''), password: settings.blueBubblesPassword, allowed: settings.allowedChatIds })).digest('hex').slice(0, 24)
}

export function blueBubblesChats(settings: IMessageConnectorSettings): string[] {
  const chats = [...new Set(settings.allowedChatIds.split(/[\s,]+/).filter(Boolean))]
  if (!chats.length || chats.length > 50 || chats.some(chat => !/^(?:iMessage|SMS);[+-];[^\s\0,]{1,400}$/.test(chat))) {
    throw new Error('Choose 1 to 50 exact BlueBubbles chat GUIDs before reading or sending.')
  }
  return chats
}

export function blueBubblesBase(settings: IMessageConnectorSettings): string {
  let url: URL
  try { url = new URL(settings.blueBubblesUrl ?? '') } catch { throw new Error('Enter the BlueBubbles server URL.') }
  if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('Use a BlueBubbles server origin without credentials, a path or query.')
  }
  if (!settings.blueBubblesPassword?.trim()) throw new Error('Enter the BlueBubbles server password.')
  return url.origin
}

async function request(settings: IMessageConnectorSettings, route: string, body?: unknown, deps: BlueBubblesDeps = {}) {
  const base = blueBubblesBase(settings)
  const url = new URL(`/api/v1/${route}`, base)
  url.searchParams.set('password', settings.blueBubblesPassword!)
  try {
    const guardedFetch: typeof fetch = (input, init) => {
      deps.assertCurrent?.()
      return (deps.fetchImpl ?? fetch)(input, init)
    }
    const { response, json } = await fetchConnectorJson(base, 'BlueBubbles', url, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal: AbortSignal.timeout(20_000),
    }, { ...deps, fetchImpl: guardedFetch, allowPrivateNetwork: true, retry: { ...deps.retry, maxAttempts: 1 } })
    if (response.status === 401 || response.status === 403) throw new Error('BlueBubbles rejected the server password. Check its API settings.')
    if (!response.ok || Number(json?.status ?? 200) >= 400) throw new Error('BlueBubbles could not complete the request. Check the server and Messages session.')
    return json?.data
  } catch (error) {
    const message = error instanceof Error ? error.message : ''
    if (message.startsWith('BlueBubbles rejected') || message.startsWith('BlueBubbles could not')) throw error
    throw new Error('BlueBubbles could not be reached securely. Check the saved server URL and connection.')
  }
}

export async function blueBubblesAccess(settings: IMessageConnectorSettings, deps?: ConnectorHttpDeps) {
  blueBubblesChats(settings)
  await request(settings, 'server/info', undefined, deps)
  return { state: 'ok' as const, detail: 'BlueBubbles server reached. Sync to verify the selected conversations.' }
}

export async function scanBlueBubbles(settings: IMessageConnectorSettings, deps?: ConnectorHttpDeps): Promise<IMessageItem[]> {
  const chats = blueBubblesChats(settings)
  const limit = Math.max(1, Math.min(settings.maxMessages || 200, 1000))
  const items: IMessageItem[] = []
  for (const chatGuid of chats) {
    const rows = await request(settings, 'message/query', { chatGuid, limit, offset: 0, sort: 'DESC', with: ['handle', 'chats'] }, deps)
    if (!Array.isArray(rows)) throw new Error('BlueBubbles returned an invalid message list.')
    for (const row of rows.slice(0, limit)) {
      if (typeof row?.guid !== 'string' || !row.guid || typeof row.text !== 'string' || !row.text.trim()) continue
      if (!Array.isArray(row.chats) || !row.chats.some((chat: any) => chat?.guid === chatGuid)) continue
      const date = Number(row.dateCreated)
      const chat = row.chats.find((value: any) => value?.guid === chatGuid)
      items.push({ id: `bb_${blueBubblesIdentity(settings)}_${Buffer.from(row.guid).toString('base64url')}`, guid: row.guid,
        chatId: chatGuid, chatName: String(chat?.displayName || row.handle?.address || chatGuid),
        handle: String(row.handle?.address ?? ''), text: row.text.slice(0, 100_000),
        isFromMe: row.isFromMe === true, hasAttachments: row.hasAttachments === true,
        date: Number.isFinite(date) && date > 0 ? date : null, replyToId: row.threadOriginatorGuid ?? null })
    }
  }
  return [...new Map(items.map(item => [item.id, item])).values()].sort((a, b) => (b.date ?? 0) - (a.date ?? 0)).slice(0, limit)
}

// Called only after owner approval or an explicit owner send. Never automatically retries.
export async function sendBlueBubbles(settings: IMessageConnectorSettings, chatGuid: string, text: string, deps?: BlueBubblesDeps) {
  if (!settings.enabled || !blueBubblesChats(settings).includes(chatGuid)) throw new Error('Choose a BlueBubbles conversation in the current connection scope before sending.')
  if (!text.trim() || text.length > 100_000 || text.includes('\0')) throw new Error('Review a valid message body before sending.')
  const data = await request(settings, 'message/text', { chatGuid, message: text, tempGuid: randomUUID(), method: 'apple-script' }, deps)
  if (typeof data?.guid !== 'string' || !data.guid) throw new Error('BlueBubbles delivery could not be confirmed. Check Messages before sending again.')
  return { ok: true, sentId: data.guid }
}
