import type { AssistantControlSnapshot, ConnectorEvent } from '../../lib/types'
import type { useGmailInbox } from './useGmailInbox'
export type ReturnTypeOfGmailInbox = ReturnType<typeof useGmailInbox>

export function mergeGmailInboxSnapshot(base: AssistantControlSnapshot | null, gmail: ReturnTypeOfGmailInbox): AssistantControlSnapshot | null {
  if (!base) return null
  const events: ConnectorEvent[] = [...gmail.snapshot.events]
  if (gmail.opened) {
    const anchor = events.find(event => event.threadKey === gmail.opened!.key && event.direction === 'inbound')
    if (anchor) {
      const ids = new Set(gmail.opened.thread.messages.map(message => message.id))
      for (let index = events.length - 1; index >= 0; index--) if (events[index].threadKey === anchor.threadKey && events[index].direction === 'inbound' && ids.has(String(events[index].sources?.[0]?.itemId))) events.splice(index, 1)
      const account = gmail.snapshot.accounts.find(item => anchor.threadKey?.startsWith(`gmail:${encodeURIComponent(item.id)}:`))
      for (const message of gmail.opened.thread.messages) events.push({
        ...anchor, direction: (message.from.match(/<([^<>]+)>/)?.[1] ?? message.from).trim().toLowerCase() === account?.email.toLowerCase() ? 'outbound' : 'inbound', id: `${anchor.id}:read:${message.id}`, actor: message.from, title: message.subject,
        sources: anchor.sources?.map(source => ({ ...source, id: `${source.id}:read:${message.id}`, itemId: message.id, title: message.subject, excerpt: message.snippet, path: source.path?.replace(/#all\/.*$/, `#all/${encodeURIComponent(message.id)}`) })),
        summary: message.text ?? message.snippet, payload: { ...anchor.payload, messageId: message.id, from: message.from, subject: message.subject, text: message.text ?? message.snippet }, createdAt: message.internalDate ?? (Date.parse(message.date) || anchor.createdAt)
      })
    }
  }
  const connected = gmail.snapshot.accounts.some(account => account.enabled && account.connected)
  return { ...base, events: [...events, ...base.events].sort((a, b) => b.createdAt - a.createdAt), connectors: base.connectors.map(connector => connector.id === 'email' && connected ? { ...connector, status: 'connected', permissionStatus: 'granted' } : connector) }
}
