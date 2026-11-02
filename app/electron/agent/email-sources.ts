import type { ConnectorSourceRef, EmailMessage } from './types'

export function emailSourceId(message: Pick<EmailMessage, 'sourceId' | 'id'>): string {
  return `email:${message.sourceId}:${message.id}`
}

export function emailSourceRef(message: EmailMessage): ConnectorSourceRef {
  return {
    id: emailSourceId(message),
    connectorId: 'email',
    sourceId: message.sourceId,
    itemId: message.id,
    title: message.subject || '(no subject)',
    path: message.path,
    excerpt: message.snippet,
    createdAt: message.date,
  }
}

export function formatEmailSourceRefs(messages: EmailMessage[]): string {
  const refs = messages.map(emailSourceRef)
  if (!refs.length) return 'Sources: none.'
  return [
    'Sources:',
    ...refs.map((ref) => `- ${ref.id} | ${ref.title}${ref.excerpt ? ` | ${ref.excerpt}` : ''}`),
  ].join('\n')
}
