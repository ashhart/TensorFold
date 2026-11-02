import { createHash } from 'node:crypto'

export function connectorSourceBinding(source: { id: string; microsoftRevision?: number; kind?: string; path?: string; url?: string; host?: string; user?: string; mailbox?: string; port?: number; tls?: boolean }): string {
  const identity = { id: source.id, microsoftRevision: source.microsoftRevision, kind: source.kind, path: source.path, url: source.url, host: source.host, user: source.user, mailbox: source.mailbox, port: source.port, tls: source.tls }
  return createHash('sha256').update(JSON.stringify(identity)).digest('hex')
}
