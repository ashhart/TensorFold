import { createHash } from 'node:crypto'

/** Host-authored identity of a scanned source, excluding contents, cursors and secrets.
 * A reused source ID cannot authorize cached material from another account/location. */
export function connectorSourceBinding(source: { id: string; kind?: string; path?: string; url?: string; host?: string; user?: string; mailbox?: string; port?: number; tls?: boolean }): string {
  const identity = { id: source.id, kind: source.kind, path: source.path, url: source.url, host: source.host, user: source.user, mailbox: source.mailbox, port: source.port, tls: source.tls }
  return createHash('sha256').update(JSON.stringify(identity)).digest('hex')
}
