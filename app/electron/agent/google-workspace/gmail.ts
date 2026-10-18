import type { GoogleWorkspaceMessage, GoogleWorkspaceMessageQuery, GoogleWorkspacePage, GoogleWorkspaceThread } from '../../../shared/google-workspace'
import { GoogleAccounts } from './accounts'
import { boundedString, googlePage, googleResourceId, responseArray, responseText, returnedPageToken } from './validation'

const BASE = 'https://gmail.googleapis.com/gmail/v1/users/me'
function header(payload: any, name: string): string {
  return responseText(responseArray(payload?.headers, 100).find(item => typeof item?.name === 'string' && item.name.toLowerCase() === name)?.value, 2048)
}
function decoded(data: unknown): string {
  if (typeof data !== 'string' || data.length > 512_000 || !/^[A-Za-z0-9_-]*={0,2}$/.test(data)) return ''
  return Buffer.from(data, 'base64url').toString('utf8').slice(0, 256_000)
}
export function parseGoogleMessage(data: any): GoogleWorkspaceMessage {
  const attachments: GoogleWorkspaceMessage['attachments'] = []
  const texts: string[] = []
  let htmlOmitted = false; let visited = 0
  const walk = (part: any, depth: number) => {
    if (!part || depth > 12 || ++visited > 100) return
    if (part.filename || part.body?.attachmentId) {
      attachments.push({ filename: responseText(part.filename, 512) || 'Unnamed attachment', mimeType: responseText(part.mimeType, 200), size: Number.isSafeInteger(part.body?.size) && part.body.size >= 0 ? part.body.size : 0 })
      return
    }
    if (part.mimeType === 'text/plain') texts.push(decoded(part.body?.data))
    if (part.mimeType === 'text/html') htmlOmitted = true
    responseArray(part.parts, 100).forEach(child => walk(child, depth + 1))
  }
  walk(data?.payload, 0)
  return { id: responseText(data?.id, 128), threadId: responseText(data?.threadId, 128), subject: header(data?.payload, 'subject'), from: header(data?.payload, 'from'), to: header(data?.payload, 'to'), date: header(data?.payload, 'date'), snippet: responseText(data?.snippet, 2048), ...(texts.length ? { text: texts.join('\n\n').slice(0, 256_000) } : {}), ...(htmlOmitted ? { htmlOmitted: true } : {}), attachments, untrustedContent: true }
}
export class GoogleGmail {
  constructor(private readonly accounts: GoogleAccounts) {}
  async listMessages(accountId: string, options: GoogleWorkspaceMessageQuery = {}): Promise<GoogleWorkspacePage<GoogleWorkspaceMessage>> {
    const page = googlePage(options.limit, options.pageToken)
    const query = options.query === undefined ? undefined : boundedString(options.query, 'Gmail search query', 1024, true)
    return this.accounts.withAccess(accountId, 'gmail.read', async (token, _revision, signal) => {
      const params = new URLSearchParams({ ...page, ...(query ? { q: query } : {}) })
      const data = await this.accounts.transport.json(`${BASE}/messages?${params}`, token, undefined, 'GET', signal)
      // Search returns summaries; metadata is fetched explicitly with a strict result bound.
      const messages = responseArray(data.messages, Number(page.maxResults))
      const items: GoogleWorkspaceMessage[] = []
      for (const message of messages) {
        const id = googleResourceId(message?.id, 'message ID')
        const details = await this.accounts.transport.json(`${BASE}/messages/${id}?format=metadata&metadataHeaders=Subject&metadataHeaders=From&metadataHeaders=To&metadataHeaders=Date`, token, undefined, 'GET', signal)
        items.push(parseGoogleMessage(details))
      }
      return { items, ...(returnedPageToken(data.nextPageToken) ? { nextPageToken: data.nextPageToken } : {}) }
    })
  }
  async readMessage(accountId: string, id: string): Promise<GoogleWorkspaceMessage> {
    const key = googleResourceId(id, 'message ID')
    return this.accounts.withAccess(accountId, 'gmail.read', async (token, _revision, signal) => parseGoogleMessage(await this.accounts.transport.json(`${BASE}/messages/${key}?format=full`, token, undefined, 'GET', signal)))
  }
  async readThread(accountId: string, id: string): Promise<GoogleWorkspaceThread> {
    const key = googleResourceId(id, 'thread ID')
    return this.accounts.withAccess(accountId, 'gmail.read', async (token, _revision, signal) => {
      const data = await this.accounts.transport.json(`${BASE}/threads/${key}?format=full`, token, undefined, 'GET', signal)
      if (!Array.isArray(data.messages) || data.messages.length > 100) throw new Error('This thread is too large. Read individual messages instead.')
      return { id: responseText(data.id, 128), messages: data.messages.map(parseGoogleMessage) }
    })
  }
}
