import type { ConnectorEvent, EmailReadResult, EmailSourceConfig } from '../types'
import type { MicrosoftMailAccount } from './types'
import { MicrosoftMailAccounts } from './accounts'
import { GRAPH, MicrosoftMailTransport } from './transport'

export class MicrosoftMailbox {
  constructor(private accounts: MicrosoftMailAccounts, private transport: MicrosoftMailTransport) { }
  private headers(account: MicrosoftMailAccount) { return { Authorization: `Bearer ${account.accessToken}`, Prefer: 'outlook.body-content-type="text", IdType="ImmutableId"', 'Content-Type': 'application/json' } }

  private check(response: Response) {
    const fail = (message: string): never => { throw Object.assign(new Error(message), { status: response.status }) }
    if (response.status === 401) fail('Microsoft sign-in needs attention. Reconnect this account.')
    if (response.status === 403) fail('Microsoft denied mail access. Review the granted permissions or tenant policy.')
    if (response.status === 429) fail('Microsoft mail is rate limited. Wait before trying again.')
    if (!response.ok) throw new Error('Microsoft mail could not complete the request. Sync or reconnect this account.')
  }

  async list(source: EmailSourceConfig, limit: number): Promise<EmailReadResult[]> {
    return this.accounts.withAccess(source, async (account, assertCurrent) => {
      const cap = Math.max(1, Math.min(Number.isFinite(limit) ? limit : 300, 2000))
      const route = '/v1.0/me/mailFolders/inbox/messages'
      const query = new URLSearchParams({ '$top': String(Math.min(cap, 100)), '$orderby': 'receivedDateTime desc', '$select': 'id,subject,from,toRecipients,receivedDateTime,isRead,bodyPreview,internetMessageId,conversationId' })
      let next: string | undefined = `${route}?${query}`
      const messages = new Map<string, EmailReadResult>()
      for (let page = 0; next && page < 20 && messages.size < cap; page++) {
        const url = new URL(next, GRAPH)
        if (url.origin !== GRAPH || url.pathname !== route || url.username || url.password || url.hash) throw new Error('Microsoft returned an invalid mail continuation. No credentials were forwarded.')
        const { response, json } = await this.transport.json(GRAPH, url.href, { headers: this.headers(account) })
        assertCurrent(); this.check(response)
        if (!Array.isArray(json.value)) throw new Error('Microsoft returned an invalid mail list.')
        for (const row of json.value) { const message = this.message(source, row); if (message) messages.set(message.id, message) }
        next = typeof json['@odata.nextLink'] === 'string' ? json['@odata.nextLink'] : undefined
      }
      return [...messages.values()].slice(0, cap)
    })
  }

  async read(source: EmailSourceConfig, id: string): Promise<EmailReadResult | null> {
    const remote = this.remoteId(source, id)
    if (!remote) return null
    return this.accounts.withAccess(source, async (account, assertCurrent) => {
      const { response, json } = await this.transport.json(GRAPH, `/v1.0/me/messages/${encodeURIComponent(remote)}?$select=id,subject,from,toRecipients,receivedDateTime,isRead,bodyPreview,body,internetMessageId,internetMessageHeaders,conversationId`, { headers: this.headers(account) })
      assertCurrent(); this.check(response)
      return this.message(source, json)
    })
  }

  private remoteId(source: EmailSourceConfig, id: string): string | null {
    const prefix = `${source.id}:`
    return id.startsWith(prefix) && /^[A-Za-z0-9_+=/-]{1,2048}$/.test(id.slice(prefix.length)) ? id.slice(prefix.length) : null
  }

  private message(source: EmailSourceConfig, row: any): EmailReadResult | null {
    if (typeof row?.id !== 'string' || !this.remoteId(source, `${source.id}:${row.id}`)) return null
    const email = (value: any) => String(value?.emailAddress?.address ?? '')
    const headers: Record<string, string> = {}
    for (const header of Array.isArray(row.internetMessageHeaders) ? row.internetMessageHeaders : []) {
      if (typeof header?.name === 'string' && typeof header.value === 'string' && ['message-id', 'in-reply-to', 'references'].includes(header.name.toLowerCase())) headers[header.name.toLowerCase()] = header.value
    }
    if (row.internetMessageId) headers['message-id'] = String(row.internetMessageId)
    return {
      id: `${source.id}:${row.id}`, sourceId: source.id, sourceLabel: source.label, path: `https://outlook.office.com/mail/`,
      subject: String(row.subject ?? ''), from: email(row.from), to: (row.toRecipients ?? []).map(email).join(', '),
      date: Number.isFinite(Date.parse(row.receivedDateTime)) ? Date.parse(row.receivedDateTime) : null, unread: row.isRead !== true,
      snippet: String(row.bodyPreview ?? '').slice(0, 1000), size: 0, messageId: row.internetMessageId ?? null,
      inReplyTo: headers['in-reply-to'] ?? null, headers,
      body: row.body?.contentType?.toLowerCase() === 'text' ? String(row.body.content ?? '').slice(0, 1_000_000) : String(row.bodyPreview ?? '')
    }
  }

  async send(source: EmailSourceConfig, event: ConnectorEvent): Promise<NonNullable<ConnectorEvent['delivery']>> {
    const snapshot = event.outbound
    if (!snapshot || event.delivery?.status !== 'sending' || event.approvalStatus !== 'approved' || event.requiresApproval || event.delivery.reviewDigest !== event.reviewDigest || snapshot.accountId !== source.id) throw new Error('Microsoft mail requires a durable exact-content approval before sending.')
    return this.accounts.withAccess(source, async (account, assertCurrent) => {
      if (!account.scopes.includes('mail.send')) throw new Error('Reconnect this Microsoft account with sending permission before reviewing a new draft.')
      const replyId = snapshot.inReplyTo ? this.remoteId(source, snapshot.inReplyTo) : null
      if (snapshot.inReplyTo && !replyId) throw new Error('This reply belongs to another Microsoft account. Review the selected conversation again.')
      const message = { subject: snapshot.subject, body: { contentType: 'Text', content: snapshot.body }, toRecipients: [{ emailAddress: { address: snapshot.recipient } }] }
      const route = replyId ? `/v1.0/me/messages/${encodeURIComponent(replyId)}/reply` : '/v1.0/me/sendMail'
      assertCurrent()
      const response = await this.transport.send(route, replyId ? { message } : { message, saveToSentItems: true }, this.headers(account), assertCurrent)
      await response.body?.cancel().catch(() => { })
      assertCurrent(); this.check(response)
      return { transport: 'microsoft_graph', status: 'queued', deliveredAt: null, path: `microsoft:${source.id}`, error: null }
    })
  }
}
