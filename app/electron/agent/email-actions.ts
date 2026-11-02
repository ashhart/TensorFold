import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { ConnectorEvent, ConnectorSourceRef, EmailDraftInput, EmailThreadResult } from './types'
import { snippet } from './email-files'
import { emailSourceRef } from './email-sources'

export interface EmailTransportResult {
  transport: 'local_outbox'
  eventId: string
  approvalId: string | null
  path: string
  deliveredAt: number
}

export function emailDraftEvent(input: EmailDraftInput, now = Date.now(), sources: ConnectorSourceRef[] = []): ConnectorEvent {
  const wantsSend = input.mode === 'send'
  const wantsReply = Boolean(input.inReplyTo || input.threadKey)
  const action = wantsSend
    ? wantsReply ? 'reply_requested' : 'send_requested'
    : wantsReply ? 'reply_draft_created' : 'draft_created'
  const label = wantsSend
    ? wantsReply ? 'Reply send request' : 'Send request'
    : wantsReply ? 'Reply draft' : 'Draft'
  return {
    id: randomUUID(),
    connectorId: 'email',
    pluginName: 'Email',
    action,
    direction: 'outbound',
    classification: 'approval',
    priority: wantsSend ? 'high' : 'normal',
    trustLevel: wantsSend ? 'approve' : 'suggest',
    requiresApproval: wantsSend,
    approvalId: wantsSend ? `approval-${randomUUID().slice(0, 8)}` : null,
    approvalStatus: wantsSend ? 'pending' : null,
    resolvedAt: null,
    resolvedBy: null,
    title: input.subject || label,
    summary: `${label} to ${input.to}: ${snippet(input.body, 220)}`,
    actor: input.to,
    threadKey: input.threadKey || input.inReplyTo || `email:draft:${randomUUID().slice(0, 8)}`,
    payloadKeys: ['to', 'subject', 'body', ...(input.threadKey ? ['thread_key'] : []), ...(input.inReplyTo ? ['in_reply_to'] : [])],
    payload: {
      mode: input.mode ?? 'draft',
      ...(input.accountId ? { accountId: input.accountId } : {}),
      to: input.to,
      subject: input.subject,
      body: input.body,
      ...(input.threadKey ? { threadKey: input.threadKey } : {}),
      ...(input.inReplyTo ? { inReplyTo: input.inReplyTo } : {}),
    },
    sources: sources.length ? sources : undefined,
    createdAt: now,
  }
}

export async function writeEmailOutboxMessage(event: ConnectorEvent, outboxDir: string, now = Date.now()): Promise<EmailTransportResult> {
  if (event.requiresApproval || event.approvalStatus !== 'approved') throw new Error('Email send events must be approved before transport.')
  if (event.connectorId !== 'email' || !['send_requested', 'reply_requested'].includes(event.action)) throw new Error('Connector event is not an email send request.')
  const input = emailDraftInputFromEvent(event)
  await fs.mkdir(outboxDir, { recursive: true, mode: 0o700 })

  await fs.chmod(outboxDir, 0o700)
  const file = path.join(outboxDir, `${new Date(now).toISOString().replace(/[:.]/g, '-')}-${safeFilePart(event.id)}.eml`)
  await fs.writeFile(file, renderOutboxEmail(event, input, now), { encoding: 'utf8', mode: 0o600 })
  await fs.chmod(file, 0o600)
  return { transport: 'local_outbox', eventId: event.id, approvalId: event.approvalId, path: file, deliveredAt: now }
}

export function emailDraftInputFromEvent(event: ConnectorEvent): EmailDraftInput {
  const payload = event.payload ?? {}
  const to = stringPayload(payload.to)
  const subject = stringPayload(payload.subject)
  const body = stringPayload(payload.body)
  if (!to || !subject || !body) throw new Error('Email send event is missing draft payload.')
  return {
    mode: payload.mode === 'send' ? 'send' : 'draft',
    to,
    subject,
    body,
    threadKey: stringPayload(payload.threadKey) || undefined,
    inReplyTo: stringPayload(payload.inReplyTo) || undefined,
  }
}

export function renderOutboxEmail(event: ConnectorEvent, input: EmailDraftInput, now: number): string {
  const sourceIds = (event.sources ?? []).map((source) => source.id).join(', ')
  const headers = [
    ['To', input.to],
    ['Subject', input.subject],
    ['Date', new Date(now).toUTCString()],
    ['Message-ID', `<${event.id}@vontra.local>`],
    ['X-TensorFold-Studio-Transport', 'local_outbox'],
    ['X-TensorFold-Studio-Connector-Event', event.id],
    ['X-TensorFold-Studio-Approval-ID', event.approvalId ?? ''],
    ['X-TensorFold-Studio-Thread-Key', event.threadKey],
    ['X-TensorFold-Studio-Sources', sourceIds],
    ...(input.inReplyTo ? [['In-Reply-To', input.inReplyTo] as [string, string]] : []),
    ['Content-Type', 'text/plain; charset=utf-8'],
    ['Content-Transfer-Encoding', '8bit'],
  ] satisfies Array<[string, string]>
  return `${headers.map(([key, value]) => `${key}: ${headerValue(value)}`).join('\r\n')}\r\n\r\n${normalizeBody(input.body)}\r\n`
}

export function stringPayload(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

export function headerValue(value: string): string {
  return String(value || '').replace(/[\r\n]+/g, ' ').trim()
}

export function normalizeBody(value: string): string {
  return String(value || '').replace(/\r\n?/g, '\n').replace(/\n/g, '\r\n')
}

export function safeFilePart(value: string): string {
  return String(value || '').replace(/[^A-Za-z0-9_.-]+/g, '-').slice(0, 80) || randomUUID()
}

export function emailThreadResolvedEvent(input: { id: string; note?: string }, thread: EmailThreadResult, now = Date.now()): ConnectorEvent {
  const sources = thread.messages.map(emailSourceRef)
  return {
    id: randomUUID(),
    connectorId: 'email',
    pluginName: 'Email',
    action: 'thread_resolved',
    direction: 'internal',
    classification: 'communication',
    priority: 'normal',
    trustLevel: 'auto_logged',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: now,
    resolvedBy: 'user',
    title: thread.root.subject || 'Email thread resolved',
    summary: `${input.note?.trim() || 'Marked resolved locally.'} ${thread.summary}`,
    actor: thread.root.from || 'Email',
    threadKey: `email:${thread.root.id}`,
    payloadKeys: ['message_id', ...(input.note ? ['note'] : [])],
    sources,
    createdAt: now,
  }
}
