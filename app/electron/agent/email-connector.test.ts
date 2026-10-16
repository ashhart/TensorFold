// Run: npx tsx electron/agent/email-connector.test.ts
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { emailDraftEvent, emailSourceId, emailSourceRef, emailThreadResolvedEvent, makeEmailSource, readEmailMessage, readEmailThread, scanEmailMessages, summarizeEmail, syncEmailConnector, validateAndMakeEmailSource, writeEmailOutboxMessage } from './email-connector'
import type { EmailConnectorSettings } from './types'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-email-'))
  const settings: EmailConnectorSettings = {
    sources: [{ id: 'src', label: 'Test Mail', path: root, kind: 'eml_dir', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
    maxMessages: 20,
    lastSyncAt: null,
  }

  await fs.writeFile(path.join(root, 'one.eml'), [
    'Message-ID: <one@example.test>',
    'Date: Fri, 19 Jun 2026 09:00:00 +0000',
    'From: Maya <maya@example.test>',
    'To: User <user@example.test>',
    'Subject: Renewal question',
    '',
    'Can you confirm pricing and rollout dates?',
  ].join('\n'), 'utf8')

  await fs.writeFile(path.join(root, 'two.eml'), [
    'Message-ID: <two@example.test>',
    'In-Reply-To: <one@example.test>',
    'References: <one@example.test>',
    'Date: Fri, 19 Jun 2026 09:10:00 +0000',
    'From: User <user@example.test>',
    'To: Maya <maya@example.test>',
    'Subject: Re: Renewal question',
    '',
    'I will send the details today.',
  ].join('\n'), 'utf8')

  const { messages } = await scanEmailMessages(settings, { limit: 10 })
  assert.equal(messages.length, 2)

  const first = messages.find((message) => message.subject === 'Renewal question')
  assert.ok(first)
  const read = await readEmailMessage(settings, `email:${first.id}`)
  assert.equal(read.id, first.id)
  assert.match(read.body, /pricing/)
  assert.equal(emailSourceId(read), `email:src:${read.id}`)
  assert.equal(emailSourceRef(read).connectorId, 'email')

  const thread = await readEmailThread(settings, `email:${first.id}`)
  assert.equal(thread.messages.length, 2)
  assert.match(thread.summary, /2 messages/)

  const summary = await summarizeEmail(settings, { limit: 10 })
  assert.match(summary.summary, /Sources:/)
  assert.match(summary.summary, /email:src:/)
  assert.equal(summary.sources.length, 2)
  assert.ok(summary.sources.every((source) => source.id.startsWith('email:src:')))

  const sourceRefs = thread.messages.map(emailSourceRef)
  const replyDraft = emailDraftEvent({ mode: 'draft', to: 'maya@example.test', subject: 'Re: Renewal question', body: 'Draft reply.', inReplyTo: first.id }, 123, sourceRefs)
  assert.equal(replyDraft.action, 'reply_draft_created')
  assert.equal(replyDraft.requiresApproval, false)
  assert.equal(replyDraft.sources?.length, 2)

  const replySend = emailDraftEvent({ mode: 'send', to: 'maya@example.test', subject: 'Re: Renewal question', body: 'Approved reply.', inReplyTo: first.id }, 124, sourceRefs)
  assert.equal(replySend.action, 'reply_requested')
  assert.equal(replySend.requiresApproval, true)
  assert.equal(replySend.approvalStatus, 'pending')
  assert.equal(replySend.sources?.[0]?.connectorId, 'email')
  const outbox = path.join(root, 'outbox')
  await assert.rejects(() => writeEmailOutboxMessage(replySend, outbox, 126), /approved before transport/)
  const approvedSend = { ...replySend, requiresApproval: false, approvalStatus: 'approved' as const, resolvedAt: 126, resolvedBy: 'test' }
  const delivered = await writeEmailOutboxMessage(approvedSend, outbox, 126)
  assert.equal(delivered.transport, 'local_outbox')
  assert.equal(delivered.eventId, replySend.id)
  assert.equal(delivered.approvalId, replySend.approvalId)
  assert.match(delivered.path, /\.eml$/)
  const rawOutbox = await fs.readFile(delivered.path, 'utf8')
  assert.match(rawOutbox, /To: maya@example\.test/)
  assert.match(rawOutbox, /Subject: Re: Renewal question/)
  assert.match(rawOutbox, new RegExp(`X-TensorFold-Studio-Connector-Event: ${replySend.id}`))
  assert.match(rawOutbox, /X-TensorFold-Studio-Approval-ID: approval-/)
  assert.match(rawOutbox, /X-TensorFold-Studio-Sources: email:src:/)
  assert.match(rawOutbox, /In-Reply-To:/)
  assert.match(rawOutbox, /Approved reply\./)
  assert.equal((await fs.stat(outbox)).mode & 0o777, 0o700, 'email outbox directory is owner-only')
  assert.equal((await fs.stat(delivered.path)).mode & 0o777, 0o600, 'email outbox message is owner-only')

  const resolved = emailThreadResolvedEvent({ id: first.id, note: 'Reply drafted.' }, thread, 125)
  assert.equal(resolved.action, 'thread_resolved')
  assert.equal(resolved.requiresApproval, false)
  assert.equal(resolved.resolvedAt, 125)
  assert.equal(resolved.sources?.length, 2)

  // INCREMENTAL SYNC + REPLY-TO CONTEXT: two.eml is In-Reply-To one@example.test, so its
  // event quotes one.eml's author + snippet; and the source cursor advances so a re-sync
  // (feeding the advanced cursor back) emits nothing. (An outbox .eml written earlier in
  // this test also lands under root, so assert on the specific events, not an exact count.)
  const sync = await syncEmailConnector(settings, [])
  const replyEvent = sync.events.find((e) => /send the details/i.test(e.summary))
  assert.ok(replyEvent, 'the reply message produced an event')
  assert.equal(replyEvent!.reply?.author, 'Maya <maya@example.test>')
  assert.match(replyEvent!.reply?.text ?? '', /pricing/i)
  assert.match(replyEvent!.summary, /In reply to Maya .*pricing/i)
  const rootEvent = sync.events.find((e) => /pricing and rollout/i.test(e.summary) && !e.reply)
  assert.ok(rootEvent, 'the root (non-reply) message has its own event without reply context')
  const advancedSource = sync.sources.find((s) => s.id === 'src')
  assert.ok(advancedSource?.cursor, 'the email source cursor advanced')
  const reSync = await syncEmailConnector({ ...settings, sources: [advancedSource!] }, [])
  assert.equal(reSync.events.length, 0, 'a re-sync with the advanced cursor emits no duplicate events')

  // #59: validateAndMakeEmailSource rejects non-existent paths so users get immediate feedback.
  const nonexistentPath = path.join(os.tmpdir(), `vontra-email-missing-${Date.now()}`)
  await assert.rejects(
    () => validateAndMakeEmailSource(nonexistentPath),
    /does not exist|not accessible/i,
  )

  // #59: validateAndMakeEmailSource rejects a file (not a directory).
  const filePath = path.join(root, 'one.eml')
  await assert.rejects(
    () => validateAndMakeEmailSource(filePath),
    /does not exist|not accessible/i,
  )

  // #59: validateAndMakeEmailSource accepts a valid directory and builds an eml_dir source.
  const validDir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-email-valid-'))
  const validSource = await validateAndMakeEmailSource(validDir)
  assert.equal(validSource.path, validDir)
  assert.equal(validSource.kind, 'eml_dir')
  assert.equal(validSource.lastError, null)

  // #59: Maildir paths are still auto-detected through the validating path.
  const maildir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-Maildir-'))
  const maildirSource = await validateAndMakeEmailSource(maildir)
  assert.equal(maildirSource.kind, 'maildir')
  await fs.rm(maildir, { recursive: true, force: true })

  // #59: the synchronous makeEmailSource + scanEmailMessages flow is unchanged.
  const syncSource = makeEmailSource(validDir)
  const validScan = await scanEmailMessages({ sources: [syncSource], maxMessages: 10, lastSyncAt: null }, { limit: 10 })
  assert.ok(Array.isArray(validScan.messages))
  await fs.rm(validDir, { recursive: true, force: true })

  // [HIGH #11] Multipart body decoding must use the matched PART's OWN Content-Transfer-Encoding,
  // not the outer message's. A typical multipart/alternative has plain outer headers (no CTE)
  // while the text/plain part declares its own base64 (or quoted-printable) encoding.
  const multipartRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-email-multipart-'))
  const plainText = 'Café prices increase by 20% on July 1st.'
  const base64Body = Buffer.from(plainText, 'utf8').toString('base64')
  await fs.writeFile(path.join(multipartRoot, 'multipart.eml'), [
    'Message-ID: <multipart@example.test>',
    'Date: Fri, 19 Jun 2026 09:20:00 +0000',
    'From: Maya <maya@example.test>',
    'To: User <user@example.test>',
    'Subject: Pricing update',
    'Content-Type: multipart/alternative; boundary="BOUNDARY123"',
    '',
    '--BOUNDARY123',
    'Content-Type: text/plain; charset="utf-8"',
    'Content-Transfer-Encoding: base64',
    '',
    base64Body,
    '--BOUNDARY123',
    'Content-Type: text/html; charset="utf-8"',
    'Content-Transfer-Encoding: base64',
    '',
    Buffer.from(`<html><body><p>${plainText}</p></body></html>`, 'utf8').toString('base64'),
    '--BOUNDARY123--',
    '',
  ].join('\n'), 'utf8')

  const multipartSettings: EmailConnectorSettings = {
    sources: [{ id: 'mp', label: 'Multipart Mail', path: multipartRoot, kind: 'eml_dir', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
    maxMessages: 20,
    lastSyncAt: null,
  }
  const { messages: multipartMessages } = await scanEmailMessages(multipartSettings, { limit: 10 })
  assert.equal(multipartMessages.length, 1)
  const multipartRead = await readEmailMessage(multipartSettings, `email:${multipartMessages[0].id}`)
  // The body must be the decoded plain text, not the raw base64 blob.
  assert.match(multipartRead.body, /Café prices increase by 20% on July 1st\./)
  assert.ok(!multipartRead.body.includes(base64Body), 'body must not contain the raw undecoded base64 blob')
  await fs.rm(multipartRoot, { recursive: true, force: true })

  await fs.rm(root, { recursive: true, force: true })
  console.log('email connector tests passed')
}

void main()
