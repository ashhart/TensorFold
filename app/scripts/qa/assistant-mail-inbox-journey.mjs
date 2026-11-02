import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { launchFixtureBrowser, openFixturePage } from './browser-session.mjs'

const output = path.resolve(process.argv[2]), base = process.argv[3] || 'http://127.0.0.1:5186/'
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser(), checks = []
try {
  const { page, errors } = await openFixturePage(browser, base)
  await page.evaluate(async () => {
    const { api } = await import('/src/lib/ipc.ts')
    const qa = globalThis.__mailQa = { enabled: false, revision: 1, synced: false, tests: 0, reads: 0, syncs: 0, prepared: 0, executions: 0, genericReplies: 0, outcome: null, launches: [], review: null, checked: false }
    const account = () => ({ provider: 'gmail', id: 'account-a', email: 'reader@example.com', revision: String(qa.revision), authorization: 'connected', read: true, send: true, inbox: qa.enabled, state: qa.checked ? 'healthy' : 'untested', checkedAt: qa.checked ? Date.now() : null, lastSuccessAt: qa.checked ? Date.now() : null, message: qa.checked ? 'Mailbox read succeeded. No message was sent.' : 'Mailbox access has not been tested. Test connection performs a read only.' })
    const key = () => `gmail:account-a:${qa.revision}:thread-a`
    const incoming = () => ({ id: `gmail-event:account-a:${qa.revision}:message-a`, connectorId: 'email', pluginName: 'Gmail · reader@example.com', action: 'message_received', direction: 'inbound', classification: 'communication', priority: 'high', trustLevel: 'suggest', requiresApproval: false, approvalStatus: null, title: 'Synthetic Gmail thread', summary: 'Synthetic Gmail preview', actor: 'Casey <casey@example.com>', threadKey: key(), payloadKeys: ['gmailInbox'], payload: { gmailInbox: true }, createdAt: Date.now() + 1000, sources: [{ id: 'gmail-source-message', connectorId: 'email', sourceId: 'gmail:account-a', sourceRevision: String(qa.revision), itemId: 'message-a', title: 'Synthetic Gmail thread' }] })
    const snapshot = () => {
      const events = qa.enabled && qa.synced ? [incoming()] : []
      if (qa.review && qa.review.revision === qa.revision && qa.enabled) {
        const email = qa.review.action.email
        events.unshift({ ...incoming(), id: `gmail-review:${qa.review.id}`, action: 'reply_requested', direction: 'outbound', actor: 'reader@example.com', requiresApproval: !qa.outcome, approvalStatus: qa.outcome ? 'approved' : 'pending', summary: email.text, reviewDigest: qa.review.digest, payload: { gmailInbox: true, accountEmail: 'reader@example.com', googleReview: { id: qa.review.id, accountId: qa.review.accountId, revision: qa.review.revision, digest: qa.review.digest }, body: email.text }, outbound: { version: 1, connectorId: 'email', accountId: 'account-a', accountRevision: String(qa.revision), recipient: 'casey@example.com', subject: email.subject, body: email.text, threadKey: key() }, ...(qa.outcome ? { delivery: { transport: 'gmail', status: 'queued', deliveredAt: null } } : {}) })
      }
      return { accounts: [{ id: 'account-a', email: 'reader@example.com', revision: qa.revision, enabled: qa.enabled, connected: true, syncedAt: qa.synced ? Date.now() : null, hasMore: true }], events }
    }
    api.mailConnections = async () => ({ googleConfigured: true, accounts: [account()] })
    api.testMailConnection = async target => { if (target.id !== 'account-a' || target.revision !== String(qa.revision)) throw new Error('Wrong account'); qa.tests++; qa.checked = true; return account() }
    api.gmailInboxSnapshot = async () => snapshot()
    api.gmailInboxEnable = async input => { if (input.accountId !== 'account-a' || input.revision !== qa.revision) throw new Error('Wrong account'); qa.enabled = input.enabled; qa.synced = false; return snapshot() }
    api.gmailInboxSync = async () => { qa.syncs++; if (qa.enabled) qa.synced = true; return snapshot() }
    api.gmailInboxThread = async input => { if (input !== key()) throw new Error('Stale account'); qa.reads++; return { id: 'thread-a', messages: [{ id: 'message-a', threadId: 'thread-a', subject: 'Synthetic Gmail thread', from: 'casey@example.com', to: 'reader@example.com', date: new Date().toISOString(), snippet: 'Synthetic Gmail preview', text: `Synthetic full Gmail body revision ${qa.revision}.`, attachments: [], untrustedContent: true }] } }
    api.gmailInboxReply = async (input, text) => { if (input !== key()) throw new Error('Stale account'); qa.prepared++; qa.outcome = null; qa.review = { id: `review-${qa.prepared}`, createdAt: Date.now(), accountId: 'account-a', accountEmail: 'reader@example.com', revision: qa.revision, digest: 'a'.repeat(64), expiresAt: Date.now() + 600000, action: { kind: 'gmail.send', email: { to: ['casey@example.com'], subject: 'Synthetic Gmail thread', text, reply: { threadId: 'thread-a', messageId: 'message-a' } } } }; return structuredClone(qa.review) }
    api.googleWorkspaceExecuteAction = async receipt => { if (receipt.digest !== qa.review.digest || receipt.revision !== qa.revision) throw new Error('Wrong review'); qa.executions++; qa.outcome = 'accepted'; return { id: receipt.id, status: 'succeeded', remoteId: 'synthetic-sent' } }
    api.googleWorkspaceCancelAction = async receipt => ({ id: receipt.id, status: 'failed', cancelled: true })
    api.replyToThread = async () => { qa.genericReplies++; throw new Error('Gmail must not enter a generic connector send') }
    globalThis.__store.setState({ launch: async input => { qa.launches.push(input); return { ok: false, error: 'Synthetic local model unavailable' } } })
    globalThis.__store.getState().setView({ name: 'assistant' })
  })
  const tab = name => page.getByRole('tab', { name, exact: true })
  await tab('Connections').click()
  const health = page.getByRole('region', { name: 'Mail connection health', exact: true })
  await health.getByText('Mailbox access has not been tested. Test connection performs a read only.', { exact: true }).waitFor()
  assert.equal(await page.evaluate(() => globalThis.__mailQa.tests), 0)
  await health.getByRole('button', { name: 'Test connection', exact: true }).click()
  await health.getByText('Mailbox read succeeded. No message was sent.', { exact: true }).waitFor()
  assert.equal(await page.evaluate(() => globalThis.__mailQa.executions), 0)
  checks.push('Connection status does not probe; explicit mailbox test shows the verified account, grants and successful read time')
  await health.getByRole('button', { name: 'Add Gmail to Inbox', exact: true }).click()
  await health.getByRole('button', { name: 'Remove Gmail from Inbox', exact: true }).waitFor()
  await tab('Inbox').click()
  await page.getByRole('button', { name: 'Refresh inbox', exact: true }).click()
  await page.getByRole('button', { name: 'Sync Gmail', exact: true }).waitFor()
  assert.equal(await page.evaluate(() => globalThis.__mailQa.syncs), 0)
  await page.getByRole('button', { name: 'Sync Gmail', exact: true }).click()
  await page.getByTestId('assistant-conversation').filter({ hasText: 'Synthetic Gmail thread' }).click()
  await page.getByRole('button', { name: 'Read Gmail conversation', exact: true }).click()
  await page.getByLabel('Conversation messages', { exact: true }).getByText('Synthetic full Gmail body revision 1.', { exact: true }).waitFor()
  checks.push('Explicitly selected Gmail appears in the unified Inbox; sync and full conversation reads are separate explicit actions')
  const draft = page.getByRole('textbox', { name: 'Reply draft', exact: true })
  await draft.fill('Keep this reply through reauthorization.')
  await page.evaluate(() => { globalThis.__mailQa.revision++; globalThis.__mailQa.synced = false })
  await page.getByRole('button', { name: 'Refresh inbox', exact: true }).click()
  await page.getByLabel('Conversation messages', { exact: true }).getByText('Synthetic full Gmail body revision 1.', { exact: true }).waitFor({ state: 'hidden' })
  await page.getByRole('button', { name: 'Sync Gmail', exact: true }).click()
  await page.getByTestId('assistant-conversation').filter({ hasText: 'Synthetic Gmail thread' }).click()
  assert.equal(await draft.inputValue(), 'Keep this reply through reauthorization.')
  checks.push('Reauthorization removes old Gmail content while preserving the same-account draft under its stable protected scope')
  await page.getByRole('button', { name: 'Ask TensorFold', exact: true }).click()
  await page.getByRole('button', { name: 'Start assistant task', exact: true }).click()
  await page.getByText('Synthetic local model unavailable', { exact: true }).waitFor()
  const handoff = await page.evaluate(() => globalThis.__mailQa.launches.at(-1))
  assert.equal(handoff.sourceThreadKey, 'gmail:account-a:2:thread-a')
  assert.ok(handoff.text.includes('Synthetic full Gmail body revision 2.'))
  assert.equal(await page.getByRole('textbox', { name: 'Assistant task draft', exact: true }).inputValue(), 'Keep this reply through reauthorization.')
  checks.push('Gmail assistant handoff reads the native thread, carries its privacy identity and retains the draft when launch fails')
  await page.getByRole('button', { name: 'Reply', exact: true }).click()
  await draft.fill('Exact reviewed fixture reply')
  await page.getByRole('button', { name: 'Review reply', exact: true }).click()
  await page.getByText('Gmail reply prepared for review. Nothing was sent.', { exact: true }).waitFor()
  assert.equal(await page.evaluate(() => globalThis.__mailQa.executions), 0)
  await page.getByRole('button', { name: 'Review pending message', exact: true }).click()
  const review = page.getByRole('dialog', { name: 'Review this message', exact: true })
  await review.getByText('reader@example.com', { exact: true }).waitFor()
  await review.getByText('casey@example.com', { exact: true }).waitFor()
  await review.getByRole('button', { name: 'Approve and send once', exact: true }).click()
  await page.getByText('Gmail accepted this message for sending.', { exact: true }).waitFor()
  assert.deepEqual(await page.evaluate(() => [globalThis.__mailQa.prepared, globalThis.__mailQa.executions, globalThis.__mailQa.genericReplies]), [1, 1, 0])
  checks.push('A Gmail reply stays pending until exact owner review; it uses the Google action route and reports acceptance rather than recipient delivery')
  await page.setViewportSize({ width: 720, height: 520 })
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
  await page.screenshot({ path: path.join(output, 'gmail-minimum-window.png') })
  assert.deepEqual(errors, [])
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify({ result: 'passed', scope: 'Actual renderer with synthetic APIs; no external messages or live accounts', checks, pageErrors: errors }, null, 2))
  console.log(checks)
} finally { await browser.close() }
