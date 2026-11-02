import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { launchFixtureBrowser, openFixturePage, waitForPaint } from './browser-session.mjs'

const output = path.resolve(process.argv[2] || '../qa-assistant')
const baseUrl = process.argv[3] || 'http://127.0.0.1:5173/'
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser()
const acceptance = [], captures = [], pageErrors = []
const record = (journey, evidence) => acceptance.push({ journey, status: 'passed', evidence })
let failure
const debug = []

async function capture(page, name, state) {
  await page.evaluate(() => {
    let label = document.getElementById('assistant-qa-label')
    if (!label) { label = document.createElement('div'); label.id = 'assistant-qa-label'; document.body.append(label) }
    label.textContent = 'Browser fixture · Synthetic inbox · No external messages sent'
    Object.assign(label.style, { position: 'fixed', bottom: '2px', right: '8px', zIndex: '9999', background: '#17202a', color: '#fff', padding: '3px 7px', borderRadius: '4px', fontSize: '10px', pointerEvents: 'none' })
  })
  await waitForPaint(page)
  await page.screenshot({ path: path.join(output, `${name}.png`) })
  captures.push({ filename: `${name}.png`, state, viewport: page.viewportSize() })
}

async function fixturePage() {
  const view = await openFixturePage(browser, baseUrl)
  pageErrors.push(view.errors)
  await view.page.evaluate(async () => {
    const { api } = await import('/src/lib/ipc.ts')
    const state = { snapshot: structuredClone(await api.getAssistantSnapshot()), mode: 'ready', reads: 0, approvalMode: 'error', attempts: 0, deliveries: 0, reviewedDigest: null, rejectRead: null }
    const source = globalThis.__store.getState().settings.email.sources[0]
    if (!source) throw new Error('The synthetic inbox needs its configured email source')
    for (const event of state.snapshot.events) if (event.connectorId === 'email') event.sources = [{ id: `fixture:${event.id}`, connectorId: 'email', sourceId: source.id }]
    globalThis.__assistantQa = state
    api.getAssistantSnapshot = async () => {
      state.reads++
      if (state.mode === 'error') throw new Error('Fixture connection is unavailable')
      if (state.mode === 'pending') return new Promise((_resolve, reject) => { state.rejectRead = reject })
      return structuredClone(state.snapshot)
    }
    api.resolveConnectorApproval = async (id, decision, digest) => {
      state.attempts++
      if (state.approvalMode === 'error') throw new Error('Fixture dispatch is unavailable; the draft remains pending')
      const event = state.snapshot.events.find(item => item.id === id)
      if (!event || !event.requiresApproval || digest !== event.reviewDigest) throw new Error('Fixture review identity did not match')
      event.requiresApproval = false
      event.approvalStatus = decision === 'approve' ? 'approved' : 'denied'
      event.delivery = { status: 'delivered', transport: 'fixture-only', deliveredAt: Date.now() }
      state.reviewedDigest = digest
      state.deliveries++
      return structuredClone(event)
    }
  })
  return view.page
}

async function openInbox(page) {
  await page.evaluate(() => globalThis.__store.getState().setView({ name: 'assistant' }))
  await page.getByRole('heading', { name: 'Your inbox', exact: true }).waitFor()
}
const replyText = 'Please keep this unfinished reply through every inbox refresh.'

try {
  const page = await fixturePage()
  await openInbox(page)
  const inbox = page.getByRole('complementary', { name: 'Assistant conversations' })
  const reply = () => page.getByRole('textbox', { name: 'Reply draft', exact: true })
  await reply().waitFor()
  assert.equal(await page.getByRole('alert').count(), 0, 'A session-only draft is not a save failure')
  await capture(page, '02-inbox-after', 'Live renderer, existing synthetic conversations; neutral session-only draft status.')
  await inbox.getByRole('combobox', { name: 'Filter inbox by source' }).selectOption('email')
  assert.equal(await page.getByTestId('assistant-conversation').count(), 1)
  await inbox.getByRole('combobox', { name: 'Filter inbox by source' }).selectOption('all')
  await inbox.getByRole('textbox', { name: 'Search inbox conversations' }).fill('launch')
  assert.ok(await page.getByTestId('assistant-conversation').count() >= 2)
  await inbox.getByRole('textbox', { name: 'Search inbox conversations' }).fill('unmatched fixture query')
  await page.getByText('No matching conversations', { exact: true }).waitFor()
  await inbox.getByRole('button', { name: 'Clear conversation search' }).click()
  const first = page.getByTestId('assistant-conversation').first()
  await first.click()
  await first.press('ArrowDown')
  assert.equal(await page.getByTestId('assistant-conversation').nth(1).getAttribute('aria-current'), 'true')
  await page.keyboard.press('ArrowUp')
  assert.equal(await first.getAttribute('aria-current'), 'true')
  record('Find and open a conversation', 'Source filter, sender/subject search, no-match recovery, and ArrowUp/ArrowDown selection execute in the production renderer.')

  await reply().fill(replyText)
  await page.evaluate(() => { globalThis.__assistantQa.mode = 'error' })
  await page.getByRole('button', { name: 'Refresh inbox', exact: true }).click()
  await page.getByText('Inbox check failed. Your conversation and draft are still here.', { exact: true }).waitFor()
  assert.equal(await reply().inputValue(), replyText)
  assert.equal(await first.getAttribute('aria-current'), 'true')
  await capture(page, '03-retained-draft-error', 'Real refresh hook failure, retained inbox selection and unfinished reply; explicit retry.')
  await page.evaluate(() => { globalThis.__assistantQa.mode = 'ready' })
  await page.getByRole('button', { name: 'Retry now', exact: true }).click()
  await page.waitForFunction(() => document.querySelector('[data-testid="assistant-freshness"]')?.textContent?.startsWith('Checked'))
  assert.equal(await reply().inputValue(), replyText)
  assert.equal(await first.getAttribute('aria-current'), 'true')
  record('Recover a failed inbox refresh', 'The actual hook retains the last snapshot, selected thread, and exact draft text, then clears its failure after Retry now.')

  await reply().focus()
  await page.evaluate(() => {
    const state = globalThis.__assistantQa
    const original = state.snapshot.events.find(event => event.connectorId === 'email')
    state.snapshot.events.push({ ...original, id: 'qa-live-inbound', createdAt: Date.now(), summary: 'Fixture follow-up arrived while you were writing.', payload: { text: 'Fixture follow-up arrived while you were writing.' } })
    state.snapshot.attention = []
  })
  await page.getByRole('region', { name: 'Conversation messages' }).getByText('Fixture follow-up arrived while you were writing.', { exact: true }).waitFor({ timeout: 22_000 })
  assert.equal(await reply().inputValue(), replyText)
  assert.equal(await reply().evaluate(element => element === document.activeElement), true)
  record('Receive live inbox updates while writing', 'The unmodified 15-second hook polls, renders a new message in the same conversation, and preserves draft text and textarea focus.')

  page.once('dialog', dialog => dialog.dismiss())
  await page.getByRole('button', { name: 'Prepare briefing', exact: true }).click()
  assert.equal(await reply().inputValue(), replyText)
  page.once('dialog', dialog => dialog.accept())
  await page.getByRole('button', { name: 'Prepare briefing', exact: true }).click()
  await page.getByRole('textbox', { name: 'Assistant task draft', exact: true }).waitFor()
  assert.equal(await page.getByRole('button', { name: 'Ask TensorFold', exact: true }).getAttribute('aria-pressed'), 'true')
  await capture(page, '04-explicit-task-mode', 'Accepted draft replacement selects Ask TensorFold; task start is distinct from an external reply.')
  await page.getByRole('button', { name: 'Reply', exact: true }).click()
  await reply().fill(replyText)
  record('Prepare work without accidentally replying', 'Cancelling replacement leaves the reply unchanged; accepting switches to assistant-task intent and shows its own action.')

  await page.evaluate(() => {
    const state = globalThis.__assistantQa
    const original = state.snapshot.events.find(event => event.connectorId === 'email')
    for (let index = 0; index < 3; index++) state.snapshot.events.unshift({
      ...original, id: `qa-approval-${index}`, createdAt: Date.now() + 3 - index,
      threadKey: index ? `email:qa-review-${index}` : original.threadKey,
      actor: 'TensorFold', title: index ? `Synthetic pending reply ${index}` : 'Renewal reply ready for review',
      action: 'reply_requested', direction: 'outbound', requiresApproval: true, approvalStatus: 'pending', priority: 'high',
      summary: 'A proposed reply is ready for your review.', reviewDigest: '1'.repeat(64),
      outbound: { version: 1, connectorId: 'email', accountId: 'synthetic-account', accountRevision: 'fixture', recipient: 'maya@example.test', subject: 'Re: Renewal', body: 'Thanks for the questions. I will confirm the rollout date before making a commitment.\n\nThis exact text is a synthetic review fixture.', threadKey: index ? `email:qa-review-${index}` : original.threadKey },
    })
    state.snapshot.attention = []
  })
  await page.getByRole('button', { name: 'Refresh inbox', exact: true }).click()
  const approvals = page.getByRole('region', { name: 'Pending approvals', exact: true })
  await approvals.getByRole('button', { name: 'Review message', exact: true }).first().waitFor()
  assert.equal(await approvals.getByRole('button', { name: 'Review message', exact: true }).count(), 2)
  await approvals.getByRole('button', { name: /Show all approvals/ }).click()
  assert.equal(await approvals.getByRole('button', { name: 'Review message', exact: true }).count(), 3)
  await approvals.getByRole('button', { name: 'Show fewer approvals', exact: true }).click()
  const review = approvals.getByRole('button', { name: 'Review message', exact: true }).first()
  await review.click()
  const dialog = page.getByRole('dialog', { name: 'Review this message', exact: true })
  await dialog.waitFor()
  assert.match(await dialog.innerText(), /maya@example.test/)
  assert.match(await dialog.getByLabel('Exact outgoing message').innerText(), /confirm the rollout date/)
  assert.equal(await page.evaluate(() => globalThis.__assistantQa.attempts), 0)
  await page.keyboard.press('Tab')
  assert.equal(await dialog.getByLabel('Exact outgoing message').evaluate(element => element === document.activeElement), true)
  await page.keyboard.press('Shift+Tab')
  assert.equal(await dialog.getByRole('button', { name: 'Approve and send once' }).evaluate(element => element === document.activeElement), true)
  await page.keyboard.press('Escape')
  await dialog.waitFor({ state: 'detached' })
  assert.equal(await review.evaluate(element => element === document.activeElement), true)
  await review.click()
  await capture(page, '05-exact-approval-review', 'Review shows exact account, recipient, subject, and body; no dispatch has occurred.')
  await dialog.getByRole('button', { name: 'Approve and send once' }).click()
  await dialog.getByRole('alert').waitFor()
  assert.equal(await page.evaluate(() => globalThis.__assistantQa.deliveries), 0)
  await capture(page, '06-approval-retry', 'Synthetic dispatch error retains the exact review payload and exposes retry through the same explicit approval.')
  await page.evaluate(() => { globalThis.__assistantQa.approvalMode = 'ready' })
  await dialog.getByRole('button', { name: 'Approve and send once' }).click()
  await dialog.waitFor({ state: 'detached' })
  assert.equal(await page.evaluate(() => globalThis.__assistantQa.deliveries), 1)
  assert.equal(await page.evaluate(() => globalThis.__assistantQa.reviewedDigest), '1'.repeat(64))
  assert.equal(await reply().inputValue(), replyText)
  record('Review and recover an approval', 'Queue bounds/expansion, exact payload, no dispatch on open, Tab trap, Escape focus return, failure retention, and one explicit fixture delivery with the exact digest all pass.')

  await page.evaluate(async () => { const state = globalThis.__store.getState(); await state.saveSettings({ ...state.settings, theme: 'dark' }) })
  await page.waitForFunction(() => document.documentElement.classList.contains('dark'))
  await page.getByText('Settings saved', { exact: true }).waitFor({ state: 'hidden', timeout: 12_000 })
  await page.setViewportSize({ width: 1024, height: 768 })
  await capture(page, '07-inbox-small-dark', 'Assistant inbox and actionable reply at 1024×768 in the actual dark theme.')
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
  record('Use a smaller dark-theme window', '1024×768 capture retains source selection, reading pane, draft, and the submit control without horizontal document overflow.')

  await page.setViewportSize({ width: 720, height: 520 })
  for (const collapsed of [false, true]) {
    await page.evaluate((collapsed) => { const state = globalThis.__store.getState(); if (state.sidebarCollapsed !== collapsed) state.toggleSidebar() }, collapsed)
    await page.getByRole('combobox', { name: 'Choose a conversation', exact: true }).scrollIntoViewIfNeeded()
    assert.equal(await page.getByRole('combobox', { name: 'Choose a conversation', exact: true }).isVisible(), true)
    await reply().scrollIntoViewIfNeeded()
    const send = page.getByRole('button', { name: 'Send reply', exact: true })
    await send.scrollIntoViewIfNeeded()
    const box = await send.boundingBox()
    assert.ok(box && box.x >= 0 && box.y >= 0 && box.x + box.width <= 720 && box.y + box.height <= 520)
    assert.equal(await send.evaluate(element => { const box = element.getBoundingClientRect(); return element.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2)) }), true)
    assert.equal(await reply().inputValue(), replyText)
    assert.ok((await page.getByRole('region', { name: 'Conversation messages' }).boundingBox())?.height >= 160)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await capture(page, `11-minimum-${collapsed ? 'collapsed' : 'expanded'}`, 'Native minimum 720×520; responsive conversation selector and reply action remain reachable, with ordinary vertical scrolling when needed.')
  }
  record('Use the native minimum window size', 'At 720×520 with the sidebar expanded and collapsed, the responsive conversation selector and reply action remain reachable, drafts persist, and no horizontal document overflow occurs.')

  const recovery = await fixturePage()
  await recovery.evaluate(() => { globalThis.__assistantQa.mode = 'pending' })
  await openInbox(recovery)
  await recovery.getByRole('heading', { name: 'Loading your inbox', exact: true }).waitFor()
  await capture(recovery, '08-initial-loading', 'Unresolved real snapshot read shows a loading state, not a false empty inbox.')
  await recovery.evaluate(() => { const fixture = globalThis.__assistantQa; fixture.mode = 'error'; fixture.rejectRead(new Error('Fixture initial inbox read failed')) })
  await recovery.getByRole('heading', { name: 'The inbox could not load', exact: true }).waitFor()
  await capture(recovery, '09-initial-error', 'Failed initial read has its own retry action; no frozen disabled draft is presented.')
  await recovery.evaluate(() => { globalThis.__assistantQa.mode = 'ready' })
  await recovery.getByRole('button', { name: 'Retry inbox', exact: true }).click()
  await recovery.getByRole('textbox', { name: 'Reply draft', exact: true }).waitFor()
  record('Recover loading and initial-read failure', 'A deferred read renders loading, rejection renders diagnostic + retry, and a successful retry restores the actionable conversation.')
  await recovery.evaluate(() => { const fixture = globalThis.__assistantQa; fixture.snapshot.events = []; fixture.snapshot.attention = [] })
  await recovery.getByRole('button', { name: 'Refresh inbox', exact: true }).click()
  await recovery.getByRole('heading', { name: 'No conversations in this view', exact: true }).waitFor()
  await capture(recovery, '10-empty-inbox', 'A successful empty inbox offers source setup and preparing a task, distinct from failure or loading.')
  await recovery.getByRole('button', { name: 'Manage sources', exact: true }).last().click()
  await recovery.getByRole('heading', { name: 'Connectors', exact: true }).waitFor()
  record('Recover an empty inbox', 'A successful empty result links to the existing Connectors settings; no false success or account connection is manufactured.')
} catch (error) {
  failure = error instanceof Error ? error.stack : String(error)
  for (const context of browser.contexts()) for (const page of context.pages()) {
    debug.push(await page.evaluate(() => ({ view: globalThis.__store?.getState().view, mode: globalThis.__assistantQa?.mode, reads: globalThis.__assistantQa?.reads, visibility: document.visibilityState, selected: document.querySelector('[data-testid="assistant-conversation"][aria-current="true"]')?.textContent, text: document.body.innerText.slice(-10000) })).catch(() => null))
    await capture(page, `failure-${debug.length}`, 'Diagnostic capture from a failed assertion; not an acceptance receipt.').catch(() => {})
  }
} finally {
  await browser.close()
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify({ capturedAt: new Date().toISOString(), renderer: 'Actual TensorFold Vite renderer', data: 'Synthetic browser fixtures only', nativeVerified: false, externalMessagesSent: false, acceptance, captures, pageErrors: pageErrors.flat(), failure, debug }, null, 2) + '\n')
}
if (failure) throw new Error(failure)
assert.deepEqual(pageErrors.flat(), [])
console.log(`Assistant GUI: ${acceptance.length} journeys passed; ${captures.length} labeled screenshots; no page errors.`)
