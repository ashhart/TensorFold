import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { launchFixtureBrowser, openFixturePage, openFixtureThread, waitForPaint } from './browser-session.mjs'

const output = path.resolve(process.argv[2] || path.join(os.tmpdir(), 'tensorfold-transcript-qa'))
const baseUrl = process.argv[3] || 'http://127.0.0.1:5173/'
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser()
const browserVersion = browser.version()
const captures = [], acceptance = [], trackedErrors = []
let failure

async function fixturePage(options) {
  const result = await openFixturePage(browser, baseUrl, options)
  result.page.setDefaultTimeout(15000)
  trackedErrors.push(result.errors)
  return result
}

async function capture(page, name, evidence) {
  await page.evaluate(() => {
    let label = document.getElementById('transcript-qa-label')
    if (!label) { label = document.createElement('div'); label.id = 'transcript-qa-label'; document.body.append(label) }
    label.textContent = 'Browser fixture · UI only · No task or approval executed'
    Object.assign(label.style, { position: 'fixed', bottom: '2px', right: '8px', zIndex: '9999', background: '#17202a', color: '#fff', padding: '3px 7px', borderRadius: '4px', fontSize: '10px', pointerEvents: 'none' })
  })
  await waitForPaint(page)
  const filename = `${name}.png`
  await page.screenshot({ path: path.join(output, filename), timeout: 20000 })
  captures.push({ filename, evidence, viewport: page.viewportSize() })
}

async function verifyConversationScroll() {
  const { page, errors } = await fixturePage()
  await page.evaluate(() => {
    const s = globalThis.__store.getState(); s.setFullWorkspaceLayout('workspace')
    const base = s.threads.find(t => t.title === 'Async/await, simply') || s.threads[0]
    const timeline = Array.from({ length: 250 }, (_, i) => [
      { id: `qa-u-${i}`, kind: 'user', text: `Reading-position fixture, turn ${i + 1}: explain this local design choice.`, createdAt: i * 2000 },
      { id: `qa-a-${i}`, kind: 'assistant', text: `Recorded response ${i + 1}. This synthetic message gives the reader a stable place to return to while other work continues.\n\nThe viewport should keep this paragraph in view after switching conversations.`, createdAt: i * 2000 + 1000 },
    ]).flat()
    const fixture = { ...base, id: 'qa-scroll', title: 'UI fixture — reading history', kind: 'chat', projectId: null, cwd: '', status: 'idle', timeline }
    globalThis.__store.setState({ threads: [...s.threads, fixture], status: { ...s.status, 'qa-scroll': 'idle' }, rightPanelHidden: true, view: { name: 'thread', id: 'qa-scroll' } })
  })
  const scroller = page.locator('[data-conversation-id="qa-scroll"]')
  await scroller.locator('[data-transcript-row]').last().waitFor()
  assert.equal(await scroller.locator('[data-transcript-row]').count(), 150)
  await scroller.evaluate(el => { el.scrollTop = 0 })
  await page.getByRole('button', { name: 'Jump to latest', exact: true }).waitFor()
  await page.getByRole('button', { name: 'Load 150 earlier rows', exact: true }).click()
  await page.waitForFunction(() => document.querySelector('[data-conversation-id="qa-scroll"]')?.querySelectorAll('[data-transcript-row]').length === 300)
  await scroller.evaluate(el => { el.scrollTop = 220 })
  const read = async () => scroller.evaluate(el => {
    const top = el.getBoundingClientRect().top
    const row = [...el.querySelectorAll('[data-transcript-row]')].find(node => node.getBoundingClientRect().bottom > top + 1)
    return { top: el.scrollTop, anchor: row?.dataset.transcriptRow, offset: row ? row.getBoundingClientRect().top - top : 0, count: el.querySelectorAll('[data-transcript-row]').length }
  })
  await page.waitForFunction(() => document.querySelector('[data-conversation-id="qa-scroll"]')?.scrollTop === 220)
  const before = await read()
  await capture(page, 'history-reading-before', 'Synthetic 500-row history with 150 explicitly loaded earlier rows.')
  await page.evaluate(() => { const s = globalThis.__store.getState(); s.openThread(s.threads.find(t => t.title === 'Add dark mode toggle').id) })
  await page.getByRole('button', { name: /Review changes/ }).waitFor()
  await page.evaluate(() => globalThis.__store.getState().openThread('qa-scroll'))
  await scroller.locator('[data-transcript-row]').last().waitFor()
  const switched = await read()
  assert.equal(switched.count, before.count); assert.equal(switched.anchor, before.anchor)
  assert.ok(Math.abs(switched.offset - before.offset) <= 1, JSON.stringify({ before, switched }))
  await page.evaluate(() => globalThis.__store.getState().setView({ name: 'home' }))
  await scroller.waitFor({ state: 'detached' })
  await page.evaluate(() => globalThis.__store.getState().openThread('qa-scroll'))
  await scroller.locator('[data-transcript-row]').last().waitFor()
  const remounted = await read()
  assert.equal(remounted.count, before.count); assert.equal(remounted.anchor, before.anchor)
  assert.ok(Math.abs(remounted.offset - before.offset) <= 1, JSON.stringify({ before, remounted }))
  await page.evaluate(() => globalThis.__store.setState(s => ({ status: { ...s.status, 'qa-scroll': 'streaming' }, threads: s.threads.map(t => t.id !== 'qa-scroll' ? t : { ...t, timeline: t.timeline.map((item, index) => index === t.timeline.length - 1 ? { ...item, text: item.text + '\n\nNew streaming content below the reader.' } : item) }) })))
  const streamed = await read(); assert.ok(Math.abs(streamed.top - before.top) <= 1, JSON.stringify({ before, streamed }))
  const composer = page.getByRole('textbox', { name: 'Message composer', exact: true })
  const draftText = 'Draft smoke: keep every character while a synthetic response streams.'
  await page.evaluate(() => {
    globalThis.__qaStreamingTicks = 0
    globalThis.__qaStreamingTimer = setInterval(() => {
      globalThis.__qaStreamingTicks += 1
      globalThis.__store.setState(s => ({ threads: s.threads.map(t => t.id !== 'qa-scroll' ? t : {
        ...t, timeline: t.timeline.map((item, index) => index === t.timeline.length - 1 ? { ...item, text: `${item.text}.` } : item),
      }) }))
    }, 40)
  })
  try {
    await composer.pressSequentially(draftText, { delay: 5, timeout: 60000 })
  } finally {
    await page.evaluate(() => clearInterval(globalThis.__qaStreamingTimer))
  }
  const streamingUpdates = await page.evaluate(() => globalThis.__qaStreamingTicks)
  assert.ok(streamingUpdates >= 2, 'Typing overlaps repeated synthetic streaming updates')
  assert.equal(await composer.inputValue(), draftText)
  const typedPosition = await read()
  assert.equal(typedPosition.anchor, before.anchor)
  assert.ok(Math.abs(typedPosition.offset - before.offset) <= 1, JSON.stringify({ before, typedPosition }))
  await page.evaluate(() => globalThis.__store.setState(s => ({
    status: { ...s.status, 'qa-scroll': 'awaiting' }, approvals: { ...s.approvals, 'qa-scroll': [{ itemId: 'qa-approval', callId: 'qa-approval' }] },
    threads: s.threads.map(t => t.id !== 'qa-scroll' ? t : { ...t, timeline: [...t.timeline, { id: 'qa-approval', approvalId: 'qa-approval', kind: 'tool', tool: 'run_command', title: 'Run', command: 'npm test', state: 'awaiting', output: 'Synthetic UI fixture: approval is pending.' }] }),
  })))
  await page.getByRole('button', { name: 'Allow', exact: true }).waitFor()
  const approval = await read(); assert.ok(Math.abs(approval.top - before.top) <= 1, JSON.stringify({ before, approval }))
  await capture(page, 'history-reading-after-approval', 'The same reading anchor after view switches, streaming, and a synthetic pending approval; no approval submitted.')
  await page.getByRole('button', { name: 'Jump to latest', exact: true }).click()
  const bottom = await scroller.evaluate(el => el.scrollHeight - el.scrollTop - el.clientHeight)
  assert.ok(bottom <= 1, String(bottom))
  await page.evaluate(() => globalThis.__store.setState(s => ({ threads: s.threads.map(t => t.id !== 'qa-scroll' ? t : { ...t, timeline: [...t.timeline, { id: 'qa-follow-check', kind: 'assistant', text: 'A final synthetic message checks following after Jump to latest.' }] }) })))
  await page.getByText('A final synthetic message checks following after Jump to latest.', { exact: true }).waitFor()
  await page.waitForFunction(() => { const el = document.querySelector('[data-conversation-id="qa-scroll"]'); return el.scrollHeight - el.scrollTop - el.clientHeight <= 1 })
  assert.equal(await composer.inputValue(), draftText)
  assert.deepEqual(errors, [])
  acceptance.push({ journey: 'Conversation reading position', status: 'passed', before, switched, remounted, streamed, approval, loadedRows: before.count, followingResumed: true,
    typingSmoke: { charactersTyped: draftText.length, syntheticUpdateIntervalMs: 40, typingDelayMs: 5, streamingUpdates, exactDraftRetained: true, typedPosition, messageSubmitted: false, performanceBenchmark: false } })
  await page.close()
}

async function verifyChanges() {
  const { page, errors } = await fixturePage()
  await openFixtureThread(page, 'Add dark mode toggle', 'changes')
  const pane = page.getByRole('complementary', { name: 'Work pane' })
  const filter = pane.getByRole('searchbox', { name: 'Filter changed files' })
  await filter.fill('App.tsx')
  await pane.getByRole('region', { name: /Diff for.*src\/App\.tsx/ }).waitFor()
  assert.equal(await pane.locator('[data-path]').count(), 1)
  const wrap = pane.getByRole('button', { name: 'Wrap lines', exact: true })
  assert.equal(await wrap.getAttribute('aria-pressed'), 'true')
  const width = await pane.getByRole('region').evaluate(element => ({ client: element.clientWidth, scroll: element.scrollWidth }))
  assert.ok(width.scroll <= width.client + 1, JSON.stringify(width))
  await wrap.press('Enter'); assert.equal(await wrap.getAttribute('aria-pressed'), 'false')
  await wrap.press('Enter')
  await pane.getByRole('button', { name: 'Collapse all', exact: true }).click()
  assert.equal(await pane.getByRole('region').count(), 0)
  await pane.getByRole('button', { name: 'Expand all', exact: true }).click()
  await page.mouse.move(5, 5)
  await capture(page, 'changes-filter-after', 'Current working-tree diff with keyboard-tested wrapping and filename filtering.')
  await page.evaluate(async () => {
    const { api } = await import('/src/lib/ipc.ts')
    globalThis.__qaGitDiff = api.gitDiff
    api.gitDiff = async () => { throw new Error('Synthetic UI fixture: git unavailable') }
  })
  await pane.getByRole('button', { name: 'Refresh', exact: true }).click()
  await pane.getByRole('alert').waitFor()
  assert.match(await pane.getByRole('alert').innerText(), /Recorded edits remain available/)
  assert.equal(await pane.getByRole('region').count(), 1)
  await capture(page, 'changes-retry-after', 'An explicitly failed fixture git read retains recorded edits and exposes Retry.')
  await page.evaluate(async () => { const { api } = await import('/src/lib/ipc.ts'); api.gitDiff = globalThis.__qaGitDiff })
  await pane.getByRole('button', { name: 'Retry', exact: true }).click()
  await pane.getByRole('alert').waitFor({ state: 'detached' })
  await filter.fill('no-matching-file')
  await pane.getByRole('button', { name: 'Clear filter', exact: true }).click()
  assert.equal(await pane.locator('[data-path]').count(), 4)
  await page.setViewportSize({ width: 1024, height: 768 })
  await page.getByRole('button', { name: /Review changes/ }).click()
  await filter.fill('App.tsx')
  await capture(page, 'changes-small-after', 'The same actual Changes component at 1024×768.')
  await pane.getByRole('button', { name: 'Close pane', exact: true }).click()
  const measurements = []
  await page.setViewportSize({ width: 720, height: 520 })
  for (const collapsed of [false, true]) {
    await page.evaluate(value => globalThis.__store.setState({ sidebarCollapsed: value }), collapsed)
    await page.getByRole('button', { name: /Review changes/ }).click()
    await filter.fill('App.tsx')
    await pane.getByRole('region', { name: /Diff for.*App\.tsx/ }).waitFor()
    const geometry = await pane.evaluate(element => ({ x: element.getBoundingClientRect().x, right: element.getBoundingClientRect().right, width: element.clientWidth, viewport: innerWidth, document: document.documentElement.scrollWidth }))
    assert.ok(geometry.x >= 0 && geometry.right <= 721 && geometry.document <= geometry.viewport, JSON.stringify(geometry))
    const diff = await pane.getByRole('region').evaluate(element => ({ client: element.clientWidth, scroll: element.scrollWidth }))
    assert.ok(diff.scroll <= diff.client + 1, JSON.stringify(diff))
    const tabLabels = await pane.getByRole('tab').evaluateAll(tabs => tabs.map(tab => {
      const label = [...tab.childNodes].find(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim())
      const range = document.createRange(); range.selectNodeContents(label)
      return { label: label.textContent, lines: range.getClientRects().length }
    }))
    assert.ok(tabLabels.every(tab => tab.lines === 1), JSON.stringify(tabLabels))
    const tabs = pane.getByRole('tab')
    await tabs.first().focus()
    for (let index = 0; index < await tabs.count(); index += 1) {
      assert.equal(await tabs.nth(index).evaluate(tab => tab === document.activeElement), true)
      await tabs.nth(index).press('Tab')
    }
    assert.equal(await pane.getByRole('button', { name: 'Close pane', exact: true }).evaluate(button => button === document.activeElement), true)
    await wrap.press('Enter'); await wrap.press('Enter')
    await page.mouse.move(5, 5)
    await capture(page, `review-minimum-sidebar-${collapsed ? 'closed' : 'open'}`, 'Actual desktop minimum 720×520; sidebar state named in filename; keyboard-tested controls remain within the window.')
    measurements.push({ sidebar: collapsed ? 'closed' : 'open', ...geometry, diff, tabLabels, keyboardReachesClose: true })
    await pane.getByRole('button', { name: 'Close pane', exact: true }).click()
    await page.getByRole('button', { name: /Worked.*3 steps.*Finished/ }).last().click()
    await page.getByRole('button', { name: /Edit src\/App.tsx Completed/ }).click()
    await page.getByRole('button', { name: 'Copy diff', exact: true }).waitFor()
    await capture(page, `activity-minimum-sidebar-${collapsed ? 'closed' : 'open'}`, 'Expanded transcript diff at the actual desktop minimum; no task executed.')
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await page.getByRole('button', { name: /Edit src\/App.tsx Completed/ }).click()
    await page.getByRole('button', { name: /Worked.*3 steps.*Finished/ }).last().click()
  }
  assert.deepEqual(errors, [])
  acceptance.push({ journey: 'Changes review', status: 'passed', filter: true, keyboardWrap: true, collapseExpand: true, failedReadFallback: true, retryRecovered: true, minimumSize: measurements })
  await page.close()
}

async function verifyToolOutput() {
  const { page, errors } = await fixturePage()
  await page.evaluate(() => {
    const state = globalThis.__store.getState(); state.setFullWorkspaceLayout('workspace')
    const base = state.threads.find(thread => thread.title === 'Add dark mode toggle')
    const start = Date.now() - 1400
    const receipt = { id: 'qa-receipt', runId: 'qa-run', threadId: 'qa-output', verdict: 'failed', summary: 'Synthetic UI fixture: test process exited with code 1.', changedFiles: [], commands: ['npm test -- --reporter=verbose'], tests: [], approvals: 0, diffs: 0, artifacts: [], errors: 1, eventReceipts: [], gaps: [], warnings: [], startedAt: start, finishedAt: Date.now(), needsReview: true }
    const fixture = { ...base, id: 'qa-output', title: 'UI fixture — long test output', projectId: null, status: 'error', timeline: [
      { id: 'qa-request', kind: 'user', text: 'UI fixture: inspect a failed test with long diagnostic output.', createdAt: start },
      { id: 'qa-command', kind: 'tool', tool: 'run_command', title: 'Run tests', command: 'npm test -- --reporter=verbose', state: 'error', meta: 'exit 1 · 1.4s', output: Array.from({ length: 64 }, (_, index) => `${String(index + 1).padStart(2, '0')}  Synthetic diagnostic: expected the saved theme preference to survive a second browser session. ${'detail/'.repeat(index % 4 === 0 ? 18 : 0)}`).join('\n'), createdAt: start + 100 },
      { id: 'qa-receipt-item', kind: 'notice', title: 'Run receipt', text: receipt.summary, runReceipt: receipt, createdAt: receipt.finishedAt },
    ] }
    globalThis.__store.setState({ threads: [...state.threads, fixture], status: { ...state.status, 'qa-output': 'error' }, rightPanelHidden: true, view: { name: 'thread', id: 'qa-output' } })
  })
  await page.getByRole('button', { name: 'Show recorded activity', exact: true }).click()
  const outputRegion = page.getByLabel('Tool output', { exact: true })
  await outputRegion.waitFor()
  assert.match(await page.getByRole('button', { name: /Run tests.*Failed/ }).innerText(), /Failed/)
  const bounds = await outputRegion.evaluate(element => ({ client: element.clientWidth, scroll: element.scrollWidth }))
  assert.ok(bounds.scroll <= bounds.client + 1, JSON.stringify(bounds))
  const beforeHeight = await outputRegion.locator('..').evaluate(element => element.clientHeight)
  await page.getByRole('button', { name: 'Expand output', exact: true }).click()
  const afterHeight = await outputRegion.locator('..').evaluate(element => element.clientHeight)
  assert.ok(afterHeight > beforeHeight)
  await page.getByRole('button', { name: 'Wrap lines', exact: true }).press('Enter')
  assert.match(await outputRegion.getAttribute('class'), /whitespace-pre(?:\s|$)/)
  await page.getByRole('button', { name: 'Wrap lines', exact: true }).press('Enter')
  await page.evaluate(async () => {
    const { api } = await import('/src/lib/ipc.ts')
    api.writeClipboard = async text => { globalThis.__qaCopied = text; return { ok: true } }
  })
  await page.getByRole('button', { name: 'Copy output', exact: true }).click()
  assert.match(await page.evaluate(() => globalThis.__qaCopied), /^\$ npm test -- --reporter=verbose\n01/)
  await page.getByRole('button', { name: 'Collapse output', exact: true }).click()
  await page.mouse.move(5, 5)
  await capture(page, 'tool-output-after', 'Synthetic failed receipt and 64 diagnostic lines; output controls and clipboard payload checked with an inert stub.')
  await page.setViewportSize({ width: 1024, height: 768 })
  await capture(page, 'tool-output-small-after', 'The same long-output fixture at 1024×768; no command executed.')
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
  const ids = await page.locator('[id]').evaluateAll(elements => elements.map(element => element.id))
  assert.equal(ids.length, new Set(ids).size, 'Disclosure IDs remain unique')
  assert.deepEqual(errors, [])
  acceptance.push({ journey: 'Long failed tool output', status: 'passed', automaticFailureDisclosure: true, keyboardWrap: true, expandHeights: { beforeHeight, afterHeight }, exactClipboardPayload: true, clipboardStubbed: true, uniqueDisclosureIds: true })
  await page.close()
}

async function verifyLiveStates() {
  const { page, errors } = await fixturePage()
  await openFixtureThread(page, 'Rotate the session signing keys')
  await page.getByRole('button', { name: /Working.*Waiting for approval.*Needs you/ }).waitFor()
  await capture(page, 'waiting-after', 'Existing pending-approval demo; no approval was submitted.')
  await openFixtureThread(page, 'Harden the billing webhook')
  await page.getByRole('button', { name: /Working.*Running/ }).waitFor()
  const animations = await page.locator('.animate-spin').evaluateAll(elements => elements.map(element => getComputedStyle(element).animationName))
  assert.ok(animations.length > 0, 'The running fixture contains status spinners')
  assert.ok(animations.every(value => value === 'none'), JSON.stringify(animations))
  await capture(page, 'running-after', 'Existing running demo with reduced-motion preferences; status text remains available.')
  assert.deepEqual(errors, [])
  acceptance.push({ journey: 'Waiting, running and reduced motion', status: 'passed', animations, approvalsSubmitted: false })
  await page.close()
}

try {
  await verifyConversationScroll()
  await verifyChanges()
  await verifyToolOutput()
  await verifyLiveStates()
} catch (error) {
  failure = error instanceof Error ? error.stack : String(error)
  for (const context of browser.contexts()) for (const page of context.pages()) {
    await capture(page, `failure-${captures.length}`, 'Diagnostic capture after a failed assertion; not a passed acceptance case.').catch(() => {})
  }
} finally {
  await browser.close()
  if (trackedErrors.flat().length) failure ??= `Renderer errors: ${trackedErrors.flat().join('; ')}`
  await writeFile(path.join(output, 'transcript-review-receipt.json'), JSON.stringify({
    status: failure ? 'failed' : 'passed', capturedAt: new Date().toISOString(), browserVersion, baseUrl,
    rendererRevision: process.env.TENSORFOLD_QA_REVISION || undefined,
    renderer: 'Actual TensorFold Vite renderer', data: 'Built-in browser demo and explicitly synthetic UI fixtures',
    nativeVerified: false, tasksExecuted: false, approvalsSubmitted: false,
    acceptance, captures, pageErrors: trackedErrors.flat(), failure,
  }, null, 2) + '\n')
}
if (failure) throw new Error(failure)
assert.deepEqual(trackedErrors.flat(), [])
console.log(`Transcript/review GUI: ${acceptance.length} journeys passed; ${captures.length} labeled screenshots; no page errors.`)
