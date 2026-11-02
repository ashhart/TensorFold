import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { launchFixtureBrowser, openFixturePage, waitForPaint } from './browser-session.mjs'

const output = path.resolve(process.argv[2] || path.join(os.tmpdir(), 'tensorfold-typing-qa'))
const urls = process.argv.slice(3)
if (!urls.length) urls.push('http://127.0.0.1:5173/')
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser()
const results = []
const draftText = 'Draft smoke: keep every character while a synthetic response streams.'

function elapsedMetrics(before, after) {
  const start = Object.fromEntries(before.metrics.map(metric => [metric.name, metric.value]))
  return Object.fromEntries(after.metrics.filter(metric => /^(TaskDuration|ScriptDuration|LayoutDuration|RecalcStyleDuration)$/.test(metric.name))
    .map(metric => [metric.name, Math.round((metric.value - (start[metric.name] || 0)) * 1000)]))
}

function profileSummary(profile) {
  const modules = new Map()
  for (const node of profile.nodes) {
    const source = node.callFrame.url.split('?')[0] || '(browser or idle)'
    modules.set(source, (modules.get(source) || 0) + (node.hitCount || 0))
  }
  return [...modules].map(([source, samples]) => ({ source, samples })).sort((a, b) => b.samples - a.samples).slice(0, 12)
}

async function verifyMessageControls(page, streaming) {
  const message = page.locator('#ti-typing-a-249')
  const expected = await page.evaluate(() => globalThis.__store.getState().threads.find(t => t.id === 'qa-typing').timeline.at(-1).text)
  assert.equal(await message.locator('p').evaluateAll(nodes => nodes.map(node => node.textContent).join('\n\n')), expected)
  assert.equal(await page.getByRole('button', { name: 'Edit user message', exact: true }).last().isDisabled(), streaming)
  assert.equal(await message.getByRole('button', { name: 'Quote assistant message', exact: true }).count(), streaming ? 0 : 1)
  assert.equal(await page.locator('#ti-typing-a-100').getByRole('button', { name: 'Quote assistant message', exact: true }).count(), 1)
  await page.evaluate(async () => {
    const { api } = await import('/src/lib/ipc.ts')
    api.patchThread = async id => globalThis.__store.getState().threads.find(t => t.id === id)
    api.prepareMessageQuote = async input => {
      globalThis.__typingQuoteRequest = input
      const source = globalThis.__store.getState().threads.find(t => t.id === input.threadId).timeline.find(item => item.id === input.itemId)
      const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source.text))
      const sourceRevision = [...new Uint8Array(hash)].map(byte => byte.toString(16).padStart(2, '0')).join('')
      const text = input.selection ?? source.text
      return { id: 'typing-quote', sourceItemId: input.itemId, sourceRevision, sourceText: text, text }
    }
    globalThis.__store.setState(s => ({ status: { ...s.status, 'qa-typing': 'idle' } }))
  })
  await message.getByRole('button', { name: 'Quote assistant message', exact: true }).waitFor()
  assert.equal(await page.getByRole('button', { name: 'Edit user message', exact: true }).last().isEnabled(), true)
  await message.getByTitle('Pin this message', { exact: true }).click()
  await message.getByTitle('Unpin this message', { exact: true }).click()
  await message.getByTitle('Pin this message', { exact: true }).waitFor()
  await message.getByRole('button', { name: 'Quote assistant message', exact: true }).click()
  const quoted = page.getByLabel('Quoted passages', { exact: true })
  await quoted.waitFor()
  assert.equal(await quoted.locator('blockquote').innerText(), expected)
  assert.deepEqual(await page.evaluate(() => globalThis.__typingQuoteRequest), { threadId: 'qa-typing', itemId: 'typing-a-249' })
  assert.equal(await page.getByRole('textbox', { name: 'Message composer', exact: true }).inputValue(), draftText)
  await page.getByRole('button', { name: 'Remove quoted passage', exact: true }).click()
  return { liveAnswerUpdated: true, busyEditGuard: true, historyQuoteAvailable: true, exactOwnerQuote: true, pinToggle: true, quoteAndPinApisStubbed: true }
}

async function measure(baseUrl, streaming) {
  const { page, errors } = await openFixturePage(browser, baseUrl)
  const client = await page.context().newCDPSession(page)
  await page.evaluate(() => {
    const s = globalThis.__store.getState()
    s.setFullWorkspaceLayout('workspace')
    const base = s.threads.find(t => t.title === 'Async/await, simply') || s.threads[0]
    const timeline = Array.from({ length: 250 }, (_, i) => [
      { id: `typing-u-${i}`, kind: 'user', text: `Reading-position fixture, turn ${i + 1}: explain this local design choice.`, createdAt: i * 2000 },
      { id: `typing-a-${i}`, kind: 'assistant', text: `Recorded response ${i + 1}. This synthetic message gives the reader a stable place to return to while other work continues.\n\nThe viewport should keep this paragraph in view after switching conversations.`, createdAt: i * 2000 + 1000 },
    ]).flat()
    const fixture = { ...base, id: 'qa-typing', title: 'UI fixture — typing during updates', kind: 'chat', projectId: null, cwd: '', status: 'idle', timeline }
    globalThis.__store.setState({ threads: [...s.threads, fixture], status: { ...s.status, 'qa-typing': 'idle' }, rightPanelHidden: true, view: { name: 'thread', id: 'qa-typing' } })
  })
  await page.locator('#ti-typing-a-249').waitFor()
  await page.locator('#ti-typing-a-249').evaluate(node => {
    const scroller = node.closest('.overflow-y-auto')
    scroller.dataset.typingQa = 'true'
    scroller.scrollTop = 0
  })
  const scroller = page.locator('[data-typing-qa="true"]')
  await page.getByRole('button', { name: 'Load 150 earlier rows', exact: true }).click()
  await page.locator('#ti-typing-a-100').waitFor()
  assert.equal(await scroller.locator('[id^="ti-typing-"]').count(), 300)
  await scroller.evaluate(element => { element.scrollTop = 220 })
  await waitForPaint(page)
  const composer = page.getByRole('textbox', { name: 'Message composer', exact: true })
  await composer.focus()
  await page.evaluate(streaming => {
    const state = { input: [], longTasks: [], updates: 0, started: performance.now(), observer: null, timer: null }
    globalThis.__typingQa = state
    document.querySelector('textarea[aria-label="Message composer"]').addEventListener('input', event => {
      state.input.push({ at: performance.now() - state.started, value: event.target.value })
    })
    state.observer = new PerformanceObserver(list => state.longTasks.push(...list.getEntries().map(entry => entry.duration)))
    state.observer.observe({ type: 'longtask' })
    if (streaming) {
      globalThis.__store.setState(s => ({ status: { ...s.status, 'qa-typing': 'streaming' } }))
      state.timer = setInterval(() => {
        state.updates += 1
        globalThis.__store.setState(s => ({ threads: s.threads.map(t => t.id !== 'qa-typing' ? t : {
          ...t, timeline: t.timeline.map((item, index) => index === t.timeline.length - 1 ? { ...item, text: `${item.text}.` } : item),
        }) }))
      }, 40)
    }
  }, streaming)
  await client.send('Performance.enable')
  await client.send('Profiler.enable')
  await client.send('Profiler.start')
  const beforeMetrics = await client.send('Performance.getMetrics')
  const started = Date.now()
  let failure
  try {
    await composer.pressSequentially(draftText, { delay: 5, timeout: 60000 })
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error)
  } finally {
    await page.evaluate(() => clearInterval(globalThis.__typingQa.timer))
  }
  const typingElapsedMs = Date.now() - started
  const afterMetrics = await client.send('Performance.getMetrics')
  const { profile } = await client.send('Profiler.stop')
  await waitForPaint(page)
  const evidence = await page.evaluate(() => {
    const state = globalThis.__typingQa
    state.observer.disconnect()
    const scroller = document.querySelector('[data-typing-qa="true"]')
    const top = scroller.getBoundingClientRect().top
    const row = [...scroller.querySelectorAll('[id^="ti-typing-"]')].find(element => element.getBoundingClientRect().bottom > top + 1)
    return { updates: state.updates, input: state.input, longTasks: state.longTasks,
      reading: { top: scroller.scrollTop, anchor: row?.id, offset: row ? row.getBoundingClientRect().top - top : null },
      finalDraft: document.querySelector('textarea[aria-label="Message composer"]').value }
  })
  const allCharactersRetained = evidence.finalDraft === draftText
    && evidence.input.length === draftText.length
    && evidence.input.every((event, index) => event.value === draftText.slice(0, index + 1))
  let messageControls
  try {
    messageControls = await verifyMessageControls(page, streaming)
  } catch (error) {
    failure ??= error instanceof Error ? error.message : String(error)
    const diagnostic = await page.evaluate(() => ({ view: globalThis.__store.getState().view,
      notice: globalThis.__store.getState().notice, request: globalThis.__typingQuoteRequest,
      draft: document.querySelector('textarea[aria-label="Message composer"]')?.value }))
    messageControls = { failure, diagnostic }
    await page.screenshot({ path: path.join(output, `typing-controls-${new URL(baseUrl).port}-${streaming ? 'streaming' : 'idle'}.png`) })
  }
  const result = { baseUrl, streaming, viewport: page.viewportSize(), timelineRows: 500, mountedMessageRows: 300,
    syntheticUpdateIntervalMs: streaming ? 40 : null, typingDelayMs: 5, actionTimeoutMs: 60000, typingElapsedMs,
    allCharactersRetained, ...evidence, messageControls, browserCpuMs: elapsedMetrics(beforeMetrics, afterMetrics), sampledModules: profileSummary(profile),
    pageErrors: errors, failure, performanceBenchmark: false, messageSubmitted: false }
  results.push(result)
  console.log(JSON.stringify({ baseUrl, streaming, typingElapsedMs, updates: evidence.updates, allCharactersRetained,
    longTasks: evidence.longTasks.length, longestTaskMs: Math.max(0, ...evidence.longTasks), browserCpuMs: result.browserCpuMs, failure }))
  await page.close()
}

try {
  for (const streaming of [false, true]) for (const baseUrl of urls) await measure(baseUrl, streaming)
} finally {
  await writeFile(path.join(output, 'transcript-typing-receipt.json'), JSON.stringify({
    capturedAt: new Date().toISOString(), browserVersion: browser.version(),
    purpose: 'Functional typing and exploratory CPU diagnosis on the shared development renderer; not a performance benchmark.',
    tasksExecuted: false, approvalsSubmitted: false, results,
  }, null, 2) + '\n')
  await browser.close()
}
assert.ok(results.every(result => result.allCharactersRetained && !result.failure && !result.pageErrors.length))
