import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { createJiti } from 'jiti'
import { launchFixtureBrowser } from './browser-session.mjs'

const jiti = createJiti(import.meta.url)
const { PRESETS } = await jiti.import('../../electron/agent/theme-presets.ts')
const { quickAskDefaults } = await jiti.import('../../shared/quick-ask.ts')
const output = process.env.TENSORFOLD_QA_OUTPUT || path.resolve('release/quick-ask-qa')
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser()
const errors = [], results = []
async function tools(page) { const button = page.getByRole('button', { name: 'More options', exact: true }); if (await button.getAttribute('aria-expanded') !== 'true') await button.click() }
const image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5gAAAABJRU5ErkJggg=='
try {
  for (const platform of ['darwin', 'win32', 'linux']) {
    const page = await browser.newPage({ viewport: { width: 760, height: 660 }, reducedMotion: 'reduce' })
    page.setDefaultTimeout(15_000)
    page.on('pageerror', error => errors.push(`${platform}: ${error.message}`))
    await page.addInitScript(({ preferences, palette, platform, image }) => {
      const local = { id: 'local-route', providerId: 'fixture', accountId: null, model: 'Example local model', label: 'Example local model', destination: 'device', location: 'This computer', state: 'ready', vision: 'observed', binding: 'a'.repeat(64) }
      const cloud = { ...local, id: 'cloud-route', destination: 'cloud', location: 'Example provider', label: 'Example cloud model' }
      const draft = () => ({ revision: 0, text: '', attachments: [], routeId: null, presetId: 'ask', web: false })
      const snapshot = { platform, preferences, shortcutError: null, routes: [local, cloud], routeErrors: [], appearance: { dark: true, palette, uiPreferences: { locale: 'en', direction: 'auto', chatScale: 1, codeScale: 1, density: 'comfortable' } }, voice: { input: 'unavailable', output: 'browser', destination: 'System speech' }, session: { id: 'session', draft: draft(), route: null, prompt: null, turns: [] } }
      const listeners = new Set(), log = [], emit = event => listeners.forEach(listener => listener(structuredClone(event)))
      let controller = 0
      globalThis.__askFixture = { snapshot, log }
      globalThis.tensorfoldQuickAsk = {
        resize: async height => { log.push({ kind: 'resize', height }) },
        pickWindow: async () => { log.push({ kind: 'window' }); return { kind: 'attached', attachments: [{ id: 'window-11111111-1111-1111-1111-111111111111-image', name: 'Document · sample.txt', kind: 'image', mime: 'image/png', dataUrl: image }, { id: 'window-11111111-1111-1111-1111-111111111111-text', name: 'Document · sample.txt · visible text', kind: 'document', mime: 'text/plain', text: 'Visible selected-window sample' }] } },
        useWindow: async () => [],
        snapshot: async () => structuredClone(snapshot),
        saveDraft: async value => { if (snapshot.session.draft.revision !== value.revision) throw new Error('stale draft'); snapshot.session.draft = { ...value, revision: value.revision + 1 }; return structuredClone(snapshot.session.draft) },
        configure: async value => { if (value.shortcut === 'Control+X') throw new Error('This shortcut is already in use.'); snapshot.preferences = value; return value },
        recordShortcut: async () => {},
        send: async input => {
          const selected = snapshot.routes.find(route => route.id === snapshot.session.draft.routeId)
          if (selected.destination !== 'device' && input.approvedRouteId !== selected.id) throw new Error('destination not approved')
          log.push({ kind: 'send', draft: structuredClone(snapshot.session.draft), input })
          const current = ++controller, turn = { id: `turn-${current}`, text: snapshot.session.draft.text, attachments: structuredClone(snapshot.session.draft.attachments), answer: 'A streamed ', reasoning: '', status: 'running', tools: [], at: Date.now() }
          snapshot.session.route = selected; snapshot.session.turns.push(turn); emit({ kind: 'turn', sessionId: snapshot.session.id, turn })
          setTimeout(() => { if (current !== controller) return; turn.answer += 'answer with **useful detail**.'; turn.status = 'complete'; snapshot.session.draft = { ...draft(), revision: snapshot.session.draft.revision + 1, routeId: selected.id }; emit({ kind: 'turn', sessionId: snapshot.session.id, turn }); emit({ kind: 'refresh' }) }, 450)
          return { requestId: turn.id }
        },
        cancel: async () => { controller++; snapshot.session.turns.at(-1).status = 'cancelled'; emit({ kind: 'refresh' }) },
        reset: async () => { controller++; snapshot.session = { ...snapshot.session, id: `session-${controller}`, turns: [], route: null, prompt: null }; emit({ kind: 'refresh' }) },
        clipboard: async () => { log.push({ kind: 'clipboard' }); return [{ id: 'clip', name: 'Clipboard text', kind: 'document', text: 'Text deliberately attached by the user.', mime: 'text/plain', status: 'ready' }] },
        displays: async () => [{ id: 'display', label: 'Example display' }],
        capture: async () => { log.push({ kind: 'capture' }); return { id: 'screen', kind: 'image', name: 'Screen capture', mime: 'image/png', dataUrl: image, status: 'ready' } },
        loadModel: async () => {}, handoff: async () => { log.push({ kind: 'handoff', draft: structuredClone(snapshot.session.draft) }); return { threadId: 'saved-conversation' } },
        copy: async text => { log.push({ kind: 'copy', text }) }, openLink: async () => {}, hide: async () => { log.push({ kind: 'hide' }) }, acknowledgeFlush: async () => {}, speak: async () => ({ ok: false }),
        onEvent: listener => { listeners.add(listener); return () => listeners.delete(listener) },
      }
    }, { preferences: quickAskDefaults(), palette: PRESETS.tensorfold.dark, platform, image })
    await page.goto(process.env.TENSORFOLD_QA_URL || 'http://127.0.0.1:5184/', { waitUntil: 'domcontentloaded' })
    const question = page.getByRole('textbox', { name: 'Question', exact: true })
    await question.waitFor()
    await page.waitForFunction(() => getComputedStyle(document.getElementById('boot')).opacity === '0')
    assert.equal(await page.evaluate(() => globalThis.__askFixture.log.filter(item => item.kind !== 'resize').length), 0, 'Opening the window does not read context')
    await page.getByRole('combobox', { name: 'Model and destination' }).selectOption('local-route')
    if (platform === 'darwin') await page.screenshot({ path: path.join(output, 'ask-tensorfold-empty.png') })
    assert.ok(await page.locator('main.ask-compact').evaluate(element => element.getBoundingClientRect().height) <= 110)
    await page.getByRole('button', { name: 'Choose a window', exact: true }).click()
    await page.getByRole('button', { name: 'Document · sample.txt', exact: true }).click()
    await page.getByRole('dialog').getByText('Visible selected-window sample').waitFor()
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: 'Remove Document · sample.txt', exact: true }).click()
    assert.equal(await page.evaluate(() => globalThis.__askFixture.snapshot.session.draft.attachments.length), 0)
    await tools(page)
    await page.getByRole('button', { name: 'Attach clipboard text or image', exact: true }).click()
    await page.getByRole('button', { name: 'Clipboard text', exact: true }).click()
    await page.getByRole('dialog').getByText('Text deliberately attached by the user.').waitFor()
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: /Remove.*Clipboard text/ }).click()
    await question.fill('A question that can be stopped')
    await page.getByRole('button', { name: 'Ask', exact: true }).click()
    await page.getByRole('button', { name: 'Stop', exact: true }).click()
    await page.waitForFunction(() => globalThis.__askFixture.snapshot.session.turns.at(-1)?.status === 'cancelled')
    assert.equal(await question.inputValue(), 'A question that can be stopped')
    await tools(page); await page.getByRole('button', { name: 'New ask', exact: true }).click()
    await question.fill('Explain why preserving drafts matters')
    await question.press('Enter')
    await page.getByText('useful detail', { exact: true }).waitFor()
    await page.waitForFunction(() => globalThis.__askFixture.snapshot.session.turns.at(-1)?.status === 'complete')
    await page.waitForFunction(() => document.querySelector('textarea')?.value === '')
    assert.equal(await page.getByRole('combobox', { name: 'Model and destination' }).isDisabled(), true)
    await page.getByRole('button', { name: 'Copy answer', exact: true }).click()
    if (platform === 'darwin') await page.screenshot({ path: path.join(output, 'ask-tensorfold-answer.png') })
    await tools(page); await page.getByRole('button', { name: 'Settings and presets', exact: true }).click()
    const shortcut = page.getByRole('textbox', { name: 'Global shortcut', exact: true })
    await shortcut.focus(); await shortcut.press('Control+x')
    await page.getByRole('button', { name: 'Save settings', exact: true }).click()
    await page.getByRole('dialog').getByRole('alert').filter({ hasText: 'already in use' }).waitFor()
    await page.keyboard.press('Escape')
    await tools(page); await page.getByRole('button', { name: 'New ask', exact: true }).click()
    await page.getByRole('combobox', { name: 'Model and destination' }).selectOption('cloud-route')
    await question.fill('A question for the chosen provider')
    const sent = await page.evaluate(() => globalThis.__askFixture.log.filter(item => item.kind === 'send').length)
    await page.getByRole('button', { name: 'Ask', exact: true }).click()
    await page.getByRole('dialog', { name: 'Review before sending' }).waitFor()
    if (platform === 'darwin') await page.screenshot({ path: path.join(output, 'ask-tensorfold-destination-review.png') })
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    assert.equal(await page.evaluate(() => globalThis.__askFixture.log.filter(item => item.kind === 'send').length), sent)
    await page.getByRole('button', { name: 'Ask', exact: true }).click()
    await page.getByRole('button', { name: 'Send to this destination', exact: true }).click()
    await page.waitForFunction(() => globalThis.__askFixture.snapshot.session.turns.at(-1)?.status === 'complete')
    await question.fill('An unfinished follow-up to carry into TF')
    await page.getByRole('button', { name: 'Continue in TensorFold', exact: true }).click()
    assert.equal(await page.evaluate(() => globalThis.__askFixture.log.find(item => item.kind === 'handoff').draft.text), 'An unfinished follow-up to carry into TF')
    await page.setViewportSize({ width: 540, height: 420 })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    if (platform === 'darwin') {
      await page.setViewportSize({ width: 760, height: 660 })
      await page.evaluate(() => { globalThis.__askFixture.snapshot.appearance.uiPreferences.locale = 'pt' })
      await tools(page); await page.getByRole('button', { name: 'New ask', exact: true }).click()
      await page.getByRole('textbox', { name: 'Pergunta', exact: true }).waitFor()
      await page.getByPlaceholder('Com o que você gostaria de ajuda?').waitFor()
      await page.screenshot({ path: path.join(output, 'ask-tensorfold-portuguese.png') })
    }
    results.push({ platform, checks: 'startup, explicit clipboard, preview, cancellation, draft recovery, streaming, model pin, copy, shortcut conflict, destination consent, handoff draft, narrow layout' })
    await page.close()
  }
  assert.deepEqual(errors, [])
  await writeFile(path.join(output, 'browser-journey.json'), JSON.stringify({ results, errors }, null, 2))
  console.log(`Ask TensorFold browser journeys passed for ${results.length} platform configurations.`)
} finally { await browser.close() }
