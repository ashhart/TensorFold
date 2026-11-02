import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { launchFixtureBrowser, openFixturePage, openFixtureThread, waitForPaint } from './browser-session.mjs'

const url = process.env.TENSORFOLD_QA_URL || 'http://127.0.0.1:5173/'
const output = path.resolve(process.env.TENSORFOLD_QA_OUTPUT || 'qa-workspace')
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser()
const checks = [], screenshots = [], errors = []
const pass = (name, evidence) => { checks.push({ name, evidence, passed: true }); console.log(`PASS ${name}`) }
let failure
async function capture(page, filename) {
  await waitForPaint(page)
  await page.screenshot({ path: path.join(output, filename) })
  screenshots.push({ filename, viewport: page.viewportSize() })
}
try {
  const fixture = await openFixturePage(browser, url)
  const { page } = fixture
  await page.evaluate(() => {
    const state = globalThis.__store.getState()
    state.setFullWorkspaceLayout('workspace')
    state.setView({ name: 'home' })
  })
  const composer = page.getByRole('textbox', { name: 'Message composer' })
  await composer.waitFor()
  const projectTrigger = page.getByRole('button', { name: 'Choose task project', exact: true })
  await projectTrigger.focus()
  await page.keyboard.press('ArrowDown')
  await page.getByRole('textbox', { name: 'Search projects' }).waitFor()
  assert.equal(await page.getByRole('textbox', { name: 'Search projects' }).evaluate(element => element === document.activeElement), true)
  await page.keyboard.press('Escape')
  assert.equal(await projectTrigger.evaluate(element => element === document.activeElement), true)
  pass('Keyboard project menu', 'ArrowDown opens and focuses search; Escape returns focus to the trigger.')
  await projectTrigger.click()
  await page.getByRole('textbox', { name: 'Search projects' }).fill('Zora')
  await page.getByRole('button', { name: 'Zora-Agent', exact: true }).last().click()
  await page.waitForFunction(() => globalThis.__store.getState().homeComposerWorkspace?.projectId === 'proj-demo')
  const draft = 'QA fixture: improve the selected project without changing another workspace.'
  await composer.fill(draft)
  await page.getByRole('button', { name: 'Attach', exact: true }).click()
  const chooseFile = page.waitForEvent('filechooser')
  await page.getByRole('button', { name: 'Attach document or image', exact: true }).click()
  await (await chooseFile).setFiles({ name: 'qa-context.txt', mimeType: 'text/plain', buffer: Buffer.from('Synthetic context for the renderer acceptance journey.\n') })
  await page.getByText('qa-context.txt', { exact: true }).waitFor()
  await page.getByRole('button', { name: 'Models', exact: true }).click()
  await page.getByTestId('local-runtime').waitFor()
  await page.getByRole('button', { name: /^New task/ }).first().click()
  await composer.waitFor()
  assert.equal(await composer.inputValue(), draft)
  await page.getByText('qa-context.txt', { exact: true }).waitFor()
  const selected = await page.evaluate(() => globalThis.__store.getState().homeComposerWorkspace)
  assert.equal(selected.projectId, 'proj-demo')
  assert.equal(selected.cwd, '/Users/example/Projects/zora-agent')
  pass('Draft, attachment and project retention', 'Typed draft, text attachment and chosen project/folder survive Models and return.')

  await page.getByRole('button', { name: /^Choose model, current/ }).click()
  const modelDialog = page.getByRole('dialog', { name: 'Choose a model', exact: true })
  await modelDialog.getByRole('textbox', { name: 'Search models' }).fill('Qwen3.5-2B')
  await modelDialog.locator('button[data-model-choice]:enabled').first().click()
  await page.waitForFunction(() => globalThis.__store.getState().settings.activeProviderId === 'tensorfold')
  const chosenModel = await page.evaluate(() => globalThis.__store.getState().settings.activeModel)
  await page.evaluate(() => {
    globalThis.__qaLaunch = null
    globalThis.__store.setState({ launch: async args => {
      globalThis.__qaLaunch = args
      globalThis.__store.getState().setNotice({ kind: 'error', message: 'Synthetic QA dispatch failure; no native task was started.' })
      return { ok: false, error: 'Synthetic QA failure' }
    } })
  })
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await page.waitForFunction(() => globalThis.__qaLaunch !== null)
  const submitted = await page.evaluate(() => globalThis.__qaLaunch)
  assert.equal(submitted.projectId, 'proj-demo')
  assert.equal(submitted.cwd, selected.cwd)
  assert.equal(submitted.providerId, 'tensorfold')
  assert.equal(submitted.model, chosenModel)
  assert.equal(submitted.attachments[0].name, 'qa-context.txt')
  assert.equal(await composer.inputValue(), draft)
  pass('Selected project/model dispatch and failure retention', 'Actual composer dispatches the chosen scope/model/attachment; an injected failed launch preserves the draft. Native execution is not exercised.')
  await capture(page, 'workspace-selected.png')
  await page.evaluate(() => globalThis.__store.getState().setNotice(null))

  await openFixtureThread(page, 'Async/await, simply')
  const chatDraft = 'A saved follow-up for this conversation.'
  await composer.fill(chatDraft)
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.getByRole('button', { name: 'Copy', exact: true }).first().click()
  const copied = await page.evaluate(() => navigator.clipboard.readText())
  assert.ok(copied.length > 20)
  pass('Conversation copy', 'A production transcript Copy action writes nonempty response/code text to the isolated browser clipboard.')
  const tools = page.getByRole('button', { name: 'More conversation tools' })
  await tools.focus()
  await page.keyboard.press('Enter')
  await page.getByRole('button', { name: 'Local helpers', exact: true }).waitFor()
  await page.waitForFunction(() => document.activeElement?.textContent?.includes('Local helpers'))
  await page.keyboard.press('ArrowDown')
  assert.equal(await page.getByRole('button', { name: 'Export readable text', exact: true }).evaluate(element => element === document.activeElement), true)
  await page.keyboard.press('Escape')
  assert.equal(await tools.evaluate(element => element === document.activeElement), true)
  pass('Secondary tool menu keyboard navigation', 'Enter opens, ArrowDown moves to export, Escape restores focus; actions retain existing guards.')
  await page.getByRole('button', { name: 'Models', exact: true }).click()
  await page.getByTestId('local-runtime').waitFor()
  await openFixtureThread(page, 'Async/await, simply')
  assert.equal(await composer.inputValue(), chatDraft)
  pass('Conversation draft retention', 'A thread draft survives switching to Models and returning.')
  await capture(page, 'conversation-actions.png')

  await openFixtureThread(page, 'Add dark mode toggle')
  await page.getByRole('button', { name: /Review changes/ }).first().click()
  await page.getByRole('complementary', { name: 'Work pane' }).waitFor()
  assert.equal(await page.getByRole('searchbox', { name: 'Filter changed files' }).isVisible(), true)
  pass('Task to review', 'Review changes opens the production Changes pane from the task context bar.')
  await page.evaluate(() => globalThis.__store.getState().closeWorkPane())
  const activity = page.getByRole('button', { name: /^Work activity:/ }).first()
  await activity.click()
  await page.getByText('Needs you', { exact: true }).last().waitFor()
  await page.keyboard.press('Escape')
  assert.equal(await activity.evaluate(element => element === document.activeElement), true)
  pass('Needs you navigation entry', 'Activity opens its actionable grouped queue; Escape returns focus without granting an approval.')

  for (const preset of ['tensorfold', 'navy', 'glass']) {
    for (const theme of ['light', 'dark']) {
      await page.evaluate(async ({ preset, theme }) => {
        const { PRESETS } = await import('/shared/theme-presets.ts')
        const state = globalThis.__store.getState()
        await state.saveSettings({ ...state.settings, preset, theme, light: { ...PRESETS[preset].light }, dark: { ...PRESETS[preset].dark } })
      }, { preset, theme })
      await waitForPaint(page)
      const dimensions = await page.evaluate(() => ({ width: document.body.scrollWidth, viewport: innerWidth }))
      assert.ok(dimensions.width <= dimensions.viewport)
      await capture(page, `theme-${preset}-${theme}.png`)
    }
  }
  pass('Built-in theme sweep', 'Production task renderer captured in TensorFold, Navy and Glass, light and dark; no body horizontal overflow. Native vibrancy is not exercised.')
  await page.evaluate(async () => {
    const { PRESETS } = await import('/shared/theme-presets.ts')
    const state = globalThis.__store.getState()
    await state.saveSettings({ ...state.settings, theme: 'light', preset: 'tensorfold', light: { ...PRESETS.tensorfold.light }, dark: { ...PRESETS.tensorfold.dark } }, { silent: true })
  })
  for (const size of [{ width: 1024, height: 768 }, { width: 720, height: 520 }]) {
    await page.setViewportSize(size)
    await capture(page, `task-${size.width}.png`)
    const stopOrSend = page.getByRole('button', { name: 'Send', exact: true })
    const box = await stopOrSend.boundingBox()
    assert.ok(box && box.x >= 0 && box.y >= 0 && box.x + box.width <= size.width && box.y + box.height <= size.height)
    await page.evaluate(() => globalThis.__store.getState().setView({ name: 'home' }))
    await composer.waitFor()
    await capture(page, `home-${size.width}.png`)
    await page.getByRole('button', { name: 'Send', exact: true }).scrollIntoViewIfNeeded()
    assert.ok(await page.getByRole('button', { name: 'Send', exact: true }).isVisible())
    await openFixtureThread(page, 'Add dark mode toggle')
  }
  pass('Supported small windows', 'Task Send remains in view at 1024×768 and the declared minimum 720×520; Home is scrollable and its Send action remains reachable.')
  const motion = await page.evaluate(() => [...document.querySelectorAll('.animate-spin, .animate-pulse')].map(element => getComputedStyle(element).animationName))
  assert.ok(motion.every(value => value === 'none'))
  pass('Reduced motion', 'Spin/pulse indicators stop animating when reduced motion is requested; status text remains.')
  errors.push(...fixture.errors)
  assert.deepEqual(errors, [])
} catch (error) {
  failure = String(error?.stack || error)
  throw error
} finally {
  await browser.close()
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify({
    fixtureData: true, backend: 'Repository browser-demo backend; launch failure injected at renderer dispatch boundary.',
    nativeVerified: false, capturedAt: new Date().toISOString(), checks, screenshots, pageErrors: errors,
    ...(failure ? { failure } : {}),
  }, null, 2) + '\n')
}
