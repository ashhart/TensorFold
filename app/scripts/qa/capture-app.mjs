import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { launchFixtureBrowser, openFixturePage, openFixtureThread, waitForPaint } from './browser-session.mjs'

const options = Object.fromEntries(process.argv.slice(2).map(value => value.replace(/^--/, '').split(/=(.*)/s).slice(0, 2)))
const baseUrl = options.url || 'http://127.0.0.1:5173/'
const output = path.resolve(options.output || 'qa-screenshots')
const label = options.label || 'candidate'
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser()
const captures = []
const errors = []

async function capture(page, name, scenario) {
  await waitForPaint(page)
  const filename = `${name}.png`
  await page.screenshot({ path: path.join(output, filename) })
  captures.push({ filename, scenario, viewport: page.viewportSize(), theme: await page.evaluate(() => document.documentElement.classList.contains('dark') ? 'dark' : 'light') })
}

try {
  const first = await openFixturePage(browser, baseUrl, { onboarding: true })
  await first.page.getByRole('heading', { name: 'How will you use TensorFold?' }).waitFor()
  await capture(first.page, '01-first-launch', 'Fresh browser profile; existing demo model/provider data; no native onboarding result claimed.')
  await first.page.getByRole('button', { name: /Continue with/ }).click()
  await first.page.getByRole('heading', { name: 'Choose a model for this Mac' }).waitFor()
  await capture(first.page, '02-model-setup', 'Onboarding model selection in the production renderer with browser-demo backend.')
  errors.push(...first.errors)
  await first.page.close()

  const workspace = await openFixturePage(browser, baseUrl)
  const { page } = workspace
  await page.evaluate(() => {
    const state = globalThis.__store.getState()
    state.setFullWorkspaceLayout('workspace')
    state.setView({ name: 'home' })
  })
  await page.getByRole('textbox', { name: 'Message composer' }).waitFor()
  await capture(page, '03-new-task', 'New task with an empty draft; fixture provider and projects.')
  await openFixtureThread(page, 'Async/await, simply')
  await capture(page, '04-conversation', 'Existing synthetic chat; production transcript, composer and history controls.')
  await openFixtureThread(page, 'Add dark mode toggle')
  await capture(page, '05-task-workspace', 'Existing synthetic project task with recorded changes; no real repository mutation.')
  await openFixtureThread(page, 'Harden the billing webhook')
  await capture(page, '06-active-work', 'Synthetic running task; real activity presentation and Stop/steer controls.')
  await openFixtureThread(page, 'Rotate the session signing keys')
  await capture(page, '07-needs-approval', 'Synthetic pending approval; no approval is granted by this capture.')
  await openFixtureThread(page, 'Upgrade to Vite 7')
  await capture(page, '08-failed-task', 'Synthetic failed task; production recovery presentation.')
  await openFixtureThread(page, 'Add dark mode toggle', 'changes')
  await page.getByRole('complementary', { name: 'Work pane' }).waitFor()
  await capture(page, '09-change-review', 'Production Changes pane with fixture diffs; no apply/commit action.')
  await page.evaluate(() => globalThis.__store.getState().setView({ name: 'models' }))
  await page.getByTestId('local-runtime').waitFor()
  await capture(page, '10-models', 'Production Models view; demo engine snapshot, not GPU inference.')
  await page.evaluate(() => globalThis.__store.getState().setView({ name: 'assistant' }))
  await page.getByRole('heading', { name: /^(Your inbox|Assistant|Conversation)$/ }).waitFor()
  await page.waitForTimeout(250)
  await capture(page, '11-assistant', 'Production Assistant; existing synthetic connector inbox, no account connection or external sync.')

  await openFixtureThread(page, 'Add dark mode toggle')
  await page.evaluate(async () => {
    const state = globalThis.__store.getState()
    await state.saveSettings({ ...state.settings, theme: 'dark' })
  })
  await page.waitForFunction(() => document.documentElement.classList.contains('dark'))
  await capture(page, '12-task-dark', 'Same task and size using the existing dark theme.')
  await page.setViewportSize({ width: 1024, height: 768 })
  await capture(page, '13-task-small-dark', 'Same task at a smaller supported window size, with the pane closed.')
  await page.evaluate(() => globalThis.__store.getState().openWorkPane('changes'))
  await page.getByRole('complementary', { name: 'Work pane' }).waitFor()
  await capture(page, '14-review-small-dark', 'Changes pane at 1024×768 using its existing floating-pane behavior.')
  errors.push(...workspace.errors)
} finally {
  await browser.close()
  await writeFile(path.join(output, 'capture-manifest.json'), JSON.stringify({
    label, capturedAt: new Date().toISOString(), evidence: 'Actual application renderer with the repository browser-demo backend.',
    nativeVerified: false, fixtureData: true, captures, pageErrors: errors,
  }, null, 2) + '\n')
}
if (errors.length) throw new Error(`Application page errors: ${errors.join('; ')}`)
console.log(`Captured ${captures.length} ${label} application views without page errors.`)
