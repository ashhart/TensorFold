import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

const require = createRequire(import.meta.url), { chromium } = require(process.env.TENSORFOLD_QA_PLAYWRIGHT || 'playwright')
const output = process.env.TENSORFOLD_QA_OUTPUT || path.resolve('release/quick-ask-qa')
const architecture = process.env.TENSORFOLD_QA_ARCH || 'arm64'
assert.ok(['arm64', 'x64'].includes(architecture))
await mkdir(output, { recursive: true })
const browser = await chromium.connectOverCDP(process.env.TENSORFOLD_QA_CDP || 'http://127.0.0.1:19967')
try {
  const context = browser.contexts()[0], main = context.pages().find(page => page.url().includes('index.html'))
  assert.ok(main, 'The isolated packaged GUI has a main window')
  main.setDefaultTimeout(20_000)
  await main.waitForFunction(() => !!window.tensorfoldQuickAskLauncher)
  let existing = null
  for (const page of context.pages()) if (await page.evaluate(() => !!window.tensorfoldQuickAsk).catch(() => false)) existing = page
  const opened = existing ? Promise.resolve(existing) : context.waitForEvent('page')
  await main.evaluate(() => window.tensorfoldQuickAskLauncher.open())
  const quick = await opened; quick.setDefaultTimeout(20_000)
  await quick.getByRole('textbox', { name: 'Question', exact: true }).waitFor()
  assert.equal(await quick.evaluate(() => !!window.tensorfoldStudio || !!window.vontra), false)
  const snapshot = await quick.evaluate(() => window.tensorfoldQuickAsk.snapshot())
  assert.equal(snapshot.platform, 'linux')
  await quick.getByRole('textbox', { name: 'Question', exact: true }).fill('Native Linux draft recovery')
  await quick.getByRole('button', { name: 'Hide Ask TensorFold', exact: true }).click()
  await main.evaluate(() => window.tensorfoldQuickAskLauncher.open())
  assert.equal(await quick.getByRole('textbox', { name: 'Question', exact: true }).inputValue(), 'Native Linux draft recovery')
  const bound = await quick.evaluate(async () => { const api = window.tensorfoldQuickAsk, state = await api.snapshot(); await api.configure({ ...state.preferences, enabled: true, shortcut: 'Control+Alt+F19' }); return api.snapshot() })
  assert.equal(bound.shortcutError, null)
  assert.equal(bound.preferences.enabled, true)
  await quick.screenshot({ path: path.join(output, 'ask-tensorfold-linux-native.png'), animations: 'disabled', timeout: 60_000 })
  await quick.evaluate(async () => { const api = window.tensorfoldQuickAsk, state = await api.snapshot(); await api.configure({ ...state.preferences, enabled: false }) })
  await writeFile(path.join(output, 'linux-native-journey.json'), JSON.stringify({ platform: snapshot.platform, architecture, url: quick.url(), checks: ['Installed Debian package GUI', 'Restricted preload excludes full-app bridge', 'Native quick window opens, hides and restores draft', 'X11 global shortcut registration accepted'], qualification: `Ubuntu 24.04 ${architecture} under local QEMU and Xvfb. Kernel sandbox evidence is recorded separately. Wayland desktop consent and physical focus behaviour remain unqualified.` }, null, 2))
  console.log('Packaged Linux native Quick Ask GUI and global shortcut checks passed.')
} finally { await browser.close() }
