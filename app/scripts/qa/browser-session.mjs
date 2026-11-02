import { createRequire } from 'node:module'
import path from 'node:path'

const require = createRequire(import.meta.url)

// This harness uses a fresh profile and the app's existing browser demo, never a signed-in browser.
export async function launchFixtureBrowser() {
  const modulePath = process.env.TENSORFOLD_QA_PLAYWRIGHT
    || (process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES ? path.join(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES, 'playwright') : 'playwright')
  const { chromium } = require(modulePath)
  return chromium.launch({
    headless: true,
    ...(process.env.TENSORFOLD_QA_CHROMIUM ? { executablePath: process.env.TENSORFOLD_QA_CHROMIUM } : {}),
    ...(process.env.TENSORFOLD_QA_NO_SANDBOX === '1' ? { args: ['--no-sandbox'] } : {}),
  })
}

export async function openFixturePage(browser, baseUrl, { onboarding = false, width = 1440, height = 1000 } = {}) {
  const page = await browser.newPage({ viewport: { width, height }, reducedMotion: 'reduce' })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  if (!onboarding) await page.addInitScript(() => localStorage.setItem('vontra.onboarded', '1'))
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' })
  await page.waitForFunction(() => globalThis.__store?.getState().ready && globalThis.__store.getState().settings, { timeout: 45_000 })
  return { page, errors }
}

export async function openFixtureThread(page, title, pane) {
  await page.evaluate(({ title, pane }) => {
    const state = globalThis.__store.getState()
    const thread = state.threads.find(item => item.title === title)
    if (!thread) throw new Error(`No built-in browser fixture named ${title}`)
    state.setFullWorkspaceLayout('workspace')
    state.openThread(thread.id)
    if (pane) globalThis.__store.getState().openWorkPane(pane)
    else globalThis.__store.getState().closeWorkPane()
  }, { title, pane })
  await page.getByRole('textbox', { name: 'Message composer' }).waitFor()
}

export async function waitForPaint(page) {
  await page.evaluate(async () => {
    await document.fonts.ready
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  })
}
