// Run against a local Web build with playwright-core and its browser binaries installed:
// node tests/copy-focus.cjs http://127.0.0.1:8080
const assert = require('node:assert/strict')
const { chromium, webkit } = require('playwright-core')

const fixture = {
  id: 'copy-example',
  title: 'Copy test',
  messages: [
    { role: 'user', content: 'Show two examples.' },
    { role: 'assistant', content: '```text\nApple\n```\n\n```text\nOrange\n```' },
  ],
  settings: {},
}

async function checkCopy(browser, mode, base) {
  const page = await browser.newPage()
  try {
    await page.route('**/api/**', route => {
      const request = route.request()
      if (new URL(request.url()).pathname === '/api/conversations/' + fixture.id) {
        return route.fulfill({ json: fixture })
      }
      return request.method() === 'GET' ? route.continue() : route.fulfill({ json: {} })
    })
    await page.addInitScript(mode => {
      window.writes = []
      const clipboard = {
        async writeText(text) {
          if (mode === 'denied') throw new Error('Clipboard denied')
          window.writes.push(text)
        },
      }
      Object.defineProperty(navigator, 'clipboard', {
        value: ['modern', 'denied'].includes(mode) ? clipboard : undefined,
      })
      document.execCommand = command => {
        if (command !== 'copy') throw new Error(command)
        if (mode === 'failure') return false
        const field = document.activeElement
        window.writes.push(field.value.slice(field.selectionStart, field.selectionEnd))
        return true
      }
    }, mode)

    await page.goto(new URL('/#chat/' + fixture.id, base).href)
    const buttons = page.locator('pre button')
    for (const [index, expected] of ['Apple\n', 'Orange\n'].entries()) {
      const button = buttons.nth(index)
      await button.focus()
      const trigger = await button.elementHandle()
      const name = await button.getAttribute('aria-label')
      const focused = () => trigger.evaluate(el => el.isConnected && el === document.activeElement)
      await page.keyboard.press(index === 0 ? 'Enter' : 'Space')
      // The toast rerenders Chat. Check both feedback and its subsequent reset.
      if (mode === 'failure') {
        await page.getByText('The browser refused the clipboard', { exact: true }).first().waitFor()
        assert.equal(await button.innerText(), 'Copy')
      } else {
        await button.getByText('Copied', { exact: true }).waitFor()
        assert.equal(await page.evaluate(() => window.writes.at(-1)), expected)
      }
      assert.ok(await focused(), 'Feedback should preserve the original focused button')
      assert.equal(await button.getAttribute('aria-label'), name)
      if (index === 1) await buttons.first().focus()
      await page.waitForTimeout(3100)
      assert.equal(await button.innerText(), 'Copy')
      assert.ok(index === 0 ? await focused() : await buttons.first().evaluate(el => el === document.activeElement))
    }
  } finally {
    await page.close()
  }
}

async function main() {
  const base = new URL(process.argv[2])
  for (const engine of [chromium, webkit]) {
    const executablePath = engine === chromium ? process.env.CHROMIUM_EXECUTABLE_PATH : undefined
    const browser = await engine.launch({ executablePath })
    try {
      for (const mode of ['modern', 'missing', 'denied', 'failure']) {
        await checkCopy(browser, mode, base)
        console.log(engine.name(), mode, 'passed')
      }
    } finally {
      await browser.close()
    }
  }
}

main().catch(error => { console.error(error); process.exitCode = 1 })
