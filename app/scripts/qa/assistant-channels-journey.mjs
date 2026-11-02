import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { launchFixtureBrowser, openFixturePage } from './browser-session.mjs'

const output = path.resolve(process.argv[2]), base = process.argv[3] || 'http://127.0.0.1:5187/'
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser(), checks = []
try {
  for (const platform of ['linux', 'win32', 'darwin']) {
    const { page, errors } = await openFixturePage(browser, base)
    await page.evaluate(async platform => {
      const { api } = await import('/src/lib/ipc.ts')
      const qa = globalThis.__channelsQa = { platform, begins: [], polls: [], cancels: [], links: [], sources: [], externalWrites: 0 }
      api.platform = platform
      const get = api.getSettings
      api.getSettings = async () => { const settings = await get(); return { ...settings, email: { ...settings.email, sources: [...settings.email.sources, ...qa.sources] } } }
      api.microsoftMailBegin = async (clientId, allowSend) => {
        qa.begins.push({ clientId, allowSend })
        return { id: 'synthetic-sign-in', userCode: 'SYNTHETIC', verificationUri: 'https://microsoft.com/devicelogin', expiresAt: Date.now() + 900_000, intervalMs: 5000 }
      }
      api.microsoftMailPoll = async id => {
        qa.polls.push(id)
        qa.sources = [{ id: 'ms-synthetic', label: 'Outlook · synthetic@example.com', kind: 'microsoft_graph', user: 'synthetic@example.com', microsoftRevision: 1, enabled: true, path: 'https://graph.microsoft.com/v1.0/me/messages', createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }]
        return { status: 'connected' }
      }
      api.microsoftMailCancel = async id => { qa.cancels.push(id) }
      api.openExternal = async url => { qa.links.push(url) }
      api.replyToThread = async () => { qa.externalWrites++; throw new Error('External writes forbidden in this fixture') }
      globalThis.__store.getState().setView({ name: 'assistant' })
    }, platform)
    await page.getByRole('tab', { name: 'Connections', exact: true }).click()
    const outlook = page.getByRole('button', { name: /Outlook \/ Microsoft 365/ })
    await outlook.waitFor()
    for (const name of ['WhatsApp', 'iMessage']) await page.getByRole('button', { name: new RegExp(`^${name} `) }).waitFor()
    await outlook.click()
    await page.getByRole('heading', { name: 'Outlook / Microsoft 365', exact: true }).first().waitFor()
    await page.getByRole('textbox', { name: 'Application client ID', exact: true }).fill('11111111-2222-3333-4444-555555555555')
    await page.getByRole('button', { name: 'Connect Microsoft account', exact: true }).click()
    await page.getByText('SYNTHETIC', { exact: true }).waitFor()
    assert.deepEqual(await page.evaluate(() => globalThis.__channelsQa.begins.map(item => item.allowSend)), [false])
    await page.getByRole('button', { name: 'Open Microsoft sign-in', exact: true }).click()
    assert.deepEqual(await page.evaluate(() => globalThis.__channelsQa.links), ['https://microsoft.com/devicelogin'])
    await page.getByRole('button', { name: 'Check sign-in', exact: true }).click()
    await page.getByText('Microsoft account connected. Sync email to read the inbox.', { exact: true }).waitFor()
    await page.getByText('Outlook · synthetic@example.com', { exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Disconnect Microsoft account', exact: true }).count(), 1)
    checks.push(`${platform} browser fixture: Assistant routes to Microsoft sign-in, defaults to read-only consent and displays the selected account`)
    await page.evaluate(async () => {
      const { requestConnectorSetup } = await import('/src/lib/store.ts')
      requestConnectorSetup('imessage')
    })
    const enabled = page.getByRole('switch', { name: 'iMessage connector', exact: true })
    if (await enabled.getAttribute('aria-checked') !== 'true') await enabled.click()
    await page.getByRole('button', { name: 'BlueBubbles', exact: true }).click()
    await page.getByRole('textbox', { name: 'Server URL', exact: true }).fill('https://messages.example.com')
    await page.getByRole('textbox', { name: 'Server URL', exact: true }).press('Tab')
    await page.getByText(/BlueBubbles works from Windows, Linux or Mac/).waitFor()
    await page.getByRole('textbox', { name: 'Server password', exact: true }).waitFor()
    await page.setViewportSize({ width: 720, height: 520 })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.evaluate(() => {
      const label = document.createElement('div'); label.textContent = `Synthetic browser fixture · ${globalThis.__channelsQa.platform} · No messages sent`
      Object.assign(label.style, { position: 'fixed', bottom: '2px', right: '6px', zIndex: '9999', background: '#17202a', color: '#fff', padding: '3px 6px', fontSize: '10px', pointerEvents: 'none' })
      document.body.append(label)
    })
    await page.screenshot({ path: path.join(output, `${platform}-bluebubbles.png`) })
    checks.push(`${platform} browser fixture: BlueBubbles setup is available with the Mac requirement stated and no horizontal overflow at 720×520`)
    assert.equal(await page.evaluate(() => globalThis.__channelsQa.externalWrites), 0)
    assert.deepEqual(errors, [])
    await page.close()
  }
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify({ result: 'passed', data: 'Synthetic browser fixtures, not native platform or live-account checks', checks, externalWrites: 0 }, null, 2))
  console.log(checks)
} finally { await browser.close() }
