import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { launchFixtureBrowser, openFixturePage } from './browser-session.mjs'

const output = path.resolve(process.argv[2]), base = process.argv[3] || 'http://127.0.0.1:5186/'
await mkdir(output, { recursive: true })
const browser = await launchFixtureBrowser(), checks = []
try {
  const { page, errors } = await openFixturePage(browser, base)
  await page.evaluate(async () => {
    const { api } = await import('/src/lib/ipc.ts')
    const qa = globalThis.__assistantGoogleQa = { revision: 1, searches: 0, mode: 'ready', launches: [], writes: 0 }
    const message = { id: 'mail', threadId: 'conversation', subject: 'Synthetic project check-in', from: 'casey@example.com',
      to: 'review@example.com', date: '2026-10-28', snippet: 'Synthetic question', text: 'Synthetic private message from revision one.', attachments: [], untrustedContent: true }
    api.googleWorkspaceStatus = async () => ({ clientConfigured: true, accounts: [{ id: 'review-account', email: 'review@example.com',
      revision: qa.revision, status: 'connected', connectedAt: 1, capabilities: ['gmail.read', 'calendar.read', 'gmail.send'] }], reviews: [], recentActions: [] })
    api.googleWorkspaceMessages = async () => {
      qa.searches++
      if (qa.mode === 'rate-limit') throw new Error('Google is rate limiting this connection. Try again later.')
      return { items: [message] }
    }
    api.googleWorkspaceThread = async () => ({ id: message.threadId, messages: [message] })
    api.googleWorkspaceCalendars = async () => ({ items: [{ id: 'primary', summary: 'Synthetic calendar', timeZone: 'Europe/London', accessRole: 'reader' }] })
    api.googleWorkspaceDrafts = async () => []
    for (const name of ['googleWorkspacePrepareAction', 'googleWorkspaceExecuteAction', 'replyToThread']) api[name] = async () => { qa.writes++; throw new Error('External writes are forbidden in this fixture') }
    globalThis.__store.setState({ launch: async payload => { qa.launches.push(payload); return { ok: false, error: 'Fixture model unavailable; draft retained' } } })
    globalThis.__store.getState().setView({ name: 'assistant' })
  })
  const tab = name => page.getByRole('tab', { name, exact: true })
  await page.getByRole('heading', { name: 'Your inbox', exact: true }).waitFor()
  await tab('Google Workspace').click()
  const mail = page.getByRole('region', { name: 'Gmail inbox', exact: true })
  await mail.waitFor()
  assert.equal(await page.evaluate(() => globalThis.__assistantGoogleQa.searches), 0)
  await mail.getByRole('button', { name: 'Search', exact: true }).click()
  await mail.getByRole('button', { name: 'Read conversation', exact: true }).click()
  await mail.getByText('Synthetic private message from revision one.', { exact: true }).waitFor()
  const googleDraft = page.getByRole('textbox', { name: 'Email message', exact: true })
  await googleDraft.fill('Keep this unsaved Google reply while reconnecting.')
  await page.waitForFunction(() => document.querySelector('select[aria-label="Google account"]')?.disabled === true)
  assert.equal(await page.getByRole('combobox', { name: 'Google account', exact: true }).isDisabled(), true)
  checks.push('Connected Google is reachable from Assistant; Gmail content is read only after an explicit action')
  await tab('Overview').click()
  const draft = page.getByRole('textbox', { name: 'Assistant task draft', exact: true })
  await draft.fill('Keep this general assistant draft through a failed launch.')
  await page.getByRole('button', { name: 'Start assistant task', exact: true }).click()
  await page.getByText('Fixture model unavailable; draft retained', { exact: true }).waitFor()
  assert.match(await draft.inputValue(), /Keep this general/)
  await tab('Inbox').click(); await tab('Overview').click()
  assert.match(await draft.inputValue(), /Keep this general/)
  const launches = await page.evaluate(() => globalThis.__assistantGoogleQa.launches)
  assert.equal(launches.length, 1)
  assert.ok(!JSON.stringify(launches[0]).includes('review@example.com'))
  checks.push('Overview retains failed and hidden drafts and does not silently attach connected Google data')
  await tab('Google Workspace').click()
  assert.match(await googleDraft.inputValue(), /Keep this unsaved Google reply/)
  await page.evaluate(() => { globalThis.__assistantGoogleQa.mode = 'rate-limit' })
  await mail.getByRole('button', { name: 'Search', exact: true }).click()
  await mail.getByText('Google is rate limiting this connection. Try again later.', { exact: true }).waitFor()
  checks.push('Known Google failures retain their specific, safe recovery guidance')
  await page.evaluate(() => { globalThis.__assistantGoogleQa.revision++ })
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await mail.getByText('Synthetic private message from revision one.', { exact: true }).waitFor({ state: 'hidden', timeout: 2500 })
  assert.equal(await mail.getByText('Synthetic private message from revision one.', { exact: true }).count(), 0,
    'Reauthorization must clear message content from the previous authorization revision')
  checks.push('Authorization revision changes remove old Gmail content')
  assert.match(await googleDraft.inputValue(), /Keep this unsaved Google reply/)
  checks.push('Unsaved Google reply text survives tab switches and same-account reauthorization')
  await page.setViewportSize({ width: 720, height: 520 })
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
  await page.evaluate(() => {
    const label = document.createElement('div'); label.textContent = 'Browser fixture · Synthetic accounts · No messages sent'
    Object.assign(label.style, { position: 'fixed', bottom: '2px', right: '6px', zIndex: '9999', background: '#17202a', color: '#fff', padding: '3px 6px', fontSize: '10px', pointerEvents: 'none' })
    document.body.append(label)
  })
  await page.screenshot({ path: path.join(output, 'google-minimum.png') })
  await page.setViewportSize({ width: 1440, height: 1000 }); await tab('Overview').click()
  await page.screenshot({ path: path.join(output, 'overview.png') })
  assert.equal(await page.evaluate(() => globalThis.__assistantGoogleQa.writes), 0)
  assert.deepEqual(errors, [])
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify({ result: 'passed', data: 'Synthetic browser fixture', nativeGoogleAccountVerified: false, checks, pageErrors: errors, externalWrites: 0 }, null, 2))
  console.log(checks)
} finally { await browser.close() }
