import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { createServer } from 'node:http'
import { once } from 'node:events'
import { mkdtemp, mkdir, readFile, realpath, rm, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'

const require = createRequire(import.meta.url)
const { _electron } = require(process.env.TENSORFOLD_QA_PLAYWRIGHT || 'playwright')
const executablePath = require('electron')
const root = path.resolve('.'), profile = await realpath(await mkdtemp(path.join(os.tmpdir(), 'tensorfold-quick-ask-qa-')))
const output = process.env.TENSORFOLD_QA_OUTPUT || path.resolve('release/quick-ask-qa')
await mkdir(output, { recursive: true })
const requests = [], errors = [], checks = []
const server = createServer(async (req, res) => {
  let body = ''; for await (const chunk of req) body += chunk
  if (req.url === '/v1/models') { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ data: [{ id: 'example-model' }] })); return }
  if (req.url !== '/v1/chat/completions') { res.writeHead(404); res.end(); return }
  const payload = JSON.parse(body); requests.push(payload)
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store' })
  res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: 'A native ' }, finish_reason: null }] })}\n\n`)
  setTimeout(() => { if (res.destroyed) return; res.end(`data: ${JSON.stringify({ choices: [{ delta: { content: 'streamed answer.' }, finish_reason: null }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`) }, 150)
})
server.listen(0, '127.0.0.1'); await once(server, 'listening')
const base = `http://127.0.0.1:${server.address().port}/v1`
let application
async function pollApi(read, ready) {
  const deadline = Date.now() + 20_000
  while (Date.now() < deadline) { const value = await read(); if (ready(value)) return value; await new Promise(resolve => setTimeout(resolve, 80)) }
  throw new Error('The native API did not reach the expected state.')
}
async function launch() {
  application = await _electron.launch({ executablePath, args: [root], env: { ...process.env, VONTRA_HOME: profile, VONTRA_HEADLESS: '0', VITE_DEV_SERVER_URL: '' }, timeout: 45_000 })
  const main = await application.firstWindow(); main.setDefaultTimeout(20_000)
  main.on('pageerror', error => errors.push(error.message))
  await main.waitForFunction(() => !!window.tensorfoldStudio && !!window.tensorfoldQuickAskLauncher)
  return main
}
async function open(main) {
  const pending = application.waitForEvent('window')
  await main.evaluate(() => window.tensorfoldQuickAskLauncher.open())
  const quick = await pending; quick.setDefaultTimeout(20_000)
  quick.on('pageerror', error => errors.push(error.message))
  await quick.waitForFunction(() => !!window.tensorfoldQuickAsk)
  await quick.getByRole('textbox', { name: 'Question', exact: true }).waitFor()
  await quick.waitForFunction(() => getComputedStyle(document.getElementById('boot')).opacity === '0')
  return quick
}
try {
  const main = await launch()
  await main.evaluate(async base => {
    const api = window.tensorfoldStudio
    const settings = await api.getSettings()
    await api.saveSettings({ ...settings, autoCheckUpdates: false, providers: [{ id: 'native-fixture', name: 'Native QA', transport: 'chat-completions', kind: 'local', source: 'custom', apiKey: '', baseUrl: base, models: ['example-model'] }], activeProviderId: 'native-fixture', activeProviderAccountId: null, activeModel: 'example-model', remoteOrchestrators: [], desktopMode: 'full', desktopOnboardingComplete: true, uiPreferences: { locale: 'en', direction: 'auto', chatScale: 1, codeScale: 1, density: 'comfortable' } })
    localStorage.setItem('vontra.onboarded', '1')
  }, base)
  await main.reload()
  await main.waitForFunction(() => document.body.classList.contains('ready'))
  console.log('Isolated native workspace ready')
  const quick = await open(main)
  assert.deepEqual(await quick.evaluate(() => ({ full: !!window.tensorfoldStudio, legacy: !!window.vontra, launcher: !!window.tensorfoldQuickAskLauncher })), { full: false, legacy: false, launcher: false })
  checks.push('The native floating window exposes only the restricted Quick Ask bridge')
  await quick.getByRole('combobox', { name: 'Model and destination' }).selectOption({ index: 1 })
  await quick.getByRole('textbox', { name: 'Question', exact: true }).fill('A synthetic native round-trip question')
  await quick.getByRole('button', { name: 'Ask', exact: true }).click()
  await quick.getByText('A native streamed answer.', { exact: true }).waitFor()
  await pollApi(() => quick.evaluate(() => window.tensorfoldQuickAsk.snapshot()), value => value.session.turns.at(-1)?.status === 'complete')
  assert.equal(requests.length, 1); assert.equal(requests[0].tools, undefined); assert.equal(requests[0].model, 'example-model')
  checks.push('A real Electron IPC request streams through the production model transport')
  await quick.getByRole('textbox', { name: 'Question', exact: true }).fill('Keep this unfinished follow-up')
  await quick.screenshot({ path: path.join(output, 'ask-tensorfold-native.png') })
  await quick.getByRole('button', { name: 'Continue in TensorFold', exact: true }).click()
  const transferred = await pollApi(() => main.evaluate(async () => {
    const thread = (await window.tensorfoldStudio.getThreads()).find(thread => thread.title === 'A synthetic native round-trip question')
    if (!thread) return null
    const draft = await window.tensorfoldStudio.getComposerDraft({ surface: 'chat', conversationId: thread.id, hostId: null, incognito: false })
    return { id: thread.id, messages: thread.apiMessages.length, text: draft.text }
  }), value => value?.text === 'Keep this unfinished follow-up')
  assert.equal(transferred.messages, 3); assert.equal(transferred.text, 'Keep this unfinished follow-up')
  checks.push('Handoff saves model history and transfers the unfinished draft into the full app')
  await main.evaluate(() => window.tensorfoldQuickAskLauncher.open())
  await quick.getByRole('button', { name: 'More options', exact: true }).click()
  await quick.getByRole('button', { name: 'New ask', exact: true }).click()
  await quick.getByRole('textbox', { name: 'Question', exact: true }).fill('Restart recovery proof')
  await quick.evaluate(async () => { const api = window.tensorfoldQuickAsk, state = await api.snapshot(); await api.configure({ ...state.preferences, enabled: true, shortcut: 'Control+Alt+F19' }) })
  assert.equal(await application.evaluate(({ globalShortcut }) => globalShortcut.isRegistered('Control+Alt+F19')), true)
  checks.push('The operating system accepted a dedicated global shortcut')
  await application.close(); application = null
  const encrypted = await readFile(path.join(profile, 'state', 'quick-ask-session.json'), 'utf8')
  assert.equal(encrypted.includes('Restart recovery proof'), false)
  assert.equal((await stat(path.join(profile, 'state', 'quick-ask-session.json'))).mode & 0o777, 0o600)
  const restartedMain = await launch(), restored = await open(restartedMain)
  assert.equal(await restored.getByRole('textbox', { name: 'Question', exact: true }).inputValue(), 'Restart recovery proof')
  checks.push('Quit flushes, encrypts and restores the draft in the owning profile')
  assert.deepEqual(errors, [])
  await writeFile(path.join(output, 'native-journey.json'), JSON.stringify({ checks, requests: requests.length, errors, qualification: 'Native macOS Electron with a synthetic local streaming API; Windows and Linux native desktop behaviour requires those systems.' }, null, 2))
  console.log(`Ask TensorFold native journey passed ${checks.length} checks.`)
} finally {
  if (application) await application.close().catch(() => {})
  server.closeAllConnections(); server.close(); await once(server, 'close')
  await rm(profile, { recursive: true, force: true })
}
