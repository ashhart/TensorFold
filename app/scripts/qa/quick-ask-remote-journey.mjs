import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { createServer } from 'node:http'
import { createServer as tcpServer } from 'node:net'
import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { once } from 'node:events'
import { mkdtemp, mkdir, realpath, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'

const require = createRequire(import.meta.url), { _electron } = require(process.env.TENSORFOLD_QA_PLAYWRIGHT || 'playwright')
const executablePath = require('electron'), root = path.resolve('.')
let token = randomBytes(32).toString('hex')
const workspace = await realpath(await mkdtemp(path.join(os.tmpdir(), 'tensorfold-ask-remote-qa-')))
const output = process.env.TENSORFOLD_QA_OUTPUT || path.resolve('release/quick-ask-qa')
await mkdir(output, { recursive: true })
const temporary = tcpServer(); temporary.listen(0, '127.0.0.1'); await once(temporary, 'listening')
const port = temporary.address().port; temporary.close(); await once(temporary, 'close')
const requests = [], errors = []
const modelServer = createServer(async (req, res) => {
  let body = ''; for await (const chunk of req) body += chunk
  if (req.url !== '/v1/chat/completions') { res.writeHead(404); res.end(); return }
  requests.push(JSON.parse(body))
  res.writeHead(200, { 'content-type': 'text/event-stream' })
  res.end(`data: ${JSON.stringify({ choices: [{ delta: { content: 'An answer from the paired QA host.' }, finish_reason: null }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
})
modelServer.listen(0, '127.0.0.1'); await once(modelServer, 'listening')
const baseUrl = `http://127.0.0.1:${modelServer.address().port}/v1`, hostUrl = `http://127.0.0.1:${port}`
let client, host
async function poll(read, condition) {
  const until = Date.now() + 30_000
  let last, failure = ''
  while (Date.now() < until) { try { const value = await read(); last = value; if (condition(value)) return value } catch (error) { failure = error.message } await new Promise(resolve => setTimeout(resolve, 100)) }
  throw new Error(`The isolated remote workflow did not reach the expected state. ${JSON.stringify({ error: last?.session?.turns?.at(-1)?.error, ui: last?.qaUiError, failure })}`)
}
try {
  const previous = process.env.VONTRA_HOME
  process.env.VONTRA_HOME = path.join(workspace, 'host')
  const jiti = createJiti(import.meta.url), { defaultSettings, saveSettings, loadSettings } = await jiti.import('../../electron/store.ts')
  const hostSettings = defaultSettings()
  await saveSettings({ ...hostSettings, autoCheckUpdates: false, providers: [{ id: 'qa-model', name: 'QA model', kind: 'local', source: 'custom', baseUrl, apiKey: '', models: ['example-model'] }], remoteOrchestrators: [], remoteControl: { ...hostSettings.remoteControl, enabled: true, authToken: token, port } })
  process.env.VONTRA_HOME = path.join(workspace, 'client')
  const clientSettings = defaultSettings()
  await saveSettings({ ...clientSettings, autoCheckUpdates: false, providers: [], activeProviderId: null, activeModel: null, desktopMode: 'full', desktopOnboardingComplete: true, remoteOrchestrators: [{ id: 'qa-studio', name: 'QA Studio', serverUrl: hostUrl, authToken: token, enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastConnectedAt: null, lastError: null }] })
  if (previous === undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME = previous
  host = spawn(executablePath, [root, '--headless-server'], { env: { ...process.env, VONTRA_HOME: path.join(workspace, 'host'), VONTRA_HEADLESS: '1', VONTRA_HEADLESS_PORT: String(port), VITE_DEV_SERVER_URL: '' }, stdio: 'ignore' })
  await poll(() => fetch(`${hostUrl}/health`).then(response => response.status), value => value === 200)
  process.env.VONTRA_HOME = path.join(workspace, 'host')
  token = (await loadSettings()).remoteControl.authToken
  process.env.VONTRA_HOME = path.join(workspace, 'client')
  const configured = await loadSettings()
  await saveSettings({ ...configured, remoteOrchestrators: configured.remoteOrchestrators.map(host => ({ ...host, authToken: token })) })
  if (previous === undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME = previous
  const unauthorized = await fetch(`${hostUrl}/v1/ask/routes`)
  assert.equal(unauthorized.status, 401); await unauthorized.text()
  client = await _electron.launch({ executablePath, args: [root], env: { ...process.env, VONTRA_HOME: path.join(workspace, 'client'), VONTRA_HEADLESS: '0', VITE_DEV_SERVER_URL: '' }, timeout: 45_000 })
  const main = await client.firstWindow(); main.setDefaultTimeout(20_000)
  main.on('pageerror', error => errors.push(error.message))
  await main.waitForFunction(() => !!window.tensorfoldQuickAskLauncher)
  await main.evaluate(() => localStorage.setItem('vontra.onboarded', '1')); await main.reload()
  await main.waitForFunction(() => document.body.classList.contains('ready'))
  const windowReady = client.waitForEvent('window')
  await main.evaluate(() => window.tensorfoldQuickAskLauncher.open())
  const quick = await windowReady; quick.setDefaultTimeout(20_000)
  await quick.waitForFunction(() => !!window.tensorfoldQuickAsk)
  const state = await poll(() => quick.evaluate(() => window.tensorfoldQuickAsk.snapshot({ refreshRoutes: true })), value => value.routes.some(route => route.hostId === 'qa-studio' && route.providerId === 'qa-model'))
  const route = state.routes.find(route => route.hostId === 'qa-studio' && route.providerId === 'qa-model')
  assert.equal(route.destination, 'remote'); assert.equal(route.location, 'QA Studio')
  await quick.getByRole('combobox', { name: 'Model and destination' }).selectOption(route.id)
  await quick.getByRole('textbox', { name: 'Question', exact: true }).fill('Remote ask proof')
  await quick.getByRole('button', { name: 'Ask', exact: true }).click()
  await quick.getByRole('dialog', { name: 'Review before sending' }).waitFor()
  assert.equal(requests.length, 0, 'No request leaves before destination approval')
  await quick.getByRole('button', { name: 'Send to this destination', exact: true }).click()
  await poll(() => quick.evaluate(async () => ({ ...await window.tensorfoldQuickAsk.snapshot(), qaUiError: document.querySelector('[role=alert]')?.textContent })), value => value.session.turns.at(-1)?.status === 'complete')
  assert.equal(requests.length, 1)
  await quick.getByRole('textbox', { name: 'Question', exact: true }).fill('CLIENT_ONLY_UNSENT_DRAFT')
  await quick.locator('input[type=file]').setInputFiles({ name: 'client-only.txt', mimeType: 'text/plain', buffer: Buffer.from('CLIENT_ONLY_UNSENT_ATTACHMENT') })
  await quick.getByRole('button', { name: 'client-only.txt', exact: true }).waitFor()
  await quick.getByRole('button', { name: 'Continue in TensorFold', exact: true }).click()
  const transferred = await poll(() => quick.evaluate(() => window.tensorfoldQuickAsk.snapshot()), state => !!state.session.handoffThreadId)
  const draft = await poll(() => main.evaluate(id => window.tensorfoldStudio.getComposerDraft({ surface: 'chat', conversationId: id, hostId: 'qa-studio', incognito: false }), transferred.session.handoffThreadId), draft => draft.text === 'CLIENT_ONLY_UNSENT_DRAFT')
  assert.equal(draft.text, 'CLIENT_ONLY_UNSENT_DRAFT')
  const hostSnapshot = await fetch(`${hostUrl}/v1/snapshot`, { headers: { authorization: `Bearer ${token}` } }).then(response => response.json())
  assert.equal(JSON.stringify(hostSnapshot).includes('CLIENT_ONLY_UNSENT'), false, 'Unsent draft text and attachment bytes remain on the client during remote handoff')
  assert.deepEqual(errors, [])
  await writeFile(path.join(output, 'remote-journey.json'), JSON.stringify({ checks: ['Two isolated TensorFold processes', 'Owner-authenticated remote model catalogue', 'Explicit destination approval before traffic', 'Production remote client and host streaming', 'Durable remote conversation handoff', 'Unsent draft and attachment remain on the client'], modelRequests: requests.length, errors }, null, 2))
  console.log('Two-process Ask TensorFold remote journey passed.')
} finally {
  if (client) await client.close().catch(() => {})
  if (host && host.exitCode === null) { const exited = once(host, 'exit'); host.kill('SIGTERM'); await Promise.race([exited, new Promise(resolve => setTimeout(resolve, 5000))]); if (host.exitCode === null) host.kill('SIGKILL') }
  modelServer.closeAllConnections(); modelServer.close(); await once(modelServer, 'close')
  await rm(workspace, { recursive: true, force: true })
}
