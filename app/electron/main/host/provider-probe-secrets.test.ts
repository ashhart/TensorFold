import assert from 'node:assert/strict'
import http from 'node:http'
import { once } from 'node:events'
import { listProviderModels, resolveProviderProbe } from './provider-runtime'
import { store } from '../state'
import type { Provider } from '../../agent/types'
import { SECRET_MASK } from '../../agent/security'

const seen: string[] = []
const server = http.createServer((req, res) => {
  seen.push(req.headers.authorization ?? '')
  res.setHeader('content-type', 'application/json')
  res.end(JSON.stringify({ data: [{ id: 'example-model' }] }))
})
server.listen(0, '127.0.0.1')
await once(server, 'listening')
const port = (server.address() as import('node:net').AddressInfo).port
const oldSettings = store.settings
const original: Provider = { id: 'probe-secret-example', name: 'Example local endpoint', kind: 'local', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${port}/saved`, apiKey: 'fictional-probe-secret', models: [] }
store.settings = { ...oldSettings, providers: [original] }
try {
  await listProviderModels({ ...original, baseUrl: `http://127.0.0.1:${port}/changed`, apiKey: '' })
  assert.equal(seen.at(-1), '', 'an explicitly empty draft key must never restore a saved key at a changed endpoint')
  const before = seen.length
  await assert.rejects(() => listProviderModels({ ...original, baseUrl: `http://127.0.0.1:${port}/changed`, apiKey: SECRET_MASK }), /saved secret.*saved URL/i)
  assert.equal(seen.length, before, 'masked endpoint swaps fail before any request')
  await listProviderModels({ ...original, apiKey: SECRET_MASK })
  assert.equal(seen.at(-1), 'Bearer fictional-probe-secret', 'the saved URL can use its own masked secret')
  await listProviderModels({ ...original, baseUrl: `http://127.0.0.1:${port}/changed`, apiKey: 'new-fictional-key' })
  assert.equal(seen.at(-1), 'Bearer new-fictional-key', 'an explicitly supplied new key belongs to the new endpoint')
  const cli: Provider = { ...original, transport: 'codex-cli', baseUrl: '', apiKey: '', cliPath: '/example/trusted/codex', cliEnv: { EXAMPLE_ACCESS_KEY: 'fictional-cli-secret' } }
  store.settings.providers = [cli]
  assert.throws(() => resolveProviderProbe({ ...cli, cliPath: '/example/changed/codex', cliEnv: { EXAMPLE_ACCESS_KEY: SECRET_MASK } }), /saved.*(?:CLI|launch|executable)/i, 'masked environment credentials cannot be rebound to a different executable')
  assert.equal(resolveProviderProbe({ ...cli, cliEnv: { EXAMPLE_ACCESS_KEY: SECRET_MASK } }).cliEnv?.EXAMPLE_ACCESS_KEY, 'fictional-cli-secret')
} finally {
  store.settings = oldSettings
  await new Promise<void>((resolve) => server.close(() => resolve()))
}
console.log('provider probe credential boundaries: ok')
