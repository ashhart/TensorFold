import assert from 'node:assert/strict'
import { CONNECTOR_JSON_MAX_BYTES, readBoundedConnectorJson } from './connector-http'

assert.deepEqual(
  await readBoundedConnectorJson(new Response('{"ok":true}', { headers: { 'content-type': 'application/json' } }), 'test API'),
  { ok: true },
)

await assert.rejects(
  () => readBoundedConnectorJson(new Response('<html>login</html>', { headers: { 'content-type': 'text/html' } }), 'test API'),
  /non-JSON/,
)

await assert.rejects(
  () => readBoundedConnectorJson(new Response('{}', { headers: { 'content-type': 'application/json', 'content-length': String(CONNECTOR_JSON_MAX_BYTES + 1) } }), 'test API'),
  /exceeds/,
)

await assert.rejects(
  () => readBoundedConnectorJson(new Response('x'.repeat(CONNECTOR_JSON_MAX_BYTES + 1), { headers: { 'content-type': 'application/json' } }), 'test API'),
  /exceeds/,
)

console.log('connector HTTP bounded JSON tests passed')

// All security checks and retries run against synthetic fetches, never a live account.
const { fetchConnectorResponse } = await import('./connector-http')
const calls: RequestInit[] = []
const lookup = async () => [{ address: '8.8.8.8', family: 4 as const }]
let tries = 0
const syntheticFetch = (async (_input: unknown, init: RequestInit = {}) => {
  calls.push(init); tries++
  return new Response('{}', { status: tries === 1 ? 503 : 200, headers: { 'content-type': 'application/json' } })
}) as typeof fetch
const abort = new AbortController()
await fetchConnectorResponse('https://connector.example.com', 'test connector', 'https://connector.example.com/items', {
  method: 'GET', signal: abort.signal, headers: { Authorization: 'Bearer synthetic-token' }, redirect: 'follow',
}, { lookup, fetchImpl: syntheticFetch, retry: { sleep: async () => {}, random: () => 0 } })
assert.equal(calls.length, 2, 'shared request retains retry behavior')
assert.ok(calls.every(init => init.redirect === 'error'), 'redirects cannot forward connector credentials')
assert.ok(calls.every(init => init.signal === abort.signal), 'cancellation remains bound to every retry')
assert.ok(calls.every(init => new Headers(init.headers).get('authorization') === 'Bearer synthetic-token'))
const before = calls.length
await assert.rejects(fetchConnectorResponse('https://connector.example.com', 'test connector', 'https://other.example.com/items', {}, { lookup, fetchImpl: syntheticFetch }), /origin|host|endpoint/i)
assert.equal(calls.length, before, 'cross-origin target is refused before fetch')
await assert.rejects(fetchConnectorResponse('http://127.0.0.1:8070', 'test connector', 'http://127.0.0.1:8070/items', {}, { fetchImpl: syntheticFetch }), /private|loopback|local|HTTPS/i)
assert.equal(calls.length, before)
tries = 1
await fetchConnectorResponse('http://127.0.0.1:8070', 'test connector', 'http://127.0.0.1:8070/items', {}, { allowPrivateNetwork: true, fetchImpl: syntheticFetch })
assert.equal(calls.length, before + 1, 'explicit private-source opt-in still works')
await assert.rejects(fetchConnectorResponse('http://169.254.169.254', 'test connector', 'http://169.254.169.254/latest', {}, { allowPrivateNetwork: true, fetchImpl: syntheticFetch }), /metadata|private|link-local/i)
assert.equal(calls.length, before + 1, 'metadata remains forbidden even with private-source opt-in')
console.log('Shared connector response origin, credential, retry, abort and private-network guards passed')
