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
