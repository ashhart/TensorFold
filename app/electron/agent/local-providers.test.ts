// Run: node electron/agent/local-providers.test.ts
import assert from 'node:assert/strict'
import { createServer, type Server } from 'node:http'
import { discoverLocalProvider, fetchOpenAiCompatibleModels } from './local-providers'
import { assertLocalProviderDiscoveryUrl } from './security'

// The exact shape a TensorFold engine answers with at /v1/models.
const TENSORFOLD_MODELS = {
  object: 'list',
  data: [{
    id: 'Qwen3.8-27B-MLX-4bit', object: 'model', created: 1_792_421_039, owned_by: 'tensorfold',
    context_length: 81_920, max_model_len: 81_920,
    reasoning: { default_enabled: true, supported_efforts: ['xhigh', 'medium', 'low', 'none'] },
  }],
}

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

async function serve(handler: (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => void): Promise<{ server: Server; base: string }> {
  const server = createServer(handler)
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as { port: number }).port
  return { server, base: `http://127.0.0.1:${port}/v1` }
}

const seen: Array<string | undefined> = []
const { server, base } = await serve((req, res) => {
  seen.push(req.headers.authorization)
  if (req.url !== '/v1/models') { res.writeHead(404).end(); return }
  if (req.headers.authorization !== 'Bearer tf-local-key') { res.writeHead(401, { 'content-type': 'application/json' }).end('{"error":"unauthorized"}'); return }
  res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(TENSORFOLD_MODELS))
})
try {
  const models = await fetchOpenAiCompatibleModels(base, 'tf-local-key')
  assert.deepEqual(models, ['Qwen3.8-27B-MLX-4bit'], 'the served model id is read from the TensorFold payload')
  assert.equal(seen[0], 'Bearer tf-local-key', 'the API key is sent as a bearer token')
  ok('OpenAI-compatible discovery reads a TensorFold /v1/models payload')

  seen.length = 0
  const found = await discoverLocalProvider('custom', base, 'tf-local-key')
  assert.equal(found.found, true)
  assert.equal(found.source, 'custom', 'a custom endpoint stays a custom provider, never the built-in engine slot')
  assert.equal(found.defaultModel, 'Qwen3.8-27B-MLX-4bit')
  assert.match(found.message, /Found 1 model/)
  ok('custom discovery returns the served model for provider setup')

  const unauthorized = await discoverLocalProvider('custom', base, 'wrong-key')
  assert.equal(unauthorized.found, false)
  assert.match(unauthorized.message, /401/)
  assert.deepEqual(unauthorized.models, [])
  ok('a rejected key reports rather than throwing')

  const missing = await discoverLocalProvider('custom', 'http://127.0.0.1:9/v1', 'key')
  assert.equal(missing.found, false)
  assert.match(missing.message, /Could not reach/)
  ok('an unreachable endpoint reports instead of throwing')

  assert.equal((await discoverLocalProvider('custom')).baseUrl, 'http://127.0.0.1:8080/v1', 'the default target is a local TensorFold engine')

  // A private LAN address is an allowed credential destination; a public one is not.
  assert.ok(await assertLocalProviderDiscoveryUrl('http://10.0.0.5:8080/v1'), 'private LAN endpoints are permitted')
  await assert.rejects(() => assertLocalProviderDiscoveryUrl('http://93.184.216.34:8080/v1'), /non-private address/)
  ok('LAN addresses pass the credential policy while public ones are refused')
} finally {
  await new Promise<void>((resolve) => server.close(() => resolve()))
}

console.log(`\n${passed} passed`)
