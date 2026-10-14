import assert from 'node:assert/strict'
import http from 'node:http'
import { once } from 'node:events'
import { probeProviderGeneration, providerProbeFailure } from './provider-generation-probe'
import type { Provider } from './types'
import type { ChatParams, StreamChunk } from './openai'

let mode: 'ok' | 'empty' | 'error' = 'ok'
const requests: Array<{ path: string; body: any }> = []
const server = http.createServer(async (req, res) => {
  let raw = ''
  for await (const part of req) raw += part
  requests.push({ path: req.url ?? '', body: JSON.parse(raw) })
  res.setHeader('content-type', 'application/json')
  if (mode === 'error') { res.statusCode = 503; res.end('{"error":"Example unavailable"}'); return }
  res.end(JSON.stringify({ choices: [{ message: { content: mode === 'ok' ? 'OK' : '' }, finish_reason: 'stop' }] }))
})
server.listen(0, '127.0.0.1')
await once(server, 'listening')
const port = (server.address() as import('node:net').AddressInfo).port
const provider: Provider = { id: 'example-probe', name: 'Example local endpoint', kind: 'local', source: 'custom', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${port}/v1`, apiKey: '', models: ['example-model'] }
try {
  const result = await probeProviderGeneration(provider, 'example-model')
  assert.equal(result.model, 'example-model')
  assert.ok(result.firstTokenMs >= 0)
  assert.equal(requests[0].path, '/v1/chat/completions', 'exercise the real generation path rather than listing models')
  assert.deepEqual(requests[0].body.messages, [{ role: 'user', content: 'Reply with OK.' }])
  assert.equal(requests[0].body.tools, undefined, 'no capability or tool is sent')
  assert.equal(JSON.stringify(result).includes('OK'), false, 'model output is neither saved nor returned to the renderer')
  mode = 'empty'
  await assert.rejects(() => probeProviderGeneration(provider, 'example-model'), /without a visible response/)
  mode = 'error'
  await assert.rejects(() => probeProviderGeneration(provider, 'example-model'), /HTTP 503/)
  let closed = false
  let observed: ChatParams | null = null
  const stream = async function* (params: ChatParams): AsyncGenerator<StreamChunk> {
    observed = params
    try { yield { type: 'content', text: 'OK' }; throw new Error('must stop on first visible token') }
    finally { closed = true }
  }
  await probeProviderGeneration(provider, 'example-model', { stream })
  assert.equal(closed, true, 'the generator is closed as soon as a response is verified')
  assert.equal((observed as ChatParams | null)?.signal?.aborted, true, 'stop any outstanding server generation')
  await assert.rejects(() => probeProviderGeneration({ ...provider, transport: 'codex-cli' }, 'example-model'), /HTTP model endpoints/)
  await assert.rejects(() => probeProviderGeneration(provider, ''), /valid model/)
  const sanitized = providerProbeFailure(new Error('HTTP 401 Example auth failed with fictional-probe-secret'))
  assert.match(sanitized, /HTTP 401/)
  assert.ok(!sanitized.includes('fictional-probe-secret'))
  assert.ok(!providerProbeFailure(new Error('Unexpected server body fictional-probe-secret')).includes('fictional-probe-secret'))
} finally {
  server.closeAllConnections()
  await new Promise<void>((resolve) => server.close(() => resolve()))
}
console.log('provider real-response probe: ok')
