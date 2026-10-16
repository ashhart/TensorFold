// Run: npx jiti electron/agent/model-gateway.test.ts
import assert from 'node:assert'
import { AI_GATEWAY_DEFAULT_MODELS, fetchAiGatewayModels, isAiGatewayProvider } from './model-gateway.ts'
import type { Provider } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ok ${name}`) }

const provider: Provider = {
  id: 'ai-gateway',
  name: 'Vercel AI Gateway',
  baseUrl: 'https://ai-gateway.vercel.sh/v1',
  apiKey: 'gateway-key',
  models: AI_GATEWAY_DEFAULT_MODELS,
  kind: 'cloud',
  source: 'ai-gateway',
}

console.log('model gateway tests:')

{
  assert.strictEqual(isAiGatewayProvider(provider), true)
  assert.strictEqual(isAiGatewayProvider({ ...provider, source: undefined, name: 'Gateway', baseUrl: 'https://ai-gateway.vercel.sh/v1' }), true)
  assert.strictEqual(isAiGatewayProvider({ ...provider, source: undefined, name: 'OpenAI', baseUrl: 'https://api.openai.com/v1' }), false)
  ok('detects AI Gateway providers without matching normal OpenAI endpoints')
}

{
  const models = await fetchAiGatewayModels(provider, async () => new Response(JSON.stringify({
    data: [{ id: 'openai/gpt-5.5' }, { id: 'anthropic/claude-opus-4.1' }, { id: 'custom/provider-model' }],
  }), { status: 200, headers: { 'content-type': 'application/json' } }) as any)
  assert.ok(models.includes('openai/gpt-5.4'))
  assert.ok(models.includes('custom/provider-model'))
  assert.strictEqual(models.filter((model) => model === 'openai/gpt-5.5').length, 1)
  ok('merges gateway defaults with discovered models without duplicates')
}

{
  const models = await fetchAiGatewayModels(provider, async () => new Response('Not found', { status: 404 }) as any)
  assert.deepStrictEqual(models, AI_GATEWAY_DEFAULT_MODELS)
  ok('falls back to curated gateway models when the gateway has no models endpoint')
}

{
  await assert.rejects(
    () => fetchAiGatewayModels(provider, async () => new Response('Unauthorized', { status: 401, statusText: 'Unauthorized' }) as any),
    /HTTP 401 Unauthorized/
  )
  ok('does not mask gateway authentication failures')
}

console.log(`model gateway tests: ${passed} passed`)
