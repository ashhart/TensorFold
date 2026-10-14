// Run: jiti src/lib/provider-discovery.test.ts
import assert from 'node:assert/strict'
import { buildDiscoveredProvider, findDiscoveryTarget, isLoopbackEndpoint, mergeDiscoveredProvider, sameLocalEndpoint } from './provider-discovery'
import type { Provider, ProviderDiscovery } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function provider(over: Partial<Provider>): Provider {
  return { id: 'p1', name: 'oMLX', transport: 'chat-completions', auth: 'api-key', baseUrl: 'http://192.168.1.9:9000/v1', apiKey: '', models: ['local-model'], kind: 'local', source: 'omlx', ...over }
}

function discovery(over: Partial<ProviderDiscovery> = {}): ProviderDiscovery {
  return { kind: 'omlx', found: true, name: 'oMLX', source: 'omlx', baseUrl: 'http://192.168.1.9:9000/v1', apiKey: 'tok', models: ['m1', 'm2'], defaultModel: 'm1', configPath: null, message: 'ok', ...over }
}

function main() {
  // sameLocalEndpoint normalizes trailing slashes + case and compares source.
  assert.ok(sameLocalEndpoint({ baseUrl: 'http://192.168.1.9:9000/v1/', source: 'omlx' }, { baseUrl: 'HTTP://192.168.1.9:9000/v1', source: 'omlx' }))
  assert.ok(!sameLocalEndpoint({ baseUrl: 'http://192.168.1.9:9000/v1', source: 'omlx' }, { baseUrl: 'http://192.168.1.9:9000/v1', source: 'ollama' }))
  assert.ok(!sameLocalEndpoint({ baseUrl: 'http://192.168.1.9:9000/v1', source: 'omlx' }, { baseUrl: 'http://10.0.0.2:9000/v1', source: 'omlx' }))
  ok('sameLocalEndpoint matches by baseUrl + source')

  // findDiscoveryTarget: id wins, then baseUrl+source fallback.
  const existing = provider({})
  assert.equal(findDiscoveryTarget([existing], discovery(), 'p1')?.id, 'p1')
  assert.equal(findDiscoveryTarget([existing], discovery())?.id, 'p1')
  assert.equal(findDiscoveryTarget([existing], discovery({ source: 'ollama' })), null)
  ok('findDiscoveryTarget resolves by id then baseUrl+source')

  // Task #45 core: a discovery that matches an existing endpoint UPDATES it,
  // and does not grow the list.
  const built = buildDiscoveredProvider(discovery(), 'new-id', existing)
  const updated = mergeDiscoveredProvider([existing], built)
  assert.equal(updated.created, false)
  assert.equal(updated.providers.length, 1)
  assert.equal(updated.providers[0].id, 'p1', 'keeps the existing provider id')
  assert.deepEqual(updated.providers[0].models, ['m1', 'm2'], 'refreshes models from discovery')
  assert.equal(updated.providers[0].apiKey, 'tok')
  ok('matching discovery updates in place (no duplicate)')

  // Simulate rapid blurs + discovery retries: repeated merges never duplicate.
  let list: Provider[] = [existing]
  for (let i = 0; i < 5; i++) {
    const b = buildDiscoveredProvider(discovery(), 'new-' + i, findDiscoveryTarget(list, discovery()))
    list = mergeDiscoveredProvider(list, b).providers
  }
  assert.equal(list.length, 1, 'five discovery passes still yield one provider')
  ok('repeated discovery passes stay at one provider')

  // A genuinely new endpoint is appended exactly once.
  const other = buildDiscoveredProvider(discovery({ baseUrl: 'http://10.0.0.5:9000/v1' }), 'fresh')
  const added = mergeDiscoveredProvider([existing], other)
  assert.equal(added.created, true)
  assert.equal(added.providers.length, 2)
  assert.equal(added.target.id, 'fresh')
  ok('new endpoint is appended once')

  // Preserves a user-edited name on the existing provider.
  const named = provider({ name: 'My Spark node' })
  const merged = mergeDiscoveredProvider([named], buildDiscoveredProvider(discovery(), 'x', named))
  assert.equal(merged.providers[0].name, 'My Spark node')
  ok('keeps a user-edited provider name')

  // Discover button is hidden for remote endpoints (#46): only loopback hosts
  // can be auto-discovered from local runtime config.
  for (const url of ['http://127.0.0.1:9000/v1', 'http://localhost:1234/v1', 'http://[::1]:9000/v1', 'http://0.0.0.0:9000']) {
    assert.ok(isLoopbackEndpoint(url), `${url} is loopback`)
  }
  for (const url of ['http://192.168.1.45:9000/v1', 'http://10.0.0.2:1234/v1', 'https://api.openai.com/v1', '', 'not a url']) {
    assert.ok(!isLoopbackEndpoint(url), `${url} is not loopback`)
  }
  ok('isLoopbackEndpoint distinguishes local from remote endpoints')

  console.log(`provider-discovery.test.ts — ${passed} checks passed`)
}

main()
