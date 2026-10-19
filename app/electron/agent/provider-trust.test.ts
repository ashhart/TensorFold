// Run: npx tsx electron/agent/provider-trust.test.ts
import assert from 'node:assert/strict'
import { classifyProvider, isExternalProvider, isLoopbackUrl, isPrivateNetworkUrl, pickPlanRunner } from '../../shared/provider-trust'

const local = (baseUrl: string, extra: Record<string, string> = {}) => ({ id: 'ollama', name: 'Ollama', kind: 'local', source: 'ollama', transport: 'chat-completions', baseUrl, ...extra })

// Loopback detection covers names, IPv4/IPv6 and the IPv4-mapped forms URL normalizes to.
for (const url of ['http://127.0.0.1:11434/v1', 'http://localhost:1234/v1', 'http://[::1]:8000/v1', 'http://127.8.9.10/v1', 'http://[::ffff:127.0.0.1]:1/v1', 'https://localhost/v1']) {
  assert.equal(isLoopbackUrl(url), true, url)
}
for (const url of ['https://api.openai.com/v1', 'http://192.168.1.20:8000/v1', 'http://10.0.0.2/v1', 'http://localhost.example.com/v1', 'http://user:pw@127.0.0.1/v1', 'ftp://127.0.0.1', 'not a url', '', 'http://[::ffff:10.0.0.1]/v1']) {
  assert.equal(isLoopbackUrl(url), false, url)
}

// Trusted local: kind local + loopback, or the built-in engine (even before its URL is live).
assert.equal(classifyProvider(local('http://127.0.0.1:11434/v1')), 'trusted-local')
assert.equal(classifyProvider({ id: 'tensorfold', source: 'tensorfold', kind: 'local', baseUrl: '' }), 'trusted-local')
assert.equal(classifyProvider({ id: 'tensorfold', source: 'tensorfold', kind: 'local', baseUrl: 'http://127.0.0.1:41000/v1' }), 'trusted-local')
assert.equal(classifyProvider(undefined, 'http://127.0.0.1:1234/v1'), 'trusted-local')

// External: cloud, LAN, a cloud URL marked local, a loopback URL not marked local, any CLI.
assert.equal(classifyProvider({ id: 'openai', kind: 'cloud', source: 'openai', baseUrl: 'https://api.openai.com/v1' }), 'external')
assert.equal(classifyProvider(local('http://192.168.1.20:8000/v1')), 'external')
assert.equal(classifyProvider(local('https://openrouter.ai/api/v1')), 'external')
assert.equal(classifyProvider({ id: 'proxy', kind: 'cloud', baseUrl: 'http://127.0.0.1:4000/v1' }), 'external')
assert.equal(classifyProvider({ id: 'tensorfold', source: 'tensorfold', kind: 'local', baseUrl: 'https://tensorfold.example.com/v1' }), 'external')
assert.equal(classifyProvider({ id: 'tensorfold', kind: 'local', baseUrl: '' }), 'external', 'id alone is not the built-in engine')
assert.equal(classifyProvider(local('http://127.0.0.1:11434/v1'), 'https://api.openai.com/v1'), 'external', 'the URL actually used counts')
assert.equal(classifyProvider(local('')), 'external')
assert.equal(classifyProvider(null), 'external')
for (const transport of ['codex-cli', 'claude-cli', 'cursor-cli', 'gemini-cli', 'pi-cli', 'opencode', 'kilo']) {
  assert.equal(isExternalProvider(local('http://127.0.0.1:4096', { transport })), true, transport)
}

// A network engine the user marked as their own is local only while the address stays private.
const trustedNetwork = (baseUrl: string) => ({ id: 'tf-remote', kind: 'local', source: 'custom', transport: 'chat-completions', baseUrl, trustedNetwork: true })
assert.equal(classifyProvider(trustedNetwork('http://192.168.1.20:8080/v1')), 'trusted-local')
assert.equal(classifyProvider(trustedNetwork('http://10.1.2.3:8080/v1')), 'trusted-local')
assert.equal(classifyProvider(trustedNetwork('http://172.20.0.5:8080/v1')), 'trusted-local')
assert.equal(classifyProvider(trustedNetwork('http://mac-mini.local:8080/v1')), 'trusted-local')
assert.equal(classifyProvider(trustedNetwork('http://127.0.0.1:8080/v1')), 'trusted-local')
assert.equal(classifyProvider(trustedNetwork('https://tensorfold.example.com/v1')), 'external', 'a public address is never trusted')
assert.equal(classifyProvider({ ...trustedNetwork('http://192.168.1.20:8080/v1'), transport: 'claude-cli' }), 'external', 'CLI transports stay external')
assert.equal(classifyProvider(trustedNetwork('http://93.184.216.34:8080/v1')), 'external', 'a public IP is never trusted')
assert.equal(classifyProvider(local('http://192.168.1.20:8000/v1')), 'external', 'an unmarked LAN provider stays external')
assert.equal(classifyProvider(local('http://172.32.0.1:8080/v1')), 'external', '172.32 is outside the private range')
assert.equal(trustedNetwork('http://10.1.2.3/v1').trustedNetwork, true)

for (const url of ['http://127.0.0.1:1/v1', 'http://10.0.0.2/v1', 'http://192.168.0.10/v1', 'http://172.16.4.4/v1', 'http://box.local/v1', 'http://[fd00::1]/v1']) {
  assert.equal(isPrivateNetworkUrl(url), true, url)
}
for (const url of ['https://api.openai.com/v1', 'http://8.8.8.8/v1', 'http://172.15.0.1/v1', 'http://172.32.0.1/v1', 'not a url', '']) {
  assert.equal(isPrivateNetworkUrl(url), false, url)
}

// Plan runner: the built-in engine first, then any trusted local provider, else none.
const openai = { id: 'openai', kind: 'cloud', baseUrl: 'https://api.openai.com/v1' }
const ollama = local('http://127.0.0.1:11434/v1')
const builtIn = { id: 'tensorfold', source: 'tensorfold', kind: 'local', baseUrl: '' }
assert.equal(pickPlanRunner([openai, ollama, builtIn]), builtIn)
assert.equal(pickPlanRunner([openai, ollama]), ollama)
assert.equal(pickPlanRunner([openai]), null)
assert.equal(pickPlanRunner(undefined), null)

console.log('provider-trust tests passed')
