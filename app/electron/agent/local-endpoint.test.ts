// Run: jiti electron/agent/local-endpoint.test.ts
//
// Local endpoint availability (VST-015): a local provider whose server is not listening
// refuses the run with a NAMED error before any model call; a listening server (any HTTP
// status) passes; cloud/CLI providers are never probed.
import assert from 'node:assert/strict'
import http from 'node:http'
import type { AddressInfo } from 'node:net'
import { assertLocalEndpointAvailable } from './local-endpoint'
import type { Provider } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const localProvider = (baseUrl: string): Provider => ({
  id: 'ollama', name: 'Ollama', transport: 'chat-completions', baseUrl, apiKey: '', models: ['llama3.1:8b'], kind: 'local', source: 'custom',
} as Provider)

async function main() {
  console.log('local-endpoint tests:')

  {
    // A listening server — even one that 404s everything — proves availability.
    const server = http.createServer((_req, res) => { res.statusCode = 404; res.end('nope') })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const port = (server.address() as AddressInfo).port
    await assertLocalEndpointAvailable(localProvider(`http://127.0.0.1:${port}/v1`))
    server.close()
    ok('a listening local server passes, regardless of HTTP status')
  }

  {
    // Reserve a port, then close it — nothing is listening there.
    const server = http.createServer()
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const port = (server.address() as AddressInfo).port
    await new Promise<void>((resolve) => server.close(() => resolve()))
    await assert.rejects(
      assertLocalEndpointAvailable(localProvider(`http://127.0.0.1:${port}/v1`)),
      (err: Error) => {
        assert.match(err.message, /Local endpoint http:\/\/127\.0\.0\.1:\d+\/v1 is unavailable/, 'the endpoint is named')
        assert.match(err.message, /Start the local server|fix the URL/, 'the fix is actionable')
        return true
      },
    )
    ok('a dead local endpoint refuses with a named, actionable error')
  }

  {
    // Non-local providers and CLI transports are never probed — a throwing fetch proves it.
    const explodingFetch = (() => { throw new Error('probe must not run') }) as unknown as typeof fetch
    await assertLocalEndpointAvailable({ ...localProvider('http://127.0.0.1:1/v1'), kind: 'cloud' } as Provider, explodingFetch)
    await assertLocalEndpointAvailable({ ...localProvider('http://127.0.0.1:1/v1'), transport: 'codex-cli' } as Provider, explodingFetch)
    await assertLocalEndpointAvailable({ ...localProvider(''), kind: 'local' } as Provider, explodingFetch)
    // The built-in engine's placeholder URL is never probed: the engine route starts it per run.
    await assertLocalEndpointAvailable({ ...localProvider('http://127.0.0.1/v1'), id: 'tensorfold', source: 'tensorfold' } as Provider, explodingFetch)
    ok('cloud providers, CLI transports, empty baseUrls and the built-in engine are skipped')
  }

  console.log(`  ${passed} passed`)
}

main().catch((err) => { console.error(err); process.exit(1) })
