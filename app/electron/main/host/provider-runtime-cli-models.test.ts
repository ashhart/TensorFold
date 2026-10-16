import assert from 'node:assert/strict'
import { defaultSettings } from '../../store'
import type { Provider } from '../../agent/types'
import { store } from '../state'
import { listProviderModels } from './provider-runtime'

const saved = store.settings
try {
  const transports = [
    ['pi-cli', 'pi-default'],
    ['cursor-cli', 'cursor-default'],
    ['gemini-cli', 'gemini-default'],
  ] as const
  for (const [transport, sentinel] of transports) {
    const provider = { id: transport, name: transport, kind: 'cloud', transport, baseUrl: '', apiKey: '', cliPath: '/bin/echo', models: [] } as Provider
    store.settings = { ...defaultSettings(), providers: [provider] }
    assert.deepEqual(await listProviderModels(provider), [sentinel], `${transport} Connect/Test uses its CLI model contract instead of fetching /models from an empty URL`)
    const missing = { ...provider, cliPath: `/definitely/not/a/vontra-${transport}` }
    store.settings = { ...defaultSettings(), providers: [missing] }
    await assert.rejects(() => listProviderModels(missing), /CLI is not installed/, `${transport} Connect/Test fails clearly when the configured binary is missing`)
  }
} finally {
  store.settings = saved
}

console.log('provider-runtime CLI models: Pi/Cursor/Gemini use binary readiness + curated model contracts')
