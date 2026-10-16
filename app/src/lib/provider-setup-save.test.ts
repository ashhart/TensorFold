import assert from 'node:assert/strict'
import { persistProviderSetup } from './provider-setup-save'
import type { Provider } from './types'

const provider: Provider = { id: 'new-provider', name: 'Example provider', kind: 'cloud', baseUrl: 'https://example.invalid/v1', apiKey: '', models: ['example-model'] }
const draft = { providers: [] as Provider[], activeProviderId: null as string | null, activeModel: 'default' }
let confirm!: () => void
const gate = new Promise<void>((resolve) => { confirm = resolve })
let reportedSuccess = false
let sent: typeof draft | null = null
const adding = persistProviderSetup(draft, provider, undefined, async (next) => { sent = next; await gate }).then(() => { reportedSuccess = true })
await Promise.resolve(); await Promise.resolve()
assert.equal(reportedSuccess, false, 'provider setup cannot confirm success while persistence is pending')
confirm(); await adding
assert.equal(reportedSuccess, true)
assert.equal(sent!.activeProviderId, provider.id)
assert.equal(sent!.activeModel, 'example-model')
await assert.rejects(persistProviderSetup(draft, provider, undefined, async () => { throw new Error('Synthetic write failure') }), /Synthetic write failure/, 'provider setup must expose a failed save to its form owner')
console.log('Provider setup awaits persistence and reports rejection')
