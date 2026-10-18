import assert from 'node:assert/strict'
import { runModelHelper } from './model-helper'
import type { ModelHelperRequest } from '../../shared/model-helper'
const request: ModelHelperRequest = { requestId: 'synthetic-id', role: 'search', threadId: 'chat-one', text: 'marker' }
let calls = 0, current = true
const options = { signal: new AbortController().signal, resolve: async () => ({ provider: {} as any, route: { providerId: 'local', accountId: null, model: 'fixture' }, assertCurrent: () => { if (!current) throw new Error('changed') } }), search: async () => ({ text: '', sources: [] }), complete: async () => { calls++; return 'Synthetic output' } }
async function main() {
  await assert.rejects(runModelHelper(request, options), /No current/); assert.equal(calls, 0)
  const result = await runModelHelper({ ...request, role: 'document_extraction' }, options)
  assert.equal(result.complete, true); assert.match(result.warning!, /Review/)
  await assert.rejects(runModelHelper({ ...request, text: 'x'.repeat(20_001) }, options), /20,000/)
  current = false; await assert.rejects(runModelHelper({ ...request, role: 'naming' }, options), /changed/); assert.equal(calls, 1)
  current = true
  await assert.rejects(runModelHelper({ ...request, role: 'naming' }, { ...options, complete: async () => { current = false; return 'partial' } }), /changed/)
  console.log('Helper jobs refuse missing sources and route drift, bound text/output and never substitute a provider')
}
void main().catch(error => { console.error(error); process.exitCode = 1 })
