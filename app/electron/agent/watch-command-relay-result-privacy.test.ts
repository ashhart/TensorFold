import assert from 'node:assert/strict'
import {createHostWatchCommandRelay} from './watch-command-relay'

const caps = {taskStart: true, taskGuide: true, taskStop: true, approvalResolve: false, transcriptReview: true, changesReview: false, engineControl: false}
for (const change of ['incognito', 'phoneLocalOnly', 'transcriptReview'] as const) {
  let currentCaps = {...caps}, mutations = 0
  const snapshot = {serverName: 'Synthetic host', activeThreadIds: ['job-test'], work: {jobs: {running: 1, queued: 0}, approvals: {pending: 0}}, threads: [
    {id: 'job-test', title: 'SYNTHETIC_LATE_PRIVATE_TITLE', kind: 'task', incognito: false, phoneLocalOnly: false, providerId: 'example', model: 'example', status: 'idle', updatedAt: 1, timeline: [{id: 'assistant-test', kind: 'assistant', text: 'SYNTHETIC_LATE_PRIVATE_REPLY'}]},
  ]}
  const relay = createHostWatchCommandRelay({now: () => 100, authorize: () => ({ok: true, sessionId: 'session-test', scope: 'paired'}), request: async (_method, path) => {
    if (path === '/v1/device-status') return {status: 200, value: {scope: 'paired', capabilities: structuredClone(currentCaps), engine: null}}
    if (path === '/v1/snapshot') return {status: 200, value: structuredClone(snapshot)}
    mutations++
    if (change === 'transcriptReview') currentCaps = {...caps, transcriptReview: false}
    else snapshot.threads[0][change] = true
    return {status: 202, value: {queued: true}}
  }})
  try {
    const command = {version: 1, id: 'request-test', action: 'reply', session: 'session-test', generatedAt: 100, threadId: 'job-test', text: 'Continue synthetic task'}
    const result = await relay.handle('fixture-only-token', command, new AbortController().signal)
    assert.equal(result.ok, true); assert.equal(mutations, 1)
    assert.equal(result.snapshot, undefined, `${change}: a successful reply returns only the receipt`)
    assert.doesNotMatch(JSON.stringify(result), /SYNTHETIC_LATE_PRIVATE_(TITLE|REPLY)/)
    const duplicate = await relay.handle('fixture-only-token', command, new AbortController().signal)
    assert.equal(duplicate.ok, true); assert.equal(duplicate.snapshot, undefined); assert.equal(mutations, 1)
  } finally { await relay.stop() }
}
console.log('Host watch reply receipts cannot return an earlier projection after private/local-only/role changes during dispatch')
