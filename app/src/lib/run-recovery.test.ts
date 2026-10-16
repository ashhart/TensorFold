import assert from 'node:assert/strict'
import { runRecovery } from './run-recovery'

for (const [text, title, destination] of [
  ['HTTP 401 unauthorized', 'Model sign-in needs attention', 'connections'],
  ['HTTP 429 quota exceeded', 'Provider limit reached', 'models'],
  ['Metal: failed to allocate model buffer', 'The model needs more memory', 'models'],
  ['context_length_exceeded', 'This conversation is too large', 'models'],
  ['fetch failed ECONNREFUSED', 'The model endpoint is unavailable', 'connections'],
  ['Provider timed out', 'The model took too long', 'models'],
  ['Denied by read-only approval mode.', 'An action was blocked', 'security'],
  ['A tool reported an unexpected response.', 'The run could not finish', null],
] as const) {
  const recovery = runRecovery(text)
  assert.equal(recovery.title, title)
  assert.equal(recovery.destination, destination)
  assert.ok(recovery.guidance.length > 0)
  assert.ok(!/automatically retry|already undone|nothing changed|success/i.test(recovery.guidance), 'recovery must not imply replay or rollback')
}
console.log('run recovery classification: ok')
