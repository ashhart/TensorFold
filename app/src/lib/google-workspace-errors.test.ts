import assert from 'node:assert/strict'
import { GOOGLE_TOKEN_FAILURES, googleRequestError } from '../../shared/google-workspace-errors'

for (const message of Object.values(GOOGLE_TOKEN_FAILURES)) {
  assert.equal(googleRequestError(new Error(message)), message)
  assert.equal(googleRequestError(new Error(`Error invoking remote method 'google-workspace:messages': Error: ${message}`)), message)
}
for (const message of ['Google is rate limiting this connection. Try again later.',
  'Could not reach Google. Check your connection.',
  'Google permission gmail.read is not granted. Reconnect with that permission selected.']) {
  assert.equal(googleRequestError(new Error(message)), message)
}
for (const value of [new Error('synthetic-secret=/synthetic/private/path'), new Error('invalid_client synthetic-secret'),
  new Error(GOOGLE_TOKEN_FAILURES.invalid_client + ' synthetic-secret'), { message: GOOGLE_TOKEN_FAILURES.invalid_client }]) {
  assert.doesNotMatch(googleRequestError(value), /synthetic-secret|synthetic\/private|Desktop OAuth/)
}
console.log('Google recovery exposes only complete static diagnostics, including wrapped IPC errors')
