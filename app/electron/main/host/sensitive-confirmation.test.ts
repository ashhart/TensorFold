import assert from 'node:assert/strict'
import { requireNativeConfirmation } from './core-ipc.ts'

const request = {
  title: 'Sensitive action',
  message: 'Confirm locally',
  detail: 'Test',
  confirmLabel: 'Confirm',
}

let calls = 0
await requireNativeConfirmation(async (input) => {
  calls++
  assert.deepEqual(input, request)
  return true
}, request)
assert.equal(calls, 1)

await assert.rejects(
  () => requireNativeConfirmation(async () => false, request),
  /cancelled by the local user/i,
  'a compromised/buggy renderer cannot proceed after the native user declines',
)

await assert.rejects(
  () => requireNativeConfirmation(undefined, request),
  /native confirmation is unavailable/i,
  'missing main-process confirmation wiring fails closed',
)

console.log('sensitive-confirmation.test.ts passed')
