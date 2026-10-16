import assert from 'node:assert/strict'
import { classifyBlockers, hasUserBlocker } from './loop-blockers'

const b = classifyBlockers([
  'Missing STRIPE_API_KEY env var',
  'Should the API return 429 or 503 when rate-limited?',
  'npm registry returned 503 (network)',
  'rm -rf / would be required',
  'Add a null check in parser.ts',
])
assert.equal(b.find((x) => /STRIPE/.test(x.issue))!.kind, 'NEEDS_USER_SECRET')
assert.equal(b.find((x) => /429 or 503/.test(x.issue))!.kind, 'NEEDS_PRODUCT_DECISION')
assert.equal(b.find((x) => /registry/.test(x.issue))!.kind, 'EXTERNAL_SERVICE_DOWN')
assert.equal(b.find((x) => /rm -rf/.test(x.issue))!.kind, 'UNSAFE_ACTION')
assert.equal(b.find((x) => /null check/.test(x.issue))!.kind, 'FIXABLE_BY_AGENT')

assert.equal(hasUserBlocker(b), true) // has non-fixable
assert.equal(hasUserBlocker(classifyBlockers(['fix the typo'])), false) // all fixable

console.log('loop-blockers tests passed')
