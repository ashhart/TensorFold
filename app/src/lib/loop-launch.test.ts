import assert from 'node:assert/strict'
import { parseVerificationDraft } from './loop-launch'

assert.deepEqual(parseVerificationDraft(' npm test\r\n\n npm run typecheck '), {
  commands: ['npm test', 'npm run typecheck'], error: null,
})
assert.ok(parseVerificationDraft(' \n ').error)
assert.ok(parseVerificationDraft(Array(33).fill('npm test').join('\n')).error)
assert.ok(parseVerificationDraft('x'.repeat(2001)).error)
assert.ok(parseVerificationDraft('npm\0test').error)
assert.deepEqual(parseVerificationDraft('npm test\nnpm test').commands, ['npm test', 'npm test'])
console.log('Loop verification input preserves command order and rejects missing or unbounded checks.')
