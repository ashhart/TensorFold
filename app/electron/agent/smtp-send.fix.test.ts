// Run: jiti electron/agent/smtp-send.fix.test.ts
// The bug: SMTP From was set to the IMAP login `user`, which is often NOT a full email
// address, producing an invalid envelope. resolveFromAddress prefers an explicit
// fromAddress and only falls back to `user` when it parses as an email.
import assert from 'node:assert/strict'
import { resolveFromAddress } from './smtp-send.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('smtp-send fix tests:')

{
  assert.equal(resolveFromAddress({ fromAddress: 'maya@example.com', user: 'maya' }), 'maya@example.com')
  ok('explicit fromAddress is used over the login username')
}

{
  assert.equal(resolveFromAddress({ user: 'maya@example.com' }), 'maya@example.com')
  ok('login username is used when it itself parses as an email')
}

{
  // A bare login username is not a valid From — must throw rather than send a broken envelope.
  assert.throws(() => resolveFromAddress({ user: 'maya' }), /valid From address/)
  ok('bare (non-email) login username throws instead of producing an invalid From')
}

{
  // domain\login style is also not a valid From.
  assert.throws(() => resolveFromAddress({ user: 'CORP\\maya' }), /valid From address/)
  ok('domain\\login username throws')
}

{
  // An explicit fromAddress that is itself invalid is rejected (no silent fallback to a bad value).
  assert.throws(() => resolveFromAddress({ fromAddress: 'not-an-email', user: 'maya@example.com' }), /not a valid email/)
  ok('an explicit but invalid fromAddress is rejected')
}

console.log(`\n${passed} checks passed.`)
