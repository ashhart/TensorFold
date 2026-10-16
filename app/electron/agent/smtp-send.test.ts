// Run: jiti electron/agent/smtp-send.test.ts
import assert from 'node:assert/strict'
import { resolveSmtpTransport, resolveReferences } from './smtp-send.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('smtp-send tests:')

{
  // The bug: IMAP accounts are created with port 993 (implicit-TLS IMAPS). That
  // port can't speak SMTP, so it must NOT be reused — fall back to 587/STARTTLS.
  const r = resolveSmtpTransport({ port: 993 })
  assert.equal(r.port, 587)
  assert.equal(r.secure, false)
  ok('IMAPS 993 falls back to SMTP submission 587 (STARTTLS)')
}

{
  // Plaintext IMAP port 143 is also an IMAP port, never SMTP.
  const r = resolveSmtpTransport({ port: 143 })
  assert.equal(r.port, 587)
  assert.equal(r.secure, false)
  ok('IMAP 143 falls back to SMTP 587')
}

{
  // Missing port -> default SMTP submission port.
  const r = resolveSmtpTransport({ port: undefined })
  assert.equal(r.port, 587)
  assert.equal(r.secure, false)
  ok('undefined port defaults to 587')
}

{
  // A genuine SMTP override is honored. 465 = implicit TLS.
  const r = resolveSmtpTransport({ port: 465 })
  assert.equal(r.port, 465)
  assert.equal(r.secure, true)
  ok('explicit 465 honored as implicit-TLS SMTP (secure:true)')
}

{
  // 587 as an explicit override stays 587, STARTTLS.
  const r = resolveSmtpTransport({ port: 587 })
  assert.equal(r.port, 587)
  assert.equal(r.secure, false)
  ok('explicit 587 stays STARTTLS (secure:false)')
}

{
  // Any non-IMAP custom SMTP port is honored as-is (e.g. 2525), STARTTLS.
  const r = resolveSmtpTransport({ port: 2525 })
  assert.equal(r.port, 2525)
  assert.equal(r.secure, false)
  ok('custom SMTP port 2525 honored, secure:false')
}

{
  // [LOW #25] The bug: the References header was set to the internal connector
  // threadKey (e.g. 'email:abc123'), not a real Message-ID — breaking mail-client
  // threading. When a real Message-ID (inReplyTo) is on the same payload, References
  // must use it, same as In-Reply-To, instead of the internal navigational threadKey.
  const r = resolveReferences('<real-message-id@example.com>', 'email:abc123')
  assert.equal(r, '<real-message-id@example.com>')
  ok('References uses the real Message-ID (inReplyTo) over the internal threadKey when both are present')
}

{
  // No real Message-ID available — omit References. An internal connector key is
  // not an RFC 5322 Message-ID and would corrupt mail-client threading.
  const r = resolveReferences(undefined, 'email:abc123')
  assert.equal(r, undefined)
  ok('References omits internal threadKey when inReplyTo is absent')
}

{
  // Neither present -> undefined (no References header).
  const r = resolveReferences(undefined, undefined)
  assert.equal(r, undefined)
  ok('References is undefined when neither inReplyTo nor threadKey is present')
}

console.log(`\n${passed} checks passed.`)
