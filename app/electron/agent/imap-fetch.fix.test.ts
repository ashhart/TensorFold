// Run: jiti electron/agent/imap-fetch.fix.test.ts
// The bug: imap-fetch (and smtp-send) carried a drifted local isPrivateIp copy that missed
// the hex IPv4-mapped IPv6 form (::ffff:7f00:1 = 127.0.0.1, ::ffff:a9fe:a9fe = 169.254.169.254).
// Both now reuse the shared security.ts isPrivateIp; this pins that it covers the hex form so a
// hostile IMAP/SMTP host whose AAAA record returns the mapped internal address is rejected.
import assert from 'node:assert/strict'
import { isPrivateIp } from './security.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('imap-fetch SSRF fix tests:')

{
  assert.equal(isPrivateIp('::ffff:7f00:1'), true) // 127.0.0.1
  ok('hex IPv4-mapped loopback (::ffff:7f00:1) is private')
}

{
  assert.equal(isPrivateIp('::ffff:a9fe:a9fe'), true) // 169.254.169.254 (cloud metadata)
  ok('hex IPv4-mapped metadata addr (::ffff:a9fe:a9fe) is private')
}

{
  // Dotted IPv4-mapped form still works.
  assert.equal(isPrivateIp('::ffff:127.0.0.1'), true)
  ok('dotted IPv4-mapped loopback is private')
}

{
  // A genuine public address is not flagged.
  assert.equal(isPrivateIp('93.184.216.34'), false)
  assert.equal(isPrivateIp('::ffff:5db8:d822'), false) // 93.184.216.34 mapped
  ok('public addresses (dotted and hex-mapped) are not flagged private')
}

console.log(`\n${passed} checks passed.`)
