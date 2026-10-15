// Run: jiti electron/agent/email-connector.fix.test.ts
// Focused tests for the data-connectors bug-fix wave:
//   - decodeQuotedPrintable reassembles UTF-8 multibyte sequences (was per-byte mojibake)
//   - imapUidOf parses IMAP UIDs and rejects file-source ids (IMAP gated by UID, not Date)
import assert from 'node:assert/strict'
import { decodeQuotedPrintable, imapUidOf } from './email-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('email-connector fix tests:')

{
  // The bug: 'caf=C3=A9' was decoded byte-by-byte to 'cafÃ©'. The C3 A9 bytes are the
  // UTF-8 encoding of 'é' and must reassemble.
  assert.equal(decodeQuotedPrintable('caf=C3=A9'), 'café')
  ok('quoted-printable reassembles a UTF-8 2-byte sequence (café)')
}

{
  // Cyrillic 'Привет' is multibyte UTF-8; per-byte decoding would corrupt it.
  const encoded = [...Buffer.from('Привет', 'utf8')].map((b) => '=' + b.toString(16).toUpperCase().padStart(2, '0')).join('')
  assert.equal(decodeQuotedPrintable(encoded), 'Привет')
  ok('quoted-printable reassembles Cyrillic (Привет)')
}

{
  // An emoji is a 4-byte UTF-8 sequence.
  const encoded = [...Buffer.from('🎉', 'utf8')].map((b) => '=' + b.toString(16).toUpperCase().padStart(2, '0')).join('')
  assert.equal(decodeQuotedPrintable(encoded), '🎉')
  ok('quoted-printable reassembles a 4-byte emoji')
}

{
  // Soft line breaks are removed; ASCII passes through unchanged.
  assert.equal(decodeQuotedPrintable('Hello=\r\n World'), 'Hello World')
  assert.equal(decodeQuotedPrintable('plain ascii text'), 'plain ascii text')
  ok('soft line breaks removed; ASCII unchanged')
}

{
  // Lowercase hex is accepted too.
  assert.equal(decodeQuotedPrintable('caf=c3=a9'), 'café')
  ok('lowercase hex escapes decode correctly')
}

{
  // A literal non-ASCII char already in the source is preserved (its UTF-8 bytes survive
  // the single charset decode).
  assert.equal(decodeQuotedPrintable('é and =C3=A9'), 'é and é')
  ok('literal non-ASCII chars are preserved alongside QP escapes')
}

{
  // IMAP ids are `<mailbox>:<uid>` with an imap:// path; the UID must be recovered.
  assert.equal(imapUidOf({ path: 'imap://acct/INBOX', id: 'INBOX:4271' }), 4271)
  ok('imapUidOf parses the UID from an imap:// message id')
}

{
  // File-source messages (no imap:// path) are not UID-gated.
  assert.equal(imapUidOf({ path: '/Users/x/Mail/cur/msg.eml', id: 'abc123' }), null)
  assert.equal(imapUidOf({ path: 'imap://acct/INBOX', id: 'no-colon' }), null)
  ok('imapUidOf returns null for non-IMAP ids and ids without a UID')
}

console.log(`\n${passed} checks passed.`)
