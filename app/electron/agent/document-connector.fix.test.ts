// Run: jiti electron/agent/document-connector.fix.test.ts
// The bug: extractPdfText threw 'Encrypted PDF extraction is not supported' on ANY byte-level
// occurrence of the literal '/Encrypt', rejecting readable non-encrypted PDFs. A PDF is encrypted
// only when its trailer references an /Encrypt dictionary via an indirect reference (/Encrypt N G R).
import assert from 'node:assert/strict'
import { isEncryptedPdf } from './document-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('document-connector fix tests:')

{
  // A real trailer encryption reference is detected.
  assert.equal(isEncryptedPdf('trailer\n<< /Size 10 /Root 1 0 R /Encrypt 9 0 R >>'), true)
  ok('trailer /Encrypt indirect reference is detected as encrypted')
}

{
  // The bare literal inside a content stream / object body is NOT encryption.
  assert.equal(isEncryptedPdf('... text about the /Encrypt entry of the trailer ...'), false)
  ok('a bare /Encrypt literal (no indirect ref) is not treated as encrypted')
}

{
  // A PDF documenting PDF internals mentioning /Encrypt as a word is readable.
  assert.equal(isEncryptedPdf('(This document explains /Encrypt and /Filter keys) Tj'), false)
  ok('a /Encrypt mention in page text is not treated as encrypted')
}

{
  // Spacing variants of a genuine reference still match.
  assert.equal(isEncryptedPdf('/Encrypt 123 0 R'), true)
  assert.equal(isEncryptedPdf('/Encrypt   45   0   R'), true)
  ok('whitespace variants of the indirect reference are detected')
}

console.log(`\n${passed} checks passed.`)
