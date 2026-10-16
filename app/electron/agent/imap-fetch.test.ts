// Run: jiti electron/agent/imap-fetch.test.ts
import assert from 'node:assert/strict'
import { extractAttachmentParts, headersToObject, isImapSource, isUnreadFromFlags, parseImapId, validateImapSource, type BodyStructureNode } from './imap-fetch.ts'
import type { EmailSourceConfig } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function makeSource(partial: Partial<EmailSourceConfig>): EmailSourceConfig {
  return {
    id: 's1', label: 'Work', path: '', kind: 'imap', enabled: true,
    createdAt: 0, updatedAt: 0, lastSyncAt: null,
    host: 'imap.example.com', port: 993, tls: true, user: 'me@example.com',
    password: 'hunter2', mailbox: 'INBOX',
    ...partial,
  }
}

console.log('imap-fetch tests:')

{
  // imapflow returns FetchMessageObject.flags as a Set<string>. The old check fell through to
  // msg.flags.includes() when '\\Seen' was absent (the common unread case) — Set has no
  // .includes, so it threw TypeError and aborted the whole fetch on the first unread message.
  assert.equal(isUnreadFromFlags(new Set(['\\Seen'])), false, 'Set with \\Seen => read')
  assert.equal(isUnreadFromFlags(new Set(['\\Flagged'])), true, 'Set without \\Seen => unread, no throw')
  assert.equal(isUnreadFromFlags(new Set()), true, 'empty Set => unread')
  assert.equal(isUnreadFromFlags(['\\Seen', '\\Answered']), false, 'array with \\Seen => read')
  assert.equal(isUnreadFromFlags(['\\Recent']), true, 'array without \\Seen => unread')
  assert.equal(isUnreadFromFlags(undefined), true, 'no flags => unread')
  assert.equal(isUnreadFromFlags(null), true, 'null flags => unread')
  ok('isUnreadFromFlags handles Set, array, and missing flags without throwing')
}

{
  assert.equal(isImapSource(makeSource({ kind: 'imap' })), true)
  assert.equal(isImapSource(makeSource({ kind: 'eml_dir' })), false)
  assert.equal(isImapSource(makeSource({ kind: 'maildir' })), false)
  ok('isImapSource distinguishes imap from file sources')
}

{
  // A complete IMAP source validates cleanly.
  assert.equal(validateImapSource(makeSource({})), null)
  // Missing host/user/password is rejected before any network attempt.
  assert.match(validateImapSource(makeSource({ host: '' }))!, /host is required/i)
  assert.match(validateImapSource(makeSource({ user: '' }))!, /username is required/i)
  assert.match(validateImapSource(makeSource({ password: '' }))!, /password is required/i)
  ok('validateImapSource requires host/user/password')
}

{
  // mailbox:uid ids round-trip through parseImapId.
  const parsed = parseImapId('INBOX:42')
  assert.equal(parsed.mailbox, 'INBOX')
  assert.equal(parsed.uid, 42)
  // A malformed id yields a null uid (readImapMessage returns null rather than fetching).
  assert.equal(parseImapId('garbage').uid, null)
  ok('parseImapId parses mailbox:uid and rejects malformed ids')
}

{
  // With `headers: true`, imapflow hands back the raw header block as a Buffer.
  // headersToObject must parse it as RFC 5322 text (lowercased keys, unfolded
  // continuations), not as a plain object (which would yield byte-index keys).
  const raw = [
    'Message-ID: <abc@example.com>',
    'References: <root@example.com>',
    ' <reply1@example.com>',
    'In-Reply-To: <reply1@example.com>',
    'Subject: Re: hello',
    '',
    '',
  ].join('\r\n')
  const buf = Buffer.from(raw, 'utf8')
  const h = headersToObject(buf)
  assert.equal(h['message-id'], '<abc@example.com>')
  // Folded References header is unfolded into a single value spanning both ids.
  assert.equal(h['references'], '<root@example.com> <reply1@example.com>')
  assert.equal(h['in-reply-to'], '<reply1@example.com>')
  assert.equal(h['subject'], 'Re: hello')
  // No spurious numeric byte-index keys from Buffer-as-object iteration.
  assert.equal(h['0'], undefined)
  // A string is parsed identically.
  assert.equal(headersToObject(raw)['message-id'], '<abc@example.com>')
  // Repeated headers (e.g. Received) are joined, not silently dropped.
  const multi = headersToObject(Buffer.from('Received: a\r\nReceived: b\r\n\r\n', 'utf8'))
  assert.equal(multi['received'], 'a, b')
  ok('headersToObject parses a raw Buffer header block into lowercased headers')
}

// ---- Attachment MIME-part extraction (pure) -----------------------------------

{
  // A multipart/mixed: an inline text body part is SKIPPED; a PDF with disposition=attachment
  // and an inline image carrying only a Content-Type name are both returned as attachments.
  const structure: BodyStructureNode = {
    part: '', type: 'multipart/mixed', childNodes: [
      { part: '1', type: 'text/plain', size: 120 }, // inline body — skipped
      { part: '2', type: 'application/pdf', disposition: 'attachment', dispositionParameters: { filename: 'invoice.pdf' }, size: 9000 },
      { part: '3', type: 'image/png', parameters: { name: 'logo.png' }, size: 4000 }, // inline image with a name => attachment
    ],
  }
  const parts = extractAttachmentParts(structure)
  assert.equal(parts.length, 2, 'inline text body part is not an attachment')
  assert.deepEqual(parts.map((p) => p.part), ['2', '3'])
  assert.equal(parts[0].filename, 'invoice.pdf')
  assert.equal(parts[0].contentType, 'application/pdf')
  assert.equal(parts[1].filename, 'logo.png', 'Content-Type name is used when disposition filename is absent')
  assert.equal(parts[1].contentType, 'image/png')
  ok('extractAttachmentParts: returns attachment+named-inline parts, skips the inline text body')
}

{
  // A simple single-part text/plain message (no childNodes, no filename) yields NO attachments —
  // it is the message body, not an attachment.
  assert.deepEqual(extractAttachmentParts({ part: '1', type: 'text/plain', size: 50 }), [])
  // text/html inline body is likewise skipped.
  assert.deepEqual(extractAttachmentParts({ part: '1', type: 'text/html', size: 80 }), [])
  // An attachment node missing a `part` id can't be fetched, so it is dropped.
  assert.deepEqual(extractAttachmentParts({ type: 'application/zip', disposition: 'attachment', dispositionParameters: { filename: 'a.zip' } }), [])
  // Null / undefined structure is safe.
  assert.deepEqual(extractAttachmentParts(null), [])
  assert.deepEqual(extractAttachmentParts(undefined), [])
  ok('extractAttachmentParts: skips inline body, partless nodes, and null structures')
}

{
  // Nested multipart (multipart/mixed > [multipart/alternative(text+html), attachment]) recurses:
  // the alternative's text+html bodies are skipped, the nested attachment is found.
  const nested: BodyStructureNode = {
    part: '', type: 'multipart/mixed', childNodes: [
      { part: '1', type: 'multipart/alternative', childNodes: [
        { part: '1.1', type: 'text/plain' },
        { part: '1.2', type: 'text/html' },
      ] },
      { part: '2', type: 'image/jpeg', disposition: 'attachment', dispositionParameters: { filename: 'photo.jpg' }, size: 12345 },
    ],
  }
  const parts = extractAttachmentParts(nested)
  assert.deepEqual(parts.map((p) => p.part), ['2'])
  assert.equal(parts[0].size, 12345, 'declared part size is carried through')
  ok('extractAttachmentParts: recurses nested multipart and skips alternative bodies')
}

{
  // The per-message cap bounds the fan-out: only the first N attachment parts are returned.
  const many: BodyStructureNode = {
    part: '', type: 'multipart/mixed',
    childNodes: Array.from({ length: 12 }, (_v, i) => ({ part: String(i + 1), type: 'application/octet-stream', disposition: 'attachment', dispositionParameters: { filename: `f${i}.bin` } })),
  }
  assert.equal(extractAttachmentParts(many, 10).length, 10, 'capped at maxParts')
  assert.equal(extractAttachmentParts(many, 3).length, 3)
  ok('extractAttachmentParts: caps the number of parts at maxParts')
}

console.log(`\n${passed} checks passed.`)
