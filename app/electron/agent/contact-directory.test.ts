// Run: jiti electron/agent/contact-directory.test.ts
import assert from 'node:assert/strict'
import {
  contactCount,
  dedupeContacts,
  displayNameFromEmailFrom,
  emailAddressFromFrom,
  emptyContactDirectory,
  isUsableName,
  learnContact,
  learnContacts,
  looksLikePhone,
  normalizeHandle,
  normalizePhone,
  resolveHandle,
  resolveHandles,
  resolveName,
  type ContactDirectory,
} from './contact-directory.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('contact-directory tests:')

// ---- Normalization (E.164 phone, handle, email) -------------------------------

{
  // E.164: punctuation/spacing stripped, "00" intl prefix -> "+", "+" preserved.
  assert.equal(normalizePhone('+44 7700 900123'), '+447700900123')
  assert.equal(normalizePhone('+1 (415) 555-0124'), '+14155550124')
  assert.equal(normalizePhone('0044 7700 900123'), '+447700900123', '00 prefix becomes +')
  assert.equal(normalizePhone('07700900123'), '07700900123', 'bare national kept digits-only (no region to infer +)')
  assert.equal(normalizePhone(''), '')
  assert.equal(normalizePhone('no digits here'), '')
  // The same human number in two formats normalizes to one key.
  assert.equal(normalizePhone('+447700900123'), normalizePhone('+44 7700 900 123'))
  ok('normalizePhone: E.164 normalize, 00->+, +-preserve, dedupe across formats')
}

{
  // normalizeHandle dispatches by platform.
  assert.equal(normalizeHandle('phone', '+44 7700 900123'), '+447700900123')
  assert.equal(normalizeHandle('phone', '447700900123@example.net'), '+447700900123', 'WhatsApp JID -> phone')
  assert.equal(normalizeHandle('email', 'mailto:Maya@example.com'), 'maya@example.com')
  assert.equal(normalizeHandle('slack', '@U024BE7LH'), 'u024be7lh')
  assert.equal(normalizeHandle('discord', 'SomeUser#0001'.toLowerCase()), 'someuser#0001')
  assert.equal(normalizeHandle('slack', ''), '')
  ok('normalizeHandle: phone/JID, email lower+mailto, id lower+@strip')
}

{
  assert.equal(looksLikePhone('+44 7700 900123'), true)
  assert.equal(looksLikePhone('(415) 555-0124'), true)
  assert.equal(looksLikePhone('maya@example.com'), false)
  assert.equal(looksLikePhone('Maya Chen'), false)
  assert.equal(looksLikePhone('U024BE7LH'), false)
  ok('looksLikePhone: phone-shaped strings only')
}

{
  // Display-name + address extraction from an email From header value.
  assert.equal(displayNameFromEmailFrom('"Maya Chen" <maya@example.com>'), 'Maya Chen')
  assert.equal(displayNameFromEmailFrom('Maya Chen <maya@example.com>'), 'Maya Chen')
  assert.equal(displayNameFromEmailFrom('maya@example.com'), '', 'bare address has no display name')
  assert.equal(displayNameFromEmailFrom('<maya@example.com>'), '', 'address-only in brackets has no name')
  assert.equal(emailAddressFromFrom('"Maya Chen" <Maya@example.com>'), 'maya@example.com')
  assert.equal(emailAddressFromFrom('maya@example.com'), 'maya@example.com')
  ok('displayNameFromEmailFrom/emailAddressFromFrom: split name vs address')
}

// ---- Usable-name gate ---------------------------------------------------------

{
  assert.equal(isUsableName('Maya Chen', '+447700900123'), true)
  assert.equal(isUsableName('', '+447700900123'), false)
  assert.equal(isUsableName('Unknown', '+447700900123'), false, 'generic placeholder rejected')
  assert.equal(isUsableName('You', '+447700900123'), false)
  assert.equal(isUsableName('+44 7700 900123', '+447700900123'), false, 'the number reformatted is not a name')
  assert.equal(isUsableName('u024be7lh', 'u024be7lh'), false, 'name == handle is not a name')
  ok('isUsableName: rejects empty/generic/handle-as-name')
}

// ---- Learning + resolution (name <-> handle) ----------------------------------

{
  let dir = emptyContactDirectory(1)
  // Learn a phone -> name from an iMessage-style observation.
  dir = learnContact(dir, { platform: 'phone', handle: '+44 7700 900123', name: 'Maya Chen', source: 'imessage' }, 10)
  assert.equal(contactCount(dir), 1)
  // Resolve forward: raw handle (any format) -> name.
  assert.equal(resolveName(dir, 'phone', '+447700900123'), 'Maya Chen')
  assert.equal(resolveName(dir, 'phone', '0044 7700 900123'), 'Maya Chen', 'different format resolves to same record')
  assert.equal(resolveName(dir, 'phone', '+15550000000'), null, 'unknown handle => null')
  // Resolve backward: "reply to <name>" -> handle record.
  const back = resolveHandle(dir, 'maya chen', 'phone')
  assert.equal(back?.handle, '+447700900123')
  assert.equal(resolveHandle(dir, 'maya')?.handle, '+447700900123', 'partial name matches')
  assert.equal(resolveHandle(dir, 'nobody'), null)
  ok('learn + resolveName + resolveHandle: bidirectional, format-insensitive')
}

{
  // Confidence: a high-signal address-book name is NOT clobbered by a low-signal one,
  // but a same-or-higher-confidence update replaces it. Count always increments.
  let dir = emptyContactDirectory(1)
  dir = learnContact(dir, { platform: 'phone', handle: '+14155550124', name: 'Alex Rivera', confidence: 'high' }, 1)
  dir = learnContact(dir, { platform: 'phone', handle: '+14155550124', name: 'alex (mobile)', confidence: 'low' }, 2)
  assert.equal(resolveName(dir, 'phone', '+14155550124'), 'Alex Rivera', 'low-confidence does not overwrite high')
  dir = learnContact(dir, { platform: 'phone', handle: '+14155550124', name: 'Alexandra Rivera', confidence: 'high' }, 3)
  assert.equal(resolveName(dir, 'phone', '+14155550124'), 'Alexandra Rivera', 'equal-confidence update replaces')
  assert.equal(dir.records['phone:+14155550124'].count, 3, 'count increments on every observation')
  ok('learnContact: confidence-gated name updates, count tracks sightings')
}

{
  // A handle observed with no usable name still registers (so the handle is known) but
  // resolveName returns null until a real name arrives.
  let dir = emptyContactDirectory(1)
  dir = learnContact(dir, { platform: 'discord', handle: '123456789', name: undefined }, 1)
  assert.equal(contactCount(dir), 1)
  assert.equal(resolveName(dir, 'discord', '123456789'), null, 'handle known, name unknown')
  dir = learnContact(dir, { platform: 'discord', handle: '123456789', name: 'Sam' }, 2)
  assert.equal(resolveName(dir, 'discord', '123456789'), 'Sam')
  ok('learnContact: nameless observation registers handle, later name fills in')
}

{
  // Platform scoping + disambiguation: two different people share a first name.
  let dir = emptyContactDirectory(1)
  dir = learnContacts(dir, [
    { platform: 'phone', handle: '+1111', name: 'Alex Stone', confidence: 'high' },
    { platform: 'slack', handle: 'UALEX', name: 'Alex Wong', confidence: 'medium' },
  ], 1)
  // Scoped resolve picks within the platform.
  assert.equal(resolveHandle(dir, 'Alex', 'slack')?.handle, 'ualex')
  assert.equal(resolveHandle(dir, 'Alex', 'phone')?.handle, '+1111')
  // Unscoped resolveHandles returns every match, best-first (high confidence first).
  const all = resolveHandles(dir, 'Alex')
  assert.deepEqual(all.map((r) => r.name), ['Alex Stone', 'Alex Wong'])
  ok('resolveHandle/resolveHandles: platform scoping + multi-match disambiguation')
}

// ---- Dedupe -------------------------------------------------------------------

{
  // Two records that normalize to the same (platform, handle) are merged; counts sum and
  // the better-named/higher-confidence record's name survives. Idempotent.
  const dir: ContactDirectory = {
    updatedAt: 1,
    records: {
      // Same human number, two different stored keys (one already-normalized, one raw-ish
      // record whose rawHandle re-normalizes onto the first).
      'phone:+447700900123': { handle: '+447700900123', rawHandle: '+44 7700 900123', platform: 'phone', name: 'Maya Chen', source: 'imessage', confidence: 'high', count: 2, updatedAt: 5 },
      'phone:447700900123dup': { handle: '447700900123dup', rawHandle: '0044 7700 900123', platform: 'phone', name: '', source: 'whatsapp', confidence: 'low', count: 3, updatedAt: 9 },
    },
  }
  const merged = dedupeContacts(dir, 100)
  assert.equal(contactCount(merged), 1, 'the two near-dupes collapse to one record')
  const rec = merged.records['phone:+447700900123']
  assert.equal(rec.name, 'Maya Chen', 'the named/higher-confidence record wins')
  assert.equal(rec.count, 5, 'counts are summed (2 + 3)')
  assert.equal(rec.updatedAt, 9, 'freshest timestamp kept')
  // Idempotent: dedupe of an already-canonical directory is unchanged in shape.
  const again = dedupeContacts(merged, 200)
  assert.equal(contactCount(again), 1)
  assert.equal(again.records['phone:+447700900123'].count, 5)
  ok('dedupeContacts: merges normalized dupes, sums counts, keeps best name, idempotent')
}

console.log(`contact-directory: ${passed} checks passed`)
