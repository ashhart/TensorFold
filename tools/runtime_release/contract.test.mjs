import test from 'node:test';
import assert from 'node:assert/strict';
import { sign } from 'node:crypto';
import { DOMAIN, mergeCatalog, preserve, verifyCatalog } from './contract.mjs';
import { signCatalog, verifyNative } from './crypto.mjs';
import { fixture, keys } from './fixtures.mjs';
test('a new engine is discoverable in a signed catalog while previous versions remain immutable', async () => {
  const key = keys(), old = mergeCatalog(null, fixture('1.2.2'));
  const doc = mergeCatalog(old, fixture());
  const raw = await signCatalog(doc, key.privateKey, key.publicKey);
  const verified = await verifyCatalog(raw, key.publicKey);
  assert.deepEqual(verified.packages.map(pkg => pkg.version), ['1.2.4', '1.2.2']);
  assert.deepEqual(verified.packages[1], old.packages[0]);
  assert.equal(mergeCatalog(verified, fixture()).packages.length, 2);
});
test('wrong roots, mutation, expiry and missing grammar cannot produce an accepted feed', async () => {
  const key = keys(), doc = mergeCatalog(null, fixture());
  const raw = await signCatalog(doc, key.privateKey, key.publicKey);
  await assert.rejects(verifyCatalog(raw, keys().publicKey));
  await assert.rejects(verifyCatalog(raw, key.publicKey, doc.expiresAt));
  const broken = structuredClone(doc); broken.packages[0].files.splice(1, 1);
  await assert.rejects(signCatalog(broken, key.privateKey, key.publicKey));
  await assert.rejects(signCatalog(doc, keys().privateKey, key.publicKey));
  const envelope = JSON.parse(raw); envelope.payload = Buffer.from(JSON.stringify({ ...doc, publishedAt: 0 })).toString('base64');
  await assert.rejects(verifyCatalog(JSON.stringify(envelope), key.publicKey));
});
test('publication cannot drop or replace earlier runtimes, including an older delayed release', () => {
  const doc = mergeCatalog(null, fixture());
  assert.throws(() => preserve(doc, mergeCatalog(null, fixture('1.2.3'))));
  const changed = fixture(); changed.sha256 = 'a'.repeat(64);
  assert.throws(() => mergeCatalog(doc, changed));
  const next = mergeCatalog(doc, fixture('1.2.3'));
  assert.deepEqual(next.packages.map(pkg => pkg.version), ['1.2.4', '1.2.3']);
});
test('native signatures are domain separated and bound to the exact archive and version', () => {
  const key = keys(), name = 'tensorfold-1.2.4-macos-arm64.tar.gz';
  const checksum = Buffer.from(`${'a'.repeat(64)}  ${name}\n`);
  const signature = sign(null, Buffer.concat([Buffer.from('tensorfold-native-checksum-v1\n'), checksum]), key.privateKey);
  assert.equal(verifyNative(checksum, signature, '1.2.4', key.publicKey), 'a'.repeat(64));
  assert.throws(() => verifyNative(checksum, signature, '1.2.3', key.publicKey));
  assert.throws(() => verifyNative(checksum, sign(null, Buffer.concat([Buffer.from(DOMAIN), checksum]), key.privateKey), '1.2.4', key.publicKey));
});
