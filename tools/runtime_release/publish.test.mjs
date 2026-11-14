import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { sign } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { fixture, keys, release } from './fixtures.mjs';
import { hash, signCatalog, trust } from './crypto.mjs';
import { mergeCatalog, preserve, verifyCatalog } from './contract.mjs';
import { prepare, verifyPrepared } from './prepare.mjs';
import { publish } from './publish.mjs';

async function world() {
  const key = keys(), root = await mkdtemp(path.join(tmpdir(), 'tf-publication-fixture-'));
  execFileSync('python3', ['-c', 'from pathlib import Path;from test_archive import Archives;import sys;Archives().archive(Path(sys.argv[1]))', root],
    { cwd: fileURLToPath(new URL('.', import.meta.url)) });
  const name = 'tensorfold-1.2.4-macos-arm64.tar.gz';
  const native = await readFile(path.join(root, name));
  const checksum = Buffer.from(`${hash(native)}  ${name}\n`);
  const signature = sign(null, Buffer.concat([Buffer.from('tensorfold-native-checksum-v1\n'), checksum]), key.privateKey);
  const blobs = new Map([[name, native], [`${name}.sha256`, checksum], [`${name}.sha256.sig`, signature]]);
  let feed = await signCatalog(mergeCatalog(null, fixture('1.2.2')), key.privateKey, key.publicKey);
  const writes = [], doc = release();
  function updateAssets() {
    doc.assets = [...blobs].map(([name, bytes]) => ({ name, size: bytes.length, digest: 'sha256:' + hash(bytes),
      browser_download_url: `https://github.com/ashhart/TensorFold/releases/download/v1.2.4/${name}` }));
  }
  updateAssets();
  let outage = false, failReadAfterPublish = false;
  const fetchImpl = async (url, init) => {
    const target = new URL(url);
    if (url === trust.feed) {
      assert.equal(init.headers.Authorization, undefined);
      if (init.method === 'POST') {
        writes.push('feed');
        if (outage) return new Response('unavailable', { status: 503 });
        const old = await verifyCatalog(feed, key.publicKey), incoming = await verifyCatalog(init.body, key.publicKey);
        preserve(old, incoming); feed = init.body;
        if (failReadAfterPublish) return new Response('lost acknowledgement', { status: 503 });
        return Response.json({ state: 'published' });
      }
      return new Response(feed);
    }
    if (target.hostname === 'api.github.com') return Response.json(doc);
    if (target.hostname === 'uploads.github.com') {
      assert.equal(init.headers.Authorization, 'Bearer fixture-scoped-token');
      writes.push('asset');
      const file = target.searchParams.get('name');
      assert.ok(!blobs.has(file), 'immutable bytes must never be replaced');
      blobs.set(file, Buffer.from(init.body)); updateAssets();
      return Response.json({ name: file });
    }
    assert.equal(init.headers.Authorization, undefined);
    const body = blobs.get(decodeURIComponent(target.pathname.split('/').at(-1)));
    return body ? new Response(body) : new Response('missing', { status: 404 });
  };
  const network = { token: 'fixture-scoped-token', fetchImpl };
  const candidate = path.join(root, 'candidate');
  const options = { network, publicKey: key.publicKey, probe: async (directory, pkg) => {
    await writeFile(path.join(directory, 'probe.json'), JSON.stringify({ version: pkg.version, schema: 1, backend: 'metal',
      grammar: true, executableSha256: pkg.files.find(file => file.path === pkg.entrypoint).sha256 }));
  } };
  const publishing = { network, publicKey: key.publicKey, privateKey: key.privateKey,
    verifyPreparedImpl: (directory, version) => verifyPrepared(directory, version, key.publicKey), sleep: async () => {} };
  return { key, root, candidate, options, publishing, doc, blobs, writes,
    feed: () => feed, outage: value => { outage = value; }, loseAck: value => { failReadAfterPublish = value; } };
}
test('release preparation and publication make the exact engine available without a GUI build', async () => {
  const f = await world();
  try {
    assert.deepEqual(await prepare('v1.2.4', f.candidate, f.options), { state: 'prepared', version: '1.2.4' });
    const result = await publish(f.candidate, f.publishing);
    assert.equal(result.verifiedLive, true);
    const catalog = await verifyCatalog(f.feed(), f.key.publicKey);
    assert.deepEqual(catalog.packages.map(pkg => pkg.version), ['1.2.4', '1.2.2']);
    assert.equal(catalog.packages[0].sha256, hash(f.blobs.get(`${catalog.packages[0].id}.tar.gz`)));
    assert.deepEqual(f.writes, ['asset', 'feed']);
    const retry = await publish(f.candidate, f.publishing);
    assert.equal(retry.state, 'current');
    assert.deepEqual(f.writes, ['asset', 'feed']);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});
test('a feed outage after asset publication can be retried without replacing the artifact', async () => {
  const f = await world();
  try {
    await prepare('v1.2.4', f.candidate, f.options);
    f.outage(true);
    await assert.rejects(publish(f.candidate, f.publishing));
    assert.equal((await verifyCatalog(f.feed(), f.key.publicKey)).packages[0].version, '1.2.2');
    f.outage(false);
    assert.equal((await publish(f.candidate, f.publishing)).state, 'published');
    assert.equal(f.writes.filter(value => value === 'asset').length, 1);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});
test('lost feed acknowledgements recover through live discovery without duplicate publication', async () => {
  const f = await world();
  try {
    await prepare('v1.2.4', f.candidate, f.options);
    f.loseAck(true);
    await assert.rejects(publish(f.candidate, f.publishing));
    const before = [...f.writes];
    assert.equal((await publish(f.candidate, f.publishing)).state, 'current');
    assert.deepEqual(f.writes, before);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});
test('wrong signatures, missing assets and altered prepared files stop before publication', async () => {
  const f = await world();
  try {
    const signatureName = 'tensorfold-1.2.4-macos-arm64.tar.gz.sha256.sig';
    const original = f.blobs.get(signatureName); f.blobs.set(signatureName, Buffer.alloc(64));
    await assert.rejects(prepare('v1.2.4', f.candidate, f.options));
    f.blobs.set(signatureName, original);
    const second = path.join(f.root, 'other'); await prepare('v1.2.4', second, f.options);
    const pkg = JSON.parse(await readFile(path.join(second, 'package.json'))); pkg.files[0].sha256 = 'a'.repeat(64);
    await writeFile(path.join(second, 'package.json'), JSON.stringify(pkg));
    await assert.rejects(publish(second, f.publishing));
    assert.deepEqual(f.writes, []);
  } finally { await rm(f.root, { recursive: true, force: true }); }
});
