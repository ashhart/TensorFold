import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { ensureAsset, findRelease, officialAsset, request, validateRelease, versionTag } from './github.mjs';
import { fixture, release } from './fixtures.mjs';
import { hash } from './crypto.mjs';
test('only canonical stable releases and unambiguous official assets are eligible', () => {
  for (const tag of ['latest', 'app-v1.0.0', 'v1.2.4-beta.1', 'v1.02.4', 'v1.2.4;echo']) assert.throws(() => versionTag(tag));
  const doc = release();
  assert.equal(validateRelease(doc, 'v1.2.4'), doc);
  for (const change of [{ draft: true }, { prerelease: true }, { tag_name: 'v1.2.3' }, { id: '../bad' }]) {
    assert.throws(() => validateRelease({ ...doc, ...change }, 'v1.2.4'));
  }
  const asset = { name: 'x.tar.gz', size: 4, browser_download_url: 'https://github.com/ashhart/TensorFold/releases/download/v1.2.4/x.tar.gz' };
  assert.throws(() => validateRelease(release('v1.2.4', [asset, asset]), 'v1.2.4'));
});
test('latest selection uses engine semver and ignores app releases and previews', async () => {
  const docs = [release('app-v9.0.0'), release('v1.2.3'), release('v1.2.4'), { ...release('v1.9.0'), prerelease: true }];
  const calls = [];
  const found = await findRelease(null, { fetchImpl: async url => {
    calls.push(url); return Response.json(url.includes('?') ? docs : release('v1.2.4'));
  } });
  assert.equal(found.tag_name, 'v1.2.4');
  assert.equal(calls.length, 2);
});
test('tokens never reach asset redirects and unexpected destinations are refused', async () => {
  let calls = 0;
  const bytes = await request('https://api.github.com/repos/ashhart/TensorFold/releases', {
    token: 'fixture-private-token', fetchImpl: async (url, init) => {
      calls++;
      if (calls === 1) {
        assert.equal(init.headers.Authorization, 'Bearer fixture-private-token');
        return new Response(null, { status: 302, headers: { location: 'https://release-assets.githubusercontent.com/fixture' } });
      }
      assert.equal(init.headers.Authorization, undefined);
      return new Response('okay');
    },
  });
  assert.equal(bytes.toString(), 'okay');
  await assert.rejects(request('https://github.com/fixture', {
    fetchImpl: async () => new Response(null, { status: 302, headers: { location: 'http://127.0.0.1/private' } }),
  }));
  await assert.rejects(request('https://attacker.example/anything'));
});
test('asset count, checksum and advertised response bounds are enforced', async () => {
  const doc = release('v1.2.4', [{ name: 'x.tar.gz', size: 4, digest: 'sha256:' + 'a'.repeat(64),
    browser_download_url: 'https://github.com/ashhart/TensorFold/releases/download/v1.2.4/x.tar.gz' }]);
  await assert.rejects(officialAsset(doc, 'missing', {}));
  await assert.rejects(officialAsset(doc, 'x.tar.gz', { fetchImpl: async () => new Response('body') }));
  await assert.rejects(request('https://github.com/fixture', { limit: 3, fetchImpl: async () => new Response('long') }));
});
test('immutable runtime assets are reused only after complete hash verification', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'tf-immutable-fixture-'));
  try {
    const pkg = fixture(); pkg.sha256 = hash(Buffer.from('body'));
    const file = path.join(directory, 'runtime.tar.gz'); await writeFile(file, 'body');
    const asset = { name: `${pkg.id}.tar.gz`, size: 4, browser_download_url: pkg.url };
    let writes = 0;
    const network = { fetchImpl: async (url, init) => {
      if (init.method !== 'GET') writes++;
      return new Response('body');
    } };
    await ensureAsset(release('v1.2.4', [asset]), pkg, file, network);
    assert.equal(writes, 0);
    await assert.rejects(ensureAsset(release('v1.2.4', [asset]), pkg, file,
      { fetchImpl: async () => new Response('evil') }));
    await assert.rejects(ensureAsset(release(), pkg, file, network), /requires a scoped GitHub token/);
    assert.equal(writes, 0);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
