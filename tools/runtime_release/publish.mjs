import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { findRelease, ensureAsset, officialAsset, request, versionTag } from './github.mjs';
import { signCatalog, trust, hash } from './crypto.mjs';
import { canonical, mergeCatalog, verifyCatalog } from './contract.mjs';
import { archiveOperation, verifyPrepared } from './prepare.mjs';
export async function verifyHostedPackage(pkg, network) {
  const bytes = await request(pkg.url, { ...network, limit: pkg.bytes });
  if (bytes.length !== pkg.bytes || hash(bytes) !== pkg.sha256) throw new Error('Hosted runtime archive hash mismatch.');
  const directory = await mkdtemp(path.join(tmpdir(), 'tf-runtime-inventory-'));
  try {
    await writeFile(path.join(directory, 'package.json'), JSON.stringify(pkg), { mode: 0o600 });
    await writeFile(path.join(directory, 'runtime.tar.gz'), bytes, { mode: 0o600 });
    await archiveOperation('inventory', pkg.version, directory);
  } finally { await rm(directory, { recursive: true, force: true }); }
}
export async function publish(directory, { privateKey, network, publicKey = trust.ed25519PublicKey,
  verifyPreparedImpl = verifyPrepared, verifyHosted = verifyHostedPackage, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)) } = {}) {
  const status = JSON.parse(await readFile(path.join(directory, 'status.json')));
  versionTag(`v${status.version}`);
  if (!['current', 'renew', 'prepared'].includes(status.state)) throw new Error('Invalid preparation status.');
  const readFeed = async () => verifyCatalog((await request(trust.feed, network)).toString('utf8'), publicKey, Date.now(), true);
  const previous = await readFeed();
  const id = `tensorfold-engine-${status.version}-darwin-arm64-metal`;
  const existing = previous.packages.find(pkg => pkg.id === id);
  if (existing && previous.expiresAt > Date.now() + 14 * 86400_000) {
    await verifyHosted(existing, network);
    return { state: 'current', version: status.version, verifiedLive: true };
  }
  let pkg = existing;
  if (!pkg) {
    if (status.state !== 'prepared') throw new Error('The selected runtime disappeared from the signed feed.');
    pkg = await verifyPreparedImpl(directory, status.version, publicKey);
    const release = await findRelease(`v${status.version}`, network);
    await ensureAsset(release, pkg, path.join(directory, `${pkg.id}.tar.gz`), network);
    const remote = await findRelease(`v${status.version}`, network);
    const official = await officialAsset(remote, `${pkg.id}.tar.gz`, network);
    if (hash(official) !== pkg.sha256) throw new Error('Runtime asset was not verified before signing.');
  }
  await verifyHosted(pkg, network);
  const latest = await readFeed();
  const next = mergeCatalog(latest, pkg);
  if (!privateKey) throw new Error('The trusted signing job requires its protected release key.');
  const signed = await signCatalog(next, privateKey, publicKey);
  await writeFile(path.join(directory, `catalog-${next.publishedAt}.json`), signed, { mode: 0o600, flag: 'wx' });
  await request(trust.feed, { ...network, method: 'POST', body: signed });
  for (let attempt = 0; attempt < 8; attempt++) {
    const live = await readFeed();
    if (live.publishedAt >= next.publishedAt && live.packages.some(item => canonical(item) === canonical(pkg))) {
      await verifyHosted(pkg, network);
      return { state: 'published', version: pkg.version, verifiedLive: true, packages: live.packages.length };
    }
    await sleep(Math.min(1000 * 2 ** attempt, 15000));
  }
  throw new Error('Published runtime feed did not become discoverable live.');
}
if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const [directory, approval] = process.argv.slice(2);
  if (!directory || approval !== '--publish') throw new Error('Usage: publish.mjs PREPARED_DIRECTORY --publish');
  try {
    const result = await publish(path.resolve(directory), {
      privateKey: process.env.TF_RUNTIME_SIGNING_KEY, network: { token: process.env.GH_TOKEN },
    });
    await mkdir(path.resolve(directory), { recursive: true });
    await writeFile(path.resolve(directory, 'publication.json'), JSON.stringify(result, null, 2) + '\n', { mode: 0o600 });
    console.log(JSON.stringify(result));
  } catch { process.exitCode = 1; console.error('Runtime publication failed. Check the release job; no automatic success was recorded.'); }
}
