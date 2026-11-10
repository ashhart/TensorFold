import { execFile as execute } from 'node:child_process';
import { promisify } from 'node:util';
import { chmod, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { findRelease, officialAsset, versionTag, request } from './github.mjs';
import { verifyNative, trust, hash } from './crypto.mjs';
import { verifyCatalog } from './contract.mjs';
const execFile = promisify(execute);
export async function archiveOperation(operation, version, directory) {
  await execFile('python3', [fileURLToPath(new URL('./archive.py', import.meta.url)), operation, version, directory],
    { timeout: 120000, maxBuffer: 1024 * 1024 });
}
export async function verifyPrepared(directory, version, publicKey = trust.ed25519PublicKey) {
  const name = `tensorfold-${version}-macos-arm64.tar.gz`;
  const checksum = await readFile(path.join(directory, `${name}.sha256`));
  const signature = await readFile(path.join(directory, `${name}.sha256.sig`));
  if (hash(await readFile(path.join(directory, name))) !== verifyNative(checksum, signature, version, publicKey)) {
    throw new Error('Prepared native archive authentication failed.');
  }
  await archiveOperation('verify', version, directory);
  const pkg = JSON.parse(await readFile(path.join(directory, 'package.json')));
  const probe = JSON.parse(await readFile(path.join(directory, 'probe.json')));
  if (probe.version !== version || probe.schema !== 1 || probe.backend !== 'metal' || probe.grammar !== true ||
    probe.executableSha256 !== pkg.files.find(file => file.path === pkg.entrypoint)?.sha256) {
    throw new Error('Native desktop compatibility receipt is invalid.');
  }
  return pkg;
}
export async function prepare(tag, directory, { probe = nativeProbe, network, publicKey = trust.ed25519PublicKey } = {}) {
  if (process.env.TF_RUNTIME_SIGNING_KEY) throw new Error('The preparation process must not receive the signing key.');
  const release = await findRelease(tag, network), version = versionTag(release.tag_name);
  const old = await verifyCatalog((await request(trust.feed, network)).toString('utf8'), publicKey, Date.now(), true);
  const id = `tensorfold-engine-${version}-darwin-arm64-metal`;
  if (old.packages.some(pkg => pkg.id === id)) {
    const state = old.expiresAt > Date.now() + 14 * 86400_000 ? 'current' : 'renew';
    await mkdir(directory, { recursive: true, mode: 0o700 });
    await writeFile(path.join(directory, 'status.json'), JSON.stringify({ state, version }), { flag: 'wx' });
    return { state, version };
  }
  await mkdir(directory, { mode: 0o700 });
  const name = `tensorfold-${version}-macos-arm64.tar.gz`;
  const checksum = await officialAsset(release, `${name}.sha256`, network, 1024);
  const signature = await officialAsset(release, `${name}.sha256.sig`, network, 64);
  const expected = verifyNative(checksum, signature, version, publicKey);
  const archive = await officialAsset(release, name, network);
  if (hash(archive) !== expected) throw new Error('Official engine archive hash verification failed.');
  for (const [file, bytes] of [[name, archive], [`${name}.sha256`, checksum], [`${name}.sha256.sig`, signature]]) {
    await writeFile(path.join(directory, file), bytes, { mode: 0o600, flag: 'wx' });
  }
  await archiveOperation('prepare', version, directory);
  const pkg = JSON.parse(await readFile(path.join(directory, 'package.json')));
  await probe(directory, pkg);
  await writeFile(path.join(directory, 'status.json'), JSON.stringify({ state: 'prepared', version }), { mode: 0o600, flag: 'wx' });
  return { state: 'prepared', version };
}
async function nativeProbe(directory, pkg) {
  if (process.platform !== 'darwin' || process.arch !== 'arm64') throw new Error('Native desktop qualification requires an arm64 Mac.');
  const source = path.join(directory, 'probe-tree');
  await mkdir(source, { mode: 0o700 });
  try {
  await execFile('python3', ['-c', 'import sys;from pathlib import Path;from archive import unpack;unpack(Path(sys.argv[1]),Path(sys.argv[2]))',
    path.join(directory, `${pkg.id}.tar.gz`), source],
  { cwd: fileURLToPath(new URL('.', import.meta.url)), timeout: 120000, maxBuffer: 1024 * 1024 });
  const binary = path.join(source, pkg.entrypoint);
  await chmod(binary, 0o755);
  const options = { cwd: source, timeout: 10000, maxBuffer: 4 * 1024 ** 2,
    env: { PATH: '/usr/bin:/bin', HOME: source, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' } };
  const { stdout } = await execFile(binary, ['--version'], options);
  if (stdout.trim() !== `tensorfold-native ${pkg.version}`) throw new Error('Runtime version probe failed.');
  const result = JSON.parse((await execFile(binary, ['capabilities', '--json'], options)).stdout);
  if (result.engine !== 'zig' || result.schema !== 1 || result.version !== pkg.version || !result.backends?.includes('metal') ||
    !result.optional_libraries?.some(library => library.name === 'TensorFold grammar' && library.found === true)) {
    throw new Error('Runtime does not satisfy the released app protocol or grammar contract.');
  }
  await writeFile(path.join(directory, 'probe.json'), JSON.stringify({ version: pkg.version, schema: 1, backend: 'metal', grammar: true,
    executableSha256: pkg.files.find(file => file.path === pkg.entrypoint).sha256 }), { mode: 0o600, flag: 'wx' });
  } finally { await rm(source, { recursive: true, force: true }); }
}
if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const [tag, directory] = process.argv.slice(2);
  if (!directory) throw new Error('Usage: prepare.mjs TAG_OR_LATEST NEW_OUTPUT_DIRECTORY');
  try { console.log(JSON.stringify(await prepare(tag === 'latest' ? null : tag, path.resolve(directory)))); }
  catch { process.exitCode = 1; console.error('Runtime preparation failed. No feed or release assets were changed.'); }
}
