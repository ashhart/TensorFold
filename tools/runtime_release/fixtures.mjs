import { generateKeyPairSync } from 'node:crypto';
import { hash } from './crypto.mjs';
export function fixture(version = '1.2.4') {
  const files = ['bin/tensorfold-native', 'lib/libtfgrammar.dylib', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md']
    .map(path => ({ path, size: 4, sha256: hash(Buffer.from(path)), executable: path.startsWith('bin/') }));
  return { id: `tensorfold-engine-${version}-darwin-arm64-metal`, version, platform: 'darwin', arch: 'arm64',
    backend: 'metal', protocol: 1, minAppVersion: '1.0.0', notes: '', bytes: 4, sha256: hash(Buffer.from(version)),
    url: `https://github.com/ashhart/TensorFold/releases/download/v${version}/tensorfold-engine-${version}-darwin-arm64-metal.tar.gz`,
    entrypoint: 'bin/tensorfold-native', files };
}
export function keys() {
  const pair = generateKeyPairSync('ed25519');
  return { privateKey: pair.privateKey, publicKey: pair.publicKey.export({ type: 'spki', format: 'pem' }).toString() };
}
export function release(tag = 'v1.2.4', assets = []) {
  return { id: 42, tag_name: tag, draft: false, prerelease: false,
    html_url: `https://github.com/ashhart/TensorFold/releases/tag/${tag}`, assets };
}
