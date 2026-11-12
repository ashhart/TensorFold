import { createHash, createPublicKey, sign, verify } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { DOMAIN, LIMIT, validateCatalog, verifyCatalog } from './contract.mjs';
export const trust = JSON.parse(await readFile(new URL('../../packaging/runtime-trust.json', import.meta.url)));
export const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export function verifyNative(checksum, signature, version, publicKey = trust.ed25519PublicKey) {
  const name = `tensorfold-${version}-macos-arm64.tar.gz`;
  const text = checksum.toString('utf8');
  if (!/^[a-f0-9]{64}$/.test(text.slice(0, 64)) || text !== `${text.slice(0, 64)}  ${name}\n` ||
    signature.length !== 64 || !verify(null, Buffer.concat([Buffer.from('tensorfold-native-checksum-v1\n'), checksum]), publicKey, signature)) {
    throw new Error('Official engine checksum signature is invalid.');
  }
  return text.slice(0, 64);
}
export async function signCatalog(doc, privateKey, publicKey = trust.ed25519PublicKey) {
  validateCatalog(doc);
  if (createPublicKey(privateKey).export({ format: 'pem', type: 'spki' }).toString().trim() !== publicKey.trim()) {
    throw new Error('Signing key does not match the established desktop root.');
  }
  const payload = Buffer.from(JSON.stringify(doc));
  const raw = JSON.stringify({ payload: payload.toString('base64'),
    signature: sign(null, Buffer.concat([Buffer.from(DOMAIN), payload]), privateKey).toString('base64') });
  if (Buffer.byteLength(raw) > LIMIT) throw new Error('Signed catalog is too large.');
  await verifyCatalog(raw, publicKey);
  return raw + '\n';
}
