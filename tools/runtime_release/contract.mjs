export const DOMAIN = 'TensorFold runtime catalog v1\n';
export const LIMIT = 512 * 1024;
export const FEED_PATH = '/studio/runtimes.json';
const hex = /^[a-f0-9]{64}$/;
export function safePath(value) {
  return typeof value === 'string' && value.length <= 240 && value.split('/').every(part =>
    /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(part) && !/[. ]$/.test(part) &&
    !/^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(part));
}
function require(value) { if (!value) throw new Error('Invalid signed runtime catalog.'); }
export function validateCatalog(doc, now = Date.now(), allowExpired = false) {
  require(doc?.schema === 1 && Number.isSafeInteger(doc.publishedAt) && doc.publishedAt >= 0 &&
    Number.isSafeInteger(doc.expiresAt) && doc.expiresAt > doc.publishedAt &&
    doc.expiresAt - doc.publishedAt <= 31 * 86400_000 && doc.publishedAt <= now + 300_000 &&
    (allowExpired || doc.expiresAt > now) && Array.isArray(doc.packages) &&
    doc.packages.length > 0 && doc.packages.length <= 128);
  const ids = new Set();
  for (const pkg of doc.packages) {
    require(pkg && typeof pkg.id === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(pkg.id) &&
      safePath(pkg.id) && !ids.has(pkg.id.toLowerCase()) && /^1\.\d{1,5}\.\d{1,5}$/.test(pkg.version) &&
      typeof pkg.minAppVersion === 'string' && /^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$/.test(pkg.minAppVersion) &&
      pkg.protocol === 1 && ['darwin', 'linux', 'win32'].includes(pkg.platform) && ['arm64', 'x64'].includes(pkg.arch) &&
      typeof pkg.backend === 'string' && /^[a-z0-9-]{1,24}$/.test(pkg.backend) &&
      typeof pkg.notes === 'string' && pkg.notes.length <= 12000 && hex.test(pkg.sha256) &&
      Number.isSafeInteger(pkg.bytes) && pkg.bytes > 0 && pkg.bytes <= 512 * 1024 ** 2 &&
      safePath(pkg.entrypoint) && Array.isArray(pkg.files) && pkg.files.length > 0 && pkg.files.length <= 512);
    ids.add(pkg.id.toLowerCase());
    const url = new URL(pkg.url);
    require(url.origin === 'https://github.com' && !url.username && !url.password && !url.search && !url.hash &&
      /^\/ashhart\/TensorFold\/releases\/download\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.tar\.gz$/.test(url.pathname) &&
      !/(?:latest|nightly)/i.test(url.pathname));
    const names = new Set();
    let size = 0;
    for (const file of pkg.files) {
      require(file && safePath(file.path) && !names.has(file.path.toLowerCase()) && hex.test(file.sha256) &&
        Number.isSafeInteger(file.size) && file.size >= 0 && typeof file.executable === 'boolean');
      size += file.size;
      require(size <= 2 * 1024 ** 3);
      names.add(file.path.toLowerCase());
    }
    require(pkg.files.some(f => f.path === pkg.entrypoint && f.executable && f.size > 0) &&
      ['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'].every(name => pkg.files.some(f => f.path === name)) &&
      pkg.files.some(f => f.path === (pkg.platform === 'darwin' ? 'lib/libtfgrammar.dylib' : 'lib/libtfgrammar.so') && f.size > 0));
    for (const file of pkg.files) require(!pkg.files.some(other => other.path.startsWith(`${file.path}/`)));
  }
  return doc;
}
function decode(value) {
  require(typeof value === 'string' && value.length <= LIMIT);
  const binary = atob(value);
  require(btoa(binary) === value);
  return Uint8Array.from(binary, char => char.charCodeAt(0));
}
export async function verifyCatalog(raw, publicKey, now = Date.now(), allowExpired = false) {
  require(typeof raw === 'string' && new TextEncoder().encode(raw).length <= LIMIT);
  const envelope = JSON.parse(raw);
  require(envelope && Object.keys(envelope).sort().join(',') === 'payload,signature');
  const payload = decode(envelope.payload), signature = decode(envelope.signature);
  require(signature.length === 64);
  const key = await crypto.subtle.importKey('spki', decode(publicKey.replace(/-----[^-]+-----|\s/g, '')),
    'Ed25519', false, ['verify']);
  const domain = new TextEncoder().encode(DOMAIN), signed = new Uint8Array(domain.length + payload.length);
  signed.set(domain); signed.set(payload, domain.length);
  require(await crypto.subtle.verify('Ed25519', key, signature, signed));
  return validateCatalog(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(payload)), now, allowExpired);
}
export function canonical(value) {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort()
    .map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
  return JSON.stringify(value);
}
export function preserve(previous, next) {
  require(next.publishedAt >= previous.publishedAt);
  for (const pkg of previous.packages) {
    require(next.packages.some(candidate => candidate.id === pkg.id && canonical(candidate) === canonical(pkg)));
  }
}
export function mergeCatalog(previous, pkg, now = Date.now()) {
  const packages = previous ? [...previous.packages] : [];
  const existing = packages.find(candidate => candidate.id.toLowerCase() === pkg.id.toLowerCase());
  require(!existing || canonical(existing) === canonical(pkg));
  if (!existing) packages.push(pkg);
  packages.sort((a, b) => {
    const av = a.version.split('.').map(Number), bv = b.version.split('.').map(Number);
    return bv[0] - av[0] || bv[1] - av[1] || bv[2] - av[2] || a.id.localeCompare(b.id);
  });
  const publishedAt = Math.max(now, (previous?.publishedAt ?? 0) + 1);
  const doc = { schema: 1, publishedAt, expiresAt: publishedAt + 30 * 86400_000, packages };
  validateCatalog(doc, now);
  if (previous) preserve(previous, doc);
  return doc;
}
