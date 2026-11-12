import { readFile } from 'node:fs/promises';
import { trust, hash } from './crypto.mjs';
const API = `https://api.github.com/repos/${trust.repository}`;
const ASSETS = new Set(['github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com',
  'github-releases.githubusercontent.com']);
export function versionTag(tag) {
  if (typeof tag !== 'string' || !/^v1\.(0|[1-9]\d{0,4})\.(0|[1-9]\d{0,4})$/.test(tag)) throw new Error('Choose a stable engine release tag.');
  return tag.slice(1);
}
export async function request(url, { token, method = 'GET', body, limit = 2 * 1024 ** 2, fetchImpl = fetch } = {}) {
  for (let redirects = 0; redirects <= 4; redirects++) {
    const target = new URL(url);
    if (target.protocol !== 'https:' || target.port || target.username || target.password ||
      !['api.github.com', 'uploads.github.com', 'tensorfold.dev', ...ASSETS].includes(target.hostname)) {
      throw new Error('Unexpected release transport origin.');
    }
    const headers = { Accept: 'application/json', 'User-Agent': 'TensorFold-runtime-publisher' };
    if (token && ['api.github.com', 'uploads.github.com'].includes(target.hostname)) headers.Authorization = `Bearer ${token}`;
    if (body) headers['Content-Type'] = target.hostname === 'uploads.github.com' ? 'application/gzip' : 'application/json';
    const response = await fetchImpl(target.href, { method, body, headers, redirect: 'manual', signal: AbortSignal.timeout(120000) });
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      await response.body?.cancel();
      if (method !== 'GET' || redirects === 4) throw new Error('Unexpected release redirect.');
      url = new URL(response.headers.get('location'), target).href;
      if (!ASSETS.has(new URL(url).hostname)) throw new Error('Unexpected asset redirect.');
      continue;
    }
    if (!response.ok) { await response.body?.cancel(); throw new Error(`Release transport failed, HTTP ${response.status}.`); }
    const length = response.headers.get('content-length');
    if (length && (!/^\d+$/.test(length) || Number(length) > limit)) {
      await response.body?.cancel(); throw new Error('Release response exceeded its bounds.');
    }
    const chunks = []; let bytes = 0;
    if (response.body) for await (const chunk of response.body) {
      bytes += chunk.byteLength;
      if (bytes > limit) throw new Error('Release response exceeded its bounds.');
      chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
  }
  throw new Error('Too many release redirects.');
}
export function validateRelease(doc, tag) {
  versionTag(tag);
  if (!doc || doc.tag_name !== tag || doc.draft !== false || doc.prerelease !== false ||
    !Number.isSafeInteger(doc.id) || doc.id <= 0 || !Array.isArray(doc.assets) || doc.assets.length > 100 ||
    doc.html_url !== `https://github.com/${trust.repository}/releases/tag/${tag}`) {
    throw new Error('Expected a published stable TensorFold engine release.');
  }
  const names = new Set();
  for (const asset of doc.assets) {
    if (!asset || typeof asset.name !== 'string' || names.has(asset.name) ||
      !Number.isSafeInteger(asset.size) || asset.size <= 0 || asset.size > 512 * 1024 ** 2 ||
      asset.browser_download_url !== `https://github.com/${trust.repository}/releases/download/${tag}/${asset.name}`) {
      throw new Error('Invalid or ambiguous release assets.');
    }
    names.add(asset.name);
  }
  return doc;
}
export async function findRelease(tag, options) {
  if (!tag) {
    const docs = JSON.parse(await request(`${API}/releases?per_page=100`, options));
    if (!Array.isArray(docs)) throw new Error('Invalid release listing.');
    const eligible = docs.filter(doc => /^v1\.\d{1,5}\.\d{1,5}$/.test(doc.tag_name) && !doc.draft && !doc.prerelease);
    eligible.sort((a, b) => {
      const av = a.tag_name.slice(1).split('.').map(Number), bv = b.tag_name.slice(1).split('.').map(Number);
      return bv[0] - av[0] || bv[1] - av[1] || bv[2] - av[2];
    });
    tag = eligible[0]?.tag_name;
  }
  versionTag(tag);
  return validateRelease(JSON.parse(await request(`${API}/releases/tags/${tag}`, options)), tag);
}
export async function officialAsset(release, name, options, limit = 512 * 1024 ** 2) {
  const asset = release.assets.find(candidate => candidate.name === name);
  if (!asset || asset.size > limit) throw new Error('Required engine release asset is absent or oversized.');
  const bytes = await request(asset.browser_download_url, { ...options, limit });
  if (bytes.length !== asset.size || asset.digest && asset.digest !== `sha256:${hash(bytes)}`) {
    throw new Error('Downloaded release asset does not match its metadata.');
  }
  return bytes;
}
export async function ensureAsset(release, pkg, file, options) {
  const bytes = await readFile(file);
  if (bytes.length !== pkg.bytes || hash(bytes) !== pkg.sha256) throw new Error('Runtime changed before upload.');
  const name = `${pkg.id}.tar.gz`;
  if (release.assets.some(asset => asset.name === name)) {
    const downloaded = await officialAsset(release, name, options);
    if (hash(downloaded) !== pkg.sha256) throw new Error('Immutable runtime asset already has different content.');
    return;
  }
  if (!options?.token) throw new Error('Publishing runtime assets requires a scoped GitHub token.');
  await request(`https://uploads.github.com/repos/${trust.repository}/releases/${release.id}/assets?name=${encodeURIComponent(name)}`,
    { ...options, method: 'POST', body: bytes });
  const published = await findRelease(release.tag_name, options);
  if (hash(await officialAsset(published, name, options)) !== pkg.sha256) throw new Error('Published runtime hash verification failed.');
}
