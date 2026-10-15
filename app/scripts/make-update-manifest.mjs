#!/usr/bin/env node
import { createHash, createPrivateKey, sign } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { assertProductionMacSigner, assertSigningKeyMatchesTrustRoot, embeddedUpdateTrustRoot } from './update-signing-trust.mjs'
import { expectedMacArtifacts, verifyMacReleaseArtifacts } from './package-mac.mjs'
import { readEngineManifest } from './fetch-engine.mjs'

const args = parseArgs(process.argv.slice(2))

if (!args.version || !args.downloadUrl || !args.artifact || !args.out) {
  console.error('Usage: node scripts/make-update-manifest.mjs --version 0.2.0 --artifact release/TensorFold-Studio-0.2.0-arm64.dmg --download-url https://example.com/TensorFold-Studio-0.2.0-arm64.dmg --notes CHANGELOG.md --release-notes-url https://example.com/changelog --out release/update.json --private-key ed25519.pem (use --unsigned-dev only for local fixtures)')
  process.exit(1)
}

const unsignedDev = args.unsignedDev === '1'
if (!args.privateKey && !unsignedDev) throw new Error('Release update manifests require --private-key. Use --unsigned-dev only for an explicit local development fixture.')
if (args.privateKey && unsignedDev) throw new Error('Choose either --private-key or --unsigned-dev, not both.')

assertPublicHttpsUrl(args.downloadUrl, 'download URL')
const releaseNotesUrl = args.releaseNotesUrl || args.notesUrl || args.changelogUrl || ''
if (releaseNotesUrl) assertPublicHttpsUrl(releaseNotesUrl, 'release notes URL')

const artifactBytes = await readFile(args.artifact)
const sha256 = createHash('sha256').update(artifactBytes).digest('hex')
const releaseNotes = args.notes ? (await readFile(args.notes, 'utf8')).trim() : ''
const now = new Date().toISOString()
const asset = {
  name: path.basename(args.artifact),
  platform: 'macos',
  arch: inferArch(args.artifact),
  browser_download_url: args.downloadUrl,
  sha256,
}
const latest = {
  version: args.version,
  releaseName: args.name || `TensorFold ${args.version}`,
  releaseNotes,
  publishedAt: now,
  downloadUrl: args.downloadUrl,
  sha256,
  assets: [asset],
  ...(releaseNotesUrl ? { releaseNotesUrl } : {}),
}

if (args.updateZip || args.updateZipUrl) {
  if (!args.updateZip || !args.updateZipUrl) throw new Error('Automatic updates require both --update-zip and --update-zip-url.')
  assertPublicHttpsUrl(args.updateZipUrl, 'automatic update ZIP URL')
  if (!new URL(args.updateZipUrl).pathname.toLowerCase().endsWith('.zip') || !args.updateZip.toLowerCase().endsWith('.zip')) throw new Error('Automatic updates require a ZIP artifact and URL.')
  const zipBytes = await readFile(args.updateZip)
  const zipSha256 = createHash('sha256').update(zipBytes).digest('hex')
  const arch = inferArch(args.updateZip)
  if (arch !== asset.arch) throw new Error('Automatic update ZIP architecture must match the installer architecture.')
  latest.automaticUpdate = { platform: 'macos', arch, zipUrl: args.updateZipUrl, sha256: zipSha256 }
  latest.assets.push({ name: path.basename(args.updateZip), platform: 'macos', arch, browser_download_url: args.updateZipUrl, sha256: zipSha256 })
}

if (args.privateKey) {
  const key = createPrivateKey(await readFile(args.privateKey, 'utf8'))
  if (key.asymmetricKeyType !== 'ed25519') throw new Error('--private-key must contain an Ed25519 private key.')
  assertSigningKeyMatchesTrustRoot(key, await embeddedUpdateTrustRoot())
  if (args.updateZip) {
    const root = fileURLToPath(new URL('..', import.meta.url))
    const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
    const metadata = { productName: pkg.build.productName, version: args.version, arch: latest.automaticUpdate.arch, appId: pkg.build.appId, engine: await readEngineManifest(root) }
    const outputRoot = path.dirname(path.resolve(args.updateZip))
    const expected = expectedMacArtifacts(outputRoot, metadata)
    if (path.resolve(args.updateZip) !== expected.zips[0]) throw new Error('Automatic update ZIP must be the exact packaged release artifact.')
    // Check the pinned signer, notarization and contained app/engine against the built bundle.
    await verifyMacReleaseArtifacts(outputRoot, ['zip'], metadata, await assertProductionMacSigner(root))
    const verifiedHash = createHash('sha256').update(await readFile(args.updateZip)).digest('hex')
    if (verifiedHash !== latest.automaticUpdate.sha256) throw new Error('Automatic update ZIP changed during release verification.')
  }
  latest.signature = sign(null, Buffer.from(canonicalJson(latest)), key).toString('base64')
}

const manifest = { latest, releases: [latest] }
await writeFile(args.out, `${JSON.stringify(manifest, null, 2)}\n`)
console.log(`Wrote ${args.out}`)
console.log(`sha256 ${sha256}`)
console.log(args.privateKey ? 'signature ed25519' : 'UNSIGNED DEVELOPMENT MANIFEST — DO NOT PUBLISH')

function parseArgs(argv) {
  const out = {}
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase())
    out[key] = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : '1'
  }
  return out
}

function inferArch(file) {
  const lower = file.toLowerCase()
  if (lower.includes('arm64') || lower.includes('aarch64')) return 'arm64'
  if (lower.includes('x64') || lower.includes('x86_64') || lower.includes('amd64')) return 'x64'
  return 'universal'
}

function assertPublicHttpsUrl(raw, label) {
  let url
  try {
    url = new URL(raw)
  } catch {
    throw new Error(`Invalid ${label}: ${raw}`)
  }
  if (url.protocol !== 'https:') throw new Error(`${label} must use https`)
  if (url.username || url.password) throw new Error(`${label} must not include credentials`)
  if (isUnsafeUpdateHost(url.hostname)) throw new Error(`${label} must use a public host`)
}

function isUnsafeUpdateHost(hostname) {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  if (!host || host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local')) return true
  const v4 = host.startsWith('::ffff:') ? host.slice(7) : host
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(v4)) {
    const [a, b] = v4.split('.').map(Number)
    if (a === 0 || a === 10 || a === 127 || a >= 224) return true
    if (a === 100 && b >= 64 && b <= 127) return true
    if (a === 169 && b === 254) return true
    if (a === 172 && b >= 16 && b <= 31) return true
    if (a === 192 && b === 168) return true
    if (a === 198 && (b === 18 || b === 19)) return true
    return false
  }
  if (!host.includes(':')) return false
  return host === '::' ||
    host === '::1' ||
    host.startsWith('fc') ||
    host.startsWith('fd') ||
    host.startsWith('fe8') ||
    host.startsWith('fe9') ||
    host.startsWith('fea') ||
    host.startsWith('feb')
}

function canonicalJson(value) {
  return JSON.stringify(sortJson(stripSignatureFields(value)))
}

function stripSignatureFields(value) {
  if (Array.isArray(value)) return value.map(stripSignatureFields)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['signature', 'ed25519Signature', 'edSignature', 'sparkleEdSignature'].includes(key))
    .map(([key, item]) => [key, stripSignatureFields(item)]))
}

function sortJson(value) {
  if (Array.isArray(value)) return value.map(sortJson)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, sortJson(item)]))
}
