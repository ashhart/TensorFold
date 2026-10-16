import assert from 'node:assert/strict'
import { createPublicKey, generateKeyPairSync, sign } from 'node:crypto'
import releaseTrust from '../../build/release-trust.json'
import { checkUpdateFeed, compareVersions, emptyUpdateStatus, MAX_UPDATE_MANIFEST_BYTES } from './update-checker'
import {
  updateTrustRootPublicKey,
  VONTRA_UPDATE_ED25519_PUBLIC_KEY,
  VONTRA_UPDATE_TRUST_ROOT_ID,
  VONTRA_UPDATE_TRUST_ROOT_PRODUCTION_READY,
} from './update-trust'

const { publicKey, privateKey } = generateKeyPairSync('ed25519')
const publicPem = publicKey.export({ type: 'spki', format: 'pem' }).toString()
const lookup = async () => [{ address: '93.184.216.34', family: 4 }]

function signedJsonRelease<T extends Record<string, unknown>>(release: T): T & { signature: string } {
  return { ...release, signature: sign(null, Buffer.from(canonicalJson(release)), privateKey).toString('base64') }
}

function response(body: unknown, contentType = 'application/json'): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': contentType },
  })
}

async function checked(body: unknown, currentVersion = '0.1.0', contentType = 'application/json') {
  return checkUpdateFeed('https://example.com/feed.json', currentVersion, undefined, {
    publicKey: publicPem,
    lookup,
    fetchImpl: (async () => response(body, contentType)) as typeof fetch,
  })
}

async function main() {
  assert.equal(emptyUpdateStatus('1.0.0', 'https://example.com/update.json').state, 'not_checked', 'An unchecked feed cannot prove that the app is current')
  assert.equal(emptyUpdateStatus('1.0.0').state, 'not_configured')
  // Beta builds must be offered the release they precede, and later betas in order.
  assert.equal(compareVersions('1.0.0', '1.0.0-beta.1'), 1)
  assert.equal(compareVersions('1.0.0-beta.2', '1.0.0-beta.1'), 1)
  assert.equal(compareVersions('1.0.0-beta.10', '1.0.0-beta.2'), 1)
  assert.equal(compareVersions('1.0.1', '1.0.0'), 1)
  assert.equal(compareVersions('v1.0.0', '1.0.0'), 0)
  assert.equal(compareVersions('1.0.0-beta.1', '1.0.0'), -1)

  assert.equal(createPublicKey(updateTrustRootPublicKey()).asymmetricKeyType, 'ed25519', 'embedded trust root is a valid Ed25519 public key')
  assert.equal(VONTRA_UPDATE_ED25519_PUBLIC_KEY, releaseTrust.update.ed25519PublicKey, 'runtime public key comes from the strict release-trust JSON')
  assert.equal(VONTRA_UPDATE_TRUST_ROOT_ID, releaseTrust.update.id, 'runtime trust-root ID comes from the same strict JSON')
  assert.equal(VONTRA_UPDATE_TRUST_ROOT_PRODUCTION_READY, releaseTrust.update.productionReady, 'runtime readiness comes from the same strict JSON')

  const basicRelease = signedJsonRelease({
    version: '0.2.0',
    downloadUrl: 'https://example.com/vontra.dmg',
    sha256: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
  })
  const basic = await checked(basicRelease)
  assert.equal(basic.state, 'available')
  assert.equal(basic.integrity, 'signature_verified')
  assert.equal(basic.signatureVerified, true)
  assert.equal(basic.checksum?.algorithm, 'sha256')
  assert.equal(basic.automaticUpdate, null, 'legacy installer-only manifests stay manual')

  const automaticRelease = signedJsonRelease({
    version: '0.2.1', downloadUrl: 'https://example.com/studio.dmg', sha256: basicRelease.sha256,
    automaticUpdate: { platform: 'macos', arch: 'arm64', zipUrl: 'https://example.com/studio.zip', sha256: 'a'.repeat(64) },
  })
  const automatic = await checked(automaticRelease)
  assert.deepEqual(automatic.automaticUpdate, { zipUrl: 'https://example.com/studio.zip', checksum: { algorithm: 'sha256', value: 'a'.repeat(64) }, arch: 'arm64' })
  await assert.rejects(() => checked({ ...automaticRelease, automaticUpdate: { ...automaticRelease.automaticUpdate, zipUrl: 'https://example.com/tampered.zip' } }), /signature verification failed/, 'the automatic ZIP URL and checksum are bound by the manifest signature')
  for (const malformed of [
    { ...automaticRelease.automaticUpdate, zipUrl: 'http://example.com/update.zip' },
    { ...automaticRelease.automaticUpdate, sha256: 'bad' },
    { ...automaticRelease.automaticUpdate, arch: 'unknown' },
    { ...automaticRelease.automaticUpdate, zipUrl: 'https://example.com/update.dmg' },
  ]) await assert.rejects(() => checked(signedJsonRelease({ ...automaticRelease, signature: undefined, automaticUpdate: malformed })), /https|checksum|platform|ZIP/)

  const richRelease = signedJsonRelease({
    version: '0.3.0',
    releaseName: 'TensorFold Studio 0.3.0',
    changelog: [
      'Native update prompt',
      { title: 'Rooms', items: ['Agent builder presets', 'Room ticker'] },
    ],
    assets: [
      { name: 'TensorFold-Studio-0.3.0-linux.AppImage', platform: 'linux', url: 'https://example.com/vontra.AppImage', sha256: 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc' },
      { name: 'TensorFold-Studio-0.3.0-arm64.dmg', platform: 'macos', browser_download_url: 'https://example.com/vontra.dmg', sha256: 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd' },
    ],
  })
  const rich = await checked({ releases: [{ version: '0.1.9' }, richRelease] }, '0.2.0')
  assert.equal(rich.latestVersion, '0.3.0')
  assert.equal(rich.downloadUrl, 'https://example.com/vontra.dmg')
  assert.equal(rich.checksum?.value, 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
  assert.match(rich.releaseNotes ?? '', /Native update prompt/)
  assert.match(rich.releaseNotes ?? '', /Rooms/)

  const linkedNotes = await checked(signedJsonRelease({
    version: '0.4.0',
    releaseName: 'Linked notes',
    changelogUrl: 'https://example.com/vontra/changelog/0.4.0',
    downloadUrl: 'https://example.com/vontra-0.4.0.dmg',
    sha256: 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
  }), '0.3.0')
  assert.equal(linkedNotes.releaseNotes, 'https://example.com/vontra/changelog/0.4.0')

  const appcastPayload = canonicalJson({
    version: '0.5.0',
    downloadUrl: 'https://example.com/vontra-0.5.0.zip',
    checksum: 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
  })
  const appcastSignature = sign(null, Buffer.from(appcastPayload), privateKey).toString('base64')
  const appcast = await checked(`
    <rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
      <channel><item>
        <title>TensorFold Studio 0.5.0</title>
        <sparkle:version>0.5.0</sparkle:version>
        <sparkle:releaseNotesLink>https://example.com/vontra/appcast-notes.html</sparkle:releaseNotesLink>
        <enclosure url="https://example.com/vontra-0.5.0.zip"
          sparkle:sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
          sparkle:edSignature="${appcastSignature}" />
      </item></channel>
    </rss>
  `, '0.4.0', 'application/xml')
  assert.equal(appcast.releaseName, 'TensorFold Studio 0.5.0')
  assert.equal(appcast.releaseNotes, 'https://example.com/vontra/appcast-notes.html')
  assert.equal(appcast.signatureVerified, true)

  await assert.rejects(() => checked({
    ...basicRelease,
    downloadUrl: 'https://example.com/tampered.dmg',
  }), /signature verification failed/i, 'signed fields cannot be tampered after signing')

  const { signature: _signature, ...unsigned } = basicRelease
  await assert.rejects(() => checked(unsigned), /did not include an Ed25519 signature/i)
  await assert.rejects(() => checked(signedJsonRelease({
    version: '0.2.0', downloadUrl: 'https://example.com/vontra.dmg',
  })), /valid SHA-256 checksum/i)

  await assert.rejects(() => checked(signedJsonRelease({
    version: '0.2.0',
    downloadUrl: 'http://example.com/vontra.dmg',
    sha256: basicRelease.sha256,
  })), /download URL must use https/)
  await assert.rejects(() => checked(signedJsonRelease({
    version: '0.2.0',
    downloadUrl: 'https://127.0.0.1/vontra.dmg',
    sha256: basicRelease.sha256,
  })), /download URL must use a public host/)
  await assert.rejects(() => checked(signedJsonRelease({
    version: '0.2.0',
    releaseNotesUrl: 'file:///tmp/vontra-notes.html',
    downloadUrl: 'https://example.com/vontra.dmg',
    sha256: basicRelease.sha256,
  })), /release notes URL must use https/)

  // Production's default root is embedded and cannot be replaced by runtime environment state.
  await assert.rejects(() => checkUpdateFeed('https://example.com/feed.json', '0.1.0', undefined, {
    lookup,
    fetchImpl: (async () => response(basicRelease)) as typeof fetch,
  }), /signature verification failed/i)

  await assert.rejects(() => checkUpdateFeed('https://example.com/feed.json', '0.1.0', undefined, {
    publicKey: publicPem,
    lookup,
    fetchImpl: (async () => response(basicRelease, 'text/plain')) as typeof fetch,
  }), /unsupported Content-Type text\/plain/i, 'manifest MIME is fail-closed')

  await assert.rejects(() => checkUpdateFeed('https://example.com/feed.json', '0.1.0', undefined, {
    publicKey: publicPem,
    lookup,
    fetchImpl: (async () => new Response(JSON.stringify(basicRelease), {
      headers: {
        'content-type': 'application/json',
        'content-length': String(MAX_UPDATE_MANIFEST_BYTES + 1),
      },
    })) as typeof fetch,
  }), new RegExp(`exceeds the ${MAX_UPDATE_MANIFEST_BYTES}-byte limit`, 'i'), 'declared oversized manifests are rejected before parsing')

  let chunkedCancelled = false
  const oversizedStream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array(MAX_UPDATE_MANIFEST_BYTES))
      controller.enqueue(new Uint8Array([0x20]))
    },
    cancel() { chunkedCancelled = true },
  })
  await assert.rejects(() => checkUpdateFeed('https://example.com/feed.json', '0.1.0', undefined, {
    publicKey: publicPem,
    lookup,
    fetchImpl: (async () => new Response(oversizedStream, {
      headers: { 'content-type': 'application/json' },
    })) as typeof fetch,
  }), new RegExp(`exceeds the ${MAX_UPDATE_MANIFEST_BYTES}-byte limit`, 'i'), 'chunked oversized manifests are capped while streaming')
  assert.equal(chunkedCancelled, true, 'oversized chunked manifest body is cancelled')

  console.log('mandatory signed update checker tests passed')
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortJson(value))
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJson)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([key]) => key !== 'signature')
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => [key, sortJson(item)]))
}

void main()
