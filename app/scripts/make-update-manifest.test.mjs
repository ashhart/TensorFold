import assert from 'node:assert/strict'
import { createHash, generateKeyPairSync } from 'node:crypto'
import { execFile } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { assertSigningKeyMatchesTrustRoot } from './update-signing-trust.mjs'

const run = promisify(execFile)
const root = fileURLToPath(new URL('..', import.meta.url))

async function main() {
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'vontra-update-manifest-'))
  try {
    const artifact = path.join(tmp, 'TensorFold-Studio-0.2.0-arm64.dmg')
    const notes = path.join(tmp, 'CHANGELOG.md')
    const out = path.join(tmp, 'update.json')
    const bytes = Buffer.from('vontra artifact fixture')
    await writeFile(artifact, bytes)
    await writeFile(notes, '# TensorFold Studio 0.2.0\n\n- Update checks are safer.\n')

    await run('node', [
      path.join(root, 'scripts/make-update-manifest.mjs'),
      '--version', '0.2.0',
      '--artifact', artifact,
      '--download-url', 'https://example.com/TensorFold-Studio-0.2.0-arm64.dmg',
      '--notes', notes,
      '--release-notes-url', 'https://example.com/vontra/0.2.0',
      '--out', out,
      '--unsigned-dev',
    ], { cwd: root })

    const manifest = JSON.parse(await readFile(out, 'utf8'))
    assert.equal(manifest.latest.version, '0.2.0')
    assert.equal(manifest.latest.sha256, createHash('sha256').update(bytes).digest('hex'))
    assert.equal(manifest.latest.assets[0].arch, 'arm64')
    assert.match(manifest.latest.releaseNotes, /Update checks are safer/)
    assert.equal(manifest.latest.releaseNotesUrl, 'https://example.com/vontra/0.2.0')
    assert.equal(manifest.latest.signature, undefined, 'explicit dev manifest is visibly unsigned')

    const zip = path.join(tmp, 'TensorFold-Studio-0.2.0-arm64.zip')
    const zipBytes = Buffer.from('automatic ZIP fixture')
    await writeFile(zip, zipBytes)
    await run(process.execPath, [path.join(root, 'scripts/make-update-manifest.mjs'), '--version', '0.2.0', '--artifact', artifact, '--download-url', 'https://example.com/studio.dmg', '--update-zip', zip, '--update-zip-url', 'https://example.com/studio.zip', '--out', out, '--unsigned-dev'], { cwd: root })
    const withZip = JSON.parse(await readFile(out, 'utf8'))
    assert.deepEqual(withZip.latest.automaticUpdate, { platform: 'macos', arch: 'arm64', zipUrl: 'https://example.com/studio.zip', sha256: createHash('sha256').update(zipBytes).digest('hex') })
    assert.equal(withZip.latest.assets.length, 2)
    await assert.rejects(() => run(process.execPath, [path.join(root, 'scripts/make-update-manifest.mjs'), '--version', '0.2.0', '--artifact', artifact, '--download-url', 'https://example.com/studio.dmg', '--update-zip', zip, '--out', out, '--unsigned-dev'], { cwd: root }), /both --update-zip and --update-zip-url/)

    await assert.rejects(
      () => run('node', [
        path.join(root, 'scripts/make-update-manifest.mjs'),
        '--version', '0.2.0',
        '--artifact', artifact,
        '--download-url', 'https://example.com/TensorFold-Studio-0.2.0-arm64.dmg',
        '--out', path.join(tmp, 'missing-signature.json'),
      ], { cwd: root }),
      /require --private-key/,
    )

    const { privateKey, publicKey } = generateKeyPairSync('ed25519')
    assert.doesNotThrow(
      () => assertSigningKeyMatchesTrustRoot(privateKey, publicKey),
      'release helper accepts the public SPKI derived from the supplied private key',
    )
    const { publicKey: wrongPublicKey } = generateKeyPairSync('ed25519')
    assert.throws(
      () => assertSigningKeyMatchesTrustRoot(privateKey, wrongPublicKey),
      /does not match the embedded TensorFold update trust root/i,
    )
    const privateKeyFile = path.join(tmp, 'update-private.pem')
    const signedOut = path.join(tmp, 'signed-update.json')
    await writeFile(privateKeyFile, privateKey.export({ type: 'pkcs8', format: 'pem' }))
    await assert.rejects(
      () => run('node', [
        path.join(root, 'scripts/make-update-manifest.mjs'),
        '--version', '0.2.0',
        '--artifact', artifact,
        '--download-url', 'https://example.com/TensorFold-Studio-0.2.0-arm64.dmg',
        '--out', signedOut,
        '--private-key', privateKeyFile,
      ], { cwd: root }),
      /does not match the embedded TensorFold update trust root/i,
      'manifest creation fails closed when the signing key cannot derive the embedded public SPKI',
    )

    const { privateKey: rsaKey } = generateKeyPairSync('rsa', { modulusLength: 2048 })
    const rsaKeyFile = path.join(tmp, 'wrong-rsa-private.pem')
    await writeFile(rsaKeyFile, rsaKey.export({ type: 'pkcs8', format: 'pem' }))
    await assert.rejects(
      () => run('node', [
        path.join(root, 'scripts/make-update-manifest.mjs'),
        '--version', '0.2.0', '--artifact', artifact,
        '--download-url', 'https://example.com/TensorFold-Studio-0.2.0-arm64.dmg',
        '--out', path.join(tmp, 'rsa-signed.json'), '--private-key', rsaKeyFile,
      ], { cwd: root }),
      /must contain an Ed25519 private key/,
    )

    await assert.rejects(
      () => run('node', [
        path.join(root, 'scripts/make-update-manifest.mjs'),
        '--version', '0.2.0',
        '--artifact', artifact,
        '--download-url', 'https://127.0.0.1/TensorFold-Studio-0.2.0-arm64.dmg',
        '--out', path.join(tmp, 'bad.json'),
        '--unsigned-dev',
      ], { cwd: root }),
      /download URL must use a public host/,
    )
  } finally {
    await rm(tmp, { recursive: true, force: true })
  }

  console.log('update manifest script tests passed')
}

void main()
