import { generateKeyPairSync, sign, createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { c } from 'tar'
import { SIGNATURE_DOMAIN, type RuntimeCatalog, type RuntimePackage } from './catalog'

export const keys = generateKeyPairSync('ed25519')
export const publicKey = keys.publicKey.export({ format: 'pem', type: 'spki' }).toString()
export function signed(doc: RuntimeCatalog): string {
  const payload = Buffer.from(JSON.stringify(doc))
  return JSON.stringify({ payload: payload.toString('base64'), signature: sign(null,
    Buffer.concat([Buffer.from(SIGNATURE_DOMAIN), payload]), keys.privateKey).toString('base64') })
}
export function document(pkg: RuntimePackage): RuntimeCatalog {
  return { schema: 1, publishedAt: Date.now() - 1000, expiresAt: Date.now() + 3600_000, packages: [pkg] }
}
export async function fixture() {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tf-runtime-fixture-')))
  const source = path.join(root, 'source')
  await fs.mkdir(path.join(source, 'bin'), { recursive: true })
  await fs.mkdir(path.join(source, 'lib'), { recursive: true })
  const contents: Record<string, string> = {
    'bin/tensorfold-native': 'synthetic runtime fixture', LICENSE: 'fixture license', NOTICE: 'fixture notice',
    'THIRD_PARTY_NOTICES.md': 'fixture notices', 'lib/libtfgrammar.dylib': 'synthetic grammar library',
  }
  const files = []
  for (const [name, data] of Object.entries(contents)) {
    const executable = name.startsWith('bin/')
    await fs.writeFile(path.join(source, name), data, { mode: executable ? 0o700 : 0o600 })
    files.push({ path: name, size: Buffer.byteLength(data), sha256: createHash('sha256').update(data).digest('hex'), executable })
  }
  const archive = path.join(root, 'runtime.tar.gz')
  await c({ gzip: true, file: archive, cwd: source, portable: true }, Object.keys(contents))
  const bytes = await fs.readFile(archive)
  const pkg: RuntimePackage = { id: '1.1.7-darwin-arm64-metal', version: '1.1.7', platform: 'darwin', arch: 'arm64', backend: 'metal',
    notes: 'Synthetic fixture release.', protocol: 1, minAppVersion: '1.0.0-beta.10',
    url: 'https://github.com/ashhart/TensorFold/releases/download/v1.1.7/runtime.tar.gz',
    sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length, entrypoint: 'bin/tensorfold-native', files }
  return { root, source, archive, bytes, pkg, cleanup: () => fs.rm(root, { recursive: true, force: true }) }
}
