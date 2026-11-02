import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createHash, createPrivateKey, createPublicKey, sign } from 'node:crypto'
import { parseArgs } from 'node:util'
import { createJiti } from 'jiti'
const { values } = parseArgs({ options: { package: { type: 'string', multiple: true }, 'signing-key': { type: 'string' }, output: { type: 'string' } } })
if (!values.package?.length || !values['signing-key'] || !values.output) throw new Error('Use --package for each platform metadata file, --signing-key and --output.')
const jiti = createJiti(import.meta.url)
const { validateRuntimeCatalog, verifyRuntimeCatalog, SIGNATURE_DOMAIN, PACKAGE_LIMIT, CATALOG_LIMIT } = await jiti.import('../electron/agent/engine/runtime/catalog.ts')
const { readPrivateJson, extractRuntime } = await jiti.import('../electron/agent/engine/runtime/storage.ts')
const { updateTrustRootPublicKey } = await jiti.import('../electron/agent/update-trust.ts')
const now = Date.now(), packages = []
for (const file of values.package) packages.push(JSON.parse(await readPrivateJson(file, CATALOG_LIMIT)))
const doc = { schema: 1, publishedAt: now, expiresAt: now + 7 * 86400_000, packages }
validateRuntimeCatalog(doc)
const payload = Buffer.from(JSON.stringify(doc))
if (payload.length * 4 / 3 + 256 > CATALOG_LIMIT) throw new Error('Combined runtime catalog is too large.')
// Verify complete archives and inventories on the signing host without executing foreign binaries.
for (const [index, pkg] of packages.entries()) {
  const archive = path.join(path.dirname(path.resolve(values.package[index])), `${pkg.id}.tar.gz`)
  const stat = await fs.lstat(archive)
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || stat.size !== pkg.bytes || stat.size > PACKAGE_LIMIT) throw new Error('Runtime archive is not a complete regular file.')
  const bytes = await fs.readFile(archive)
  if (createHash('sha256').update(bytes).digest('hex') !== pkg.sha256) throw new Error('Runtime archive hash mismatch.')
  const temp = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tf-sign-runtime-')))
  try { await extractRuntime(archive, temp, pkg) } finally { await fs.rm(temp, { recursive: true, force: true }) }
}
const stat = await fs.lstat(values['signing-key'])
if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || process.platform !== 'win32' && (stat.mode & 0o077)) throw new Error('Use a protected signing key file.')
const key = createPrivateKey(await fs.readFile(values['signing-key']))
if (createPublicKey(key).export({ type: 'spki', format: 'pem' }).trim() !== updateTrustRootPublicKey().trim()) throw new Error('Signing key does not match the embedded production root.')
const raw = JSON.stringify({ payload: payload.toString('base64'), signature: sign(null, Buffer.concat([Buffer.from(SIGNATURE_DOMAIN), payload]), key).toString('base64') })
verifyRuntimeCatalog(raw)
await fs.writeFile(values.output, raw + '\n', { mode: 0o600, flag: 'wx' })
console.log(`Verified and signed ${packages.length} runtime packages locally. Nothing was uploaded.`)
