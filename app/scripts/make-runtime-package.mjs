import { promises as fs } from 'node:fs'
import path from 'node:path'
import { createHash, createPrivateKey, createPublicKey, sign } from 'node:crypto'
import { parseArgs } from 'node:util'
import { c } from 'tar'
import { createJiti } from 'jiti'
const { values } = parseArgs({ options: Object.fromEntries(['source', 'output', 'id', 'version', 'platform', 'arch', 'backend', 'min-app', 'url', 'notes', 'signing-key'].map(key => [key, { type: 'string' }])) })
for (const key of ['source', 'output', 'id', 'version', 'platform', 'arch', 'backend', 'min-app', 'url']) if (!values[key]) throw new Error(`Missing --${key}.`)
const jiti = createJiti(import.meta.url)
const { validateRuntimeCatalog, verifyRuntimeCatalog, safeRuntimePath, runtimeId, SIGNATURE_DOMAIN } = await jiti.import('../electron/agent/engine/runtime/catalog.ts')
const { updateTrustRootPublicKey } = await jiti.import('../electron/agent/update-trust.ts')
const { probeRuntime } = await jiti.import('../electron/agent/engine/runtime/probe.ts')
runtimeId(values.id)
if (!['darwin', 'linux', 'win32'].includes(values.platform) || !['x64', 'arm64'].includes(values.arch)) throw new Error('Unsupported runtime platform.')
let key = null
if (values['signing-key']) {
const keyFile = await fs.lstat(values['signing-key'])
if (!keyFile.isFile() || keyFile.isSymbolicLink() || keyFile.nlink !== 1 || process.platform !== 'win32' && (keyFile.mode & 0o077)) throw new Error('Use a private signing-key file.')
key = createPrivateKey(await fs.readFile(values['signing-key']))
if (createPublicKey(key).export({ type: 'spki', format: 'pem' }).trim() !== updateTrustRootPublicKey().trim()) throw new Error('Signing key does not match the embedded production trust root.')
}
const source = await fs.realpath(values.source)
const output = path.resolve(values.output)
await fs.mkdir(output, { recursive: true, mode: 0o700 })
if (output === source || output.startsWith(source + path.sep)) throw new Error('Output must be outside the runtime source.')
const files = []
async function walk(dir, prefix = '') {
  for (const name of (await fs.readdir(dir)).sort()) {
    const rel = prefix + name, file = path.join(dir, name), stat = await fs.lstat(file)
    if (!safeRuntimePath(rel) || stat.isSymbolicLink()) throw new Error('Runtime source contains unsupported paths or links.')
    if (stat.isDirectory()) await walk(file, `${rel}/`)
    else {
      if (!stat.isFile() || stat.nlink !== 1) throw new Error('Runtime source must contain regular files only.')
      const hash = createHash('sha256')
      const handle = await fs.open(file)
      try { for await (const chunk of handle.createReadStream()) hash.update(chunk) } finally { await handle.close() }
      files.push({ path: rel, size: stat.size, sha256: hash.digest('hex'), executable: rel === (values.platform === 'win32' ? 'bin/tensorfold-native.exe' : 'bin/tensorfold-native') })
    }
  }
}
await walk(source)
const archive = path.join(output, `${values.id}.tar.gz`)
if (await fs.lstat(archive).then(() => true, e => { if (e.code === 'ENOENT') return false; throw e })) throw new Error('Output archive already exists.')
await c({ file: archive, cwd: source, gzip: true, portable: true, noMtime: true, noDirRecurse: true }, files.map(f => f.path))
const bytes = await fs.readFile(archive)
const pkg = { id: values.id, version: values.version, platform: values.platform, arch: values.arch, backend: values.backend,
  notes: values.notes ? await fs.readFile(values.notes, 'utf8') : '', protocol: 1, minAppVersion: values['min-app'], url: values.url,
  sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length,
  entrypoint: values.platform === 'win32' ? 'bin/tensorfold-native.exe' : 'bin/tensorfold-native', files }
// Probe the host runtime and its matching grammar library before signing.
if (pkg.platform !== process.platform || pkg.arch !== process.arch) throw new Error('Create runtime packages on their native build host.')
await probeRuntime(source, pkg)
const now = Date.now()
const doc = { schema: 1, publishedAt: now, expiresAt: now + 7 * 86400_000, packages: [pkg] }
validateRuntimeCatalog(doc)
await fs.writeFile(path.join(output, `${pkg.id}.json`), JSON.stringify(pkg, null, 2) + '\n', { mode: 0o600, flag: 'wx' })
if (key) {
  const payload = Buffer.from(JSON.stringify(doc))
  const catalog = JSON.stringify({ payload: payload.toString('base64'), signature: sign(null, Buffer.concat([Buffer.from(SIGNATURE_DOMAIN), payload]), key).toString('base64') })
  verifyRuntimeCatalog(catalog)
  await fs.writeFile(path.join(output, 'runtimes.json'), catalog + '\n', { mode: 0o600, flag: 'wx' })
}
console.log(`Verified runtime package ${pkg.id}: ${pkg.bytes} bytes, SHA-256 ${pkg.sha256}. ${key ? 'Catalog signed locally.' : 'Unsigned metadata prepared for the trusted signing host.'} Nothing was uploaded.`)
