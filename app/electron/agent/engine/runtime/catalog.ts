import { verify } from 'node:crypto'
import { updateTrustRootPublicKey } from '../../update-trust'
import { compareSemver, parseSemver } from '../../cli-version-advisory'
import type { EngineRuntimeRelease } from '../../types/engine-runtime'

export const RUNTIME_FEED = 'https://tensorfold.dev/studio/runtimes.json'
export const CATALOG_LIMIT = 512 * 1024
export const PACKAGE_LIMIT = 512 * 1024 * 1024
export const INVENTORY_LIMIT = 2 * 1024 ** 3
export const SIGNATURE_DOMAIN = 'TensorFold runtime catalog v1\n'
export interface RuntimeFile { path: string; size: number; sha256: string; executable: boolean }
export interface RuntimePackage extends EngineRuntimeRelease {
  protocol: 1
  minAppVersion: string
  url: string
  sha256: string
  bytes: number
  entrypoint: string
  files: RuntimeFile[]
}
export interface RuntimeCatalog { schema: 1; publishedAt: number; expiresAt: number; packages: RuntimePackage[] }
export interface SignedRuntimeCatalog { payload: string; signature: string }
const HEX = /^[a-f0-9]{64}$/
export function runtimeId(value: unknown): string {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value) || !safeRuntimePath(value)) throw new Error('Choose a runtime from the verified catalog.')
  return value
}
export function safeRuntimePath(value: unknown): value is string {
  return typeof value === 'string' && value.length <= 240 && value.split('/').every(part =>
    /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(part) && !/[. ]$/.test(part) &&
    !/^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(part))
}
function invalid(): never { throw new Error('Invalid signed runtime catalog.') }
export function verifyRuntimeCatalog(raw: string, options: { publicKey?: string; now?: number; installed?: boolean } = {}): RuntimeCatalog {
  if (Buffer.byteLength(raw) > CATALOG_LIMIT) invalid()
  const envelope: SignedRuntimeCatalog = JSON.parse(raw)
  if (!envelope || Object.keys(envelope).sort().join(',') !== 'payload,signature' ||
    typeof envelope.payload !== 'string' || typeof envelope.signature !== 'string') invalid()
  const payload = Buffer.from(envelope.payload, 'base64')
  const signature = Buffer.from(envelope.signature, 'base64')
  if (payload.toString('base64') !== envelope.payload || signature.toString('base64') !== envelope.signature ||
    signature.length !== 64 || !verify(null, Buffer.concat([Buffer.from(SIGNATURE_DOMAIN), payload]),
      options.publicKey ?? updateTrustRootPublicKey(), signature)) throw new Error('Runtime catalog signature verification failed.')
  return validateRuntimeCatalog(JSON.parse(payload.toString('utf8')), options)
}
export function validateRuntimeCatalog(doc: RuntimeCatalog, options: { now?: number; installed?: boolean } = {}): RuntimeCatalog {
  const now = options.now ?? Date.now()
  if (doc?.schema !== 1 || !Number.isSafeInteger(doc.publishedAt) || !Number.isSafeInteger(doc.expiresAt) ||
    doc.expiresAt <= doc.publishedAt || doc.expiresAt - doc.publishedAt > 31 * 86400_000 ||
    !options.installed && (doc.publishedAt > now + 300_000 || doc.expiresAt <= now) ||
    !Array.isArray(doc.packages) || doc.packages.length > 128) invalid()
  const ids = new Set<string>()
  for (const pkg of doc.packages) {
    runtimeId(pkg.id)
    const version = typeof pkg.version === 'string' && /^1\.\d{1,5}\.\d{1,5}$/.test(pkg.version) ? parseSemver(pkg.version) : null
    if (ids.has(pkg.id.toLowerCase()) || !version || typeof pkg.minAppVersion !== 'string' || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(pkg.minAppVersion) || !parseSemver(pkg.minAppVersion) || pkg.protocol !== 1 ||
      !['darwin', 'linux', 'win32'].includes(pkg.platform) || !['x64', 'arm64'].includes(pkg.arch) ||
      typeof pkg.backend !== 'string' || !/^[a-z0-9-]{1,24}$/.test(pkg.backend) ||
      typeof pkg.notes !== 'string' || pkg.notes.length > 12000 || !HEX.test(pkg.sha256) ||
      !Number.isSafeInteger(pkg.bytes) || pkg.bytes < 1 || pkg.bytes > PACKAGE_LIMIT ||
      !safeRuntimePath(pkg.entrypoint) || !Array.isArray(pkg.files) || !pkg.files.length || pkg.files.length > 512) invalid()
    ids.add(pkg.id.toLowerCase())
    const url = new URL(pkg.url)
    if (url.protocol !== 'https:' || url.hostname !== 'github.com' || url.username || url.password || url.port ||
      url.search || url.hash || !/^\/ashhart\/TensorFold\/releases\/download\/[^/]+\/[^/]+\.tar\.gz$/.test(url.pathname) ||
      /(?:latest|nightly)/i.test(url.pathname)) invalid()
    const names = new Set<string>()
    let size = 0
    for (const file of pkg.files) {
      if (!safeRuntimePath(file.path) || names.has(file.path.toLowerCase()) || !HEX.test(file.sha256) ||
        !Number.isSafeInteger(file.size) || file.size < 0 || typeof file.executable !== 'boolean') invalid()
      names.add(file.path.toLowerCase()); size += file.size
      if (size > INVENTORY_LIMIT) invalid()
    }
    if (!pkg.files.some(f => f.path === pkg.entrypoint && f.executable && f.size > 0) ||
      !['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'].every(name => pkg.files.some(f => f.path === name))) invalid()
    const library = pkg.platform === 'darwin' ? 'lib/libtfgrammar.dylib' : 'lib/libtfgrammar.so'
    if (!pkg.files.some(f => f.path === library && f.size > 0)) invalid()
    for (const file of pkg.files) if (pkg.files.some(other => other.path !== file.path && other.path.startsWith(`${file.path}/`))) invalid()
  }
  return doc
}
export function compatibleRuntime(pkg: RuntimePackage, appVersion: string, platform: string, arch: string): boolean {
  const app = parseSemver(appVersion), min = parseSemver(pkg.minAppVersion)
  return !!app && !!min && pkg.platform === platform && pkg.arch === arch && pkg.protocol === 1 && compareSemver(app, min) >= 0
}
export function publicRuntime(pkg: RuntimePackage): EngineRuntimeRelease {
  const { id, version, platform, arch, backend, notes } = pkg
  return { id, version, platform, arch, backend, notes }
}
