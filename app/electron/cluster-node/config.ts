import { promises as fs, constants } from 'node:fs'
import path from 'node:path'
import { randomBytes, timingSafeEqual } from 'node:crypto'
import type { ClusterNodeConfig } from './types'

export class NodeHttpError extends Error { constructor(readonly status: number, message: string) { super(message) } }
export function isLoopback(host: string): boolean { return host === '127.0.0.1' || host === '::1' || host === 'localhost' }
export function validateNodeConfig(input: unknown): ClusterNodeConfig {
  const value = input as ClusterNodeConfig
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !['schema', 'dataDir', 'enginePath', 'cacheDir', 'host', 'port', 'tls', 'fabricLibraryPath', 'advertise'].includes(key)) || value.schema !== 1 || !path.isAbsolute(value.dataDir ?? '') || value.dataDir.length > 4096 || typeof value.host !== 'string' || !value.host || /[\s\x00-\x1f\x7f/]/.test(value.host) || !Number.isSafeInteger(value.port) || value.port < 0 || value.port > 65535 || value.advertise !== undefined && typeof value.advertise !== 'boolean') throw new Error('Choose a valid private node configuration.')
  for (const field of ['enginePath', 'cacheDir', 'fabricLibraryPath'] as const) if (value[field] !== undefined && (!path.isAbsolute(value[field]!) || value[field]!.length > 4096 || /[\x00-\x1f\x7f]/.test(value[field]!))) throw new Error('Node paths must be explicit absolute paths.')
  if (!isLoopback(value.host) && !value.tls) throw new Error('Non-loopback node management requires an explicit TLS certificate and key.')
  if (value.tls && (Object.keys(value.tls).some(key => !['certFile', 'keyFile'].includes(key)) || !path.isAbsolute(value.tls.certFile ?? '') || !path.isAbsolute(value.tls.keyFile ?? ''))) throw new Error('Choose explicit TLS certificate and key files.')
  return structuredClone(value)
}
/** Do not follow a symlink namespace into another profile or disclose tokens through a shared directory. */
export async function privateDirectory(directory: string): Promise<void> {
  const absolute = path.resolve(directory)
  let current = path.parse(absolute).root
  for (const segment of absolute.slice(current.length).split(path.sep)) {
    if (!segment) continue
    current = path.join(current, segment)
    let stat = await fs.lstat(current).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return null; throw error })
    if (!stat) { await fs.mkdir(current, { mode: 0o700 }); stat = await fs.lstat(current) }
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('The node data namespace must contain only real directories.')
  }
  const stat = await fs.stat(absolute)
  if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) throw new Error('The node data directory must belong to its current user.')
  if (stat.mode & 0o077) throw new Error('Create a private node data directory with permissions 0700 before starting it.')
}
export async function privateRead(file: string, maxBytes: number): Promise<Buffer> {
  const handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    const stat = await handle.stat()
    if (!stat.isFile() || stat.size > maxBytes || stat.mode & 0o077 || typeof process.getuid === 'function' && stat.uid !== process.getuid()) throw new Error('The node private file is unavailable, oversized or shared with another user.')
    const result = await handle.readFile()
    if (result.length > maxBytes) throw new Error('The node private file exceeds its size limit.')
    return result
  } finally { await handle.close() }
}
export async function readToken(file: string): Promise<string> {
  const token = (await privateRead(file, 128)).toString('utf8').trim()
  if (!/^[a-f0-9]{64}$/.test(token)) throw new Error('Node credentials must contain a private 256-bit token.')
  return token
}
export async function createOrReadTokens(dataDir: string, files?: { management: string; inference: string }): Promise<{ management: string; inference: string }> {
  await privateDirectory(dataDir)
  const names = files ?? { management: path.join(dataDir, 'management.token'), inference: path.join(dataDir, 'inference.token') }
  const obtain = async (file: string) => {
    if (!path.isAbsolute(file)) throw new Error('Credential files must use explicit absolute paths.')
    const handle = await fs.open(file, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600).catch((error: NodeJS.ErrnoException) => { if (error.code === 'EEXIST') return null; throw error })
    if (handle) { try { await handle.writeFile(randomBytes(32).toString('hex') + '\n'); await handle.sync() } finally { await handle.close() } }
    return readToken(file)
  }
  const management = await obtain(names.management), inference = await obtain(names.inference)
  if (timingSafeEqual(Buffer.from(management), Buffer.from(inference))) throw new Error('Management and inference credentials must be different.')
  return { management, inference }
}
export function authorized(header: string | undefined, expected: string): boolean {
  const match = /^Bearer ([a-f0-9]{64})$/.exec(header ?? '')
  return !!match && timingSafeEqual(Buffer.from(match[1]), Buffer.from(expected))
}
export function publicMessage(error: unknown): string {
  return error instanceof NodeHttpError ? error.message : 'The node operation could not finish; inspect its local configuration and owned engine state.'
}
