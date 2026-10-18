import { createHash, generateKeyPairSync, randomBytes } from 'node:crypto'
import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import type { ClusterSshKey } from '../../../../shared/cluster-workspace'

export interface ClusterKeyFs {
  lstat(file: string): Promise<{ isDirectory(): boolean; isFile(): boolean; isSymbolicLink(): boolean; mode: number; uid?: number; size?: number }>
  mkdir(file: string, options: { mode: number }): Promise<unknown>
  readFile(file: string): Promise<Buffer>
  open(file: string, flags: number, mode: number): Promise<{ writeFile(value: string): Promise<void>; sync(): Promise<void>; close(): Promise<void> }>
}
export interface ClusterKeyDeps {
  profileRoot(): string; fs?: ClusterKeyFs; uid?: number
  generatePair?: () => { privateKey: string; publicKey: string }
}
const magic = Buffer.from('openssh-key-v1\0')
const field = (value: Buffer | string) => { const bytes = Buffer.from(value); const size = Buffer.alloc(4); size.writeUInt32BE(bytes.length); return Buffer.concat([size, bytes]) }
const integer = (value: number) => { const result = Buffer.alloc(4); result.writeUInt32BE(value); return result }
export function ed25519PublicKey(publicBytes: Buffer): string {
  if (publicBytes.length !== 32) throw new Error('Invalid managed SSH public key.')
  return `ssh-ed25519 ${Buffer.concat([field('ssh-ed25519'), field(publicBytes)]).toString('base64')} TensorFold-managed-cluster`
}
/** Production-only CSPRNG; tests inject nonfunctional fixture encodings rather than create keys. */
export function generateManagedEd25519(): { privateKey: string; publicKey: string } {
  const pair = generateKeyPairSync('ed25519')
  const publicDer = pair.publicKey.export({ type: 'spki', format: 'der' }) as Buffer
  const privateDer = pair.privateKey.export({ type: 'pkcs8', format: 'der' }) as Buffer
  if (publicDer.length !== 44 || privateDer.length !== 48 || !privateDer.subarray(0, 16).equals(Buffer.from('302e020100300506032b657004220420', 'hex'))) throw new Error('Unsupported managed SSH key encoding.')
  const publicBytes = publicDer.subarray(-32), check = randomBytes(4)
  const publicBlob = Buffer.concat([field('ssh-ed25519'), field(publicBytes)])
  let body = Buffer.concat([check, check, field('ssh-ed25519'), field(publicBytes), field(Buffer.concat([privateDer.subarray(-32), publicBytes])), field('TensorFold-managed-cluster')])
  const padding = 8 - body.length % 8; body = Buffer.concat([body, Buffer.from(Array.from({ length: padding }, (_, index) => index + 1))])
  const encoded = Buffer.concat([magic, field('none'), field('none'), field(Buffer.alloc(0)), integer(1), field(publicBlob), field(body)]).toString('base64').match(/.{1,70}/g)!.join('\n')
  return { privateKey: `-----BEGIN OPENSSH PRIVATE KEY-----\n${encoded}\n-----END OPENSSH PRIVATE KEY-----\n`, publicKey: ed25519PublicKey(publicBytes) + '\n' }
}
function publicBlob(raw: string): Buffer {
  const match = /^ssh-ed25519 ([A-Za-z0-9+/]+={0,2})(?: [^\r\n]*)?\n?$/.exec(raw)
  if (!match) throw new Error('Managed SSH public key is invalid; existing files were preserved.')
  const blob = Buffer.from(match[1], 'base64')
  if (blob.length !== 51 || blob.readUInt32BE(0) !== 11 || blob.subarray(4, 15).toString() !== 'ssh-ed25519' || blob.readUInt32BE(15) !== 32) throw new Error('Managed SSH public key is invalid; existing files were preserved.')
  return blob
}
function privateBlob(raw: string): Buffer {
  const match = /^-----BEGIN OPENSSH PRIVATE KEY-----\n([A-Za-z0-9+/=\r\n]+)-----END OPENSSH PRIVATE KEY-----\n?$/.exec(raw)
  if (!match) throw new Error('Managed SSH private key is invalid; existing files were preserved.')
  const bytes = Buffer.from(match[1].replace(/\s/g, ''), 'base64')
  let offset = magic.length
  if (!bytes.subarray(0, offset).equals(magic)) throw new Error('Managed SSH private key is invalid.')
  const next = () => { if (offset + 4 > bytes.length) throw new Error('Managed SSH private key is truncated.'); const length = bytes.readUInt32BE(offset); offset += 4; if (length > bytes.length - offset) throw new Error('Managed SSH private key is truncated.'); const value = bytes.subarray(offset, offset + length); offset += length; return value }
  if (next().toString() !== 'none' || next().toString() !== 'none' || next().length !== 0 || bytes.readUInt32BE(offset) !== 1) throw new Error('Managed SSH key format is unsupported.')
  offset += 4
  const publicBytes = next(), body = next()
  if (offset !== bytes.length || body.length < 8 || !body.subarray(0, 4).equals(body.subarray(4, 8))) throw new Error('Managed SSH private key is invalid.')
  let cursor = 8
  const part = () => { if (cursor + 4 > body.length) throw new Error('Managed SSH private key is truncated.'); const length = body.readUInt32BE(cursor); cursor += 4; if (length > body.length - cursor) throw new Error('Managed SSH private key is truncated.'); const value = body.subarray(cursor, cursor + length); cursor += length; return value }
  if (part().toString() !== 'ssh-ed25519') throw new Error('Managed SSH private key is unsupported.')
  const rawPublic = part(), privateBytes = part(); part()
  if (rawPublic.length !== 32 || privateBytes.length !== 64 || !privateBytes.subarray(32).equals(rawPublic) || !publicBytes.equals(Buffer.concat([field('ssh-ed25519'), field(rawPublic)]))) throw new Error('Managed SSH private key does not match its public key.')
  const padding = body.subarray(cursor)
  if (!padding.length || padding.length > 8 || [...padding].some((value, index) => value !== index + 1)) throw new Error('Managed SSH private key padding is invalid.')
  return publicBytes
}
export function managedSshKeyPath(profileRoot: string): string { return path.join(profileRoot, 'clusters', 'ssh', 'id_ed25519') }
export function createManagedSshKeyStore(deps: ClusterKeyDeps) {
  const io = deps.fs ?? fs, uid = deps.uid ?? process.getuid?.()
  const inspect = async (create: boolean): Promise<ClusterSshKey> => {
    const root = deps.profileRoot(), file = managedSshKeyPath(root), directory = path.dirname(file)
    const current = () => { if (deps.profileRoot() !== root) throw new Error('The active profile changed during SSH key setup.') }
    const check = async (target: string, kind: 'directory' | 'file', privateMode = true) => {
      const stat = await io.lstat(target)
      if (stat.isSymbolicLink() || (kind === 'directory' ? !stat.isDirectory() : !stat.isFile()) || uid !== undefined && stat.uid !== undefined && stat.uid !== uid || (stat.mode & (privateMode ? 0o077 : 0o022)) !== 0 || kind === 'file' && stat.size !== undefined && stat.size > (privateMode ? 16_384 : 4096)) throw new Error('Managed SSH storage must be owned by this user, private and free of symbolic links; existing files were preserved.')
    }
    await check(root, 'directory')
    for (const dir of [path.join(root, 'clusters'), directory]) {
      try { await check(dir, 'directory') }
      catch (error: any) { if (error?.code !== 'ENOENT' || !create) { if (error?.code === 'ENOENT') return { available: false, publicKey: null, fingerprint: null, message: 'No app-owned SSH key exists in this profile.' }; throw error }; current(); await io.mkdir(dir, { mode: 0o700 }); await check(dir, 'directory') }
    }
    let privateData: Buffer, publicData: Buffer
    try { await check(file, 'file'); await check(file + '.pub', 'file', false); privateData = await io.readFile(file); publicData = await io.readFile(file + '.pub') }
    catch (error: any) {
      if (error?.code !== 'ENOENT') throw error
      // A half-created pair is recoverable data, never permission to overwrite the other half.
      const exists = async (target: string) => { try { await io.lstat(target); return true } catch (failure: any) { if (failure.code === 'ENOENT') return false; throw failure } }
      if (await exists(file) || await exists(file + '.pub')) throw new Error('A partial managed SSH key pair exists; preserve and repair it before creating another.')
      if (!create) return { available: false, publicKey: null, fingerprint: null, message: 'No app-owned SSH key exists in this profile.' }
      current(); const pair = (deps.generatePair ?? generateManagedEd25519)()
      if (!privateBlob(pair.privateKey).equals(publicBlob(pair.publicKey))) throw new Error('Generated SSH key pair does not match.')
      for (const [target, value, mode] of [[file, pair.privateKey, 0o600], [file + '.pub', pair.publicKey, 0o644]] as const) {
        current(); await check(root, 'directory'); await check(directory, 'directory')
        const handle = await io.open(target, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | (constants.O_NOFOLLOW ?? 0), mode)
        try { current(); await handle.writeFile(value); await handle.sync(); current() } finally { await handle.close() }
      }
      privateData = Buffer.from(pair.privateKey); publicData = Buffer.from(pair.publicKey)
    }
    current()
    if (privateData.length > 16_384 || publicData.length > 4096 || !privateBlob(privateData.toString()).equals(publicBlob(publicData.toString()))) throw new Error('Managed SSH key pair is invalid; existing files were preserved.')
    return { available: true, publicKey: publicData.toString().trim(), fingerprint: 'SHA256:' + createHash('sha256').update(publicBlob(publicData.toString())).digest('base64').replace(/=+$/, ''), message: 'App-owned SSH key is ready. Only its public key may be installed on a remote account.' }
  }
  return { inspect: () => inspect(false), create: () => inspect(true) }
}
