import { createCipheriv, createDecipheriv, randomBytes, scrypt } from 'node:crypto'

const MAGIC = Buffer.from('TFBACKUP1\0')
const HEADER_BYTES = MAGIC.length + 16 + 12 + 4
export const MAX_BACKUP_BYTES = 64 * 1024 * 1024
export const MAX_BACKUP_JSON_BYTES = 48 * 1024 * 1024

function passwordBytes(password: string): Buffer {
  if (typeof password !== 'string' || password.length < 8 || password.length > 1024) throw new Error('Backup password must contain 8 to 1024 characters.')
  return Buffer.from(password, 'utf8')
}

async function derive(password: string, salt: Buffer): Promise<Buffer> {
  const input = passwordBytes(password)
  try {
    return await new Promise<Buffer>((resolve, reject) => {
      scrypt(input, salt, 32, { N: 32768, r: 8, p: 1, maxmem: 64 * 1024 * 1024 }, (error, key) => error ? reject(error) : resolve(key))
    })
  } finally { input.fill(0) }
}

export async function encryptWorkspaceBackup(plaintext: Uint8Array, password: string): Promise<Uint8Array> {
  if (!plaintext.byteLength || plaintext.byteLength > MAX_BACKUP_JSON_BYTES) throw new Error('Backup exceeds the supported archive size.')
  const salt = randomBytes(16), iv = randomBytes(12)
  const header = Buffer.concat([MAGIC, salt, iv, Buffer.alloc(4)])
  header.writeUInt32BE(plaintext.byteLength, header.length - 4)
  const key = await derive(password, salt)
  try {
    const cipher = createCipheriv('aes-256-gcm', key, iv)
    cipher.setAAD(header)
    const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()])
    return Buffer.concat([header, cipher.getAuthTag(), ciphertext])
  } finally { key.fill(0) }
}

export async function decryptWorkspaceBackup(input: Uint8Array, password: string): Promise<Buffer> {
  if (!(input instanceof Uint8Array) || input.byteLength < HEADER_BYTES + 17 || input.byteLength > MAX_BACKUP_BYTES) throw new Error('Invalid or oversized backup archive.')
  const bytes = Buffer.from(input)
  if (!bytes.subarray(0, MAGIC.length).equals(MAGIC) || bytes.readUInt32BE(HEADER_BYTES - 4) !== bytes.length - HEADER_BYTES - 16
    || bytes.length - HEADER_BYTES - 16 > MAX_BACKUP_JSON_BYTES) throw new Error('Invalid backup archive header.')
  const key = await derive(password, bytes.subarray(MAGIC.length, MAGIC.length + 16))
  try {
    const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(MAGIC.length + 16, MAGIC.length + 28))
    decipher.setAAD(bytes.subarray(0, HEADER_BYTES))
    decipher.setAuthTag(bytes.subarray(HEADER_BYTES, HEADER_BYTES + 16))
    return Buffer.concat([decipher.update(bytes.subarray(HEADER_BYTES + 16)), decipher.final()])
  } catch { throw new Error('Backup password or integrity check failed.') }
  finally { key.fill(0); bytes.fill(0) }
}
