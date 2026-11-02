import { open, lstat } from 'node:fs/promises'

export function appImageLayout(prefix, size) {
  if (!Buffer.isBuffer(prefix) || prefix.length < 96 || prefix.toString('hex', 0, 4) !== '7f454c46' || prefix[4] !== 2 || prefix[5] !== 1 || prefix.toString('hex', 8, 11) !== '414902') throw new Error('Expected a 64-bit type-2 AppImage.')
  const machine = prefix.readUInt16LE(18)
  if (![62, 183].includes(machine)) throw new Error('Unsupported AppImage architecture.')
  const offsets = []
  let cursor = 0
  while ((cursor = prefix.indexOf('hsqs', cursor)) !== -1) {
    if (cursor + 96 <= prefix.length) {
      const used = prefix.readBigUInt64LE(cursor + 40), block = prefix.readUInt32LE(cursor + 12)
      if (prefix.readUInt16LE(cursor + 28) === 4 && prefix.readUInt16LE(cursor + 30) === 0 && [65536, 131072, 262144, 524288, 1048576].includes(block) && used >= 96n && used <= BigInt(size - cursor)) offsets.push(cursor)
    }
    cursor++
  }
  if (offsets.length !== 1) throw new Error('AppImage must contain one unambiguous SquashFS payload.')
  return { offset: offsets[0], arch: machine === 62 ? 'x64' : 'arm64' }
}
export async function readAppImageLayout(file) {
  const stat = await lstat(file)
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 2147483648) throw new Error('Expected a regular AppImage of at most 2 GiB.')
  const handle = await open(file, 'r')
  try {
    const prefix = Buffer.alloc(Math.min(stat.size, 2 * 1024 * 1024))
    await handle.read(prefix, 0, prefix.length, 0)
    return { ...appImageLayout(prefix, stat.size), prefix }
  } finally { await handle.close() }
}
