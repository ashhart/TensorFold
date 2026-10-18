import { deflateSync } from 'node:zlib'
/** Original eight-pixel red test square; never loads an image or asset from disk. */
export function qualificationImage(): string {
  const chunk = (name: string, data: Buffer) => {
    const type = Buffer.from(name), length = Buffer.alloc(4), crc = Buffer.alloc(4)
    length.writeUInt32BE(data.length)
    let checksum = 0xffffffff
    for (const byte of Buffer.concat([type, data])) { checksum ^= byte; for (let i = 0; i < 8; i++) checksum = checksum & 1 ? 0xedb88320 ^ checksum >>> 1 : checksum >>> 1 }
    crc.writeUInt32BE((checksum ^ 0xffffffff) >>> 0)
    return Buffer.concat([length, type, data, crc])
  }
  const header = Buffer.alloc(13); header.writeUInt32BE(8); header.writeUInt32BE(8, 4); header[8] = 8; header[9] = 2
  const pixels = Buffer.alloc(8 * 25)
  for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) pixels[row * 25 + 1 + col * 3] = 255
  return `data:image/png;base64,${Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', deflateSync(pixels)), chunk('IEND', Buffer.alloc(0))]).toString('base64')}`
}
