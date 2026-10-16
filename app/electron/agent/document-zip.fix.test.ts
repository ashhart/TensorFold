// Run: jiti electron/agent/document-zip.fix.test.ts
// The bug: readZipEntries read compressedSize from the local file header but ignored
// general-purpose bit 3 (data descriptor). Streaming ZIP writers (used by many real .xlsx
// files) set bit 3 and store compressedSize=0 in the local header, so dataEnd=dataStart, an
// empty buffer was inflated/skipped, and the sheet yielded no rows. readZipEntries must now
// recover the compressed extent via the trailing data descriptor.
import assert from 'node:assert/strict'
import { deflateRawSync, crc32 } from 'node:zlib'
import { readZipEntries } from './document-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

// Build a single-entry ZIP written in STREAMING style: general-purpose bit 3 set, local
// header sizes = 0, and a trailing data descriptor (with the optional 0x08074b50 signature).
function streamedZip(name: string, content: Buffer, opts: { signature?: boolean } = {}): Buffer {
  const deflated = deflateRawSync(content)
  const crc = crc32(content)
  const nameBuf = Buffer.from(name, 'utf8')

  const local = Buffer.alloc(30)
  local.writeUInt32LE(0x04034b50, 0)   // local file header signature
  local.writeUInt16LE(20, 4)           // version needed
  local.writeUInt16LE(0x08, 6)         // general-purpose flags: bit 3 (data descriptor)
  local.writeUInt16LE(8, 8)            // method: deflate
  local.writeUInt16LE(0, 10)           // time
  local.writeUInt16LE(0, 12)           // date
  local.writeUInt32LE(0, 14)           // crc-32 (0 in local header, real value in descriptor)
  local.writeUInt32LE(0, 18)           // compressed size = 0 (streamed)
  local.writeUInt32LE(0, 22)           // uncompressed size = 0 (streamed)
  local.writeUInt16LE(nameBuf.length, 26)
  local.writeUInt16LE(0, 28)           // extra length

  const descParts: Buffer[] = []
  if (opts.signature !== false) {
    const sig = Buffer.alloc(4)
    sig.writeUInt32LE(0x08074b50, 0)
    descParts.push(sig)
  }
  const desc = Buffer.alloc(12)
  desc.writeUInt32LE(crc >>> 0, 0)
  desc.writeUInt32LE(deflated.length, 4)
  desc.writeUInt32LE(content.length, 8)
  descParts.push(desc)

  // Central directory + EOCD so readZipEntries' scan has a terminating signature.
  const central = Buffer.alloc(46)
  central.writeUInt32LE(0x02014b50, 0)
  central.writeUInt16LE(nameBuf.length, 28)

  const localBlock = Buffer.concat([local, nameBuf, deflated, ...descParts])
  return Buffer.concat([localBlock, central, nameBuf])
}

console.log('document-connector streamed-zip fix tests:')

{
  // Streamed entry WITH the data-descriptor signature.
  const zip = streamedZip('sheet.txt', Buffer.from('hello streamed world'))
  const entries = readZipEntries(zip)
  assert.equal(entries.get('sheet.txt')?.toString('utf8'), 'hello streamed world')
  ok('streamed entry with descriptor signature is read (was silently empty/skipped)')
}

{
  // Streamed entry WITHOUT the optional data-descriptor signature (12-byte descriptor).
  const zip = streamedZip('sheet.txt', Buffer.from('no-sig descriptor body'), { signature: false })
  const entries = readZipEntries(zip)
  assert.equal(entries.get('sheet.txt')?.toString('utf8'), 'no-sig descriptor body')
  ok('streamed entry without descriptor signature is read via the 12-byte fallback')
}

console.log(`\n${passed} checks passed.`)
