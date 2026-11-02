export function assertWindowsPe(bytes, label = 'Windows engine') {
  if (bytes.length < 128 || bytes.toString('ascii', 0, 2) !== 'MZ') throw new Error(`${label} must be a native x64 PE file.`)
  const offset = bytes.readUInt32LE(0x3c)
  if (offset + 24 > bytes.length || bytes.toString('ascii', offset, offset + 4) !== 'PE\0\0' || bytes.readUInt16LE(offset + 4) !== 0x8664) throw new Error(`${label} must be a native x64 PE file.`)
}
export function embeddedCudaImageCount(bytes) {
  const magic = Buffer.from([0x50, 0xed, 0x55, 0xba])
  let at = 0, count = 0
  while ((at = bytes.indexOf(magic, at)) >= 0) {
    if (at + 16 <= bytes.length && bytes.readUInt16LE(at + 4) === 1) {
      const header = bytes.readUInt16LE(at + 6), size = bytes.readBigUInt64LE(at + 8)
      if (header >= 16 && header <= 4096 && size > 0 && size <= BigInt(bytes.length - at - header)) count++
    }
    at += 4
  }
  return count
}
export function assertWindowsEngineCapabilities(doc, version) {
  if (doc?.schema !== 1 || doc.engine !== 'zig' || doc.version !== version || !Array.isArray(doc.backends) || !doc.backends.includes('cuda') ||
    !Array.isArray(doc.optional_libraries) || !doc.optional_libraries.some(lib => lib.name === 'TensorFold grammar' && lib.found === true)) {
    throw new Error('The packaged Windows engine must report its matching version, CUDA backend and working grammar library.')
  }
}
