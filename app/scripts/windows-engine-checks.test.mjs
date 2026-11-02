import assert from 'node:assert/strict'
import { assertWindowsPe, embeddedCudaImageCount, assertWindowsEngineCapabilities } from './windows-engine-checks.mjs'
const pe = Buffer.alloc(256); pe.write('MZ'); pe.writeUInt32LE(128, 0x3c); pe.write('PE\0\0', 128); pe.writeUInt16LE(0x8664, 132)
assert.doesNotThrow(() => assertWindowsPe(pe))
pe.writeUInt16LE(0xaa64, 132); assert.throws(() => assertWindowsPe(pe), /native x64 PE/)
assert.throws(() => assertWindowsPe(Buffer.from('MZ')), /native x64 PE/)
const blob = Buffer.alloc(48); Buffer.from([0x50, 0xed, 0x55, 0xba]).copy(blob); blob.writeUInt16LE(1, 4); blob.writeUInt16LE(16, 6); blob.writeBigUInt64LE(32n, 8)
assert.equal(embeddedCudaImageCount(Buffer.concat([blob, blob])), 2)
blob.writeBigUInt64LE(999n, 8); assert.equal(embeddedCudaImageCount(blob), 0)
const caps = { schema: 1, engine: 'zig', version: '1.1.6', backends: ['cuda'], optional_libraries: [{ name: 'TensorFold grammar', found: true }] }
assert.doesNotThrow(() => assertWindowsEngineCapabilities(caps, '1.1.6'))
for (const bad of [{ ...caps, version: '1.1.4' }, { ...caps, schema: 2 }, { ...caps, backends: [] }, { ...caps, optional_libraries: [] }]) assert.throws(() => assertWindowsEngineCapabilities(bad, '1.1.6'), /working grammar/)
console.log('Windows native architecture, embedded image and grammar/capability packaging guards passed.')
