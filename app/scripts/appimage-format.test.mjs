import assert from 'node:assert/strict'
import { appImageLayout } from './appimage-format.mjs'
const prefix = Buffer.alloc(512)
prefix.write('7f454c4602010100414902', 0, 'hex'); prefix.writeUInt16LE(62, 18)
function superblock(offset) {
  prefix.write('hsqs', offset); prefix.writeUInt32LE(131072, offset + 12)
  prefix.writeUInt16LE(4, offset + 28); prefix.writeBigUInt64LE(192n, offset + 40)
}
superblock(128)
assert.deepEqual(appImageLayout(prefix, 1024), { offset: 128, arch: 'x64' })
prefix.writeUInt16LE(183, 18); assert.equal(appImageLayout(prefix, 1024).arch, 'arm64')
superblock(320); assert.throws(() => appImageLayout(prefix, 1024), /unambiguous/)
prefix.fill(0, 320); prefix.writeBigUInt64LE(3000n, 168); assert.throws(() => appImageLayout(prefix, 1024), /unambiguous/)
prefix[8] = 0; assert.throws(() => appImageLayout(prefix, 1024), /type-2/)
assert.throws(() => appImageLayout(Buffer.alloc(20), 20), /type-2/)
console.log('AppImage parser rejects ambiguous, truncated, oversized and unsupported archive layouts.')
