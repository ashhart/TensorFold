import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { verifyWindowsSandboxHelper } from './windows-sandbox-helper'
import { windowsSandboxArguments } from './windows-sandbox'
import { localWindowsPath } from './windows-sandbox-paths'

for (const unsafe of ['\\\\server\\share', '\\\\?\\C:\\device', 'C:relative', 'file://server/share', 'C:\\file:stream']) {
  assert.throws(() => localWindowsPath(unsafe), /local|streams/)
}

const args = windowsSandboxArguments({ root: 'C:\\QA\\private', workspace: 'C:\\QA\\private\\workspace', scratch: 'C:\\QA\\private\\scratch',
  command: 'C:\\Windows\\System32\\cmd.exe', args: ['/d', '/s', '/c', 'echo fixture'], allowNetwork: false, allowWrite: false, parentPid: 100, timeoutMs: 5000 })
assert.deepEqual(args.slice(3, 9), ['C:\\QA\\private\\scratch', '0', '0', '5000', '100', 'abi1'])
assert.equal(args[9], '--')
assert.throws(() => windowsSandboxArguments({ root: '\\\\server\\share', workspace: 'C:\\QA\\workspace', scratch: 'C:\\QA\\scratch',
  command: 'cmd.exe', args: [], allowNetwork: true, allowWrite: true, parentPid: 1, timeoutMs: 0 }), /local absolute/)
const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-windows-helper-test-'))
try {
  const binary = path.join(folder, 'helper.exe'), metadata = path.join(folder, 'manifest.json')
  const bytes = Buffer.alloc(128); bytes.write('MZ')
  const manifest = { abi: 1, platform: 'win32', arch: 'x64', sha256: createHash('sha256').update(bytes).digest('hex') }
  await fs.writeFile(binary, bytes); await fs.writeFile(metadata, JSON.stringify(manifest))
  assert.equal(verifyWindowsSandboxHelper(binary, metadata).sha256, manifest.sha256)
  bytes[90] = 1; await fs.writeFile(binary, bytes)
  assert.throws(() => verifyWindowsSandboxHelper(binary, metadata), /integrity/)
  await fs.writeFile(metadata, JSON.stringify({ ...manifest, abi: 2 }))
  assert.throws(() => verifyWindowsSandboxHelper(binary, metadata), /unsupported/)
} finally { await fs.rm(folder, { recursive: true, force: true }) }
console.log('Windows launch contract denies implicit network/write access and rejects tampered or incompatible native helpers.')
