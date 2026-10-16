import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { LocalBackend, SshBackend } from './workspace-backend'
import { readOpenedTextFile, withFileReadBudget, WORKSPACE_READ_MAX_BYTES } from './bounded-file-read'
import { readBoundedSftpFile } from './ssh-connection'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-bounded-read-'))
const backend = new LocalBackend(root, async () => ({ code: 0, stdout: '', stderr: '' }))
try {
  const file = path.join(root, 'large.txt')
  const handle = await fs.open(file, 'w')
  await handle.write('first\nsecond\n')
  await handle.truncate(WORKSPACE_READ_MAX_BYTES * 4)
  await handle.close()
  await assert.rejects(() => backend.readFile(file), /read limit/)
  const range = await backend.readFileRange(file, { offset: 1, limit: 2, maxChars: 100 })
  assert.equal(range.text, '    1\tfirst\n    2\tsecond', 'early line selection never buffers the large remainder')
  await assert.rejects(() => backend.readFile(root), /non-regular/)
  const aborter = new AbortController()
  aborter.abort()
  await assert.rejects(() => backend.readFile(file, { signal: aborter.signal }), /aborted/)

  const utf8 = path.join(root, 'utf8.txt')
  await fs.writeFile(utf8, 'a'.repeat(65535) + '€\nlast\n')
  const selected = await backend.readFileRange(utf8, { offset: 2, limit: 2, maxChars: 100 })
  assert.equal(selected.text, '    2\tlast\n    3\t', 'UTF-8 crossing chunk boundaries preserves later lines and trailing newline')
  await assert.rejects(() => backend.readFileRange(utf8, { offset: 2, limit: 1, maxChars: 100, maxBytes: 128 }), /scan limit/)
} finally { await fs.rm(root, { recursive: true, force: true }) }

let reads = 0
let closes = 0
await assert.rejects(() => withFileReadBudget({}, (budget) => readOpenedTextFile({
  stat: async () => ({ isFile: false, size: 0 }),
  read: async () => { reads++; return 0 },
  close: async () => { closes++ },
}, budget)), /non-regular/)
assert.equal(reads, 0, 'special file metadata is rejected before any read')
assert.equal(closes, 1)

const requests: number[] = []
await assert.rejects(() => withFileReadBudget({ maxBytes: 4 }, (budget) => readOpenedTextFile({
  stat: async () => ({ isFile: true, size: 4 }),
  read: async (buffer, length) => { requests.push(length); buffer.fill(65); return length },
  close: async () => {},
}, budget)), /scan limit/)
assert.deepEqual(requests, [4, 1], 'a growing file is detected with one bounded EOF probe')

function sftpFixture(mode = 0o100644, size = 8) {
  const calls = { reads: 0, closes: 0, destroyed: 0 }
  let pendingRead: (() => void) | undefined
  const raw = {
    open(_path: string, _flags: string, cb: Function) { cb(null, Buffer.from('fixture')) },
    fstat(_handle: Buffer, cb: Function) { cb(null, { mode, size }) },
    read(_handle: Buffer, _buffer: Buffer, _offset: number, _length: number, _position: number, _cb: Function) { calls.reads++; pendingRead?.() },
    close(_handle: Buffer, cb: Function) { calls.closes++; cb() },
    destroy() { calls.destroyed++ },
  }
  return { raw, calls, whenRead: (fn: () => void) => { pendingRead = fn } }
}
{
  const { raw, calls } = sftpFixture(0o010644)
  await assert.rejects(() => readBoundedSftpFile(raw, '/fixture'), /non-regular/)
  assert.equal(calls.reads, 0)
  assert.equal(calls.closes, 1)
}
{
  const { raw, calls } = sftpFixture(0o100644, WORKSPACE_READ_MAX_BYTES + 1)
  await assert.rejects(() => readBoundedSftpFile(raw, '/fixture'), /read limit/)
  assert.equal(calls.reads, 0)
}
{
  const { raw, calls, whenRead } = sftpFixture()
  const aborter = new AbortController()
  whenRead(() => aborter.abort())
  await assert.rejects(() => readBoundedSftpFile(raw, '/fixture', { signal: aborter.signal }), /aborted/)
  assert.equal(calls.closes, 1, 'cancelling a pending SFTP read closes the opened handle')
  assert.equal(calls.destroyed, 1, 'cancellation releases the SFTP channel and pending requests')
}
{
  const { raw, calls } = sftpFixture()
  await assert.rejects(() => readBoundedSftpFile(raw, '/fixture', { timeoutMs: 10 }), /timed out/)
  assert.equal(calls.closes, 1, 'read timeout closes the opened SFTP handle')
  assert.equal(calls.destroyed, 1)
}
{
  const aborter = new AbortController()
  let observedSignal: AbortSignal | undefined
  const remote = new SshBackend('/fixture', {
    exec: async (_cmd, options) => {
      observedSignal = options?.signal
      queueMicrotask(() => aborter.abort())
      return new Promise(() => {})
    },
    sftp: async () => { throw new Error('must not open SFTP after cancellation') },
  })
  await assert.rejects(() => remote.readFile('/fixture/file', { signal: aborter.signal }), /aborted/)
  assert.equal(observedSignal?.aborted, true, 'cancellation reaches canonicalization before SFTP starts')
}
console.log('bounded local and SFTP read tests passed')
