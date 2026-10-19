import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { downloadVerifiedUpdate, reverifyDownloadedUpdate } from './update-download'

const bytes = Buffer.from('fixture update payload with multiple partial writes')
const checksum = { algorithm: 'sha256' as const, value: createHash('sha256').update(bytes).digest('hex') }
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'update-io-'))
const originalOpen = fs.open
let behavior: 'partial' | 'zero' | 'error' = 'partial'
let writeCount = 0

fs.open = (async (...args: Parameters<typeof fs.open>) => {
  const handle = await originalOpen(...args)
  if (String(args[0]).endsWith('.part')) {
    const write = handle.write.bind(handle)
    handle.write = (async (buffer: Buffer, offset = 0, length = buffer.length - offset, position = null) => {
      writeCount++
      if (behavior === 'error') throw new Error('fixture storage failure')
      if (behavior === 'zero') return { bytesWritten: 0, buffer }
      return write(buffer, offset, Math.min(length, 3), position)
    }) as typeof handle.write
  }
  return handle
}) as typeof fs.open

const input = { downloadUrl: 'https://download.example/update.zip', checksum, manifestSignatureVerified: true, downloadRoot: root }
const lookup = async () => [{ address: '93.184.216.34', family: 4 }]

try {
  const result = await downloadVerifiedUpdate(input, { lookup, fetchImpl: (async () => new Response(bytes)) as typeof fetch })
  assert.deepEqual(await fs.readFile(result.path), bytes, 'verified updates must contain every byte after short writes')
  assert.ok(writeCount > 1, 'a partial write must be retried')
  await reverifyDownloadedUpdate(result, root)
  await fs.rm(path.dirname(result.path), { recursive: true, force: true })

  behavior = 'zero'
  await assert.rejects(
    downloadVerifiedUpdate(input, { lookup, fetchImpl: (async () => new Response(bytes)) as typeof fetch }),
    /write.*progress|zero.*write/i,
    'a zero-byte write must fail instead of committing or looping forever',
  )
  assert.deepEqual(await fs.readdir(root), [])

  behavior = 'error'
  let cancelled = false
  const stream = new ReadableStream<Uint8Array>({
    start(controller) { controller.enqueue(bytes) },
    cancel() { cancelled = true },
  })
  const response = new Response(stream)
  await assert.rejects(downloadVerifiedUpdate(input, { lookup, fetchImpl: (async () => response) as typeof fetch }), /fixture storage failure/)
  assert.equal(cancelled, true, 'failed storage must cancel the network body')
  assert.equal(response.body?.locked, false, 'failed storage must release the reader lock')
  assert.deepEqual(await fs.readdir(root), [])
} finally {
  fs.open = originalOpen
  await fs.rm(root, { recursive: true, force: true })
}

console.log('Verified update I/O regressions passed')
