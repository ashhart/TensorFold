import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { reverifyDownloadedUpdate } from './update-download'

const run = promisify(execFile)
const child = process.argv.indexOf('--fifo-child')

if (child >= 0) {
  const root = process.argv[child + 1]
  const candidate = path.join(root, 'update.zip')
  const bytes = await fs.readFile(candidate)
  const info = await fs.lstat(candidate, { bigint: true })
  const originalOpen = fs.open
  let replaced = false
  fs.open = (async (...args: Parameters<typeof fs.open>) => {
    if (String(args[0]) === candidate && !replaced) {
      replaced = true
      await fs.rm(candidate)
      await run('mkfifo', [candidate])
    }
    return originalOpen(...args)
  }) as typeof fs.open
  try {
    await assert.rejects(reverifyDownloadedUpdate({
      path: candidate,
      finalUrl: 'https://download.example/update.zip',
      sha256: createHash('sha256').update(bytes).digest('hex'),
      bytes: bytes.length,
      redirects: 0,
      fileIdentity: { device: String(info.dev), inode: String(info.ino) },
    }, root), /file identity changed before re-verification/)
    assert.equal(replaced, true, 'the FIFO must replace a file after its initial identity checks')
  } finally {
    fs.open = originalOpen
  }
} else if (process.platform !== 'win32') {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'update-fifo-'))
  try {
    await fs.writeFile(path.join(root, 'update.zip'), 'fixture update payload')
    await run(process.execPath, [
      path.resolve('node_modules/jiti/lib/jiti-cli.mjs'),
      path.resolve('electron/agent/update-download-special.test.ts'), '--fifo-child', root,
    ], { timeout: 3000 })
    console.log('Verified update special-file race regression passed')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
} else {
  console.log('SKIP: POSIX named pipes are unavailable on Windows')
}
