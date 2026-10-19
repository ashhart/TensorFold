import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { readArtifactTextBounded, readArtifactTextPrefix } from './artifact-files'

const run = promisify(execFile)
const mode = process.argv.indexOf('--fifo-child')
if (mode >= 0) {
  const reader = process.argv[mode + 1] === 'prefix' ? readArtifactTextPrefix : readArtifactTextBounded
  await reader(process.argv[mode + 2], 100)
} else if (process.platform !== 'win32') {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'artifact-fifo-'))
  try {
    const fifo = path.join(root, 'artifact.txt')
    await run('mkfifo', [fifo])
    for (const reader of ['prefix', 'bounded']) {
      await assert.rejects(
        run(process.execPath, [
          path.resolve('node_modules/jiti/lib/jiti-cli.mjs'),
          path.resolve('electron/agent/artifact-files-special.test.ts'), '--fifo-child', reader, fifo,
        ], { timeout: 3000 }),
        (error: Error & { stderr?: string; killed?: boolean }) => {
          const rejected = /Artifact path is not a regular file/.test(`${error.message}\n${error.stderr ?? ''}`)
          if (!rejected) console.error({ message: error.message, stderr: error.stderr, killed: error.killed })
          return rejected
        },
        `${reader} must reject a FIFO without waiting for a writer`,
      )
    }
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
  console.log('Artifact special-file regressions passed')
} else {
  console.log('SKIP: POSIX named pipes are unavailable on Windows')
}
