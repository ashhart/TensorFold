import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { scanGitHygiene } from './git-hygiene'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'hygiene-hidden-'))
const files = ['.env.local', '.env.production', '.npmrc', '.config/service/settings.env']
try {
  for (const file of files) {
    await fs.mkdir(path.dirname(path.join(root, file)), { recursive: true })
    await fs.writeFile(path.join(root, file), `API_KEY=${['fixture', 'private', 'credential'].join('-')}\n`)
  }
  await fs.mkdir(path.join(root, '.cache'))
  await fs.writeFile(path.join(root, '.cache', 'ignored.txt'), 'ordinary generated cache data\n')
  const report = await scanGitHygiene(root)
  for (const file of files) {
    assert.ok(
      report.findings.some((finding) => finding.file === file && finding.kind === 'env_secret'),
      `hidden credential carrier ${file} must participate in the commit preflight`,
    )
  }
  assert.equal(report.blocked, true)
  assert.equal(report.scannedFiles, files.length, 'explicit generated-cache exclusions remain effective')
} finally {
  await fs.rm(root, { recursive: true, force: true })
}

console.log('Hidden-file Git hygiene regressions passed')
