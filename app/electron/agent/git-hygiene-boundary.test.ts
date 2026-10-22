import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { sanitizeGitHygieneFindings } from './git-hygiene'
import type { GitHygieneFinding } from './types'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'hygiene-boundary-'))
const workspace = path.join(root, 'workspace')
const original = `API_KEY=${['fixture', 'private', 'value'].join('-')}\n`

function finding(file: string): GitHygieneFinding {
  return {
    id: file, source: 'working_tree', kind: 'env_secret', severity: 'critical',
    file, line: 1, commit: null, excerpt: '', match: '', suggestion: '', autoFixable: true,
  }
}

try {
  await fs.mkdir(workspace)
  const outside = path.join(root, 'outside.txt')
  await fs.writeFile(outside, original)
  await fs.symlink(outside, path.join(workspace, 'linked.txt'))
  const escaped = await sanitizeGitHygieneFindings(workspace, [finding('linked.txt')])
  assert.equal(await fs.readFile(outside, 'utf8'), original, 'sanitization must not mutate a symlink target outside the workspace')
  assert.deepEqual(escaped.changedFiles, [])
  assert.deepEqual(escaped.skippedFindings, ['linked.txt'])

  const directory = path.join(root, 'outside-dir')
  await fs.mkdir(directory)
  await fs.writeFile(path.join(directory, 'data.txt'), original)
  await fs.symlink(directory, path.join(workspace, 'parent'))
  const parent = await sanitizeGitHygieneFindings(workspace, [finding('parent/data.txt')])
  assert.equal(await fs.readFile(path.join(directory, 'data.txt'), 'utf8'), original)
  assert.deepEqual(parent.changedFiles, [])

  await fs.mkdir(path.join(workspace, '.git'))
  await fs.writeFile(path.join(workspace, '.git', 'config'), original)
  const metadata = await sanitizeGitHygieneFindings(workspace, [finding('.git/config')])
  assert.equal(await fs.readFile(path.join(workspace, '.git', 'config'), 'utf8'), original)
  assert.deepEqual(metadata.changedFiles, [], 'renderer-supplied findings cannot authorize Git metadata writes')

  await fs.writeFile(path.join(workspace, '.env'), original)
  const allowed = await sanitizeGitHygieneFindings(workspace, [finding('.env')])
  assert.deepEqual(allowed.changedFiles, ['.env'], 'explicit secret remediation remains available')
  assert.equal(await fs.readFile(path.join(workspace, '.env'), 'utf8'), 'API_KEY= [redacted]\n')
} finally {
  await fs.rm(root, { recursive: true, force: true })
}

console.log('Git hygiene workspace boundary regressions passed')
