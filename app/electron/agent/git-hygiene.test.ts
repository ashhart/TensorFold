import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { sanitizeGitHygieneFindings, scanGitHygiene } from './git-hygiene'
import type { GitHygieneFinding } from './types'

async function main(): Promise<void> {
  const root = await mkdtemp(path.join(tmpdir(), 'vontra-hygiene-'))

  try {
    const patternName = ['ENV', 'SECRET', 'RE'].join('_')
    const tokenWords = ['API[_-]?KEY', 'ACCESS[_-]?TOKEN', 'AUTH[_-]?TOKEN', 'SECRET', 'PASSWORD']
    const person = ['a', 'sh'].join('')
    const secretName = ['API', 'KEY'].join('_')
    const secretValue = ['super', 'secret', 'value'].join('')
    const privateEmail = [person, 'company.co.uk'].join('@')
    const privatePath = ['/Users', person, 'Documents', 'Projects'].join('/')
    const patterns = [
      `const ${patternName} = /\\b[A-Z0-9_]*(?:${tokenWords.join('|')})\\s*[:=]\\s*['"]?[^'"\\s#]{8,}/gi`,
      `const exampleAddress = "${['maya', 'example.com'].join('@')}"`,
      `const docsPath = "${['/Users', 'example', 'Projects'].join('/')}"`,
    ].join('\n')
    await writeFile(path.join(root, 'patterns.ts'), patterns, 'utf8')
    await writeFile(path.join(root, 'fixture.test.ts'), [
      'const messageId = "fixture@example.test"',
      'const token = { apiKey: "secret-a", accessToken: "oauth-token" }',
      'const sample = "AKIAABCDEFGHIJKLMNOP"',
      'const privateKeyFixture = "-----BEGIN RSA PRIVATE KEY-----\\nMIIabc123==\\n-----END RSA PRIVATE KEY-----"',
    ].join('\n'), 'utf8')

    const patternReport = await scanGitHygiene(root, '')
    assert.equal(patternReport.summary.critical, 0, 'regex definitions should not be treated as real env secrets')
    assert.equal(patternReport.summary.medium, 0, 'reserved example fixtures should not block hygiene')

    const fakeFinding: GitHygieneFinding = {
      id: 'fake_pattern_hit',
      source: 'working_tree',
      kind: 'env_secret',
      severity: 'critical',
      file: 'patterns.ts',
      line: 1,
      commit: null,
      excerpt: patterns.split('\n')[0],
      match: 'ENV_SECRET_RE = /...',
      suggestion: 'fake',
      autoFixable: true,
    }
    const sanitizePattern = await sanitizeGitHygieneFindings(root, [fakeFinding])
    assert.deepEqual(sanitizePattern.changedFiles, [], 'pattern definition lines should be skipped by sanitizer')
    assert.deepEqual(sanitizePattern.skippedFindings, ['fake_pattern_hit'])
    assert.equal(await readFile(path.join(root, 'patterns.ts'), 'utf8'), patterns)

    await writeFile(path.join(root, 'secrets.txt'), [
      `${secretName}=${secretValue}`,
      `owner=${privateEmail}`,
      `local=${privatePath}`,
    ].join('\n'), 'utf8')

    const report = await scanGitHygiene(root, '')
    assert.ok(report.findings.some((f) => f.kind === 'env_secret' && f.file === 'secrets.txt'))
    assert.ok(report.findings.some((f) => f.kind === 'email' && f.file === 'secrets.txt'))
    assert.ok(report.findings.some((f) => f.kind === 'absolute_path' && f.file === 'secrets.txt'))

    const result = await sanitizeGitHygieneFindings(root, report.findings.filter((f) => f.file === 'secrets.txt'))
    assert.deepEqual(result.changedFiles, ['secrets.txt'])
    const sanitized = await readFile(path.join(root, 'secrets.txt'), 'utf8')
    assert.ok(!sanitized.includes(secretValue))
    assert.ok(!sanitized.includes(privateEmail))
    assert.ok(!sanitized.includes(privatePath))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

main()
  .then(() => console.log('git-hygiene tests passed'))
  .catch((err) => {
    console.error(err)
    process.exitCode = 1
  })
