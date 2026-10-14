// Run: jiti electron/agent/pr-publish.test.ts
import assert from 'node:assert/strict'
import { buildPrPublishCommands, parsePrUrlFromGhOutput, publishLocalPr, type PrPublishRunner } from './pr-publish.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('pr-publish tests:')

const base = { branch: 'feat/x', baseBranch: 'main', title: 'Add X', bodyMarkdown: '## Summary\ndid X', worktree: '/w' }

// ---- command construction: git push then gh pr create --draft ----
{
  const cmds = buildPrPublishCommands(base)
  assert.equal(cmds.length, 2)
  assert.deepEqual(cmds[0].argv, ['git', 'push', '-u', 'origin', 'feat/x'])
  assert.equal(cmds[0].cwd, '/w')
  const gh = cmds[1].argv
  assert.equal(gh[0], 'gh'); assert.equal(gh[1], 'pr'); assert.equal(gh[2], 'create')
  assert.ok(gh.includes('--draft'), 'draft by default')
  assert.ok(gh.includes('--base') && gh[gh.indexOf('--base') + 1] === 'main')
  assert.ok(gh.includes('--head') && gh[gh.indexOf('--head') + 1] === 'feat/x')
  assert.ok(gh.includes('--title') && gh[gh.indexOf('--title') + 1] === 'Add X')
  assert.ok(gh.includes('--body') && gh[gh.indexOf('--body') + 1] === '## Summary\ndid X', 'body passed as a single argv element (no shell)')
  ok('builds git push + gh pr create --draft with base/head/title/body as argv')
}

// ---- draft:false drops the --draft flag; custom remote honored ----
{
  const cmds = buildPrPublishCommands({ ...base, draft: false, remote: 'upstream' })
  assert.equal(cmds[0].argv[3], 'upstream', 'custom remote used in push')
  assert.ok(!cmds[1].argv.includes('--draft'), 'no --draft when draft:false')
  ok('draft:false omits --draft; custom remote used')
}

// ---- null base branch omits --base (gh falls back to the repo default) ----
{
  const cmds = buildPrPublishCommands({ ...base, baseBranch: null })
  assert.ok(!cmds[1].argv.includes('--base'), 'no --base when baseBranch is null')
  ok('null baseBranch omits --base')
}

// ---- parse the PR url out of gh output ----
{
  assert.equal(parsePrUrlFromGhOutput('https://github.com/example-org/example-repo/pull/42\n'), 'https://github.com/example-org/example-repo/pull/42')
  assert.equal(parsePrUrlFromGhOutput('Creating draft pull request\nhttps://github.com/o/r/pull/7'), 'https://github.com/o/r/pull/7')
  assert.equal(parsePrUrlFromGhOutput('no url here'), null)
  ok('parsePrUrlFromGhOutput extracts the pull URL or null')
}

// A scripted runner: outcomes shift per call; records the argv it was given.
function scriptedRunner(outcomes: Array<{ ok: boolean; stdout?: string; stderr?: string }>): PrPublishRunner & { calls: string[][] } {
  const calls: string[][] = []
  return {
    calls,
    run: async (argv: string[]) => { calls.push(argv); return { stdout: '', stderr: '', ...(outcomes.shift() ?? { ok: false }) } },
  }
}

// ---- gate blocks: nothing runs when remote publish isn't approved ----
{
  const runner = scriptedRunner([{ ok: true }])
  const r = await publishLocalPr(base, { allowed: false, reason: 'needs approval' }, runner)
  assert.equal(r.ok, false)
  assert.match(r.error || '', /needs approval/)
  assert.equal(runner.calls.length, 0, 'no git/gh runs when the gate blocks')
  ok('gate not allowed → no commands run, returns the gate reason')
}

// ---- happy path: push then gh create → returns the parsed PR url ----
{
  const runner = scriptedRunner([
    { ok: true, stdout: 'Branch pushed' },
    { ok: true, stdout: 'https://github.com/example-org/example-repo/pull/99' },
  ])
  const r = await publishLocalPr(base, { allowed: true, reason: 'approved' }, runner)
  assert.equal(r.ok, true)
  assert.equal(r.prUrl, 'https://github.com/example-org/example-repo/pull/99')
  assert.equal(runner.calls.length, 2)
  assert.deepEqual(runner.calls[0], ['git', 'push', '-u', 'origin', 'feat/x'])
  ok('allowed → pushes then opens the draft PR, returns the URL')
}

// ---- push fails → stops before gh, surfaces the error ----
{
  const runner = scriptedRunner([{ ok: false, stderr: 'rejected: non-fast-forward' }])
  const r = await publishLocalPr(base, { allowed: true, reason: 'approved' }, runner)
  assert.equal(r.ok, false)
  assert.match(r.error || '', /non-fast-forward/)
  assert.equal(runner.calls.length, 1, 'gh pr create is not attempted after a failed push')
  ok('failed push stops the sequence and surfaces stderr')
}

// ---- gh create succeeds but prints no url → ok:false with a clear message ----
{
  const runner = scriptedRunner([{ ok: true }, { ok: true, stdout: 'created but weird output' }])
  const r = await publishLocalPr(base, { allowed: true, reason: 'approved' }, runner)
  assert.equal(r.ok, false, 'no parseable URL is treated as a failure to confirm')
  ok('gh success without a URL is reported as unconfirmed')
}

console.log(`\n${passed} checks passed.`)
