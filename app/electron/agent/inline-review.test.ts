import assert from 'node:assert/strict'
import { reviewDiff, reviewDiffWithRubric, parseRubric, type CheckRule, type ReviewFile } from './inline-review'

let ok = 0
const pass = (msg: string) => {
  ok++
  console.log(`ok: ${msg}`)
}

// --- findings are anchored to NEW-file line numbers on added lines ---
{
  const before = 'const a = 1\nconst b = 2\nconst c = 3'
  const after = 'const a = 1\nconsole.log(b)\nconst c = 3'
  const rules: CheckRule[] = [
    { id: 'no-console', message: 'raw console call', severity: 'error', pattern: /console\.\w+/ },
  ]
  const res = reviewDiff([{ path: 'src/x.ts', before, after }], rules)
  assert.equal(res.findings.length, 1, 'one finding')
  const f = res.findings[0]
  assert.equal(f.ruleId, 'no-console')
  assert.equal(f.side, 'added')
  assert.equal(f.line, 2, 'anchored to new line 2')
  assert.equal(f.match, 'console.log')
  assert.equal(f.blocking, true, 'error >= default threshold error → blocking')
  assert.equal(res.blocked, true)
  pass('added-line finding anchored to new line number')
}

// --- default scope is added-only: a match on unchanged context must NOT fire ---
{
  const before = 'console.log(1)\nconst b = 2'
  const after = 'console.log(1)\nconst b = 3'
  const rules: CheckRule[] = [{ id: 'nc', message: 'console', severity: 'warning', pattern: /console\.log/ }]
  const res = reviewDiff([{ path: 'a.ts', before, after }], rules)
  // console.log(1) is unchanged context → not flagged under default 'added' scope.
  assert.equal(res.findings.length, 0, 'context line not flagged by default')
  pass('default scope=added ignores unchanged context')
}

// --- scope=removed anchors to OLD line numbers ---
{
  const before = 'line1\ndebugger\nline3'
  const after = 'line1\nline3'
  const rules: CheckRule[] = [
    { id: 'no-debugger', message: 'debugger removed', severity: 'info', pattern: /debugger/, scope: 'removed' },
  ]
  const res = reviewDiff([{ path: 'a.ts', before, after }], rules)
  assert.equal(res.findings.length, 1)
  assert.equal(res.findings[0].side, 'removed')
  assert.equal(res.findings[0].line, 2, 'old line 2')
  pass('scope=removed anchors to old line number')
}

// --- ranking by severity desc, then path, then line ---
{
  const files: ReviewFile[] = [
    { path: 'b.ts', before: '', after: 'x\ny\nz' },
    { path: 'a.ts', before: '', after: 'p\nq' },
  ]
  const rules: CheckRule[] = [
    { id: 'warn', message: 'w', severity: 'warning', pattern: /./, scope: 'added' },
    { id: 'crit', message: 'c', severity: 'critical', pattern: /q/, scope: 'added' },
  ]
  const res = reviewDiff(files, rules)
  // critical must sort first
  assert.equal(res.findings[0].severity, 'critical', 'critical ranked first')
  assert.equal(res.findings[0].path, 'a.ts')
  // remaining warnings are path-then-line ordered: a.ts before b.ts
  const warns = res.findings.filter((f) => f.severity === 'warning')
  assert.ok(warns.length > 0)
  assert.equal(warns[0].path, 'a.ts', 'a.ts warnings before b.ts (path order)')
  // within a file, line order ascending
  const aWarns = warns.filter((f) => f.path === 'a.ts')
  assert.deepEqual(aWarns.map((f) => f.line), [1, 2], 'ascending line order within file')
  pass('ranking: severity desc → path → line')
}

// --- blockingThreshold controls the blocked flag ---
{
  const rules: CheckRule[] = [{ id: 'w', message: 'w', severity: 'warning', pattern: /x/ }]
  const files: ReviewFile[] = [{ path: 'a.ts', before: '', after: 'x' }]
  const lax = reviewDiff(files, rules, { blockingThreshold: 'error' })
  assert.equal(lax.blocked, false, 'warning below error threshold → not blocked')
  const strict = reviewDiff(files, rules, { blockingThreshold: 'warning' })
  assert.equal(strict.blocked, true, 'warning meets warning threshold → blocked')
  pass('blockingThreshold gates blocked flag')
}

// --- forceBlocking overrides threshold ---
{
  const rules: CheckRule[] = [{ id: 'w', message: 'w', severity: 'info', pattern: /x/, forceBlocking: true }]
  const res = reviewDiff([{ path: 'a.ts', before: '', after: 'x' }], rules, { blockingThreshold: 'critical' })
  assert.equal(res.blocked, true, 'forceBlocking wins over high threshold')
  pass('forceBlocking overrides threshold')
}

// --- pathInclude / pathExclude filtering ---
{
  const rules: CheckRule[] = [
    { id: 'test-only', message: 'no focus', severity: 'error', pattern: /\.only/, pathInclude: '.test.' },
    { id: 'src-only', message: 'no todo', severity: 'warning', pattern: /TODO/, pathExclude: '.test.' },
  ]
  const files: ReviewFile[] = [
    { path: 'a.test.ts', before: '', after: 'it.only(1)\nTODO' },
    { path: 'a.ts', before: '', after: 'it.only(2)\nTODO' },
  ]
  const res = reviewDiff(files, rules)
  const ids = res.findings.map((f) => `${f.ruleId}#${f.path}`).sort()
  // test-only fires only in the .test file; src-only fires only in the non-test file
  assert.deepEqual(ids, ['src-only#a.ts', 'test-only#a.test.ts'])
  pass('pathInclude/pathExclude filter by file path')
}

// --- pre-computed diff input path ---
{
  const rules: CheckRule[] = [{ id: 'q', message: 'q', severity: 'warning', pattern: /Q/ }]
  const res = reviewDiff(
    [{ path: 'a.ts', diff: [{ type: 'add', text: 'Q here', newNo: 42 }, { type: 'ctx', text: 'Q ctx', oldNo: 41, newNo: 41 }] }],
    rules,
  )
  assert.equal(res.findings.length, 1, 'only added line matched (default scope)')
  assert.equal(res.findings[0].line, 42, 'used supplied newNo')
  pass('accepts pre-computed DiffLine[]')
}

// --- message placeholder substitution with capture groups ---
{
  const rules: CheckRule[] = [
    { id: 'key', message: 'hardcoded key: $1', severity: 'critical', pattern: /apiKey\s*=\s*['"]([^'"]+)['"]/ },
  ]
  const res = reviewDiff([{ path: 'a.ts', before: '', after: "apiKey = 'sk-abc123'" }], rules)
  assert.equal(res.findings[0].message, 'hardcoded key: sk-abc123', 'placeholder filled from group 1')
  pass('message placeholders substitute capture groups')
}

// --- multiple matches on one line each yield a finding (no infinite loop) ---
{
  const rules: CheckRule[] = [{ id: 'x', message: 'x', severity: 'info', pattern: /x/ }]
  const res = reviewDiff([{ path: 'a.ts', before: '', after: 'x x x' }], rules)
  assert.equal(res.findings.length, 3, 'three matches on one line')
  pass('multiple matches per line, no infinite loop')
}

// --- maxFindings cap keeps highest severity ---
{
  const rules: CheckRule[] = [
    { id: 'lo', message: 'lo', severity: 'info', pattern: /a/ },
    { id: 'hi', message: 'hi', severity: 'critical', pattern: /a/ },
  ]
  const res = reviewDiff([{ path: 'f.ts', before: '', after: 'a' }], rules, { maxFindings: 1 })
  assert.equal(res.findings.length, 1)
  assert.equal(res.findings[0].severity, 'critical', 'cap kept the critical one')
  pass('maxFindings keeps highest severity')
}

// --- parseRubric: bracket severity, id:message, /regex/ ---
{
  const rubric = [
    '# My checks',
    '- [error] no-console: forbid raw console.* /console\\.\\w+/',
    '- [warning] todo-left: leftover marker `TODO`',
    'critical secret-key possible hardcoded secret /api[_-]?key/i',
    'this line is prose and should be ignored',
    '',
  ].join('\n')
  const rules = parseRubric(rubric)
  assert.equal(rules.length, 3, 'three rules parsed, prose ignored')
  assert.equal(rules[0].id, 'no-console')
  assert.equal(rules[0].severity, 'error')
  assert.equal(rules[0].message, 'forbid raw console.*')
  assert.equal(rules[1].severity, 'warning')
  assert.equal(rules[1].id, 'todo-left')
  // backtick literal is escaped-substring
  assert.equal(typeof rules[1].pattern, 'string')
  assert.equal(rules[2].id, 'secret-key')
  assert.equal(rules[2].severity, 'critical')
  pass('parseRubric parses bracket + bare-word severities')
}

// --- reviewDiffWithRubric end to end ---
{
  const rubric = '- [error] nc: no console /console\\.\\w+/'
  const res = reviewDiffWithRubric([{ path: 'a.ts', before: 'x=1', after: 'console.warn(1)' }], rubric)
  assert.equal(res.findings.length, 1)
  assert.equal(res.findings[0].ruleId, 'nc')
  assert.equal(res.findings[0].match, 'console.warn')
  assert.equal(res.blocked, true)
  pass('reviewDiffWithRubric wires rubric → findings')
}

// --- duplicate rubric ids get de-duped ---
{
  const rubric = ['error dup: first /aaa/', 'error dup: second /bbb/'].join('\n')
  const rules = parseRubric(rubric)
  assert.equal(rules.length, 2)
  assert.notEqual(rules[0].id, rules[1].id, 'ids de-duplicated')
  assert.equal(rules[0].id, 'dup')
  assert.equal(rules[1].id, 'dup-2')
  pass('duplicate rubric ids de-duplicated')
}

// --- counts tally by severity ---
{
  const rules: CheckRule[] = [
    { id: 'c', message: 'c', severity: 'critical', pattern: /c/ },
    { id: 'w', message: 'w', severity: 'warning', pattern: /w/ },
  ]
  const res = reviewDiff([{ path: 'f.ts', before: '', after: 'c\nw\nw' }], rules)
  assert.equal(res.counts.critical, 1)
  assert.equal(res.counts.warning, 2)
  assert.equal(res.counts.error, 0)
  assert.equal(res.counts.info, 0)
  pass('counts tally by severity')
}

// --- per-file counts preserve input order ---
{
  const rules: CheckRule[] = [{ id: 'x', message: 'x', severity: 'info', pattern: /x/ }]
  const res = reviewDiff(
    [
      { path: 'z.ts', before: '', after: 'x\nx' },
      { path: 'a.ts', before: '', after: 'x' },
    ],
    rules,
  )
  assert.deepEqual(res.files, [
    { path: 'z.ts', findings: 2 },
    { path: 'a.ts', findings: 1 },
  ])
  pass('per-file counts preserve supplied order')
}

console.log(`\n${ok} assertions passed`)
