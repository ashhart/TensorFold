import assert from 'node:assert/strict'
import { computeDiff, diffStat } from './diff'

// --- normal LCS path still works ---
{
  const before = 'a\nb\nc'
  const after = 'a\nx\nc'
  const diff = computeDiff(before, after)
  const stat = diffStat(diff)
  assert.equal(stat.added, 1, 'one added line')
  assert.equal(stat.removed, 1, 'one removed line')
  assert.ok(diff.some((l) => l.type === 'del' && l.text === 'b'), 'b deleted')
  assert.ok(diff.some((l) => l.type === 'add' && l.text === 'x'), 'x added')
  console.log('ok: small LCS diff')
}

// --- large-file guard: must NOT allocate the O(n*m) table or OOM ---
{
  // ~60k lines each side -> dense table would be ~3.6e9 cells (tens of GB).
  const N = 60_000
  const before = Array.from({ length: N }, (_, i) => `line ${i}`).join('\n')
  // Change only the middle line; common prefix/suffix dominate.
  const afterArr = Array.from({ length: N }, (_, i) => `line ${i}`)
  afterArr[30_000] = 'CHANGED'
  const after = afterArr.join('\n')

  const t0 = Date.now()
  const diff = computeDiff(before, after, 3)
  const ms = Date.now() - t0
  assert.ok(ms < 2000, `coarse diff should be fast, took ${ms}ms`)

  const stat = diffStat(diff)
  // Coarse diff trims common prefix/suffix, so only the changed middle remains.
  assert.equal(stat.removed, 1, `exactly one removed line, got ${stat.removed}`)
  assert.equal(stat.added, 1, `exactly one added line, got ${stat.added}`)
  assert.ok(diff.some((l) => l.type === 'del' && l.text === 'line 30000'), 'old middle deleted')
  assert.ok(diff.some((l) => l.type === 'add' && l.text === 'CHANGED'), 'new middle added')

  // Line numbers must be correct (1-based) on the changed line.
  const del = diff.find((l) => l.type === 'del' && l.text === 'line 30000')!
  assert.equal(del.oldNo, 30_001, `del oldNo, got ${del.oldNo}`)
  const add = diff.find((l) => l.type === 'add' && l.text === 'CHANGED')!
  assert.equal(add.newNo, 30_001, `add newNo, got ${add.newNo}`)
  console.log(`ok: large-file coarse diff (${ms}ms, ${diff.length} display lines)`)
}

// --- coarse path: pure append at end of a large file ---
{
  const N = 50_000
  const before = Array.from({ length: N }, (_, i) => `L${i}`).join('\n')
  const after = before + '\nAPPENDED'
  const diff = computeDiff(before, after, 3)
  const stat = diffStat(diff)
  assert.equal(stat.removed, 0, `no removals on append, got ${stat.removed}`)
  assert.equal(stat.added, 1, `one addition on append, got ${stat.added}`)
  assert.ok(diff.some((l) => l.type === 'add' && l.text === 'APPENDED'), 'appended line present')
  console.log('ok: large-file append')
}

// --- coarse path produces context around the change ---
{
  const N = 50_000
  const beforeArr = Array.from({ length: N }, (_, i) => `row ${i}`)
  const afterArr = beforeArr.slice()
  afterArr[25_000] = 'EDIT'
  const diff = computeDiff(beforeArr.join('\n'), afterArr.join('\n'), 3)
  const ctx = diff.filter((l) => l.type === 'ctx')
  assert.ok(ctx.length > 0, 'coarse diff includes context lines')
  assert.ok(diff.some((l) => l.type === 'hunk'), 'coarse diff includes a hunk header')
  console.log(`ok: coarse context (${ctx.length} ctx lines)`)
}

// --- identical input produces no add/del rows ---
{
  const same = 'a\nb\nc'
  const diff = computeDiff(same, same)
  const stat = diffStat(diff)
  assert.equal(stat.added, 0, 'no adds on identical input')
  assert.equal(stat.removed, 0, 'no dels on identical input')
  console.log('ok: identical input no-op')
}

// --- mid-line edit (sets up word-level diff in VST-021) ---
{
  const before = 'const x = 1\nreturn x'
  const after = 'const x = 2\nreturn x'
  const diff = computeDiff(before, after)
  const dels = diff.filter((l) => l.type === 'del')
  const adds = diff.filter((l) => l.type === 'add')
  assert.equal(dels.length, 1, 'one del line')
  assert.equal(adds.length, 1, 'one add line')
  assert.equal(dels[0].text, 'const x = 1', 'del is the old line')
  assert.equal(adds[0].text, 'const x = 2', 'add is the new line')
  assert.equal(dels[0].oldNo, 1, 'del oldNo is 1')
  assert.equal(adds[0].newNo, 1, 'add newNo is 1')
  console.log('ok: mid-line edit')
}

// --- CRLF files diff correctly (line content excludes the CR from display) ---
{
  const before = 'a\r\nb\r\nc'
  const after = 'a\r\nx\r\nc'
  const diff = computeDiff(before, after)
  const stat = diffStat(diff)
  assert.equal(stat.added, 1, 'CRLF: one add')
  assert.equal(stat.removed, 1, 'CRLF: one del')
  // The del/add text should preserve the CR so the source is faithfully shown;
  // line splitting is on \n only (jsdiff convention).
  assert.ok(diff.some((l) => l.type === 'del' && l.text.startsWith('b')), 'CRLF del present')
  console.log('ok: CRLF diff')
}

// --- empty inputs ---
{
  assert.equal(computeDiff('', '').length, 0, 'both empty -> no lines')
  const fromEmpty = computeDiff('', 'a\nb')
  assert.equal(diffStat(fromEmpty).added, 2, 'empty -> 2 lines added')
  assert.equal(diffStat(fromEmpty).removed, 0, 'empty -> nothing removed')
  const toEmpty = computeDiff('a\nb', '')
  assert.equal(diffStat(toEmpty).removed, 2, '2 lines -> empty removed')
  assert.equal(diffStat(toEmpty).added, 0, 'nothing added')
  console.log('ok: empty inputs')
}

console.log('ALL TESTS PASSED')
