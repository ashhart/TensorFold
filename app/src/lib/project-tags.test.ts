// Run: jiti src/lib/project-tags.test.ts
// Covers the pure tag-normalization + filter helpers used by the project tag
// editor and the Code/Chats sidebar filter.
import assert from 'node:assert/strict'
import {
  ALL_FILTER,
  UNTAGGED_FILTER,
  addTag,
  collectTags,
  hasUntaggedProjects,
  normalizeTag,
  projectMatchesTag,
  removeTag,
  tagTone,
  toggleTag,
} from './project-tags.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('normalizeTag tests:')
;(() => {
  assert.equal(normalizeTag('  Work '), 'work', 'trims + lowercases')
  assert.equal(normalizeTag('Client   A'), 'client a', 'collapses internal whitespace')
  assert.equal(normalizeTag(''), '', 'blank -> empty')
  assert.equal(normalizeTag('   '), '', 'whitespace-only -> empty')
  assert.equal(normalizeTag(null), '', 'null -> empty')
  assert.equal(normalizeTag(undefined), '', 'undefined -> empty')
  assert.equal(normalizeTag('All'), '', "reserved 'all' sentinel rejected")
  assert.equal(normalizeTag('UNTAGGED'), '', "reserved 'untagged' sentinel rejected")
  ok('normalizeTag trims/lowercases/collapses and rejects blanks + sentinels')
})()

console.log('\naddTag / removeTag / toggleTag tests:')
;(() => {
  assert.deepEqual(addTag([], 'Work'), ['work'], 'adds normalized')
  assert.deepEqual(addTag(['work'], 'WORK'), ['work'], 'dedupes case-insensitively')
  assert.deepEqual(addTag(undefined, 'personal'), ['personal'], 'tolerates undefined base')
  assert.deepEqual(addTag(['work'], '   '), ['work'], 'blank add is a no-op')

  const base = ['work', 'personal']
  const copy = addTag(base, 'client')
  assert.deepEqual(base, ['work', 'personal'], 'addTag does not mutate input')
  assert.deepEqual(copy, ['work', 'personal', 'client'], 'addTag returns extended copy')

  assert.deepEqual(removeTag(['work', 'personal'], 'Work'), ['personal'], 'removes normalized match')
  assert.deepEqual(removeTag(['work'], 'missing'), ['work'], 'remove of absent tag is a no-op')

  assert.deepEqual(toggleTag(['work'], 'work'), [], 'toggle removes when present')
  assert.deepEqual(toggleTag(['work'], 'personal'), ['work', 'personal'], 'toggle adds when absent')
  assert.deepEqual(toggleTag(['work'], '  '), ['work'], 'toggle of blank is a no-op')
  ok('add/remove/toggle normalize, dedupe, and never mutate inputs')
})()

console.log('\nprojectMatchesTag tests:')
;(() => {
  const work = { tags: ['work'] }
  const both = { tags: ['work', 'personal'] }
  const none = { tags: [] as string[] }
  const undef = {} as { tags?: string[] }

  assert.equal(projectMatchesTag(work, ALL_FILTER), true, "'all' matches everything")
  assert.equal(projectMatchesTag(none, ALL_FILTER), true, "'all' matches untagged too")

  assert.equal(projectMatchesTag(work, 'work'), true, 'tag membership matches')
  assert.equal(projectMatchesTag(work, 'WORK'), true, 'filter is normalized before compare')
  assert.equal(projectMatchesTag(work, 'personal'), false, 'non-member tag excluded')
  assert.equal(projectMatchesTag(both, 'personal'), true, 'matches any of multiple tags')

  assert.equal(projectMatchesTag(none, UNTAGGED_FILTER), true, "'untagged' matches no-tag project")
  assert.equal(projectMatchesTag(undef, UNTAGGED_FILTER), true, "'untagged' matches missing tags field")
  assert.equal(projectMatchesTag(work, UNTAGGED_FILTER), false, "'untagged' excludes a tagged project")
  ok('projectMatchesTag handles all/untagged/specific filters with normalization')
})()

console.log('\ncollectTags / hasUntaggedProjects tests:')
;(() => {
  const projects = [
    { tags: ['work', 'personal'] },
    { tags: ['Work'] }, // case-variant collapses onto 'work'
    { tags: ['client a'] },
    { tags: [] as string[] },
    {} as { tags?: string[] },
  ]
  // 'work' appears most (2), then the rest tie at 1 -> alphabetical.
  assert.deepEqual(collectTags(projects), ['work', 'client a', 'personal'], 'frequency-desc then alpha')
  assert.deepEqual(collectTags([]), [], 'no projects -> no tags')
  assert.deepEqual(collectTags([{ tags: [] }]), [], 'only-untagged -> no tags')

  assert.equal(hasUntaggedProjects(projects), true, 'detects an untagged project')
  assert.equal(hasUntaggedProjects([{ tags: ['work'] }]), false, 'all tagged -> false')
  ok('collectTags orders by frequency+alpha and dedupes; hasUntaggedProjects detects bare projects')
})()

;(() => {
  // tagTone: Personal is green (positive), Work is purple (accent), custom tags are neutral.
  assert.match(tagTone('personal'), /positive/, 'personal → green')
  assert.match(tagTone('Personal'), /positive/, 'case-insensitive')
  assert.match(tagTone('work'), /accent/, 'work → purple/accent')
  assert.match(tagTone('client-a'), /ink-/, 'a custom tag is neutral')
  ok('tagTone: personal=green, work=purple, custom=neutral')
})()

console.log(`\n${passed} passed`)
