// Run: jiti electron/agent/prompt-registry.test.ts
import assert from 'node:assert/strict'
import {
  commitVersion,
  diffBodies,
  diffVersions,
  emptyRegistry,
  hashBody,
  linkRun,
  normalizeBody,
  resolveLabel,
  rollbackLabel,
  runsForVersion,
  versionForRun,
  versionIdFor,
  versionsOf,
} from './prompt-registry.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('prompt-registry tests:')

// ---- normalize + hash are stable and whitespace-insensitive ----
{
  assert.equal(normalizeBody('a\r\nb  \n c \t\n'), 'a\nb\n c')
  assert.equal(hashBody('hello  '), hashBody('hello'))
  assert.notEqual(hashBody('hello'), hashBody('world'))
  ok('normalizeBody strips CRLF + trailing ws; hashBody is content-addressed')
}

// ---- first commit creates a version with seq 1 + a stable id ----
{
  const r0 = emptyRegistry()
  const { registry, version, deduped } = commitVersion(r0, { kind: 'persona', key: 'coder', body: 'Be terse.' }, 100)
  assert.equal(deduped, false)
  assert.equal(version.seq, 1)
  assert.equal(version.kind, 'persona')
  assert.equal(version.key, 'coder')
  assert.equal(version.createdAt, 100)
  assert.equal(version.id, versionIdFor('persona', 'coder', hashBody('Be terse.')))
  assert.equal(registry.versions.length, 1)
  ok('first commit creates seq-1 version with content-addressed id')
}

// ---- identical body dedupes (no new row), only moves the label ----
{
  let r = emptyRegistry()
  r = commitVersion(r, { kind: 'skill', key: 's', body: 'X', label: 'active' }, 1).registry
  const second = commitVersion(r, { kind: 'skill', key: 's', body: 'X\n\n', label: 'active' }, 2)
  assert.equal(second.deduped, true)
  assert.equal(second.registry.versions.length, 1, 'no duplicate version for identical normalized body')
  assert.equal(second.registry.versions[0].seq, 1)
  ok('identical (normalized) body dedupes to the same version')
}

// ---- changed body creates seq 2; versionsOf ordered oldest→newest ----
{
  let r = emptyRegistry()
  const v1 = commitVersion(r, { kind: 'persona', key: 'p', body: 'one' }, 1)
  r = v1.registry
  const v2 = commitVersion(r, { kind: 'persona', key: 'p', body: 'two' }, 2)
  r = v2.registry
  assert.equal(v2.version.seq, 2)
  assert.notEqual(v1.version.id, v2.version.id)
  const list = versionsOf(r, 'persona', 'p')
  assert.deepEqual(list.map((x) => x.seq), [1, 2])
  // a different key does not share the seq counter
  const other = commitVersion(r, { kind: 'persona', key: 'q', body: 'one' }, 3)
  assert.equal(other.version.seq, 1)
  ok('changed body increments seq; seq is per (kind,key)')
}

// ---- label resolution + rollback ----
{
  let r = emptyRegistry()
  const c1 = commitVersion(r, { kind: 'config', key: 'router', body: 'A', label: 'prod' }, 1)
  r = c1.registry
  const c2 = commitVersion(r, { kind: 'config', key: 'router', body: 'B', label: 'prod' }, 2)
  r = c2.registry
  // prod now points at v2
  assert.equal(resolveLabel(r, 'config', 'router', 'prod')?.id, c2.version.id)
  // roll prod back to v1
  r = rollbackLabel(r, 'config', 'router', 'prod', c1.version.id, 3)
  assert.equal(resolveLabel(r, 'config', 'router', 'prod')?.id, c1.version.id)
  assert.equal(resolveLabel(r, 'config', 'router', 'missing'), null)
  ok('label resolves to latest, rollbackLabel repoints it to an earlier version')
}

// ---- rollback rejects foreign / unknown versions ----
{
  let r = emptyRegistry()
  const a = commitVersion(r, { kind: 'config', key: 'router', body: 'A' }, 1)
  r = a.registry
  const b = commitVersion(r, { kind: 'config', key: 'other', body: 'A2' }, 2)
  r = b.registry
  assert.throws(() => rollbackLabel(r, 'config', 'router', 'prod', b.version.id, 3), /does not belong/)
  assert.throws(() => rollbackLabel(r, 'config', 'router', 'prod', 'nope', 3), /unknown versionId/)
  ok('rollbackLabel rejects unknown and cross-artifact version ids')
}

// ---- run linkage: link, resolve, reverse-lookup, overwrite ----
{
  let r = emptyRegistry()
  const c = commitVersion(r, { kind: 'subagent', key: 'reviewer', body: 'review carefully' }, 1)
  r = c.registry
  r = linkRun(r, 'run-1', c.version.id, 10)
  assert.equal(versionForRun(r, 'run-1')?.id, c.version.id)
  assert.equal(versionForRun(r, 'run-x'), null)
  assert.deepEqual(runsForVersion(r, c.version.id).map((x) => x.runId), ['run-1'])
  // re-link the same run overwrites (idempotent per runId)
  const c2 = commitVersion(r, { kind: 'subagent', key: 'reviewer', body: 'review harder' }, 11)
  r = c2.registry
  r = linkRun(r, 'run-1', c2.version.id, 12)
  assert.equal(r.runs.length, 1)
  assert.equal(versionForRun(r, 'run-1')?.id, c2.version.id)
  ok('linkRun links + resolves + reverse-looks-up + overwrites per runId')
}

// ---- linkRun rejects unknown versions ----
{
  const r = emptyRegistry()
  assert.throws(() => linkRun(r, 'run-1', 'ghost', 1), /unknown versionId/)
  ok('linkRun refuses to dangle a run off an unknown version')
}

// ---- rollback + run replay: point label at the version a good run used ----
{
  let r = emptyRegistry()
  const good = commitVersion(r, { kind: 'persona', key: 'p', body: 'good voice', label: 'active' }, 1)
  r = good.registry
  r = linkRun(r, 'run-good', good.version.id, 2)
  const bad = commitVersion(r, { kind: 'persona', key: 'p', body: 'bad voice', label: 'active' }, 3)
  r = bad.registry
  assert.equal(resolveLabel(r, 'persona', 'p', 'active')?.body, 'bad voice')
  // roll active back to whatever the good run used
  const goodVersion = versionForRun(r, 'run-good')!
  r = rollbackLabel(r, 'persona', 'p', 'active', goodVersion.id, 4)
  assert.equal(resolveLabel(r, 'persona', 'p', 'active')?.body, 'good voice')
  ok('replay: roll active label back to the version a known-good run used')
}

// ---- diff between two committed versions ----
{
  let r = emptyRegistry()
  const v1 = commitVersion(r, { kind: 'skill', key: 'd', body: 'line1\nline2\nline3' }, 1)
  r = v1.registry
  const v2 = commitVersion(r, { kind: 'skill', key: 'd', body: 'line1\nline2b\nline3\nline4' }, 2)
  r = v2.registry
  const diff = diffVersions(r, v1.version.id, v2.version.id)
  assert.equal(diff.identical, false)
  assert.equal(diff.removed, 1) // line2
  assert.equal(diff.added, 2)   // line2b + line4
  // context lines preserved
  assert.ok(diff.lines.some((l) => l.kind === 'context' && l.text === 'line1'))
  assert.ok(diff.lines.some((l) => l.kind === 'remove' && l.text === 'line2'))
  assert.ok(diff.lines.some((l) => l.kind === 'add' && l.text === 'line4'))
  ok('diffVersions computes add/remove/context line diff')
}

// ---- diff identical bodies + diff against empty baseline ----
{
  const same = diffBodies('a\nb', 'a\nb')
  assert.equal(same.identical, true)
  assert.equal(same.added, 0)
  assert.equal(same.removed, 0)

  let r = emptyRegistry()
  const v1 = commitVersion(r, { kind: 'config', key: 'x', body: 'l1\nl2' }, 1)
  r = v1.registry
  const fromEmpty = diffVersions(r, null, v1.version.id)
  assert.equal(fromEmpty.added, 2)
  assert.equal(fromEmpty.removed, 0)
  assert.equal(fromEmpty.fromId, null)
  ok('diff detects identical bodies and diffs against an empty baseline')
}

// ---- purity: commit/link/rollback never mutate the input registry ----
{
  const r0 = emptyRegistry()
  const c = commitVersion(r0, { kind: 'persona', key: 'p', body: 'x', label: 'active' }, 1)
  assert.equal(r0.versions.length, 0, 'input registry untouched by commit')
  assert.equal(r0.labels.length, 0)
  const linked = linkRun(c.registry, 'run', c.version.id, 2)
  assert.equal(c.registry.runs.length, 0, 'input registry untouched by linkRun')
  const rolled = rollbackLabel(linked, 'persona', 'p', 'active', c.version.id, 3)
  assert.notEqual(rolled.labels, linked.labels)
  ok('all mutators are pure — inputs never mutated')
}

console.log(`\nprompt-registry: ${passed} checks passed`)
