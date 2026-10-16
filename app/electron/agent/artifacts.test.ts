// Run: jiti electron/agent/artifacts.test.ts
//
// First-class artifacts (VST-011): creation + kind inference, bounded previews,
// redaction (safe-copy semantics + binary honesty), load normalization, thread scoping.
import assert from 'node:assert/strict'
import {
  ARTIFACT_PREVIEW_CHARS,
  MAX_INLINE_ARTIFACT_CHARS,
  artifactPathAccessKind,
  artifactPreview,
  artifactsForThread,
  createArtifactRecord,
  inferArtifactKind,
  normalizeArtifactRecord,
  redactArtifactRecord,
} from './artifacts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('artifacts tests:')

{
  assert.equal(inferArtifactKind({ path: '/x/report.pdf' }), 'pdf')
  assert.equal(inferArtifactKind({ path: '/x/data.csv' }), 'spreadsheet')
  assert.equal(inferArtifactKind({ path: '/x/shot.png' }), 'image')
  assert.equal(inferArtifactKind({ mimeType: 'image/webp' }), 'image')
  assert.equal(inferArtifactKind({ kind: 'screenshot', path: '/x/whatever.bin' }), 'screenshot', 'explicit kind wins')
  assert.equal(inferArtifactKind({ path: '/x/unknown.zzz' }), 'file')
  ok('kind inference: explicit > mime > extension > file')
}

{
  const capturedRemote = createArtifactRecord({
    title: 'remote',
    path: '/srv/project/report.txt',
    provenance: { threadId: 'missing', workspace: { kind: 'ssh', hostId: 'host-1', path: '/srv/project' } },
  })
  assert.equal(artifactPathAccessKind(capturedRemote), 'ssh', 'captured provenance wins even after its thread disappears')
  const legacyRemote = createArtifactRecord({ title: 'legacy remote', path: '/srv/project/report.txt', provenance: { threadId: 't-ssh' } })
  assert.equal(artifactPathAccessKind(legacyRemote), 'unknown', 'legacy records fail closed even if their mutable thread is later reassigned')
  const local = createArtifactRecord({
    title: 'local',
    path: '/tmp/report.txt',
    provenance: { threadId: 't-local', workspace: { kind: 'local', path: '/tmp' } },
  })
  assert.equal(artifactPathAccessKind(local), 'local')
  assert.deepEqual(normalizeArtifactRecord(JSON.parse(JSON.stringify(capturedRemote)))?.provenance.workspace, capturedRemote.provenance.workspace)
  ok('path provenance distinguishes local, SSH, and unknown legacy artifacts without local-path guessing')
}

{
  const record = createArtifactRecord({
    title: '  Release   report ',
    inlineText: 'All good.',
    provenance: { threadId: 't1', tool: 'save_artifact', sourceRef: 'npm test' },
  }, 1000, 'art_1')
  assert.equal(record.title, 'Release report')
  assert.equal(record.kind, 'file')
  assert.equal(record.inlineText, 'All good.')
  assert.equal(record.sizeBytes, 9)
  assert.equal(record.redaction, 'unredacted')
  assert.equal(record.provenance.threadId, 't1')

  assert.throws(() => createArtifactRecord({ title: '', inlineText: 'x', provenance: { threadId: 't1' } }), /title/)
  assert.throws(() => createArtifactRecord({ title: 'x', provenance: { threadId: 't1' } }), /path or inline/)
  assert.throws(() => createArtifactRecord({ title: 'x', path: '/a', inlineText: 'b', provenance: { threadId: 't1' } }), /not both/)
  assert.throws(() => createArtifactRecord({ title: 'x', inlineText: 'b', provenance: { threadId: '' } }), /provenance/)

  const huge = createArtifactRecord({ title: 'big', inlineText: 'x'.repeat(MAX_INLINE_ARTIFACT_CHARS + 500), provenance: { threadId: 't1' } })
  assert.equal(huge.inlineText!.length, MAX_INLINE_ARTIFACT_CHARS)
  assert.equal(huge.inlineTruncated, true, 'truncation is recorded, never silent')
  ok('create: validation, provenance required, bounded inline with visible truncation')
}

{
  const textual = createArtifactRecord({ title: 'log', kind: 'log', inlineText: 'line\n'.repeat(600), provenance: { threadId: 't1' } })
  const preview = artifactPreview(textual)
  assert.equal(preview.excerpt!.length, ARTIFACT_PREVIEW_CHARS)
  assert.equal(preview.truncated, true)
  assert.match(preview.meta, /log/)

  const binary = createArtifactRecord({ title: 'shot', kind: 'screenshot', path: '/x/shot.png', sizeBytes: 2048, provenance: { threadId: 't1' } })
  const binaryPreview = artifactPreview(binary)
  assert.equal(binaryPreview.excerpt, null, 'binary kinds preview as metadata only')
  assert.match(binaryPreview.meta, /2\.0 KB/)
  ok('preview: bounded excerpt for text, metadata card for binary')
}

{
  const record = createArtifactRecord({ title: 'report', kind: 'report', inlineText: 'Contact maya@example.com — token sk-abc1234567890abcdefghij shipped.', provenance: { threadId: 't1' } }, 1000)
  const { artifact: redacted, content } = redactArtifactRecord(record, null, 2000)
  assert.equal(redacted.redaction, 'redacted')
  assert.ok(!content!.includes('maya@example.com'), 'emails are masked')
  assert.ok(!content!.includes('sk-abc1234567890abcdefghij'), 'secrets are masked')
  assert.ok(redacted.redactionReport.length > 0, 'the report says WHAT was masked')
  assert.ok(!redacted.inlineText!.includes('maya@example.com'), 'inline records carry the safe copy')

  const shot = createArtifactRecord({ title: 'shot', kind: 'screenshot', path: '/x/s.png', provenance: { threadId: 't1' } })
  const binary = redactArtifactRecord(shot, null, 2000)
  assert.equal(binary.artifact.redaction, 'not_required', 'binary artifacts say text redaction does not apply — never a fake "redacted"')
  ok('redact: masked safe copy + report; binary flips to not_required honestly')
}

{
  const good = createArtifactRecord({ title: 'a', inlineText: 'x', provenance: { threadId: 't1' } })
  assert.ok(normalizeArtifactRecord(JSON.parse(JSON.stringify(good))), 'round-trips')
  assert.equal(normalizeArtifactRecord(null), null)
  assert.equal(normalizeArtifactRecord({ id: 'a' }), null, 'missing fields degrade to null')
  assert.equal(normalizeArtifactRecord({ ...good, provenance: {} }), null, 'provenance is mandatory')
  ok('normalize: corrupt records drop out instead of poisoning the list')
}

{
  const a = createArtifactRecord({ title: 'a', inlineText: 'x', provenance: { threadId: 't1' } })
  const b = createArtifactRecord({ title: 'b', inlineText: 'x', provenance: { threadId: 't2' } })
  assert.deepEqual(artifactsForThread([a, b], 't1').map((x) => x.title), ['a'])
  ok('thread scoping: an artifact belongs to its run thread')
}

console.log(`  ${passed} passed`)
