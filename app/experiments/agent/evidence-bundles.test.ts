import assert from 'node:assert/strict'
import {
  attachEvidenceBundleRefs,
  createEvidenceBundleRecord,
  exportEvidenceBundleRecord,
  normalizeEvidenceBundleMetadata,
  normalizeEvidenceBundleRecord,
  redactEvidenceBundleMetadata,
  summarizeEvidenceBundleCounts,
  validateEvidenceBundleInput,
  validateEvidenceBundleRecord,
} from './evidence-bundles.ts'
import { SECRET_MASK } from '../../electron/agent/security.ts'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ok ${name}`)
}

const OPENAI = 'sk-' + 'A1b2C3d4'.repeat(5)
const GH = 'ghp_' + 'a1B2c3D4'.repeat(5)
const BEARER = 'abcDEF1234567890XYZ'
const MASK = SECRET_MASK

{
  const bundle = createEvidenceBundleRecord({
    scope: 'mission',
    scopeId: 'mission-1',
    title: '  Verify evidence gates  ',
    sourceIds: ['thread-1', 'thread-1', ' checkpoint-1 '],
    artifactIds: 'run-export-1, screenshot-1; screenshot-1',
    metadata: { cwd: '/Users/example/repo', retries: 2 },
  }, 100)

  assert.equal(bundle.schema, 'vontra.evidence-bundle/1')
  assert.equal(bundle.id, 'bundle-mission-mission-1-2s')
  assert.equal(bundle.scope, 'mission')
  assert.equal(bundle.scopeId, 'mission-1')
  assert.equal(bundle.title, 'Verify evidence gates')
  assert.deepEqual(bundle.sourceIds, ['thread-1', 'checkpoint-1'])
  assert.deepEqual(bundle.artifactIds, ['run-export-1', 'screenshot-1'])
  assert.equal(bundle.createdAt, 100)
  assert.equal(bundle.updatedAt, 100)
  assert.deepEqual(validateEvidenceBundleRecord(bundle), { ok: true, errors: [] })
  ok('creates a normalized mission evidence bundle')
}

{
  const bundle = normalizeEvidenceBundleRecord({
    scope: 'ticket',
    scopeId: 'ticket-7',
    sourceIds: ['worker-thread', 'worker-thread'],
    artifactIds: [],
    metadata: { note: '  inspected   src/App.tsx  ', nested: { value: Number.POSITIVE_INFINITY }, skipped: undefined },
    createdAt: 10,
    updatedAt: 12,
  })

  assert.equal(bundle.title, 'Room ticket evidence')
  assert.deepEqual(bundle.sourceIds, ['worker-thread'])
  assert.deepEqual(bundle.artifactIds, [])
  assert.deepEqual(bundle.metadata, { note: 'inspected src/App.tsx', nested: { value: null } })
  assert.equal(bundle.createdAt, 10)
  assert.equal(bundle.updatedAt, 12)
  ok('normalizes ticket records and metadata without needing runtime services')
}

{
  const bundle = createEvidenceBundleRecord({
    scope: 'pr_loop',
    scopeId: 'lpr_123',
    sourceIds: ['impl-thread'],
    artifactIds: ['diff-1'],
  }, 200)
  const attached = attachEvidenceBundleRefs(bundle, {
    sourceIds: ['impl-thread', 'review-thread', 'mission-1'],
    artifactIds: 'diff-1; test-log-1',
  }, 250)
  const unchanged = attachEvidenceBundleRefs(attached, { sourceIds: ['review-thread'], artifactIds: ['test-log-1'] }, 300)

  assert.deepEqual(attached.sourceIds, ['impl-thread', 'review-thread', 'mission-1'])
  assert.deepEqual(attached.artifactIds, ['diff-1', 'test-log-1'])
  assert.equal(attached.updatedAt, 250)
  assert.equal(unchanged.updatedAt, 250)
  ok('attaches source and artifact ids idempotently')
}

{
  const metadata = {
    command: `curl -H "Authorization: Bearer ${BEARER}" https://example.com/report`,
    githubToken: GH,
    contact: 'owner@example.com',
    nested: {
      apiKey: OPENAI,
      password: 'hunter2hunter2',
      safe: 'src/App.tsx',
      host: '10.0.0.5',
    },
  }
  const redacted = redactEvidenceBundleMetadata(metadata, { anonymize: true })

  assert.ok(!JSON.stringify(redacted.metadata).includes(BEARER), 'bearer token removed')
  assert.ok(!JSON.stringify(redacted.metadata).includes(GH), 'github token removed')
  assert.ok(!JSON.stringify(redacted.metadata).includes(OPENAI), 'api key removed')
  assert.ok(!JSON.stringify(redacted.metadata).includes('hunter2hunter2'), 'key-named password removed')
  assert.ok(!JSON.stringify(redacted.metadata).includes('owner@example.com'), 'email removed')
  assert.ok(!JSON.stringify(redacted.metadata).includes('10.0.0.5'), 'ip removed when anonymizing')
  assert.equal((redacted.metadata.nested as any).apiKey, MASK)
  assert.ok(redacted.redactionReport.some((item) => item.kind === 'assignment' && item.count >= 2))
  assert.ok(redacted.redactionReport.some((item) => item.kind === 'bearer'))
  assert.ok(redacted.redactionReport.some((item) => item.kind === 'email'))
  assert.ok(redacted.redactionReport.some((item) => item.kind === 'ip'))
  ok('redacts sensitive exported metadata values and key-named secrets')
}

{
  const bundle = createEvidenceBundleRecord({
    scope: 'incident',
    scopeId: 'inc-1',
    metadata: { token: 'plain-secret-value', note: 'safe note' },
  }, 300)
  const exported = exportEvidenceBundleRecord(bundle)
  const raw = exportEvidenceBundleRecord(bundle, { redact: false })

  assert.equal(exported.redacted, true)
  assert.equal(exported.record.metadata.token, MASK)
  assert.equal(exported.record.metadata.note, 'safe note')
  assert.ok(exported.redactionReport.some((item) => item.kind === 'assignment'))
  assert.equal(raw.redacted, false)
  assert.equal(raw.record.metadata.token, 'plain-secret-value')
  assert.deepEqual(raw.redactionReport, [])
  ok('exports bundle records with metadata redaction on by default')
}

{
  assert.deepEqual(validateEvidenceBundleInput({ scope: 'mission', scopeId: 'm1' }), { ok: true, errors: [] })
  const invalidInput = validateEvidenceBundleInput({ scope: 'room', scopeId: '', createdAt: Number.NaN })
  assert.equal(invalidInput.ok, false)
  assert.match(invalidInput.errors.join('\n'), /scope must/)
  assert.match(invalidInput.errors.join('\n'), /scopeId is required/)
  assert.match(invalidInput.errors.join('\n'), /createdAt/)
  assert.throws(() => createEvidenceBundleRecord({ scope: 'mission', scopeId: '' }, 1), /scopeId is required/)

  const invalidRecord = validateEvidenceBundleRecord({ schema: 'bad', scope: 'assistant_output', scopeId: 'out-1' })
  assert.equal(invalidRecord.ok, false)
  assert.match(invalidRecord.errors.join('\n'), /schema/)
  assert.match(invalidRecord.errors.join('\n'), /id is required/)
  assert.match(invalidRecord.errors.join('\n'), /title is required/)
  ok('validates required input and record fields')
}

{
  const metadata = normalizeEvidenceBundleMetadata({
    keep: true,
    list: ['one', undefined, Number.NaN],
    fn: () => 'skip',
  })
  assert.deepEqual(metadata, { keep: true, list: ['one', null, null] })
  ok('keeps bundle metadata JSON-shaped')
}

{
  const bundles = [
    createEvidenceBundleRecord({ scope: 'mission', scopeId: 'm1', sourceIds: ['thread-1'], artifactIds: ['export-1'] }, 1),
    createEvidenceBundleRecord({ scope: 'ticket', scopeId: 't1', sourceIds: ['thread-1', 'thread-2'] }, 2),
    createEvidenceBundleRecord({ scope: 'assistant_output', scopeId: 'a1' }, 3),
    { scope: 'room', scopeId: '', sourceIds: ['bad-source'] },
  ]
  const summary = summarizeEvidenceBundleCounts(bundles)

  assert.equal(summary.total, 4)
  assert.equal(summary.byScope.mission, 1)
  assert.equal(summary.byScope.ticket, 1)
  assert.equal(summary.byScope.assistant_output, 1)
  assert.equal(summary.sourceIds, 3)
  assert.equal(summary.artifactIds, 1)
  assert.equal(summary.empty, 1)
  assert.equal(summary.invalid, 1)
  ok('summarizes bundle counts by scope with unique refs and invalid records')
}

console.log(`\nevidence-bundles tests: ${passed} passed`)
