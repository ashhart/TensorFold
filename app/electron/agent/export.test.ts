// Run: npx tsx electron/agent/export.test.ts
import assert from 'node:assert'
import { redactText, aggregate } from './export-redaction.ts'
import { buildRunExport } from './export.ts'
import type { Thread } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const OPENAI = 'sk-' + 'A1b2C3d4'.repeat(5)
const AWS = 'AKIAABCDEFGHIJKLMNOP'
const GH = 'ghp_' + 'a1B2c3D4'.repeat(5)
const JWT = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
const PRIV = '-----BEGIN RSA PRIVATE KEY-----\nMIIabc123==\n-----END RSA PRIVATE KEY-----'
const MASK = '••••••••'

// ── redactText ───────────────────────────────────────────────────────────────
{
  for (const [label, secret, kind] of [
    ['openai key', OPENAI, 'api_key'],
    ['aws key', AWS, 'aws_key'],
    ['github token', GH, 'github_token'],
    ['jwt', JWT, 'jwt'],
  ] as const) {
    const r = redactText(`token here: ${secret} done`)
    assert.ok(!r.text.includes(secret), `${label} value removed`)
    assert.ok(r.text.includes(MASK), `${label} replaced with mask`)
    assert.ok(r.hits.includes(kind), `${label} hit recorded as ${kind}`)
  }
  ok('masks api keys / tokens / jwt and records hit kinds')
}

{
  const r = redactText(PRIV)
  assert.ok(!r.text.includes('MIIabc123'), 'private key body removed')
  assert.ok(r.hits.includes('private_key'), 'private_key hit recorded')
  ok('masks PEM private key blocks')
}

{
  // Regression: namespaced (underscore/hyphen-prefixed) keys must redact — \bword\b can't match
  // across an underscore, so DATABASE_PASSWORD etc. previously leaked through "redacted" exports.
  for (const sample of ['DATABASE_PASSWORD=superSecretValue1', 'APP_SECRET=plainlongvalue123', 'MY_API_KEY=abcdef123456', 'db_password: hunter2hunter2', 'service-access-token=zzzzzzzzzzzz']) {
    const r = redactText(sample)
    assert.ok(r.text.includes(MASK), `${sample} → value masked`)
    assert.ok(r.hits.includes('assignment'), `${sample} → assignment hit`)
    const secretPart = sample.split(/[=:]/).slice(1).join('').trim()
    assert.ok(!r.text.includes(secretPart), `${sample} → raw value gone`)
  }
  // Connection-string inline password.
  const cs = redactText('DB at postgres://admin:s3cretp4ss@db.internal:5432/app')
  assert.ok(!cs.text.includes('s3cretp4ss'), 'connection-string password masked')
  assert.ok(cs.text.includes('postgres://admin:'), 'connection-string user/scheme kept')
  ok('masks namespaced key=value secrets + connection-string passwords')
}

{
  const r = redactText('contact user@example.com at /Users/example/secret/thing')
  assert.ok(!r.text.includes('user@example.com'), 'email removed')
  assert.ok(r.text.includes('<redacted-email>'), 'email placeholder used')
  assert.ok(r.text.includes('/Users/<user>/'), 'home path username masked')
  assert.ok(!r.text.includes('/Users/example/'), 'real username not present')
  ok('masks email + home-path username (PII)')
}

{
  const plain = 'just a normal sentence about cart.ts and a fix'
  assert.strictEqual(redactText(plain).text, plain, 'clean text unchanged')
  assert.strictEqual(redactText(plain).hits.length, 0, 'no false-positive hits')
  ok('leaves clean text untouched (no false positives)')
}

{
  const opaque = 'q7Kp2_Zm9Ax4Vn8Ld3Rt6Yw1Bc5Hs0JuEeNf'
  const r = redactText(`probe ${opaque} complete`)
  assert.ok(!r.text.includes(opaque), 'keyword-less high-entropy credential removed')
  assert.ok(r.hits.includes('opaque_secret'))
  for (const safe of [
    'ordinarylowercasewordsthatarelongenoughbutnotrandom',
    '550e8400-e29b-41d4-a716-447700900501',
    'a3f5c8d714e9b2c06a1d8f9433c7a012d439ef12',
    '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
    '/tmp/q7Kp2_Zm9Ax4Vn8Ld3Rt6Yw1Bc5Hs0JuEeNf/report.json',
  ]) assert.ok(redactText(safe).text.includes(safe), `safe value preserved: ${safe}`)
  assert.ok(!redactText(`https://api.example/${opaque}/resource`).text.includes(opaque), 'opaque URL path credentials are not mistaken for filesystem paths')
  ok('masks opaque entropy while preserving prose, UUIDs, hashes, and paths')
}

{
  const noIp = redactText('host 10.0.0.5 responded')
  assert.ok(noIp.text.includes('10.0.0.5'), 'IP kept without anonymize')
  const anon = redactText('host 10.0.0.5 responded', true)
  assert.ok(!anon.text.includes('10.0.0.5'), 'IP masked when anonymizing')
  assert.ok(anon.hits.includes('ip'), 'ip hit recorded under anonymize')
  ok('IP only masked under anonymize')
}

{
  const report = aggregate(['api_key', 'api_key', 'email'])
  assert.deepStrictEqual(report.find((r) => r.kind === 'api_key'), { kind: 'api_key', count: 2 }, 'aggregate counts')
  ok('aggregate tallies hit kinds')
}

// ── buildRunExport ───────────────────────────────────────────────────────────
function thread(over: Partial<Thread> = {}): Thread {
  return {
    id: 'th_1', title: 'Fix checkout', kind: 'task', projectId: null, cwd: '/Users/example/repo',
    providerId: 'openai', model: 'gpt', createdAt: 1, updatedAt: 2, status: 'idle',
    apiMessages: [
      { role: 'user', content: `deploy with ${OPENAI}` } as any,
      { role: 'assistant', content: 'done' } as any,
    ],
    timeline: [
      { id: 'i1', kind: 'reasoning', text: 'thinking about it' },
      { id: 'i2', kind: 'tool', tool: 'bash', title: 'run', output: `key=${GH}`, state: 'ok' },
      { id: 'i3', kind: 'assistant', text: 'patched cart.ts' },
    ],
    ...over,
  } as Thread
}

{
  const b = buildRunExport(thread(), {}, 1000)
  assert.strictEqual(b.schema, 'vontra.run-export/1', 'schema set')
  assert.strictEqual(b.redacted, true, 'redacted on by default')
  assert.strictEqual(b.exportedAt, 1000, 'caller timestamp used')
  assert.ok(!JSON.stringify(b).includes(OPENAI), 'message secret not present in bundle')
  assert.ok(!JSON.stringify(b).includes(GH), 'tool-output secret not present in bundle')
  assert.ok(b.redactionReport.length > 0, 'redaction report populated')
  assert.strictEqual(b.stats.toolCalls, 1, 'tool calls counted')
  ok('buildRunExport redacts secrets across messages + tool output')
}

{
  const b = buildRunExport(thread({
    timeline: [
      { id: 'read', kind: 'tool', tool: 'read_file', title: 'Read app', path: '/Users/example/repo/src/App.tsx', output: 'file contents', state: 'ok', createdAt: 10 },
      {
        id: 'edit',
        kind: 'tool',
        tool: 'edit_file',
        title: 'Edit app',
        path: '/Users/example/repo/src/App.tsx',
        diff: [{ type: 'add', text: `+ const key = "${OPENAI}"` }],
        output: 'Updated src/App.tsx',
        meta: '+1 -0',
        state: 'ok',
        createdAt: 20,
      },
      { id: 'cmd', kind: 'tool', tool: 'run_command', title: 'Run tests', command: `npm test -- --token=${GH}`, output: 'Tests passed. See https://example.com/report.', meta: 'exit 0', state: 'ok', createdAt: 30 },
      { id: 'shot', kind: 'tool', tool: 'browser_screenshot', title: 'Screenshot', path: '/Users/example/Desktop/result.png', output: 'Captured result.png', state: 'ok', createdAt: 40 },
      { id: 'approval', kind: 'tool', tool: 'write_file', title: 'Write denied', approval: `Denied because ${AWS}`, state: 'denied', createdAt: 50 },
      { id: 'assistant', kind: 'assistant', text: 'Implemented and verified with tests.', createdAt: 60 },
    ],
  }), {}, 1000)
  assert.ok(b.evidence, 'evidence bundle present')
  assert.ok(b.chronicle, 'chronicle present')
  assert.ok(b.timeline.every((event) => event.id), 'timeline events carry trace ids')
  assert.ok(b.evidence!.touchedFiles.some((file) => file.path.includes('/Users/<user>/repo/src/App.tsx')), 'touched file path redacted')
  assert.ok(!JSON.stringify(b.evidence).includes('/Users/example/'), 'raw home path absent from evidence')
  assert.equal(b.evidence!.diffs.length, 1, 'diff evidence captured')
  assert.ok(!JSON.stringify(b.evidence!.diffs).includes(OPENAI), 'diff secret redacted')
  assert.equal(b.evidence!.commandOutputs.length, 1, 'command output evidence captured')
  assert.ok(!JSON.stringify(b.evidence!.commandOutputs).includes(GH), 'command secret redacted')
  assert.equal(b.evidence!.screenshots.length, 1, 'screenshot evidence captured')
  assert.equal(b.evidence!.citations[0].url, 'https://example.com/report', 'citation URL captured')
  assert.equal(b.evidence!.approvals.length, 1, 'approval evidence captured')
  assert.ok(!JSON.stringify(b.evidence!.approvals).includes(AWS), 'approval secret redacted')
  assert.equal(b.evidence!.assistantClaims.length, 1, 'assistant claim refs captured')
  assert.equal(b.evidence!.assistantClaims[0].itemId, 'assistant')
  assert.deepEqual(b.evidence!.assistantClaims[0].sourceItemIds, ['read', 'edit', 'cmd', 'shot', 'approval'])
  assert.equal(b.evidence!.assistantClaims[0].supported, true)
  assert.equal(b.evidence!.gaps.length, 0, 'evidence-backed claim has no gap')
  assert.match(b.chronicle!.summary, /6 events/)
  assert.match(b.chronicle!.summary, /1 command outputs/)
  assert.match(b.chronicle!.summary, /1 assistant claims/)
  ok('buildRunExport emits redacted evidence bundle + run chronicle')
}

{
  const b = buildRunExport(thread({
    timeline: [
      { id: 'assistant', kind: 'assistant', text: `Implemented the patch with ${OPENAI} and verified the test suite is green.`, createdAt: 10 },
    ],
  }), {}, 1000)
  assert.equal(b.evidence!.gaps.length, 1)
  assert.equal(b.evidence!.gaps[0].kind, 'claim_without_evidence')
  assert.match(b.evidence!.gaps[0].reason, /successful command output/)
  assert.ok(!b.evidence!.gaps[0].claim.includes(OPENAI), 'gap claim redacted')
  assert.equal(b.evidence!.assistantClaims.length, 1)
  assert.deepEqual(b.evidence!.assistantClaims[0].sourceItemIds, [])
  assert.equal(b.evidence!.assistantClaims[0].supported, false)
  assert.match(b.chronicle!.summary, /1 evidence gaps/)
  ok('buildRunExport flags verification claims without command evidence')
}

{
  const b = buildRunExport(thread(), { mode: 'share' }, 1)
  assert.strictEqual(b.anonymized, true, 'share mode anonymizes')
  assert.ok(b.thread.id.startsWith('anon-'), 'thread id hashed')
  assert.strictEqual(b.thread.cwd, undefined, 'cwd dropped when anonymized')
  assert.strictEqual(b.thread.provider, undefined, 'provider dropped when anonymized')
  ok('share packet anonymizes identifiers')
}

{
  const full = buildRunExport(thread(), { mode: 'full' }, 1)
  assert.ok(full.timeline.some((e) => e.kind === 'reasoning'), 'full keeps reasoning')
  const evalB = buildRunExport(thread(), { mode: 'eval' }, 1)
  assert.ok(!evalB.timeline.some((e) => e.kind === 'reasoning'), 'eval drops reasoning')
  assert.strictEqual(evalB.anonymized, true, 'eval anonymizes')
  ok('eval replay drops reasoning + anonymizes')
}

{
  const b = buildRunExport(thread(), { redact: false }, 1)
  assert.strictEqual(b.redacted, false, 'redact can be disabled')
  assert.ok(JSON.stringify(b).includes(OPENAI), 'raw secret present when redaction off')
  assert.strictEqual(b.redactionReport.length, 0, 'no report when redaction off')
  ok('redaction can be explicitly disabled')
}

{
  // Regression: compaction summaries are LLM-generated folds of earlier transcript and can
  // echo secrets. In default 'full' mode (redact true, anonymize false) they must be scrubbed.
  const b = buildRunExport(thread({
    compactions: [
      { id: 'c1', at: 5, trigger: 'auto', messagesFolded: 12, tokensBefore: 9000, tokensAfter: 1200, summaryExcerpt: `Earlier the tool printed key=${OPENAI} and the user pasted ${GH}.`, kindsDropped: ['tool', 'reasoning'] },
    ],
  }), {}, 1000)
  assert.ok(b.compactions, 'compactions retained in full mode')
  assert.equal(b.compactions!.length, 1, 'compaction entry preserved')
  assert.ok(!JSON.stringify(b.compactions).includes(OPENAI), 'compaction summary secret (openai) redacted')
  assert.ok(!JSON.stringify(b.compactions).includes(GH), 'compaction summary secret (github) redacted')
  assert.ok(JSON.stringify(b.compactions).includes(MASK), 'compaction summary carries mask')
  assert.equal(b.compactions![0].id, 'c1', 'compaction non-text fields preserved')
  assert.equal(b.compactions![0].messagesFolded, 12, 'compaction bookkeeping preserved')
  assert.deepEqual(b.compactions![0].kindsDropped, ['tool', 'reasoning'], 'compaction kindsDropped preserved')
  assert.ok(!JSON.stringify(b).includes(OPENAI), 'no compaction secret anywhere in full bundle')
  // anonymize still drops the array entirely.
  const share = buildRunExport(thread({
    compactions: [
      { id: 'c1', at: 5, trigger: 'auto', messagesFolded: 1, tokensBefore: 9, tokensAfter: 1, summaryExcerpt: 'plain summary', kindsDropped: [] },
    ],
  }), { mode: 'share' }, 1)
  assert.strictEqual(share.compactions, undefined, 'compactions dropped when anonymized')
  ok('buildRunExport scrubs compaction summaries in full mode and drops them when anonymized')
}

console.log(`\n${passed} export tests passed.`)
