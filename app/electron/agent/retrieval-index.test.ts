import assert from 'node:assert/strict'
import {
  buildRetrievalIndex,
  formatRetrievalSearchResult,
  recencyWeight,
  searchRetrievalIndex,
} from './retrieval-index'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

console.log('retrieval-index tests:')

{
  const index = buildRetrievalIndex({
    codeFiles: [
      { path: 'src/approvals.ts', content: 'export function approveRenewal() { return "source-backed approval" }', modifiedAt: 10 },
    ],
    documents: [
      { id: 'docs:docs:brief', connectorId: 'docs', sourceId: 'docs', itemId: 'brief', title: 'Launch brief', excerpt: 'Reviewer evidence and approval notes.' },
    ],
    connectorEvents: [
      {
        id: 'evt-email',
        connectorId: 'email',
        pluginName: 'Email',
        action: 'message_received',
        direction: 'inbound',
        classification: 'communication',
        priority: 'high',
        trustLevel: 'suggest',
        requiresApproval: false,
        approvalId: null,
        approvalStatus: null,
        resolvedAt: null,
        resolvedBy: null,
        title: 'Renewal question',
        summary: 'Client asked whether approval evidence is ready.',
        actor: 'Maya',
        threadKey: 'email:renewal',
        payloadKeys: ['subject'],
        sources: [{ id: 'email:src:one', connectorId: 'email', sourceId: 'src', itemId: 'one', title: 'Renewal question', excerpt: 'approval evidence' }],
        createdAt: 20,
      },
      {
        id: 'evt-slack',
        connectorId: 'slack',
        pluginName: 'Messages',
        action: 'message_received',
        direction: 'inbound',
        classification: 'communication',
        priority: 'normal',
        trustLevel: 'notify',
        requiresApproval: false,
        approvalId: null,
        approvalStatus: null,
        resolvedAt: null,
        resolvedBy: null,
        title: 'Launch thread',
        summary: 'Chat says release-note ownership needs approval.',
        actor: '#launch',
        threadKey: 'slack:launch',
        payloadKeys: ['message'],
        sources: [{ id: 'slack:src:msg', connectorId: 'slack', sourceId: 'src', itemId: 'msg', title: '#launch', excerpt: 'ownership needs approval' }],
        createdAt: 21,
      },
    ],
    threads: [{
      id: 'thread-1',
      title: 'Approval chat',
      kind: 'task',
      cwd: '/workspace/vontra',
      providerId: null,
      model: null,
      createdAt: 1,
      updatedAt: 30,
      status: 'idle',
      apiMessages: [{ role: 'user', content: 'Find the approval trail.' }],
      timeline: [{ id: 'tl-1', kind: 'assistant', text: 'The chat summary cites renewal approval evidence.', createdAt: 30 }],
      compactions: [{ id: 'cmp-1', at: 31, trigger: 'auto', messagesFolded: 4, tokensBefore: 9000, tokensAfter: 600, summaryExcerpt: 'Folded approval summary for renewal work.', kindsDropped: ['assistant'] }],
    }],
    missions: [{
      id: 'mission-1',
      title: 'Approval mission',
      checkpoints: [{ id: 'chk-1', sequence: 1, label: 'Evidence', summary: 'Sandbox artifact proves approval check passed.', status: 'running', createdAt: 40, artifactRefs: ['sbox-123'] }],
    } as any],
    projectMemory: [{
      id: 'mem-1',
      projectId: 'project-1',
      kind: 'summary',
      text: 'Durable summary says approval evidence must cite connector sources.',
      sourceType: 'connector',
      sourceRef: 'email:src:one',
      sourceIds: ['email:src:one', 'slack:src:msg'],
      confidence: 'high',
      privacyScope: 'project',
      owner: null,
      lastVerifiedAt: 50,
      createdAt: 50,
      updatedAt: 50,
    }],
    artifacts: [{ id: 'artifact-1', title: 'Patch bundle', text: 'Artifact contains approval diff and logs.', sourceIds: ['artifact:patch-1'], createdAt: 60 }],
  }, 100)

  assert.equal(index.stats.total, index.records.length)
  assert.deepEqual(index.stats.byKind, {
    code: 1,
    doc: 1,
    chat: 3,
    email: 1,
    summary: 3,
    artifact: 2,
  })
  assert.ok(index.records.find((record) => record.kind === 'summary' && record.sourceIds.includes('email:src:one')))
  assert.ok(index.records.find((record) => record.kind === 'artifact' && record.sourceIds.includes('artifact:sbox-123')))
  ok('builds one retrieval index across code, docs, chats, emails, summaries, and artifacts')

  const result = searchRetrievalIndex(index, { query: 'approval evidence', limit: 4 })
  assert.equal(result.hits.length, 4)
  assert.ok(result.hits[0].score >= result.hits[1].score)
  assert.ok(result.hits.some((hit) => hit.record.sourceIds.includes('email:src:one')))
  assert.ok(result.hits.every((hit) => hit.matchedTerms.includes('approval') || hit.matchedTerms.includes('evidence')))
  ok('ranks retrieval hits and keeps source ids attached')

  const formatted = formatRetrievalSearchResult(result)
  assert.match(formatted, /Retrieval index:/)
  assert.match(formatted, /Sources:/)
  assert.match(formatted, /email:src:one/)
  ok('formats retrieval hits with prompt-visible citations')
}

{
  // Recency decay: fresh ≈ full weight, one half-life ≈ midpoint, very old → floor.
  const now = 1_000 * 86_400_000
  assert.ok(recencyWeight(now, now) > 0.99, 'just-created keeps ~full weight')
  assert.ok(Math.abs(recencyWeight(now - 30 * 86_400_000, now, 30) - 0.75) < 0.02, 'one half-life ~midpoint (0.5..1)')
  assert.ok(recencyWeight(now - 3650 * 86_400_000, now, 30) < 0.55, 'very old decays to the floor')
  assert.equal(recencyWeight(0, now), 0.5, 'unknown createdAt -> floor')
  ok('recencyWeight: decays score with age toward a floor')
}

{
  // [LOW #28] normalizeRetrievalRecord must not permanently concatenate title/path into
  // record.text -- doing so double-counts matches (title-match bonus AND an inflated
  // text-match bonus that's only true because of the appended title) and pollutes both
  // the dedup hash and excerpt() with display metadata instead of real body content.
  // Use the raw `records` input so this isolates normalizeRetrievalRecord itself, not
  // any upstream connector-specific text building.
  const genuineBody = [
    'The unrelated memo genuinely discusses revenue at length: revenue grew across every region,',
    'revenue per seat improved, revenue retention held steady, revenue mix shifted toward services,',
    'revenue recognition timing was clarified, revenue forecasts were revised upward, revenue from',
    'new logos outpaced revenue from renewals, revenue per rep climbed for the fourth straight quarter,',
    'revenue concentration risk dropped, revenue from upsells doubled, revenue overall beat plan,',
    'revenue per channel was reviewed, and revenue guidance for next quarter was finalized.',
  ].join(' ')
  const index = buildRetrievalIndex({
    records: [
      {
        id: 'doc:q4',
        kind: 'doc',
        title: 'Q4 Revenue Report',
        path: 'docs/finance/q4-report.md',
        text: 'Quarterly numbers and headcount notes with no mention of the forbidden word.',
      },
      {
        id: 'doc:genuine',
        kind: 'doc',
        title: 'Unrelated memo',
        path: 'docs/misc/memo.md',
        text: genuineBody,
      },
    ],
  }, 100)

  const titleRecord = index.records.find((record) => record.title === 'Q4 Revenue Report')
  assert.ok(titleRecord, 'title record indexed')
  assert.ok(
    !titleRecord!.text.toLowerCase().includes('revenue'),
    `record.text must stay body-only and not gain "revenue" via title/path concatenation, got: ${titleRecord!.text}`,
  )
  assert.ok(!titleRecord!.text.includes('Q4 Revenue Report'), 'record.text must not contain the appended title')
  assert.ok(!titleRecord!.text.includes('docs/finance/q4-report.md'), 'record.text must not contain the appended path')
  ok('normalizeRetrievalRecord keeps text as body-only content (no title/path concatenation)')

  // The title-only record legitimately matches via the separate title-match bonus, but
  // must NOT outrank a record that genuinely discusses "revenue" at length in its body --
  // pre-fix it did, purely because the appended title polluted the text-match scoring.
  const result = searchRetrievalIndex(index, { query: 'revenue', limit: 10 })
  const titleHit = result.hits.find((hit) => hit.record.title === 'Q4 Revenue Report')
  const genuineHit = result.hits.find((hit) => hit.record.title === 'Unrelated memo')
  assert.ok(genuineHit, 'genuinely-matching body record is a hit')
  assert.ok(titleHit, 'title-only record still matches via the separate title-match bonus')
  assert.ok(
    genuineHit!.score > titleHit!.score,
    `genuine body match (${genuineHit!.score}) must outrank a title-only match inflated by text concatenation (${titleHit!.score})`,
  )
  ok('search ranks genuine body matches above a title-match record with no body mention')

  const formatted = formatRetrievalSearchResult(searchRetrievalIndex(index, { query: 'revenue', limit: 10, kinds: 'doc' }))
  const titleLineIndex = formatted.indexOf('Q4 Revenue Report')
  const block = formatted.slice(titleLineIndex, formatted.indexOf('\n\n', titleLineIndex))
  const excerptLine = block.split('\n').pop() ?? ''
  assert.ok(
    !excerptLine.includes('docs/finance/q4-report.md'),
    `formatted excerpt line must not surface the appended path as if it were body content: ${excerptLine}`,
  )
  ok('excerpt() (via formatRetrievalSearchResult) reflects real body content only, not appended title/path metadata')
}

console.log(`\n${passed} checks passed.`)
