// Run: jiti src/lib/daily-digest.test.ts
import assert from 'node:assert/strict'
import { formatDailyDigest } from './daily-digest.ts'
import type { AwayBriefItem } from './away-brief.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('daily-digest tests:')

const item = (over: Partial<AwayBriefItem>): AwayBriefItem => ({
  id: over.id ?? 'i', kind: over.kind ?? 'connector', subjectId: over.subjectId ?? 's',
  title: over.title ?? 'T', detail: over.detail ?? 'D', tone: over.tone ?? 'normal',
  createdAt: over.createdAt ?? 0, sourceIds: over.sourceIds ?? [],
})

// ---- empty → a clear "nothing to report" briefing, still headed with the date ----
{
  const md = formatDailyDigest([], { dateLabel: 'Wed 1 Jul' })
  assert.match(md, /Daily brief — Wed 1 Jul/)
  assert.match(md, /No activity to report/i)
  ok('empty digest still renders a dated header + "no activity"')
}

// ---- groups by tone tier: critical/high = attention, normal = updates, low/good = FYI ----
{
  const items = [
    item({ id: 'a', tone: 'critical', kind: 'mission', title: 'Prod deploy stalled', detail: 'waiting on approval' }),
    item({ id: 'b', tone: 'high', kind: 'pr_loop', title: 'PR needs review', detail: '3 comments' }),
    item({ id: 'c', tone: 'normal', kind: 'room', title: 'Room finished', detail: 'built feature X' }),
    item({ id: 'd', tone: 'low', kind: 'connector', title: 'New Slack thread', detail: 'fyi' }),
    item({ id: 'e', tone: 'good', kind: 'automation', title: 'Nightly passed', detail: 'all green' }),
  ]
  const md = formatDailyDigest(items, { dateLabel: 'Wed 1 Jul' })
  const attn = md.indexOf('Needs attention')
  const upd = md.indexOf('Updates')
  const fyi = md.indexOf('FYI')
  assert.ok(attn > -1 && upd > -1 && fyi > -1, 'all three tiers present')
  assert.ok(attn < upd && upd < fyi, 'attention before updates before FYI')
  // critical + high land under attention; normal under updates; low + good under FYI
  assert.ok(md.indexOf('Prod deploy stalled') > attn && md.indexOf('Prod deploy stalled') < upd)
  assert.ok(md.indexOf('Room finished') > upd && md.indexOf('Room finished') < fyi)
  assert.ok(md.indexOf('Nightly passed') > fyi)
  // item lines carry the kind + title + detail
  assert.match(md, /\[mission\].*Prod deploy stalled.*waiting on approval/)
  ok('groups tiers in order, tags kind, includes title + detail')
}

// ---- headline counts: N need you · M updates ----
{
  const items = [item({ tone: 'critical', id: '1' }), item({ tone: 'high', id: '2' }), item({ tone: 'normal', id: '3' })]
  const md = formatDailyDigest(items, { dateLabel: 'X' })
  assert.match(md, /2 need you/, 'critical+high counted as "need you"')
  ok('headline summarizes attention vs other counts')
}

// ---- within a tier, newest first ----
{
  const items = [
    item({ id: 'old', tone: 'normal', title: 'OLD', createdAt: 100 }),
    item({ id: 'new', tone: 'normal', title: 'NEW', createdAt: 999 }),
  ]
  const md = formatDailyDigest(items, { dateLabel: 'X' })
  assert.ok(md.indexOf('NEW') < md.indexOf('OLD'), 'newer item listed first within a tier')
  ok('items sorted newest-first within a tier')
}

// ---- maxItems caps the output and notes the remainder ----
{
  const items = Array.from({ length: 10 }, (_, i) => item({ id: `x${i}`, tone: 'normal', title: `T${i}`, createdAt: i }))
  const md = formatDailyDigest(items, { dateLabel: 'X', maxItems: 3 })
  assert.match(md, /\+7 more/, 'notes how many were truncated')
  ok('maxItems caps items and notes the remainder')
}

// ---- custom title honored ----
{
  const md = formatDailyDigest([], { dateLabel: 'X', title: 'Morning rundown' })
  assert.match(md, /Morning rundown — X/)
  ok('custom title honored')
}

console.log(`\n${passed} checks passed.`)
