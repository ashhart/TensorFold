import assert from 'node:assert/strict'
import {
  approvalsShouldStick,
  computeCommandStatusSummary,
  disclosureDefaultOpen,
  limitMissionRows,
  liveJobIsActive,
  summaryDefaultExpanded,
  type Tone,
} from './model'

// --- computeCommandStatusSummary: derives a single glanceable status from metric tones ---
{
  const summary = computeCommandStatusSummary({
    attentionCount: 0,
    attentionMaxTone: 'good',
    approvalsCount: 0,
    liveJobsTone: 'good',
    connectorTone: 'good',
    jobsStatus: { queued: 2, stalled: 0, failed: 0, deadLetter: 0 },
    missionCount: 1,
  })
  assert.equal(summary.maxTone, 'good', 'all-good metrics resolve to a good max tone')
  assert.equal(summary.attentionCount, 0)
  assert.equal(summary.approvalsCount, 0)
  assert.equal(summary.missionCount, 1)
  assert.equal(summary.jobsStatus.queued, 2)
  assert.equal(summaryDefaultExpanded(summary.maxTone), false, 'good tone defaults collapsed')
}

{
  const summary = computeCommandStatusSummary({
    attentionCount: 3,
    attentionMaxTone: 'critical',
    approvalsCount: 2,
    liveJobsTone: 'high',
    connectorTone: 'good',
    jobsStatus: { queued: 0, stalled: 1, failed: 0, deadLetter: 1 },
    missionCount: 0,
  })
  assert.equal(summary.maxTone, 'critical', 'a critical signal wins the max tone')
  assert.equal(summaryDefaultExpanded(summary.maxTone), true, 'critical tone defaults expanded')
}

{
  // 'high' is the worst tone present -> expanded by default
  const summary = computeCommandStatusSummary({
    attentionCount: 1,
    attentionMaxTone: 'normal',
    approvalsCount: 1,
    liveJobsTone: 'high',
    connectorTone: 'normal',
    jobsStatus: { queued: 0, stalled: 1, failed: 0, deadLetter: 0 },
    missionCount: 1,
  })
  assert.equal(summary.maxTone, 'high')
  assert.equal(summaryDefaultExpanded(summary.maxTone), true, 'high tone defaults expanded')
}

{
  const summary = computeCommandStatusSummary({
    attentionCount: 0,
    attentionMaxTone: 'good',
    approvalsCount: 0,
    liveJobsTone: 'good',
    connectorTone: 'good',
    readinessTone: 'high',
    jobsStatus: { queued: 0, stalled: 0, failed: 0, deadLetter: 0 },
    missionCount: 0,
  })
  assert.equal(summary.maxTone, 'high', 'readiness issues raise the command centre summary')
}

// --- summaryDefaultExpanded: only critical/high force an open dashboard ---
{
  const cases: Array<[Tone, boolean]> = [
    ['critical', true],
    ['high', true],
    ['normal', false],
    ['low', false],
    ['good', false],
  ]
  for (const [tone, expected] of cases) {
    assert.equal(summaryDefaultExpanded(tone), expected, `summaryDefaultExpanded(${tone}) === ${expected}`)
  }
}

// --- disclosureDefaultOpen: collapses quiet sections, auto-expands busy / critical ones ---
{
  // below threshold and quiet -> closed
  assert.equal(disclosureDefaultOpen(2, 3), false, 'closed when count <= threshold')
  // strictly above threshold -> open
  assert.equal(disclosureDefaultOpen(5, 3), true, 'open when count > threshold')
  // at threshold -> still closed (strictly greater)
  assert.equal(disclosureDefaultOpen(3, 3), false, 'closed when count === threshold')
  // a critical item forces open even when under threshold
  assert.equal(disclosureDefaultOpen(1, 3, 'critical'), true, 'critical forces open under threshold')
  // a non-critical tone under threshold stays closed
  assert.equal(disclosureDefaultOpen(1, 3, 'high'), false, 'non-critical tone does not force open')
  // empty section never opens
  assert.equal(disclosureDefaultOpen(0, 0), false, 'zero items stays closed even at threshold 0')
}

// --- limitMissionRows: show top N, signal overflow with "view all" ---
{
  const missions = [1, 2, 3, 4, 5].map((n) => ({ id: String(n) }))
  const limited = limitMissionRows(missions, 3)
  assert.equal(limited.visible.length, 3, 'caps the visible list to the limit')
  assert.equal(limited.visible[0].id, '1', 'preserves order')
  assert.equal(limited.hasMore, true, 'flags overflow when total exceeds limit')
  assert.equal(limited.total, 5, 'reports the full total')
  assert.equal(limited.hidden, 2, 'reports how many are hidden')
}

{
  const missions = [{ id: 'a' }, { id: 'b' }]
  const limited = limitMissionRows(missions, 3)
  assert.equal(limited.visible.length, 2, 'shows all when under the limit')
  assert.equal(limited.hasMore, false, 'no overflow when total <= limit')
  assert.equal(limited.hidden, 0)
}

{
  const limited = limitMissionRows([], 3)
  assert.equal(limited.visible.length, 0)
  assert.equal(limited.hasMore, false)
  assert.equal(limited.total, 0)
}

// --- approvalsShouldStick: keep the approvals panel pinned only when something is waiting ---
{
  assert.equal(approvalsShouldStick(2), true, 'sticky when approvals are pending')
  assert.equal(approvalsShouldStick(0), false, 'not sticky when the queue is clear')
}

// --- liveJobIsActive: the Live Jobs panel must not pulse / say "Active" for terminal or stuck jobs (LOW #30) ---
{
  assert.equal(liveJobIsActive('running'), true, 'running is genuinely active')
  assert.equal(liveJobIsActive('queued'), true, 'queued is genuinely active')
  assert.equal(liveJobIsActive('stalled'), true, 'stalled is still in-flight, not terminal')
  assert.equal(liveJobIsActive('waiting approval'), true, 'waiting approval is still in-flight, not terminal')
  assert.equal(liveJobIsActive('failed'), false, 'a failed durable job is terminal, not active')
  assert.equal(liveJobIsActive('dead letter'), false, 'a dead-lettered durable job is terminal, not active')
  assert.equal(liveJobIsActive('paused'), false, 'a paused durable job is not making progress')
}

console.log('command-centre layout/declutter tests passed')
