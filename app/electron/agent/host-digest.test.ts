// Run: jiti electron/agent/host-digest.test.ts
import assert from 'node:assert/strict'
import { buildHostDigest } from './host-digest.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('host-digest tests:')

const NOW = 1_000_000_000_000
const HOUR = 3_600_000
const within = NOW - 2 * HOUR // inside a 24h window
const old = NOW - 48 * HOUR // outside it

// ---- empty → dated header + "no activity" ----
{
  const md = buildHostDigest({ now: NOW, dateLabel: 'Wed 1 Jul' })
  assert.match(md, /Daily brief — Wed 1 Jul/)
  assert.match(md, /No activity to report/i)
  ok('empty digest renders a dated header + "no activity"')
}

// ---- missions needing attention land under "Needs attention" ----
{
  const md = buildHostDigest({
    now: NOW, dateLabel: 'X',
    missions: [
      { status: 'waiting_approval', name: 'Deploy prod', updatedAt: within },
      { status: 'running', name: 'Busy mission', updatedAt: within }, // not attention, not "done"
    ],
  })
  assert.match(md, /Needs attention/)
  assert.ok(md.indexOf('Deploy prod') > md.indexOf('Needs attention'))
  assert.ok(!md.includes('Busy mission'), 'a running mission is neither attention nor a completed update')
  ok('waiting_approval mission surfaces under Needs attention')
}

// ---- failed jobs = attention; completed jobs = updates; window respected ----
{
  const md = buildHostDigest({
    now: NOW, dateLabel: 'X',
    jobs: [
      { kind: 'room', status: 'failed', label: 'Build broke', updatedAt: within },
      { kind: 'room', status: 'completed', label: 'Feature shipped', updatedAt: within },
      { kind: 'room', status: 'completed', label: 'Ancient job', updatedAt: old }, // outside window
    ],
  })
  const attn = md.indexOf('Needs attention')
  const upd = md.indexOf('Updates')
  assert.ok(md.indexOf('Build broke') > attn && md.indexOf('Build broke') < upd, 'failed job under attention')
  assert.ok(md.indexOf('Feature shipped') > upd, 'completed job under updates')
  assert.ok(!md.includes('Ancient job'), 'jobs outside the window are excluded')
  ok('failed→attention, completed→updates, out-of-window excluded')
}

// ---- inbound connector events are summarized as a count in Updates ----
{
  const md = buildHostDigest({
    now: NOW, dateLabel: 'X',
    connectorEvents: [
      { direction: 'inbound', title: 'msg1', createdAt: within },
      { direction: 'inbound', title: 'msg2', createdAt: within },
      { direction: 'outbound', title: 'sent', createdAt: within }, // not counted
      { direction: 'inbound', title: 'old', createdAt: old }, // out of window
    ],
  })
  assert.match(md, /2 new inbound message/, 'counts only in-window inbound events')
  ok('inbound connector events counted in Updates')
}

// ---- pending-approval connector events surface under attention ----
{
  const md = buildHostDigest({
    now: NOW, dateLabel: 'X',
    connectorEvents: [{ direction: 'inbound', requiresApproval: true, approvalStatus: 'pending', title: 'Approve send?', createdAt: within }],
  })
  assert.ok(md.indexOf('Approve send?') > md.indexOf('Needs attention'))
  ok('pending-approval connector event surfaces under Needs attention')
}

// ---- headline counts need-you vs updates ----
{
  const md = buildHostDigest({
    now: NOW, dateLabel: 'X',
    missions: [{ status: 'stalled', name: 'M', updatedAt: within }],
    jobs: [{ kind: 'room', status: 'completed', label: 'J', updatedAt: within }],
  })
  assert.match(md, /1 need you/)
  assert.match(md, /1 update/)
  ok('headline summarizes need-you vs updates')
}

console.log(`\n${passed} checks passed.`)
