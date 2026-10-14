// Run: jiti src/lib/usage-windows.test.ts
import assert from 'node:assert/strict'
import {
  usageWindowView,
  usageWindowViews,
  orderUsageWindows,
  worstUsageTone,
  formatUsageEta,
  type UsageWindow,
} from './usage-windows'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('usage-windows tests:')

const now = 1_000_000_000_000
const HOUR = 60 * 60_000

function window(extra: Partial<UsageWindow> = {}): UsageWindow {
  return { id: 'w', label: 'Session (5h)', kind: 'session', used: 0, limit: 100, resetsAt: now + HOUR, startedAt: now - HOUR, ...extra }
}

// --- pace: even spend at the halfway point is on-track ---
{
  // Window is 2h long (started 1h ago, resets in 1h) → 50% elapsed. Used 50% → on track.
  const view = usageWindowView(window({ used: 50 }), now)
  assert.equal(view.usedPercent, 50)
  assert.equal(view.expectedPercent, 50, 'expected marker sits at the elapsed fraction')
  assert.equal(view.pace, 'on_track')
  assert.equal(view.tone, 'normal')
  ok('usageWindowView: even spend at the elapsed fraction → on_track')
}

// --- behind: spending faster than the even line ---
{
  const view = usageWindowView(window({ used: 85 }), now) // 85% used at 50% elapsed
  assert.equal(view.pace, 'behind')
  assert.equal(view.tone, 'warn')
  assert.ok(view.paceDetail.includes('ahead of an even pace'))
  ok('usageWindowView: used far above the even line → behind/warn')
}

// --- behind + nearly full escalates to critical ---
{
  const view = usageWindowView(window({ used: 95 }), now)
  assert.equal(view.pace, 'behind')
  assert.equal(view.tone, 'critical', '>=90% used while behind is critical')
  ok('usageWindowView: behind and >=90% used → critical')
}

// --- ahead: spending slower than the even line ---
{
  const view = usageWindowView(window({ used: 20 }), now) // 20% used at 50% elapsed
  assert.equal(view.pace, 'ahead')
  assert.equal(view.tone, 'good')
  assert.equal(view.remaining, 80)
  ok('usageWindowView: used well below the even line → ahead/good')
}

// --- exhausted ---
{
  const view = usageWindowView(window({ used: 100 }), now)
  assert.equal(view.pace, 'exhausted')
  assert.equal(view.tone, 'critical')
  assert.equal(view.usedPercent, 100)
  assert.ok(view.paceDetail.includes('Limit reached'))
  ok('usageWindowView: used >= limit → exhausted/critical')
}

// --- over-limit clamps to 100% not beyond ---
{
  const view = usageWindowView(window({ used: 250 }), now)
  assert.equal(view.usedPercent, 100, 'percent clamps at 100')
  assert.equal(view.remaining, 0, 'remaining floors at 0')
  ok('usageWindowView: over-limit clamps percent and remaining')
}

// --- no pace reference (no startedAt): tone from fullness only ---
{
  const view = usageWindowView(window({ used: 80, startedAt: null }), now)
  assert.equal(view.expectedPercent, null, 'no pace marker without a start time')
  assert.equal(view.pace, 'unknown')
  assert.equal(view.tone, 'warn', '80% used → warn on fullness alone')
  ok('usageWindowView: no start time → pace unknown, tone from fullness')
}

// --- no limit: pure counter ---
{
  const view = usageWindowView(window({ used: 5000, limit: null, resetsAt: null }), now)
  assert.equal(view.usedPercent, null)
  assert.equal(view.remaining, null)
  assert.equal(view.tone, 'muted')
  assert.equal(view.etaText, null)
  ok('usageWindowView: no limit → muted counter, no eta')
}

// --- eta formatting ---
{
  assert.equal(formatUsageEta(now + 30_000, now), 'resets in <1m')
  assert.equal(formatUsageEta(now + 45 * 60_000, now), 'resets in 45m')
  assert.equal(formatUsageEta(now + 3 * HOUR, now), 'resets in 3h')
  assert.equal(formatUsageEta(now + 2 * 24 * HOUR, now), 'resets in 2d')
  assert.equal(formatUsageEta(now - 1000, now), 'resetting')
  assert.equal(formatUsageEta(null, now), null)
  ok('formatUsageEta: minute/hour/day buckets + past + null')
}

// --- ordering + worst-tone rollup ---
{
  const windows: UsageWindow[] = [
    { id: 'model', label: 'Opus', kind: 'model', used: 10, limit: 100, resetsAt: now + HOUR, startedAt: now - HOUR },
    { id: 'weekly', label: 'Weekly', kind: 'weekly', used: 95, limit: 100, resetsAt: now + 5 * HOUR, startedAt: now - 5 * HOUR },
    { id: 'session', label: 'Session', kind: 'session', used: 50, limit: 100, resetsAt: now + HOUR, startedAt: now - HOUR },
  ]
  const ordered = orderUsageWindows(windows).map((w) => w.id)
  assert.deepEqual(ordered, ['session', 'weekly', 'model'], 'session → weekly → model ordering')
  const views = usageWindowViews(windows, now)
  assert.equal(views[0].id, 'session')
  assert.equal(worstUsageTone(views), 'critical', 'the weekly window at 95% while behind drives the worst tone')
  ok('order + worstUsageTone: session-first ordering, worst window drives the badge')
}

console.log(`\n${passed} passed`)
