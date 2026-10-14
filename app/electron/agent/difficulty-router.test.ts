// Run: jiti electron/agent/difficulty-router.test.ts
import assert from 'node:assert/strict'
import {
  classifyTaskTier,
  classifyBatch,
  modelsForTier,
  DEFAULT_TIER_POLICY,
  type TierPolicy,
} from './difficulty-router.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ok ${name}`) }

// --- baseline by job class -------------------------------------------------
{
  assert.equal(classifyTaskTier({ jobClass: 'ack' }).tier, 'cheap')
  assert.equal(classifyTaskTier({ jobClass: 'scoring' }).tier, 'cheap')
  assert.equal(classifyTaskTier({ jobClass: 'compaction' }).tier, 'cheap')
  assert.equal(classifyTaskTier({ jobClass: 'plan' }).tier, 'frontier')
  assert.equal(classifyTaskTier({ jobClass: 'review' }).tier, 'frontier')
  assert.equal(classifyTaskTier({ jobClass: 'edit' }).tier, 'standard')
  ok('job-class baselines map to expected tiers')
}

// --- unknown job class falls back to default baseline ----------------------
{
  const d = classifyTaskTier({ jobClass: 'unknown' })
  assert.equal(d.baseline, 'standard')
  assert.equal(d.tier, 'standard')
  ok('unknown job class uses default baseline')
}

// --- hard text cues raise a standard job to frontier -----------------------
{
  const d = classifyTaskTier({ jobClass: 'edit', text: 'Refactor the auth architecture across multiple files and prove correctness' })
  assert.equal(d.tier, 'frontier')
  assert.ok(d.reasons.some((r) => r.signal === 'text-hard'))
  ok('hard cues escalate an edit job to frontier')
}

// --- easy text cues lower a chat job to cheap ------------------------------
{
  const d = classifyTaskTier({ jobClass: 'chat', text: 'ok thanks, got it' })
  assert.equal(d.tier, 'cheap')
  assert.ok(d.reasons.some((r) => r.signal === 'text-easy' || r.signal === 'text-short'))
  ok('trivial cues drop a chat job to cheap')
}

// --- very short input drops tier (no hard cues) ----------------------------
{
  const d = classifyTaskTier({ jobClass: 'edit', text: 'yes' })
  assert.equal(d.tier, 'cheap')
  ok('very short input drops to cheap')
}

// --- large context raises difficulty ---------------------------------------
{
  const low = classifyTaskTier({ jobClass: 'summarize', approxInputTokens: 1000 })
  const high = classifyTaskTier({ jobClass: 'summarize', approxInputTokens: 50_000 })
  assert.equal(low.tier, 'cheap')
  assert.ok(high.difficulty > low.difficulty)
  assert.ok(high.reasons.some((r) => r.signal === 'large-context'))
  ok('large context raises difficulty above small context')
}

// --- many tool calls + multi-file raise difficulty -------------------------
{
  const d = classifyTaskTier({ jobClass: 'build', expectedToolCalls: 12, filesToEdit: 5 })
  assert.equal(d.tier, 'frontier')
  assert.ok(d.reasons.some((r) => r.signal === 'many-tools'))
  assert.ok(d.reasons.some((r) => r.signal === 'multi-file'))
  ok('many tools + multi-file escalate a build to frontier')
}

// --- retry escalation raises difficulty ------------------------------------
{
  const base = classifyTaskTier({ jobClass: 'edit', text: 'fix the null check' })
  const retry = classifyTaskTier({ jobClass: 'edit', text: 'fix the null check', retryEscalation: true })
  assert.ok(retry.difficulty > base.difficulty)
  assert.ok(retry.reasons.some((r) => r.signal === 'retry'))
  ok('retry escalation raises difficulty')
}

// --- never-downgrade keeps plan at frontier despite easy cues --------------
{
  const d = classifyTaskTier({ jobClass: 'plan', text: 'ok thanks, quick one-liner please' })
  assert.equal(d.tier, 'frontier')
  assert.ok(d.reasons.some((r) => r.signal === 'never-downgrade' || r.signal === 'job-class'))
  ok('never-downgrade pins plan to frontier baseline')
}

// --- policy maxTier caps a frontier decision -------------------------------
{
  const budgetPolicy: TierPolicy = { ...DEFAULT_TIER_POLICY, maxTier: 'standard', neverDowngradeBaseline: [] }
  const d = classifyTaskTier({ jobClass: 'plan', text: 'architect a full rewrite' }, budgetPolicy)
  assert.equal(d.tier, 'standard')
  assert.ok(d.reasons.some((r) => r.signal === 'policy-max'))
  ok('policy maxTier caps frontier down to standard')
}

// --- policy minTier lifts a cheap decision ---------------------------------
{
  const qualityPolicy: TierPolicy = { ...DEFAULT_TIER_POLICY, minTier: 'standard' }
  const d = classifyTaskTier({ jobClass: 'ack', text: 'ok' }, qualityPolicy)
  assert.equal(d.tier, 'standard')
  assert.ok(d.reasons.some((r) => r.signal === 'policy-min'))
  ok('policy minTier lifts cheap up to standard')
}

// --- per-task forceMinTier / forceMaxTier ----------------------------------
{
  const forcedUp = classifyTaskTier({ jobClass: 'ack', text: 'ok', forceMinTier: 'frontier' })
  assert.equal(forcedUp.tier, 'frontier')
  const forcedDown = classifyTaskTier({ jobClass: 'plan', text: 'architect a rewrite', forceMaxTier: 'cheap', ...{} })
  // never-downgrade for plan still applies before force-max? force-max applied after -> cheap wins as ceiling
  assert.equal(forcedDown.tier, 'cheap')
  ok('per-task force floor/ceiling override tier')
}

// --- determinism -----------------------------------------------------------
{
  const a = classifyTaskTier({ jobClass: 'debug', text: 'root cause the race condition' })
  const b = classifyTaskTier({ jobClass: 'debug', text: 'root cause the race condition' })
  assert.deepEqual(a, b)
  ok('classification is deterministic')
}

// --- difficulty stays within [0,1] -----------------------------------------
{
  const d = classifyTaskTier({
    jobClass: 'plan',
    text: 'complex tricky subtle security refactor migration rewrite prove invariant concurrency',
    approxInputTokens: 200_000,
    expectedToolCalls: 40,
    filesToEdit: 20,
    retryEscalation: true,
  })
  assert.ok(d.difficulty >= 0 && d.difficulty <= 1)
  assert.equal(d.tier, 'frontier')
  ok('difficulty clamps within [0,1] under extreme signals')
}

// --- modelsForTier ordering + fallback -------------------------------------
{
  const tierModels = {
    cheap: ['haiku', 'o4-mini'],
    standard: ['sonnet'],
    frontier: ['opus', 'gpt-5.5'],
  }
  assert.deepEqual(modelsForTier({ tier: 'frontier' }, tierModels), ['opus', 'gpt-5.5', 'sonnet', 'haiku', 'o4-mini'])
  assert.deepEqual(modelsForTier({ tier: 'cheap' }, tierModels), ['haiku', 'o4-mini', 'sonnet', 'opus', 'gpt-5.5'])
  // empty tier falls back to adjacent tiers
  assert.deepEqual(modelsForTier({ tier: 'standard' }, { frontier: ['opus'], cheap: ['haiku'] }), ['opus', 'haiku'])
  // dedupes across tiers
  assert.deepEqual(modelsForTier({ tier: 'cheap' }, { cheap: ['x'], standard: ['x', 'y'] }), ['x', 'y'])
  ok('modelsForTier orders by tier with adjacent fallback and dedupe')
}

// --- classifyBatch ---------------------------------------------------------
{
  const out = classifyBatch([
    { jobClass: 'ack', text: 'thanks' },
    { jobClass: 'plan', text: 'design the system' },
  ])
  assert.equal(out.length, 2)
  assert.equal(out[0].tier, 'cheap')
  assert.equal(out[1].tier, 'frontier')
  ok('classifyBatch classifies each signal')
}

console.log(`\n${passed} checks passed`)
