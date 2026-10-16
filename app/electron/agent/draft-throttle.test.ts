// Run: jiti electron/agent/draft-throttle.test.ts
import assert from 'node:assert/strict'
import { canEmitDraftEdit, makeDraftEditThrottle, DEFAULT_DRAFT_EDIT_INTERVAL_MS, DRAFT_EDIT_INTERVALS_MS, draftEditIntervalForPlatform } from './draft-throttle.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('draft-throttle tests:')

// ---- the default interval keeps a safe margin under Telegram's edit rate ----
{
  assert.equal(DEFAULT_DRAFT_EDIT_INTERVAL_MS, 1500, 'default spacing is 1.5s')
  assert.ok(DEFAULT_DRAFT_EDIT_INTERVAL_MS >= 1500, 'spacing is at least the task-mandated 1.5s')
  ok('default interval is 1.5s')
}

// ---- pure core: first edit always allowed; later edits gated by the interval ----
{
  // No prior emit -> always allowed, whatever the clock.
  assert.equal(canEmitDraftEdit(null, 0, 1500), true, 'first edit allowed with no prior')
  assert.equal(canEmitDraftEdit(null, 999999, 1500), true)
  // Within the interval -> blocked. At/after the interval -> allowed.
  assert.equal(canEmitDraftEdit(1000, 1000, 1500), false, 'zero elapsed is blocked')
  assert.equal(canEmitDraftEdit(1000, 2499, 1500), false, 'just under the interval is blocked')
  assert.equal(canEmitDraftEdit(1000, 2500, 1500), true, 'exactly the interval is allowed')
  assert.equal(canEmitDraftEdit(1000, 5000, 1500), true, 'well past the interval is allowed')
  ok('canEmitDraftEdit: first allowed, then spaced by the interval')
}

// ---- pure core: non-positive interval disables throttling; backwards clock is conservative ----
{
  assert.equal(canEmitDraftEdit(1000, 1000, 0), true, 'interval 0 disables throttling')
  assert.equal(canEmitDraftEdit(1000, 1000, -5), true, 'negative interval disables throttling')
  // A clock that appears to go backwards must not be read as "interval elapsed".
  assert.equal(canEmitDraftEdit(5000, 4000, 1500), false, 'backwards clock is treated as not-yet')
  ok('canEmitDraftEdit: interval<=0 is unthrottled; backwards clock blocks')
}

// ---- stateful wrapper: emits, then suppresses until the interval, then emits again ----
{
  const t = makeDraftEditThrottle(1500)
  assert.equal(t.shouldEmit(0), true, 'first frame fires immediately')
  assert.equal(t.shouldEmit(100), false, 'a frame 100ms later is dropped')
  assert.equal(t.shouldEmit(1499), false, 'still dropped just under the interval')
  assert.equal(t.shouldEmit(1500), true, 'fires again once the interval elapses')
  assert.equal(t.shouldEmit(1600), false, 'and the clock restarts from the last emit')
  assert.equal(t.shouldEmit(3000), true, 'fires after another full interval')
  ok('makeDraftEditThrottle: spaces emitted edits by the interval, restarting each time')
}

// ---- stateful wrapper: a dropped frame does NOT advance the clock (only an emitted one does) ----
{
  const t = makeDraftEditThrottle(1000)
  assert.equal(t.shouldEmit(0), true)
  // Several rapid frames inside the window are all dropped and must not push the next-allowed time
  // forward — otherwise a steady fast stream could starve edits forever.
  assert.equal(t.shouldEmit(200), false)
  assert.equal(t.shouldEmit(400), false)
  assert.equal(t.shouldEmit(600), false)
  assert.equal(t.shouldEmit(1000), true, 'allowed exactly one interval after the LAST EMIT, not the last drop')
  ok('makeDraftEditThrottle: dropped frames do not move the gate')
}

// ---- stateful wrapper: markFinal resets the clock (a reused throttle starts from the final) ----
{
  const t = makeDraftEditThrottle(1000)
  assert.equal(t.shouldEmit(0), true)
  t.markFinal(5000) // the final commit happened at t=5000 (bypasses shouldEmit)
  assert.equal(t.shouldEmit(5500), false, 'an edit 500ms after the final is still throttled')
  assert.equal(t.shouldEmit(6000), true, 'and allowed a full interval after the final')
  ok('makeDraftEditThrottle: markFinal advances the gate to the final commit')
}

// ---- stateful wrapper: default interval is the 1.5s constant ----
{
  const t = makeDraftEditThrottle()
  assert.equal(t.shouldEmit(0), true)
  assert.equal(t.shouldEmit(1499), false, 'default-built throttle uses the 1.5s spacing')
  assert.equal(t.shouldEmit(1500), true)
  ok('makeDraftEditThrottle: defaults to the 1.5s interval')
}

// ---- per-platform edit interval: each editing channel respects its OWN rate limit ----
{
  // The three draft-streaming channels each have an explicit spacing tuned to their edit ceiling.
  assert.equal(draftEditIntervalForPlatform('telegram'), 1500, 'telegram = 1.5s under editMessageText')
  assert.equal(draftEditIntervalForPlatform('slack'), 1200, 'slack = 1.2s under chat.update Tier-3')
  assert.equal(draftEditIntervalForPlatform('discord'), 1000, 'discord = 1s under the channel edit bucket')
  // The table and the lookup agree (no drift between the constant map and the accessor).
  assert.equal(draftEditIntervalForPlatform('slack'), DRAFT_EDIT_INTERVALS_MS.slack)
  assert.equal(draftEditIntervalForPlatform('discord'), DRAFT_EDIT_INTERVALS_MS.discord)
  assert.equal(draftEditIntervalForPlatform('telegram'), DRAFT_EDIT_INTERVALS_MS.telegram)
  ok('draftEditIntervalForPlatform: per-platform spacing matches the limit table')
}

// ---- unknown / missing platform falls back to the safe default ----
{
  assert.equal(draftEditIntervalForPlatform('imessage'), DEFAULT_DRAFT_EDIT_INTERVAL_MS, 'a non-editing platform uses the default')
  assert.equal(draftEditIntervalForPlatform('whatsapp'), DEFAULT_DRAFT_EDIT_INTERVAL_MS)
  assert.equal(draftEditIntervalForPlatform(undefined), DEFAULT_DRAFT_EDIT_INTERVAL_MS, 'undefined falls back')
  assert.equal(draftEditIntervalForPlatform(null), DEFAULT_DRAFT_EDIT_INTERVAL_MS, 'null falls back')
  assert.equal(draftEditIntervalForPlatform(''), DEFAULT_DRAFT_EDIT_INTERVAL_MS, 'empty string falls back')
  ok('draftEditIntervalForPlatform: unknown/absent platform -> default 1.5s')
}

// ---- a per-platform throttle actually spaces edits at the SELECTED interval ----
{
  // Discord's 1s spacing: a frame at 999ms is dropped, at 1000ms it fires — proving the selected
  // interval (not the 1.5s default) gates the channel's edits.
  const discord = makeDraftEditThrottle(draftEditIntervalForPlatform('discord'))
  assert.equal(discord.shouldEmit(0), true, 'first discord frame fires')
  assert.equal(discord.shouldEmit(999), false, 'dropped just under discord 1s')
  assert.equal(discord.shouldEmit(1000), true, 'fires at discord 1s')
  // Slack's 1.2s spacing differs: at 1000ms (which Discord would allow) Slack is still throttled.
  const slack = makeDraftEditThrottle(draftEditIntervalForPlatform('slack'))
  assert.equal(slack.shouldEmit(0), true)
  assert.equal(slack.shouldEmit(1000), false, 'slack still throttled at 1s (its limit is 1.2s)')
  assert.equal(slack.shouldEmit(1200), true, 'fires at slack 1.2s')
  ok('makeDraftEditThrottle honours the per-platform interval (discord 1s vs slack 1.2s)')
}

console.log(`\n${passed} passed`)
