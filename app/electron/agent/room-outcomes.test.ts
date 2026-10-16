// Run: jiti electron/agent/room-outcomes.test.ts
//
// Room outcomes (VST-019): every mode's outcome contract (stop condition + expected
// artifact + seat/moderator behavior + budget), the FINAL-text gates per mode, and
// parity with the renderer's mode picker options.
import assert from 'node:assert/strict'
import {
  normalizeRoomMode,
  ROOM_MODES,
  ROOM_OUTCOME_PROFILES,
  roomModePromptBlock,
  roomModeDefaults,
  roomOutcomeGate,
  roomOutcomeProfile,
} from './room-outcomes'
import { ROOM_MODE_OPTIONS } from '../../src/lib/room-modes'
import { ROOM_DELIBERATION_PROFILES } from './room-collaboration'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('room-outcomes tests:')

{
  // Every mode has the full contract, its budget maps to a real deliberation profile,
  // and the prompt block carries all five pieces.
  const efforts = new Set(ROOM_DELIBERATION_PROFILES.map((profile) => profile.id))
  for (const mode of ROOM_MODES) {
    const profile = ROOM_OUTCOME_PROFILES[mode]
    assert.equal(profile.id, mode)
    for (const field of [profile.stopCondition, profile.expectedArtifact.description, profile.seatBehavior, profile.moderatorBehavior] as string[]) {
      assert.ok(field.length > 20, `${mode}: contract fields are substantive`)
    }
    assert.ok(efforts.has(profile.budget), `${mode}: budget '${profile.budget}' is a real deliberation profile`)
    const block = roomModePromptBlock(profile)
    assert.match(block, /Stop condition:/)
    assert.match(block, /Expected final artifact:/)
    assert.match(block, /Seat behavior:/)
    assert.match(block, /Moderator behavior:/)
  }
  ok('all six modes: full contract, real budget mapping, five-part prompt block')
}

{
  assert.deepEqual(roomModeDefaults(), { mode: 'implement', deliberationEffort: 'deep' })
  for (const mode of ROOM_MODES) assert.equal(roomModeDefaults(mode).deliberationEffort, ROOM_OUTCOME_PROFILES[mode].budget)
  ok('new rooms default to an implement outcome and mode changes inherit the real mode budget')
}

{
  // EACH ROOM MODE (spec test): its gate passes a conforming final and names what a
  // non-conforming final is missing.
  const cases: Array<{ mode: (typeof ROOM_MODES)[number]; good: string; bad: string; missing: RegExp }> = [
    { mode: 'plan', good: 'Plan:\n1. Extract the parser\n2. Add tests\n3. Ship behind a flag', bad: 'We should probably refactor the parser at some point.', missing: /ordered plan/ },
    { mode: 'implement', good: 'Implemented the cache in src/cache.ts and updated the loader. Tests: npm test passed (34/34).', bad: 'The approach we would take is a Map cache.', missing: /what actually changed|verification evidence/ },
    { mode: 'review', good: 'Changes requested: the retry loop swallows errors (loop.ts:88) and the new flag is undocumented.', bad: 'Overall the code looks reasonable to me.', missing: /explicit verdict/ },
    { mode: 'debug', good: 'Root cause: the watcher double-fires because chokidar emits rename+change; fix by debouncing 50ms in watch.ts.', bad: 'The tests are flaky sometimes on CI.', missing: /root cause/ },
    { mode: 'incident', good: 'Mitigated by rolling back v2.1. Root cause: the migration locked the jobs table. Follow-ups: add lock timeout, stage migrations.', bad: 'The outage seems over now.', missing: /mitigation|root cause|follow-up/ },
    { mode: 'research', good: 'Findings: the SDK retries 3x by default (source: sdk/client.ts). Conclusion: no wrapper retry needed.', bad: 'It probably retries on its own.', missing: /conclusion|sources/ },
  ]
  for (const c of cases) {
    const good = roomOutcomeGate(c.mode, { finalText: c.good })
    assert.equal(good.satisfied, true, `${c.mode}: conforming final passes (${JSON.stringify(good.missing)})`)
    const bad = roomOutcomeGate(c.mode, { finalText: c.bad })
    assert.equal(bad.satisfied, false, `${c.mode}: empty-gesture final is blocked`)
    assert.ok(bad.missing.some((m) => c.missing.test(m)), `${c.mode}: the missing piece is named (${bad.missing.join('; ')})`)
  }
  ok('each room mode: conforming finals pass, empty gestures are blocked with the gap named')
}

{
  // Legacy rooms: no mode → no profile → no gate. Unknown strings normalize to null.
  assert.equal(normalizeRoomMode(undefined), null)
  assert.equal(normalizeRoomMode('huddle'), null)
  assert.equal(normalizeRoomMode('review'), 'review')
  assert.equal(roomOutcomeProfile(null), null)
  assert.equal(roomOutcomeProfile('debug')?.budget, 'deep')
  ok('legacy rooms keep no-gate behavior; unknown modes normalize to null')
}

{
  // Renderer parity: the mode picker offers exactly the six modes with matching labels.
  assert.deepEqual(ROOM_MODE_OPTIONS.map((option) => option.id), ROOM_MODES)
  for (const option of ROOM_MODE_OPTIONS) {
    assert.equal(option.label, ROOM_OUTCOME_PROFILES[option.id].label, `${option.id}: picker label matches the profile`)
    assert.ok(option.detail.length > 20)
  }
  ok('renderer mode picker matches the outcome profiles (ids + labels)')
}

console.log(`  ${passed} passed`)
