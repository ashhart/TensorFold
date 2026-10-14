import assert from 'node:assert/strict'
import {
  MID_RUN_NUDGE_MUTATION_THRESHOLD,
  STOP_REMINDER_MAX_PER_RUN,
  midRunTodoNudge,
  normalizePlanSteps,
  openPlanSteps,
  planProgress,
  renderPlanTree,
  stopTodoReminder,
} from './todo-tracker'

console.log('todo tracker tests:')

// -- normalizePlanSteps: OMP statuses, blockers, subtasks, legacy shape --
{
  const steps = normalizePlanSteps([
    { text: 'Ship the fix', status: 'in_progress' },
    { text: 'Add fixture for regression', status: 'pending', subtask: true },
    { text: 'Unblock CI', status: 'blocked', blocker: 'runner capacity' },
    { text: 'Old rewrite approach', status: 'dropped' },
    { text: 'Reproduce the crash', status: 'done' },
  ])
  assert.equal(steps.length, 5)
  assert.deepEqual(steps[1], { id: 's1', text: 'Add fixture for regression', done: false, status: 'pending', depth: 1 })
  assert.equal(steps[2].blocker, 'runner capacity')
  assert.equal(steps[3].status, 'dropped')
  assert.equal(steps[4].done, true)
  // Legacy Codex shape keeps working, including done-via-boolean.
  const legacy = normalizePlanSteps([{ text: 'a', status: 'pending' }, { text: 'b', done: true }, 'bare string'])
  assert.deepEqual(legacy.map((s) => [s.text, s.done, s.status]), [['a', false, 'pending'], ['b', true, 'done'], ['bare string', false, 'pending']])
  assert.deepEqual(normalizePlanSteps('nope'), [])
  console.log('  ✓ normalizePlanSteps maps statuses, blockers, subtasks, and the legacy shape')
}

// -- planProgress: half credit for in_progress, blocked counts, fraction --
{
  const progress = planProgress(normalizePlanSteps([
    { text: 'one', status: 'done' },
    { text: 'two', status: 'in_progress' },
    { text: 'three', status: 'blocked' },
    { text: 'four', status: 'pending' },
  ]))
  assert.equal(progress.done, 1)
  assert.equal(progress.open, 3)
  assert.equal(progress.blocked, 1)
  assert.equal(progress.inProgress, 1)
  assert.equal(progress.fraction, 1.5 / 4)
  assert.equal(planProgress([]).fraction, 0)
  assert.equal(openPlanSteps(normalizePlanSteps([{ text: 'x', status: 'dropped' }, { text: 'y', status: 'done' }])).length, 0)
  console.log('  ✓ planProgress credits done + half in_progress and reports blocked')
}

// -- renderPlanTree: status glyphs and subtask indent --
{
  const tree = renderPlanTree(normalizePlanSteps([
    { text: 'top', status: 'in_progress' },
    { text: 'child', status: 'pending', subtask: true },
    { text: 'stuck', status: 'blocked', blocker: 'upstream api' },
    { text: 'gone', status: 'dropped' },
    { text: 'finished', status: 'done' },
  ]))
  assert.equal(tree, '▸ top\n  ○ child\n■ stuck — upstream api\n× gone\n✓ finished')
  console.log('  ✓ renderPlanTree renders glyphs, blockers, and subtask indents')
}

// -- midRunTodoNudge: threshold, per-run budget, open-item requirement --
{
  const plan = normalizePlanSteps([{ text: 'a' }, { text: 'b', status: 'in_progress' }])
  assert.equal(midRunTodoNudge(MID_RUN_NUDGE_MUTATION_THRESHOLD - 1, plan, 0), null)
  assert.match(midRunTodoNudge(MID_RUN_NUDGE_MUTATION_THRESHOLD, plan, 0)!, /2 plan items still open/)
  assert.equal(midRunTodoNudge(MID_RUN_NUDGE_MUTATION_THRESHOLD, plan, 2), null)
  const allDone = normalizePlanSteps([{ text: 'a', status: 'done' }])
  assert.equal(midRunTodoNudge(99, allDone, 0), null)
  console.log('  ✓ midRunTodoNudge fires at the mutation threshold, twice per run at most')
}

// -- stopTodoReminder: budget, question skip, message shape --
{
  const plan = normalizePlanSteps([
    { text: 'ship it', status: 'in_progress' },
    { text: 'nested detail', status: 'pending', subtask: true },
    { text: 'blocked step', status: 'blocked', blocker: 'review' },
  ])
  const reminder = stopTodoReminder(plan, 0, 'I am done with the main work now.')
  assert.ok(reminder)
  assert.equal(reminder.open.length, 3)
  assert.match(reminder.message, /You stopped with 3 open plan items/)
  assert.match(reminder.message, /\(Reminder 1\/3\)/)
  assert.equal(stopTodoReminder(plan, STOP_REMINDER_MAX_PER_RUN, 'final'), null)
  assert.equal(stopTodoReminder(normalizePlanSteps([{ text: 'done', status: 'done' }]), 0, 'final'), null)
  // The assistant ending on a question to the user means it is their turn — no reminder.
  assert.equal(stopTodoReminder(plan, 0, 'I need one thing first: should I delete the old table?'), null)
  assert.equal(stopTodoReminder(plan, 0, '¿Qué versión prefieres?'), null)
  assert.ok(stopTodoReminder(plan, 0, 'Everything is finished and verified. The fix is in place.'), 'plain statements still remind')
  console.log('  ✓ stopTodoReminder budgets attempts and skips user-facing questions')
}

console.log('todo tracker tests passed')
