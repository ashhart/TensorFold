// Run: npx jiti electron/agent/interrupt-registry.test.ts
import assert from 'node:assert/strict'
import {
  InterruptRegistry,
  InterruptDeniedError,
  type InterruptEvent,
} from './interrupt-registry'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ok ${name}`) }

console.log('interrupt registry tests:')

// Deterministic factory: monotonic ids + monotonic clock, no globals.
function makeRegistry(events?: InterruptEvent[]) {
  let id = 0
  let clock = 100
  return new InterruptRegistry({
    nextId: () => `int_${++id}`,
    now: () => clock++,
    onEvent: events ? (e) => events.push(e) : undefined,
  })
}

// ── park returns a pending promise that stays pending until resolved ──────────
{
  const reg = makeRegistry()
  const { id, promise } = reg.park({ kind: 'approval', tool: 'run_command', label: 'shell', args: { command: 'ls' } })
  assert.equal(id, 'int_1')
  assert.equal(reg.size, 1)
  assert.equal(reg.has('int_1'), true)

  let settled = false
  promise.then(() => { settled = true }, () => { settled = true })
  await Promise.resolve() // give the microtask queue a chance — must remain pending
  assert.equal(settled, false, 'promise must stay pending while parked')
  ok('park parks a pending interrupt')
}

// ── allow (no edit) resolves with the original args, edited=false ─────────────
{
  const reg = makeRegistry()
  const { id, promise } = reg.park({ kind: 'approval', tool: 'run_command', label: 'shell', args: { command: 'ls' } })
  reg.resolve(id, { type: 'allow' })
  const outcome = await promise
  assert.deepEqual(outcome, { type: 'allow', args: { command: 'ls' }, edited: false })
  assert.equal(reg.size, 0, 'resolving removes the entry')
  assert.equal(reg.has(id), false)
  ok('allow keeps original args and clears the entry')
}

// ── allow WITH edited args resolves with the edit, edited=true ────────────────
{
  const reg = makeRegistry()
  const { id, promise } = reg.park({ kind: 'approval', tool: 'run_command', label: 'shell', args: { command: 'rm -rf /' } })
  reg.resolve(id, { type: 'allow', args: { command: 'rm -rf ./tmp' } })
  const outcome = await promise
  assert.equal(outcome.type, 'allow')
  assert.deepEqual((outcome as any).args, { command: 'rm -rf ./tmp' })
  assert.equal((outcome as any).edited, true)
  ok('allow with edited args resumes with the edit (edit-args-then-resume)')
}

// ── deny rejects with InterruptDeniedError carrying id + reason ───────────────
{
  const reg = makeRegistry()
  const { id, promise } = reg.park({ kind: 'approval', tool: 'email_compose', label: 'send' })
  reg.resolve(id, { type: 'deny', reason: 'not this recipient' })
  await assert.rejects(promise, (err: unknown) => {
    assert.ok(err instanceof InterruptDeniedError)
    assert.equal(err.interruptId, id)
    assert.equal(err.reason, 'not this recipient')
    return true
  })
  assert.equal(reg.size, 0)
  ok('deny rejects with InterruptDeniedError(id, reason)')
}

// ── question → answer resolves with the free-form value ───────────────────────
{
  const reg = makeRegistry()
  const { id, promise } = reg.park({ kind: 'question', tool: 'ask_user', label: 'which file?', prompt: 'file?' })
  reg.resolve(id, { type: 'answer', value: 'src/index.ts' })
  const outcome = await promise
  assert.deepEqual(outcome, { type: 'answer', value: 'src/index.ts' })
  ok('question resolves via answer')
}

// ── choice accepts a valid option, rejects an invalid one WITHOUT consuming ───
{
  const reg = makeRegistry()
  const { id, promise } = reg.park({ kind: 'choice', tool: 'pick', label: 'model', options: ['a', 'b'] })
  assert.throws(() => reg.resolve(id, { type: 'answer', value: 'z' }), /not one of \[a, b\]/)
  assert.equal(reg.has(id), true, 'a bad resolve must NOT consume the parked request')
  reg.resolve(id, { type: 'answer', value: 'b' })
  const outcome = await promise
  assert.deepEqual(outcome, { type: 'answer', value: 'b' })
  ok('choice validates options and a bad resolve is retry-able')
}

// ── kind/resolution mismatch is rejected both ways ────────────────────────────
{
  const reg = makeRegistry()
  const approval = reg.park({ kind: 'approval', tool: 't', label: 'l' })
  assert.throws(() => reg.resolve(approval.id, { type: 'answer', value: 1 }), /'allow' or 'deny'/)
  assert.equal(reg.has(approval.id), true)

  const question = reg.park({ kind: 'question', tool: 't', label: 'l' })
  assert.throws(() => reg.resolve(question.id, { type: 'allow' }), /do not take 'allow'/)
  assert.equal(reg.has(question.id), true)
  ok('resolution kind must match request kind')
}

// ── unknown id and double-resolve both throw ──────────────────────────────────
{
  const reg = makeRegistry()
  assert.throws(() => reg.resolve('nope', { type: 'allow' }), /No parked interrupt/)
  const { id } = reg.park({ kind: 'approval', tool: 't', label: 'l' })
  reg.resolve(id, { type: 'allow' })
  assert.throws(() => reg.resolve(id, { type: 'allow' }), /No parked interrupt/)
  ok('unknown id and double-resolve throw')
}

// ── invalid requests are rejected at park time ────────────────────────────────
{
  const reg = makeRegistry()
  assert.throws(() => reg.park({ kind: 'bogus' as any, tool: 't', label: 'l' }), /Unknown interrupt kind/)
  assert.throws(() => reg.park({ kind: 'approval', tool: '', label: 'l' }), /needs a tool/)
  assert.throws(() => reg.park({ kind: 'choice', tool: 't', label: 'l' }), /non-empty options/)
  ok('park validates the request')
}

// ── list()/peek() reflect parked state deterministically (oldest first) ───────
{
  const reg = makeRegistry()
  const a = reg.park({ kind: 'approval', tool: 'a', label: 'A' })
  const b = reg.park({ kind: 'question', tool: 'b', label: 'B', prompt: 'p' })
  const listed = reg.list()
  assert.deepEqual(listed.map((p) => p.id), [a.id, b.id])
  assert.equal(reg.peek(a.id)?.request.tool, 'a')
  assert.equal(reg.peek('missing'), undefined)
  reg.resolve(a.id, { type: 'allow' })
  assert.deepEqual(reg.list().map((p) => p.id), [b.id])
  ok('list/peek expose parked state, oldest first')
}

// ── cancelAll denies every parked interrupt and empties the registry ──────────
{
  const reg = makeRegistry()
  const a = reg.park({ kind: 'approval', tool: 'a', label: 'A' })
  const b = reg.park({ kind: 'approval', tool: 'b', label: 'B' })
  const rejA = assert.rejects(a.promise, InterruptDeniedError)
  const rejB = assert.rejects(b.promise, InterruptDeniedError)
  const cancelled = reg.cancelAll('run cancelled')
  assert.deepEqual(cancelled.sort(), [a.id, b.id].sort())
  assert.equal(reg.size, 0)
  await rejA
  await rejB
  ok('cancelAll denies all parked interrupts')
}

// ── onEvent observes the lifecycle deterministically ──────────────────────────
{
  const events: InterruptEvent[] = []
  const reg = makeRegistry(events)
  const { id } = reg.park({ kind: 'approval', tool: 'run_command', label: 'shell', args: { command: 'ls' } })
  reg.resolve(id, { type: 'allow', args: { command: 'ls -la' } })
  assert.equal(events.length, 2)
  assert.equal(events[0].type, 'parked')
  assert.equal(events[1].type, 'resolved')
  assert.equal((events[1] as any).resolution, 'allow')
  assert.equal((events[1] as any).edited, true)
  // clock is injected → timestamps are deterministic (park at 100, resolve at 101).
  assert.equal((events[0] as any).at, 100)
  assert.equal((events[1] as any).at, 101)
  ok('onEvent observes park/resolve with deterministic timestamps')
}

// ── a throwing observer never breaks the registry ────────────────────────────
{
  let id = 0
  const reg = new InterruptRegistry({
    nextId: () => `int_${++id}`,
    now: () => 0,
    onEvent: () => { throw new Error('observer boom') },
  })
  const parked = reg.park({ kind: 'approval', tool: 't', label: 'l' })
  assert.doesNotThrow(() => reg.resolve(parked.id, { type: 'allow' }))
  ok('throwing observer is swallowed')
}

console.log(`\n${passed} interrupt registry tests passed`)
