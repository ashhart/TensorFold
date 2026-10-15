// Run: jiti electron/agent/steering.test.ts
//
// Live mid-turn steering channel + cue pump (VST-033). Pure — no engine processes.
import assert from 'node:assert/strict'
import { createSteeringChannel, providerSupportsLiveSteering, runSteerCuePump, type SteerCue } from './steering'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('steering tests:')

// --- channel ---
{
  const channel = createSteeringChannel()
  assert.equal(channel.canAccept, false)
  assert.equal(await channel.offer({ id: 'a', text: 'hi' }), false)

  const seen: string[] = []
  const off = channel.connect(async (input) => { seen.push(input.text); return true })
  assert.equal(channel.canAccept, true)
  assert.equal(await channel.offer({ id: 'a', text: 'first' }), true)
  assert.deepEqual(seen, ['first'])

  off()
  assert.equal(channel.canAccept, false)
  assert.equal(await channel.offer({ id: 'b', text: 'after-off' }), false)
  assert.deepEqual(seen, ['first'])
  ok('offer only reaches a connected consumer; disconnect stops delivery')
}

{
  const channel = createSteeringChannel()
  let ready = false
  channel.connect(async () => true, () => ready)
  assert.equal(channel.canAccept, false)
  assert.equal(await channel.offer({ id: 'a', text: 'x' }), false, 'not ready → refused without reaching consumer')
  ready = true
  assert.equal(channel.canAccept, true)
  assert.equal(await channel.offer({ id: 'a', text: 'x' }), true)
  ok('isReady gates acceptance (codex active-turn precondition shape)')
}

{
  const channel = createSteeringChannel()
  const first: string[] = []
  const second: string[] = []
  const offFirst = channel.connect(async (i) => { first.push(i.text); return true })
  channel.connect(async (i) => { second.push(i.text); return true })
  await channel.offer({ id: 'a', text: 'x' })
  // The stale disconnect from the replaced consumer must not evict the new one.
  offFirst()
  assert.equal(channel.canAccept, true)
  await channel.offer({ id: 'b', text: 'y' })
  assert.deepEqual(first, [])
  assert.deepEqual(second, ['x', 'y'])
  ok('reconnect replaces the consumer; a stale disconnect cannot evict the replacement')
}

{
  const channel = createSteeringChannel()
  channel.connect(async () => { throw new Error('engine died mid-write') })
  assert.equal(await channel.offer({ id: 'a', text: 'x' }), false)
  const channel2 = createSteeringChannel()
  channel2.connect(async () => true, () => { throw new Error('isReady blew up') })
  assert.equal(channel2.canAccept, false)
  assert.equal(await channel2.offer({ id: 'a', text: 'x' }), false)
  ok('consumer/isReady exceptions surface as refusal, never as a crash')
}

// --- transport gate ---
{
  assert.equal(providerSupportsLiveSteering({ transport: 'codex-cli' } as any), true)
  assert.equal(providerSupportsLiveSteering({ transport: 'claude-cli' } as any), true)
  assert.equal(providerSupportsLiveSteering({ transport: 'pi-cli' } as any), true)
  assert.equal(providerSupportsLiveSteering({ transport: 'opencode' } as any), false)
  assert.equal(providerSupportsLiveSteering({ transport: 'kilo' } as any), false)
  assert.equal(providerSupportsLiveSteering({ transport: 'cursor-cli' } as any), false)
  assert.equal(providerSupportsLiveSteering({ transport: 'gemini-cli' } as any), false)
  assert.equal(providerSupportsLiveSteering({ transport: undefined } as any), false)
  assert.equal(providerSupportsLiveSteering(null), false)
  ok('only codex/claude/pi transports steer live')
}

// --- pump ---
function pumpHarness() {
  const channel = createSteeringChannel()
  const aborter = new AbortController()
  const queue: SteerCue[] = []
  let wakeWaiters: Array<() => void> = []
  const steered: string[] = []
  const rejected: string[] = []
  const enqueue = (cue: SteerCue) => {
    queue.push(cue)
    const waiters = wakeWaiters
    wakeWaiters = []
    for (const w of waiters) w()
  }
  const pump = runSteerCuePump({
    channel,
    signal: aborter.signal,
    pollMs: 5,
    waitForCue: (timeoutMs, signal) => new Promise((resolve) => {
      if (queue.length || signal.aborted) return resolve()
      const timer = setTimeout(resolve, timeoutMs)
      wakeWaiters.push(() => { clearTimeout(timer); resolve() })
      signal.addEventListener('abort', () => { clearTimeout(timer); resolve() }, { once: true })
    }),
    drainCues: () => queue.splice(0, queue.length),
    onSteered: (cue) => steered.push(cue.itemId),
    onRejected: (cue) => rejected.push(cue.itemId),
  })
  return { channel, aborter, enqueue, steered, rejected, queue, pump }
}

const tick = () => new Promise((r) => setTimeout(r, 25))

{
  const h = pumpHarness()
  h.enqueue({ itemId: 'cue-1', text: 'before engine ready' })
  await tick()
  assert.equal(h.queue.length, 1, 'no consumer → cue stays queued for the turn-end follow-up path')
  assert.deepEqual(h.steered, [])

  const taken: string[] = []
  h.channel.connect(async (i) => { taken.push(i.text); return true })
  await tick()
  assert.deepEqual(taken, ['before engine ready'])
  assert.deepEqual(h.steered, ['cue-1'])
  assert.equal(h.queue.length, 0)

  h.enqueue({ itemId: 'cue-2', text: 'mid-turn' })
  await tick()
  assert.deepEqual(h.steered, ['cue-1', 'cue-2'])
  h.aborter.abort()
  await h.pump
  ok('pump leaves cues queued until the engine is ready, then delivers in order')
}

{
  const h = pumpHarness()
  h.channel.connect(async () => false)
  h.enqueue({ itemId: 'cue-1', text: 'engine refuses' })
  await tick()
  assert.deepEqual(h.rejected, ['cue-1'])
  assert.deepEqual(h.steered, [])
  h.aborter.abort()
  await h.pump
  ok('a drained cue the engine refuses surfaces as an explicit rejection')
}

{
  const h = pumpHarness()
  h.aborter.abort()
  await h.pump
  h.enqueue({ itemId: 'cue-1', text: 'after abort' })
  await tick()
  assert.equal(h.queue.length, 1, 'aborted pump never drains')
  assert.deepEqual(h.steered, [])
  assert.deepEqual(h.rejected, [])
  ok('abort stops the pump without touching the queue')
}

console.log(`  ${passed} passed`)
