// Tests createRoomSeatTurn (#3415): the per-seat window is a stall watchdog refreshed by chunks, not a wall-clock cap.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { setTimeout as nap } from 'node:timers/promises'
import { createRoomSeatTurn, seatFirstOutputWindow, type SeatRuntime } from './room-seat-runtime'
import { runCliSeatTurn } from './room-cli-seat'
import { initSeatHeartbeat } from './room-heartbeats'
import { builtinEngineProvider } from './engine/provider'
import type { RoomRunDeps } from './rooms'

const provider = (id: string, transport?: any): any => ({ id, name: id, baseUrl: 'http://local/v1', apiKey: '', kind: 'local', models: ['m'], transport })
const seat = (id: string, name: string, model: string): any => ({ id, name, role: name, providerId: 'p1', providerAccountId: null, model })
const makeRuntime = (name = 'Alpha'): SeatRuntime => {
  const s = seat('s1', name, 'mA')
  return {
    seat: s,
    reasoningEffort: 'medium',
    messages: [{ role: 'system', content: 'primer' }],
    recent: [],
    heartbeat: initSeatHeartbeat(s, 20, Date.now()),
  }
}
const makeDeps = (over: Partial<RoomRunDeps> = {}): RoomRunDeps => ({
  room: { id: 'room1', name: 'Room', problem: '', seats: [], deliberationEffort: 'quick', events: [], memory: [], memorySummary: '', state: 'queued', finalText: '', seatHeartbeats: {}, collaborationMode: 'huddle', tickets: [] } as any,
  project: { id: 'p1', name: 'P', cwd: '/tmp/room', workspace: null } as any,
  providers: [provider('p1')],
  problem: 'X',
  temperature: 0.2,
  reasoningEffort: 'medium',
  emit: () => {},
  signal: new AbortController().signal,
  ...over,
})
// A fake chat that mirrors streamChat: it honors the turn signal, so an abort stops the iteration.
function observingChat(chunks: string[], gapMs: number) {
  return (async function* (p: any) {
    for (const text of chunks) {
      await nap(gapMs)
      if (p.signal.aborted) throw new Error('aborted')
      yield { type: 'content' as const, text }
    }
  }) as any
}

const WINDOW_MS = 100
const CHUNKS = Array.from({ length: 20 }, (_, i) => `${i}`) // 20 x ~20ms = ~400ms, 4x the window

test('a seat that keeps streaming beats the idle window and returns the full reply', async () => {
  const deltas: string[] = []
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, chat: observingChat(CHUNKS, 20) }), (e: any) => { if (e.type === 'delta') deltas.push(e.chunk) })
  const out = await runSeatTurn(makeRuntime(), provider('p1'))

  assert.equal(out.timedOut, false, 'a productive stream is not aborted')
  assert.equal(out.reply, CHUNKS.join(''), 'every chunk of the long reply is kept')
  assert.deepEqual(deltas, CHUNKS, 'each chunk was forwarded as a delta')
})

test('a seat that never yields is skipped as timed out within about the window', async () => {
  const hangingChat = (async function* (p: any) {
    await new Promise<void>((_, reject) => p.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true }))
    yield { type: 'content' as const, text: 'never reached' }
  }) as any
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, chat: hangingChat }), () => {})

  const startedAt = Date.now()
  const out = await runSeatTurn(makeRuntime(), provider('p1'))

  assert.deepEqual(out, { reply: '', timedOut: true }, 'a stalled seat is aborted and skipped')
  assert.ok(Date.now() - startedAt < WINDOW_MS * 4, 'the watchdog fires on the idle window, not later')
})

test('a seat that yields one chunk and then stalls is still timed out', async () => {
  const stallChat = (async function* (p: any) {
    await nap(5)
    if (p.signal.aborted) throw new Error('aborted')
    yield { type: 'content' as const, text: 'a' }
    await new Promise<void>((_, reject) => p.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true }))
    yield { type: 'content' as const, text: 'b' }
  }) as any
  const deltas: string[] = []
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, chat: stallChat }), (e: any) => { if (e.type === 'delta') deltas.push(e.chunk) })

  const out = await runSeatTurn(makeRuntime(), provider('p1'))

  assert.equal(out.timedOut, true, 'the stall after the first chunk trips the watchdog')
  assert.equal(out.reply, '', 'a timed-out turn contributes no partial reply')
  assert.deepEqual(deltas, ['a'], 'the chunk that arrived was still forwarded')
})

test('turnTimeoutMs 0 disables the watchdog for a slow but complete stream', async () => {
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: 0, chat: observingChat(CHUNKS, 20) }), () => {})
  const out = await runSeatTurn(makeRuntime(), provider('p1'))

  assert.equal(out.timedOut, false, 'no watchdog means no timeout')
  assert.equal(out.reply, CHUNKS.join(''), 'the slow stream finishes')

  // A non-finite window is no watchdog either; it must not arm a 1 ms timer.
  const nanOut = await createRoomSeatTurn(makeDeps({ turnTimeoutMs: Number.NaN, chat: observingChat(CHUNKS, 20) }), () => {})(makeRuntime(), provider('p1'))

  assert.equal(nanOut.timedOut, false, 'NaN is no watchdog')
  assert.equal(nanOut.reply, CHUNKS.join(''))
})

test('a CLI seat turn refreshes the watchdog from its onDelta progress', async () => {
  const cliSeatTurn = (async (params: any, onDelta: (text: string) => void) => {
    let reply = ''
    for (const text of CHUNKS) {
      await nap(20)
      if (params.signal?.aborted) throw new Error('aborted')
      reply += text
      onDelta(text)
    }
    return reply
  }) as any
  const deltas: string[] = []
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, cliSeatTurn }), (e: any) => { if (e.type === 'delta') deltas.push(e.chunk) })

  const out = await runSeatTurn(makeRuntime(), provider('p1', 'claude-cli'))

  assert.equal(out.timedOut, false, 'a streaming CLI seat is not cut off at the window')
  assert.equal(out.reply, CHUNKS.join(''))
  assert.deepEqual(deltas, CHUNKS)
})

test('no stale timer cuts the next one', async () => {
  // Capture the per-turn signal so a timer left armed by turn 1 shows up as an abort on it.
  const seen: AbortSignal[] = []
  const inner = observingChat(['a', 'b'], 20)
  const chat = (async function* (p: any) {
    if (!seen.length) seen.push(p.signal)
    yield* inner(p)
  }) as any
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, chat }), () => {})
  const first = await runSeatTurn(makeRuntime('Alpha'), provider('p1'))
  await nap(WINDOW_MS + 60) // a leaked timer from the first turn would have fired by now

  assert.equal(first.timedOut, false, 'the first turn is a healthy stream')
  assert.equal(seen.length, 1, 'the first turn handed its own signal to the transport')
  assert.equal(seen[0].aborted, false, 'no timer left over from the first turn aborted it')

  const second = await runSeatTurn(makeRuntime('Beta'), provider('p1'))

  assert.equal(second.timedOut, false, 'the next turn is not cut by a timer left over from the previous one')
  assert.equal(second.reply, 'ab')
})

test('a CLI seat that resolves cleanly on abort is still reported as timed out', async () => {
  // Mirrors the real CLI transports: they return (not throw) once their signal aborts.
  const cliSeatTurn = (async (params: any, onDelta: (text: string) => void) => {
    await nap(5)
    if (params.signal?.aborted) return ''
    onDelta('a')
    await nap(WINDOW_MS + 120)
    return 'a'
  }) as any
  const deltas: string[] = []
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, cliSeatTurn }), (e: any) => { if (e.type === 'delta') deltas.push(e.chunk) })

  const out = await runSeatTurn(makeRuntime(), provider('p1', 'claude-cli'))

  assert.deepEqual(out, { reply: '', timedOut: true }, 'a clean resolve after the watchdog fired is still a stall')
  assert.deepEqual(deltas, ['a'], 'the chunk that arrived was still forwarded')
})

test('a built-in engine seat keeps the engine first-byte allowance before its first chunk', async () => {
  const engineProvider: any = builtinEngineProvider()
  // Cold prefill: the first chunk lands after the idle window but well inside the engine allowance.
  const prefillChat = (async function* (p: any) {
    await nap(WINDOW_MS + 50)
    if (p.signal.aborted) throw new Error('aborted')
    yield { type: 'content' as const, text: 'a' }
    await nap(10)
    if (p.signal.aborted) throw new Error('aborted')
    yield { type: 'content' as const, text: 'b' }
  }) as any
  const prefill = await createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, chat: prefillChat }), () => {})(makeRuntime(), engineProvider)

  assert.equal(prefill.timedOut, false, 'a slow cold prefill is not a stall')
  assert.equal(prefill.reply, 'ab', 'the whole reply is kept')

  // After the first chunk the plain idle window applies again.
  const stallChat = (async function* (p: any) {
    await nap(WINDOW_MS + 50)
    if (p.signal.aborted) throw new Error('aborted')
    yield { type: 'content' as const, text: 'a' }
    await new Promise<void>((_, reject) => p.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true }))
    yield { type: 'content' as const, text: 'b' }
  }) as any
  const stalled = await createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, chat: stallChat }), () => {})(makeRuntime(), engineProvider)

  assert.deepEqual(stalled, { reply: '', timedOut: true }, 'the stall after the first chunk still trips the idle window')
})

test('a room-signal abort in the middle of a stalled turn is not a seat timeout', async () => {
  const controller = new AbortController()
  const hangingChat = (async function* (p: any) {
    await new Promise<void>((_, reject) => p.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true }))
    yield { type: 'content' as const, text: 'never reached' }
  }) as any
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: 10_000, chat: hangingChat, signal: controller.signal }), () => {})
  const started = runSeatTurn(makeRuntime(), provider('p1'))
  await nap(10)
  controller.abort()

  await assert.rejects(() => started, /aborted/, 'a user stop propagates instead of reporting a bogus timeout')
})

test('a room signal already aborted before the turn starts never reaches the model', async () => {
  const controller = new AbortController()
  controller.abort()
  const entry: boolean[] = []
  const chat = (async function* (p: any) {
    if (!entry.length) entry.push(p.signal.aborted)
    if (p.signal.aborted) throw new Error('aborted')
    yield { type: 'content' as const, text: 'never reached' }
  }) as any
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, chat, signal: controller.signal }), () => {})

  await assert.rejects(() => runSeatTurn(makeRuntime(), provider('p1')), /aborted/, 'the abort propagates out of the turn')
  assert.deepEqual(entry, [true], 'the transport saw an already-aborted turn signal on entry')
})

test('seatFirstOutputWindow gives the engine the first-byte allowance and everyone else the idle window', () => {
  const engineProvider: any = builtinEngineProvider()
  assert.equal(seatFirstOutputWindow(engineProvider, 180_000), 600000, 'a cold built-in prefill keeps the engine allowance')
  assert.equal(seatFirstOutputWindow(provider('p1'), 180_000), 180000, 'an API provider keeps the idle window')
  assert.equal(seatFirstOutputWindow(engineProvider, 0), 0, 'no watchdog')
  assert.equal(seatFirstOutputWindow(engineProvider, Number.NaN), 0, 'a non-finite window is no watchdog')
  // The room's own deliberation budget caps the engine allowance, never shortens the idle window.
  assert.equal(seatFirstOutputWindow(engineProvider, 180_000, 0), 600000, 'no budget means the full engine allowance')
  assert.equal(seatFirstOutputWindow(engineProvider, 180_000, 120_000), 180000, 'the idle window wins over a smaller room budget')
  assert.equal(seatFirstOutputWindow(engineProvider, 600_000, 120_000), 600000, 'a longer idle window still wins')
  assert.equal(seatFirstOutputWindow(provider('p1'), 180_000, 120_000), 180000, 'a non-engine seat ignores the budget')
  assert.equal(seatFirstOutputWindow(engineProvider, 0, 120_000), 0, 'no watchdog, budget or not')
})

test('a non-content CLI chunk still refreshes the watchdog through onActivity', async () => {
  // Each gap is under the window, but the 120ms since the last *content* chunk is over it:
  // only the reasoning chunk's onActivity call can keep the seat alive.
  const source = (async function* (params: any) {
    // Mirrors the real CLI transports: each read stops as soon as the turn signal is aborted.
    if (params.signal?.aborted) return
    yield { type: 'content', text: 'a' }
    await nap(60)
    if (params.signal?.aborted) return
    yield { type: 'reasoning', text: 'thinking\n' }
    await nap(60)
    if (params.signal?.aborted) return
    yield { type: 'content', text: 'b' }
  }) as any
  const cliSeatTurn = (params: any, onDelta: (text: string) => void, onActivity?: () => void) =>
    runCliSeatTurn(params, onDelta, source, onActivity)
  const deltas: string[] = []
  const runSeatTurn = createRoomSeatTurn(makeDeps({ turnTimeoutMs: WINDOW_MS, cliSeatTurn }), (e: any) => { if (e.type === 'delta') deltas.push(e.chunk) })

  const out = await runSeatTurn(makeRuntime(), provider('p1', 'claude-cli'))

  assert.deepEqual(out, { reply: 'ab', timedOut: false }, 'the reasoning chunk counted as progress')
  assert.deepEqual(deltas, ['a', 'b'], 'only content chunks are forwarded as deltas')
})
