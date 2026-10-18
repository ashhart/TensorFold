// Run: jiti electron/agent/loop-steering-watcher.test.ts
//
// Regression: the homegrown steer watcher (VST-047) must be retired on EVERY
// stream-attempt exit, not just the success/steer-restart paths. Before the fix,
// a failed attempt left its watcher polling the thread cue seam: it could eat a
// Guide cue queued during the retry backoff (parked in the dead watcher, already
// out of the host queue) and it leaked past run end, draining cues meant for the
// thread's NEXT run.
import assert from 'node:assert/strict'
import { runAgent } from './loop.ts'
import type { AgentCue, Provider, Thread } from './types'

const ok = (name: string) => console.log(`  ✓ ${name}`)
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function waitFor(cond: () => boolean, timeoutMs: number, label: string) {
  const start = Date.now()
  while (!cond()) {
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${label}`)
    await sleep(10)
  }
}

function sseResponse(parts: string[]): Response {
  const body = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder()
      for (const p of parts) controller.enqueue(enc.encode(p))
      controller.close()
    },
  })
  return new Response(body as any, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

console.log('steering watcher lifecycle tests:')

const CUE_TEXT = 'please switch to snake_case'

const thread: Thread = {
  id: 't1', title: 't', kind: 'chat', cwd: process.cwd(),
  providerId: 'p1', model: 'm', createdAt: Date.now(), updatedAt: Date.now(),
  status: 'idle', apiMessages: [{ role: 'user', content: 'hi' }], timeline: [],
} as Thread

const provider: Provider = {
  id: 'p1', name: 'Local', transport: 'chat-completions', kind: 'local',
  baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: ['m'],
}

// Host cue seam: a real queue plus event-driven waiters, mirroring thread-runtime.
const cueQueue: AgentCue[] = []
const waiters: Array<() => void> = []
let waitForCueCalls = 0
let drainCalls = 0
const waitForCue = async (timeoutMs: number, signal: AbortSignal) => {
  waitForCueCalls++
  if (cueQueue.length || signal.aborted) return
  await new Promise<void>((resolve) => {
    const timer = setTimeout(done, timeoutMs)
    const wake = () => done()
    waiters.push(wake)
    function done() {
      clearTimeout(timer)
      signal.removeEventListener('abort', done)
      const i = waiters.indexOf(wake)
      if (i >= 0) waiters.splice(i, 1)
      resolve()
    }
    signal.addEventListener('abort', done, { once: true })
  })
}
const drainCues = (): AgentCue[] => {
  drainCalls++
  return cueQueue.splice(0, cueQueue.length)
}
const pushCue = (text: string) => {
  cueQueue.push({ id: `cue-${cueQueue.length + 1}`, itemId: `cue-item-${cueQueue.length + 1}`, text, createdAt: Date.now() })
  for (const wake of waiters.splice(0, waiters.length)) wake()
}

const events: any[] = []
const requests: any[] = []
let fetchCalls = 0
const originalFetch = globalThis.fetch
globalThis.fetch = (async (_input: any, init: any) => {
  fetchCalls++
  requests.push(JSON.parse(init.body))
  // Attempt 1: an explicit rate-limit refusal before output enters safe retry backoff.
  if (fetchCalls === 1) {
    return new Response('rate limited', { status: 429, statusText: 'Too Many Requests' })
  }
  // Attempt 2: hangs until the fresh steer watcher aborts it (the cue is already
  // queued by then). Falls back to a normal finish after 3s so a regression fails
  // the assertions below instead of hanging the suite.
  if (fetchCalls === 2) {
    const sig: AbortSignal | undefined = init?.signal
    const body = new ReadableStream({
      start(controller) {
        const enc = new TextEncoder()
        const finish = () => {
          try {
            controller.enqueue(enc.encode('data: {"choices":[{"index":0,"delta":{"content":"late reply"},"finish_reason":"stop"}]}\n\n'))
            controller.enqueue(enc.encode('data: [DONE]\n\n'))
            controller.close()
          } catch { /* already errored */ }
        }
        const timer = setTimeout(finish, 3000)
        const fail = () => {
          clearTimeout(timer)
          try { controller.error(new Error('Provider stream was aborted')) } catch { /* already closed */ }
        }
        if (sig?.aborted) fail()
        else sig?.addEventListener('abort', fail, { once: true })
      },
    })
    return new Response(body as any, { status: 200, headers: { 'content-type': 'text/event-stream' } })
  }
  // Re-request after the steer: complete normally.
  return sseResponse([
    'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n',
    'data: [DONE]\n\n',
  ])
}) as any

try {
  const runDone = runAgent({
    thread,
    provider,
    model: 'm',
    approvalMode: 'auto',
    temperature: 0,
    reasoningEffort: 'low',
    emit: (e) => events.push(e),
    requestApproval: async () => 'allow',
    signal: new AbortController().signal,
    persist: () => {},
    drainCues,
    waitForCue,
  })

  // Queue the Guide cue while attempt 1 is dead and the loop sits in retry backoff.
  await waitFor(
    () => events.some((e) => e.type === 'item_add' && String(e.item?.text ?? '').includes('Provider hiccup')),
    5000,
    'retry backoff notice',
  )
  pushCue(CUE_TEXT)
  await runDone

  // (a) The cue survived the retry transition and reached the model on a later
  // request. Two delivery shapes are valid: the attempt-1 watcher drains it during
  // the backoff and the attempt-end finally re-injects it via unclaimedCues (the
  // next request carries it directly), or it stays queued until attempt 2's fresh
  // watcher steers (a re-request then carries it). Both end with the cue in a
  // request body; the regression was the cue vanishing into a dead watcher.
  assert.ok(fetchCalls >= 2, `expected the retry to issue a second request, got ${fetchCalls} fetch calls`)
  const reachedModel = requests.slice(1).some((body) =>
    (body.messages ?? []).some((m: any) => typeof m.content === 'string' && m.content.includes(CUE_TEXT)))
  assert.ok(reachedModel, 'Guide cue queued during the retry transition reached the model (not eaten by the stale watcher)')
  const cueApplied = events.some((e) => e.type === 'item_update' && e.id === 'cue-item-1' && e.patch?.state === 'ok')
  assert.ok(cueApplied, 'cue item was marked applied')
  ok('cue queued during a retry transition reaches the model on the next attempt')

  // (b) The run saw a retry; once it completes, no watcher may keep polling the seam.
  const waitCallsAtEnd = waitForCueCalls
  const drainCallsAtEnd = drainCalls
  await sleep(1200) // > 2x the watcher's 400ms poll interval
  assert.equal(waitForCueCalls, waitCallsAtEnd, 'no watcher keeps calling waitForCue after run end')
  assert.equal(drainCalls, drainCallsAtEnd, 'no watcher keeps draining cues after run end')
  assert.equal(waiters.length, 0, 'no cue waiters remain registered after run end')
  ok('no watcher or waiter survives a run that experienced a retry')
} finally {
  globalThis.fetch = originalFetch
}

console.log('\n2 checks passed.')
console.log('ALL TESTS PASSED')
