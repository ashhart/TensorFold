// Behavioral test for runRoom: proves two DISTINCT models actually exchange messages, and pins the
// multi-model fixes (per-turn timeout, prose broadcast, non-failure when one seat dies).
import assert from 'node:assert/strict'
import { runRoom } from './rooms.ts'

const provider = (id: string): any => ({ id, name: id, baseUrl: 'http://local/v1', apiKey: '', kind: 'local', models: [] })
const seat = (id: string, name: string, providerId: string, model: string): any => ({ id, name, role: name, providerId, providerAccountId: null, model })
const makeRoom = (seats: any[]): any => ({
  id: 'room1', name: 'Room', problem: '', seats, deliberationEffort: 'quick',
  events: [], memory: [], memorySummary: '', state: 'queued', finalText: '', seatHeartbeats: {},
  collaborationMode: 'huddle', tickets: [],
})
const project: any = { id: 'p1', name: 'P', cwd: '/tmp/room', workspace: null }

// A fake chat transport that records each call (model + messages it received) and replays scripted
// replies per model. `hang` models block until their turn signal aborts (to exercise the timeout).
function recordingChat(scripts: Record<string, string[]>, hang: Set<string> = new Set()) {
  const calls: { model: string; messages: { role: string; content: string }[] }[] = []
  const q: Record<string, string[]> = Object.fromEntries(Object.entries(scripts).map(([m, a]) => [m, [...a]]))
  const chat = async function* (p: any) {
    calls.push({ model: p.model, messages: p.messages.map((m: any) => ({ role: m.role, content: m.content })) })
    if (hang.has(p.model)) {
      await new Promise((_, rej) => p.signal.addEventListener('abort', () => rej(new Error('aborted')), { once: true }))
    }
    const reply = (q[p.model] && q[p.model].shift()) || 'noted\nACTION: YIELD'
    yield { type: 'content' as const, text: reply }
  }
  return { chat: chat as any, calls }
}

async function run() {
  // 1) HAPPY PATH — two distinct models talk and finalize.
  {
    const { chat, calls } = recordingChat({
      mA: ['I propose we cache results in a Map for O(1) lookups.\nACTION: ASK Beta do you agree with the Map cache?'],
      mB: ['Agreed, a Map cache is right.\nACTION: FINAL Use a Map cache for O(1) lookups.'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA'), seat('s2', 'Beta', 'pB', 'mB')])
    await runRoom({ room, project, providers: [provider('pA'), provider('pB')], problem: 'Speed up lookups?', temperature: 0.2, reasoningEffort: 'medium', emit: (e) => events.push(e), signal: new AbortController().signal, maxTurns: 6, chat })

    assert.ok(calls.some((c) => c.model === 'mA'), 'seat Alpha ran on its own model mA')
    assert.ok(calls.some((c) => c.model === 'mB'), 'seat Beta ran on its own model mB')
    const beta = calls.find((c) => c.model === 'mB')!
    assert.ok(beta.messages.some((m) => m.content.includes('cache results in a Map')),
      "Beta received Alpha's REASONING prose (broadcast delivered the full message, bug #3)")
    assert.ok(beta.messages.some((m) => /Alpha handed work to you/i.test(m.content)),
      'Beta received a direct handoff cue before answering Alpha')
    assert.ok(events.some((e) => e.type === 'final'), 'room reached FINAL')
    assert.equal(room.state, 'complete')
    console.log('ok - two distinct models exchange messages and finalize')
  }

  // 2) TIMEOUT (#6) — a hung model is skipped so the other model still finishes the room.
  {
    const { chat } = recordingChat(
      { mB: ['Alpha is stuck; I will decide.\nACTION: FINAL Proceed with plan B.'] },
      new Set(['mA']), // Alpha hangs on its turn
    )
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA'), seat('s2', 'Beta', 'pB', 'mB')])
    await runRoom({ room, project, providers: [provider('pA'), provider('pB')], problem: 'Decide', temperature: 0.2, reasoningEffort: 'medium', emit: (e) => events.push(e), signal: new AbortController().signal, maxTurns: 6, turnTimeoutMs: 40, chat })

    assert.ok(events.some((e) => e.type === 'error' && /timed out/i.test(e.message)), 'hung seat timed out (not a frozen room)')
    assert.ok(events.some((e) => e.type === 'final'), 'the other model still finalized')
    assert.equal(room.state, 'complete')
    console.log('ok - a hung model is skipped; the room is not frozen')
  }

  // 3) FAILED-STATE (#8) — one seat with no provider errors, but the contributing model completes.
  {
    const { chat } = recordingChat({
      mA: ['First, the API boundary.\nACTION: SPEAK', 'Second, the data model.\nACTION: SPEAK'],
    })
    const events: any[] = []
    // Beta points at provider 'pB' which is NOT supplied → "provider not configured" each pass.
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA'), seat('s2', 'Beta', 'pB', 'mB')])
    await runRoom({ room, project, providers: [provider('pA')], problem: 'Design', temperature: 0.2, reasoningEffort: 'medium', emit: (e) => events.push(e), signal: new AbortController().signal, maxTurns: 2, chat })

    assert.ok(events.some((e) => e.type === 'error' && /provider not configured/i.test(e.message)), 'the broken seat is surfaced')
    // A contributing model that never reaches FINAL leaves the room needing the user
    // (honest "Needs you"), NOT failed — the broken seat must not fail the room.
    assert.equal(room.state, 'waiting_user', 'a contributing model with no FINAL leaves the room waiting on the user, not failed')
    console.log('ok - a broken seat does not fail a room where another model collaborated')
  }

  // 4) DUPLICATE NAMES (#1) — two seats sharing a name must NOT collapse; both distinct models run.
  {
    const { chat, calls } = recordingChat({
      mA: ['Point one from me.\nACTION: SPEAK'],
      mB: ['Point two, and we are done.\nACTION: FINAL Done.'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Bob', 'pA', 'mA'), seat('s2', 'Bob', 'pB', 'mB')]) // SAME name, distinct ids/models
    await runRoom({ room, project, providers: [provider('pA'), provider('pB')], problem: 'X', temperature: 0.2, reasoningEffort: 'medium', emit: (e) => events.push(e), signal: new AbortController().signal, maxTurns: 6, chat })

    assert.ok(calls.some((c) => c.model === 'mA') && calls.some((c) => c.model === 'mB'), 'both same-named seats ran their distinct models')
    console.log('ok - duplicate seat names do not drop a model from the loop')
  }

  // 5) FINAL GATE NOISE — identical blocker checks are logged once, not once per seat/pass.
  {
    const { chat } = recordingChat({
      mA: [
        'Looks complete.\nACTION: FINAL Ship it.',
        'Trying again.\nACTION: FINAL Ship it.',
      ],
      mB: [
        'Also complete.\nACTION: FINAL Ship it.',
        'Trying again too.\nACTION: FINAL Ship it.',
      ],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA'), seat('s2', 'Beta', 'pB', 'mB')])
    await runRoom({
      room,
      project,
      providers: [provider('pA'), provider('pB')],
      problem: 'Build the page',
      temperature: 0.2,
      reasoningEffort: 'medium',
      emit: (e) => events.push(e),
      signal: new AbortController().signal,
      maxTurns: 2,
      finalGate: () => ['room task(s) still open: Build the page.'],
      chat,
    })

    const finalBlocked = events.filter((event) => event.type === 'event' && event.item?.kind === 'check' && /Final blocked:/i.test(event.item.text))
    assert.equal(finalBlocked.length, 1, 'identical final blockers should produce one transcript event')
    assert.equal(room.state, 'waiting_user')
    console.log('ok - repeated final blockers are not spammed to every seat')
  }

  // 6) USER STEERING — a user event in the room is surfaced as latest direction.
  {
    const { chat, calls } = recordingChat({
      mA: ['I will focus the review.\nACTION: FINAL Review auth first.'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA')])
    room.events.push({ id: 'steer-1', tick: 1, kind: 'user', seat: 'you', target: 'all', text: '@reviewer focus auth first', createdAt: Date.now() })
    await runRoom({ room, project, providers: [provider('pA')], problem: 'Review the app', temperature: 0.2, reasoningEffort: 'medium', emit: (e) => events.push(e), signal: new AbortController().signal, maxTurns: 2, chat })

    assert.ok(calls[0].messages.some((m) => /USER STEERING/i.test(m.content) && /focus auth first/i.test(m.content)), 'seat prompt includes latest user steering')
    console.log('ok - user steering is surfaced in the next room turn')
  }

  // 7) OUTCOME GATE (VST-019) — a review-mode room may not FINAL without a verdict; the
  //    moderator names the missing artifact and the corrected FINAL completes the room.
  {
    const { chat, calls } = recordingChat({
      mA: [
        'I read the diff and it seems fine overall.\nACTION: FINAL Overall the code looks reasonable to me.',
        'Committing to a verdict.\nACTION: FINAL Approved: the retry fix is correct and the tests cover the regression.',
      ],
      mB: ['Nothing to add.\nACTION: YIELD'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA'), seat('s2', 'Beta', 'pB', 'mB')])
    room.mode = 'review'
    await runRoom({ room, project, providers: [provider('pA'), provider('pB')], problem: 'Review the retry fix.', temperature: 0.2, reasoningEffort: 'medium', emit: (e) => events.push(e), signal: new AbortController().signal, maxTurns: 6, chat })

    assert.equal(room.state, 'complete', 'the corrected FINAL completes the room')
    assert.match(room.finalText, /Approved:/, 'the surviving final carries the verdict')
    const blocked = room.events.find((e: any) => e.kind === 'check' && /Review outcome incomplete/.test(e.text))
    assert.ok(blocked, 'the verdict-free FINAL was blocked with the missing artifact named')
    assert.match(blocked!.text, /explicit verdict/)
    const alphaPrompts = calls.filter((c) => c.model === 'mA').flatMap((c) => c.messages.map((m) => m.content))
    assert.ok(alphaPrompts.some((m) => /ROOM OUTCOME/.test(m) && /Stop condition:/.test(m)), 'seats see the mode contract in their primer')
    console.log('ok - a review room cannot FINAL without a verdict; the fixed final completes it')
  }

  // 8) ORCH-1 — an illegal start transition is reported and never applied.
  {
    const { chat, calls } = recordingChat({ mA: ['ACTION: FINAL should not run'] })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA')])
    room.state = 'archived'
    await runRoom({ room, project, providers: [provider('pA')], problem: 'Do not reopen', temperature: 0.2, reasoningEffort: 'medium', emit: (e) => events.push(e), signal: new AbortController().signal, maxTurns: 1, chat })

    assert.equal(room.state, 'archived', 'archived -> running is rejected instead of being applied')
    assert.equal(calls.length, 0, 'an illegally-started room never runs a model turn')
    assert.ok(events.some((event) => event.type === 'error' && /archived.*running/i.test(event.message)), 'typed transition failure is observable')
    console.log('ok - illegal room starts are reported without mutating state')
  }

  // 9) ORCH-1 catch regression — once FINAL commits complete, a later delivery exception
  // must not apply the illegal complete -> failed transition or publish a failed lifecycle.
  {
    const { chat } = recordingChat({ mA: ['Conclusive result.\nACTION: FINAL Ship the result.'] })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA')])
    let threwAfterComplete = false
    await runRoom({
      room,
      project,
      providers: [provider('pA')],
      problem: 'Finish safely',
      temperature: 0.2,
      reasoningEffort: 'medium',
      signal: new AbortController().signal,
      maxTurns: 1,
      chat,
      emit: (event) => {
        if (event.type === 'final' && !threwAfterComplete) {
          threwAfterComplete = true
          throw new Error('renderer delivery failed after completion')
        }
        events.push(event)
      },
    })

    assert.equal(threwAfterComplete, true, 'test exercised the post-completion catch path')
    assert.equal(room.state, 'complete', 'complete -> failed is illegal and completion remains authoritative')
    assert.ok(events.some((event) => event.type === 'error' && /complete.*failed/i.test(event.message)), 'illegal catch transition is reported')
    assert.ok(events.some((event) => event.type === 'error' && event.fatal && /renderer delivery failed/.test(event.message)), 'the underlying delivery failure is still surfaced')
    assert.equal(events.some((event) => event.type === 'status' && event.state === 'failed'), false, 'no failed status is emitted for the retained completed room')
    console.log('ok - post-FINAL catch cannot rewrite complete to failed')
  }

  console.log('\nrooms-run behavioral tests passed')
}

await run()
