// Behavioral tests for the supervisor collaboration mode and reply-correlated
// envelopes in runRoom: ASSIGN auto-starts ticket work (supervisor mode only),
// and a trailing "RE: #<id>" correlates an answer to the ask it addresses.
import assert from 'node:assert/strict'
import { runRoom } from './rooms.ts'

const provider = (id: string): any => ({ id, name: id, baseUrl: 'http://local/v1', apiKey: '', kind: 'local', models: [] })
const seat = (id: string, name: string, providerId: string, model: string): any => ({ id, name, role: name, providerId, providerAccountId: null, model })
const makeRoom = (seats: any[], collaborationMode: string): any => ({
  id: 'room1', projectId: 'p1', name: 'Room', problem: '', seats, deliberationEffort: 'quick',
  events: [], memory: [], memorySummary: '', state: 'queued', finalText: '', seatHeartbeats: {},
  collaborationMode, createdAt: 0, updatedAt: 0,
})
const project: any = { id: 'p1', name: 'P', cwd: '/tmp/room', workspace: null }

// Fake chat transport: records calls and replays scripted replies per model.
function recordingChat(scripts: Record<string, string[]>) {
  const calls: { model: string; messages: { role: string; content: string }[] }[] = []
  const q: Record<string, string[]> = Object.fromEntries(Object.entries(scripts).map(([m, a]) => [m, [...a]]))
  const chat = async function* (p: any) {
    calls.push({ model: p.model, messages: p.messages.map((m: any) => ({ role: m.role, content: m.content })) })
    const reply = (q[p.model] && q[p.model].shift()) || 'noted\nACTION: YIELD'
    yield { type: 'content' as const, text: reply }
  }
  return { chat: chat as any, calls }
}

const baseDeps = (room: any, events: any[], chat: any, extra: any = {}) => ({
  room, project, providers: [provider('pA'), provider('pB')], problem: 'Build the feature',
  temperature: 0.2, reasoningEffort: 'medium', emit: (e: any) => events.push(e),
  signal: new AbortController().signal, maxTurns: 4, chat, ...extra,
})

async function run() {
  // 1) SUPERVISOR MODE — an ASSIGN auto-starts ticket work through the onAssign hook,
  //    and the notice lands on the room timeline.
  {
    const { chat } = recordingChat({
      mA: ['Decomposed the goal. Builder takes the login ticket.\nACTION: ASSIGN Builder Implement login form'],
      mB: ['Worker is running; tracking it.\nACTION: FINAL Ticket assigned and worker started.'],
    })
    const events: any[] = []
    const assigns: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'supervisor')
    await runRoom(baseDeps(room, events, chat, {
      onAssign: async (args: any) => { assigns.push(args); return `Auto-started build worker for "Implement login form" with Builder.` },
    }))

    assert.equal(assigns.length, 1, 'supervisor ASSIGN triggered the auto-start hook exactly once')
    assert.equal(assigns[0].seat, 'Planner')
    assert.equal(assigns[0].target, 'Builder')
    assert.ok(assigns[0].text.includes('Implement login form'))
    assert.ok(events.some((e) => e.type === 'event' && e.item?.tool === 'auto_assign' && e.item?.target === 'Builder'),
      'the auto-start notice was committed to the room timeline')
    console.log('ok - supervisor mode ASSIGN auto-starts ticket work')
  }

  // 2) NON-SUPERVISOR MODES — the hook is NOT invoked; manual CLAIM stays the convention.
  {
    const { chat } = recordingChat({
      mA: ['Builder, take the login ticket.\nACTION: ASSIGN Builder Implement login form'],
      mB: ['On it.\nACTION: FINAL Done.'],
    })
    const events: any[] = []
    const assigns: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'balanced')
    await runRoom(baseDeps(room, events, chat, {
      onAssign: async (args: any) => { assigns.push(args); return 'started' },
    }))

    assert.equal(assigns.length, 0, 'balanced mode never auto-starts on ASSIGN')
    assert.ok(!events.some((e) => e.type === 'event' && e.item?.tool === 'auto_assign'))
    console.log('ok - non-supervisor modes leave ASSIGN manual')
  }

  // 3) REPLY ENVELOPES — an answer citing "RE: #<id>" correlates to the ask item.
  {
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'balanced')
    // Builder reads the ref (#id8) out of the broadcast ask and cites it in the reply.
    const chat = async function* (p: any) {
      if (p.model === 'mA') { yield { type: 'content' as const, text: 'Quick check.\nACTION: ASK Builder do you agree with the Map cache?' }; return }
      const seen = p.messages.map((m: any) => m.content).join('\n')
      const ref = seen.match(/ASK, to Builder #([\w-]{8})/)?.[1]
      assert.ok(ref, 'the addressed ask carries a visible ref id in seat context')
      yield { type: 'content' as const, text: `Yes, agreed.\nACTION: ASK Planner confirmed, thanks RE: #${ref}` }
    }
    await runRoom(baseDeps(room, events, chat as any))

    const ask = events.find((e) => e.type === 'event' && e.item?.kind === 'ask' && e.item?.seat === 'Planner')
    const reply = events.find((e) => e.type === 'event' && e.item?.kind === 'ask' && e.item?.seat === 'Builder')
    assert.ok(ask, 'the original ask is on the timeline')
    assert.ok(reply, 'the reply is on the timeline')
    assert.equal(reply.item.replyTo, ask.item.id, 'RE: #<id8> resolves to the full ask event id')
    assert.equal(reply.item.target, 'Planner')
    console.log('ok - RE: #<id> correlates a reply to its ask')
  }

  // 4) LENIENT PARSING — a plain ASK without RE still parses and targets normally.
  {
    const { chat } = recordingChat({
      mA: ['Thoughts?\nACTION: ASK Builder what do you think?'],
      mB: ['Fine.\nACTION: FINAL settled'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'balanced')
    await runRoom(baseDeps(room, events, chat))

    const ask = events.find((e) => e.type === 'event' && e.item?.kind === 'ask' && e.item?.seat === 'Planner')
    assert.ok(ask, 'plain ASK parsed')
    assert.equal(ask.item.target, 'Builder')
    assert.equal(ask.item.replyTo, undefined, 'no RE marker means no replyTo')
    assert.ok(events.some((e) => e.type === 'final'), 'room still finalizes')
    console.log('ok - plain ASK without RE still parses')
  }

  // 5) SUPERVISOR GATE — an ASSIGN from a NON-planning seat never auto-starts; the room
  //    gets a moderator notice instead.
  {
    const { chat } = recordingChat({
      mA: ['Waiting on the builder.\nACTION: YIELD'],
      mB: ['Planner, run the numbers.\nACTION: ASSIGN Planner Re-run the cost estimate', 'Done.\nACTION: FINAL wrapped'],
    })
    const events: any[] = []
    const assigns: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'supervisor')
    await runRoom(baseDeps(room, events, chat, {
      onAssign: async (args: any) => { assigns.push(args); return 'started' },
    }))

    assert.equal(assigns.length, 0, 'a non-supervisor seat ASSIGN must not auto-start work')
    const notice = events.find((e) => e.type === 'event' && e.item?.tool === 'auto_assign')
    assert.ok(notice, 'a moderator notice was committed for the ignored ASSIGN')
    assert.match(notice.item.text, /only the supervisor seat/)
    console.log('ok - non-supervisor ASSIGN is ignored with a moderator notice')
  }

  // 6) ADDRESSEE PARSE — the token right after ASSIGN wins over later seat mentions.
  {
    const { chat } = recordingChat({
      mA: ['Splitting the work.\nACTION: ASSIGN Builder take the auth ticket since Planner is busy'],
      mB: ['On it.\nACTION: FINAL done'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'balanced')
    await runRoom(baseDeps(room, events, chat))

    const assign = events.find((e) => e.type === 'event' && e.item?.kind === 'assign')
    assert.ok(assign, 'ASSIGN committed')
    assert.equal(assign.item.target, 'Builder', 'first token after ASSIGN is the addressee, not the later mention')
    console.log('ok - ASSIGN addressee is the first token, not the last seat mention')
  }

  // 7) GHOST ADDRESSEE — in supervisor mode an ASSIGN that resolves to no seat produces
  //    a moderator notice instead of a silent no-op.
  {
    const { chat } = recordingChat({
      mA: ['Handing off.\nACTION: ASSIGN Ghost take the deploy ticket', 'Done.\nACTION: FINAL wrapped'],
      mB: ['noted\nACTION: YIELD'],
    })
    const events: any[] = []
    const assigns: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'supervisor')
    await runRoom(baseDeps(room, events, chat, {
      onAssign: async (args: any) => { assigns.push(args); return 'started' },
    }))

    assert.equal(assigns.length, 0, 'no auto-start for a ghost addressee')
    const notice = events.find((e) => e.type === 'event' && e.item?.tool === 'auto_assign')
    assert.ok(notice, 'moderator notice committed for the unresolved ASSIGN')
    assert.match(notice.item.text, /did not name a room seat/)
    console.log('ok - ASSIGN to a nonexistent seat emits a moderator notice')
  }

  // 8) RE: GUARD — lowercase prose like "re: testing" must not fabricate a replyTo.
  {
    const { chat } = recordingChat({
      mA: ['Question.\nACTION: ASK Builder what about re: testing'],
      mB: ['Fine.\nACTION: FINAL settled'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'balanced')
    await runRoom(baseDeps(room, events, chat))

    const ask = events.find((e) => e.type === 'event' && e.item?.kind === 'ask')
    assert.ok(ask, 'ASK committed')
    assert.equal(ask.item.replyTo, undefined, 'prose ending in "re: testing" is not a reply citation')
    console.log('ok - lowercase prose re: does not set replyTo')
  }

  // 9) EMPTY REPLY — an empty/reasoning-only seat reply is a silent YIELD: no empty
  //    assistant message, no committed empty event, and the room still progresses.
  {
    const { chat } = recordingChat({
      mA: ['', 'Wrapped up.\nACTION: FINAL done'],
      mB: ['Present and tracking.\nACTION: SPEAK present', 'noted\nACTION: YIELD'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Planner', 'pA', 'mA'), seat('s2', 'Builder', 'pB', 'mB')], 'balanced')
    await runRoom(baseDeps(room, events, chat))

    assert.ok(!events.some((e) => e.type === 'event' && e.item?.seat === 'Planner' && !(e.item?.text ?? '').trim()),
      'no empty event was committed for the empty reply')
    assert.ok(events.some((e) => e.type === 'final'), 'room still finalized after the empty turn')
    console.log('ok - empty seat reply is treated as a silent YIELD')
  }

  console.log('rooms supervisor/envelope tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
