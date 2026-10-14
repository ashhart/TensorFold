// Behavioral tests for live shared room memory: a seat's memory_write shows up in
// the next seat's context during the run, and the scratchpad survives the room's
// JSON persistence round-trip (it is re-injected into the primer after a restart).
import assert from 'node:assert/strict'
import { runRoom } from './rooms.ts'
import { createSharedMemory, writeBlock } from './shared-room-memory.ts'

const provider = (id: string): any => ({ id, name: id, baseUrl: 'http://local/v1', apiKey: '', kind: 'local', models: [] })
const seat = (id: string, name: string, providerId: string, model: string): any => ({ id, name, role: name, providerId, providerAccountId: null, model })
const makeRoom = (seats: any[]): any => ({
  id: 'room1', projectId: 'p1', name: 'Room', problem: '', seats, deliberationEffort: 'quick',
  events: [], memory: [], memorySummary: '', state: 'queued', finalText: '', seatHeartbeats: {},
  collaborationMode: 'balanced', sharedMemory: createSharedMemory(), createdAt: 0, updatedAt: 0,
})
const project: any = { id: 'p1', name: 'P', cwd: '/tmp/room', workspace: null }

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
  room, project, providers: [provider('pA'), provider('pB')], problem: 'Speed up lookups',
  temperature: 0.2, reasoningEffort: 'medium', emit: (e: any) => events.push(e),
  signal: new AbortController().signal, maxTurns: 4, chat, ...extra,
})

async function run() {
  // 1) LIVE WIRING — Alpha's memory_write is injected into Beta's context on Beta's
  //    next turn (the "[room] shared memory updated" snapshot).
  {
    const { chat, calls } = recordingChat({
      mA: ['Recording the decision.\nACTION: TOOL memory_write {"name":"decisions","content":"Use a Map cache for O(1) lookups"}'],
      mB: ['Saw the decision, we are aligned.\nACTION: FINAL Use a Map cache.'],
    })
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA'), seat('s2', 'Beta', 'pB', 'mB')])
    // The room tool handler (memory_write in workspace-runtime-ipc) writes through
    // writeBlock against room.sharedMemory; mirror that here.
    const tools = {
      memory_write: async (args: any, seatName: string) => {
        const result = writeBlock(room.sharedMemory, { name: args.name, content: args.content, author: seatName, mode: args.mode })
        assert.ok(result.ok)
        return `Memory "${args.name}" updated.`
      },
    }
    await runRoom(baseDeps(room, events, chat, { tools }))

    assert.equal(room.sharedMemory.blocks.decisions?.content, 'Use a Map cache for O(1) lookups')
    const beta = calls.find((c) => c.model === 'mB')!
    assert.ok(
      beta.messages.some((m) => /shared memory updated/i.test(m.content) && m.content.includes('Use a Map cache for O(1) lookups')),
      "Beta's next turn carries Alpha's memory_write",
    )
    assert.ok(events.some((e) => e.type === 'final'), 'room finalized')
    console.log("ok - a seat's memory_write appears in the next seat's context")
  }

  // 2) PERSISTENCE ROUND-TRIP — the scratchpad survives rooms.json serialization and
  //    is re-injected into each seat's system primer on the next run.
  {
    const events: any[] = []
    const room = makeRoom([seat('s1', 'Alpha', 'pA', 'mA'), seat('s2', 'Beta', 'pB', 'mB')])
    writeBlock(room.sharedMemory, { name: 'decisions', content: 'Cache in a Map', author: 'Alpha' })
    writeBlock(room.sharedMemory, { name: 'blockers', content: 'None', author: 'Beta', mode: 'append-bullet' })
    const restored = JSON.parse(JSON.stringify(room)) // what loadRooms() returns after a restart

    const { chat: chat2, calls } = recordingChat({ mA: ['done\nACTION: FINAL finished'], mB: [] })
    await runRoom(baseDeps(restored, events, chat2))

    const primer = calls[0].messages.find((m) => m.role === 'system')!
    assert.ok(primer.content.includes('SHARED MEMORY'), 'primer carries the shared memory heading')
    assert.ok(primer.content.includes('Cache in a Map'), 'decisions block survived the round-trip')
    assert.ok(primer.content.includes('## BLOCKERS'), 'blockers block survived the round-trip')
    console.log('ok - shared memory survives the room persistence round-trip')
  }

  console.log('rooms shared memory tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
