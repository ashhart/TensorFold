// Run: node electron/agent/room-collaboration.test.ts
import assert from 'node:assert'
import {
  buildRoomMemoryDiff,
  buildRoomRunMemoryEntry,
  deliberationBudgetText,
  deliberationProfile,
  formatRoomMemory,
  inferSeatLane,
  normalizeCollaborationMode,
  normalizeDeliberationEffort,
  roomShouldCloseOut,
  seatCollaborationInstructions,
} from './room-collaboration.ts'
import type { Provider, Room, Seat } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const cloud: Provider = {
  id: 'openai',
  name: 'OpenAI',
  baseUrl: 'https://api.openai.com/v1',
  apiKey: '',
  models: ['gpt-4o'],
  kind: 'cloud',
  source: 'openai',
}

const local: Provider = {
  id: 'ollama',
  name: 'Ollama',
  baseUrl: 'http://127.0.0.1:11434/v1',
  apiKey: 'ollama',
  models: ['qwen2.5-coder:14b'],
  kind: 'local',
  source: 'ollama',
}

function seat(role: string, providerId = cloud.id, model = 'gpt-4o'): Seat {
  return { id: role, name: role.split(/\s+/)[0], role, providerId, model }
}

function room(): Room {
  return {
    id: 'room-1',
    projectId: 'project-1',
    name: 'Test room',
    seats: [seat('Planner'), seat('Builder', local.id, 'qwen2.5-coder:14b')],
    collaborationMode: 'frontier_plans_local_builds',
    deliberationEffort: 'normal',
    memory: [],
    memorySummary: '',
    problem: 'Build the connector',
    state: 'running',
    events: [
      { id: 'e1', tick: 1, kind: 'task', seat: 'you', target: 'all', text: 'Build the connector', createdAt: 1 },
      { id: 'e2', tick: 2, kind: 'claim', seat: 'Builder', target: 'all', text: 'Claimed email source scan', createdAt: 2 },
      { id: 'e3', tick: 3, kind: 'tool', seat: 'Builder', target: 'all', text: 'read_file - read source', tool: 'read_file', path: 'electron/agent/email-connector.ts', createdAt: 3 },
      { id: 'e4', tick: 4, kind: 'final', seat: 'Planner', target: 'all', text: 'Connector shape agreed.', createdAt: 4 },
    ],
    finalText: 'Connector shape agreed.',
    createdAt: 1,
    updatedAt: 4,
  }
}

console.log('room collaboration tests:')

{
  assert.strictEqual(normalizeCollaborationMode(undefined), 'frontier_plans_local_builds')
  assert.strictEqual(normalizeCollaborationMode('balanced'), 'balanced')
  ok('normalizes collaboration mode defaults')
}

{
  assert.strictEqual(normalizeDeliberationEffort(undefined), 'normal')
  assert.strictEqual(deliberationProfile('deep').maxTurns, 16)
  assert.match(deliberationBudgetText(deliberationProfile('quick')), /4 room turns/)
  ok('normalizes room deliberation effort and exposes budgets')
}

{
  assert.strictEqual(inferSeatLane(seat('Lead Architect')), 'planning')
  assert.strictEqual(inferSeatLane(seat('Security Reviewer')), 'review')
  assert.strictEqual(inferSeatLane(seat('Local Engineer')), 'development')
  assert.strictEqual(inferSeatLane({ ...seat('Anything'), lane: 'operations' }), 'operations')
  ok('infers useful seat lanes from role text or explicit lane')
}

{
  const planner = seat('Planner')
  const builder = seat('Builder', local.id, local.models[0])
  const planPrompt = seatCollaborationInstructions({ collaborationMode: 'frontier_plans_local_builds', deliberationEffort: 'deep' }, planner, cloud)
  const buildPrompt = seatCollaborationInstructions({ collaborationMode: 'frontier_plans_local_builds', deliberationEffort: 'deep' }, builder, local)
  assert.match(planPrompt, /planning lane/i)
  assert.match(planPrompt, /cloud\/frontier seat/i)
  assert.match(planPrompt, /ROOM DELIBERATION: Deep/)
  assert.match(buildPrompt, /local build lane/i)
  assert.match(buildPrompt, /local seat/i)
  ok('generates lane-aware collaboration instructions')
}

{
  const r = room()
  r.memory = [{ id: 'm1', kind: 'decision', text: 'Use REST first.', source: 'manual', createdAt: 1, updatedAt: 1 }]
  const formatted = formatRoomMemory(r)
  assert.match(formatted, /Use REST first/)
  const entry = buildRoomRunMemoryEntry(r, r.problem, 10)
  assert(entry)
  assert.match(entry.text, /Connector shape agreed/)
  assert.match(entry.text, /electron\/agent\/email-connector\.ts/)
  ok('formats room memory and builds deterministic run memory')
}

{
  const previous = [
    { id: 'm1', kind: 'decision' as const, text: 'Use REST first.', source: 'manual' as const, createdAt: 1, updatedAt: 1 },
    { id: 'm2', kind: 'finding' as const, text: 'Email source exists.', source: 'room' as const, createdAt: 2, updatedAt: 2 },
  ]
  const next = [
    { id: 'm3', kind: 'summary' as const, text: 'Connector shape agreed.', source: 'room' as const, createdAt: 3, updatedAt: 3 },
    previous[0],
  ]
  const diff = buildRoomMemoryDiff(previous, next, 'old summary', 'new summary')
  assert.deepStrictEqual(diff.added.map((entry) => entry.id), ['m3'])
  assert.deepStrictEqual(diff.removed.map((entry) => entry.id), ['m2'])
  assert.strictEqual(diff.previousCount, 2)
  assert.strictEqual(diff.nextCount, 2)
  assert.strictEqual(diff.summaryChanged, true)
  ok('builds a room memory diff after a run')
}

{
  // A room should get a "closing call" nudge on its final stretch so seats commit
  // (create tickets + FINAL) instead of deliberating until the budget hard-stops.
  const NORMAL = 5 * 60 * 1000
  assert.strictEqual(roomShouldCloseOut(0, 8, 0, NORMAL), false, 'start of a normal room is not closing out')
  assert.strictEqual(roomShouldCloseOut(2, 8, 100_000, NORMAL), false, 'under half the time budget is not closing out')
  assert.strictEqual(roomShouldCloseOut(7, 8, 1_000, NORMAL), true, 'the last allowed turn is a closing call')
  assert.strictEqual(roomShouldCloseOut(2, 8, 160_000, NORMAL), true, 'past half the time budget is a closing call')
  // 'max' effort has no time budget (0): only the last turn triggers the closing call.
  assert.strictEqual(roomShouldCloseOut(2, 48, 9_000_000, 0), false, 'no time budget + not last turn is not closing out')
  assert.strictEqual(roomShouldCloseOut(47, 48, 9_000_000, 0), true, 'no time budget + last turn is a closing call')
  ok('flags the final stretch of a room as a closing call')
}

console.log(`room collaboration tests passed (${passed})`)
