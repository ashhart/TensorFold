// Run: npx tsx src/lib/room-agent-tree.test.ts
import assert from 'node:assert/strict'
import { buildRoomAgentTree } from './room-agent-tree'
import type { Room } from './types'

const baseRoom: Room = {
  id: 'room-1',
  projectId: 'project-1',
  name: 'Worker room',
  seats: [
    { id: 'planner', name: 'Planner', role: 'Plans handoffs', providerId: 'openai', model: 'gpt-4.1', fanOutBudget: 'paired' },
    { id: 'builder', name: 'Builder', role: 'Builds the slice', providerId: 'ollama', model: 'qwen', fanOutBudget: 'single' },
  ],
  problem: 'Ship hierarchy UI',
  state: 'running',
  events: [
    { id: 'task', tick: 1, kind: 'task', seat: 'you', target: 'all', text: 'Ship hierarchy UI', createdAt: 1 },
    { id: 'claim', tick: 2, kind: 'claim', seat: 'Planner', target: 'all', text: 'I will split the work.', createdAt: 2 },
    { id: 'delegate-1', tick: 3, kind: 'tool', seat: 'Planner', target: 'all', tool: 'delegate', text: 'delegate: Sub-agent result: Found the room panel.', createdAt: 3 },
    { id: 'delegate-2', tick: 4, kind: 'tool', seat: 'Planner', target: 'all', tool: 'delegate', text: 'delegate: Sub-agent failed: Model limit.', createdAt: 4 },
    { id: 'read', tick: 5, kind: 'tool', seat: 'Builder', target: 'all', tool: 'read_file', text: 'read_file: ok', path: 'src/components/Rooms.tsx', createdAt: 5 },
  ],
  finalText: '',
  createdAt: 1,
  updatedAt: 5,
}

{
  const tree = buildRoomAgentTree(baseRoom)
  assert.equal(tree.length, 2)
  assert.equal(tree[0].name, 'Planner')
  assert.equal(tree[0].fanOutBudget, 'paired')
  assert.equal(tree[0].children.length, 2)
  assert.equal(tree[0].children[0].name, 'Planner sub-agent 1')
  assert.equal(tree[0].children[0].detail, 'Found the room panel.')
  assert.equal(tree[0].children[0].status, 'done')
  assert.equal(tree[0].children[1].status, 'failed')
  assert.equal(tree[1].children.length, 0)
  assert.equal(tree[1].detail, 'Used read_file')
}

{
  const richRoom: Room = {
    ...baseRoom,
    events: [
      {
        id: 'rich-delegate',
        tick: 1,
        kind: 'tool',
        seat: 'Builder',
        target: 'all',
        tool: 'delegate',
        text: 'delegate: queued',
        createdAt: 10,
        args: { task: 'Inspect terminal affordances', name: 'Terminal scout', status: 'running' },
      } as Room['events'][number],
    ],
  }
  const tree = buildRoomAgentTree(richRoom)
  assert.equal(tree[1].children[0].name, 'Terminal scout')
  assert.equal(tree[1].children[0].detail, 'Inspect terminal affordances')
  assert.equal(tree[1].children[0].status, 'active')
}

{
  const rawId = '8f14e45f-ea4c-4a2a-8f05-33f4fd2b7a11'
  const noisyRoom: Room = {
    ...baseRoom,
    events: [
      {
        id: 'raw-name',
        tick: 1,
        kind: 'tool',
        seat: 'Planner',
        target: 'all',
        tool: 'delegate',
        text: 'delegate: running',
        createdAt: 10,
        args: {
          id: rawId,
          name: `subagent-${rawId}`,
          task: 'Keep the delegate under Planner',
          status: 'running',
          subAgent: { id: rawId, name: 'Planner scout' },
        },
      } as Room['events'][number],
    ],
  }
  const tree = buildRoomAgentTree(noisyRoom)
  assert.equal(tree[0].children[0].name, 'Planner scout')
  assert.equal(tree[0].children[0].detail, 'Keep the delegate under Planner')
}

{
  const nestedRoom: Room = {
    ...baseRoom,
    seats: [
      { id: 'bob', name: 'Bob', role: 'Builds', providerId: 'openai', model: 'gpt-5.4', fanOutBudget: 'paired' },
      { id: 'vera', name: 'Vera', role: 'Reviews', providerId: 'anthropic', model: 'opus-4.8', fanOutBudget: 'paired' },
    ],
    events: [
      {
        id: 'bob-child-before-parent',
        tick: 1,
        kind: 'tool',
        seat: 'Bob',
        target: 'all',
        tool: 'delegate',
        text: 'delegate: running',
        createdAt: 12,
        args: { id: 'bob-verifier', parentAgentId: 'bob-builder', name: 'Bob verifier', task: 'Check Bob builder output', status: 'running' },
      } as Room['events'][number],
      {
        id: 'bob-parent',
        tick: 2,
        kind: 'tool',
        seat: 'Bob',
        target: 'all',
        tool: 'delegate',
        text: 'delegate: Sub-agent result: Built the first slice.',
        createdAt: 10,
        args: { id: 'bob-builder', name: 'Bob builder', task: 'Build the first slice', status: 'succeeded' },
      } as Room['events'][number],
      {
        id: 'vera-parent',
        tick: 3,
        kind: 'tool',
        seat: 'Vera',
        target: 'all',
        tool: 'delegate',
        text: 'delegate: queued',
        createdAt: 11,
        args: { id: 'vera-reviewer', name: 'Vera reviewer', task: 'Review Bob changes', status: 'queued' },
      } as Room['events'][number],
    ],
  }
  const tree = buildRoomAgentTree(nestedRoom)
  const bob = tree[0]
  const vera = tree[1]
  assert.equal(bob.name, 'Bob')
  assert.equal(bob.children.length, 1)
  assert.equal(bob.children[0].name, 'Bob builder')
  assert.equal(bob.children[0].children.length, 1)
  assert.equal(bob.children[0].children[0].name, 'Bob verifier')
  assert.equal(bob.children[0].children[0].status, 'active')
  assert.equal(vera.children.length, 1)
  assert.equal(vera.children[0].name, 'Vera reviewer')
  assert.equal(vera.children[0].children.length, 0)
}

{
  const unnamedNestedRoom: Room = {
    ...baseRoom,
    seats: [
      { id: 'bob', name: 'Bob', role: 'Builds', providerId: 'openai', model: 'gpt-5.4', fanOutBudget: 'paired' },
    ],
    events: [
      {
        id: 'unnamed-child-first',
        tick: 1,
        kind: 'tool',
        seat: 'Bob',
        target: 'all',
        tool: 'delegate',
        text: 'delegate: running',
        createdAt: 12,
        args: { id: 'bob-unnamed-child', parentAgentId: 'bob-unnamed-parent', task: 'Check the unnamed parent', status: 'running' },
      } as Room['events'][number],
      {
        id: 'unnamed-parent-later',
        tick: 2,
        kind: 'tool',
        seat: 'Bob',
        target: 'all',
        tool: 'delegate',
        text: 'delegate: Sub-agent result: Parent complete.',
        createdAt: 10,
        args: { id: 'bob-unnamed-parent', task: 'Own the unnamed child', status: 'succeeded' },
      } as Room['events'][number],
    ],
  }
  const tree = buildRoomAgentTree(unnamedNestedRoom)
  assert.equal(tree[0].children.length, 1)
  assert.equal(tree[0].children[0].name, 'Bob sub-agent 1')
  assert.equal(tree[0].children[0].children[0].name, 'Bob sub-agent 1 sub-agent 1')
  assert.equal(tree[0].children[0].children[0].detail, 'Check the unnamed parent')
}

{
  const withSubAgentObjects: Room = {
    ...baseRoom,
    seats: [
      { id: 'bob', name: 'Bob', role: 'Builds', providerId: 'openai', model: 'gpt-5.4' },
    ],
    events: [
      {
        id: 'parent-object',
        tick: 1,
        kind: 'tool',
        seat: 'Bob',
        target: 'all',
        tool: 'launch_subagent',
        text: 'launch_subagent: ok',
        createdAt: 20,
        subAgent: { id: 'agent-parent', name: 'Parent scout', task: 'Map files', status: 'succeeded' },
      } as any,
      {
        id: 'child-object',
        tick: 2,
        kind: 'tool',
        seat: 'Bob',
        target: 'all',
        tool: 'launch_subagent',
        text: 'launch_subagent: ok',
        createdAt: 21,
        subAgent: {
          id: 'agent-child',
          name: 'Child scout',
          task: 'Validate files',
          status: 'succeeded',
          parent: { kind: 'agent', id: 'agent-parent', label: 'Parent scout' },
        },
      } as any,
    ],
  }
  const tree = buildRoomAgentTree(withSubAgentObjects)
  assert.equal(tree[0].children.length, 1)
  assert.equal(tree[0].children[0].name, 'Parent scout')
  assert.equal(tree[0].children[0].children.length, 1)
  assert.equal(tree[0].children[0].children[0].name, 'Child scout')
  assert.equal(tree[0].children[0].children[0].detail, 'Validate files')
}

console.log('room-agent-tree tests passed')
